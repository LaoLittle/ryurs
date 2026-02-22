use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ryurs_dynarec::{CpuState, ExitReason, StepResult};
use ryurs_gpu_wgpu::nvdrv::{NvdrvIoctlKind, NvdrvService};

use super::service_manager::ServiceManager;
use super::syscall::{SyscallDisposition, dispatch_supervisor_call};
use crate::emulator::DisplayFrame;
use crate::memory::SharedGuestMemory;

pub const MAIN_PROCESS_HANDLE: u32 = 1;
pub const MAIN_THREAD_HANDLE: u32 = 1;
pub const SELF_THREAD_HANDLE: u32 = 0xffff_8000;
pub const SELF_PROCESS_HANDLE: u32 = 0xffff_8001;
const FIRST_DYNAMIC_PROCESS_HANDLE: u32 = 2;
const FIRST_DYNAMIC_THREAD_HANDLE: u32 = 2;
const FIRST_USER_HANDLE: u32 = 2;
const HANDLE_WAIT_MASK: u32 = 0x4000_0000;
const FIRST_SERVICE_SESSION_ID: u32 = 1;
const FIRST_EVENT_ID: u32 = 1;
const DEFAULT_THREAD_PRIORITY: i32 = 44;
const DEFAULT_CORE_MASK: u64 = 0x0f;
const THREAD_TLS_PAGE_SIZE: u64 = 0x1000;
const APPLET_MESSAGE_CHANGE_INTO_FOREGROUND: u32 = 1;
const APPLET_MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadCoreMaskError {
    InvalidHandle,
    InvalidCombination,
}

fn configured_core_mask() -> u64 {
    static CORE_MASK_OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();

    CORE_MASK_OVERRIDE
        .get_or_init(|| {
            std::env::var("RYURS_CORE_MASK").ok().and_then(|value| {
                if let Some(hex) = value.strip_prefix("0x") {
                    u64::from_str_radix(hex, 16).ok()
                } else {
                    value.parse::<u64>().ok()
                }
            })
        })
        .unwrap_or(DEFAULT_CORE_MASK)
}

fn random_core_id_from_mask(seed: u64, mask: u64) -> i32 {
    let mut slots = [0u8; 64];
    let mut count = 0usize;
    for core in 0..64u8 {
        if ((mask >> core) & 1) != 0 {
            slots[count] = core;
            count += 1;
        }
    }

    if count == 0 {
        return 0;
    }

    let mut mixed = seed ^ 0x9e37_79b9_7f4a_7c15;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;

    i32::from(slots[(mixed as usize) % count])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessLifecycle {
    Created,
    Runnable,
    Exited,
}

#[derive(Debug, Clone)]
struct GuestProcess {
    handle: u32,
    pid: u64,
    lifecycle: ProcessLifecycle,
    main_thread_handle: u32,
    entrypoint: u64,
    argument: u64,
    stack_top: u64,
    random_entropy: [u64; 4],
    next_user_handle: u32,
    handle_table: HashMap<u32, HandleObject>,
    memory: SharedGuestMemory,
    /// Start of the heap virtual-address region (Ryujinx: HeapRegionStart).
    heap_region_start: u64,
    /// End of the heap virtual-address region (Ryujinx: HeapRegionEnd).
    heap_region_end: u64,
    /// Maximum physical memory the heap may consume
    /// (Ryujinx: _heapCapacity = _memoryUsageCapacity - stackSize - imageSize).
    heap_capacity: u64,
    /// Current end of the committed heap (grows from heap_region_start).
    current_heap_addr: u64,
    /// Total size of loaded image segments (used for GetInfo used-memory).
    image_size: u64,
    /// Main thread stack size.
    main_thread_stack_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandleObject {
    Process(u32),
    Thread(u32),
    ServiceSession(u32),
    Event(u32),
    SharedMemory {
        size: u64,
        permission: u32,
    },
    TransferMemory {
        address: u64,
        size: u64,
        permission: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct EventState {
    signaled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceSessionKind {
    Sm,
    AppletOE,
    Apm,
    Lm,
    Friend,
    EctxAw,
    AocU,
    Pctl,
    ViU,
    Nvdrv,
    Hid,
    Time,
    AudrenU,
    AudOut,
    AudOutSession,
    FspSrv,
    Acc,
    Nifm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainObjectKind {
    GenericRoot,
    AppletProxyService,
    ApplicationProxy,
    CommonStateGetter,
    SelfController,
    WindowController,
    AudioController,
    DisplayController,
    ProcessWindingController,
    LibraryAppletCreator,
    ApplicationFunctions,
    DebugFunctions,
    ApmManager,
    ApmSession,
    LmLogService,
    LmLogger,
    FriendServiceCreator,
    FriendService,
    FspStorage,
    AmStorage,
    AmStorageAccessor,
    BaasManager,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViSessionRole {
    Root,
    ApplicationDisplay,
    SystemDisplay,
    ManagerDisplay,
    BinderRelay,
    BinderIndirect,
}

#[derive(Debug, Clone)]
struct ServiceSessionState {
    kind: ServiceSessionKind,
    vi_role: Option<ViSessionRole>,
    domain_enabled: bool,
    next_domain_object_id: u32,
    domain_objects: HashMap<u32, DomainObjectKind>,
    event_handles: HashMap<(u32, u32), u32>,
    applet_resource_user_id: u64,
}

impl ServiceSessionState {
    fn new(kind: ServiceSessionKind, applet_resource_user_id: u64) -> Self {
        Self {
            kind,
            vi_role: (kind == ServiceSessionKind::ViU).then_some(ViSessionRole::Root),
            domain_enabled: false,
            next_domain_object_id: 2,
            domain_objects: HashMap::new(),
            event_handles: HashMap::new(),
            applet_resource_user_id,
        }
    }

    fn root_domain_kind(&self) -> DomainObjectKind {
        match self.kind {
            ServiceSessionKind::AppletOE => DomainObjectKind::AppletProxyService,
            ServiceSessionKind::Apm => DomainObjectKind::ApmManager,
            ServiceSessionKind::Lm => DomainObjectKind::LmLogService,
            ServiceSessionKind::Friend => DomainObjectKind::FriendServiceCreator,
            ServiceSessionKind::Sm
            | ServiceSessionKind::EctxAw
            | ServiceSessionKind::AocU
            | ServiceSessionKind::Pctl
            | ServiceSessionKind::ViU
            | ServiceSessionKind::Nvdrv
            | ServiceSessionKind::Hid
            | ServiceSessionKind::Time
            | ServiceSessionKind::AudrenU
            | ServiceSessionKind::AudOut
            | ServiceSessionKind::AudOutSession
            | ServiceSessionKind::FspSrv
            | ServiceSessionKind::Acc
            | ServiceSessionKind::Nifm => DomainObjectKind::GenericRoot,
        }
    }
}

fn entropy_words(seed: u64) -> [u64; 4] {
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    let mut state = seed ^ 0xd1b5_4a32_d192_ed03;
    [
        splitmix64(&mut state),
        splitmix64(&mut state),
        splitmix64(&mut state),
        splitmix64(&mut state),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AddressWaitKey {
    process_handle: u32,
    address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThreadLifecycle {
    Created,
    Runnable,
    Blocked,
    Safepoint,
    Exited,
}

#[derive(Debug, Clone)]
struct GuestThread {
    handle: u32,
    process_handle: u32,
    thread_uid: u64,
    priority: i32,
    core_id: i32,
    affinity_mask: u64,
    core_binding_fixed: bool,
    lifecycle: ThreadLifecycle,
    running_on_worker: bool,
    wait_address: Option<u64>,
    address_key_address: Option<u64>,
    address_key_value: u32,
    wait_sync_handles: Vec<u32>,
    wait_sync_out_index_ptr: Option<u64>,
    wake_deadline: Option<Instant>,
    context: CpuState,
}

#[derive(Debug, Clone)]
pub struct WorkerTask {
    pub thread_handle: u32,
    pub process_handle: u32,
    pub state: CpuState,
    pub memory: SharedGuestMemory,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerTrap {
    pub pc: u64,
    pub reason: ExitReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinderBufferState {
    Free,
    Dequeued,
    Queued,
    Acquired,
}

#[derive(Debug, Clone, Copy)]
pub struct BinderQueueItem {
    pub slot: u32,
    pub frame: u64,
}

#[derive(Debug)]
pub struct KernelState {
    pub tick_count: u64,
    pub frame_count: u64,
    next_process_handle: u32,
    next_process_pid: u64,
    next_thread_handle: u32,
    next_thread_uid: u64,
    next_service_session_id: u32,
    next_event_id: u32,
    processes: Vec<GuestProcess>,
    threads: Vec<GuestThread>,
    ready_queue: VecDeque<u32>,
    address_waiters: HashMap<AddressWaitKey, VecDeque<u32>>,
    service_sessions: HashMap<u32, ServiceSessionState>,
    events: HashMap<u32, EventState>,
    applet_messages: VecDeque<u32>,
    next_display_layer_id: u64,
    safepoint_requested: bool,
    gpu_adapter_name: Option<String>,
    worker_trap: Option<WorkerTrap>,
    nvdrv: NvdrvService,
    /// Binder VSync event handles (signaled periodically to simulate display refresh).
    binder_vsync_handles: Vec<(u32, u32)>, // (process_handle, event_handle)
    /// Counter: consecutive ArbitrateLock calls from the main thread.
    /// When this exceeds MAIN_THREAD_YIELD_THRESHOLD, request a safepoint
    /// so worker threads get CPU time.
    main_thread_lock_spins: u32,
    /// When set, the main thread is "blocked" on this mutex address.
    /// The emulator should skip main thread execution and run workers.
    main_thread_yield_mutex: Option<u64>,
    /// Counter for the number of times we've polled during the current yield.
    main_thread_yield_poll_count: u64,
    /// Deferred sleep duration for the main thread.  Set by svc_sleep_thread
    /// so that `step()` can release the kernel lock before sleeping.
    main_thread_pending_sleep: Option<std::time::Duration>,
    /// Audio out buffer tags that have been "played" and are ready to be returned
    /// by GetReleasedAudioOutBuffers / GetReleasedAudioOutBuffersAuto.
    audio_out_released_tags: VecDeque<u64>,
    /// Audio out buffer tags that have been appended but not yet "played" —
    /// they wait for a simulated playback delay before being released.
    audio_out_pending_tags: VecDeque<(u64, Instant)>,
    /// Event handle for the audio out buffer event (RegisterBufferEvent).
    audio_out_buffer_event: Option<u32>,
    /// Decrypted RomFS data for FspStorage::Read (IFileSystemProxy::OpenDataStorageByCurrentProcess).
    pub romfs_data: Option<Arc<Vec<u8>>>,
    /// Handle ID for the HID shared memory (set when IAppletResource::GetSharedMemoryHandle returns).
    hid_shared_memory_handle: Option<u32>,
    /// Guest virtual address where HID shared memory is mapped (set by svc_map_shared_memory).
    hid_shared_memory_address: Option<u64>,
    /// Monotonic counter for HID Npad sampling numbers.
    hid_sampling_number: u64,
    /// Binder graphic buffers stored per slot via SetPreallocatedBuffer (transaction 14).
    /// Up to 64 slots; each slot stores the raw GraphicBuffer blob (from slot offset onward).
    binder_graphic_buffers: Vec<Option<Vec<u8>>>,
    /// Lightweight producer/consumer state per binder slot.
    binder_slot_states: Vec<BinderBufferState>,
    /// Next slot to return from DequeueBuffer. Cycles through allocated slots.
    binder_dequeue_slot: u32,
    /// Total number of preallocated buffer slots.
    binder_num_slots: u32,
    /// Most recently queued buffer slot from Binder QueueBuffer.
    binder_queued_slot: Option<u32>,
    /// FIFO of queued buffers awaiting consumer-side acquisition.
    binder_queue: VecDeque<BinderQueueItem>,
    /// Frame number of the most recent queued binder buffer awaiting presentation.
    binder_pending_frame: Option<u64>,
    /// Channel for sending framebuffer pixels to the display renderer.
    /// When set, QueueBuffer will copy the guest framebuffer and send it.
    frame_tx: Option<std::sync::mpsc::SyncSender<DisplayFrame>>,
}

impl Default for KernelState {
    fn default() -> Self {
        Self {
            tick_count: 0,
            frame_count: 0,
            next_process_handle: FIRST_DYNAMIC_PROCESS_HANDLE,
            next_process_pid: 2,
            next_thread_handle: FIRST_DYNAMIC_THREAD_HANDLE,
            next_thread_uid: 2,
            next_service_session_id: FIRST_SERVICE_SESSION_ID,
            next_event_id: FIRST_EVENT_ID,
            processes: Vec::new(),
            threads: Vec::new(),
            ready_queue: VecDeque::new(),
            address_waiters: HashMap::new(),
            service_sessions: HashMap::new(),
            events: HashMap::new(),
            applet_messages: VecDeque::new(),
            next_display_layer_id: 1,
            safepoint_requested: false,
            gpu_adapter_name: None,
            worker_trap: None,
            nvdrv: NvdrvService::default(),
            binder_vsync_handles: Vec::new(),
            main_thread_lock_spins: 0,
            main_thread_yield_mutex: None,
            main_thread_yield_poll_count: 0,
            main_thread_pending_sleep: None,
            audio_out_released_tags: VecDeque::new(),
            audio_out_pending_tags: VecDeque::new(),
            audio_out_buffer_event: None,
            romfs_data: None,
            hid_shared_memory_handle: None,
            hid_shared_memory_address: None,
            hid_sampling_number: 0,
            binder_graphic_buffers: vec![None; 64],
            binder_slot_states: vec![BinderBufferState::Free; 64],
            binder_dequeue_slot: 0,
            binder_num_slots: 0,
            binder_queued_slot: None,
            binder_queue: VecDeque::new(),
            binder_pending_frame: None,
            frame_tx: None,
        }
    }
}

impl KernelState {
    fn trace_worker_steps_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("RYURS_TRACE_WORKER_STEPS")
                .map(|value| value != "0")
                .unwrap_or(false)
        })
    }

    fn trace_worker_step_limit() -> usize {
        static LIMIT: OnceLock<usize> = OnceLock::new();
        *LIMIT.get_or_init(|| {
            std::env::var("RYURS_TRACE_WORKER_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(512)
        })
    }

    fn trace_worker_step(handle: u32, step: StepResult, state: &CpuState) {
        static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);

        if !Self::trace_worker_steps_enabled() {
            return;
        }

        let index = TRACE_COUNT.fetch_add(1, Ordering::Relaxed);
        if index >= Self::trace_worker_step_limit() {
            return;
        }

        eprintln!(
            "[worker-step] idx={} tid={} step={:?} pc=0x{:016x} x0=0x{:016x} x1=0x{:016x} x2=0x{:016x} sp=0x{:016x}",
            index,
            handle,
            step,
            state.pc,
            state.read_x(0),
            state.read_x(1),
            state.read_x(2),
            state.sp,
        );
    }

    pub fn reset_threads(&mut self, main_state: &CpuState, main_memory: &SharedGuestMemory) {
        self.next_process_handle = FIRST_DYNAMIC_PROCESS_HANDLE;
        self.next_process_pid = 2;
        self.next_thread_handle = FIRST_DYNAMIC_THREAD_HANDLE;
        self.next_thread_uid = 2;
        self.next_service_session_id = FIRST_SERVICE_SESSION_ID;
        self.next_event_id = FIRST_EVENT_ID;
        self.processes.clear();
        self.threads.clear();
        self.ready_queue.clear();
        self.address_waiters.clear();
        self.service_sessions.clear();
        self.events.clear();
        self.applet_messages.clear();
        self.next_display_layer_id = 1;
        self.safepoint_requested = false;
        self.worker_trap = None;
        self.binder_vsync_handles.clear();
        self.main_thread_yield_mutex = None;

        self.applet_messages
            .push_back(APPLET_MESSAGE_FOCUS_STATE_CHANGED);
        self.applet_messages
            .push_back(APPLET_MESSAGE_CHANGE_INTO_FOREGROUND);

        let default_affinity_mask = configured_core_mask();
        let main_core_id = random_core_id_from_mask(1, default_affinity_mask);

        let mut main_handle_table = HashMap::new();
        main_handle_table.insert(MAIN_THREAD_HANDLE, HandleObject::Thread(MAIN_THREAD_HANDLE));

        self.processes.push(GuestProcess {
            handle: MAIN_PROCESS_HANDLE,
            pid: 1,
            lifecycle: ProcessLifecycle::Runnable,
            main_thread_handle: MAIN_THREAD_HANDLE,
            entrypoint: main_state.pc,
            argument: main_state.read_x(0),
            stack_top: main_state.sp,
            random_entropy: entropy_words(1),
            next_user_handle: FIRST_USER_HANDLE,
            handle_table: main_handle_table,
            memory: main_memory.clone(),
            heap_region_start: 0,
            heap_region_end: 0,
            heap_capacity: 0,
            current_heap_addr: 0,
            image_size: 0,
            main_thread_stack_size: 0,
        });
        self.threads.push(GuestThread {
            handle: MAIN_THREAD_HANDLE,
            process_handle: MAIN_PROCESS_HANDLE,
            thread_uid: 1,
            priority: DEFAULT_THREAD_PRIORITY,
            core_id: main_core_id,
            affinity_mask: default_affinity_mask,
            core_binding_fixed: false,
            lifecycle: ThreadLifecycle::Runnable,
            running_on_worker: false,
            wait_address: None,
            address_key_address: None,
            address_key_value: 0,
            wait_sync_handles: Vec::new(),
            wait_sync_out_index_ptr: None,
            wake_deadline: None,
            context: main_state.clone(),
        });
    }

    pub fn on_cpu_block(&mut self, memory: &SharedGuestMemory) {
        self.tick_count = self.tick_count.wrapping_add(1);

        // Simulate ~60 Hz VSync: signal binder events every 512 blocks.
        // This gives blocked threads a chance to process frames.
        if self.tick_count & 0x1FF == 0 {
            self.signal_binder_vsync_events();
        }

        // Update HID shared memory with Npad state every 256 blocks (~60 Hz equivalent).
        if self.tick_count & 0xFF == 0 {
            self.update_hid_shared_memory(memory);
        }

        // Promote pending audio buffers whose playback delay has elapsed.
        if self.tick_count & 0x3F == 0 {
            self.tick_audio_buffers();
        }
    }

    /// Lightweight tick during yield-poll: VSync + HID + audio buffers.
    /// Audio ticking is needed so the audio thread (which may hold the
    /// mutex we're waiting on) can see released buffers and exit its
    /// polling loop.
    pub fn on_cpu_block_yield(&mut self, memory: &SharedGuestMemory) {
        self.tick_count = self.tick_count.wrapping_add(1);

        if self.tick_count & 0x1FF == 0 {
            self.signal_binder_vsync_events();
        }

        if self.tick_count & 0xFF == 0 {
            self.update_hid_shared_memory(memory);
        }

        // Promote pending audio buffers so the mutex owner can finish.
        if self.tick_count & 0x3F == 0 {
            self.tick_audio_buffers();
        }
    }

    /// Record that the main thread called ArbitrateLock.
    /// If it spins too many times without other SVCs, set a yield flag
    /// so the emulator gives worker threads CPU time.
    pub fn record_main_thread_lock_spin(&mut self) {
        const MAIN_THREAD_YIELD_THRESHOLD: u32 = 8;
        self.main_thread_lock_spins += 1;
        if self.main_thread_lock_spins >= MAIN_THREAD_YIELD_THRESHOLD {
            self.main_thread_lock_spins = 0;
        }
    }

    /// Returns true if the main thread should yield to let workers run.
    pub fn main_thread_should_yield(&self) -> bool {
        self.main_thread_lock_spins == 0 && self.has_worker_work()
    }

    /// Returns true if the main thread is "blocked" on a mutex,
    /// meaning the emulator should skip main thread execution.
    pub fn main_thread_is_yielding(&self) -> bool {
        self.main_thread_yield_mutex.is_some()
    }

    /// Returns the mutex address the main thread is yield-waiting on, if any.
    pub fn main_thread_yield_mutex_address(&self) -> Option<u64> {
        self.main_thread_yield_mutex
    }

    /// Set the main thread as "blocked" on a contended mutex.
    pub fn set_main_thread_yield_mutex(&mut self, mutex_address: u64) {
        self.main_thread_yield_mutex = Some(mutex_address);
        self.main_thread_yield_poll_count = 0;
    }

    /// Return (and increment) the yield poll counter.
    pub fn yield_poll_count(&mut self) -> u64 {
        let c = self.main_thread_yield_poll_count;
        self.main_thread_yield_poll_count += 1;
        c
    }

    /// Set a deferred sleep for the main thread.
    /// Called by svc_sleep_thread; the actual sleep happens in step()
    /// after the kernel lock is released.
    pub fn set_main_thread_pending_sleep(&mut self, dur: std::time::Duration) {
        self.main_thread_pending_sleep = Some(dur);
    }

    /// Take the pending sleep duration, if any.
    pub fn take_main_thread_pending_sleep(&mut self) -> Option<std::time::Duration> {
        self.main_thread_pending_sleep.take()
    }

    /// Check if the mutex the main thread is blocked on has become free.
    /// If so, clear the yield flag and return true.
    pub fn try_clear_main_thread_yield(&mut self, memory: &SharedGuestMemory) -> bool {
        if let Some(addr) = self.main_thread_yield_mutex {
            if let Some(mutex_value) = memory.read_u32(addr) {
                // Mutex is free if the word is 0 or only has the main thread's tag.
                if mutex_value == 0 || mutex_value == MAIN_THREAD_HANDLE {
                    eprintln!(
                        "[yield] main thread mutex 0x{addr:016x} freed (value=0x{mutex_value:08x})"
                    );
                    self.main_thread_yield_mutex = None;
                    return true;
                }
                // Periodic diagnostic log
                if self.tick_count & 0xFFF == 0 {
                    eprintln!(
                        "[yield] main thread waiting on mutex 0x{addr:016x} value=0x{mutex_value:08x} tick={}",
                        self.tick_count
                    );
                    for t in &self.threads {
                        if t.handle != MAIN_THREAD_HANDLE {
                            eprintln!(
                                "[yield-debug] thread {} lifecycle={:?} on_worker={} wait_addr={:?} deadline={:?}",
                                t.handle,
                                t.lifecycle,
                                t.running_on_worker,
                                t.wait_address,
                                t.wake_deadline
                                    .map(|d| d.duration_since(std::time::Instant::now()))
                            );
                        }
                    }
                }
            } else {
                // Can't read mutex word → unblock to avoid permanent hang.
                self.main_thread_yield_mutex = None;
                return true;
            }
        }
        false
    }

    /// Force-clear the yield mutex (e.g., timeout or worker triggered release).
    pub fn clear_main_thread_yield(&mut self) {
        self.main_thread_yield_mutex = None;
    }

    /// Reset the lock spin counter when the main thread does something
    /// other than ArbitrateLock.
    pub fn reset_main_thread_lock_spins(&mut self) {
        self.main_thread_lock_spins = 0;
    }

    /// Push an audio-out buffer tag into the "pending" queue.
    /// Buffers are moved to the released queue after a realistic playback
    /// delay, simulating the audio hardware consuming samples.
    pub fn audio_out_append_buffer(&mut self, tag: u64) {
        self.audio_out_pending_tags.push_back((tag, Instant::now()));
    }

    /// Drain all released audio buffer tags (for GetReleasedAudioOutBuffers).
    /// Also promotes any pending buffers whose playback delay has elapsed.
    pub fn audio_out_take_released(&mut self) -> Vec<u64> {
        // Move pending buffers that have "played" (≥5ms since append) to released.
        let now = Instant::now();
        let playback_delay = Duration::from_micros(100);
        while let Some(&(tag, appended_at)) = self.audio_out_pending_tags.front() {
            if now.duration_since(appended_at) >= playback_delay {
                self.audio_out_pending_tags.pop_front();
                self.audio_out_released_tags.push_back(tag);
            } else {
                break;
            }
        }
        self.audio_out_released_tags.drain(..).collect()
    }

    /// Promote any pending audio buffers whose playback delay has elapsed,
    /// and signal the buffer event if any became available.
    pub fn tick_audio_buffers(&mut self) {
        if self.audio_out_pending_tags.is_empty() {
            return;
        }
        let now = Instant::now();
        let playback_delay = Duration::from_micros(100);
        let mut promoted = false;
        while let Some(&(tag, appended_at)) = self.audio_out_pending_tags.front() {
            if now.duration_since(appended_at) >= playback_delay {
                self.audio_out_pending_tags.pop_front();
                self.audio_out_released_tags.push_back(tag);
                promoted = true;
            } else {
                break;
            }
        }
        if promoted {
            if let Some(ev_handle) = self.audio_out_buffer_event {
                let _ = self.set_event_signaled_for_handle(MAIN_THREAD_HANDLE, ev_handle, true);
            }
        }
    }

    /// Register the audio out buffer event handle.
    pub fn set_audio_out_buffer_event(&mut self, handle: u32) {
        self.audio_out_buffer_event = Some(handle);
    }

    /// Get the audio out buffer event handle (if registered).
    pub fn audio_out_buffer_event(&self) -> Option<u32> {
        self.audio_out_buffer_event
    }

    /// Record the HID shared memory handle returned by IAppletResource::GetSharedMemoryHandle.
    pub fn set_hid_shared_memory_handle(&mut self, handle: u32) {
        self.hid_shared_memory_handle = Some(handle);
    }

    /// Check if a shared memory handle is the HID one; if so, record the guest address.
    pub fn check_hid_shared_memory_mapping(&mut self, handle: u32, guest_address: u64) {
        if self.hid_shared_memory_handle == Some(handle) {
            eprintln!(
                "[hid] shared memory mapped at guest address 0x{:016x}",
                guest_address
            );
            self.hid_shared_memory_address = Some(guest_address);
        }
    }

    /// Get the guest address of HID shared memory (if mapped).
    pub fn hid_shared_memory_address(&self) -> Option<u64> {
        self.hid_shared_memory_address
    }

    /// Update HID shared memory with Npad controller state.
    /// Simulates a Pro Controller connected on player 1 (NpadIdType_No1).
    ///
    /// HID shared memory layout (from libnx):
    ///   Offset 0x0000: DebugPad
    ///   Offset 0x0400: TouchScreen
    ///   Offset 0x3400: Mouse
    ///   Offset 0x3800: Keyboard
    ///   Offset 0x3C00: Digitizer
    ///   Offset 0x4C00: HomeButton
    ///   Offset 0x4E00: SleepButton
    ///   Offset 0x5000: CaptureButton
    ///   Offset 0x5200: InputDetector
    ///   Offset 0x5A00: UniquePad
    ///   Offset 0x9A00: Npad (10 entries × 0x5000 each)
    ///
    /// NpadInternalState layout (within each 0x5000 entry):
    ///   +0x00: u32 style_set (HidNpadStyleTag bits)
    ///   +0x04: u32 joy_assignment_mode
    ///   +0x08: HidNpadFullKeyColorState { u32 attribute, u32 main_color, u32 sub_color }
    ///   +0x14: HidNpadJoyColorState { u32 attribute, u32 left_main, u32 left_sub, u32 right_main, u32 right_sub }
    ///   +0x28: HidNpadCommonLifo full_key_lifo
    ///     header: { u64 unused, u64 buffer_count, u64 tail, u64 count }
    ///     storage[17]: { u64 sampling_number, HidNpadCommonState }
    ///       HidNpadCommonState: { u64 sampling_number, u64 buttons, (s32,s32) stick_l, (s32,s32) stick_r, u32 attributes, u32 reserved }
    ///     Each storage entry = 0x30 bytes, lifo total = 0x20 header + 17×0x30 = 0x350
    ///   ... (more lifos, six-axis, etc.)
    ///   +0x4188: u32 device_type
    ///   +0x418C: u32 reserved
    ///   +0x4190: u64 system_properties
    ///   +0x4198: u32 system_button_properties
    ///   +0x419C: u32 battery_level[3]
    pub fn update_hid_shared_memory(&mut self, memory: &SharedGuestMemory) {
        let base = match self.hid_shared_memory_address {
            Some(addr) => addr,
            None => return,
        };

        self.hid_sampling_number += 1;
        let sn = self.hid_sampling_number;

        // Npad entry 0 (Player 1) at offset 0x9A00
        let npad0 = base + 0x9A00;

        // style_set = NpadFullKey (Pro Controller) = BIT(0) = 1
        memory.write_u32(npad0 + 0x00, 1);
        // joy_assignment_mode = Dual (0)
        memory.write_u32(npad0 + 0x04, 0);

        // FullKeyColorState: attribute=1 (ReadableColors), body color=0x303030 (dark gray), button=0xE6E6E6
        memory.write_u32(npad0 + 0x08, 1); // attribute = HasColor
        memory.write_u32(npad0 + 0x0C, 0x303030FF); // body RGBA
        memory.write_u32(npad0 + 0x10, 0xE6E6E6FF); // button RGBA

        // JoyColorState: attribute=0 (no color)
        memory.write_u32(npad0 + 0x14, 0);

        // full_key_lifo at +0x28:
        let lifo_base = npad0 + 0x28;
        // header.unused = 0
        memory.write_u64(lifo_base + 0x00, 0);
        // header.buffer_count = 17
        memory.write_u64(lifo_base + 0x08, 17);
        // header.tail = index of latest entry (sn - 1) % 17
        let tail = (sn.wrapping_sub(1)) % 17;
        memory.write_u64(lifo_base + 0x10, tail);
        // header.count = number of writes
        memory.write_u64(lifo_base + 0x18, sn);

        // Write the current entry at storage[tail]
        let entry_base = lifo_base + 0x20 + tail * 0x30;
        // sampling_number (atomic storage outer)
        memory.write_u64(entry_base + 0x00, sn);
        // HidNpadCommonState: sampling_number
        memory.write_u64(entry_base + 0x08, sn);
        // buttons = 0 (nothing pressed)
        memory.write_u64(entry_base + 0x10, 0);
        // analog_stick_l = (0, 0)
        memory.write_u32(entry_base + 0x18, 0);
        memory.write_u32(entry_base + 0x1C, 0);
        // analog_stick_r = (0, 0)
        memory.write_u32(entry_base + 0x20, 0);
        memory.write_u32(entry_base + 0x24, 0);
        // attributes = BIT(0) IsConnected | BIT(1) IsWired = 3
        memory.write_u32(entry_base + 0x28, 3);
        // reserved = 0
        memory.write_u32(entry_base + 0x2C, 0);

        // device_type at +0x4188: BIT(0) = FullKey
        memory.write_u32(npad0 + 0x4188, 1);
        // reserved
        memory.write_u32(npad0 + 0x418C, 0);
        // system_properties: is_powered | is_abxy_button_oriented | is_plus_available | is_minus_available | is_directional_buttons_available
        // bits: [0..3] is_charging=0, [3..6] is_powered=0b001 (one controller powered),
        //       [11] is_abxy_button_oriented=1, [13] is_plus_available=1, [14] is_minus_available=1, [15] is_directional_buttons_available=1
        // = BIT(3) | BIT(11) | BIT(13) | BIT(14) | BIT(15) = 0x8 | 0x800 | 0x2000 | 0x4000 | 0x8000 = 0xE808
        memory.write_u64(npad0 + 0x4190, 0xE808);
        // system_button_properties = 0
        memory.write_u32(npad0 + 0x4198, 0);
        // battery_level[0] = 4 (full), [1] = 4, [2] = 4
        memory.write_u32(npad0 + 0x419C, 4);
        memory.write_u32(npad0 + 0x41A0, 4);
        memory.write_u32(npad0 + 0x41A4, 4);

        // Also write the handheld_lifo at +0x28 + 0x350 = +0x378 with the same data
        // (in case the game checks handheld mode)
        let hh_lifo_base = npad0 + 0x378;
        memory.write_u64(hh_lifo_base + 0x00, 0);
        memory.write_u64(hh_lifo_base + 0x08, 17);
        memory.write_u64(hh_lifo_base + 0x10, tail);
        memory.write_u64(hh_lifo_base + 0x18, sn);
        let hh_entry = hh_lifo_base + 0x20 + tail * 0x30;
        memory.write_u64(hh_entry + 0x00, sn);
        memory.write_u64(hh_entry + 0x08, sn);
        memory.write_u64(hh_entry + 0x10, 0);
        memory.write_u32(hh_entry + 0x18, 0);
        memory.write_u32(hh_entry + 0x1C, 0);
        memory.write_u32(hh_entry + 0x20, 0);
        memory.write_u32(hh_entry + 0x24, 0);
        memory.write_u32(hh_entry + 0x28, 3);
        memory.write_u32(hh_entry + 0x2C, 0);

        // Also set npad entry 8 (Handheld) at offset 0x9A00 + 8*0x5000 = 0x31A00
        // with style_set = Handheld BIT(1) = 2
        let npad8 = base + 0x9A00 + 8 * 0x5000;
        memory.write_u32(npad8 + 0x00, 2); // style_set = NpadHandheld
        memory.write_u32(npad8 + 0x04, 0); // joy_assignment_mode
        // handheld_lifo at +0x378
        let hh8_lifo = npad8 + 0x378;
        memory.write_u64(hh8_lifo + 0x00, 0);
        memory.write_u64(hh8_lifo + 0x08, 17);
        memory.write_u64(hh8_lifo + 0x10, tail);
        memory.write_u64(hh8_lifo + 0x18, sn);
        let hh8_entry = hh8_lifo + 0x20 + tail * 0x30;
        memory.write_u64(hh8_entry + 0x00, sn);
        memory.write_u64(hh8_entry + 0x08, sn);
        memory.write_u64(hh8_entry + 0x10, 0);
        memory.write_u32(hh8_entry + 0x18, 0);
        memory.write_u32(hh8_entry + 0x1C, 0);
        memory.write_u32(hh8_entry + 0x20, 0);
        memory.write_u32(hh8_entry + 0x24, 0);
        memory.write_u32(hh8_entry + 0x28, 3);
        memory.write_u32(hh8_entry + 0x2C, 0);
        // device_type: HandheldLeft | HandheldRight = BIT(2) | BIT(3) = 0xC
        memory.write_u32(npad8 + 0x4188, 0xC);
        memory.write_u64(npad8 + 0x4190, 0xE808);
        memory.write_u32(npad8 + 0x4198, 0);
        memory.write_u32(npad8 + 0x419C, 4);
        memory.write_u32(npad8 + 0x41A0, 4);
        memory.write_u32(npad8 + 0x41A4, 4);
    }

    /// Register a binder VSync event handle for periodic signaling.
    pub fn register_binder_vsync_event(&mut self, process_handle: u32, event_handle: u32) {
        if !self
            .binder_vsync_handles
            .iter()
            .any(|(p, h)| *p == process_handle && *h == event_handle)
        {
            self.binder_vsync_handles
                .push((process_handle, event_handle));
        }
    }

    /// Signal all registered binder VSync event handles.
    pub fn signal_binder_vsync_events(&mut self) {
        let handles: Vec<(u32, u32)> = self.binder_vsync_handles.clone();
        for (process_handle, event_handle) in handles {
            // Look up the event by scanning the process handle table.
            let event_id = self
                .processes
                .iter()
                .find(|p| p.handle == process_handle)
                .and_then(|p| p.handle_table.get(&event_handle).copied())
                .and_then(|obj| match obj {
                    HandleObject::Event(id) => Some(id),
                    _ => None,
                });
            if let Some(event_id) = event_id {
                if let Some(event) = self.events.get_mut(&event_id) {
                    event.signaled = true;
                }
                self.wake_waiting_threads_for_signaled_handle(process_handle, event_handle);
            }
        }
    }

    pub fn has_worker_work(&self) -> bool {
        !self.ready_queue.is_empty()
    }

    pub fn worker_thread_handles_snapshot(&self) -> Vec<u32> {
        self.threads
            .iter()
            .filter(|thread| {
                thread.handle != MAIN_THREAD_HANDLE && thread.lifecycle != ThreadLifecycle::Exited
            })
            .map(|thread| thread.handle)
            .collect()
    }

    pub fn thread_is_live(&self, thread_handle: u32) -> bool {
        self.thread_index(thread_handle)
            .map(|index| self.threads[index].lifecycle != ThreadLifecycle::Exited)
            .unwrap_or(false)
    }

    pub fn safepoint_is_active(&self) -> bool {
        self.safepoint_requested
    }

    pub fn request_safepoint_interrupt(&mut self) {
        if self.safepoint_requested {
            return;
        }
        self.safepoint_requested = true;

        let queued: Vec<u32> = self.ready_queue.drain(..).collect();
        for handle in queued {
            if let Some(index) = self.thread_index(handle)
                && handle != MAIN_THREAD_HANDLE
                && self.threads[index].lifecycle == ThreadLifecycle::Runnable
            {
                self.threads[index].lifecycle = ThreadLifecycle::Safepoint;
            }
        }

        for thread in &mut self.threads {
            if thread.handle == MAIN_THREAD_HANDLE || thread.running_on_worker {
                continue;
            }
            if thread.lifecycle == ThreadLifecycle::Runnable {
                thread.lifecycle = ThreadLifecycle::Safepoint;
            }
        }
    }

    pub fn release_safepoint_interrupt(&mut self) {
        if !self.safepoint_requested {
            return;
        }
        self.safepoint_requested = false;

        let mut enqueue = Vec::new();
        for thread in &mut self.threads {
            if thread.lifecycle != ThreadLifecycle::Safepoint {
                continue;
            }
            thread.lifecycle = ThreadLifecycle::Runnable;
            if thread.handle != MAIN_THREAD_HANDLE && !thread.running_on_worker {
                enqueue.push(thread.handle);
            }
        }

        for handle in enqueue {
            self.enqueue_thread(handle);
        }
    }

    pub fn worker_threads_quiesced_for_safepoint(&self) -> bool {
        self.threads.iter().all(|thread| {
            if thread.handle == MAIN_THREAD_HANDLE {
                return true;
            }
            !thread.running_on_worker && thread.lifecycle != ThreadLifecycle::Runnable
        })
    }

    pub fn set_gpu_adapter_name(&mut self, adapter_name: Option<String>) {
        self.gpu_adapter_name = adapter_name;
    }

    /// Set the channel used to send framebuffer pixels to the display renderer.
    pub fn set_frame_sender(&mut self, tx: std::sync::mpsc::SyncSender<DisplayFrame>) {
        self.frame_tx = Some(tx);
    }

    /// Returns a reference to the frame sender, if one has been configured.
    pub fn frame_sender(&self) -> Option<&std::sync::mpsc::SyncSender<DisplayFrame>> {
        self.frame_tx.as_ref()
    }

    pub fn has_gpu_backend(&self) -> bool {
        self.gpu_adapter_name.is_some()
    }

    pub fn gpu_adapter_name(&self) -> Option<&str> {
        self.gpu_adapter_name.as_deref()
    }

    pub fn process_handle_for_thread(&self, thread_handle: u32) -> Option<u32> {
        let index = self.thread_index(thread_handle)?;
        Some(self.threads[index].process_handle)
    }

    pub fn random_entropy_word(&self, thread_handle: u32, sub_id: i64) -> Option<u64> {
        let index = usize::try_from(sub_id).ok()?;
        if index > 3 {
            return None;
        }

        let process_handle = self.process_handle_for_thread(thread_handle)?;
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| process.random_entropy[index])
    }

    pub fn process_pid_for_handle(&self, requester_thread_handle: u32, handle: u32) -> Option<u64> {
        let object = self.resolve_handle_object(requester_thread_handle, handle)?;

        match object {
            HandleObject::Process(process_handle) => self.process_pid(process_handle),
            HandleObject::Thread(thread_handle) => {
                let thread_index = self.thread_index(thread_handle)?;
                self.process_pid(self.threads[thread_index].process_handle)
            }
            HandleObject::ServiceSession(_)
            | HandleObject::Event(_)
            | HandleObject::SharedMemory { .. }
            | HandleObject::TransferMemory { .. } => None,
        }
    }

    pub fn create_shared_memory_handle(
        &mut self,
        requester_thread_handle: u32,
        size: u64,
        permission: u32,
    ) -> Option<u32> {
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;
        self.allocate_handle(
            process_handle,
            HandleObject::SharedMemory { size, permission },
        )
    }

    pub fn shared_memory_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(u64, u32)> {
        match self.resolve_handle_object(requester_thread_handle, handle)? {
            HandleObject::SharedMemory { size, permission } => Some((size, permission)),
            HandleObject::Process(_)
            | HandleObject::Thread(_)
            | HandleObject::ServiceSession(_)
            | HandleObject::Event(_)
            | HandleObject::TransferMemory { .. } => None,
        }
    }

    pub fn create_transfer_memory_handle(
        &mut self,
        requester_thread_handle: u32,
        address: u64,
        size: u64,
        permission: u32,
    ) -> Option<u32> {
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;
        self.allocate_handle(
            process_handle,
            HandleObject::TransferMemory {
                address,
                size,
                permission,
            },
        )
    }

    pub fn transfer_memory_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(u64, u64, u32)> {
        match self.resolve_handle_object(requester_thread_handle, handle)? {
            HandleObject::TransferMemory {
                address,
                size,
                permission,
            } => Some((address, size, permission)),
            HandleObject::Process(_)
            | HandleObject::Thread(_)
            | HandleObject::ServiceSession(_)
            | HandleObject::Event(_)
            | HandleObject::SharedMemory { .. } => None,
        }
    }

    pub fn process_address_space_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(u64, u64)> {
        let process_handle = self.resolve_process_handle(requester_thread_handle, handle)?;
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| {
                let base = process.memory.base_address();
                let size = process.memory.size_bytes();
                (base, size)
            })
    }

    /// Returns `(heap_region_start, heap_region_end, heap_capacity, current_heap_addr)`.
    pub fn process_heap_region_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(u64, u64, u64, u64)> {
        let process_handle = self.resolve_process_handle(requester_thread_handle, handle)?;
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| {
                (
                    process.heap_region_start,
                    process.heap_region_end,
                    process.heap_capacity,
                    process.current_heap_addr,
                )
            })
    }

    /// Update `current_heap_addr` after a successful SetHeapSize.
    pub fn set_process_current_heap_addr(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        addr: u64,
    ) {
        let Some(process_handle) = self.resolve_process_handle(requester_thread_handle, handle)
        else {
            return;
        };
        if let Some(process) = self
            .processes
            .iter_mut()
            .find(|p| p.handle == process_handle && p.lifecycle != ProcessLifecycle::Exited)
        {
            process.current_heap_addr = addr;
        }
    }

    /// Returns `(image_size, main_thread_stack_size)` for used-memory computation.
    pub fn process_image_info_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(u64, u64)> {
        let process_handle = self.resolve_process_handle(requester_thread_handle, handle)?;
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| (process.image_size, process.main_thread_stack_size))
    }

    /// Configure the heap region for the main process.
    /// Called from the emulator after loading the program image.
    pub fn configure_main_process_heap(
        &mut self,
        heap_region_start: u64,
        heap_region_end: u64,
        image_size: u64,
        main_thread_stack_size: u64,
    ) {
        if let Some(process) = self
            .processes
            .iter_mut()
            .find(|p| p.handle == MAIN_PROCESS_HANDLE)
        {
            process.heap_region_start = heap_region_start;
            process.heap_region_end = heap_region_end;
            process.current_heap_addr = heap_region_start;
            process.image_size = image_size;
            process.main_thread_stack_size = main_thread_stack_size;
            // Ryujinx: heapCapacity = memoryUsageCapacity - stackSize - imageSize
            // where memoryUsageCapacity = HeapRegionEnd - HeapRegionStart
            let region_size = heap_region_end.saturating_sub(heap_region_start);
            process.heap_capacity = region_size.saturating_sub(main_thread_stack_size + image_size);
        }
    }

    pub fn thread_priority_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<i32> {
        let thread_handle = self.resolve_thread_handle(requester_thread_handle, handle)?;
        let index = self.thread_index(thread_handle)?;
        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return None;
        }

        Some(self.threads[index].priority)
    }

    pub fn thread_uid_for_handle(&self, requester_thread_handle: u32, handle: u32) -> Option<u64> {
        let thread_handle = self.resolve_thread_handle(requester_thread_handle, handle)?;
        let index = self.thread_index(thread_handle)?;
        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return None;
        }

        Some(self.threads[index].thread_uid)
    }

    pub fn thread_core_mask_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<(i32, u64)> {
        let thread_handle = self.resolve_thread_handle(requester_thread_handle, handle)?;
        let index = self.thread_index(thread_handle)?;
        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return None;
        }

        Some((
            self.threads[index].core_id,
            self.threads[index].affinity_mask,
        ))
    }

    pub fn set_thread_core_mask(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        ideal_core_id: i32,
        affinity_mask: u64,
    ) -> Result<(), ThreadCoreMaskError> {
        let Some(thread_handle) = self.resolve_thread_handle(requester_thread_handle, handle)
        else {
            return Err(ThreadCoreMaskError::InvalidHandle);
        };

        let Some(index) = self.thread_index(thread_handle) else {
            return Err(ThreadCoreMaskError::InvalidHandle);
        };

        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return Err(ThreadCoreMaskError::InvalidHandle);
        }

        let available_mask = configured_core_mask();
        let effective_mask = affinity_mask & available_mask;
        if effective_mask == 0 {
            return Err(ThreadCoreMaskError::InvalidCombination);
        }

        let (core_id, fixed_binding) = if ideal_core_id >= 0 {
            if ideal_core_id >= 64 || ((effective_mask >> ideal_core_id) & 1) == 0 {
                return Err(ThreadCoreMaskError::InvalidCombination);
            }
            (ideal_core_id, true)
        } else if ideal_core_id == -1 {
            let current_core = self.threads[index].core_id;
            if current_core >= 0 && current_core < 64 && ((effective_mask >> current_core) & 1) != 0
            {
                (current_core, false)
            } else {
                let seed = self.threads[index].thread_uid ^ self.tick_count;
                (random_core_id_from_mask(seed, effective_mask), false)
            }
        } else {
            return Err(ThreadCoreMaskError::InvalidCombination);
        };

        self.threads[index].core_id = core_id;
        self.threads[index].affinity_mask = effective_mask;
        self.threads[index].core_binding_fixed = fixed_binding;

        Ok(())
    }

    pub fn set_thread_priority(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        priority: i32,
    ) -> bool {
        let Some(thread_handle) = self.resolve_thread_handle(requester_thread_handle, handle)
        else {
            return false;
        };

        let Some(index) = self.thread_index(thread_handle) else {
            return false;
        };

        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return false;
        }

        self.threads[index].priority = priority;
        true
    }

    pub fn connect_to_named_port(
        &mut self,
        requester_thread_handle: u32,
        name: &str,
    ) -> Option<u32> {
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;
        let kind = ServiceManager::resolve_named_port_kind(name)?;

        self.allocate_service_session_handle(process_handle, kind)
    }

    pub fn service_session_for_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<ServiceSessionKind> {
        let session_id = self.resolve_service_session_id(requester_thread_handle, handle)?;
        self.service_sessions
            .get(&session_id)
            .map(|session| session.kind)
    }

    pub fn service_session_vi_role(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<ViSessionRole> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::ViU {
            return None;
        }

        session.vi_role
    }

    pub fn service_session_set_vi_role(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        role: ViSessionRole,
    ) -> bool {
        let Some(session) = self.resolve_service_session_mut(requester_thread_handle, handle)
        else {
            return false;
        };
        if session.kind != ServiceSessionKind::ViU {
            return false;
        }

        session.vi_role = Some(role);
        true
    }

    pub fn open_named_service_session(
        &mut self,
        requester_thread_handle: u32,
        name: &str,
    ) -> Option<u32> {
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;
        let kind = ServiceManager::resolve_sm_service_kind(name)?;
        self.allocate_service_session_handle(process_handle, kind)
    }

    /// Clone the service session identified by `session_handle`, returning a
    /// new handle with copied session state (including domain mode/object map).
    /// This is the kernel-side implementation of CMIF control command 2/4.
    pub fn clone_service_session(
        &mut self,
        requester_thread_handle: u32,
        session_handle: u32,
    ) -> Option<u32> {
        self.clone_service_session_as(requester_thread_handle, session_handle, None)
    }

    /// Clone the service session, optionally overriding the kind.
    pub fn clone_service_session_as(
        &mut self,
        requester_thread_handle: u32,
        session_handle: u32,
        override_kind: Option<ServiceSessionKind>,
    ) -> Option<u32> {
        let mut session = self
            .resolve_service_session(requester_thread_handle, session_handle)?
            .clone();
        if let Some(kind) = override_kind {
            session.kind = kind;
        }
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;

        let mut session_id = self.next_service_session_id.max(FIRST_SERVICE_SESSION_ID);
        while self.service_sessions.contains_key(&session_id) {
            session_id = session_id.checked_add(1)?;
        }

        self.next_service_session_id = session_id
            .checked_add(1)
            .unwrap_or(FIRST_SERVICE_SESSION_ID);
        self.service_sessions.insert(session_id, session);

        if let Some(handle) =
            self.allocate_handle(process_handle, HandleObject::ServiceSession(session_id))
        {
            Some(handle)
        } else {
            self.service_sessions.remove(&session_id);
            None
        }
    }

    pub fn convert_service_session_to_domain(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<u32> {
        let session = self.resolve_service_session_mut(requester_thread_handle, handle)?;
        if !session.domain_enabled {
            session.domain_enabled = true;
            session.next_domain_object_id = 2;
            session.domain_objects.clear();
            session.event_handles.clear();
            session.domain_objects.insert(1, session.root_domain_kind());
        }

        Some(1)
    }

    pub fn service_session_domain_object_kind(
        &self,
        requester_thread_handle: u32,
        handle: u32,
        object_id: u32,
    ) -> Option<DomainObjectKind> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if !session.domain_enabled {
            return None;
        }

        session.domain_objects.get(&object_id).copied()
    }

    pub fn service_session_applet_resource_user_id(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<u64> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        Some(session.applet_resource_user_id)
    }

    pub fn service_session_set_applet_resource_user_id(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        applet_resource_user_id: u64,
    ) -> bool {
        let Some(session) = self.resolve_service_session_mut(requester_thread_handle, handle)
        else {
            return false;
        };
        session.applet_resource_user_id = applet_resource_user_id;
        true
    }

    pub fn service_session_add_domain_object(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        object_kind: DomainObjectKind,
    ) -> Option<u32> {
        let session = self.resolve_service_session_mut(requester_thread_handle, handle)?;
        if !session.domain_enabled {
            return None;
        }

        let mut object_id = session.next_domain_object_id.max(2);
        while session.domain_objects.contains_key(&object_id) {
            object_id = object_id.checked_add(1)?;
        }

        session.domain_objects.insert(object_id, object_kind);
        session.next_domain_object_id = object_id.checked_add(1).unwrap_or(2);
        Some(object_id)
    }

    pub fn service_session_remove_domain_object(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        object_id: u32,
    ) -> bool {
        let Some(session) = self.resolve_service_session_mut(requester_thread_handle, handle)
        else {
            return false;
        };
        if !session.domain_enabled || object_id == 1 {
            return false;
        }

        let removed = session.domain_objects.remove(&object_id).is_some();
        if removed {
            session
                .event_handles
                .retain(|(event_object_id, _), _| *event_object_id != object_id);
        }

        removed
    }

    pub fn service_session_get_or_create_event_handle(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        object_id: u32,
        command_id: u32,
    ) -> Option<u32> {
        let session_id = self.resolve_service_session_id(requester_thread_handle, handle)?;
        let key = (object_id, command_id);

        if let Some(cached_handle) = self
            .service_sessions
            .get(&session_id)
            .and_then(|session| session.event_handles.get(&key))
            .copied()
        {
            return Some(cached_handle);
        }

        let event_handle = self.create_event_handle(requester_thread_handle)?;
        let session = self.service_sessions.get_mut(&session_id)?;
        session.event_handles.insert(key, event_handle);
        Some(event_handle)
    }

    pub fn service_session_event_handle(
        &self,
        requester_thread_handle: u32,
        handle: u32,
        object_id: u32,
        command_id: u32,
    ) -> Option<u32> {
        let session_id = self.resolve_service_session_id(requester_thread_handle, handle)?;
        self.service_sessions
            .get(&session_id)
            .and_then(|session| session.event_handles.get(&(object_id, command_id)).copied())
    }

    pub fn applet_message_available(&self) -> bool {
        !self.applet_messages.is_empty()
    }

    pub fn applet_pop_message(&mut self) -> Option<u32> {
        self.applet_messages.pop_front()
    }

    /// Increment the frame counter and return the new value.
    pub fn increment_frame_count(&mut self) -> u64 {
        self.frame_count += 1;
        self.frame_count
    }

    /// Store a preallocated graphic buffer for a given slot (SetPreallocatedBuffer).
    pub fn set_preallocated_buffer(&mut self, slot: u32, buffer: Vec<u8>) {
        let slot = slot as usize;
        if slot >= self.binder_graphic_buffers.len() {
            self.binder_graphic_buffers.resize(slot + 1, None);
        }
        if slot >= self.binder_slot_states.len() {
            self.binder_slot_states
                .resize(slot + 1, BinderBufferState::Free);
        }
        eprintln!(
            "[binder] SetPreallocatedBuffer slot={} buffer_len={}",
            slot,
            buffer.len()
        );
        self.binder_graphic_buffers[slot] = Some(buffer);
        self.binder_slot_states[slot] = BinderBufferState::Free;
        self.binder_queue.retain(|item| item.slot as usize != slot);
        // Track how many slots are allocated.
        self.binder_num_slots = self
            .binder_graphic_buffers
            .iter()
            .rposition(|b| b.is_some())
            .map(|i| i as u32 + 1)
            .unwrap_or(0);
    }

    /// Get the preallocated graphic buffer for a given slot (RequestBuffer).
    pub fn get_preallocated_buffer(&self, slot: u32) -> Option<&[u8]> {
        self.binder_graphic_buffers
            .get(slot as usize)
            .and_then(|b| b.as_deref())
    }

    pub fn set_queued_buffer_slot(&mut self, slot: u32) {
        self.binder_queued_slot = Some(slot);
    }

    pub fn queued_buffer_slot(&self) -> Option<u32> {
        self.binder_queued_slot
    }

    pub fn set_binder_pending_frame(&mut self, frame: u64) {
        self.binder_pending_frame = Some(frame);
    }

    pub fn take_binder_pending_frame(&mut self) -> Option<u64> {
        self.binder_pending_frame.take()
    }

    pub fn queued_buffer_count(&self) -> usize {
        self.binder_queue.len()
    }

    pub fn queue_binder_buffer(&mut self, slot: u32, frame: u64, async_mode: bool) -> usize {
        self.binder_queued_slot = Some(slot);

        let slot_index = slot as usize;
        if slot_index >= self.binder_slot_states.len() {
            self.binder_slot_states
                .resize(slot_index + 1, BinderBufferState::Free);
        }
        self.binder_slot_states[slot_index] = BinderBufferState::Queued;

        // Approximate Android's droppable-front behavior for async producers.
        if async_mode && let Some(replaced) = self.binder_queue.pop_front() {
            let replaced_index = replaced.slot as usize;
            if replaced_index < self.binder_slot_states.len() {
                self.binder_slot_states[replaced_index] = BinderBufferState::Free;
            }
        }

        self.binder_queue.retain(|item| item.slot != slot);
        self.binder_queue.push_back(BinderQueueItem { slot, frame });
        self.binder_queue.len()
    }

    pub fn acquire_next_binder_buffer(&mut self) -> Option<BinderQueueItem> {
        let item = self.binder_queue.pop_front()?;
        let slot_index = item.slot as usize;
        if slot_index < self.binder_slot_states.len() {
            self.binder_slot_states[slot_index] = BinderBufferState::Acquired;
        }
        Some(item)
    }

    pub fn release_binder_buffer(&mut self, slot: u32) {
        let slot_index = slot as usize;
        if slot_index < self.binder_slot_states.len() {
            self.binder_slot_states[slot_index] = BinderBufferState::Free;
        }
    }

    /// Dequeue the next available buffer slot.
    pub fn dequeue_buffer_slot(&mut self) -> u32 {
        if self.binder_num_slots == 0 {
            return 0;
        }

        for offset in 0..self.binder_num_slots {
            let slot = (self.binder_dequeue_slot + offset) % self.binder_num_slots;
            let slot_index = slot as usize;
            let has_buffer = self
                .binder_graphic_buffers
                .get(slot_index)
                .and_then(|buffer| buffer.as_ref())
                .is_some();
            let is_free = self
                .binder_slot_states
                .get(slot_index)
                .copied()
                .unwrap_or(BinderBufferState::Free)
                == BinderBufferState::Free;
            if has_buffer && is_free {
                if slot_index >= self.binder_slot_states.len() {
                    self.binder_slot_states
                        .resize(slot_index + 1, BinderBufferState::Free);
                }
                self.binder_slot_states[slot_index] = BinderBufferState::Dequeued;
                self.binder_dequeue_slot = (slot + 1) % self.binder_num_slots;
                return slot;
            }
        }

        let slot = self.binder_dequeue_slot % self.binder_num_slots;
        let slot_index = slot as usize;
        if slot_index >= self.binder_slot_states.len() {
            self.binder_slot_states
                .resize(slot_index + 1, BinderBufferState::Free);
        }
        self.binder_slot_states[slot_index] = BinderBufferState::Dequeued;
        self.binder_dequeue_slot = (slot + 1) % self.binder_num_slots;
        eprintln!("[binder] DequeueBuffer fallback -> slot={slot}");
        slot
    }

    /// Returns all allocated NvMap handles that could be framebuffers
    /// by scanning every service session for NvdrvService data.
    pub fn framebuffer_candidates(&self) -> Vec<(u64, u32)> {
        let candidates = self.nvdrv.framebuffer_candidates();
        if !candidates.is_empty() {
            return candidates;
        }
        Vec::new()
    }

    pub fn nvmap_info(&self, handle: i32) -> Option<(u64, u32, i32)> {
        self.nvdrv.nvmap_info(handle)
    }

    pub fn nvdrv_open_device(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        path: &str,
    ) -> Option<i32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        self.nvdrv.open_device(path)
    }

    pub fn nvdrv_close_device(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        fd: i32,
    ) -> bool {
        let Some(session) = self.resolve_service_session(requester_thread_handle, handle) else {
            return false;
        };
        if session.kind != ServiceSessionKind::Nvdrv {
            return false;
        }

        self.nvdrv.close_device(fd)
    }

    pub fn nvdrv_ioctl_result(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        kind: NvdrvIoctlKind,
        fd: i32,
        ioctl: u32,
        args: &mut [u8],
        inline_buffer: &mut [u8],
    ) -> Option<u32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(
            self.nvdrv
                .ioctl_result(kind, fd, ioctl, args, inline_buffer),
        )
    }

    pub fn nvdrv_process_gpfifo(
        &mut self,
        inline_buffer: &[u8],
        memory_reader: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
        memory_writer: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.nvdrv
            .process_gpfifo_entries(inline_buffer, memory_reader, memory_writer);
    }

    pub fn nvdrv_initialize_payload(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<Vec<u8>> {
        let has_gpu_backend = self.has_gpu_backend();
        let adapter_name = self.gpu_adapter_name().map(str::to_string);

        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(
            self.nvdrv
                .initialize_payload(has_gpu_backend, adapter_name.as_deref()),
        )
    }

    pub fn nvdrv_query_event_command(
        &self,
        requester_thread_handle: u32,
        handle: u32,
        fd: i32,
        event_id: u32,
    ) -> Option<u32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        self.nvdrv.query_event_command(fd, event_id)
    }

    pub fn nvdrv_should_auto_signal_event(&self, fd: i32, event_id: u32) -> bool {
        self.nvdrv.should_auto_signal_event(fd, event_id)
    }

    pub fn nvdrv_map_shared_memory_result(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        payload: &[u8],
    ) -> Option<u32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(self.nvdrv.map_shared_memory(payload))
    }

    pub fn nvdrv_get_status_payload(
        &self,
        requester_thread_handle: u32,
        handle: u32,
        payload: &[u8],
    ) -> Option<Vec<u8>> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(self.nvdrv.get_status_payload(payload))
    }

    pub fn nvdrv_set_client_pid_result(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        payload: &[u8],
    ) -> Option<u32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(self.nvdrv.set_client_pid(payload))
    }

    pub fn nvdrv_finish_initialize_result(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        payload: &[u8],
    ) -> Option<u32> {
        let session = self.resolve_service_session(requester_thread_handle, handle)?;
        if session.kind != ServiceSessionKind::Nvdrv {
            return None;
        }

        Some(self.nvdrv.finish_initialize(payload))
    }

    pub fn create_event_handle(&mut self, requester_thread_handle: u32) -> Option<u32> {
        self.create_event_handle_with_state(requester_thread_handle, false)
    }

    fn create_event_handle_with_state(
        &mut self,
        requester_thread_handle: u32,
        signaled: bool,
    ) -> Option<u32> {
        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;
        let mut event_id = self.next_event_id.max(FIRST_EVENT_ID);
        while self.events.contains_key(&event_id) {
            event_id = event_id.checked_add(1)?;
        }

        self.events.insert(event_id, EventState { signaled });
        self.next_event_id = event_id.checked_add(1).unwrap_or(FIRST_EVENT_ID);

        if let Some(handle) = self.allocate_handle(process_handle, HandleObject::Event(event_id)) {
            eprintln!(
                "[kernel:event] created event_id={} handle={} signaled={}",
                event_id, handle, signaled
            );
            Some(handle)
        } else {
            self.events.remove(&event_id);
            None
        }
    }

    pub fn set_event_signaled_for_handle(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
        signaled: bool,
    ) -> bool {
        let Some(HandleObject::Event(event_id)) =
            self.resolve_handle_object(requester_thread_handle, handle)
        else {
            return false;
        };

        let Some(event) = self.events.get_mut(&event_id) else {
            return false;
        };

        event.signaled = signaled;

        if signaled
            && let Some(process_handle) = self.process_handle_for_thread(requester_thread_handle)
        {
            self.wake_waiting_threads_for_signaled_handle(process_handle, handle);
        }

        true
    }

    pub fn clear_event_signal_for_handle(&mut self, requester_thread_handle: u32, handle: u32) {
        let Some(HandleObject::Event(event_id)) =
            self.resolve_handle_object(requester_thread_handle, handle)
        else {
            return;
        };
        if let Some(event) = self.events.get_mut(&event_id) {
            event.signaled = false;
        }
    }

    pub fn synchronization_handle_is_signaled(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<bool> {
        match self.resolve_handle_object(requester_thread_handle, handle)? {
            HandleObject::Thread(thread_handle) => {
                let index = self.thread_index(thread_handle)?;
                Some(self.threads[index].lifecycle == ThreadLifecycle::Exited)
            }
            HandleObject::Process(process_handle) => {
                let index = self.process_index(process_handle)?;
                Some(self.processes[index].lifecycle == ProcessLifecycle::Exited)
            }
            HandleObject::Event(event_id) => self.events.get(&event_id).map(|event| event.signaled),
            HandleObject::ServiceSession(_) => Some(false),
            HandleObject::SharedMemory { .. } | HandleObject::TransferMemory { .. } => None,
        }
    }

    pub fn close_handle(&mut self, requester_thread_handle: u32, handle: u32) -> bool {
        if handle == SELF_THREAD_HANDLE || handle == SELF_PROCESS_HANDLE {
            return false;
        }

        let Some(process_handle) = self.process_handle_for_thread(requester_thread_handle) else {
            return false;
        };
        let Some(process_index) = self.process_index(process_handle) else {
            return false;
        };

        let Some(removed) = self.processes[process_index].handle_table.remove(&handle) else {
            return false;
        };

        match removed {
            HandleObject::ServiceSession(session_id) => {
                self.service_sessions.remove(&session_id);
            }
            HandleObject::Event(event_id) => {
                self.events.remove(&event_id);
            }
            HandleObject::Process(_)
            | HandleObject::Thread(_)
            | HandleObject::SharedMemory { .. }
            | HandleObject::TransferMemory { .. } => {}
        }

        true
    }

    #[cfg(test)]
    pub fn process_main_thread_handle(
        &self,
        requester_thread_handle: u32,
        process_handle: u32,
    ) -> Option<u32> {
        let process_handle =
            self.resolve_process_handle(requester_thread_handle, process_handle)?;
        let process = self.processes.iter().find(|process| {
            process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
        })?;

        process
            .handle_table
            .iter()
            .find_map(|(handle, object)| match object {
                HandleObject::Thread(thread_handle)
                    if *thread_handle == process.main_thread_handle =>
                {
                    Some(*handle)
                }
                _ => None,
            })
    }

    #[cfg(test)]
    pub fn process_bootstrap_for_handle(
        &self,
        requester_thread_handle: u32,
        process_handle: u32,
    ) -> Option<(u64, u64, u64)> {
        let process_handle =
            self.resolve_process_handle(requester_thread_handle, process_handle)?;
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| (process.entrypoint, process.argument, process.stack_top))
    }

    #[cfg(test)]
    pub fn is_thread_safepoint_parked(&self, thread_handle: u32) -> bool {
        self.thread_index(thread_handle)
            .map(|index| self.threads[index].lifecycle == ThreadLifecycle::Safepoint)
            .unwrap_or(false)
    }

    pub fn waiter_count_for_address(&self, thread_handle: u32, address: u64) -> usize {
        let Some(process_handle) = self.process_handle_for_thread(thread_handle) else {
            return 0;
        };

        let key = AddressWaitKey {
            process_handle,
            address,
        };
        self.address_waiters
            .get(&key)
            .map(|waiters| waiters.len())
            .unwrap_or(0)
    }

    pub fn resolve_process_handle(
        &self,
        requester_thread_handle: u32,
        process_handle: u32,
    ) -> Option<u32> {
        match self.resolve_handle_object(requester_thread_handle, process_handle)? {
            HandleObject::Process(handle) => Some(handle),
            HandleObject::Thread(_)
            | HandleObject::ServiceSession(_)
            | HandleObject::Event(_)
            | HandleObject::SharedMemory { .. }
            | HandleObject::TransferMemory { .. } => None,
        }
    }

    pub fn terminate_process(&mut self, requester_thread_handle: u32, process_handle: u32) -> bool {
        let Some(process_handle) =
            self.resolve_process_handle(requester_thread_handle, process_handle)
        else {
            return false;
        };

        self.terminate_process_internal(process_handle)
    }

    fn terminate_process_internal(&mut self, process_handle: u32) -> bool {
        let Some(process_index) = self
            .processes
            .iter()
            .position(|process| process.handle == process_handle)
        else {
            return false;
        };

        if self.processes[process_index].lifecycle == ProcessLifecycle::Exited {
            return true;
        }

        let thread_handles: Vec<u32> = self
            .threads
            .iter()
            .filter(|thread| {
                thread.process_handle == process_handle
                    && thread.lifecycle != ThreadLifecycle::Exited
            })
            .map(|thread| thread.handle)
            .collect();

        for handle in thread_handles {
            if let Some(index) = self.thread_index(handle) {
                let context = self.threads[index].context.clone();
                self.exit_thread_internal(handle, &context);
            }
        }

        self.processes[process_index].lifecycle = ProcessLifecycle::Exited;
        true
    }

    pub fn exit_process(&mut self, thread_handle: u32, state: &CpuState) -> bool {
        let Some(process_handle) = self.process_handle_for_thread(thread_handle) else {
            return false;
        };

        let thread_handles: Vec<u32> = self
            .threads
            .iter()
            .filter(|thread| {
                thread.process_handle == process_handle
                    && thread.lifecycle != ThreadLifecycle::Exited
            })
            .map(|thread| thread.handle)
            .collect();

        for handle in thread_handles {
            if handle == thread_handle {
                self.exit_thread_internal(handle, state);
            } else if let Some(index) = self.thread_index(handle) {
                let context = self.threads[index].context.clone();
                self.exit_thread_internal(handle, &context);
            }
        }

        if let Some(process_index) = self
            .processes
            .iter()
            .position(|process| process.handle == process_handle)
        {
            self.processes[process_index].lifecycle = ProcessLifecycle::Exited;
        }

        process_handle == MAIN_PROCESS_HANDLE
    }

    pub fn take_worker_trap(&mut self) -> Option<WorkerTrap> {
        self.worker_trap.take()
    }

    pub fn handle_supervisor_call(
        &mut self,
        start_pc: u64,
        thread_handle: u32,
        imm16: u16,
        state: &mut CpuState,
        memory: &SharedGuestMemory,
    ) -> StepResult {
        match dispatch_supervisor_call(self, thread_handle, imm16, state, memory) {
            SyscallDisposition::Continue => {
                self.sync_thread_context(thread_handle, state);
                StepResult::Advanced {
                    start_pc,
                    end_pc: state.pc,
                }
            }
            SyscallDisposition::ThreadExit => {
                self.exit_thread_internal(thread_handle, state);
                if thread_handle == MAIN_THREAD_HANDLE {
                    StepResult::Trap {
                        pc: state.pc,
                        reason: ExitReason::ThreadExit,
                    }
                } else {
                    StepResult::Advanced {
                        start_pc,
                        end_pc: state.pc,
                    }
                }
            }
            SyscallDisposition::Unhandled => {
                state.clear_exit();
                StepResult::Trap {
                    pc: state.pc,
                    reason: ExitReason::UnknownSupervisorCall(imm16),
                }
            }
        }
    }

    pub fn resolve_thread_step(
        &mut self,
        thread_handle: u32,
        step: StepResult,
        state: &mut CpuState,
        memory: &SharedGuestMemory,
    ) -> StepResult {
        match step {
            StepResult::Trap {
                reason: ExitReason::SupervisorCall(imm16),
                pc,
            } => self.handle_supervisor_call(pc, thread_handle, imm16, state, memory),
            other => other,
        }
    }

    pub fn create_thread(
        &mut self,
        creator_handle: u32,
        entrypoint: u64,
        argument: u64,
        stack_top: u64,
        state_template: &CpuState,
        priority: i32,
    ) -> u32 {
        let process_handle = self
            .thread_index(creator_handle)
            .map(|index| self.threads[index].process_handle)
            .unwrap_or(MAIN_PROCESS_HANDLE);
        self.sync_thread_context(creator_handle, state_template);

        let mut thread_state = state_template.clone();
        thread_state.pc = entrypoint;
        thread_state.sp = stack_top & !0xf;
        let thread_ptr = stack_top & !(THREAD_TLS_PAGE_SIZE - 1);
        thread_state.tpidr_el0 = 0;
        thread_state.tpidrro_el0 = thread_ptr;
        thread_state.tpidr_el1 = 0;
        thread_state.write_x(0, argument);
        thread_state.write_x(18, 1);
        thread_state.write_x(30, state_template.read_x(30));
        thread_state.clear_exit();

        let thread_handle = self.next_thread_handle;
        self.next_thread_handle = self.next_thread_handle.saturating_add(1);
        let thread_uid = self.next_thread_uid;
        self.next_thread_uid = self.next_thread_uid.saturating_add(1);
        let affinity_mask = configured_core_mask();
        let core_id = random_core_id_from_mask(thread_uid, affinity_mask);

        let handle = self
            .allocate_handle(process_handle, HandleObject::Thread(thread_handle))
            .unwrap_or(0);
        thread_state.write_x(1, u64::from(handle));

        self.threads.push(GuestThread {
            handle: thread_handle,
            process_handle,
            thread_uid,
            priority,
            core_id,
            affinity_mask,
            core_binding_fixed: false,
            lifecycle: ThreadLifecycle::Created,
            running_on_worker: false,
            wait_address: None,
            address_key_address: None,
            address_key_value: 0,
            wait_sync_handles: Vec::new(),
            wait_sync_out_index_ptr: None,
            wake_deadline: None,
            context: thread_state,
        });

        handle
    }

    pub fn create_process(
        &mut self,
        creator_handle: u32,
        entrypoint: u64,
        argument: u64,
        stack_top: u64,
        state_template: &CpuState,
        memory: SharedGuestMemory,
    ) -> (u32, u32) {
        let creator_process_handle = self
            .thread_index(creator_handle)
            .map(|index| self.threads[index].process_handle)
            .unwrap_or(MAIN_PROCESS_HANDLE);
        let creator_priority = self
            .thread_index(creator_handle)
            .map(|index| self.threads[index].priority)
            .unwrap_or(DEFAULT_THREAD_PRIORITY);
        self.sync_thread_context(creator_handle, state_template);

        let process_handle = self.next_process_handle;
        self.next_process_handle = self.next_process_handle.saturating_add(1);
        let pid = self.next_process_pid;
        self.next_process_pid = self.next_process_pid.saturating_add(1);

        let mut thread_state = state_template.clone();
        thread_state.pc = entrypoint;
        thread_state.sp = stack_top & !0xf;
        let thread_ptr = stack_top & !(THREAD_TLS_PAGE_SIZE - 1);
        thread_state.tpidr_el0 = 0;
        thread_state.tpidrro_el0 = thread_ptr;
        thread_state.tpidr_el1 = 0;
        thread_state.write_x(0, argument);
        thread_state.write_x(18, 1);
        thread_state.write_x(30, state_template.read_x(30));
        thread_state.clear_exit();
        let _ = memory.write_u64(thread_ptr.wrapping_add(0x1f8), thread_ptr);
        let _ = memory.write_u32(thread_ptr.wrapping_add(0x1b0), MAIN_THREAD_HANDLE);

        let thread_handle = self.next_thread_handle;
        self.next_thread_handle = self.next_thread_handle.saturating_add(1);
        let thread_uid = self.next_thread_uid;
        self.next_thread_uid = self.next_thread_uid.saturating_add(1);
        let affinity_mask = configured_core_mask();
        let core_id = random_core_id_from_mask(thread_uid, affinity_mask);

        let mut process_handle_table = HashMap::new();
        process_handle_table.insert(MAIN_THREAD_HANDLE, HandleObject::Thread(thread_handle));
        thread_state.write_x(1, u64::from(MAIN_THREAD_HANDLE));

        self.processes.push(GuestProcess {
            handle: process_handle,
            pid,
            lifecycle: ProcessLifecycle::Created,
            main_thread_handle: thread_handle,
            entrypoint,
            argument,
            stack_top: thread_state.sp,
            random_entropy: entropy_words(pid),
            next_user_handle: FIRST_USER_HANDLE,
            handle_table: process_handle_table,
            memory,
            heap_region_start: 0,
            heap_region_end: 0,
            heap_capacity: 0,
            current_heap_addr: 0,
            image_size: 0,
            main_thread_stack_size: 0,
        });

        self.threads.push(GuestThread {
            handle: thread_handle,
            process_handle,
            thread_uid,
            priority: creator_priority,
            core_id,
            affinity_mask,
            core_binding_fixed: false,
            lifecycle: ThreadLifecycle::Created,
            running_on_worker: false,
            wait_address: None,
            address_key_address: None,
            address_key_value: 0,
            wait_sync_handles: Vec::new(),
            wait_sync_out_index_ptr: None,
            wake_deadline: None,
            context: thread_state,
        });

        let process_user_handle = self
            .allocate_handle(
                creator_process_handle,
                HandleObject::Process(process_handle),
            )
            .unwrap_or(0);
        let thread_user_handle = self
            .allocate_handle(creator_process_handle, HandleObject::Thread(thread_handle))
            .unwrap_or(0);

        (process_user_handle, thread_user_handle)
    }

    pub fn start_process(&mut self, requester_thread_handle: u32, process_handle: u32) -> bool {
        let Some(process_handle) =
            self.resolve_process_handle(requester_thread_handle, process_handle)
        else {
            return false;
        };

        let Some(process_index) = self
            .processes
            .iter()
            .position(|process| process.handle == process_handle)
        else {
            return false;
        };

        if self.processes[process_index].lifecycle == ProcessLifecycle::Exited {
            return false;
        }

        self.processes[process_index].lifecycle = ProcessLifecycle::Runnable;
        let (main_thread_handle, entrypoint, argument, stack_top) = {
            let process = &self.processes[process_index];
            (
                process.main_thread_handle,
                process.entrypoint,
                process.argument,
                process.stack_top,
            )
        };

        if let Some(index) = self.thread_index(main_thread_handle) {
            if self.threads[index].lifecycle == ThreadLifecycle::Created {
                let thread = &mut self.threads[index];
                thread.context.pc = entrypoint;
                thread.context.sp = stack_top;
                thread.context.write_x(0, argument);
                thread.context.write_x(1, u64::from(MAIN_THREAD_HANDLE));
                thread.context.clear_exit();
                thread.lifecycle =
                    if self.safepoint_requested && main_thread_handle != MAIN_THREAD_HANDLE {
                        ThreadLifecycle::Safepoint
                    } else {
                        ThreadLifecycle::Runnable
                    };
                if main_thread_handle != MAIN_THREAD_HANDLE
                    && self.threads[index].lifecycle == ThreadLifecycle::Runnable
                {
                    self.enqueue_thread(main_thread_handle);
                }
            }
        }

        let thread_handles: Vec<u32> = self
            .threads
            .iter()
            .filter(|thread| {
                thread.process_handle == process_handle
                    && thread.lifecycle == ThreadLifecycle::Created
                    && thread.handle != main_thread_handle
            })
            .map(|thread| thread.handle)
            .collect();

        for handle in thread_handles {
            if let Some(index) = self.thread_index(handle) {
                self.threads[index].lifecycle = if self.safepoint_requested {
                    ThreadLifecycle::Safepoint
                } else {
                    ThreadLifecycle::Runnable
                };
                if handle != MAIN_THREAD_HANDLE
                    && self.threads[index].lifecycle == ThreadLifecycle::Runnable
                {
                    self.enqueue_thread(handle);
                }
            }
        }

        true
    }

    pub fn start_thread(&mut self, requester_thread_handle: u32, handle: u32) -> bool {
        let Some(handle) = self.resolve_thread_handle(requester_thread_handle, handle) else {
            return false;
        };

        let Some(index) = self.thread_index(handle) else {
            return false;
        };

        let process_handle = self.threads[index].process_handle;
        if !self.is_process_runnable(process_handle) {
            return false;
        }

        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return false;
        }

        self.threads[index].lifecycle = if self.safepoint_requested && handle != MAIN_THREAD_HANDLE
        {
            ThreadLifecycle::Safepoint
        } else {
            ThreadLifecycle::Runnable
        };
        let process_handle = self.threads[index].process_handle;
        if handle != MAIN_THREAD_HANDLE
            && self.is_process_runnable(process_handle)
            && self.threads[index].lifecycle == ThreadLifecycle::Runnable
        {
            self.enqueue_thread(handle);
        }

        true
    }

    pub fn exit_thread(&mut self, handle: u32, state: &CpuState) {
        self.exit_thread_internal(handle, state);
    }

    pub fn yield_thread(&mut self, handle: u32, state: &CpuState) {
        self.sync_thread_context(handle, state);
        if handle != MAIN_THREAD_HANDLE && self.is_thread_runnable(handle) {
            if self.safepoint_requested {
                if let Some(index) = self.thread_index(handle) {
                    self.threads[index].lifecycle = ThreadLifecycle::Safepoint;
                }
            } else {
                self.enqueue_thread(handle);
            }
        }
    }

    pub fn sleep_thread(&mut self, handle: u32, state: &CpuState, timeout_ns: i64) {
        self.sync_thread_context(handle, state);

        // timeout_ns == 0: yield (let other threads run) and return immediately.
        if timeout_ns == 0 {
            self.yield_thread(handle, state);
            return;
        }

        // timeout_ns < 0 (e.g. -1 / UINT64_MAX): block the thread forever
        // (no wake deadline).  This is used by _appletInfiniteSleepLoop in
        // libnx's clean exit path — the thread must NOT return.
        // timeout_ns > 0: block until the deadline elapses.
        if let Some(index) = self.thread_index(handle) {
            let thread = &mut self.threads[index];
            if thread.lifecycle == ThreadLifecycle::Exited {
                return;
            }

            thread.lifecycle = ThreadLifecycle::Blocked;
            thread.wait_address = None;
            thread.address_key_address = None;
            thread.address_key_value = 0;
            thread.wait_sync_handles.clear();
            thread.wait_sync_out_index_ptr = None;
            thread.wake_deadline = if timeout_ns > 0 {
                Some(Instant::now() + Duration::from_nanos(timeout_ns as u64))
            } else {
                None // infinite sleep — no deadline, thread stays blocked
            };
            thread.running_on_worker = false;
        }
    }

    pub fn wait_for_synchronization_handles(
        &mut self,
        handle: u32,
        state: &CpuState,
        sync_handles: &[u32],
        out_index_ptr: u64,
        timeout_ns: i64,
    ) {
        self.sync_thread_context(handle, state);

        if let Some(index) = self.thread_index(handle) {
            let thread = &mut self.threads[index];
            if thread.lifecycle == ThreadLifecycle::Exited {
                return;
            }

            thread.lifecycle = ThreadLifecycle::Blocked;
            thread.wait_address = None;
            thread.address_key_address = None;
            thread.address_key_value = 0;
            thread.wait_sync_handles.clear();
            thread.wait_sync_handles.extend_from_slice(sync_handles);
            thread.wait_sync_out_index_ptr = Some(out_index_ptr);
            thread.wake_deadline = if timeout_ns > 0 {
                Some(Instant::now() + Duration::from_nanos(timeout_ns as u64))
            } else {
                None
            };
            thread.running_on_worker = false;
        }
    }

    pub fn wait_for_address(
        &mut self,
        handle: u32,
        state: &CpuState,
        address: u64,
        timeout_ns: i64,
    ) {
        self.wait_for_address_with_key(handle, state, address, timeout_ns, address, 0);
    }

    pub fn wait_for_address_with_key(
        &mut self,
        handle: u32,
        state: &CpuState,
        wait_address: u64,
        timeout_ns: i64,
        key_address: u64,
        key_value: u32,
    ) {
        self.sync_thread_context(handle, state);

        if let Some(index) = self.thread_index(handle) {
            let thread = &mut self.threads[index];
            if thread.lifecycle == ThreadLifecycle::Exited {
                return;
            }

            thread.lifecycle = ThreadLifecycle::Blocked;
            thread.wait_address = Some(wait_address);
            thread.address_key_address = Some(key_address);
            thread.address_key_value = key_value;
            thread.wait_sync_handles.clear();
            thread.wait_sync_out_index_ptr = None;
            thread.wake_deadline = if timeout_ns > 0 {
                Some(Instant::now() + Duration::from_nanos(timeout_ns as u64))
            } else {
                None
            };
            thread.running_on_worker = false;

            let key = AddressWaitKey {
                process_handle: thread.process_handle,
                address: wait_address,
            };
            self.address_waiters
                .entry(key)
                .or_default()
                .push_back(handle);
        }
    }

    pub fn thread_address_key(&self, thread_handle: u32) -> Option<(u64, u32)> {
        let index = self.thread_index(thread_handle)?;
        let thread = &self.threads[index];
        if thread.lifecycle == ThreadLifecycle::Exited {
            return None;
        }

        Some((thread.address_key_address?, thread.address_key_value))
    }

    /// Check if the thread with process-level handle `game_handle`
    /// (looked up from `requester_handle`'s process) is Blocked and
    /// waiting on some address.  Returns the `wait_address` if so.
    pub fn thread_blocked_on_address(
        &self,
        requester_handle: u32,
        game_handle: u32,
    ) -> Option<u64> {
        let internal_handle = self.resolve_thread_handle(requester_handle, game_handle)?;
        let index = self.thread_index(internal_handle)?;
        let thread = &self.threads[index];
        if thread.lifecycle != ThreadLifecycle::Blocked {
            return None;
        }
        thread.wait_address
    }

    /// Like `thread_blocked_on_address`, but only returns the address if
    /// the thread is blocked on a **mutex** wait (svc_arbitrate_lock),
    /// not a condvar wait (svc_wait_process_wide_key_atomic).
    pub fn move_blocked_thread_to_address_wait(
        &mut self,
        thread_handle: u32,
        wait_address: u64,
        timeout_ns: i64,
    ) -> bool {
        let Some(index) = self.thread_index(thread_handle) else {
            return false;
        };

        if self.threads[index].lifecycle != ThreadLifecycle::Blocked {
            return false;
        }

        if let Some(current_wait_address) = self.threads[index].wait_address {
            let process_handle = self.threads[index].process_handle;
            self.remove_from_wait_queue(thread_handle, process_handle, current_wait_address);
        }

        let process_handle = self.threads[index].process_handle;
        self.threads[index].wait_address = Some(wait_address);
        self.threads[index].wait_sync_handles.clear();
        self.threads[index].wait_sync_out_index_ptr = None;
        self.threads[index].wake_deadline = if timeout_ns > 0 {
            Some(Instant::now() + Duration::from_nanos(timeout_ns as u64))
        } else {
            None
        };
        self.threads[index].running_on_worker = false;

        self.address_waiters
            .entry(AddressWaitKey {
                process_handle,
                address: wait_address,
            })
            .or_default()
            .push_back(thread_handle);
        true
    }

    pub fn take_waiters_for_address(
        &mut self,
        signaler_handle: u32,
        address: u64,
        count: i32,
    ) -> Vec<u32> {
        let Some(process_handle) = self.process_handle_for_thread(signaler_handle) else {
            return Vec::new();
        };

        let wake_count = if count <= 0 {
            usize::MAX
        } else {
            count as usize
        };

        let key = AddressWaitKey {
            process_handle,
            address,
        };

        let mut handles = Vec::new();
        let mut should_remove = false;

        if let Some(waiters) = self.address_waiters.get_mut(&key) {
            while handles.len() < wake_count {
                let Some(handle) = waiters.pop_front() else {
                    break;
                };
                handles.push(handle);
            }
            should_remove = waiters.is_empty();
        }

        if should_remove {
            self.address_waiters.remove(&key);
        }

        handles
    }

    pub fn wake_thread(&mut self, thread_handle: u32) -> bool {
        self.unblock_thread(thread_handle)
    }

    pub fn signal_to_address(&mut self, signaler_handle: u32, address: u64, count: i32) -> usize {
        let handles = self.take_waiters_for_address(signaler_handle, address, count);

        let mut woken = 0usize;
        for handle in handles {
            if self.unblock_thread(handle) {
                woken += 1;
            }
        }

        woken
    }

    pub fn wake_timed_threads(&mut self) {
        let now = Instant::now();
        let mut ready_handles = Vec::new();

        for thread in &mut self.threads {
            if thread.lifecycle != ThreadLifecycle::Blocked {
                continue;
            }

            if let Some(deadline) = thread.wake_deadline {
                if deadline <= now {
                    ready_handles.push(thread.handle);
                }
            }
        }

        for handle in ready_handles {
            let _ = self.unblock_thread(handle);
        }
    }

    fn wake_waiting_threads_for_signaled_handle(
        &mut self,
        process_handle: u32,
        signaled_handle: u32,
    ) {
        let mut to_wake = Vec::new();

        for thread in &self.threads {
            if thread.lifecycle != ThreadLifecycle::Blocked
                || thread.process_handle != process_handle
            {
                continue;
            }

            if thread.wait_sync_handles.is_empty() || thread.wait_sync_out_index_ptr.is_none() {
                continue;
            }

            if !thread.wait_sync_handles.contains(&signaled_handle) {
                continue;
            }

            let selected_index =
                thread
                    .wait_sync_handles
                    .iter()
                    .enumerate()
                    .find_map(|(index, handle)| {
                        self.synchronization_handle_is_signaled_for_process(process_handle, *handle)
                            .filter(|signaled| *signaled)
                            .map(|_| index as u32)
                    });

            if let Some(selected_index) = selected_index {
                to_wake.push((
                    thread.handle,
                    thread.wait_sync_out_index_ptr.unwrap_or(0),
                    selected_index,
                ));
            }
        }

        if to_wake.is_empty() {
            return;
        }

        let Some(process_index) = self.process_index(process_handle) else {
            return;
        };
        let process_memory = self.processes[process_index].memory.clone();

        for (thread_handle, out_index_ptr, selected_index) in to_wake {
            let _ = process_memory.write_u32(out_index_ptr, selected_index);
            if let Some(index) = self.thread_index(thread_handle) {
                self.threads[index].context.write_x(0, 0);
                self.threads[index]
                    .context
                    .write_x(1, u64::from(selected_index));
            }
            let _ = self.unblock_thread(thread_handle);
        }
    }

    pub fn time_until_next_wakeup(&self) -> Option<Duration> {
        let now = Instant::now();
        self.threads
            .iter()
            .filter_map(|thread| {
                if thread.lifecycle == ThreadLifecycle::Blocked {
                    thread.wake_deadline
                } else {
                    None
                }
            })
            .min()
            .map(|deadline| deadline.saturating_duration_since(now))
    }

    #[allow(unused)]
    pub fn take_runnable_worker_thread(&mut self) -> Option<WorkerTask> {
        if self.safepoint_requested {
            return None;
        }

        while let Some(handle) = self.ready_queue.pop_front() {
            let Some(index) = self.thread_index(handle) else {
                continue;
            };

            let thread = &mut self.threads[index];
            if thread.lifecycle != ThreadLifecycle::Runnable || thread.running_on_worker {
                continue;
            }

            let Some(process_index) = self
                .processes
                .iter()
                .position(|process| process.handle == thread.process_handle)
            else {
                continue;
            };

            if self.processes[process_index].lifecycle != ProcessLifecycle::Runnable {
                continue;
            }

            thread.running_on_worker = true;
            return Some(WorkerTask {
                thread_handle: handle,
                process_handle: thread.process_handle,
                state: thread.context.clone(),
                memory: self.processes[process_index].memory.clone(),
            });
        }

        None
    }

    pub fn take_runnable_worker_thread_for(&mut self, thread_handle: u32) -> Option<WorkerTask> {
        if self.safepoint_requested {
            return None;
        }

        let index = self.thread_index(thread_handle)?;
        if self.threads[index].lifecycle != ThreadLifecycle::Runnable
            || self.threads[index].running_on_worker
        {
            return None;
        }

        let process_index = self
            .processes
            .iter()
            .position(|process| process.handle == self.threads[index].process_handle)?;
        if self.processes[process_index].lifecycle != ProcessLifecycle::Runnable {
            return None;
        }

        if let Some(position) = self
            .ready_queue
            .iter()
            .position(|queued| *queued == thread_handle)
        {
            let _ = self.ready_queue.remove(position);
        }

        self.threads[index].running_on_worker = true;
        Some(WorkerTask {
            thread_handle,
            process_handle: self.threads[index].process_handle,
            state: self.threads[index].context.clone(),
            memory: self.processes[process_index].memory.clone(),
        })
    }

    pub fn complete_worker_step(
        &mut self,
        handle: u32,
        process_handle: u32,
        mut state: CpuState,
        step: StepResult,
        memory: &SharedGuestMemory,
    ) {
        let Some(index) = self.thread_index(handle) else {
            return;
        };

        if self.threads[index].process_handle != process_handle {
            return;
        }

        self.threads[index].running_on_worker = false;
        if self.threads[index].lifecycle == ThreadLifecycle::Exited {
            return;
        }

        let resolved = self.resolve_thread_step(handle, step, &mut state, memory);
        Self::trace_worker_step(handle, resolved, &state);
        self.handle_worker_resolved_step(handle, state, resolved, memory);
    }

    fn handle_worker_resolved_step(
        &mut self,
        handle: u32,
        state: CpuState,
        step: StepResult,
        memory: &SharedGuestMemory,
    ) {
        let Some(index) = self.thread_index(handle) else {
            return;
        };

        match step {
            StepResult::Advanced { .. } => {
                if self.threads[index].lifecycle == ThreadLifecycle::Runnable {
                    self.threads[index].context = state;
                    if self.safepoint_requested && handle != MAIN_THREAD_HANDLE {
                        self.threads[index].lifecycle = ThreadLifecycle::Safepoint;
                    } else {
                        self.enqueue_thread(handle);
                    }
                }
            }
            StepResult::Trap {
                reason: ExitReason::ThreadExit,
                ..
            } => {
                eprintln!(
                    "[worker-exit] thread {} ThreadExit at pc=0x{:016x}",
                    handle, state.pc
                );
                self.exit_thread_internal(handle, &state);
            }
            StepResult::Trap { pc, reason } => {
                eprintln!(
                    "[worker-trap] thread {} TRAP at pc=0x{pc:016x} reason={reason:?} x0=0x{:016x} x1=0x{:016x} x2=0x{:016x} x3=0x{:016x} x8=0x{:016x} x19=0x{:016x} x20=0x{:016x} x21=0x{:016x} x22=0x{:016x} x30=0x{:016x} sp=0x{:016x}",
                    handle,
                    state.read_x(0),
                    state.read_x(1),
                    state.read_x(2),
                    state.read_x(3),
                    state.read_x(8),
                    state.read_x(19),
                    state.read_x(20),
                    state.read_x(21),
                    state.read_x(22),
                    state.read_x(30),
                    state.sp,
                );

                // For MemoryFault on worker threads, try to fix up and retry.
                let recovered = false;

                if !recovered {
                    // Dump instructions around the crash site for disassembly analysis.
                    let code_start = pc.saturating_sub(128);
                    let code_end = pc + 64;
                    let mut code_dump = String::new();
                    for addr in (code_start..code_end).step_by(4) {
                        if let Some(insn) = memory.read_u32(addr) {
                            let marker = if addr == pc { " <-- FAULT" } else { "" };
                            code_dump
                                .push_str(&format!("  [{:016x}] 0x{:08x}{}\n", addr, insn, marker));
                        }
                    }
                    eprintln!(
                        "[worker-trap] code dump around pc=0x{:016x}:\n{}",
                        pc, code_dump
                    );

                    // Also dump caller instructions around x30 to see the call site.
                    let x30 = state.read_x(30);
                    let caller_start = x30.saturating_sub(32);
                    let caller_end = x30 + 32;
                    let mut caller_dump = String::new();
                    for addr in (caller_start..caller_end).step_by(4) {
                        if let Some(insn) = memory.read_u32(addr) {
                            let marker = if addr == x30 { " <-- RETURN" } else { "" };
                            caller_dump
                                .push_str(&format!("  [{:016x}] 0x{:08x}{}\n", addr, insn, marker));
                        }
                    }
                    eprintln!(
                        "[worker-trap] caller code around x30=0x{:016x}:\n{}",
                        x30, caller_dump
                    );

                    // Dump memory around x1 (array base) for crash analysis.
                    let x1 = state.read_x(1);
                    if x1 > 0x80000000 && x1 < 0x900000000 {
                        let dump_start = x1.saturating_sub(16);
                        let dump_end = x1 + 256;
                        let mut dump = String::new();
                        for addr in (dump_start..dump_end).step_by(8) {
                            if let Some(val) = memory.read_u64(addr) {
                                dump.push_str(&format!("  [{:016x}] = 0x{:016x}\n", addr, val));
                            }
                        }
                        eprintln!(
                            "[worker-trap] memory dump around x1=0x{:016x}:\n{}",
                            x1, dump
                        );
                    }
                    // Dump memory around x20 (saved original x0 = 'this' pointer).
                    let x20 = state.read_x(20);
                    if x20 > 0x80000000 && x20 < 0x900000000 {
                        let mut dump = String::new();
                        for addr in (x20..x20 + 256).step_by(8) {
                            if let Some(val) = memory.read_u64(addr) {
                                dump.push_str(&format!("  [{:016x}] = 0x{:016x}\n", addr, val));
                            }
                        }
                        eprintln!(
                            "[worker-trap] memory dump around x20=0x{:016x}:\n{}",
                            x20, dump
                        );
                    }
                    // Dump stack around sp to find return addresses.
                    let sp = state.sp;
                    if sp > 0x80000000 && sp < 0x900000000 {
                        let mut dump = String::new();
                        for addr in (sp..sp + 256).step_by(8) {
                            if let Some(val) = memory.read_u64(addr) {
                                dump.push_str(&format!("  [{:016x}] = 0x{:016x}\n", addr, val));
                            }
                        }
                        eprintln!("[worker-trap] stack dump sp=0x{:016x}:\n{}", sp, dump);
                    }
                    self.exit_thread_internal(handle, &state);
                    // Don't propagate worker traps as fatal — cooperative
                    // scheduling can cause spurious MemoryFault crashes
                    // when worker threads race ahead of initialization
                    // that the main thread hasn't completed yet.
                    // Just log and continue; the thread is already exited.
                    eprintln!(
                        "[worker-trap] thread {} exited due to {:?} at pc=0x{:016x} (non-fatal)",
                        handle, reason, pc
                    );

                    // If the main thread was yield-waiting on a mutex
                    // owned by this crashed thread, release the mutex so
                    // the main thread can continue.
                    eprintln!(
                        "[worker-trap] checking yield mutex: main_yield={:?} handle={}",
                        self.main_thread_yield_mutex, handle
                    );
                    if let Some(yield_addr) = self.main_thread_yield_mutex {
                        if let Some(mutex_val) = memory.read_u32(yield_addr) {
                            let owner_game_handle = mutex_val & !HANDLE_WAIT_MASK;
                            let resolved =
                                self.resolve_thread_handle(MAIN_THREAD_HANDLE, owner_game_handle);
                            eprintln!(
                                "[worker-trap] yield_addr=0x{:x} mutex_val=0x{:x} owner_game={} resolved={:?} crashed_internal={}",
                                yield_addr, mutex_val, owner_game_handle, resolved, handle
                            );
                            // Check if this game-visible handle resolves to
                            // the internal handle of the crashed thread.
                            let is_crashed_thread =
                                owner_game_handle != 0 && resolved == Some(handle);
                            if is_crashed_thread {
                                eprintln!(
                                    "[worker-trap] releasing mutex 0x{:016x} held by crashed thread {} (game handle {})",
                                    yield_addr, handle, owner_game_handle
                                );
                                memory.write_u32(yield_addr, 0);
                                self.main_thread_yield_mutex = None;
                            }
                        }
                    }
                }
            }
        }
    }

    fn enqueue_thread(&mut self, handle: u32) {
        if !self.ready_queue.contains(&handle) {
            self.ready_queue.push_back(handle);
        }
    }

    fn remove_from_wait_queue(&mut self, handle: u32, process_handle: u32, address: u64) {
        let key = AddressWaitKey {
            process_handle,
            address,
        };
        let should_remove = {
            let Some(waiters) = self.address_waiters.get_mut(&key) else {
                return;
            };

            if let Some(position) = waiters.iter().position(|queued| *queued == handle) {
                waiters.remove(position);
            }

            waiters.is_empty()
        };

        if should_remove {
            self.address_waiters.remove(&key);
        }
    }

    fn unblock_thread(&mut self, handle: u32) -> bool {
        let Some(index) = self.thread_index(handle) else {
            return false;
        };

        if self.threads[index].lifecycle != ThreadLifecycle::Blocked {
            return false;
        }

        if let Some(address) = self.threads[index].wait_address {
            let process_handle = self.threads[index].process_handle;
            self.remove_from_wait_queue(handle, process_handle, address);
        }

        self.threads[index].lifecycle = if self.safepoint_requested && handle != MAIN_THREAD_HANDLE
        {
            ThreadLifecycle::Safepoint
        } else {
            ThreadLifecycle::Runnable
        };
        self.threads[index].wait_address = None;
        self.threads[index].address_key_address = None;
        self.threads[index].address_key_value = 0;
        self.threads[index].wait_sync_handles.clear();
        self.threads[index].wait_sync_out_index_ptr = None;
        self.threads[index].wake_deadline = None;

        if handle != MAIN_THREAD_HANDLE
            && self.threads[index].lifecycle == ThreadLifecycle::Runnable
        {
            self.enqueue_thread(handle);
        }

        true
    }

    fn is_thread_runnable(&self, handle: u32) -> bool {
        self.thread_index(handle)
            .map(|index| self.threads[index].lifecycle == ThreadLifecycle::Runnable)
            .unwrap_or(false)
    }

    fn is_process_runnable(&self, process_handle: u32) -> bool {
        self.processes
            .iter()
            .find(|process| process.handle == process_handle)
            .map(|process| process.lifecycle == ProcessLifecycle::Runnable)
            .unwrap_or(false)
    }

    fn sync_thread_context(&mut self, handle: u32, state: &CpuState) {
        if let Some(index) = self.thread_index(handle) {
            if self.threads[index].lifecycle != ThreadLifecycle::Exited {
                self.threads[index].context = state.clone();
            }
        }
    }

    fn exit_thread_internal(&mut self, handle: u32, state: &CpuState) {
        if let Some(index) = self.thread_index(handle) {
            if let Some(address) = self.threads[index].wait_address {
                let process_handle = self.threads[index].process_handle;
                self.remove_from_wait_queue(handle, process_handle, address);
            }
            self.threads[index].context = state.clone();
            self.threads[index].lifecycle = ThreadLifecycle::Exited;
            self.threads[index].running_on_worker = false;
            self.threads[index].wait_address = None;
            self.threads[index].address_key_address = None;
            self.threads[index].address_key_value = 0;
            self.threads[index].wait_sync_handles.clear();
            self.threads[index].wait_sync_out_index_ptr = None;
            self.threads[index].wake_deadline = None;
        }
    }

    fn synchronization_handle_is_signaled_for_process(
        &self,
        process_handle: u32,
        handle: u32,
    ) -> Option<bool> {
        let process_index = self.process_index(process_handle)?;
        let object = self.processes[process_index]
            .handle_table
            .get(&handle)
            .copied()?;

        match object {
            HandleObject::Thread(thread_handle) => {
                let index = self.thread_index(thread_handle)?;
                Some(self.threads[index].lifecycle == ThreadLifecycle::Exited)
            }
            HandleObject::Process(wait_process_handle) => {
                let index = self.process_index(wait_process_handle)?;
                Some(self.processes[index].lifecycle == ProcessLifecycle::Exited)
            }
            HandleObject::Event(event_id) => self.events.get(&event_id).map(|event| event.signaled),
            HandleObject::ServiceSession(_) => Some(false),
            HandleObject::SharedMemory { .. } | HandleObject::TransferMemory { .. } => None,
        }
    }

    fn process_pid(&self, process_handle: u32) -> Option<u64> {
        self.processes
            .iter()
            .find(|process| {
                process.handle == process_handle && process.lifecycle != ProcessLifecycle::Exited
            })
            .map(|process| process.pid)
    }

    fn resolve_handle_object(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<HandleObject> {
        if handle == SELF_THREAD_HANDLE {
            return Some(HandleObject::Thread(requester_thread_handle));
        }

        let process_handle = self.process_handle_for_thread(requester_thread_handle)?;

        if handle == SELF_PROCESS_HANDLE {
            return Some(HandleObject::Process(process_handle));
        }

        let process_index = self.process_index(process_handle)?;
        self.processes[process_index]
            .handle_table
            .get(&handle)
            .copied()
    }

    pub fn resolve_thread_handle(&self, requester_thread_handle: u32, handle: u32) -> Option<u32> {
        match self.resolve_handle_object(requester_thread_handle, handle)? {
            HandleObject::Thread(thread_handle) => Some(thread_handle),
            HandleObject::Process(_)
            | HandleObject::ServiceSession(_)
            | HandleObject::Event(_)
            | HandleObject::SharedMemory { .. }
            | HandleObject::TransferMemory { .. } => None,
        }
    }

    fn process_index(&self, process_handle: u32) -> Option<usize> {
        self.processes
            .iter()
            .position(|process| process.handle == process_handle)
    }

    pub fn allocate_display_layer_id(&mut self) -> u64 {
        let id = self.next_display_layer_id;
        self.next_display_layer_id += 1;
        id
    }

    fn allocate_handle(&mut self, owner_process_handle: u32, object: HandleObject) -> Option<u32> {
        let process_index = self.process_index(owner_process_handle)?;
        let process = &mut self.processes[process_index];

        let mut handle = process.next_user_handle.max(FIRST_USER_HANDLE);
        while process.handle_table.contains_key(&handle) || handle == 0 {
            handle = handle.checked_add(1)?;
        }

        process.handle_table.insert(handle, object);
        process.next_user_handle = handle.checked_add(1).unwrap_or(FIRST_USER_HANDLE);
        Some(handle)
    }

    fn allocate_service_session_handle(
        &mut self,
        owner_process_handle: u32,
        kind: ServiceSessionKind,
    ) -> Option<u32> {
        let applet_resource_user_id = self
            .process_index(owner_process_handle)
            .and_then(|index| self.processes.get(index).map(|process| process.pid))
            .unwrap_or(1);

        let mut session_id = self.next_service_session_id.max(FIRST_SERVICE_SESSION_ID);
        while self.service_sessions.contains_key(&session_id) {
            session_id = session_id.checked_add(1)?;
        }

        self.next_service_session_id = session_id
            .checked_add(1)
            .unwrap_or(FIRST_SERVICE_SESSION_ID);
        self.service_sessions.insert(
            session_id,
            ServiceSessionState::new(kind, applet_resource_user_id),
        );

        if let Some(handle) = self.allocate_handle(
            owner_process_handle,
            HandleObject::ServiceSession(session_id),
        ) {
            Some(handle)
        } else {
            self.service_sessions.remove(&session_id);
            None
        }
    }

    fn resolve_service_session_id(&self, requester_thread_handle: u32, handle: u32) -> Option<u32> {
        match self.resolve_handle_object(requester_thread_handle, handle)? {
            HandleObject::ServiceSession(session_id) => Some(session_id),
            HandleObject::Process(_)
            | HandleObject::Thread(_)
            | HandleObject::Event(_)
            | HandleObject::SharedMemory { .. }
            | HandleObject::TransferMemory { .. } => None,
        }
    }

    fn resolve_service_session(
        &self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<&ServiceSessionState> {
        let session_id = self.resolve_service_session_id(requester_thread_handle, handle)?;
        self.service_sessions.get(&session_id)
    }

    fn resolve_service_session_mut(
        &mut self,
        requester_thread_handle: u32,
        handle: u32,
    ) -> Option<&mut ServiceSessionState> {
        let session_id = self.resolve_service_session_id(requester_thread_handle, handle)?;
        self.service_sessions.get_mut(&session_id)
    }

    fn thread_index(&self, handle: u32) -> Option<usize> {
        self.threads
            .iter()
            .position(|thread| thread.handle == handle)
    }
}

#[cfg(test)]
mod tests {
    use super::{DomainObjectKind, KernelState, MAIN_THREAD_HANDLE};
    use crate::memory::SharedGuestMemory;
    use ryurs_dynarec::{CpuState, StepResult};

    const MAIN_PC: u64 = 0x8000_0000;
    const THREAD_PC: u64 = 0x8000_1000;
    const THREAD_SP: u64 = 0x8400_0000;
    const THREAD_ARG: u64 = 0x1234;

    fn setup_kernel_with_runnable_worker() -> (KernelState, SharedGuestMemory, u32) {
        let mut kernel = KernelState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let mut state = CpuState::default();
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let thread_handle = kernel.create_thread(
            MAIN_THREAD_HANDLE,
            THREAD_PC,
            THREAD_ARG,
            THREAD_SP,
            &state,
            44,
        );
        assert!(kernel.start_thread(MAIN_THREAD_HANDLE, thread_handle));

        (kernel, memory, thread_handle)
    }

    #[test]
    fn safepoint_blocks_and_resumes_runnable_workers() {
        let (mut kernel, _memory, _thread_user_handle) = setup_kernel_with_runnable_worker();

        kernel.request_safepoint_interrupt();
        assert!(kernel.safepoint_is_active());
        assert!(kernel.worker_threads_quiesced_for_safepoint());
        assert!(kernel.take_runnable_worker_thread().is_none());

        kernel.release_safepoint_interrupt();
        assert!(!kernel.safepoint_is_active());
        assert!(
            kernel.take_runnable_worker_thread().is_some(),
            "worker should become runnable again after safepoint release"
        );
    }

    #[test]
    fn running_worker_parks_at_safepoint_after_step_completion() {
        let (mut kernel, memory, _thread_user_handle) = setup_kernel_with_runnable_worker();

        let task = kernel
            .take_runnable_worker_thread()
            .expect("worker task should be available before safepoint");

        kernel.request_safepoint_interrupt();
        assert!(kernel.safepoint_is_active());

        let mut advanced_state = task.state.clone();
        let start_pc = advanced_state.pc;
        advanced_state.pc = advanced_state.pc.wrapping_add(4);

        kernel.complete_worker_step(
            task.thread_handle,
            task.process_handle,
            advanced_state,
            StepResult::Advanced {
                start_pc,
                end_pc: start_pc.wrapping_add(4),
            },
            &memory,
        );

        assert!(kernel.is_thread_safepoint_parked(task.thread_handle));
        assert!(kernel.worker_threads_quiesced_for_safepoint());
        assert!(kernel.take_runnable_worker_thread().is_none());

        kernel.release_safepoint_interrupt();
        let resumed = kernel
            .take_runnable_worker_thread()
            .expect("worker should resume once safepoint is released");
        assert_eq!(resumed.thread_handle, task.thread_handle);
    }

    #[test]
    fn clone_service_session_preserves_domain_state() {
        let mut kernel = KernelState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let mut state = CpuState::default();
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let session_handle = kernel
            .open_named_service_session(MAIN_THREAD_HANDLE, "fsp-srv")
            .expect("fsp-srv session should open");
        assert_eq!(
            kernel.convert_service_session_to_domain(MAIN_THREAD_HANDLE, session_handle),
            Some(1)
        );

        let object_id = kernel
            .service_session_add_domain_object(
                MAIN_THREAD_HANDLE,
                session_handle,
                DomainObjectKind::Generic,
            )
            .expect("domain object should be allocated");

        let cloned_handle = kernel
            .clone_service_session(MAIN_THREAD_HANDLE, session_handle)
            .expect("cloned session handle should be allocated");

        assert_eq!(
            kernel.service_session_domain_object_kind(MAIN_THREAD_HANDLE, cloned_handle, 1),
            Some(DomainObjectKind::GenericRoot)
        );
        assert_eq!(
            kernel
                .service_session_domain_object_kind(MAIN_THREAD_HANDLE, cloned_handle, object_id,),
            Some(DomainObjectKind::Generic)
        );
    }
}
