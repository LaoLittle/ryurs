use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tracing::{info, warn};

use ryurs_dynarec::{
    CpuState, DecodeError, DynarecEngine, DynarecOptions, ExceptionLevel, ExitReason, StepResult,
    arm_registered_execution_context_pages, decode_block,
    disarm_registered_execution_context_pages, install_fault_handler,
};
use ryurs_gpu_wgpu::WgpuBackend;

use crate::config::EmulatorConfig;
use crate::hle::{KernelState, MAIN_THREAD_HANDLE, ServiceManager};
use crate::keys::KeySet;
use crate::loader::{
    ProgramImage, ProgramRuntimeHints, parse_nca_program_image, parse_nsp_program_image,
    parse_program_image, words_to_le_bytes,
};
use crate::memory::SharedGuestMemory;
use crate::memory::set_memory_watch_pc;

#[derive(Debug, Clone, Copy)]
pub struct RunReport {
    pub executed_blocks: usize,
    pub stop_reason: ExitReason,
    pub end_pc: u64,
    pub last_block_start_pc: Option<u64>,
    pub last_block_start_instruction: Option<u32>,
    pub fault_instruction: Option<u32>,
    pub fault_prev_instruction: Option<u32>,
    pub fault_prev2_instruction: Option<u32>,
    pub fault_prev3_instruction: Option<u32>,
    pub fault_address: Option<u64>,
    pub decode_error_pc: Option<u64>,
    pub decode_error_opcode: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmulatorSnapshot {
    pub pc: u64,
    pub sp: u64,
    pub nzcv: u32,
    pub current_el: ExceptionLevel,
    pub x0: u64,
    pub x1: u64,
    pub x2: u64,
    pub x3: u64,
    pub kernel_ticks: u64,
}

#[derive(Debug, Clone)]
pub struct DisplayFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct Emulator {
    pub state: CpuState,
    memory: SharedGuestMemory,
    dynarec: Arc<DynarecEngine>,
    block_max_instructions: usize,
    reset_vector: u64,
    high_memory_end: u64,
    kernel: Arc<Mutex<KernelState>>,
    worker_pool: GuestWorkerPool,
    _services: ServiceManager,
    gpu: Option<WgpuBackend>,
}

const PROCESS_FLAG_IS_64_BIT: u8 = 1 << 0;
const PROCESS_FLAG_ADDRESS_SPACE_SHIFT: u8 = 1;
const PROCESS_FLAG_ADDRESS_SPACE_MASK: u8 = 0x7 << PROCESS_FLAG_ADDRESS_SPACE_SHIFT;
const PROCESS_FLAG_ADDRESS_SPACE_32_BIT: u8 = 0 << PROCESS_FLAG_ADDRESS_SPACE_SHIFT;

fn trace_block_registers(block_start_pc: u64, state: &CpuState) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU32, Ordering as AtomOrd};

    static TRACE_PCS: OnceLock<Vec<u64>> = OnceLock::new();
    static TRACE_LIMIT: OnceLock<u32> = OnceLock::new();
    static TRACE_COUNT: AtomicU32 = AtomicU32::new(0);

    let pcs = TRACE_PCS.get_or_init(|| {
        std::env::var("RYURS_TRACE_BLOCK_REGS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|value| {
                        let value = value.trim();
                        if value.is_empty() {
                            return None;
                        }

                        let value = value
                            .strip_prefix("0x")
                            .or_else(|| value.strip_prefix("0X"))
                            .unwrap_or(value);
                        u64::from_str_radix(value, 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    if !pcs.contains(&block_start_pc) {
        return;
    }

    let limit = *TRACE_LIMIT.get_or_init(|| {
        std::env::var("RYURS_TRACE_BLOCK_LIMIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(32)
    });
    let index = TRACE_COUNT.fetch_add(1, AtomOrd::Relaxed);
    if index >= limit {
        return;
    }

    eprintln!(
        "[block-regs] pc=0x{block_start_pc:016x} sp=0x{:016x} x0=0x{:016x} x1=0x{:016x} x2=0x{:016x} x3=0x{:016x} x6=0x{:016x} x7=0x{:016x} x8=0x{:016x} x9=0x{:016x}",
        state.sp,
        state.read_x(0),
        state.read_x(1),
        state.read_x(2),
        state.read_x(3),
        state.read_x(6),
        state.read_x(7),
        state.read_x(8),
        state.read_x(9),
    );
    eprintln!(
        "[block-regs] pc=0x{block_start_pc:016x} x10=0x{:016x} x11=0x{:016x} x14=0x{:016x} x15=0x{:016x} x16=0x{:016x} x17=0x{:016x} x18=0x{:016x} x19=0x{:016x}",
        state.read_x(10),
        state.read_x(11),
        state.read_x(14),
        state.read_x(15),
        state.read_x(16),
        state.read_x(17),
        state.read_x(18),
        state.read_x(19),
    );
    eprintln!(
        "[block-regs] pc=0x{block_start_pc:016x} x20=0x{:016x} x21=0x{:016x} x22=0x{:016x} x23=0x{:016x} x24=0x{:016x} x29=0x{:016x} x30=0x{:016x}",
        state.read_x(20),
        state.read_x(21),
        state.read_x(22),
        state.read_x(23),
        state.read_x(24),
        state.read_x(29),
        state.read_x(30),
    );
}

fn trace_block_memory_after(block_start_pc: u64, memory: &SharedGuestMemory) {
    use std::sync::OnceLock;

    #[derive(Debug, Clone, Copy)]
    struct TraceBlockMemoryRange {
        pc: u64,
        address: u64,
        words: usize,
    }

    static RANGES: OnceLock<Vec<TraceBlockMemoryRange>> = OnceLock::new();

    let ranges = RANGES.get_or_init(|| {
        std::env::var("RYURS_TRACE_BLOCK_MEMORY_AFTER")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|entry| {
                        let (pc, rest) = entry.trim().split_once('@')?;
                        let (address, words) = rest.split_once(':')?;
                        let parse_hex = |value: &str| {
                            let value = value
                                .trim()
                                .strip_prefix("0x")
                                .or_else(|| value.trim().strip_prefix("0X"))
                                .unwrap_or(value.trim());
                            u64::from_str_radix(value, 16).ok()
                        };

                        Some(TraceBlockMemoryRange {
                            pc: parse_hex(pc)?,
                            address: parse_hex(address)?,
                            words: words.trim().parse::<usize>().ok()?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    });

    for range in ranges
        .iter()
        .copied()
        .filter(|range| range.pc == block_start_pc)
    {
        if range.words == 0 {
            continue;
        }

        let mut values = Vec::with_capacity(range.words);
        let mut ok = true;
        for index in 0..range.words {
            let addr = range.address + (index as u64) * 4;
            match memory.fetch_u32(addr) {
                Some(value) => values.push(value),
                None => {
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            let joined = values
                .iter()
                .map(|value| format!("0x{value:08x}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "[block-mem-after] pc=0x{block_start_pc:016x} addr=0x{:016x} words={} {}",
                range.address, range.words, joined
            );
        } else {
            eprintln!(
                "[block-mem-after] pc=0x{block_start_pc:016x} addr=0x{:016x} words={} <unreadable>",
                range.address, range.words
            );
        }
    }
}

fn trace_block_memory_changes(block_start_pc: u64, memory: &SharedGuestMemory) {
    use std::sync::{Mutex, OnceLock};

    #[derive(Debug, Clone, Copy)]
    struct TraceMemoryWatch {
        address: u64,
        words: usize,
    }

    static WATCH: OnceLock<Option<TraceMemoryWatch>> = OnceLock::new();
    static LAST_VALUES: OnceLock<Mutex<Option<Vec<u32>>>> = OnceLock::new();

    let watch = WATCH.get_or_init(|| {
        let raw = std::env::var("RYURS_TRACE_BLOCK_MEMORY_CHANGES").ok()?;
        let (address, words) = raw.trim().split_once(':')?;
        let address = address
            .trim()
            .strip_prefix("0x")
            .or_else(|| address.trim().strip_prefix("0X"))
            .unwrap_or(address.trim());
        Some(TraceMemoryWatch {
            address: u64::from_str_radix(address, 16).ok()?,
            words: words.trim().parse::<usize>().ok()?,
        })
    });
    let Some(watch) = watch else {
        return;
    };
    if watch.words == 0 {
        return;
    }

    let mut values = Vec::with_capacity(watch.words);
    for index in 0..watch.words {
        let addr = watch.address + (index as u64) * 4;
        let Some(value) = memory.fetch_u32(addr) else {
            return;
        };
        values.push(value);
    }

    let last_values = LAST_VALUES.get_or_init(|| Mutex::new(None));
    let mut guard = last_values
        .lock()
        .expect("trace block memory changes lock poisoned");
    if guard.as_ref() == Some(&values) {
        return;
    }

    let joined = values
        .iter()
        .map(|value| format!("0x{value:08x}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "[block-mem-change] pc=0x{block_start_pc:016x} addr=0x{:016x} words={} {}",
        watch.address, watch.words, joined
    );
    *guard = Some(values);
}
const PROCESS_FLAG_ADDRESS_SPACE_64_BIT_DEPRECATED: u8 = 1 << PROCESS_FLAG_ADDRESS_SPACE_SHIFT;
const PROCESS_FLAG_ADDRESS_SPACE_32_BIT_NO_ALIAS: u8 = 2 << PROCESS_FLAG_ADDRESS_SPACE_SHIFT;
const PROCESS_FLAG_ADDRESS_SPACE_64_BIT: u8 = 3 << PROCESS_FLAG_ADDRESS_SPACE_SHIFT;
const DEFAULT_MAIN_THREAD_STACK_SIZE: u64 = 0x10_0000;
const THREAD_EXIT_TRAMPOLINE_ADDR: u64 = 0x1000_0000;
const SVC_EXIT_THREAD: u32 = 0xd400_0141;
const BRK_PANIC: u32 = 0xd420_0000;
const STARTUP_FRAME_OFFSET: u64 = 0x80;
const MOD0_MAGIC: u32 = u32::from_le_bytes(*b"MOD0");
const MEMORY_STATE_CODE_STATIC: u32 = 0x3;
const MEMORY_STATE_HEAP: u32 = 0x5;
const MEMORY_STATE_STACK: u32 = 0x0b;
const MEMORY_STATE_THREAD_LOCAL: u32 = 0x0c;
const MEMORY_PERMISSION_READ: u32 = 0x1;
const MEMORY_PERMISSION_READ_WRITE: u32 = 0x3;
const MEMORY_PERMISSION_READ_EXECUTE: u32 = 0x5;
const PAGE_SIZE: u64 = 0x1000;
const PAGE_MASK: u64 = !(PAGE_SIZE - 1);

fn align_up_page(value: u64) -> u64 {
    (value.saturating_add(PAGE_SIZE - 1)) & PAGE_MASK
}

fn default_process_flags() -> u8 {
    PROCESS_FLAG_IS_64_BIT | PROCESS_FLAG_ADDRESS_SPACE_64_BIT
}

fn low_region_for_process_flags(process_flags: u8, reset_vector: u64) -> Option<(u64, u64)> {
    let (start, end) = match process_flags & PROCESS_FLAG_ADDRESS_SPACE_MASK {
        PROCESS_FLAG_ADDRESS_SPACE_32_BIT | PROCESS_FLAG_ADDRESS_SPACE_32_BIT_NO_ALIAS => {
            (0x0020_0000, 0x4000_0000)
        }
        PROCESS_FLAG_ADDRESS_SPACE_64_BIT_DEPRECATED => (0x0800_0000, 0x8000_0000),
        PROCESS_FLAG_ADDRESS_SPACE_64_BIT => (0x1000, reset_vector),
        _ => (0x1000, reset_vector),
    };

    let end = end.min(reset_vector);
    if end <= start {
        None
    } else {
        Some((start, end))
    }
}

struct WorkerNotifier {
    threads: Mutex<Vec<thread::Thread>>,
}

impl WorkerNotifier {
    fn new() -> Self {
        Self {
            threads: Mutex::new(Vec::new()),
        }
    }

    fn register_current_thread(&self) {
        self.threads
            .lock()
            .expect("worker notifier lock poisoned")
            .push(thread::current());
    }

    fn unpark_all(&self) {
        let threads = self.threads.lock().expect("worker notifier lock poisoned");
        for thread in threads.iter() {
            thread.unpark();
        }
    }
}

struct WorkerContext {
    guest_thread_handle: u32,
    dynarec: Arc<DynarecEngine>,
    kernel: Arc<Mutex<KernelState>>,
    shutdown: Arc<AtomicBool>,
    notifier: Arc<WorkerNotifier>,
}

struct GuestWorkerPool {
    shutdown: Arc<AtomicBool>,
    notifier: Arc<WorkerNotifier>,
    dynarec: Arc<DynarecEngine>,
    kernel: Arc<Mutex<KernelState>>,
    workers: Vec<(u32, JoinHandle<()>)>,
}

impl GuestWorkerPool {
    fn new(dynarec: Arc<DynarecEngine>, kernel: Arc<Mutex<KernelState>>) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let notifier = Arc::new(WorkerNotifier::new());

        Self {
            shutdown,
            notifier,
            dynarec,
            kernel,
            workers: Vec::new(),
        }
    }

    fn sync_guest_threads(&mut self) {
        let mut index = 0usize;
        while index < self.workers.len() {
            if self.workers[index].1.is_finished() {
                let (_, handle) = self.workers.remove(index);
                let _ = handle.join();
            } else {
                index += 1;
            }
        }

        let guest_threads = self
            .kernel
            .lock()
            .expect("kernel lock poisoned")
            .worker_thread_handles_snapshot();

        for guest_thread_handle in guest_threads {
            if self
                .workers
                .iter()
                .any(|(running_thread, _)| *running_thread == guest_thread_handle)
            {
                continue;
            }

            let context = WorkerContext {
                guest_thread_handle,
                dynarec: Arc::clone(&self.dynarec),
                kernel: Arc::clone(&self.kernel),
                shutdown: Arc::clone(&self.shutdown),
                notifier: Arc::clone(&self.notifier),
            };

            let handle = thread::spawn(move || worker_loop(context));
            self.workers.push((guest_thread_handle, handle));
        }
    }

    fn wake_workers(&self) {
        self.notifier.unpark_all();
    }
}

impl Drop for GuestWorkerPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notifier.unpark_all();
        while let Some((_, worker)) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(context: WorkerContext) {
    context.notifier.register_current_thread();

    loop {
        if context.shutdown.load(Ordering::Acquire) {
            break;
        }

        let (task, thread_live, wait_timeout) = {
            let mut kernel = context.kernel.lock().expect("kernel lock poisoned");
            kernel.wake_timed_threads();
            let thread_live = kernel.thread_is_live(context.guest_thread_handle);
            let task = if thread_live {
                kernel.take_runnable_worker_thread_for(context.guest_thread_handle)
            } else {
                None
            };
            (task, thread_live, kernel.time_until_next_wakeup())
        };

        if !thread_live {
            break;
        }

        let Some(task) = task else {
            if let Some(timeout) = wait_timeout {
                thread::park_timeout(timeout);
            } else {
                thread::park();
            }
            continue;
        };

        let mut state = task.state;
        let result = {
            let mut memory = task.memory.locked_view();
            match context.dynarec.step_with_memory(&mut state, &mut memory) {
                Ok(result) => result,
                Err(_) => StepResult::Trap {
                    pc: state.pc,
                    reason: ExitReason::DecodeFault,
                },
            }
        };

        let has_more_work = {
            let mut kernel = context.kernel.lock().expect("kernel lock poisoned");
            kernel.complete_worker_step(
                task.thread_handle,
                task.process_handle,
                state,
                result,
                &task.memory,
            );
            kernel.has_worker_work()
        };

        if has_more_work {
            context.notifier.unpark_all();
        }
    }
}

impl Emulator {
    fn maybe_apply_guest_workaround(&mut self) -> Option<StepResult> {
        use std::sync::atomic::{AtomicBool, Ordering as AtomOrd};

        // 2048.nro tries to load `sdmc:/switch/2048/state` during startup.
        // Our current FSP sdmc file stubs return a fake object with zeroed reads,
        // which immediately wipes the freshly spawned board back to all-zero.
        // Skip that load routine until sdmc file semantics are implemented.
        static LOGGED_2048_SAVE_SKIP: AtomicBool = AtomicBool::new(false);

        if self.state.pc == 0x8000_1f60
            && self.memory.fetch_u32(0x8000_1f60) == Some(0x4f00_041f)
            && self.memory.fetch_u32(0x8000_1f64) == Some(0xa9b8_7bfd)
            && self.memory.fetch_u32(0x8000_1f90) == Some(0x9400_7b5c)
        {
            if !LOGGED_2048_SAVE_SKIP.swap(true, AtomOrd::Relaxed) {
                eprintln!(
                    "[guest-workaround] skipping 2048 save-state load because sdmc file IO is not implemented yet"
                );
            }

            let start_pc = self.state.pc;
            let end_pc = self.state.read_x(30);
            self.state.pc = end_pc;
            return Some(StepResult::Advanced { start_pc, end_pc });
        }

        None
    }

    pub fn new(config: EmulatorConfig) -> Result<Self> {
        install_fault_handler()?;

        let mut state = CpuState::default();
        state.pc = config.reset_vector;

        let high_memory_end = config
            .reset_vector
            .saturating_add(config.memory_size as u64);

        let dynarec = Arc::new(DynarecEngine::new(DynarecOptions {
            block_max_instructions: config.block_max_instructions,
            force_interpreter: config.force_interpreter,
        })?);

        let memory = SharedGuestMemory::new(config.reset_vector, config.memory_size);
        info!(
            host_page_size = memory.host_page_size(),
            alias_mode = ?memory.alias_mode(),
            "initialized guest memory backend"
        );
        memory.clear_low_regions();
        if let Some((start, end)) =
            low_region_for_process_flags(default_process_flags(), config.reset_vector)
        {
            let _ = memory.map_low_region(start, end.saturating_sub(start));
        }

        let stack_size = align_up_page(DEFAULT_MAIN_THREAD_STACK_SIZE);
        let stack_end = high_memory_end & PAGE_MASK;
        state.sp = stack_end.saturating_sub(0x10) & !0xf;

        let thread_ptr = stack_end
            .saturating_sub(stack_size)
            .saturating_sub(PAGE_SIZE);
        state.tpidr_el0 = 0;
        state.tpidrro_el0 = thread_ptr;
        state.tpidr_el1 = 0;
        let _ = memory.write_u64(thread_ptr.wrapping_add(0x1f8), thread_ptr);

        state.write_x(0, 0);
        state.write_x(1, u64::from(MAIN_THREAD_HANDLE));
        state.write_x(18, 1);
        state.write_x(30, THREAD_EXIT_TRAMPOLINE_ADDR);

        Self::install_thread_exit_trampoline(&memory);
        Self::install_startup_return_frame(&memory, state.sp);

        let services = ServiceManager::with_defaults();
        info!(
            count = services.names().len(),
            "registered HLE service stubs"
        );

        let gpu = if config.gpu_enabled {
            match WgpuBackend::new() {
                Ok(backend) => {
                    info!(adapter = backend.adapter_name(), "initialized wgpu backend");
                    Some(backend)
                }
                Err(err) => {
                    warn!(error = %err, "wgpu initialization failed, running CPU-only");
                    None
                }
            }
        } else {
            None
        };

        let kernel = Arc::new(Mutex::new(KernelState::default()));
        {
            let mut guard = kernel.lock().expect("kernel lock poisoned");
            guard.set_gpu_adapter_name(
                gpu.as_ref()
                    .map(|backend| backend.adapter_name().to_string()),
            );
            guard.reset_threads(&state, &memory);
        }

        let mut worker_pool = GuestWorkerPool::new(Arc::clone(&dynarec), Arc::clone(&kernel));
        worker_pool.sync_guest_threads();

        Ok(Self {
            state,
            memory,
            dynarec,
            block_max_instructions: config.block_max_instructions,
            reset_vector: config.reset_vector,
            high_memory_end,
            kernel,
            worker_pool,
            _services: services,
            gpu,
        })
    }

    /// Set a channel for sending framebuffer pixels to the display renderer.
    /// Must be called before `run_steps()`.
    pub fn set_frame_sender(&self, tx: std::sync::mpsc::SyncSender<DisplayFrame>) {
        let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
        kernel.set_frame_sender(tx);
    }

    pub fn load_program_bytes(&mut self, load_address: u64, bytes: &[u8]) -> Result<()> {
        let image = ProgramImage::raw(load_address, bytes);
        self.load_program_image(&image)
    }

    pub fn load_program_words(&mut self, load_address: u64, words: &[u32]) -> Result<()> {
        let bytes = words_to_le_bytes(words);
        self.load_program_bytes(load_address, &bytes)
    }

    pub fn load_program_auto(
        &mut self,
        fallback_load_address: u64,
        bytes: &[u8],
    ) -> Result<ProgramImage> {
        let image = parse_program_image(bytes, fallback_load_address)?;
        self.load_program_image(&image)?;
        Ok(image)
    }

    pub fn load_program_nca(
        &mut self,
        fallback_load_address: u64,
        bytes: &[u8],
        key_set: &KeySet,
    ) -> Result<ProgramImage> {
        let image = parse_nca_program_image(bytes, fallback_load_address, key_set)?;
        self.load_program_image(&image)?;
        Ok(image)
    }

    pub fn load_program_nsp(
        &mut self,
        fallback_load_address: u64,
        bytes: &[u8],
        key_set: &KeySet,
    ) -> Result<ProgramImage> {
        let image = parse_nsp_program_image(bytes, fallback_load_address, key_set)?;
        self.load_program_image(&image)?;
        Ok(image)
    }

    pub fn load_program_image(&mut self, image: &ProgramImage) -> Result<()> {
        self.memory.clear_high_regions();
        for segment in &image.segments {
            self.memory.load(segment.address, &segment.bytes)?;
        }

        // Apply ELF relocations from the MOD0/.dynamic section for each loaded module.
        for &(text_base, bias) in &image.module_bases {
            self.apply_elf_relocations(text_base, bias);
        }

        self.rebuild_query_memory_regions(image);

        self.apply_runtime_hints(image.runtime_hints);
        self.state.pc = image.entry;
        {
            let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
            kernel.reset_threads(&self.state, &self.memory);

            // Store RomFS data for FspStorage::Read if the image provides it.
            if let Some(ref romfs) = image.romfs_data {
                eprintln!("[emulator] loaded RomFS data: {} bytes", romfs.len());
                kernel.romfs_data = Some(std::sync::Arc::new(romfs.clone()));
            }
        }

        // Compute heap region layout (Ryujinx-style).
        self.configure_heap_region(image);

        self.dynarec.invalidate_all();
        self.worker_pool =
            GuestWorkerPool::new(Arc::clone(&self.dynarec), Arc::clone(&self.kernel));
        self.worker_pool.sync_guest_threads();
        self.worker_pool.wake_workers();
        Ok(())
    }

    /// Execute one JIT block without processing SVCs through the kernel.
    /// Returns the raw StepResult from the dynarec. If the result is a
    /// SupervisorCall, the caller must handle it (e.g., call `resolve_svc`).
    pub fn step_raw(&mut self) -> Result<StepResult> {
        self.memory.clear_last_fault();
        let result = {
            let mut memory = self.memory.locked_view();
            self.dynarec
                .step_with_memory(&mut self.state, &mut memory)?
        };
        Ok(result)
    }

    /// Resolve a SupervisorCall after `step_raw` returned one.
    pub fn resolve_svc(&mut self, result: StepResult) -> StepResult {
        self.kernel
            .lock()
            .expect("kernel lock poisoned")
            .resolve_thread_step(MAIN_THREAD_HANDLE, result, &mut self.state, &self.memory)
    }

    pub fn step(&mut self) -> Result<StepResult> {
        // Yield-polling: when the main thread is blocked on a mutex, poll
        // for up to YIELD_BUDGET_MS of wall-clock time per step() call.
        // This doesn't consume block budget, giving workers real time
        // to release the contended mutex.
        let mut should_wake_workers;
        let yield_deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        loop {
            self.worker_pool.sync_guest_threads();

            should_wake_workers = {
                let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
                kernel.wake_timed_threads();
                kernel.has_worker_work()
            };

            if self
                .kernel
                .lock()
                .expect("kernel lock poisoned")
                .safepoint_is_active()
            {
                if should_wake_workers {
                    self.worker_pool.wake_workers();
                }
                if let Some(gpu) = &self.gpu {
                    gpu.submit_idle();
                }
                return Ok(StepResult::Advanced {
                    start_pc: self.state.pc,
                    end_pc: self.state.pc,
                });
            }

            // When the main thread is "blocked" on a contended mutex, don't
            // execute its JIT block.  Instead, give workers time to run so
            // the mutex owner can release the lock.
            {
                let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
                if kernel.main_thread_is_yielding() {
                    // Check if the mutex has been released.
                    if kernel.try_clear_main_thread_yield(&self.memory) {
                        // Mutex is free — resume main thread execution.
                    } else if std::time::Instant::now() < yield_deadline {
                        // Still held and within budget — wake workers and give
                        // them a generous time slice (5ms) to make progress
                        // without the main thread contending for the kernel lock.
                        kernel.on_cpu_block_yield(&self.memory);
                        let _yield_count = kernel.yield_poll_count();
                        drop(kernel);
                        self.worker_pool.wake_workers();
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        self.worker_pool.sync_guest_threads();
                        continue;
                    } else {
                        // Budget exhausted — count as 1 block so caller advances.
                        kernel.on_cpu_block_yield(&self.memory);
                        drop(kernel);
                        self.worker_pool.wake_workers();
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        self.worker_pool.sync_guest_threads();
                        return Ok(StepResult::Advanced {
                            start_pc: self.state.pc,
                            end_pc: self.state.pc,
                        });
                    }
                }
            }

            break; // Not yielding — proceed to execute a guest block.
        } // end yield-poll loop

        // sdmc load workaround for now (not implemented yet).
        if let Some(result) = self.maybe_apply_guest_workaround() {
            return Ok(result);
        }

        self.memory.clear_last_fault();

        let block_start_pc = self.state.pc;
        set_memory_watch_pc(block_start_pc);
        trace_block_registers(block_start_pc, &self.state);
        let result = {
            let mut memory = self.memory.locked_view();
            self.dynarec
                .step_with_memory(&mut self.state, &mut memory)?
        };
        trace_block_memory_after(block_start_pc, &self.memory);
        trace_block_memory_changes(block_start_pc, &self.memory);

        // Trace main thread block PCs in the steady-state loop.
        // Only active when RYURS_TRACE_PC=1 environment variable is set.
        {
            use std::sync::OnceLock;
            use std::sync::atomic::{AtomicU32, Ordering as AtomOrd};
            static TRACE_PC: OnceLock<bool> = OnceLock::new();
            static LOOKUP_DUMPS: AtomicU32 = AtomicU32::new(0);
            let trace_pc = *TRACE_PC.get_or_init(|| {
                std::env::var("RYURS_TRACE_PC")
                    .map(|v| v == "1")
                    .unwrap_or(false)
            });

            // Dump the search string when we enter the linked list lookup function.
            if block_start_pc == 0x800007a0 && LOOKUP_DUMPS.fetch_add(1, AtomOrd::Relaxed) < 5 {
                let string_addr = self.state.read_x(0);
                let mut s = Vec::new();
                for i in 0..128u64 {
                    if let Some(b) = self
                        .memory
                        .fetch_u32(string_addr + i)
                        .map(|w| (w & 0xFF) as u8)
                    {
                        if b == 0 {
                            break;
                        }
                        s.push(b);
                    } else {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&s);
                eprintln!("[lookup] string at x0=0x{:016x}: {:?}", string_addr, text);
            }
            if trace_pc {
                let tick = self.kernel.lock().expect("kernel lock").tick_count;
                if tick > 10000 && tick % 512 == 0 {
                    eprintln!(
                        "[pc-trace] tick={} pc=0x{:016x} -> 0x{:016x} x0=0x{:016x} x1=0x{:016x}",
                        tick,
                        block_start_pc,
                        self.state.pc,
                        self.state.read_x(0),
                        self.state.read_x(1),
                    );
                }
            }
        }

        let result = self
            .kernel
            .lock()
            .expect("kernel lock poisoned")
            .resolve_thread_step(MAIN_THREAD_HANDLE, result, &mut self.state, &self.memory);

        // Check whether the main thread requested a deferred sleep
        // (e.g. svc_sleep_thread).  We take the duration *before* dropping
        // the lock, then sleep *after* dropping it so workers can run
        // with an uncontested lock during the sleep.
        let pending_sleep;
        {
            let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
            kernel.on_cpu_block(&self.memory);
            should_wake_workers |= kernel.has_worker_work();

            // When the main thread is in a tight ArbitrateLock spin loop,
            // yield to let worker threads make progress.
            if kernel.main_thread_should_yield() {
                should_wake_workers = true;
            }
            pending_sleep = kernel.take_main_thread_pending_sleep();
        }

        self.worker_pool.sync_guest_threads();

        if should_wake_workers {
            self.worker_pool.wake_workers();
            // Give worker threads a small time slice to execute.
            std::thread::yield_now();
        }

        // Perform the deferred sleep with the kernel lock released.
        if let Some(dur) = pending_sleep {
            std::thread::sleep(dur);
        }

        if matches!(result, StepResult::Advanced { .. }) {
            if let Some(trap) = self
                .kernel
                .lock()
                .expect("kernel lock poisoned")
                .take_worker_trap()
            {
                return Ok(StepResult::Trap {
                    pc: trap.pc,
                    reason: trap.reason,
                });
            }
        }

        if let Some(gpu) = &self.gpu {
            gpu.submit_idle();
        }

        Ok(result)
    }

    pub fn run_steps(&mut self, max_blocks: usize) -> Result<RunReport> {
        let mut report = RunReport {
            executed_blocks: 0,
            stop_reason: ExitReason::Running,
            end_pc: self.state.pc,
            last_block_start_pc: None,
            last_block_start_instruction: None,
            fault_instruction: None,
            fault_prev_instruction: None,
            fault_prev2_instruction: None,
            fault_prev3_instruction: None,
            fault_address: None,
            decode_error_pc: None,
            decode_error_opcode: None,
        };

        for _ in 0..max_blocks {
            match self.step()? {
                StepResult::Advanced { start_pc, end_pc } => {
                    report.executed_blocks += 1;
                    report.last_block_start_pc = Some(start_pc);
                    report.last_block_start_instruction = self.memory.fetch_u32(start_pc);
                    report.end_pc = end_pc;
                }
                StepResult::Trap { pc, reason } => {
                    report.end_pc = pc;
                    report.stop_reason = reason;
                    report.fault_instruction = self.memory.fetch_u32(pc);
                    report.fault_prev_instruction = pc
                        .checked_sub(4)
                        .and_then(|prev_pc| self.memory.fetch_u32(prev_pc));
                    report.fault_prev2_instruction = pc
                        .checked_sub(8)
                        .and_then(|prev_pc| self.memory.fetch_u32(prev_pc));
                    report.fault_prev3_instruction = pc
                        .checked_sub(12)
                        .and_then(|prev_pc| self.memory.fetch_u32(prev_pc));
                    report.fault_address = self.memory.last_fault_address();

                    if let ExitReason::DecodeFault = reason {
                        if let Some((decode_pc, decode_opcode)) =
                            self.inspect_decode_fault_details(pc)
                        {
                            report.decode_error_pc = Some(decode_pc);
                            report.decode_error_opcode = Some(decode_opcode);
                        }
                    }

                    return Ok(report);
                }
            }
        }

        report.end_pc = self.state.pc;
        Ok(report)
    }

    fn inspect_decode_fault_details(&self, start_pc: u64) -> Option<(u64, u32)> {
        match decode_block(start_pc, self.dynarec_block_limit(), |pc| {
            self.memory.fetch_u32(pc)
        }) {
            Ok(_) => None,
            Err(DecodeError::UnsupportedOpcode { pc, opcode }) => Some((pc, opcode)),
            Err(DecodeError::MissingInstruction { .. }) => None,
        }
    }

    fn dynarec_block_limit(&self) -> usize {
        self.block_max_instructions
    }

    /// Compute and register the heap region in the kernel, following Ryujinx's
    /// address-space layout for 64-bit processes (Addr39).
    ///
    /// Ryujinx places regions after the code region in order:
    ///   Alias (64 GiB) | Heap (6 GiB) | Stack (2 GiB) | TlsIo (64 GiB)
    /// We simplify: heap starts right after the highest used address
    /// (code segments + stack), aligned up to REGION_ALIGNMENT (2 MiB).
    /// Heap region size = 6 GiB (0x1_8000_0000) matching Ryujinx default.
    fn configure_heap_region(&mut self, image: &ProgramImage) {
        const REGION_ALIGNMENT: u64 = 0x20_0000; // 2 MiB
        const HEAP_REGION_SIZE: u64 = 0x1_8000_0000; // 6 GiB

        // Find the highest address used by code segments.
        let code_end = image
            .segments
            .iter()
            .filter(|s| !s.bytes.is_empty())
            .map(|s| s.address + s.bytes.len() as u64)
            .max()
            .unwrap_or(self.reset_vector);

        // Image size = total bytes of all segments (for GetInfo used-memory).
        let image_size: u64 = image.segments.iter().map(|s| s.bytes.len() as u64).sum();

        // Stack sits at the top of the fastmem span.
        let stack_size = image
            .runtime_hints
            .main_thread_stack_size
            .unwrap_or(DEFAULT_MAIN_THREAD_STACK_SIZE)
            .max(PAGE_SIZE);
        let stack_size = align_up_page(stack_size);

        // The highest used address is max(code_end, stack region end).
        // Stack is at high_memory_end - stack_size .. high_memory_end, plus
        // one page for TLS below the stack.
        let highest_used = self.high_memory_end.max(code_end);

        // Heap region starts after the highest used address, aligned up.
        let heap_region_start = (highest_used + REGION_ALIGNMENT - 1) & !(REGION_ALIGNMENT - 1);
        let heap_region_end = heap_region_start + HEAP_REGION_SIZE;

        self.kernel
            .lock()
            .expect("kernel lock poisoned")
            .configure_main_process_heap(
                heap_region_start,
                heap_region_end,
                image_size,
                stack_size,
            );
    }

    fn rebuild_query_memory_regions(&mut self, image: &ProgramImage) {
        let mut segments = image
            .segments
            .iter()
            .filter_map(|segment| {
                (!segment.bytes.is_empty()).then_some((segment.address, segment.bytes.len() as u64))
            })
            .collect::<Vec<_>>();
        segments.sort_by_key(|(address, _)| *address);

        self.memory.clear_high_regions();

        if segments.is_empty() {
            return;
        }

        let mut module_bases = segments
            .iter()
            .filter_map(|(address, _)| {
                self.looks_like_module_text_base(*address)
                    .then_some(*address)
            })
            .collect::<Vec<_>>();
        module_bases.sort_unstable();
        module_bases.dedup();

        let mut module_segment_ordinals = BTreeMap::<u64, usize>::new();

        for (address, size) in segments {
            let module_base = module_bases
                .iter()
                .copied()
                .take_while(|base| *base <= address)
                .last();

            let (state, permission) = if let Some(base) = module_base {
                let ordinal = module_segment_ordinals.entry(base).or_insert(0);
                let region = match *ordinal {
                    0 => (MEMORY_STATE_CODE_STATIC, MEMORY_PERMISSION_READ_EXECUTE),
                    1 => (MEMORY_STATE_CODE_STATIC, MEMORY_PERMISSION_READ),
                    _ => (MEMORY_STATE_HEAP, MEMORY_PERMISSION_READ_WRITE),
                };
                *ordinal += 1;
                region
            } else {
                (MEMORY_STATE_HEAP, MEMORY_PERMISSION_READ_WRITE)
            };

            let _ = self
                .memory
                .map_high_region(address, size, state, permission);
        }
    }

    fn looks_like_module_text_base(&self, base: u64) -> bool {
        let Some(mod0_delta) = self.memory.read_u32(base.wrapping_add(4)) else {
            return false;
        };

        if mod0_delta == 0 || mod0_delta > 0x20_0000 {
            return false;
        }

        let mod0 = base.wrapping_add(mod0_delta as u64);
        self.memory.read_u32(mod0) == Some(MOD0_MAGIC)
    }

    /// Apply ELF relocations from the MOD0/.dynamic section of the loaded
    /// program.  This processes R_AARCH64_RELATIVE entries and patches
    /// absolute addresses so that the program works at its loaded base.
    fn apply_elf_relocations(&mut self, text_base: u64, relocation_bias: u64) {
        // The NSO text segment starts with a branch instruction followed by a
        // MOD0 offset at +4.
        let Some(mod0_delta) = self.memory.read_u32(text_base.wrapping_add(4)) else {
            return;
        };
        if mod0_delta == 0 || mod0_delta > 0x20_0000 {
            return;
        }
        let mod0 = text_base.wrapping_add(mod0_delta as u64);
        if self.memory.read_u32(mod0) != Some(MOD0_MAGIC) {
            eprintln!("[reloc] MOD0 magic mismatch at {mod0:#x}");
            return;
        }

        // MOD0 layout:
        //   +0x00: magic "MOD0"
        //   +0x04: dynamic_offset (i32, relative to MOD0)
        //   +0x08: bss_start_offset (i32, relative to MOD0)
        //   +0x0C: bss_end_offset (i32, relative to MOD0)
        //   +0x10: eh_start_offset (i32, relative to MOD0)
        //   +0x14: eh_end_offset (i32, relative to MOD0)
        //   +0x18: module_object_offset (i32, relative to MOD0)
        let Some(dyn_delta) = self.memory.read_u32(mod0.wrapping_add(4)) else {
            return;
        };
        let dynamic_addr = mod0.wrapping_add(dyn_delta as u64);

        // Parse .dynamic entries to find DT_RELA, DT_RELASZ, DT_RELAENT,
        // DT_JMPREL, DT_PLTRELSZ, DT_REL, DT_RELSZ.
        const DT_NULL: u64 = 0;
        const DT_RELA: u64 = 7;
        const DT_RELASZ: u64 = 8;
        const DT_RELAENT: u64 = 9;
        const DT_JMPREL: u64 = 23;
        const DT_PLTRELSZ: u64 = 2;
        const DT_REL: u64 = 17;
        const DT_RELSZ: u64 = 18;
        const DT_RELCOUNT: u64 = 0x6ffffff9;
        const DT_RELACOUNT: u64 = 0x6ffffffa;

        let mut rela_addr: u64 = 0;
        let mut rela_size: u64 = 0;
        let mut rela_ent: u64 = 24; // default for Elf64_Rela
        let mut jmprel_addr: u64 = 0;
        let mut jmprel_size: u64 = 0;
        let mut rel_addr: u64 = 0;
        let mut rel_size: u64 = 0;

        let mut cur = dynamic_addr;
        for _ in 0..4096 {
            let Some(tag) = self.memory.read_u64(cur) else {
                break;
            };
            let Some(val) = self.memory.read_u64(cur.wrapping_add(8)) else {
                break;
            };
            cur = cur.wrapping_add(16);

            if tag == DT_NULL {
                break;
            }

            match tag {
                DT_RELA => rela_addr = val.wrapping_add(relocation_bias),
                DT_RELASZ => rela_size = val,
                DT_RELAENT => rela_ent = val,
                DT_JMPREL => jmprel_addr = val.wrapping_add(relocation_bias),
                DT_PLTRELSZ => jmprel_size = val,
                DT_REL => rel_addr = val.wrapping_add(relocation_bias),
                DT_RELSZ => rel_size = val,
                _ => {}
            }
        }

        const R_AARCH64_RELATIVE: u64 = 0x403; // 1027

        let mut applied = 0u64;
        let mut skipped = 0u64;

        // Process DT_RELA entries.
        if rela_addr != 0 && rela_size != 0 && rela_ent >= 24 {
            let count = rela_size / rela_ent;
            for i in 0..count {
                let entry = rela_addr.wrapping_add(i * rela_ent);
                let Some(r_offset) = self.memory.read_u64(entry) else {
                    continue;
                };
                let Some(r_info) = self.memory.read_u64(entry.wrapping_add(8)) else {
                    continue;
                };
                let Some(r_addend) = self.memory.read_u64(entry.wrapping_add(16)) else {
                    continue;
                };
                let r_type = r_info & 0xFFFF_FFFF;

                if r_type == R_AARCH64_RELATIVE {
                    let target = relocation_bias.wrapping_add(r_offset);
                    let value = relocation_bias.wrapping_add(r_addend);
                    self.memory.write_u64(target, value);
                    applied += 1;
                } else {
                    skipped += 1;
                }
            }
        }

        // Process DT_JMPREL (PLT) entries — usually also RELA format.
        if jmprel_addr != 0 && jmprel_size != 0 {
            let jmprel_ent = rela_ent.max(24);
            let count = jmprel_size / jmprel_ent;
            for i in 0..count {
                let entry = jmprel_addr.wrapping_add(i * jmprel_ent);
                let Some(r_offset) = self.memory.read_u64(entry) else {
                    continue;
                };
                let Some(r_info) = self.memory.read_u64(entry.wrapping_add(8)) else {
                    continue;
                };
                let Some(r_addend) = self.memory.read_u64(entry.wrapping_add(16)) else {
                    continue;
                };
                let r_type = r_info & 0xFFFF_FFFF;

                if r_type == R_AARCH64_RELATIVE {
                    let target = relocation_bias.wrapping_add(r_offset);
                    let value = relocation_bias.wrapping_add(r_addend);
                    self.memory.write_u64(target, value);
                    applied += 1;
                } else {
                    skipped += 1;
                }
            }
        }

        eprintln!(
            "[reloc] applied {applied} R_AARCH64_RELATIVE relocations, skipped {skipped} others \
             (rela @ {rela_addr:#x} size {rela_size:#x}, jmprel @ {jmprel_addr:#x} size {jmprel_size:#x})"
        );
    }

    fn apply_runtime_hints(&mut self, hints: ProgramRuntimeHints) {
        let process_flags = hints.process_flags.unwrap_or_else(default_process_flags);
        let stack_size = align_up_page(
            hints
                .main_thread_stack_size
                .unwrap_or(DEFAULT_MAIN_THREAD_STACK_SIZE)
                .max(PAGE_SIZE),
        );

        self.memory.clear_low_regions();

        if let Some((start, end)) = low_region_for_process_flags(process_flags, self.reset_vector) {
            let _ = self.memory.map_low_region(start, end.saturating_sub(start));
        }

        let stack_end = self.high_memory_end & PAGE_MASK;
        self.state.sp = stack_end.saturating_sub(0x10) & !0xf;

        let thread_ptr = stack_end
            .saturating_sub(stack_size)
            .saturating_sub(PAGE_SIZE);
        let stack_start = thread_ptr.saturating_add(PAGE_SIZE);

        let _ = self.memory.map_high_region(
            thread_ptr,
            PAGE_SIZE,
            MEMORY_STATE_THREAD_LOCAL,
            MEMORY_PERMISSION_READ_WRITE,
        );
        let _ = self.memory.map_high_region(
            stack_start,
            stack_size,
            MEMORY_STATE_STACK,
            MEMORY_PERMISSION_READ_WRITE,
        );

        self.state.tpidr_el0 = 0;
        self.state.tpidrro_el0 = thread_ptr;
        self.state.tpidr_el1 = 0;

        let _ = self
            .memory
            .write_u64(thread_ptr.wrapping_add(0x1f8), thread_ptr);

        Self::install_thread_exit_trampoline(&self.memory);
        Self::install_startup_return_frame(&self.memory, self.state.sp);

        // Mirror Ryujinx's main-thread bootstrap register contract.
        self.state.write_x(0, 0);
        self.state.write_x(1, u64::from(MAIN_THREAD_HANDLE));
        self.state.write_x(18, 1);
        self.state.write_x(30, THREAD_EXIT_TRAMPOLINE_ADDR);
    }

    fn install_thread_exit_trampoline(memory: &SharedGuestMemory) {
        let _ = memory.write_u32(THREAD_EXIT_TRAMPOLINE_ADDR, SVC_EXIT_THREAD);
        let _ = memory.write_u32(THREAD_EXIT_TRAMPOLINE_ADDR.wrapping_add(4), BRK_PANIC);
    }

    fn install_startup_return_frame(memory: &SharedGuestMemory, sp: u64) {
        let _ = memory.write_u64(sp.wrapping_add(STARTUP_FRAME_OFFSET), 0);
        let _ = memory.write_u64(
            sp.wrapping_add(STARTUP_FRAME_OFFSET + 8),
            THREAD_EXIT_TRAMPOLINE_ADDR,
        );
    }

    pub fn snapshot(&self) -> EmulatorSnapshot {
        let kernel_ticks = self.kernel.lock().expect("kernel lock poisoned").tick_count;

        EmulatorSnapshot {
            pc: self.state.pc,
            sp: self.state.sp,
            nzcv: self.state.nzcv,
            current_el: self.state.current_el(),
            x0: self.state.read_x(0),
            x1: self.state.read_x(1),
            x2: self.state.read_x(2),
            x3: self.state.read_x(3),
            kernel_ticks,
        }
    }

    pub fn read_memory_u64(&self, address: u64) -> Option<u64> {
        self.memory.read_u64(address)
    }

    pub fn read_memory_u32(&self, address: u64) -> Option<u32> {
        self.memory.read_u32(address)
    }

    pub fn guest_base_address(&self) -> u64 {
        self.memory.guest_base_address()
    }

    pub fn guest_memory_size(&self) -> usize {
        self.memory.guest_memory_size()
    }

    /// Return a raw byte slice covering the entire guest memory region.
    ///
    /// # Safety
    /// The caller must ensure no concurrent writes are occurring.
    pub unsafe fn guest_memory_slice(&self) -> &[u8] {
        unsafe { self.memory.guest_memory_slice() }
    }

    pub fn sparse_pages(&self) -> Vec<(u64, Vec<u8>)> {
        self.memory.sparse_pages()
    }

    /// Switch to interpreter-only mode (disables JIT compilation and chaining).
    /// Also invalidates all cached compiled blocks so they are re-decoded as interpreted.
    pub fn set_force_interpreter(&self, force: bool) {
        self.dynarec.set_force_interpreter(force);
        if force {
            self.dynarec.invalidate_all();
        }
    }

    pub fn request_safepoint_interrupt(&self) {
        let _ = arm_registered_execution_context_pages();
        {
            let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
            kernel.request_safepoint_interrupt();
        }
        self.worker_pool.wake_workers();
    }

    pub fn release_safepoint_interrupt(&self) {
        let _ = disarm_registered_execution_context_pages();
        {
            let mut kernel = self.kernel.lock().expect("kernel lock poisoned");
            kernel.release_safepoint_interrupt();
        }
        self.worker_pool.wake_workers();
    }

    pub fn safepoint_quiesced(&self) -> bool {
        self.kernel
            .lock()
            .expect("kernel lock poisoned")
            .worker_threads_quiesced_for_safepoint()
    }
}
