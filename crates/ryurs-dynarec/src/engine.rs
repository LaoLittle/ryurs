use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::CString;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

#[cfg(unix)]
use std::sync::atomic::AtomicU64;

use anyhow::Result;

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
use anyhow::anyhow;

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "x86_64"
))]
use std::arch::asm;

use crate::arch::{execute_compiled_block_with_dispatcher_landing, JitDispatchFrame};
use crate::decoder::{
    decode_block, DecodeError, DecodedBlock, DecodedInstruction, ExtendType, PairAddressMode,
    ShiftType, SystemRegister,
};
use crate::memory::{FaultClass, MemoryAccessKind, MemoryStrategy, PteRegion};
use crate::runtime::{
    CpuState, ExceptionKind, ExecutionContext, ExitReason, SysRegError, DCZID_EL0_VALUE,
};
use crate::translator::{BlockCompiler, CompiledBlock, HostPcMapEntry};

pub trait MemoryInterface {
    fn fetch_u32(&mut self, address: u64) -> Option<u32>;

    fn fastmem_region(&mut self) -> Option<FastmemRegion> {
        None
    }

    fn pte_region(&mut self) -> Option<PteRegion> {
        None
    }

    fn classify_guest_access(&mut self, _address: u64, _kind: MemoryAccessKind) -> FaultClass {
        FaultClass::MemoryFault
    }

    fn read_u8(&mut self, _address: u64) -> Option<u8> {
        None
    }

    fn write_u8(&mut self, _address: u64, _value: u8) -> bool {
        false
    }

    fn read_u16(&mut self, _address: u64) -> Option<u16> {
        None
    }

    fn write_u16(&mut self, _address: u64, _value: u16) -> bool {
        false
    }

    fn read_u32(&mut self, _address: u64) -> Option<u32> {
        None
    }

    fn write_u32(&mut self, _address: u64, _value: u32) -> bool {
        false
    }

    fn read_u64(&mut self, _address: u64) -> Option<u64> {
        None
    }

    fn write_u64(&mut self, _address: u64, _value: u64) -> bool {
        false
    }
}

#[derive(Clone, Copy)]
pub struct FastmemRegion {
    pub guest_base: u64,
    pub host_ptr: *mut u8,
    pub size: u64,
}

struct FetchOnly<'a, F>
where
    F: FnMut(u64) -> Option<u32>,
{
    fetch: &'a mut F,
}

impl<F> MemoryInterface for FetchOnly<'_, F>
where
    F: FnMut(u64) -> Option<u32>,
{
    fn fetch_u32(&mut self, address: u64) -> Option<u32> {
        (self.fetch)(address)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DynarecOptions {
    pub block_max_instructions: usize,
    pub force_interpreter: bool,
}

impl Default for DynarecOptions {
    fn default() -> Self {
        Self {
            block_max_instructions: 48,
            force_interpreter: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StepResult {
    Advanced { start_pc: u64, end_pc: u64 },
    Trap { pc: u64, reason: ExitReason },
}

#[derive(Clone)]
enum CachedBlock {
    Compiled(CompiledBlock),
    Interpreted(DecodedBlock),
}

pub struct DynarecEngine {
    options: DynarecOptions,
    force_interpreter: std::sync::atomic::AtomicBool,
    block_cache: RwLock<HashMap<u64, Arc<CachedBlock>>>,
    arena: Arc<crate::jit_arena::JitArena>,
}

#[derive(Clone, Copy)]
struct ContextPageRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy)]
struct RegisteredContextPage {
    page: *mut u8,
    page_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SignalFaultRecord {
    fault_address: usize,
    is_write: bool,
    kind: FaultClass,
}

struct ThreadJitStack {
    base: *mut u8,
    size: usize,
    top: *mut u8,
}

// Raw page pointers refer to process-owned VM mappings and remain valid while registered.
unsafe impl Send for RegisteredContextPage {}
unsafe impl Sync for RegisteredContextPage {}

fn context_page_registry() -> &'static Mutex<HashMap<std::thread::ThreadId, RegisteredContextPage>>
{
    static REGISTRY: OnceLock<Mutex<HashMap<std::thread::ThreadId, RegisteredContextPage>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_current_context_page(page: *mut u8, page_size: usize) {
    let thread_id = std::thread::current().id();
    context_page_registry()
        .lock()
        .expect("context page registry lock poisoned")
        .insert(thread_id, RegisteredContextPage { page, page_size });
}

fn unregister_current_context_page() {
    let thread_id = std::thread::current().id();
    let _ = context_page_registry()
        .lock()
        .expect("context page registry lock poisoned")
        .remove(&thread_id);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContextPageProtection {
    NoAccess,
    ReadWrite,
}

#[cfg(unix)]
fn set_context_page_protection(
    page: *mut u8,
    page_size: usize,
    protection: ContextPageProtection,
) -> bool {
    let prot = match protection {
        ContextPageProtection::NoAccess => libc::PROT_NONE,
        ContextPageProtection::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
    };

    // SAFETY: caller provides a page-aligned live mapping and a valid protection mode.
    unsafe { libc::mprotect(page.cast::<libc::c_void>(), page_size, prot) == 0 }
}

#[cfg(not(unix))]
fn set_context_page_protection(
    _page: *mut u8,
    _page_size: usize,
    _protection: ContextPageProtection,
) -> bool {
    false
}

#[cfg(unix)]
static CONTEXT_PAGE_NAME_COUNTER: AtomicU64 = AtomicU64::new(1);

#[cfg(unix)]
fn map_shared_context_page(page_size: usize) -> Option<(*mut u8, *mut u8)> {
    let pid = unsafe { libc::getpid() };

    for _ in 0..8 {
        let nonce = CONTEXT_PAGE_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("/ryurs_ctx_{pid}_{nonce}");
        let c_name = CString::new(name).ok()?;
        let fd = unsafe {
            libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600,
            )
        };
        if fd < 0 {
            continue;
        }

        // SAFETY: name is valid and was just created by this thread.
        unsafe {
            let _ = libc::shm_unlink(c_name.as_ptr());
        }

        if unsafe { libc::ftruncate(fd, page_size as libc::off_t) } != 0 {
            // SAFETY: `fd` is valid and owned by this scope.
            unsafe {
                libc::close(fd);
            }
            return None;
        }

        let page = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if page == libc::MAP_FAILED {
            // SAFETY: `fd` is valid and owned by this scope.
            unsafe {
                libc::close(fd);
            }
            return None;
        }

        let alias = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        // SAFETY: `fd` is valid and owned by this scope.
        unsafe {
            libc::close(fd);
        }

        if alias == libc::MAP_FAILED {
            // SAFETY: `page` came from a successful mmap call above.
            unsafe {
                libc::munmap(page, page_size);
            }
            return None;
        }

        return Some((page.cast::<u8>(), alias.cast::<u8>()));
    }

    None
}

#[cfg(not(unix))]
fn map_shared_context_page(page_size: usize) -> Option<(*mut u8, *mut u8)> {
    // SAFETY: `calloc` allocates a writable host buffer or returns null.
    let ptr = unsafe { libc::calloc(1, page_size).cast::<u8>() };
    (!ptr.is_null()).then_some((ptr, ptr))
}

#[cfg(unix)]
unsafe fn unmap_context_page(page: *mut u8, page_size: usize) {
    // SAFETY: caller guarantees this pointer/size pair came from `mmap` and is still live.
    let _ = unsafe { libc::munmap(page.cast::<libc::c_void>(), page_size) };
}

#[cfg(not(unix))]
unsafe fn unmap_context_page(page: *mut u8, _page_size: usize) {
    // SAFETY: caller guarantees this pointer came from `calloc` and is still live.
    unsafe {
        libc::free(page.cast::<libc::c_void>());
    }
}

struct ThreadExecutionContextPage {
    page: *mut u8,
    alias: *mut u8,
    page_size: usize,
}

impl ThreadExecutionContextPage {
    fn allocate() -> Option<Self> {
        let page_size = page_size::get();
        let (page, alias) = map_shared_context_page(page_size)?;

        // SAFETY: `alias` maps at least one writable page and is properly aligned.
        unsafe {
            alias
                .cast::<ExecutionContext>()
                .write(ExecutionContext::default());
        }

        register_current_context_page(page, page_size);
        Some(Self {
            page,
            alias,
            page_size,
        })
    }

    fn execution_context_ptr(&self) -> *mut ExecutionContext {
        self.page.cast::<ExecutionContext>()
    }

    fn alias_context_mut(&mut self) -> &mut ExecutionContext {
        // SAFETY: `alias` points to writable memory owned by this thread-local context page.
        // When we call this method, the jit block is either trapped or returned to dispatcher,
        // we have exclusive access to the alias page and can safely mutate the execution context through it.
        unsafe { &mut *self.alias.cast::<ExecutionContext>() }
    }

    fn disarm(&self) -> bool {
        set_context_page_protection(self.page, self.page_size, ContextPageProtection::ReadWrite)
    }

    fn range(&self) -> ContextPageRange {
        ContextPageRange {
            start: self.page as usize,
            end: (self.page as usize).saturating_add(self.page_size),
        }
    }
}

impl Drop for ThreadExecutionContextPage {
    fn drop(&mut self) {
        unregister_current_context_page();
        TLS_CONTEXT_PAGE_RANGE.with(|slot| slot.set(None));

        // SAFETY: these pointers are owned by this thread-local page object.
        unsafe {
            unmap_context_page(self.alias, self.page_size);
            if self.page != self.alias {
                unmap_context_page(self.page, self.page_size);
            }
        }
    }
}

thread_local! {
    static TLS_MEMORY: Cell<Option<ThreadMemoryFns>> = const { Cell::new(None) };
    static TLS_COMPILER: Cell<*mut BlockCompiler> = const { Cell::new(std::ptr::null_mut()) };
    static TLS_HOST_STACK_SNAPSHOT: Cell<usize> = const { Cell::new(0) };
    static TLS_HOST_STACK_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TLS_ACTIVE_DISPATCH_FRAME: Cell<*mut JitDispatchFrame> = const { Cell::new(std::ptr::null_mut()) };
    static TLS_CONTEXT_PAGE_RANGE: Cell<Option<ContextPageRange>> = const { Cell::new(None) };
    static TLS_THREAD_EXECUTION_CONTEXT_PAGE: RefCell<Option<ThreadExecutionContextPage>> = const { RefCell::new(None) };
    static TLS_SIGNAL_FAULT_RECORD: Cell<Option<SignalFaultRecord>> = const { Cell::new(None) };
    static TLS_THREAD_JIT_STACK: RefCell<Option<ThreadJitStack>> = const { RefCell::new(None) };
}

fn with_thread_execution_context_page<R>(
    f: impl FnOnce(&mut ThreadExecutionContextPage) -> R,
) -> Option<R> {
    TLS_THREAD_EXECUTION_CONTEXT_PAGE.with(|slot| {
        let mut page_slot = slot.borrow_mut();
        if page_slot.is_none() {
            let page = ThreadExecutionContextPage::allocate()?;
            TLS_CONTEXT_PAGE_RANGE.with(|range| range.set(Some(page.range())));
            *page_slot = Some(page);
        }

        let page = page_slot
            .as_mut()
            .expect("execution context page must be initialized");
        TLS_CONTEXT_PAGE_RANGE.with(|range| range.set(Some(page.range())));
        Some(f(page))
    })
}

fn set_signal_fault_record(record: Option<SignalFaultRecord>) {
    TLS_SIGNAL_FAULT_RECORD.with(|slot| slot.set(record));
}

fn take_signal_fault_record() -> Option<SignalFaultRecord> {
    TLS_SIGNAL_FAULT_RECORD.with(|slot| {
        let record = slot.get();
        slot.set(None);
        record
    })
}

impl ThreadJitStack {
    fn allocate(size: usize) -> Option<Self> {
        let page_size = page_size::get();
        let aligned_size = size.div_ceil(page_size) * page_size;
        let total_size = aligned_size.checked_add(page_size * 2)?;

        #[cfg(unix)]
        unsafe {
            let mapping = libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            if mapping == libc::MAP_FAILED {
                return None;
            }
            let base = mapping.cast::<u8>();
            let usable = base.add(page_size);
            let top = usable.add(aligned_size);
            let _ = libc::mprotect(base.cast(), page_size, libc::PROT_NONE);
            let _ = libc::mprotect(top.cast(), page_size, libc::PROT_NONE);
            Some(Self {
                base,
                size: total_size,
                top,
            })
        }

        #[cfg(not(unix))]
        {
            let mut bytes = vec![0u8; aligned_size].into_boxed_slice();
            let base = bytes.as_mut_ptr();
            let top = unsafe { base.add(aligned_size) };
            std::mem::forget(bytes);
            Some(Self {
                base,
                size: aligned_size,
                top,
            })
        }
    }

    fn top(&self) -> *mut u8 {
        self.top
    }
}

impl Drop for ThreadJitStack {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = libc::munmap(self.base.cast(), self.size);
        }
    }
}

fn with_thread_jit_stack<R>(f: impl FnOnce(&ThreadJitStack) -> R) -> Option<R> {
    TLS_THREAD_JIT_STACK.with(|slot| {
        let mut stack = slot.borrow_mut();
        if stack.is_none() {
            *stack = Some(ThreadJitStack::allocate(1024 * 1024)?);
        }
        Some(f(stack.as_ref().expect("jit stack should be initialized")))
    })
}

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
fn fault_hits_thread_context_page(fault_address: usize) -> bool {
    TLS_CONTEXT_PAGE_RANGE.with(|slot| {
        let Some(range) = slot.get() else {
            return false;
        };

        fault_address >= range.start && fault_address < range.end
    })
}

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
fn disarm_current_context_page() -> bool {
    TLS_CONTEXT_PAGE_RANGE.with(|slot| {
        let Some(range) = slot.get() else {
            return false;
        };
        set_context_page_protection(
            range.start as *mut u8,
            range.end.saturating_sub(range.start),
            ContextPageProtection::ReadWrite,
        )
    })
}

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
unsafe fn read_fault_address(info: *mut libc::siginfo_t) -> Option<usize> {
    let info = unsafe { info.as_ref() }?;
    // SAFETY: the signal payload is valid for synchronous faults.
    Some(unsafe { info.si_addr() } as usize)
}

pub fn arm_registered_execution_context_pages() -> usize {
    let registry = context_page_registry()
        .lock()
        .expect("context page registry lock poisoned");

    registry
        .values()
        .filter(|page| {
            set_context_page_protection(page.page, page.page_size, ContextPageProtection::NoAccess)
        })
        .count()
}

pub fn disarm_registered_execution_context_pages() -> usize {
    let registry = context_page_registry()
        .lock()
        .expect("context page registry lock poisoned");

    registry
        .values()
        .filter(|page| {
            set_context_page_protection(page.page, page.page_size, ContextPageProtection::ReadWrite)
        })
        .count()
}

#[inline(always)]
fn capture_host_stack_pointer() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let sp: usize;
        // SAFETY: reading the stack pointer register is side-effect free.
        unsafe {
            asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    #[cfg(target_arch = "aarch64")]
    {
        let sp: usize;
        // SAFETY: reading the stack pointer register is side-effect free.
        unsafe {
            asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    #[cfg(target_arch = "riscv64")]
    {
        let sp: usize;
        // SAFETY: reading the stack pointer register is side-effect free.
        unsafe {
            asm!("mv {}, sp", out(reg) sp, options(nomem, nostack));
        }
        sp
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        0
    }
}

#[allow(dead_code)]
pub fn host_stack_snapshot_from_tls() -> Option<usize> {
    if let Some((_, landing_sp)) = dispatcher_landing_from_tls() {
        return Some(landing_sp);
    }

    let snapshot = TLS_HOST_STACK_SNAPSHOT.with(|slot| slot.get());
    if snapshot == 0 {
        return None;
    }

    Some(snapshot)
}

#[allow(dead_code)]
pub fn dispatcher_landing_from_tls() -> Option<(usize, usize)> {
    let frame_ptr = TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.get());
    let frame = unsafe { frame_ptr.as_ref() }?;
    if frame.landing_pc == 0 || frame.landing_sp == 0 {
        return None;
    }

    Some((frame.landing_pc, frame.landing_sp))
}

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
fn map_guest_pc_for_active_host_pc(host_pc: usize) -> Option<u64> {
    let frame_ptr = TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.get());
    let frame = unsafe { frame_ptr.as_ref() }?;

    if frame.host_code_size == 0 || frame.host_pc_map_len == 0 {
        return None;
    }

    let rel = host_pc.checked_sub(frame.host_code_start)?;
    let rel = u32::try_from(rel).ok()?;
    if rel >= frame.host_code_size {
        return None;
    }

    let entries =
        unsafe { std::slice::from_raw_parts(frame.host_pc_map_ptr, frame.host_pc_map_len) };
    let idx = entries.partition_point(|entry| entry.host_offset_start <= rel);
    let entry = entries.get(idx.checked_sub(1)?)?;
    (rel >= entry.host_offset_start && rel < entry.host_offset_end).then_some(entry.guest_pc)
}

#[allow(dead_code)]
pub unsafe fn return_to_dispatcher_from_signal(fault_pc_hint: Option<u64>) -> bool {
    let frame_ptr = TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.get());
    let Some(frame) = (unsafe { frame_ptr.as_mut() }) else {
        return false;
    };

    let Some(state) = (unsafe { frame.state_ptr.as_mut() }) else {
        return false;
    };

    state.pc = fault_pc_hint.unwrap_or(frame.block_pc);
    state.set_dispatcher_interrupt();

    frame.landing_pc != 0 && frame.landing_sp != 0
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn read_macos_fault_is_write(ucontext: *mut libc::c_void) -> Option<bool> {
    let uc = unsafe { (ucontext as *mut DarwinUContext64).as_ref() }?;
    let mcontext = unsafe { uc.uc_mcontext.as_ref() }?;
    Some((mcontext.__es.__esr & 0x40) != 0)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
unsafe fn read_macos_fault_is_write(_ucontext: *mut libc::c_void) -> Option<bool> {
    None
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn read_linux_fault_is_write(ucontext: *mut libc::c_void) -> Option<bool> {
    let uc = unsafe { (ucontext as *mut libc::ucontext_t).as_ref() }?;
    Some((uc.uc_mcontext.gregs[libc::REG_ERR as usize] & 2) != 0)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
unsafe fn read_linux_fault_is_write(_ucontext: *mut libc::c_void) -> Option<bool> {
    None
}

struct HostStackScope {
    previous_snapshot: usize,
    previous_active: bool,
    previous_frame_ptr: *mut JitDispatchFrame,
}

impl HostStackScope {
    fn enter(frame_ptr: *mut JitDispatchFrame) -> Self {
        let snapshot = capture_host_stack_pointer();
        let previous_snapshot = TLS_HOST_STACK_SNAPSHOT.with(|slot| {
            let previous = slot.get();
            slot.set(snapshot);
            previous
        });
        let previous_active = TLS_HOST_STACK_ACTIVE.with(|slot| {
            let previous = slot.get();
            slot.set(true);
            previous
        });
        let previous_frame_ptr = TLS_ACTIVE_DISPATCH_FRAME.with(|slot| {
            let previous = slot.get();
            slot.set(frame_ptr);
            previous
        });

        Self {
            previous_snapshot,
            previous_active,
            previous_frame_ptr,
        }
    }

    #[allow(dead_code)]
    fn is_active() -> bool {
        TLS_HOST_STACK_ACTIVE.with(|slot| slot.get())
    }
}

impl Drop for HostStackScope {
    fn drop(&mut self) {
        TLS_HOST_STACK_SNAPSHOT.with(|slot| slot.set(self.previous_snapshot));
        TLS_HOST_STACK_ACTIVE.with(|slot| slot.set(self.previous_active));
        TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.set(self.previous_frame_ptr));
    }
}

#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
))]
static FAULT_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
struct DarwinArmExceptionState64 {
    __far: u64,
    __esr: u32,
    __exception: u32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
struct DarwinArmThreadState64 {
    __x: [u64; 29],
    __fp: u64,
    __lr: u64,
    __sp: u64,
    __pc: u64,
    __cpsr: u32,
    __pad: u32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
struct DarwinMContext64 {
    __es: DarwinArmExceptionState64,
    __ss: DarwinArmThreadState64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
struct DarwinUContext64 {
    uc_onstack: libc::c_int,
    uc_sigmask: libc::sigset_t,
    uc_stack: libc::stack_t,
    uc_link: *mut DarwinUContext64,
    uc_mcsize: libc::size_t,
    uc_mcontext: *mut DarwinMContext64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn read_macos_aarch64_fault_pc(ucontext: *mut libc::c_void) -> Option<usize> {
    let uc = unsafe { (ucontext as *mut DarwinUContext64).as_ref() }?;
    let mcontext = unsafe { uc.uc_mcontext.as_ref() }?;
    Some(mcontext.__ss.__pc as usize)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn rewrite_macos_aarch64_ucontext(
    ucontext: *mut libc::c_void,
    landing_pc: usize,
    landing_sp: usize,
) -> bool {
    let Some(uc) = (unsafe { (ucontext as *mut DarwinUContext64).as_mut() }) else {
        return false;
    };
    let Some(mcontext) = (unsafe { uc.uc_mcontext.as_mut() }) else {
        return false;
    };

    mcontext.__ss.__pc = landing_pc as u64;
    mcontext.__ss.__sp = landing_sp as u64;
    true
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
extern "C" fn jit_fault_signal_handler_macos(
    signum: libc::c_int,
    info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    #[inline(never)]
    fn fatal_signal_exit(signum: libc::c_int) -> ! {
        unsafe { libc::_exit(128 + signum) }
    }

    let Some((landing_pc, landing_sp)) = dispatcher_landing_from_tls() else {
        fatal_signal_exit(signum);
    };
    let Some(fault_address) = (unsafe { read_fault_address(info) }) else {
        fatal_signal_exit(signum);
    };
    if fault_hits_thread_context_page(fault_address) {
        let _ = disarm_current_context_page();
        set_signal_fault_record(Some(SignalFaultRecord {
            fault_address,
            is_write: unsafe { read_macos_fault_is_write(ucontext) }.unwrap_or(false),
            kind: FaultClass::Polling,
        }));
    } else {
        set_signal_fault_record(Some(SignalFaultRecord {
            fault_address,
            is_write: unsafe { read_macos_fault_is_write(ucontext) }.unwrap_or(false),
            kind: FaultClass::MemoryFault,
        }));
    }

    let fault_guest_pc_hint =
        unsafe { read_macos_aarch64_fault_pc(ucontext) }.and_then(map_guest_pc_for_active_host_pc);

    let redirected = unsafe {
        return_to_dispatcher_from_signal(fault_guest_pc_hint)
            && rewrite_macos_aarch64_ucontext(ucontext, landing_pc, landing_sp)
    };
    if !redirected {
        fatal_signal_exit(signum);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn install_fault_handler() -> Result<()> {
    if FAULT_HANDLER_INSTALLED.swap(true, AtomicOrdering::AcqRel) {
        return Ok(());
    }

    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        if libc::sigemptyset(&mut action.sa_mask) != 0 {
            return Err(anyhow!(
                "sigemptyset failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        action.sa_flags = libc::SA_SIGINFO;
        action.sa_sigaction = jit_fault_signal_handler_macos as *const () as usize;

        if libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) != 0 {
            return Err(anyhow!(
                "sigaction(SIGSEGV) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        if libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()) != 0 {
            return Err(anyhow!(
                "sigaction(SIGBUS) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
    }

    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn read_linux_fault_pc(ucontext: *mut libc::c_void) -> Option<usize> {
    let uc = unsafe { (ucontext as *mut libc::ucontext_t).as_ref() }?;
    Some(uc.uc_mcontext.gregs[libc::REG_RIP as usize] as usize)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn rewrite_linux_ucontext(
    ucontext: *mut libc::c_void,
    landing_pc: usize,
    landing_sp: usize,
) -> bool {
    let Some(uc) = (unsafe { (ucontext as *mut libc::ucontext_t).as_mut() }) else {
        return false;
    };

    uc.uc_mcontext.gregs[libc::REG_RIP as usize] = landing_pc as libc::greg_t;
    uc.uc_mcontext.gregs[libc::REG_RSP as usize] = landing_sp as libc::greg_t;
    true
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn read_linux_fault_pc(ucontext: *mut libc::c_void) -> Option<usize> {
    let uc = unsafe { (ucontext as *mut libc::ucontext_t).as_ref() }?;
    Some(uc.uc_mcontext.pc as usize)
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn rewrite_linux_ucontext(
    ucontext: *mut libc::c_void,
    landing_pc: usize,
    landing_sp: usize,
) -> bool {
    let Some(uc) = (unsafe { (ucontext as *mut libc::ucontext_t).as_mut() }) else {
        return false;
    };

    uc.uc_mcontext.pc = landing_pc as u64;
    uc.uc_mcontext.sp = landing_sp as u64;
    true
}

#[cfg(all(target_os = "linux", target_arch = "riscv64"))]
unsafe fn read_linux_fault_pc(ucontext: *mut libc::c_void) -> Option<usize> {
    let uc = unsafe { (ucontext as *mut libc::ucontext_t).as_ref() }?;
    Some(uc.uc_mcontext.__gregs[libc::REG_PC] as usize)
}

#[cfg(all(target_os = "linux", target_arch = "riscv64"))]
unsafe fn rewrite_linux_ucontext(
    ucontext: *mut libc::c_void,
    landing_pc: usize,
    landing_sp: usize,
) -> bool {
    let Some(uc) = (unsafe { (ucontext as *mut libc::ucontext_t).as_mut() }) else {
        return false;
    };

    uc.uc_mcontext.__gregs[libc::REG_PC] = landing_pc as libc::c_ulong;
    uc.uc_mcontext.__gregs[libc::REG_SP] = landing_sp as libc::c_ulong;
    true
}

#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
))]
extern "C" fn jit_fault_signal_handler_linux(
    signum: libc::c_int,
    info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    #[inline(never)]
    fn fatal_signal_exit(signum: libc::c_int) -> ! {
        unsafe { libc::_exit(128 + signum) }
    }

    let Some((landing_pc, landing_sp)) = dispatcher_landing_from_tls() else {
        fatal_signal_exit(signum);
    };
    let Some(fault_address) = (unsafe { read_fault_address(info) }) else {
        fatal_signal_exit(signum);
    };
    if fault_hits_thread_context_page(fault_address) {
        let _ = disarm_current_context_page();
        set_signal_fault_record(Some(SignalFaultRecord {
            fault_address,
            is_write: unsafe { read_linux_fault_is_write(ucontext) }.unwrap_or(false),
            kind: FaultClass::Polling,
        }));
    } else {
        set_signal_fault_record(Some(SignalFaultRecord {
            fault_address,
            is_write: unsafe { read_linux_fault_is_write(ucontext) }.unwrap_or(false),
            kind: FaultClass::MemoryFault,
        }));
    }

    let fault_guest_pc_hint =
        unsafe { read_linux_fault_pc(ucontext) }.and_then(map_guest_pc_for_active_host_pc);

    let redirected = unsafe {
        return_to_dispatcher_from_signal(fault_guest_pc_hint)
            && rewrite_linux_ucontext(ucontext, landing_pc, landing_sp)
    };
    if !redirected {
        fatal_signal_exit(signum);
    }
}

#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
))]
pub fn install_fault_handler() -> Result<()> {
    if FAULT_HANDLER_INSTALLED.swap(true, AtomicOrdering::AcqRel) {
        return Ok(());
    }

    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        if libc::sigemptyset(&mut action.sa_mask) != 0 {
            return Err(anyhow!(
                "sigemptyset failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        action.sa_flags = libc::SA_SIGINFO;
        action.sa_sigaction = jit_fault_signal_handler_linux as *const () as usize;

        if libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) != 0 {
            return Err(anyhow!(
                "sigaction(SIGSEGV) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        if libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()) != 0 {
            return Err(anyhow!(
                "sigaction(SIGBUS) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
    }

    Ok(())
}

#[cfg(not(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    )
)))]
pub fn install_fault_handler() -> Result<()> {
    Ok(())
}

#[derive(Clone, Copy)]
struct ThreadMemoryFns {
    ctx: *mut (),
    fastmem: Option<FastmemRegion>,
    read_u8: unsafe fn(*mut (), u64) -> Option<u8>,
    write_u8: unsafe fn(*mut (), u64, u8) -> bool,
    read_u16: unsafe fn(*mut (), u64) -> Option<u16>,
    write_u16: unsafe fn(*mut (), u64, u16) -> bool,
    read_u32: unsafe fn(*mut (), u64) -> Option<u32>,
    write_u32: unsafe fn(*mut (), u64, u32) -> bool,
    read_u64: unsafe fn(*mut (), u64) -> Option<u64>,
    write_u64: unsafe fn(*mut (), u64, u64) -> bool,
}

struct ThreadMemoryBinding;

fn classify_bound_signal_fault<M>(
    memory: &mut M,
    fastmem_region: Option<FastmemRegion>,
    record: SignalFaultRecord,
) -> FaultClass
where
    M: MemoryInterface,
{
    match record.kind {
        FaultClass::Polling => FaultClass::Polling,
        FaultClass::GranularityMismatch => FaultClass::GranularityMismatch,
        FaultClass::HostFault => FaultClass::HostFault,
        FaultClass::MemoryFault => {
            let Some(region) = fastmem_region else {
                return FaultClass::MemoryFault;
            };
            let start = region.host_ptr as usize;
            let end = start.saturating_add(region.size as usize);
            if record.fault_address < start || record.fault_address >= end {
                return FaultClass::HostFault;
            }
            let guest_address = region.guest_base + (record.fault_address - start) as u64;
            memory.classify_guest_access(
                guest_address,
                if record.is_write {
                    MemoryAccessKind::Write
                } else {
                    MemoryAccessKind::Read
                },
            )
        }
    }
}

fn preferred_memory_strategy<M>(memory: &mut M) -> MemoryStrategy
where
    M: MemoryInterface,
{
    if memory.fastmem_region().is_some() {
        MemoryStrategy::Fastmem
    } else if memory.pte_region().is_some() {
        MemoryStrategy::PteChecked
    } else {
        MemoryStrategy::Memhelper
    }
}

fn downgraded_memory_strategy<M>(memory: &mut M, strategy: MemoryStrategy) -> MemoryStrategy
where
    M: MemoryInterface,
{
    match strategy {
        MemoryStrategy::Fastmem if memory.pte_region().is_some() => MemoryStrategy::PteChecked,
        MemoryStrategy::Fastmem => MemoryStrategy::Memhelper,
        MemoryStrategy::PteChecked => MemoryStrategy::Memhelper,
        MemoryStrategy::Memhelper => MemoryStrategy::Interpret,
        MemoryStrategy::Interpret => MemoryStrategy::Interpret,
    }
}

impl ThreadMemoryBinding {
    fn bind<M>(memory: &mut M) -> Self
    where
        M: MemoryInterface,
    {
        unsafe fn read_u8_impl<M: MemoryInterface>(ctx: *mut (), address: u64) -> Option<u8> {
            // Safety: The context is guaranteed to be valid and exclusively owned by the current thread for the duration of this binding.
            unsafe { (&mut *(ctx as *mut M)).read_u8(address) }
        }

        unsafe fn write_u8_impl<M: MemoryInterface>(ctx: *mut (), address: u64, value: u8) -> bool {
            unsafe { (&mut *(ctx as *mut M)).write_u8(address, value) }
        }

        unsafe fn read_u16_impl<M: MemoryInterface>(ctx: *mut (), address: u64) -> Option<u16> {
            unsafe { (&mut *(ctx as *mut M)).read_u16(address) }
        }

        unsafe fn write_u16_impl<M: MemoryInterface>(
            ctx: *mut (),
            address: u64,
            value: u16,
        ) -> bool {
            unsafe { (&mut *(ctx as *mut M)).write_u16(address, value) }
        }

        unsafe fn read_u32_impl<M: MemoryInterface>(ctx: *mut (), address: u64) -> Option<u32> {
            unsafe { (&mut *(ctx as *mut M)).read_u32(address) }
        }

        unsafe fn write_u32_impl<M: MemoryInterface>(
            ctx: *mut (),
            address: u64,
            value: u32,
        ) -> bool {
            unsafe { (&mut *(ctx as *mut M)).write_u32(address, value) }
        }

        unsafe fn read_u64_impl<M: MemoryInterface>(ctx: *mut (), address: u64) -> Option<u64> {
            unsafe { (&mut *(ctx as *mut M)).read_u64(address) }
        }

        unsafe fn write_u64_impl<M: MemoryInterface>(
            ctx: *mut (),
            address: u64,
            value: u64,
        ) -> bool {
            unsafe { (&mut *(ctx as *mut M)).write_u64(address, value) }
        }

        let fns = ThreadMemoryFns {
            ctx: memory as *mut M as *mut (),
            fastmem: memory.fastmem_region(),
            read_u8: read_u8_impl::<M>,
            write_u8: write_u8_impl::<M>,
            read_u16: read_u16_impl::<M>,
            write_u16: write_u16_impl::<M>,
            read_u32: read_u32_impl::<M>,
            write_u32: write_u32_impl::<M>,
            read_u64: read_u64_impl::<M>,
            write_u64: write_u64_impl::<M>,
        };

        TLS_MEMORY.with(|slot| {
            slot.set(Some(fns));
        });
        Self
    }
}

impl Drop for ThreadMemoryBinding {
    fn drop(&mut self) {
        TLS_MEMORY.with(|slot| {
            slot.set(None);
        });
    }
}

fn fastmem_host_ptr(region: FastmemRegion, address: u64, access_size: u64) -> Option<*mut u8> {
    if region.host_ptr.is_null() {
        return None;
    }

    let offset = address.checked_sub(region.guest_base)?;
    let end = offset.checked_add(access_size)?;
    if end > region.size {
        return None;
    }

    let offset = usize::try_from(offset).ok()?;
    let host_base = region.host_ptr as usize;
    let host_addr = host_base.checked_add(offset)?;
    Some(host_addr as *mut u8)
}

fn thread_fastmem_read_u8(region: Option<FastmemRegion>, address: u64) -> Option<u8> {
    let host_ptr = fastmem_host_ptr(region?, address, 1)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    Some(unsafe { std::ptr::read_unaligned(host_ptr.cast::<u8>()) })
}

fn thread_fastmem_write_u8(region: Option<FastmemRegion>, address: u64, value: u8) -> Option<bool> {
    let host_ptr = fastmem_host_ptr(region?, address, 1)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    unsafe {
        std::ptr::write_unaligned(host_ptr.cast::<u8>(), value);
    }
    Some(true)
}

fn thread_fastmem_read_u16(region: Option<FastmemRegion>, address: u64) -> Option<u16> {
    let host_ptr = fastmem_host_ptr(region?, address, 2)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    let value = unsafe { std::ptr::read_unaligned(host_ptr.cast::<u16>()) };
    Some(u16::from_le(value))
}

fn thread_fastmem_write_u16(
    region: Option<FastmemRegion>,
    address: u64,
    value: u16,
) -> Option<bool> {
    let host_ptr = fastmem_host_ptr(region?, address, 2)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    unsafe {
        std::ptr::write_unaligned(host_ptr.cast::<u16>(), value.to_le());
    }
    Some(true)
}

fn thread_fastmem_read_u32(region: Option<FastmemRegion>, address: u64) -> Option<u32> {
    let host_ptr = fastmem_host_ptr(region?, address, 4)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    let value = unsafe { std::ptr::read_unaligned(host_ptr.cast::<u32>()) };
    Some(u32::from_le(value))
}

fn thread_fastmem_write_u32(
    region: Option<FastmemRegion>,
    address: u64,
    value: u32,
) -> Option<bool> {
    let host_ptr = fastmem_host_ptr(region?, address, 4)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    unsafe {
        std::ptr::write_unaligned(host_ptr.cast::<u32>(), value.to_le());
    }
    Some(true)
}

fn thread_fastmem_read_u64(region: Option<FastmemRegion>, address: u64) -> Option<u64> {
    let host_ptr = fastmem_host_ptr(region?, address, 8)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    let value = unsafe { std::ptr::read_unaligned(host_ptr.cast::<u64>()) };
    Some(u64::from_le(value))
}

fn thread_fastmem_write_u64(
    region: Option<FastmemRegion>,
    address: u64,
    value: u64,
) -> Option<bool> {
    let host_ptr = fastmem_host_ptr(region?, address, 8)?;
    // SAFETY: `fastmem_host_ptr` validates the full in-bounds range for this access.
    unsafe {
        std::ptr::write_unaligned(host_ptr.cast::<u64>(), value.to_le());
    }
    Some(true)
}

fn thread_mem_read_u8(address: u64) -> Option<u8> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(value) = thread_fastmem_read_u8(fns.fastmem, address) {
            return Some(value);
        }
        Some(unsafe { (fns.read_u8)(fns.ctx, address) }).flatten()
    })
}

fn thread_mem_write_u8(address: u64, value: u8) -> Option<bool> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(ok) = thread_fastmem_write_u8(fns.fastmem, address, value) {
            return Some(ok);
        }
        Some(unsafe { (fns.write_u8)(fns.ctx, address, value) })
    })
}

fn thread_mem_read_u16(address: u64) -> Option<u16> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(value) = thread_fastmem_read_u16(fns.fastmem, address) {
            return Some(value);
        }
        Some(unsafe { (fns.read_u16)(fns.ctx, address) }).flatten()
    })
}

fn thread_mem_write_u16(address: u64, value: u16) -> Option<bool> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(ok) = thread_fastmem_write_u16(fns.fastmem, address, value) {
            return Some(ok);
        }
        Some(unsafe { (fns.write_u16)(fns.ctx, address, value) })
    })
}

fn thread_mem_read_u32(address: u64) -> Option<u32> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(value) = thread_fastmem_read_u32(fns.fastmem, address) {
            return Some(value);
        }
        Some(unsafe { (fns.read_u32)(fns.ctx, address) }).flatten()
    })
}

fn thread_mem_write_u32(address: u64, value: u32) -> Option<bool> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(ok) = thread_fastmem_write_u32(fns.fastmem, address, value) {
            return Some(ok);
        }
        Some(unsafe { (fns.write_u32)(fns.ctx, address, value) })
    })
}

fn thread_mem_read_u64(address: u64) -> Option<u64> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(value) = thread_fastmem_read_u64(fns.fastmem, address) {
            return Some(value);
        }
        Some(unsafe { (fns.read_u64)(fns.ctx, address) }).flatten()
    })
}

fn thread_mem_write_u64(address: u64, value: u64) -> Option<bool> {
    TLS_MEMORY.with(|slot| {
        let fns = slot.get()?;
        if let Some(ok) = thread_fastmem_write_u64(fns.fastmem, address, value) {
            return Some(ok);
        }
        Some(unsafe { (fns.write_u64)(fns.ctx, address, value) })
    })
}

type EagerDirectTarget = (u64, usize);
type EagerDirectTargets = [Option<EagerDirectTarget>; 2];

fn compile_with_thread_local(
    arena: &Arc<crate::jit_arena::JitArena>,
    block: &DecodedBlock,
    eager_direct_targets: EagerDirectTargets,
    strategy: MemoryStrategy,
) -> Result<CompiledBlock> {
    TLS_COMPILER.with(|slot| {
        let mut compiler_ptr = slot.get();
        if compiler_ptr.is_null() {
            let compiler = Box::new(BlockCompiler::new(arena.clone())?);
            compiler_ptr = Box::into_raw(compiler);
            slot.set(compiler_ptr);
        }

        // SAFETY: pointer is initialized once per thread and leaked for process lifetime.
        unsafe {
            compiler_ptr
                .as_mut()
                .expect("thread-local compiler must be initialized")
                .compile(block, eager_direct_targets, strategy)
        }
    })
}

fn eager_direct_branch_targets(block: &DecodedBlock) -> [Option<u64>; 2] {
    let Some(last) = block.instructions.last() else {
        return [None, None];
    };

    match last {
        DecodedInstruction::B { target } | DecodedInstruction::Bl { target } => {
            [Some(*target), None]
        }
        DecodedInstruction::BCond { target, .. }
        | DecodedInstruction::Cbz { target, .. }
        | DecodedInstruction::Cbnz { target, .. }
        | DecodedInstruction::Cbz32 { target, .. }
        | DecodedInstruction::Cbnz32 { target, .. }
        | DecodedInstruction::Tbz { target, .. }
        | DecodedInstruction::Tbnz { target, .. } => [Some(*target), Some(block.fallthrough_pc)],
        _ => [None, None],
    }
}

fn decode_guest_block<M>(
    start_pc: u64,
    block_max_instructions: usize,
    memory: &mut M,
) -> Result<DecodedBlock, DecodeError>
where
    M: MemoryInterface,
{
    decode_block(start_pc, block_max_instructions, |pc| memory.fetch_u32(pc))
}

fn mark_memory_fault(state_ptr: *mut CpuState) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };
    state.set_memory_fault();
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_read_u8(state_ptr: *mut CpuState, address: u64) -> u64 {
    match thread_mem_read_u8(address) {
        Some(value) => u64::from(value),
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_read_u16(state_ptr: *mut CpuState, address: u64) -> u64 {
    match thread_mem_read_u16(address) {
        Some(value) => u64::from(value),
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_read_u32(state_ptr: *mut CpuState, address: u64) -> u64 {
    match thread_mem_read_u32(address) {
        Some(value) => u64::from(value),
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_read_u64(state_ptr: *mut CpuState, address: u64) -> u64 {
    match thread_mem_read_u64(address) {
        Some(value) => value,
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_write_u8(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u32 {
    match thread_mem_write_u8(address, value as u8) {
        Some(true) => 1,
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_write_u16(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u32 {
    match thread_mem_write_u16(address, value as u16) {
        Some(true) => 1,
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_write_u32(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u32 {
    match thread_mem_write_u32(address, value as u32) {
        Some(true) => 1,
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mem_write_u64(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u32 {
    match thread_mem_write_u64(address, value) {
        Some(true) => 1,
        _ => {
            mark_memory_fault(state_ptr);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_state_clrex(state_ptr: *mut CpuState) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };

    state.clear_exclusive_reservation();
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_uminp_vector(state_ptr: *mut CpuState, packed: u64) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };

    let vd = (packed & 0xFF) as u8;
    let vn = ((packed >> 8) & 0xFF) as u8;
    let vm = ((packed >> 16) & 0xFF) as u8;
    let q = ((packed >> 24) & 1) != 0;
    let size = ((packed >> 32) & 0xFF) as u8;
    execute_uminp_vector(state, vd, vn, vm, q, size);
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_umaxp_vector(state_ptr: *mut CpuState, packed: u64) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };

    let vd = (packed & 0xFF) as u8;
    let vn = ((packed >> 8) & 0xFF) as u8;
    let vm = ((packed >> 16) & 0xFF) as u8;
    let q = ((packed >> 24) & 1) != 0;
    let size = ((packed >> 32) & 0xFF) as u8;
    execute_umaxp_vector(state, vd, vn, vm, q, size);
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_shrn_vector(state_ptr: *mut CpuState, packed: u64) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };

    let vd = (packed & 0xFF) as u8;
    let vn = ((packed >> 8) & 0xFF) as u8;
    let q = ((packed >> 16) & 1) != 0;
    let size = ((packed >> 24) & 0xFF) as u8;
    let shift = ((packed >> 32) & 0xFF) as u8;
    execute_shrn_vector(state, vd, vn, q, size, shift);
}

fn sys_reg_from_index(idx: u64) -> Option<SystemRegister> {
    Some(match idx {
        0 => SystemRegister::CurrentEl,
        1 => SystemRegister::Nzcv,
        2 => SystemRegister::Daif,
        3 => SystemRegister::Spsel,
        4 => SystemRegister::CntfrqEl0,
        5 => SystemRegister::CtrEl0,
        6 => SystemRegister::DczidEl0,
        7 => SystemRegister::CntpctEl0,
        8 => SystemRegister::CntvctEl0,
        9 => SystemRegister::TpidrEl0,
        10 => SystemRegister::TpidrroEl0,
        11 => SystemRegister::TpidrEl1,
        12 => SystemRegister::SpEl0,
        13 => SystemRegister::SpEl1,
        14 => SystemRegister::SpEl2,
        15 => SystemRegister::SpEl3,
        16 => SystemRegister::SpsrEl1,
        17 => SystemRegister::ElrEl1,
        18 => SystemRegister::SpsrEl2,
        19 => SystemRegister::ElrEl2,
        20 => SystemRegister::SpsrEl3,
        21 => SystemRegister::ElrEl3,
        22 => SystemRegister::VbarEl1,
        23 => SystemRegister::VbarEl2,
        24 => SystemRegister::VbarEl3,
        _ => return None,
    })
}

pub fn sys_reg_to_index(reg: &SystemRegister) -> u64 {
    match reg {
        SystemRegister::CurrentEl => 0,
        SystemRegister::Nzcv => 1,
        SystemRegister::Daif => 2,
        SystemRegister::Spsel => 3,
        SystemRegister::CntfrqEl0 => 4,
        SystemRegister::CtrEl0 => 5,
        SystemRegister::DczidEl0 => 6,
        SystemRegister::CntpctEl0 => 7,
        SystemRegister::CntvctEl0 => 8,
        SystemRegister::TpidrEl0 => 9,
        SystemRegister::TpidrroEl0 => 10,
        SystemRegister::TpidrEl1 => 11,
        SystemRegister::SpEl0 => 12,
        SystemRegister::SpEl1 => 13,
        SystemRegister::SpEl2 => 14,
        SystemRegister::SpEl3 => 15,
        SystemRegister::SpsrEl1 => 16,
        SystemRegister::ElrEl1 => 17,
        SystemRegister::SpsrEl2 => 18,
        SystemRegister::ElrEl2 => 19,
        SystemRegister::SpsrEl3 => 20,
        SystemRegister::ElrEl3 => 21,
        SystemRegister::VbarEl1 => 22,
        SystemRegister::VbarEl2 => 23,
        SystemRegister::VbarEl3 => 24,
    }
}

/// JIT helper for MRS instruction. Returns the register value, or sets privilege fault.
#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_mrs(state_ptr: *mut CpuState, reg_idx: u64) -> u64 {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return 0;
    };

    let Some(reg) = sys_reg_from_index(reg_idx) else {
        state.set_privilege_fault();
        return 0;
    };

    match state.read_sys_reg(reg) {
        Ok(value) => value,
        Err(_) => {
            state.set_privilege_fault();
            0
        }
    }
}

/// JIT helper for MSR instruction. Returns 0 on success, sets privilege fault on failure.
#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_msr(state_ptr: *mut CpuState, reg_idx: u64, value: u64) {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return;
    };

    let Some(reg) = sys_reg_from_index(reg_idx) else {
        state.set_privilege_fault();
        return;
    };

    match state.write_sys_reg(reg, value) {
        Ok(()) => {}
        Err(_) => {
            state.set_privilege_fault();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_record_link_miss(
    execution_context_ptr: *mut ExecutionContext,
    source_pc: u64,
    next_pc: u64,
) {
    let Some(execution_context) = (unsafe { execution_context_ptr.as_mut() }) else {
        return;
    };

    execution_context.late_link_source_pc = source_pc;
    execution_context.late_link_next_pc = next_pc;
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_exclusive_load_u32(state_ptr: *mut CpuState, address: u64) -> u64 {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return 0;
    };

    match thread_mem_read_u32(address) {
        Some(value) => {
            state.set_exclusive_reservation(address, 4);
            u64::from(value)
        }
        None => {
            state.set_memory_fault();
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_exclusive_load_u64(state_ptr: *mut CpuState, address: u64) -> u64 {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return 0;
    };

    match thread_mem_read_u64(address) {
        Some(value) => {
            state.set_exclusive_reservation(address, 8);
            value
        }
        None => {
            state.set_memory_fault();
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_exclusive_store_u32(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u64 {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return 1;
    };

    let mut status = 1u64;
    if state.check_exclusive_reservation(address, 4) {
        match thread_mem_write_u32(address, value as u32) {
            Some(true) => status = 0,
            _ => state.set_memory_fault(),
        }
    }

    state.clear_exclusive_reservation();
    status
}

#[unsafe(no_mangle)]
pub extern "C" fn ryurs_jit_exclusive_store_u64(
    state_ptr: *mut CpuState,
    address: u64,
    value: u64,
) -> u64 {
    let Some(state) = (unsafe { state_ptr.as_mut() }) else {
        return 1;
    };

    let mut status = 1u64;
    if state.check_exclusive_reservation(address, 8) {
        match thread_mem_write_u64(address, value) {
            Some(true) => status = 0,
            _ => state.set_memory_fault(),
        }
    }

    state.clear_exclusive_reservation();
    status
}

impl DynarecEngine {
    pub fn new(options: DynarecOptions) -> Result<Self> {
        install_fault_handler()?;
        let arena = Arc::new(
            crate::jit_arena::JitArena::new()
                .map_err(|e| anyhow::anyhow!("failed to create JIT arena: {e}"))?,
        );
        let force = options.force_interpreter;
        Ok(Self {
            options,
            force_interpreter: std::sync::atomic::AtomicBool::new(force),
            block_cache: RwLock::new(HashMap::new()),
            arena,
        })
    }

    pub fn set_force_interpreter(&self, force: bool) {
        self.force_interpreter
            .store(force, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn invalidate_all(&self) {
        self.block_cache
            .write()
            .expect("dynarec block cache poisoned")
            .clear();
    }

    pub fn block_return_slot_ptr(&self, block_pc: u64) -> Option<usize> {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        match cache.get(&block_pc)?.as_ref() {
            CachedBlock::Compiled(block) => Some(block.return_address_slot_ptr()),
            CachedBlock::Interpreted(_) => None,
        }
    }

    pub fn block_return_link_entry_count(&self, block_pc: u64) -> Option<usize> {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        match cache.get(&block_pc)?.as_ref() {
            CachedBlock::Compiled(block) => Some(block.return_link_entry_count()),
            CachedBlock::Interpreted(_) => None,
        }
    }

    pub fn block_return_links(&self, block_pc: u64) -> Option<Vec<(u64, usize)>> {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        match cache.get(&block_pc)?.as_ref() {
            CachedBlock::Compiled(block) => Some(block.snapshot_return_links()),
            CachedBlock::Interpreted(_) => None,
        }
    }

    pub fn block_dispatch_target(&self, block_pc: u64) -> Option<usize> {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        match cache.get(&block_pc)?.as_ref() {
            CachedBlock::Compiled(block) => Some(block.dispatch_target()),
            CachedBlock::Interpreted(_) => None,
        }
    }

    pub fn initialize_block_return_target(
        &self,
        block_pc: u64,
        expected_next_pc: u64,
        target: usize,
    ) -> bool {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        let Some(cached) = cache.get(&block_pc) else {
            return false;
        };

        let CachedBlock::Compiled(block) = cached.as_ref() else {
            return false;
        };

        block.initialize_return_link(expected_next_pc, target);
        true
    }

    pub fn patch_block_return_target(
        &self,
        block_pc: u64,
        expected_next_pc: u64,
        target: usize,
    ) -> bool {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        let Some(cached) = cache.get(&block_pc) else {
            return false;
        };

        let CachedBlock::Compiled(block) = cached.as_ref() else {
            return false;
        };

        block.patch_return_link(expected_next_pc, target);
        true
    }

    pub fn clear_block_return_target(&self, block_pc: u64) -> bool {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        let Some(cached) = cache.get(&block_pc) else {
            return false;
        };

        let CachedBlock::Compiled(block) = cached.as_ref() else {
            return false;
        };

        block.clear_return_link();
        true
    }

    pub fn patch_block_return_edge(
        &self,
        source_pc: u64,
        expected_next_pc: u64,
        target_block_pc: u64,
    ) -> bool {
        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");

        let Some(source_cached) = cache.get(&source_pc) else {
            return false;
        };
        let Some(target_cached) = cache.get(&target_block_pc) else {
            return false;
        };

        let CachedBlock::Compiled(source_block) = source_cached.as_ref() else {
            return false;
        };
        let CachedBlock::Compiled(target_block) = target_cached.as_ref() else {
            return false;
        };

        source_block.patch_return_link(expected_next_pc, target_block.entry_ptr());
        true
    }

    fn eager_direct_link_hints(&self, block: &DecodedBlock) -> EagerDirectTargets {
        let branch_targets = eager_direct_branch_targets(block);
        if !branch_targets.iter().any(Option::is_some) {
            return [None, None];
        }

        let cache = self
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");

        let mut hints = [None, None];
        for (index, target_pc) in branch_targets.into_iter().enumerate() {
            let Some(target_pc) = target_pc else {
                continue;
            };

            let Some(target_cached) = cache.get(&target_pc) else {
                continue;
            };

            let target_entry = match target_cached.as_ref() {
                CachedBlock::Compiled(target) => target.entry_ptr(),
                CachedBlock::Interpreted(_) => continue,
            };

            hints[index] = Some((target_pc, target_entry));
        }

        hints
    }

    fn initialize_eager_return_links(
        block: &CompiledBlock,
        eager_direct_targets: EagerDirectTargets,
    ) {
        for (target_pc, target_entry) in eager_direct_targets.into_iter().flatten() {
            block.initialize_return_link(target_pc, target_entry);
        }
    }

    fn ensure_compiled_block_for_late_link<M>(&self, block_pc: u64, memory: &mut M) -> bool
    where
        M: MemoryInterface,
    {
        {
            let cache = self
                .block_cache
                .read()
                .expect("dynarec block cache poisoned");
            if let Some(cached) = cache.get(&block_pc) {
                return matches!(cached.as_ref(), CachedBlock::Compiled(_));
            }
        }

        let decoded_block =
            match decode_guest_block(block_pc, self.options.block_max_instructions, memory) {
                Ok(block) => block,
                Err(_) => return false,
            };

        if decoded_block
            .instructions
            .iter()
            .any(DecodedInstruction::requires_interpreter)
        {
            let mut cache = self
                .block_cache
                .write()
                .expect("dynarec block cache poisoned");
            cache
                .entry(block_pc)
                .or_insert_with(|| Arc::new(CachedBlock::Interpreted(decoded_block)));
            return false;
        }

        let eager_direct_targets = self.eager_direct_link_hints(&decoded_block);
        let strategy = preferred_memory_strategy(memory);
        let compiled = match compile_with_thread_local(
            &self.arena,
            &decoded_block,
            eager_direct_targets,
            strategy,
        ) {
            Ok(compiled) => compiled,
            Err(_) => return false,
        };
        Self::initialize_eager_return_links(&compiled, eager_direct_targets);

        let mut cache = self
            .block_cache
            .write()
            .expect("dynarec block cache poisoned");
        let entry = cache
            .entry(block_pc)
            .or_insert_with(|| Arc::new(CachedBlock::Compiled(compiled)));
        matches!(entry.as_ref(), CachedBlock::Compiled(_))
    }

    fn try_patch_late_link_edge<M>(&self, source_pc: u64, next_pc: u64, memory: &mut M)
    where
        M: MemoryInterface,
    {
        // PTE-backed memory is still correctness-first. Avoid compiling and patching
        // late-link edges on that path until the direct-edge strategy is stable.
        if memory.pte_region().is_some() {
            return;
        }

        if self.patch_block_return_edge(source_pc, next_pc, next_pc) {
            return;
        }

        if !self.ensure_compiled_block_for_late_link(next_pc, memory) {
            return;
        }

        let _ = self.patch_block_return_edge(source_pc, next_pc, next_pc);
    }

    pub fn step<F>(&self, state: &mut CpuState, fetch: &mut F) -> Result<StepResult>
    where
        F: FnMut(u64) -> Option<u32>,
    {
        let mut memory = FetchOnly { fetch };
        self.step_with_memory(state, &mut memory)
    }

    pub fn step_with_memory<M>(&self, state: &mut CpuState, memory: &mut M) -> Result<StepResult>
    where
        M: MemoryInterface,
    {
        let start_pc = state.pc;

        // A null PC usually indicates an unresolved branch target in guest code.
        // Treat it as a decode trap to surface the real failure mode.
        if start_pc == 0 {
            state.set_decode_fault();
            return Ok(StepResult::Trap {
                pc: 0,
                reason: ExitReason::DecodeFault,
            });
        }

        let cached_block = {
            let cache = self
                .block_cache
                .read()
                .expect("dynarec block cache poisoned");
            cache.get(&start_pc).cloned()
        };

        let cached_block = if let Some(block) = cached_block {
            block
        } else {
            let decoded_block =
                match decode_guest_block(start_pc, self.options.block_max_instructions, memory) {
                    Ok(block) => block,
                    Err(err) => {
                        mark_decode_error(state, err);
                        return Ok(StepResult::Trap {
                            pc: start_pc,
                            reason: state.exit_reason(),
                        });
                    }
                };

            let cache_entry = if self
                .force_interpreter
                .load(std::sync::atomic::Ordering::Relaxed)
                || decoded_block
                    .instructions
                    .iter()
                    .any(DecodedInstruction::requires_interpreter)
            {
                Arc::new(CachedBlock::Interpreted(decoded_block))
            } else {
                let eager_direct_targets = self.eager_direct_link_hints(&decoded_block);
                let initial_strategy = preferred_memory_strategy(memory);
                let compiled = compile_with_thread_local(
                    &self.arena,
                    &decoded_block,
                    eager_direct_targets,
                    initial_strategy,
                )?;
                Self::initialize_eager_return_links(&compiled, eager_direct_targets);
                Arc::new(CachedBlock::Compiled(compiled))
            };

            let mut cache = self
                .block_cache
                .write()
                .expect("dynarec block cache poisoned");
            cache
                .entry(start_pc)
                .or_insert_with(|| cache_entry.clone())
                .clone()
        };

        state.clear_exit();
        if let Some(region) = memory.fastmem_region() {
            state.set_fastmem_region(region.guest_base, region.host_ptr, region.size);
        } else {
            state.clear_fastmem_region();
        }
        if let Some(region) = memory.pte_region() {
            state.set_pte_region(
                region.guest_base,
                region.page_count,
                region.pte_ptr.cast::<u64>(),
            );
        } else {
            state.clear_pte_region();
        }

        match cached_block.as_ref() {
            CachedBlock::Compiled(block) => {
                let _binding = ThreadMemoryBinding::bind(memory);
                set_signal_fault_record(None);
                let (host_pc_map_ptr, host_pc_map_len) = if block.host_pc_map.is_empty() {
                    (
                        std::ptr::NonNull::<HostPcMapEntry>::dangling().as_ptr()
                            as *const HostPcMapEntry,
                        0usize,
                    )
                } else {
                    (block.host_pc_map.as_ptr(), block.host_pc_map.len())
                };
                let mut execution_state = state.clone();
                let mut late_link_miss: Option<(u64, u64)> = None;
                let mut fell_back_to_interpreter = false;

                let Some(()) = with_thread_execution_context_page(|execution_page| {
                    let execution_context_ptr = execution_page.execution_context_ptr();
                    let execution_context = execution_page.alias_context_mut();
                    execution_context.context = execution_state.clone();
                    execution_context.return_address = 0;
                    execution_context.late_link_source_pc = u64::MAX;
                    execution_context.late_link_next_pc = 0;

                    let context_state_ptr = std::ptr::addr_of_mut!(execution_context.context);
                    let mut dispatch_frame = JitDispatchFrame::new(
                        context_state_ptr,
                        start_pc,
                        block.code_start,
                        block.code_size,
                        host_pc_map_ptr,
                        host_pc_map_len,
                    );
                    let _host_stack_scope = HostStackScope::enter(&mut dispatch_frame);
                    let jit_stack_top = with_thread_jit_stack(|stack| stack.top())
                        .expect("jit stack should be available");
                    let next_pc = unsafe {
                        execute_compiled_block_with_dispatcher_landing(
                            block.entry,
                            execution_context_ptr,
                            execution_context.context.fastmem_guest_base,
                            jit_stack_top,
                            &mut dispatch_frame,
                        )
                    };

                    if !matches!(
                        execution_context.context.exit_reason(),
                        ExitReason::DispatcherInterrupt
                    ) {
                        execution_context.context.pc = next_pc;
                    }

                    if execution_context.late_link_source_pc != u64::MAX {
                        late_link_miss = Some((
                            execution_context.late_link_source_pc,
                            execution_context.late_link_next_pc,
                        ));
                    }

                    execution_state = execution_context.context.clone();
                }) else {
                    state.set_memory_fault();
                    return Ok(StepResult::Trap {
                        pc: start_pc,
                        reason: ExitReason::MemoryFault,
                    });
                };

                if let Some(record) = take_signal_fault_record() {
                    execution_state.pc = execution_state.pc.max(start_pc);
                    let fastmem_region = memory.fastmem_region();
                    match classify_bound_signal_fault(memory, fastmem_region, record) {
                        FaultClass::Polling => execution_state.set_dispatcher_interrupt(),
                        FaultClass::GranularityMismatch => execution_state.set_granularity_fault(),
                        FaultClass::MemoryFault | FaultClass::HostFault => {
                            execution_state.set_memory_fault()
                        }
                    }
                }

                if matches!(execution_state.exit_reason(), ExitReason::GranularityFault)
                    && block.strategy == MemoryStrategy::Fastmem
                {
                    let fallback_pc = execution_state.pc;
                    let fallback_strategy = downgraded_memory_strategy(memory, block.strategy);
                    self.block_cache
                        .write()
                        .expect("dynarec block cache poisoned")
                        .remove(&start_pc);
                    let decoded_block = match decode_guest_block(
                        fallback_pc,
                        self.options.block_max_instructions,
                        memory,
                    ) {
                        Ok(block) => block,
                        Err(err) => {
                            mark_decode_error(&mut execution_state, err);
                            let reason = execution_state.exit_reason();
                            *state = execution_state;
                            return Ok(StepResult::Trap {
                                pc: fallback_pc,
                                reason,
                            });
                        }
                    };
                    let eager_direct_targets = self.eager_direct_link_hints(&decoded_block);
                    let recompiled = compile_with_thread_local(
                        &self.arena,
                        &decoded_block,
                        eager_direct_targets,
                        fallback_strategy,
                    )?;
                    Self::initialize_eager_return_links(&recompiled, eager_direct_targets);
                    self.block_cache
                        .write()
                        .expect("dynarec block cache poisoned")
                        .insert(fallback_pc, Arc::new(CachedBlock::Compiled(recompiled)));
                } else if matches!(execution_state.exit_reason(), ExitReason::MemoryFault) {
                    fell_back_to_interpreter = true;
                    let fallback_pc = execution_state.pc;

                    self.block_cache
                        .write()
                        .expect("dynarec block cache poisoned")
                        .remove(&start_pc);

                    execution_state.clear_exit();
                    execution_state.pc = fallback_pc;

                    let decoded_block = match decode_guest_block(
                        fallback_pc,
                        self.options.block_max_instructions,
                        memory,
                    ) {
                        Ok(block) => block,
                        Err(err) => {
                            mark_decode_error(&mut execution_state, err);
                            *state = execution_state;
                            return Ok(StepResult::Trap {
                                pc: fallback_pc,
                                reason: state.exit_reason(),
                            });
                        }
                    };

                    self.block_cache
                        .write()
                        .expect("dynarec block cache poisoned")
                        .insert(
                            fallback_pc,
                            Arc::new(CachedBlock::Interpreted(decoded_block.clone())),
                        );

                    execute_interpreted_block(&mut execution_state, &decoded_block, memory);
                }

                if !fell_back_to_interpreter
                    && matches!(execution_state.exit_reason(), ExitReason::Running)
                {
                    if let Some((source_pc, next_pc)) = late_link_miss {
                        self.try_patch_late_link_edge(source_pc, next_pc, memory);
                    } else {
                        self.try_patch_late_link_edge(start_pc, execution_state.pc, memory);
                    }
                }

                *state = execution_state;
            }
            CachedBlock::Interpreted(block) => {
                execute_interpreted_block(state, block, memory);
            }
        }

        Ok(resolve_post_step(state, start_pc))
    }
}

fn resolve_post_step(state: &mut CpuState, start_pc: u64) -> StepResult {
    match state.exit_reason() {
        ExitReason::Running => StepResult::Advanced {
            start_pc,
            end_pc: state.pc,
        },
        ExitReason::DispatcherInterrupt => {
            state.clear_exit();
            StepResult::Advanced {
                start_pc,
                end_pc: state.pc,
            }
        }
        ExitReason::ThreadExit => StepResult::Trap {
            pc: state.pc,
            reason: ExitReason::ThreadExit,
        },
        ExitReason::SupervisorCall(imm) => StepResult::Trap {
            pc: state.pc,
            reason: ExitReason::SupervisorCall(imm),
        },
        ExitReason::UnknownSupervisorCall(imm) => StepResult::Trap {
            pc: state.pc,
            reason: ExitReason::UnknownSupervisorCall(imm),
        },
        ExitReason::HypervisorCall(_) => {
            state.clear_exit();
            if state.take_exception(ExceptionKind::Hvc).is_err() {
                state.set_privilege_fault();
                return StepResult::Trap {
                    pc: state.pc,
                    reason: state.exit_reason(),
                };
            }

            StepResult::Advanced {
                start_pc,
                end_pc: state.pc,
            }
        }
        ExitReason::SecureMonitorCall(_) => {
            state.clear_exit();
            if state.take_exception(ExceptionKind::Smc).is_err() {
                state.set_privilege_fault();
                return StepResult::Trap {
                    pc: state.pc,
                    reason: state.exit_reason(),
                };
            }

            StepResult::Advanced {
                start_pc,
                end_pc: state.pc,
            }
        }
        ExitReason::ExceptionReturn => {
            state.clear_exit();
            if state.exception_return().is_err() {
                state.set_privilege_fault();
                return StepResult::Trap {
                    pc: state.pc,
                    reason: state.exit_reason(),
                };
            }

            StepResult::Advanced {
                start_pc,
                end_pc: state.pc,
            }
        }
        reason => StepResult::Trap {
            pc: state.pc,
            reason,
        },
    }
}

fn execute_interpreted_block<M>(state: &mut CpuState, block: &DecodedBlock, memory: &mut M)
where
    M: MemoryInterface,
{
    let mut pc = block.start_pc;

    for instruction in &block.instructions {
        match instruction {
            DecodedInstruction::Nop
            | DecodedInstruction::Yield
            | DecodedInstruction::Wfe
            | DecodedInstruction::Wfi
            | DecodedInstruction::Sev
            | DecodedInstruction::Sevl
            | DecodedInstruction::Xpaclri
            | DecodedInstruction::Bti { .. } => {
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovZ { rd, imm16, shift } => {
                state.write_x(*rd, (u64::from(*imm16)) << *shift);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovZ32 { rd, imm16, shift } => {
                let value = (u32::from(*imm16)) << *shift;
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovN { rd, imm16, shift } => {
                state.write_x(*rd, !((u64::from(*imm16)) << *shift));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovN32 { rd, imm16, shift } => {
                let value = !((u32::from(*imm16)) << *shift);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovK { rd, imm16, shift } => {
                let original = state.read_x(*rd);
                let clear_mask = !((0xffffu64) << *shift);
                let merged = (original & clear_mask) | ((u64::from(*imm16)) << *shift);
                state.write_x(*rd, merged);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovK32 { rd, imm16, shift } => {
                let original = state.read_x(*rd) as u32;
                let clear_mask = !((0xffffu32) << *shift);
                let merged = (original & clear_mask) | ((u32::from(*imm16)) << *shift);
                state.write_x(*rd, u64::from(merged));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndImm { rd, rn, imm } => {
                let result = state.read_x(*rn) & *imm;
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndImm32 { rd, rn, imm } => {
                let result = (state.read_x(*rn) as u32) & *imm;
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrImm { rd, rn, imm } => {
                let result = state.read_x(*rn) | *imm;
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrImm32 { rd, rn, imm } => {
                let result = (state.read_x(*rn) as u32) | *imm;
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EorImm { rd, rn, imm } => {
                let result = state.read_x(*rn) ^ *imm;
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EorImm32 { rd, rn, imm } => {
                let result = (state.read_x(*rn) as u32) ^ *imm;
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndsImm { rd, rn, imm } => {
                let result = state.read_x(*rn) & *imm;
                state.write_x(*rd, result);
                state.nzcv = (((result >> 63) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndsImm32 { rd, rn, imm } => {
                let result = (state.read_x(*rn) as u32) & *imm;
                state.write_x(*rd, u64::from(result));
                state.nzcv = (((result >> 31) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LslImm { rd, rn, amount } => {
                state.write_x(*rd, state.read_x(*rn).wrapping_shl(u32::from(*amount)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LslImm32 { rd, rn, amount } => {
                let value = (state.read_x(*rn) as u32).wrapping_shl(u32::from(*amount));
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LsrImm { rd, rn, amount } => {
                state.write_x(*rd, state.read_x(*rn).wrapping_shr(u32::from(*amount)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LsrImm32 { rd, rn, amount } => {
                let value = (state.read_x(*rn) as u32).wrapping_shr(u32::from(*amount));
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AsrImm { rd, rn, amount } => {
                let value = ((state.read_x(*rn) as i64) >> amount) as u64;
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AsrImm32 { rd, rn, amount } => {
                let value = ((state.read_x(*rn) as i32) >> amount) as u32;
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Sbfm { rd, rn, immr, imms } => {
                let value = sbfm64(state.read_x(*rn), *immr, *imms);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Sbfm32 { rd, rn, immr, imms } => {
                let value = sbfm32(state.read_x(*rn) as u32, *immr, *imms);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Ubfm { rd, rn, immr, imms } => {
                let value = ubfm64(state.read_x(*rn), *immr, *imms);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Ubfm32 { rd, rn, immr, imms } => {
                let value = ubfm32(state.read_x(*rn) as u32, *immr, *imms);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Bfm { rd, rn, immr, imms } => {
                let dst = state.read_x(*rd);
                let src = state.read_x(*rn);
                let value = bfm64(dst, src, *immr, *imms);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Bfm32 { rd, rn, immr, imms } => {
                let dst = state.read_x(*rd) as u32;
                let src = state.read_x(*rn) as u32;
                let value = bfm32(dst, src, *immr, *imms);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddImm {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = read_gp_or_sp(state, *rn);
                let rhs = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                write_gp_or_sp(state, *rd, lhs.wrapping_add(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddImm32 {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = read_w_or_wsp(state, *rn);
                let rhs = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                write_w_or_wsp(state, *rd, lhs.wrapping_add(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubImm {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = read_gp_or_sp(state, *rn);
                let rhs = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                write_gp_or_sp(state, *rd, lhs.wrapping_sub(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubImm32 {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = read_w_or_wsp(state, *rn);
                let rhs = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                write_w_or_wsp(state, *rd, lhs.wrapping_sub(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsImm {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_add(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsImm32 {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_add32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsImm {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_sub(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsImm32 {
                rd,
                rn,
                imm12,
                shift12,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_sub32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs.wrapping_add(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs.wrapping_add(rhs)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs.wrapping_sub(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs.wrapping_sub(rhs)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_add(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_add32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_sub(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_sub32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddExt {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = read_gp_or_sp(state, *rn);
                let rhs = apply_extend(state.read_x(*rm), *extend, *amount);
                write_gp_or_sp(state, *rd, lhs.wrapping_add(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddExt32 {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = read_w_or_wsp(state, *rn);
                let rhs = apply_extend32(state.read_x(*rm), *extend, *amount);
                write_w_or_wsp(state, *rd, lhs.wrapping_add(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubExt {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = read_gp_or_sp(state, *rn);
                let rhs = apply_extend(state.read_x(*rm), *extend, *amount);
                write_gp_or_sp(state, *rd, lhs.wrapping_sub(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubExt32 {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = read_w_or_wsp(state, *rn);
                let rhs = apply_extend32(state.read_x(*rm), *extend, *amount);
                write_w_or_wsp(state, *rd, lhs.wrapping_sub(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsExt {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_extend(state.read_x(*rm), *extend, *amount);
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_add(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddsExt32 {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_extend32(state.read_x(*rm), *extend, *amount);
                let result = lhs.wrapping_add(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_add32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsExt {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_extend(state.read_x(*rm), *extend, *amount);
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, result);
                state.update_nzcv_sub(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SubsExt32 {
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_extend32(state.read_x(*rm), *extend, *amount);
                let result = lhs.wrapping_sub(rhs);
                state.write_x(*rd, u64::from(result));
                state.update_nzcv_sub32(lhs, rhs, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs & rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs & rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BicReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs & !rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BicReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs & !rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs | rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs | rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrnReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs | !rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrnReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs | !rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EorReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs ^ rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EorReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs ^ rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EonReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                state.write_x(*rd, lhs ^ !rhs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::EonReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                state.write_x(*rd, u64::from(lhs ^ !rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndsReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                let result = lhs & rhs;
                state.write_x(*rd, result);
                state.nzcv = (((result >> 63) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndsReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                let result = lhs & rhs;
                state.write_x(*rd, u64::from(result));
                state.nzcv = (((result >> 31) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BicsReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn);
                let rhs = apply_shift(state.read_x(*rm), *shift, *amount);
                let result = lhs & !rhs;
                state.write_x(*rd, result);
                state.nzcv = (((result >> 63) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BicsReg32 {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = apply_shift32(state.read_x(*rm) as u32, *shift, *amount);
                let result = lhs & !rhs;
                state.write_x(*rd, u64::from(result));
                state.nzcv = (((result >> 31) as u32) << 31) | ((result == 0) as u32) << 30;
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csel {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn)
                } else {
                    state.read_x(*rm)
                };
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csel32 {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn) as u32
                } else {
                    state.read_x(*rm) as u32
                };
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csinc {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn)
                } else {
                    state.read_x(*rm).wrapping_add(1)
                };
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csinc32 {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn) as u32
                } else {
                    (state.read_x(*rm) as u32).wrapping_add(1)
                };
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csinv {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn)
                } else {
                    !state.read_x(*rm)
                };
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csinv32 {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn) as u32
                } else {
                    !(state.read_x(*rm) as u32)
                };
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csneg {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn)
                } else {
                    state.read_x(*rm).wrapping_neg()
                };
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Csneg32 {
                rd,
                rn,
                rm,
                condition,
            } => {
                let value = if state.condition_holds(*condition) {
                    state.read_x(*rn) as u32
                } else {
                    (state.read_x(*rm) as u32).wrapping_neg()
                };
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmpReg {
                rn,
                rm,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn);
                    let rhs = state.read_x(*rm);
                    let result = lhs.wrapping_sub(rhs);
                    state.update_nzcv_sub(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmpReg32 {
                rn,
                rm,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn) as u32;
                    let rhs = state.read_x(*rm) as u32;
                    let result = lhs.wrapping_sub(rhs);
                    state.update_nzcv_sub32(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmpImm {
                rn,
                imm5,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn);
                    let rhs = u64::from(*imm5);
                    let result = lhs.wrapping_sub(rhs);
                    state.update_nzcv_sub(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmpImm32 {
                rn,
                imm5,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn) as u32;
                    let rhs = u32::from(*imm5);
                    let result = lhs.wrapping_sub(rhs);
                    state.update_nzcv_sub32(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmnReg {
                rn,
                rm,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn);
                    let rhs = state.read_x(*rm);
                    let result = lhs.wrapping_add(rhs);
                    state.update_nzcv_add(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmnReg32 {
                rn,
                rm,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn) as u32;
                    let rhs = state.read_x(*rm) as u32;
                    let result = lhs.wrapping_add(rhs);
                    state.update_nzcv_add32(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmnImm {
                rn,
                imm5,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn);
                    let rhs = u64::from(*imm5);
                    let result = lhs.wrapping_add(rhs);
                    state.update_nzcv_add(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CcmnImm32 {
                rn,
                imm5,
                nzcv,
                condition,
            } => {
                if state.condition_holds(*condition) {
                    let lhs = state.read_x(*rn) as u32;
                    let rhs = u32::from(*imm5);
                    let result = lhs.wrapping_add(rhs);
                    state.update_nzcv_add32(lhs, rhs, result);
                } else {
                    state.nzcv = u32::from(*nzcv & 0x0f) << 28;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Lslv { rd, rn, rm } => {
                let value = state
                    .read_x(*rn)
                    .wrapping_shl((state.read_x(*rm) & 63) as u32);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Lslv32 { rd, rn, rm } => {
                let amount = (state.read_x(*rm) & 31) as u32;
                let value = (state.read_x(*rn) as u32).wrapping_shl(amount);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Lsrv { rd, rn, rm } => {
                let value = state
                    .read_x(*rn)
                    .wrapping_shr((state.read_x(*rm) & 63) as u32);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Lsrv32 { rd, rn, rm } => {
                let amount = (state.read_x(*rm) & 31) as u32;
                let value = (state.read_x(*rn) as u32).wrapping_shr(amount);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Asrv { rd, rn, rm } => {
                let shift = (state.read_x(*rm) & 63) as u32;
                let value = ((state.read_x(*rn) as i64) >> shift) as u64;
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Asrv32 { rd, rn, rm } => {
                let shift = (state.read_x(*rm) & 31) as u32;
                let value = ((state.read_x(*rn) as i32) >> shift) as u32;
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rorv { rd, rn, rm } => {
                let shift = (state.read_x(*rm) & 63) as u32;
                let value = state.read_x(*rn).rotate_right(shift);
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rorv32 { rd, rn, rm } => {
                let shift = (state.read_x(*rm) & 31) as u32;
                let value = (state.read_x(*rn) as u32).rotate_right(shift);
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Extr { rd, rn, rm, lsb } => {
                let lsb = u32::from(*lsb & 63);
                let low = state.read_x(*rm);
                let high = state.read_x(*rn);
                let value = if lsb == 0 {
                    low
                } else {
                    (low >> lsb) | (high << (64 - lsb))
                };
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Extr32 { rd, rn, rm, lsb } => {
                let lsb = u32::from(*lsb & 31);
                let low = state.read_x(*rm) as u32;
                let high = state.read_x(*rn) as u32;
                let value = if lsb == 0 {
                    low
                } else {
                    (low >> lsb) | (high << (32 - lsb))
                };
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rbit { rd, rn } => {
                state.write_x(*rd, state.read_x(*rn).reverse_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rbit32 { rd, rn } => {
                let value = (state.read_x(*rn) as u32).reverse_bits();
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Clz { rd, rn } => {
                let value = state.read_x(*rn).leading_zeros();
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Clz32 { rd, rn } => {
                let value = (state.read_x(*rn) as u32).leading_zeros();
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rev { rd, rn } => {
                let value = state.read_x(*rn).swap_bytes();
                state.write_x(*rd, value);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rev32 { rd, rn } => {
                let value = (state.read_x(*rn) as u32).swap_bytes();
                state.write_x(*rd, u64::from(value));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Mul { rd, rn, rm } => {
                let lhs = state.read_x(*rn);
                let rhs = state.read_x(*rm);
                state.write_x(*rd, lhs.wrapping_mul(rhs));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Mul32 { rd, rn, rm } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = state.read_x(*rm) as u32;
                state.write_x(*rd, u64::from(lhs.wrapping_mul(rhs)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Madd { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn);
                let rhs = state.read_x(*rm);
                let addend = state.read_x(*ra);
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, product.wrapping_add(addend));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Madd32 { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = state.read_x(*rm) as u32;
                let addend = state.read_x(*ra) as u32;
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, u64::from(product.wrapping_add(addend)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Msub { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn);
                let rhs = state.read_x(*rm);
                let minuend = state.read_x(*ra);
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, minuend.wrapping_sub(product));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Msub32 { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = state.read_x(*rm) as u32;
                let minuend = state.read_x(*ra) as u32;
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, u64::from(minuend.wrapping_sub(product)));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Smaddl { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn) as u32 as i32 as i64;
                let rhs = state.read_x(*rm) as u32 as i32 as i64;
                let addend = state.read_x(*ra) as i64;
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, product.wrapping_add(addend) as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Smsubl { rd, rn, rm, ra } => {
                let lhs = state.read_x(*rn) as u32 as i32 as i64;
                let rhs = state.read_x(*rm) as u32 as i32 as i64;
                let minuend = state.read_x(*ra) as i64;
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, minuend.wrapping_sub(product) as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Umaddl { rd, rn, rm, ra } => {
                let lhs = u64::from(state.read_x(*rn) as u32);
                let rhs = u64::from(state.read_x(*rm) as u32);
                let addend = state.read_x(*ra);
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, product.wrapping_add(addend));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Umsubl { rd, rn, rm, ra } => {
                let lhs = u64::from(state.read_x(*rn) as u32);
                let rhs = u64::from(state.read_x(*rm) as u32);
                let minuend = state.read_x(*ra);
                let product = lhs.wrapping_mul(rhs);
                state.write_x(*rd, minuend.wrapping_sub(product));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Smulh { rd, rn, rm } => {
                let lhs = state.read_x(*rn) as i64 as i128;
                let rhs = state.read_x(*rm) as i64 as i128;
                let high = (lhs.wrapping_mul(rhs) >> 64) as i64;
                state.write_x(*rd, high as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Umulh { rd, rn, rm } => {
                let lhs = u128::from(state.read_x(*rn));
                let rhs = u128::from(state.read_x(*rm));
                let high = (lhs.wrapping_mul(rhs) >> 64) as u64;
                state.write_x(*rd, high);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Udiv { rd, rn, rm } => {
                let lhs = state.read_x(*rn);
                let rhs = state.read_x(*rm);
                let result = if rhs == 0 { 0 } else { lhs / rhs };
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Udiv32 { rd, rn, rm } => {
                let lhs = state.read_x(*rn) as u32;
                let rhs = state.read_x(*rm) as u32;
                let result = if rhs == 0 { 0 } else { lhs / rhs };
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Sdiv { rd, rn, rm } => {
                let lhs = state.read_x(*rn) as i64;
                let rhs = state.read_x(*rm) as i64;
                let result = if rhs == 0 {
                    0
                } else if lhs == i64::MIN && rhs == -1 {
                    i64::MIN
                } else {
                    lhs / rhs
                };
                state.write_x(*rd, result as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Sdiv32 { rd, rn, rm } => {
                let lhs = state.read_x(*rn) as i32;
                let rhs = state.read_x(*rm) as i32;
                let result = if rhs == 0 {
                    0
                } else if lhs == i32::MIN && rhs == -1 {
                    i32::MIN
                } else {
                    lhs / rhs
                };
                state.write_x(*rd, u64::from(result as u32));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LdxrU64 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        state.set_exclusive_reservation(address, 8);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdxrU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        state.set_exclusive_reservation(address, 4);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdaxrU64 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        state.set_exclusive_reservation(address, 8);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdaxrU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        state.set_exclusive_reservation(address, 4);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdxpU64 { rt, rt2, rn }
            | DecodedInstruction::LdaxpU64 { rt, rt2, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let second = address.wrapping_add(8);

                match (memory.read_u64(address), memory.read_u64(second)) {
                    (Some(v0), Some(v1)) => {
                        state.write_x(*rt, v0);
                        state.write_x(*rt2, v1);
                        state.set_exclusive_reservation(address, 16);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdxpU32 { rt, rt2, rn }
            | DecodedInstruction::LdaxpU32 { rt, rt2, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let second = address.wrapping_add(4);

                match (memory.read_u32(address), memory.read_u32(second)) {
                    (Some(v0), Some(v1)) => {
                        state.write_x(*rt, u64::from(v0));
                        state.write_x(*rt2, u64::from(v1));
                        state.set_exclusive_reservation(address, 8);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdarU64 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdarU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdarbU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdarhU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StxrU64 { rs, rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let mut status = 1u64;

                if state.check_exclusive_reservation(address, 8) {
                    let value = state.read_x(*rt);
                    if memory.write_u64(address, value) {
                        status = 0;
                    } else {
                        state.set_memory_fault();
                        state.pc = pc;
                        state.clear_exclusive_reservation();
                        return;
                    }
                }

                state.clear_exclusive_reservation();
                state.write_x(*rs, status);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::StxpU64 { rs, rt, rt2, rn }
            | DecodedInstruction::StlxpU64 { rs, rt, rt2, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let mut status = 1u64;

                if state.check_exclusive_reservation(address, 16) {
                    let v0 = state.read_x(*rt);
                    let v1 = state.read_x(*rt2);
                    let ok_first = memory.write_u64(address, v0);
                    let ok_second = memory.write_u64(address.wrapping_add(8), v1);
                    if ok_first && ok_second {
                        status = 0;
                    } else {
                        state.set_memory_fault();
                        state.pc = pc;
                        state.clear_exclusive_reservation();
                        return;
                    }
                }

                state.clear_exclusive_reservation();
                state.write_x(*rs, status);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::StxrU32 { rs, rt, rn }
            | DecodedInstruction::StlxrU32 { rs, rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let mut status = 1u64;

                if state.check_exclusive_reservation(address, 4) {
                    let value = state.read_x(*rt) as u32;
                    if memory.write_u32(address, value) {
                        status = 0;
                    } else {
                        state.set_memory_fault();
                        state.pc = pc;
                        state.clear_exclusive_reservation();
                        return;
                    }
                }

                state.clear_exclusive_reservation();
                state.write_x(*rs, status);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::StlxrU64 { rs, rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let mut status = 1u64;

                if state.check_exclusive_reservation(address, 8) {
                    let value = state.read_x(*rt);
                    if memory.write_u64(address, value) {
                        status = 0;
                    } else {
                        state.set_memory_fault();
                        state.pc = pc;
                        state.clear_exclusive_reservation();
                        return;
                    }
                }

                state.clear_exclusive_reservation();
                state.write_x(*rs, status);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::StxpU32 { rs, rt, rt2, rn }
            | DecodedInstruction::StlxpU32 { rs, rt, rt2, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let mut status = 1u64;

                if state.check_exclusive_reservation(address, 8) {
                    let v0 = state.read_x(*rt) as u32;
                    let v1 = state.read_x(*rt2) as u32;
                    let ok_first = memory.write_u32(address, v0);
                    let ok_second = memory.write_u32(address.wrapping_add(4), v1);
                    if ok_first && ok_second {
                        status = 0;
                    } else {
                        state.set_memory_fault();
                        state.pc = pc;
                        state.clear_exclusive_reservation();
                        return;
                    }
                }

                state.clear_exclusive_reservation();
                state.write_x(*rs, status);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::StlrU64 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let value = state.read_x(*rt);
                if memory.write_u64(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StlrU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let value = state.read_x(*rt) as u32;
                if memory.write_u32(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StlrbU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let value = state.read_x(*rt) as u8;
                if memory.write_u8(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StlrhU32 { rt, rn } => {
                let address = read_gp_or_sp(state, *rn);
                let value = state.read_x(*rt) as u16;
                if memory.write_u16(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrImmU32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 2;
                let address = base.wrapping_add(offset);

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrImmU32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 2;
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u32;

                if memory.write_u32(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrImmU16 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 1;
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrImmU16 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 1;
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u16;

                if memory.write_u16(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrImmU8 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12);
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrImmU8 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12);
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u8;

                if memory.write_u8(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrsbImm64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12);
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i8) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbImm32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12);
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i8) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshImm64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 1;
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i16) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshImm32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 1;
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i16) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrswImm64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 2;
                let address = base.wrapping_add(offset);

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i32) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrRegU64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 3) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrRegU64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 3) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt);

                if memory.write_u64(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrRegU32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 2) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrRegU32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 2) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u32;

                if memory.write_u32(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrRegU16 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 1) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrRegU16 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 1) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u16;

                if memory.write_u16(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrRegU8 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 0) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrRegU8 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 0) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt) as u8;

                if memory.write_u8(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrsbReg64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 0) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i8) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbReg32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 0) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i8) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshReg64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 1) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i16) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshReg32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 1) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i16) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrswReg64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 2) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i32) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpReg128 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 4) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let lo = memory.read_u64(address);
                let hi = memory.read_u64(address.wrapping_add(8));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        state.write_v128(*rt, lo, hi);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpReg32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 2) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpReg64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 3) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_v128(*rt, value, 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrFpReg128 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 4) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let (lo, hi) = state.read_v128(*rt);

                let ok_lo = memory.write_u64(address, lo);
                let ok_hi = memory.write_u64(address.wrapping_add(8), hi);
                if ok_lo && ok_hi {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpReg32 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 2) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u32(address, lo as u32) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpReg64 {
                rt,
                rn,
                rm,
                option,
                shift,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let Some(offset) = resolve_ls_reg_offset(state, *rm, *option, *shift, 3) else {
                    state.set_decode_fault();
                    state.pc = pc.wrapping_add(4);
                    return;
                };
                let address = base.wrapping_add(offset);
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u64(address, lo) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdurI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::SturI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt);

                if memory.write_u64(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPostI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        let updated = add_signed_offset(base, i64::from(*imm9));
                        write_gp_or_sp(state, *rn, updated);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrPreI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        write_gp_or_sp(state, *rn, address);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrPostI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;
                let value = state.read_x(*rt);

                if memory.write_u64(address, value) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt);

                if memory.write_u64(address, value) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdurI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::SturI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u32;

                if memory.write_u32(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPostI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        let updated = add_signed_offset(base, i64::from(*imm9));
                        write_gp_or_sp(state, *rn, updated);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrPreI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        write_gp_or_sp(state, *rn, address);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrPostI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;
                let value = state.read_x(*rt) as u32;

                if memory.write_u32(address, value) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u32;

                if memory.write_u32(address, value) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPostI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        let updated = add_signed_offset(base, i64::from(*imm9));
                        write_gp_or_sp(state, *rn, updated);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrPreI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        write_gp_or_sp(state, *rn, address);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrPostI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;
                let value = state.read_x(*rt) as u8;

                if memory.write_u8(address, value) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u8;

                if memory.write_u8(address, value) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPostI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        let updated = add_signed_offset(base, i64::from(*imm9));
                        write_gp_or_sp(state, *rn, updated);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrPreI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        write_gp_or_sp(state, *rn, address);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrPostI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base;
                let value = state.read_x(*rt) as u16;

                if memory.write_u16(address, value) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u16;

                if memory.write_u16(address, value) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdurI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::SturI16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u16;

                if memory.write_u16(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdurI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from(value));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::SturI8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let value = state.read_x(*rt) as u8;

                if memory.write_u8(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdursbI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i8) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdursbI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i8) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurshI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i16) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurshI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_x(*rt, u64::from((value as i16) as i32 as u32));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurswI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_x(*rt, (value as i32) as i64 as u64);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            // --- Sign-extending post/pre-index loads ---
            DecodedInstruction::LdrshPostI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                match memory.read_u16(base) {
                    Some(v) => {
                        state.write_x(*rt, u64::from((v as i16) as i32 as u32));
                        write_gp_or_sp(state, *rn, add_signed_offset(base, i64::from(*imm9)));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshPostI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                match memory.read_u16(base) {
                    Some(v) => {
                        state.write_x(*rt, (v as i16) as i64 as u64);
                        write_gp_or_sp(state, *rn, add_signed_offset(base, i64::from(*imm9)));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshPreI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let addr = add_signed_offset(base, i64::from(*imm9));
                match memory.read_u16(addr) {
                    Some(v) => {
                        state.write_x(*rt, u64::from((v as i16) as i32 as u32));
                        write_gp_or_sp(state, *rn, addr);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrshPreI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let addr = add_signed_offset(base, i64::from(*imm9));
                match memory.read_u16(addr) {
                    Some(v) => {
                        state.write_x(*rt, (v as i16) as i64 as u64);
                        write_gp_or_sp(state, *rn, addr);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbPostI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                match memory.read_u8(base) {
                    Some(v) => {
                        state.write_x(*rt, u64::from((v as i8) as i32 as u32));
                        write_gp_or_sp(state, *rn, add_signed_offset(base, i64::from(*imm9)));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbPostI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                match memory.read_u8(base) {
                    Some(v) => {
                        state.write_x(*rt, (v as i8) as i64 as u64);
                        write_gp_or_sp(state, *rn, add_signed_offset(base, i64::from(*imm9)));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbPreI32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let addr = add_signed_offset(base, i64::from(*imm9));
                match memory.read_u8(addr) {
                    Some(v) => {
                        state.write_x(*rt, u64::from((v as i8) as i32 as u32));
                        write_gp_or_sp(state, *rn, addr);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrsbPreI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let addr = add_signed_offset(base, i64::from(*imm9));
                match memory.read_u8(addr) {
                    Some(v) => {
                        state.write_x(*rt, (v as i8) as i64 as u64);
                        write_gp_or_sp(state, *rn, addr);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrswPostI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                match memory.read_u32(base) {
                    Some(v) => {
                        state.write_x(*rt, (v as i32) as i64 as u64);
                        write_gp_or_sp(state, *rn, add_signed_offset(base, i64::from(*imm9)));
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrswPreI64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let addr = add_signed_offset(base, i64::from(*imm9));
                match memory.read_u32(addr) {
                    Some(v) => {
                        state.write_x(*rt, (v as i32) as i64 as u64);
                        write_gp_or_sp(state, *rn, addr);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdpI64 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 8;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first = memory.read_u64(address);
                let second = memory.read_u64(address.wrapping_add(8));
                match (first, second) {
                    (Some(v0), Some(v1)) => {
                        state.write_x(*rt, v0);
                        state.write_x(*rt2, v1);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StpI64 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 8;
                let (address, writeback) = pair_address(base, offset, *mode);
                let v0 = state.read_x(*rt);
                let v1 = state.read_x(*rt2);

                let ok_first = memory.write_u64(address, v0);
                let ok_second = memory.write_u64(address.wrapping_add(8), v1);
                if ok_first && ok_second {
                    if let Some(new_base) = writeback {
                        write_gp_or_sp(state, *rn, new_base);
                    }
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdpI32 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 4;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first = memory.read_u32(address);
                let second = memory.read_u32(address.wrapping_add(4));
                match (first, second) {
                    (Some(v0), Some(v1)) => {
                        state.write_x(*rt, u64::from(v0));
                        state.write_x(*rt2, u64::from(v1));
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::Ldpsw {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 4;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first = memory.read_u32(address);
                let second = memory.read_u32(address.wrapping_add(4));
                match (first, second) {
                    (Some(v0), Some(v1)) => {
                        state.write_x(*rt, (v0 as i32) as i64 as u64);
                        state.write_x(*rt2, (v1 as i32) as i64 as u64);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StpI32 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 4;
                let (address, writeback) = pair_address(base, offset, *mode);
                let v0 = state.read_x(*rt) as u32;
                let v1 = state.read_x(*rt2) as u32;

                let ok_first = memory.write_u32(address, v0);
                let ok_second = memory.write_u32(address.wrapping_add(4), v1);
                if ok_first && ok_second {
                    if let Some(new_base) = writeback {
                        write_gp_or_sp(state, *rn, new_base);
                    }
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdpFp32 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 4;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first = memory.read_u32(address);
                let second = memory.read_u32(address.wrapping_add(4));
                match (first, second) {
                    (Some(v0), Some(v1)) => {
                        state.write_v128(*rt, u64::from(v0), 0);
                        state.write_v128(*rt2, u64::from(v1), 0);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StpFp32 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 4;
                let (address, writeback) = pair_address(base, offset, *mode);
                let (v0_lo, _) = state.read_v128(*rt);
                let (v1_lo, _) = state.read_v128(*rt2);

                let ok_first = memory.write_u32(address, v0_lo as u32);
                let ok_second = memory.write_u32(address.wrapping_add(4), v1_lo as u32);
                if ok_first && ok_second {
                    if let Some(new_base) = writeback {
                        write_gp_or_sp(state, *rn, new_base);
                    }
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdpFp64 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 8;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first = memory.read_u64(address);
                let second = memory.read_u64(address.wrapping_add(8));
                match (first, second) {
                    (Some(v0), Some(v1)) => {
                        state.write_v128(*rt, v0, 0);
                        state.write_v128(*rt2, v1, 0);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StpFp64 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 8;
                let (address, writeback) = pair_address(base, offset, *mode);
                let (v0_lo, _) = state.read_v128(*rt);
                let (v1_lo, _) = state.read_v128(*rt2);

                let ok_first = memory.write_u64(address, v0_lo);
                let ok_second = memory.write_u64(address.wrapping_add(8), v1_lo);
                if ok_first && ok_second {
                    if let Some(new_base) = writeback {
                        write_gp_or_sp(state, *rn, new_base);
                    }
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdpV128 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 16;
                let (address, writeback) = pair_address(base, offset, *mode);

                let first_lo = memory.read_u64(address);
                let first_hi = memory.read_u64(address.wrapping_add(8));
                let second_lo = memory.read_u64(address.wrapping_add(16));
                let second_hi = memory.read_u64(address.wrapping_add(24));
                match (first_lo, first_hi, second_lo, second_hi) {
                    (Some(v0_lo), Some(v0_hi), Some(v1_lo), Some(v1_hi)) => {
                        state.write_v128(*rt, v0_lo, v0_hi);
                        state.write_v128(*rt2, v1_lo, v1_hi);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(state, *rn, new_base);
                        }
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StpV128 {
                rt,
                rt2,
                rn,
                imm7,
                mode,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = i64::from(*imm7) * 16;
                let (address, writeback) = pair_address(base, offset, *mode);
                let (v0_lo, v0_hi) = state.read_v128(*rt);
                let (v1_lo, v1_hi) = state.read_v128(*rt2);

                let ok0 = memory.write_u64(address, v0_lo);
                let ok1 = memory.write_u64(address.wrapping_add(8), v0_hi);
                let ok2 = memory.write_u64(address.wrapping_add(16), v1_lo);
                let ok3 = memory.write_u64(address.wrapping_add(24), v1_hi);
                if ok0 && ok1 && ok2 && ok3 {
                    if let Some(new_base) = writeback {
                        write_gp_or_sp(state, *rn, new_base);
                    }
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdurFp8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurFp16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_v128(*rt, value, 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdurFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let lo = memory.read_u64(address);
                let hi = memory.read_u64(address.wrapping_add(8));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        state.write_v128(*rt, lo, hi);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::SturFp8 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u8(address, lo as u8) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::SturFp16 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u16(address, lo as u16) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::SturFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u32(address, lo as u32) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::SturFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u64(address, lo) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::SturFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, hi) = state.read_v128(*rt);

                let ok_lo = memory.write_u64(address, lo);
                let ok_hi = memory.write_u64(address.wrapping_add(8), hi);
                if ok_lo && ok_hi {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPostFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let lo = memory.read_u64(base);
                let hi = memory.read_u64(base.wrapping_add(8));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        state.write_v128(*rt, lo, hi);
                        let updated = add_signed_offset(base, i64::from(*imm9));
                        write_gp_or_sp(state, *rn, updated);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrPreFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let lo = memory.read_u64(address);
                let hi = memory.read_u64(address.wrapping_add(8));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        state.write_v128(*rt, lo, hi);
                        write_gp_or_sp(state, *rn, address);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrPostFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let (lo, hi) = state.read_v128(*rt);
                let ok_lo = memory.write_u64(base, lo);
                let ok_hi = memory.write_u64(base.wrapping_add(8), hi);

                if ok_lo && ok_hi {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreFp128 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, hi) = state.read_v128(*rt);
                let ok_lo = memory.write_u64(address, lo);
                let ok_hi = memory.write_u64(address.wrapping_add(8), hi);

                if ok_lo && ok_hi {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            // 64-bit FP post/pre-indexed load/store
            DecodedInstruction::LdrPostFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                if let Some(val) = memory.read_u64(base) {
                    state.write_v128(*rt, val, 0);
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPreFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                if let Some(val) = memory.read_u64(address) {
                    state.write_v128(*rt, val, 0);
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPostFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u64(base, lo) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreFp64 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u64(address, lo) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            // 32-bit FP post/pre-indexed load/store
            DecodedInstruction::LdrPostFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                if let Some(val) = memory.read_u32(base) {
                    state.write_v128(*rt, val as u64, 0);
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrPreFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                if let Some(val) = memory.read_u32(address) {
                    state.write_v128(*rt, val as u64, 0);
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPostFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u32(base, lo as u32) {
                    let updated = add_signed_offset(base, i64::from(*imm9));
                    write_gp_or_sp(state, *rn, updated);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrPreFp32 { rt, rn, imm9 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = add_signed_offset(base, i64::from(*imm9));
                let (lo, _) = state.read_v128(*rt);
                if memory.write_u32(address, lo as u32) {
                    write_gp_or_sp(state, *rn, address);
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::SimdLdMulti {
                q,
                nregs,
                interleave,
                size,
                rn,
                vt,
                post_index,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                let total_bytes = bytes_per_reg * u64::from(*nregs);
                let elem_size: u64 = 1u64 << *size; // 1, 2, 4, or 8
                let elems_per_reg = bytes_per_reg / elem_size;

                // Read all bytes from memory
                let mut buf = vec![0u8; total_bytes as usize];
                let mut ok = true;
                for i in 0..total_bytes {
                    match memory.read_u8(base.wrapping_add(i)) {
                        Some(b) => buf[i as usize] = b,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }

                // Distribute to registers
                let n = *nregs as u64;
                for reg_idx in 0..n {
                    let vreg = (u64::from(*vt) + reg_idx) & 31;
                    let mut lo = 0u64;
                    let mut hi = 0u64;
                    for lane in 0..elems_per_reg {
                        let mem_idx = if *interleave {
                            // Deinterleave: element at position (lane * n + reg_idx)
                            (lane * n + reg_idx) * elem_size
                        } else {
                            // Consecutive: register reg_idx, lane
                            (reg_idx * elems_per_reg + lane) * elem_size
                        };
                        let mut elem = 0u64;
                        for b in 0..elem_size {
                            elem |= (buf[(mem_idx + b) as usize] as u64) << (b * 8);
                        }
                        let bit_offset = lane * elem_size * 8;
                        if bit_offset < 64 {
                            lo |= elem << bit_offset;
                        } else {
                            hi |= elem << (bit_offset - 64);
                        }
                    }
                    state.write_v128(vreg as u8, lo, hi);
                }

                // Post-index writeback
                if let Some(rm) = post_index {
                    let offset = if *rm == 31 {
                        total_bytes
                    } else {
                        state.read_x(*rm)
                    };
                    write_gp_or_sp(state, *rn, base.wrapping_add(offset));
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdStMulti {
                q,
                nregs,
                interleave,
                size,
                rn,
                vt,
                post_index,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                let total_bytes = bytes_per_reg * u64::from(*nregs);
                let elem_size: u64 = 1u64 << *size;
                let elems_per_reg = bytes_per_reg / elem_size;
                let n = *nregs as u64;

                // Build buffer from registers
                let mut buf = vec![0u8; total_bytes as usize];
                for reg_idx in 0..n {
                    let vreg = (u64::from(*vt) + reg_idx) & 31;
                    let (lo, hi) = state.read_v128(vreg as u8);
                    for lane in 0..elems_per_reg {
                        let bit_offset = lane * elem_size * 8;
                        let elem = if bit_offset < 64 {
                            (lo >> bit_offset) & ((1u64 << (elem_size * 8)) - 1)
                        } else {
                            (hi >> (bit_offset - 64)) & ((1u64 << (elem_size * 8)) - 1)
                        };
                        let mem_idx = if *interleave {
                            (lane * n + reg_idx) * elem_size
                        } else {
                            (reg_idx * elems_per_reg + lane) * elem_size
                        };
                        for b in 0..elem_size {
                            buf[(mem_idx + b) as usize] = ((elem >> (b * 8)) & 0xff) as u8;
                        }
                    }
                }

                // Write buffer to memory
                let mut ok = true;
                for i in 0..total_bytes {
                    if !memory.write_u8(base.wrapping_add(i), buf[i as usize]) {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }

                // Post-index writeback
                if let Some(rm) = post_index {
                    let offset = if *rm == 31 {
                        total_bytes
                    } else {
                        state.read_x(*rm)
                    };
                    write_gp_or_sp(state, *rn, base.wrapping_add(offset));
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdLdReplicate {
                q,
                vt,
                rn,
                size,
                nregs,
                post_index,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let elem_size: u64 = 1u64 << *size;
                let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                let elems_per_reg = bytes_per_reg / elem_size;

                let mut addr = base;
                let mut ok = true;
                for reg_idx in 0..(*nregs as u64) {
                    let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                    // Read one element
                    let elem: u64 = match *size {
                        0 => match memory.read_u8(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        1 => match memory.read_u16(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        2 => match memory.read_u32(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        3 => match memory.read_u64(addr) {
                            Some(v) => v,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        _ => unreachable!(),
                    };
                    addr = addr.wrapping_add(elem_size);

                    // Replicate to all lanes
                    let mut lo = 0u64;
                    let mut hi = 0u64;
                    for lane in 0..elems_per_reg {
                        let bit_offset = lane * elem_size * 8;
                        if bit_offset < 64 {
                            lo |= elem << bit_offset;
                        } else {
                            hi |= elem << (bit_offset - 64);
                        }
                    }
                    state.write_v128(vreg, lo, hi);
                }
                if !ok {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }

                if let Some(rm) = post_index {
                    let offset = if *rm == 31 {
                        elem_size * u64::from(*nregs)
                    } else {
                        state.read_x(*rm)
                    };
                    write_gp_or_sp(state, *rn, base.wrapping_add(offset));
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdLdSingle {
                q: _,
                vt,
                rn,
                size,
                index,
                nregs,
                post_index,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let elem_size: u64 = 1u64 << *size;
                let elem_bits = elem_size * 8;
                let elem_mask: u64 = if elem_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << elem_bits) - 1
                };

                let mut addr = base;
                let mut ok = true;
                for reg_idx in 0..(*nregs as u64) {
                    let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                    let elem: u64 = match *size {
                        0 => match memory.read_u8(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        1 => match memory.read_u16(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        2 => match memory.read_u32(addr) {
                            Some(v) => v as u64,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        3 => match memory.read_u64(addr) {
                            Some(v) => v,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        _ => unreachable!(),
                    };
                    addr = addr.wrapping_add(elem_size);

                    // Insert element into existing register value at [index]
                    let (mut lo, mut hi) = state.read_v128(vreg);
                    let bit_offset = u64::from(*index) * elem_bits;
                    if bit_offset < 64 {
                        lo = (lo & !(elem_mask << bit_offset)) | ((elem & elem_mask) << bit_offset);
                    } else {
                        let off = bit_offset - 64;
                        hi = (hi & !(elem_mask << off)) | ((elem & elem_mask) << off);
                    }
                    state.write_v128(vreg, lo, hi);
                }
                if !ok {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }

                if let Some(rm) = post_index {
                    let offset = if *rm == 31 {
                        elem_size * u64::from(*nregs)
                    } else {
                        state.read_x(*rm)
                    };
                    write_gp_or_sp(state, *rn, base.wrapping_add(offset));
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdStSingle {
                q: _,
                vt,
                rn,
                size,
                index,
                nregs,
                post_index,
            } => {
                let base = read_gp_or_sp(state, *rn);
                let elem_size: u64 = 1u64 << *size;
                let elem_bits = elem_size * 8;
                let elem_mask: u64 = if elem_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << elem_bits) - 1
                };

                let mut addr = base;
                let mut ok = true;
                for reg_idx in 0..(*nregs as u64) {
                    let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                    let (lo, hi) = state.read_v128(vreg);
                    let bit_offset = u64::from(*index) * elem_bits;
                    let elem = if bit_offset < 64 {
                        (lo >> bit_offset) & elem_mask
                    } else {
                        (hi >> (bit_offset - 64)) & elem_mask
                    };

                    let wrote = match *size {
                        0 => memory.write_u8(addr, elem as u8),
                        1 => memory.write_u16(addr, elem as u16),
                        2 => memory.write_u32(addr, elem as u32),
                        3 => memory.write_u64(addr, elem),
                        _ => unreachable!(),
                    };
                    if !wrote {
                        ok = false;
                        break;
                    }
                    addr = addr.wrapping_add(elem_size);
                }
                if !ok {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }

                if let Some(rm) = post_index {
                    let offset = if *rm == 31 {
                        elem_size * u64::from(*nregs)
                    } else {
                        state.read_x(*rm)
                    };
                    write_gp_or_sp(state, *rn, base.wrapping_add(offset));
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSaddl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Saddl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUaddl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Uaddl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSaddw {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Saddw, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUaddw {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Uaddw, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSsubl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Ssubl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUsubl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Usubl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSsubw {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Ssubw, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUsubw {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Usubw, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSabdl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Sabdl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUabdl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Uabdl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSmlal {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Smlal, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUmlal {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Umlal, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSmlsl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Smlsl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUmlsl {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Umlsl, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSmull {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Smull, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUmull {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_three_diff(state, *q, SimdWidenOp::Umull, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdAddhn {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_narrow(state, *q, SimdNarrowOp::Addhn, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdRaddhn {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_narrow(state, *q, SimdNarrowOp::Raddhn, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSubhn {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_narrow(state, *q, SimdNarrowOp::Subhn, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdRsubhn {
                q,
                size,
                rd,
                rn,
                rm,
            } => {
                execute_simd_narrow(state, *q, SimdNarrowOp::Rsubhn, *size, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSshll {
                q,
                size,
                shift,
                rd,
                rn,
            } => {
                execute_simd_shift_long(state, *q, true, *size, *shift, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUshll {
                q,
                size,
                shift,
                rd,
                rn,
            } => {
                execute_simd_shift_long(state, *q, false, *size, *shift, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdSshr {
                q,
                size,
                shift,
                rd,
                rn,
            } => {
                execute_simd_shift_imm(state, *q, SimdShiftOp::Sshr, *size, *shift, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdUshr {
                q,
                size,
                shift,
                rd,
                rn,
            } => {
                execute_simd_shift_imm(state, *q, SimdShiftOp::Ushr, *size, *shift, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SimdShl {
                q,
                size,
                shift,
                rd,
                rn,
            } => {
                execute_simd_shift_imm(state, *q, SimdShiftOp::Shl, *size, *shift, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtRoundFromFloat {
                rd,
                rn,
                sf,
                ftype,
                rmode,
                is_unsigned,
            } => {
                execute_fcvt_round_from_float(state, *rd, *rn, *sf, *ftype, *rmode, *is_unsigned);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::LdrFpImm8 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12));

                match memory.read_u8(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpImm16 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 1);

                match memory.read_u16(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpImm32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 2);

                match memory.read_u32(address) {
                    Some(value) => {
                        state.write_v128(*rt, u64::from(value), 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpImm64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 3);

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_v128(*rt, value, 0);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::LdrFpImm128 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 4);
                let lo = memory.read_u64(address);
                let hi = memory.read_u64(address.wrapping_add(8));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        state.write_v128(*rt, lo, hi);
                        pc = pc.wrapping_add(4);
                    }
                    _ => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrFpImm8 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12));
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u8(address, lo as u8) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpImm16 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 1);
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u16(address, lo as u16) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpImm32 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 2);
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u32(address, lo as u32) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpImm64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 3);
                let (lo, _) = state.read_v128(*rt);

                if memory.write_u64(address, lo) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::StrFpImm128 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let address = base.wrapping_add(u64::from(*imm12) << 4);
                let (lo, hi) = state.read_v128(*rt);

                let ok_lo = memory.write_u64(address, lo);
                let ok_hi = memory.write_u64(address.wrapping_add(8), hi);
                if ok_lo && ok_hi {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::LdrImmU64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 3;
                let address = base.wrapping_add(offset);

                match memory.read_u64(address) {
                    Some(value) => {
                        state.write_x(*rt, value);
                        pc = pc.wrapping_add(4);
                    }
                    None => {
                        state.set_memory_fault();
                        state.pc = pc;
                        return;
                    }
                }
            }
            DecodedInstruction::StrImmU64 { rt, rn, imm12 } => {
                let base = read_gp_or_sp(state, *rn);
                let offset = u64::from(*imm12) << 3;
                let address = base.wrapping_add(offset);
                let value = state.read_x(*rt);

                if memory.write_u64(address, value) {
                    pc = pc.wrapping_add(4);
                } else {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
            }
            DecodedInstruction::B { target } => {
                state.pc = *target;
                return;
            }
            DecodedInstruction::Bl { target } => {
                state.write_x(30, pc.wrapping_add(4));
                state.pc = *target;
                return;
            }
            DecodedInstruction::BCond { target, condition } => {
                if state.condition_holds(*condition) {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Cbz { rt, target } => {
                if state.read_x(*rt) == 0 {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Cbnz { rt, target } => {
                if state.read_x(*rt) != 0 {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Cbz32 { rt, target } => {
                if (state.read_x(*rt) as u32) == 0 {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Cbnz32 { rt, target } => {
                if (state.read_x(*rt) as u32) != 0 {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Tbz { rt, bit, target } => {
                let bit_set = ((state.read_x(*rt) >> *bit) & 1) != 0;
                if !bit_set {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Tbnz { rt, bit, target } => {
                let bit_set = ((state.read_x(*rt) >> *bit) & 1) != 0;
                if bit_set {
                    state.pc = *target;
                } else {
                    state.pc = pc.wrapping_add(4);
                }
                return;
            }
            DecodedInstruction::Adr { rd, target } | DecodedInstruction::Adrp { rd, target } => {
                state.write_x(*rd, *target);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Blr { rn } => {
                state.write_x(30, pc.wrapping_add(4));
                state.pc = state.read_x(*rn);
                return;
            }
            DecodedInstruction::Br { rn } | DecodedInstruction::Ret { rn } => {
                state.pc = state.read_x(*rn);
                return;
            }
            DecodedInstruction::Brk { imm16 } => {
                state.set_breakpoint(*imm16);
                state.pc = pc.wrapping_add(4);
                return;
            }
            DecodedInstruction::Svc { imm16 } => {
                state.set_svc(*imm16);
                state.pc = pc.wrapping_add(4);
                return;
            }
            DecodedInstruction::Hvc { imm16 } => {
                state.set_hvc(*imm16);
                state.pc = pc.wrapping_add(4);
                return;
            }
            DecodedInstruction::Smc { imm16 } => {
                state.set_smc(*imm16);
                state.pc = pc.wrapping_add(4);
                return;
            }
            DecodedInstruction::ScvtfS32 { rd, rn } => {
                execute_scvtf_s32(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfS32 { rd, rn } => {
                let unsigned = state.read_x(*rn) as u32;
                let converted = (unsigned as f32).to_bits();
                write_v_scalar_f32(state, *rd, converted);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfD32 { rd, rn } => {
                execute_ucvtf_d32(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ScvtfD32 { rd, rn } => {
                execute_scvtf_d32(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfSReg32 { rd, rn } => {
                execute_ucvtf_s_reg32(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ScvtfSReg32 { rd, rn } => {
                execute_scvtf_s_reg32(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfDReg64 { rd, rn } => {
                execute_ucvtf_d_reg64(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ScvtfDReg64 { rd, rn } => {
                execute_scvtf_d_reg64(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ScvtfS64 { rd, rn } => {
                let signed = state.read_x(*rn) as i64;
                let converted = (signed as f32).to_bits();
                write_v_scalar_f32(state, *rd, converted);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ScvtfD64 { rd, rn } => {
                let signed = state.read_x(*rn) as i64;
                let converted = (signed as f64).to_bits();
                write_v_scalar_f64(state, *rd, converted);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfS64 { rd, rn } => {
                let unsigned = state.read_x(*rn);
                let converted = (unsigned as f32).to_bits();
                write_v_scalar_f32(state, *rd, converted);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UcvtfD64 { rd, rn } => {
                let unsigned = state.read_x(*rn);
                let converted = (unsigned as f64).to_bits();
                write_v_scalar_f64(state, *rd, converted);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFadd { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fadd);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFsub { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fsub);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmul { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmul);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFdiv { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fdiv);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmax { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmax);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmin { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmin);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmaxnm { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmaxnm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFminnm { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fminnm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmulx { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmulx);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFabd { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fabd);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFaddp { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Faddp);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmaxp { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmaxp);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFminp { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fminp);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrecps { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Frecps);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrsqrts { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Frsqrts);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmeq { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fcmeq);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmge { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fcmge);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmgt { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fcmgt);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFacge { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Facge);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFacgt { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Facgt);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmla { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmla);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmls { vd, vn, vm, q, sz } => {
                execute_fp_vector_three_same(state, *vd, *vn, *vm, *q, *sz, FpVecOp::Fmls);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrecpe { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frecpe);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrsqrte { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frsqrte);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFabs { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fabs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFneg { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fneg);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFsqrt { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fsqrt);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtns { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtns);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtnu { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtnu);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtms { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtms);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtmu { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtmu);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtas { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtas);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtau { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtau);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtps { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtps);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtpu { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtpu);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtzs { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtzs);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcvtzu { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcvtzu);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmeq0 { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcmeq0);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmge0 { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcmge0);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmgt0 { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcmgt0);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmle0 { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcmle0);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFcmlt0 { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Fcmlt0);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecScvtf { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Scvtf);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecUcvtf { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Ucvtf);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrint32x { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frint32x);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrintn { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frintn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrintm { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frintm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrintp { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frintp);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrintz { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frintz);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrinta { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frinta);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrintx { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frintx);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFrinti { vd, vn, q, sz } => {
                execute_fp_vector_unary(state, *vd, *vn, *q, *sz, FpVecUnaryOp::Frinti);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmulElem {
                vd,
                vn,
                vm,
                index,
                q,
                sz,
            } => {
                execute_fp_vector_by_element(
                    state,
                    *vd,
                    *vn,
                    *vm,
                    *index,
                    *q,
                    *sz,
                    FpVecElemOp::Fmul,
                );
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmlaElem {
                vd,
                vn,
                vm,
                index,
                q,
                sz,
            } => {
                execute_fp_vector_by_element(
                    state,
                    *vd,
                    *vn,
                    *vm,
                    *index,
                    *q,
                    *sz,
                    FpVecElemOp::Fmla,
                );
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmlsElem {
                vd,
                vn,
                vm,
                index,
                q,
                sz,
            } => {
                execute_fp_vector_by_element(
                    state,
                    *vd,
                    *vn,
                    *vm,
                    *index,
                    *q,
                    *sz,
                    FpVecElemOp::Fmls,
                );
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::VecFmulxElem {
                vd,
                vn,
                vm,
                index,
                q,
                sz,
            } => {
                execute_fp_vector_by_element(
                    state,
                    *vd,
                    *vn,
                    *vm,
                    *index,
                    *q,
                    *sz,
                    FpVecElemOp::Fmulx,
                );
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FaddS { rd, rn, rm } => {
                execute_fadd_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FsubS { rd, rn, rm } => {
                execute_fsub_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmulS { rd, rn, rm } => {
                execute_fmul_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmaxS { rd, rn, rm } => {
                execute_fmax_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FminS { rd, rn, rm } => {
                execute_fmin_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmaxnmS { rd, rn, rm } => {
                let lhs = f32::from_bits(read_v_scalar_f32(state, *rn));
                let rhs = f32::from_bits(read_v_scalar_f32(state, *rm));
                let result = if lhs.is_nan() && rhs.is_nan() {
                    f32::NAN
                } else if lhs.is_nan() {
                    rhs
                } else if rhs.is_nan() {
                    lhs
                } else {
                    lhs.max(rhs)
                };
                write_v_scalar_f32(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FminnmS { rd, rn, rm } => {
                let lhs = f32::from_bits(read_v_scalar_f32(state, *rn));
                let rhs = f32::from_bits(read_v_scalar_f32(state, *rm));
                let result = if lhs.is_nan() && rhs.is_nan() {
                    f32::NAN
                } else if lhs.is_nan() {
                    rhs
                } else if rhs.is_nan() {
                    lhs
                } else {
                    lhs.min(rhs)
                };
                write_v_scalar_f32(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FdivS { rd, rn, rm } => {
                execute_fdiv_s(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FaddD { rd, rn, rm } => {
                execute_fadd_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FsubD { rd, rn, rm } => {
                execute_fsub_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmulD { rd, rn, rm } => {
                execute_fmul_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmaxD { rd, rn, rm } => {
                execute_fmax_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FminD { rd, rn, rm } => {
                execute_fmin_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmaxnmD { rd, rn, rm } => {
                let lhs = f64::from_bits(read_v_scalar_f64(state, *rn));
                let rhs = f64::from_bits(read_v_scalar_f64(state, *rm));
                let result = if lhs.is_nan() && rhs.is_nan() {
                    f64::NAN
                } else if lhs.is_nan() {
                    rhs
                } else if rhs.is_nan() {
                    lhs
                } else {
                    lhs.max(rhs)
                };
                write_v_scalar_f64(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FminnmD { rd, rn, rm } => {
                let lhs = f64::from_bits(read_v_scalar_f64(state, *rn));
                let rhs = f64::from_bits(read_v_scalar_f64(state, *rm));
                let result = if lhs.is_nan() && rhs.is_nan() {
                    f64::NAN
                } else if lhs.is_nan() {
                    rhs
                } else if rhs.is_nan() {
                    lhs
                } else {
                    lhs.min(rhs)
                };
                write_v_scalar_f64(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FdivD { rd, rn, rm } => {
                execute_fdiv_d(state, *rd, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FnmulS { rd, rn, rm } => {
                let a = f32::from_bits(read_v_scalar_f32(state, *rn));
                let b = f32::from_bits(read_v_scalar_f32(state, *rm));
                let result = -(a * b);
                write_v_scalar_f32(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FnmulD { rd, rn, rm } => {
                let a = f64::from_bits(read_v_scalar_f64(state, *rn));
                let b = f64::from_bits(read_v_scalar_f64(state, *rm));
                let result = -(a * b);
                write_v_scalar_f64(state, *rd, result.to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovImmS { rd, imm8 } => {
                execute_fmov_imm_s(state, *rd, *imm8);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovImmD { rd, imm8 } => {
                execute_fmov_imm_d(state, *rd, *imm8);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovSFromW { rd, rn } => {
                execute_fmov_s_from_w(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovWFromS { rd, rn } => {
                execute_fmov_w_from_s(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovDFromX { rd, rn } => {
                execute_fmov_d_from_x(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FmovXFromD { rd, rn } => {
                execute_fmov_x_from_d(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtDFromS { rd, rn } => {
                execute_fcvt_d_from_s(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtSFromD { rd, rn } => {
                execute_fcvt_s_from_d(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuXFromD { rd, rn } => {
                execute_fcvtzu_x_from_d(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            // FCVTZS (scalar integer) - FP to signed int, round toward zero
            DecodedInstruction::FcvtzsWFromS { rd, rn } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let result = if source.is_nan() {
                    0i32
                } else if source >= i32::MAX as f32 {
                    i32::MAX
                } else if source <= i32::MIN as f32 {
                    i32::MIN
                } else {
                    source as i32
                };
                state.write_x(*rd, result as u32 as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsXFromS { rd, rn } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = source as f64;
                let result = if scaled.is_nan() {
                    0i64
                } else if scaled >= i64::MAX as f64 {
                    i64::MAX
                } else if scaled <= i64::MIN as f64 {
                    i64::MIN
                } else {
                    scaled as i64
                };
                state.write_x(*rd, result as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsWFromD { rd, rn } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let result = if source.is_nan() {
                    0i32
                } else if source >= i32::MAX as f64 {
                    i32::MAX
                } else if source <= i32::MIN as f64 {
                    i32::MIN
                } else {
                    source as i32
                };
                state.write_x(*rd, result as u32 as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsXFromD { rd, rn } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let result = if source.is_nan() {
                    0i64
                } else if source >= i64::MAX as f64 {
                    i64::MAX
                } else if source <= i64::MIN as f64 {
                    i64::MIN
                } else {
                    source as i64
                };
                state.write_x(*rd, result as u64);
                pc = pc.wrapping_add(4);
            }
            // FCVTZU (scalar integer) - FP to unsigned int, round toward zero
            DecodedInstruction::FcvtzuWFromS { rd, rn } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let result = if source.is_nan() || source < 0.0 {
                    0u32
                } else if source >= u32::MAX as f32 {
                    u32::MAX
                } else {
                    source as u32
                };
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuXFromS { rd, rn } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = source as f64;
                let result = if scaled.is_nan() || scaled < 0.0 {
                    0u64
                } else if scaled >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    scaled as u64
                };
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuWFromD { rd, rn } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let result = if source.is_nan() || source < 0.0 {
                    0u32
                } else if source >= u32::MAX as f64 {
                    u32::MAX
                } else {
                    source as u32
                };
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsFixedWFromS { rd, rn, fbits } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = source * (1u64 << *fbits) as f32;
                let result = if scaled.is_nan() {
                    0i32
                } else if scaled >= i32::MAX as f32 {
                    i32::MAX
                } else if scaled <= i32::MIN as f32 {
                    i32::MIN
                } else {
                    scaled as i32
                };
                state.write_x(*rd, result as u32 as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsFixedXFromS { rd, rn, fbits } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = (source as f64) * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() {
                    0i64
                } else if scaled >= i64::MAX as f64 {
                    i64::MAX
                } else if scaled <= i64::MIN as f64 {
                    i64::MIN
                } else {
                    scaled as i64
                };
                state.write_x(*rd, result as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsFixedWFromD { rd, rn, fbits } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let scaled = source * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() {
                    0i32
                } else if scaled >= i32::MAX as f64 {
                    i32::MAX
                } else if scaled <= i32::MIN as f64 {
                    i32::MIN
                } else {
                    scaled as i32
                };
                state.write_x(*rd, result as u32 as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzsFixedXFromD { rd, rn, fbits } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let scaled = source * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() {
                    0i64
                } else if scaled >= i64::MAX as f64 {
                    i64::MAX
                } else if scaled <= i64::MIN as f64 {
                    i64::MIN
                } else {
                    scaled as i64
                };
                state.write_x(*rd, result as u64);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuFixedWFromS { rd, rn, fbits } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = source * (1u64 << *fbits) as f32;
                let result = if scaled.is_nan() || scaled < 0.0 {
                    0u32
                } else if scaled >= u32::MAX as f32 {
                    u32::MAX
                } else {
                    scaled as u32
                };
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuFixedXFromS { rd, rn, fbits } => {
                let source = f32::from_bits(read_v_scalar_f32(state, *rn));
                let scaled = (source as f64) * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() || scaled < 0.0 {
                    0u64
                } else if scaled >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    scaled as u64
                };
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuFixedWFromD { rd, rn, fbits } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let scaled = source * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() || scaled < 0.0 {
                    0u32
                } else if scaled >= u32::MAX as f64 {
                    u32::MAX
                } else {
                    scaled as u32
                };
                state.write_x(*rd, u64::from(result));
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtzuFixedXFromD { rd, rn, fbits } => {
                let source = f64::from_bits(read_v_scalar_f64(state, *rn));
                let scaled = source * (1u64 << *fbits) as f64;
                let result = if scaled.is_nan() || scaled < 0.0 {
                    0u64
                } else if scaled >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    scaled as u64
                };
                state.write_x(*rd, result);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcvtmuXFromD { rd, rn } => {
                execute_fcvtmu_x_from_d(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovXFromVD { rd, vn, index } => {
                execute_mov_x_from_v_d(state, *rd, *vn, *index);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MovVD1FromX { vd, rn } => {
                execute_mov_v_d1_from_x(state, *vd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::InsGeneralVector {
                vd,
                rn,
                dst_index,
                size,
            } => {
                let src_val = state.read_x(*rn);
                let (mut d_lo, mut d_hi) = state.read_v128(*vd);
                let elem_bits = 8u32 << *size;
                let mask = if elem_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << elem_bits) - 1
                };
                let insert_val = src_val & mask;
                let dst_bit = u32::from(*dst_index) * elem_bits;
                if dst_bit < 64 {
                    d_lo = (d_lo & !(mask << dst_bit)) | (insert_val << dst_bit);
                } else {
                    let shift = dst_bit - 64;
                    d_hi = (d_hi & !(mask << shift)) | (insert_val << shift);
                }
                state.write_v128(*vd, d_lo, d_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::InsElement {
                vd,
                vn,
                dst_index,
                src_index,
                size,
            } => {
                let (lo, hi) = state.read_v128(*vn);
                let (mut d_lo, mut d_hi) = state.read_v128(*vd);
                let elem_bits = 8u32 << *size;
                let mask = if elem_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << elem_bits) - 1
                };
                // Extract source element
                let src_bit = u32::from(*src_index) * elem_bits;
                let src_val = if src_bit < 64 {
                    (lo >> src_bit) & mask
                } else {
                    (hi >> (src_bit - 64)) & mask
                };
                // Insert into destination element
                let dst_bit = u32::from(*dst_index) * elem_bits;
                if dst_bit < 64 {
                    d_lo = (d_lo & !(mask << dst_bit)) | (src_val << dst_bit);
                } else {
                    let shift = dst_bit - 64;
                    d_hi = (d_hi & !(mask << shift)) | (src_val << shift);
                }
                state.write_v128(*vd, d_lo, d_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DupV16BFromW { vd, rn } => {
                execute_dup_v16b_from_w(state, *vd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DupV4SFromW { vd, rn } => {
                execute_dup_v4s_from_w(state, *vd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MoviV2DZero { vd } | DecodedInstruction::MoviV4SZero { vd } => {
                execute_movi_v2d_zero(state, *vd);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::MoviVector { vd, imm_lo, imm_hi }
            | DecodedInstruction::MvniVector { vd, imm_lo, imm_hi } => {
                state.write_v128(*vd, *imm_lo, *imm_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrImmVector { vd, imm_lo, imm_hi } => {
                let (lo, hi) = state.read_v128(*vd);
                state.write_v128(*vd, lo | imm_lo, hi | imm_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BicImmVector { vd, imm_lo, imm_hi } => {
                let (lo, hi) = state.read_v128(*vd);
                state.write_v128(*vd, lo & !imm_lo, hi & !imm_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_add_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AddpVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_addp_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UminpVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_uminp_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UmaxpVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_umaxp_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::ShrnVector {
                vd,
                vn,
                q,
                size,
                shift,
            } => {
                execute_shrn_vector(state, *vd, *vn, *q, *size, *shift);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::AndVector { vd, vn, vm, q } => {
                execute_and_vector(state, *vd, *vn, *vm, *q);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::OrrVector { vd, vn, vm, q } => {
                execute_orr_vector(state, *vd, *vn, *vm, *q);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CmeqVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_cmeq_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CmhsVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_cmhs_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DupGeneralVector { vd, rn, q, size } => {
                let val = state.read_x(*rn);
                let (lo, hi) = match size {
                    0 => {
                        let b = (val & 0xff) as u8;
                        let lane = u64::from_le_bytes([b; 8]);
                        (lane, if *q { lane } else { 0 })
                    }
                    1 => {
                        let h = (val & 0xffff) as u16;
                        let mut bytes = [0u8; 8];
                        for i in 0..4 {
                            bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                        }
                        let lane = u64::from_le_bytes(bytes);
                        (lane, if *q { lane } else { 0 })
                    }
                    2 => {
                        let w = (val & 0xffff_ffff) as u32;
                        let lo = u64::from(w) | (u64::from(w) << 32);
                        (lo, if *q { lo } else { 0 })
                    }
                    3 => (val, if *q { val } else { 0 }),
                    _ => unreachable!(),
                };
                state.write_v128(*vd, lo, hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DupElementVector {
                vd,
                vn,
                q,
                size,
                index,
            } => {
                let (src_lo, src_hi) = state.read_v128(*vn);
                let elem_bits = (1u64 << *size) * 8;
                let bit_offset = u64::from(*index) * elem_bits;
                let elem_mask = if elem_bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << elem_bits) - 1
                };
                let elem = if bit_offset < 64 {
                    (src_lo >> bit_offset) & elem_mask
                } else {
                    (src_hi >> (bit_offset - 64)) & elem_mask
                };
                let (lo, hi) = match *size {
                    0 => {
                        let b = elem as u8;
                        let lane = u64::from_le_bytes([b; 8]);
                        (lane, if *q { lane } else { 0 })
                    }
                    1 => {
                        let h = elem as u16;
                        let mut bytes = [0u8; 8];
                        for i in 0..4 {
                            bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                        }
                        let lane = u64::from_le_bytes(bytes);
                        (lane, if *q { lane } else { 0 })
                    }
                    2 => {
                        let lo_val = (elem & 0xffff_ffff) | ((elem & 0xffff_ffff) << 32);
                        (lo_val, if *q { lo_val } else { 0 })
                    }
                    3 => (elem, if *q { elem } else { 0 }),
                    _ => unreachable!(),
                };
                state.write_v128(*vd, lo, hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CmhiVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_cmhi_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UshlVector {
                vd,
                vn,
                vm,
                q,
                size,
            } => {
                execute_ushl_vector(state, *vd, *vn, *vm, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::BifVector { vd, vn, vm, q } => {
                execute_bif_vector(state, *vd, *vn, *vm, *q);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CmeqVectorZero { vd, vn, q, size } => {
                execute_cmeq_vector_zero(state, *vd, *vn, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::NegVector { vd, vn, q, size } => {
                let (lo, hi) = state.read_v128(*vn);
                let bytes = *size;
                let elem_mask = match bytes {
                    0 => 0xff_u64,
                    1 => 0xffff_u64,
                    2 => 0xffff_ffff_u64,
                    3 => u64::MAX,
                    _ => unreachable!(),
                };
                let elem_bits = 8u32 << bytes;
                let elems_per_u64 = 64 / elem_bits;
                let mut neg_lo = 0u64;
                let mut neg_hi = 0u64;
                for i in 0..elems_per_u64 {
                    let shift = i * elem_bits;
                    let val = (lo >> shift) & elem_mask;
                    let negated = (0u64.wrapping_sub(val)) & elem_mask;
                    neg_lo |= negated << shift;
                }
                if *q {
                    for i in 0..elems_per_u64 {
                        let shift = i * elem_bits;
                        let val = (hi >> shift) & elem_mask;
                        let negated = (0u64.wrapping_sub(val)) & elem_mask;
                        neg_hi |= negated << shift;
                    }
                }
                state.write_v128(*vd, neg_lo, neg_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Rev64Vector { vd, vn, q, size } => {
                execute_rev64_vector(state, *vd, *vn, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::CntVector { vd, vn, q } => {
                execute_cnt_vector(state, *vd, *vn, *q);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::NotVector { vd, vn, q } => {
                let (lo, hi) = state.read_v128(*vn);
                let result_hi = if *q { !hi } else { 0 };
                state.write_v128(*vd, !lo, result_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::RbitVector { vd, vn, q } => {
                let (lo, hi) = state.read_v128(*vn);
                let rev = |v: u64| -> u64 {
                    let mut r = 0u64;
                    for i in 0..8 {
                        let byte = ((v >> (i * 8)) & 0xff) as u8;
                        let rb = byte.reverse_bits();
                        r |= (rb as u64) << (i * 8);
                    }
                    r
                };
                let result_hi = if *q { rev(hi) } else { 0 };
                state.write_v128(*vd, rev(lo), result_hi);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::UaddlvVector { vd, vn, q, size } => {
                execute_uaddlv_vector(state, *vd, *vn, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::SaddlvVector { vd, vn, q, size } => {
                execute_saddlv_vector(state, *vd, *vn, *q, *size);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcmpS { rn, rm } => {
                execute_fcmp_s(state, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcmpD { rn, rm } => {
                execute_fcmp_d(state, *rn, *rm);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcselS {
                rd,
                rn,
                rm,
                condition,
            } => {
                let bits = if state.condition_holds(*condition) {
                    read_v_scalar_f32(state, *rn)
                } else {
                    read_v_scalar_f32(state, *rm)
                };
                write_v_scalar_f32(state, *rd, bits);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FcselD {
                rd,
                rn,
                rm,
                condition,
            } => {
                let bits = if state.condition_holds(*condition) {
                    read_v_scalar_f64(state, *rn)
                } else {
                    read_v_scalar_f64(state, *rm)
                };
                write_v_scalar_f64(state, *rd, bits);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FrintmS { rd, rn } => {
                execute_frintm_s(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FrintmD { rd, rn } => {
                execute_frintm_d(state, *rd, *rn);
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FnegS { rd, rn } => {
                let val = f32::from_bits(read_v_scalar_f32(state, *rn));
                write_v_scalar_f32(state, *rd, (-val).to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FnegD { rd, rn } => {
                let val = f64::from_bits(read_v_scalar_f64(state, *rn));
                write_v_scalar_f64(state, *rd, (-val).to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FabsS { rd, rn } => {
                let val = f32::from_bits(read_v_scalar_f32(state, *rn));
                write_v_scalar_f32(state, *rd, val.abs().to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FabsD { rd, rn } => {
                let val = f64::from_bits(read_v_scalar_f64(state, *rn));
                write_v_scalar_f64(state, *rd, val.abs().to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FsqrtS { rd, rn } => {
                let val = f32::from_bits(read_v_scalar_f32(state, *rn));
                write_v_scalar_f32(state, *rd, val.sqrt().to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::FsqrtD { rd, rn } => {
                let val = f64::from_bits(read_v_scalar_f64(state, *rn));
                write_v_scalar_f64(state, *rd, val.sqrt().to_bits());
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DcZva { rt } => {
                if execute_dc_zva(state, memory, *rt).is_err() {
                    state.set_memory_fault();
                    state.pc = pc;
                    return;
                }
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::DcCivac { .. }
            | DecodedInstruction::Dmb { .. }
            | DecodedInstruction::Dsb { .. }
            | DecodedInstruction::Isb { .. } => {
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Clrex { .. } => {
                state.clear_exclusive_reservation();
                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Eret => {
                state.set_eret();
                state.pc = pc.wrapping_add(4);
                return;
            }
            DecodedInstruction::Mrs { rt, reg } => {
                match state.read_sys_reg(*reg) {
                    Ok(value) => state.write_x(*rt, value),
                    Err(_) => {
                        state.set_privilege_fault();
                        state.pc = pc.wrapping_add(4);
                        return;
                    }
                }

                pc = pc.wrapping_add(4);
            }
            DecodedInstruction::Msr { reg, rt } => {
                let value = state.read_x(*rt);
                match state.write_sys_reg(*reg, value) {
                    Ok(()) => {
                        pc = pc.wrapping_add(4);
                    }
                    Err(SysRegError::PrivilegeFault | SysRegError::ReadOnly) => {
                        state.set_privilege_fault();
                        state.pc = pc.wrapping_add(4);
                        return;
                    }
                }
            }
        }
    }

    state.pc = block.fallthrough_pc;
}

fn read_gp_or_sp(state: &CpuState, reg: u8) -> u64 {
    if reg == 31 {
        state.sp
    } else {
        state.read_x(reg)
    }
}

fn write_gp_or_sp(state: &mut CpuState, reg: u8, value: u64) {
    if reg == 31 {
        state.sp = value;
    } else {
        state.write_x(reg, value);
    }
}

fn read_w_or_wsp(state: &CpuState, reg: u8) -> u32 {
    if reg == 31 {
        state.sp as u32
    } else {
        state.read_x(reg) as u32
    }
}

fn write_w_or_wsp(state: &mut CpuState, reg: u8, value: u32) {
    if reg == 31 {
        state.sp = u64::from(value);
    } else {
        state.write_x(reg, u64::from(value));
    }
}

fn execute_scvtf_s32(state: &mut CpuState, rd: u8, rn: u8) {
    let signed = state.read_x(rn) as u32 as i32;
    let converted = (signed as f32).to_bits();
    write_v_scalar_f32(state, rd, converted);
}

fn execute_ucvtf_d32(state: &mut CpuState, rd: u8, rn: u8) {
    let unsigned = state.read_x(rn) as u32;
    let converted = (unsigned as f64).to_bits();
    write_v_scalar_f64(state, rd, converted);
}

fn execute_scvtf_d32(state: &mut CpuState, rd: u8, rn: u8) {
    let signed = state.read_x(rn) as u32 as i32;
    let converted = (signed as f64).to_bits();
    write_v_scalar_f64(state, rd, converted);
}

fn execute_ucvtf_s_reg32(state: &mut CpuState, rd: u8, rn: u8) {
    let unsigned = read_v_scalar_f32(state, rn);
    let converted = (unsigned as f32).to_bits();
    write_v_scalar_f32(state, rd, converted);
}

fn execute_scvtf_s_reg32(state: &mut CpuState, rd: u8, rn: u8) {
    let signed = read_v_scalar_f32(state, rn) as i32;
    let converted = (signed as f32).to_bits();
    write_v_scalar_f32(state, rd, converted);
}

fn execute_ucvtf_d_reg64(state: &mut CpuState, rd: u8, rn: u8) {
    let unsigned = read_v_scalar_f64(state, rn);
    let converted = (unsigned as f64).to_bits();
    write_v_scalar_f64(state, rd, converted);
}

fn execute_scvtf_d_reg64(state: &mut CpuState, rd: u8, rn: u8) {
    let signed = read_v_scalar_f64(state, rn) as i64;
    let converted = (signed as f64).to_bits();
    write_v_scalar_f64(state, rd, converted);
}

fn execute_fdiv_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = (lhs / rhs).to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fdiv_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = (lhs / rhs).to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fadd_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = (lhs + rhs).to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fadd_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = (lhs + rhs).to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fsub_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = (lhs - rhs).to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fsub_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = (lhs - rhs).to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fmul_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = (lhs * rhs).to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fmul_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = (lhs * rhs).to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fmax_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = if lhs.is_nan() || rhs.is_nan() {
        f32::NAN
    } else {
        lhs.max(rhs)
    }
    .to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fmax_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else {
        lhs.max(rhs)
    }
    .to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fmin_s(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));
    let result = if lhs.is_nan() || rhs.is_nan() {
        f32::NAN
    } else {
        lhs.min(rhs)
    }
    .to_bits();
    write_v_scalar_f32(state, rd, result);
}

fn execute_fmin_d(state: &mut CpuState, rd: u8, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));
    let result = if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else {
        lhs.min(rhs)
    }
    .to_bits();
    write_v_scalar_f64(state, rd, result);
}

fn execute_fmov_imm_s(state: &mut CpuState, rd: u8, imm8: u8) {
    write_v_scalar_f32(state, rd, expand_fmov_imm8_f32(imm8));
}

fn execute_fmov_imm_d(state: &mut CpuState, rd: u8, imm8: u8) {
    write_v_scalar_f64(state, rd, expand_fmov_imm8_f64(imm8));
}

fn execute_fmov_s_from_w(state: &mut CpuState, rd: u8, rn: u8) {
    let bits = state.read_x(rn) as u32;
    write_v_scalar_f32(state, rd, bits);
}

fn execute_fmov_w_from_s(state: &mut CpuState, rd: u8, rn: u8) {
    let bits = read_v_scalar_f32(state, rn);
    state.write_x(rd, u64::from(bits));
}

fn execute_fmov_d_from_x(state: &mut CpuState, rd: u8, rn: u8) {
    let bits = state.read_x(rn);
    state.write_v128(rd, bits, 0);
}

fn execute_fmov_x_from_d(state: &mut CpuState, rd: u8, rn: u8) {
    let (bits, _) = state.read_v128(rn);
    state.write_x(rd, bits);
}

fn execute_fcvt_d_from_s(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f32::from_bits(read_v_scalar_f32(state, rn));
    let converted = (source as f64).to_bits();
    write_v_scalar_f64(state, rd, converted);
}

fn execute_fcvt_s_from_d(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f64::from_bits(read_v_scalar_f64(state, rn));
    let converted = (source as f32).to_bits();
    write_v_scalar_f32(state, rd, converted);
}

fn execute_fcvtzu_x_from_d(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f64::from_bits(read_v_scalar_f64(state, rn));
    let converted = source.trunc() as u64;
    state.write_x(rd, converted);
}

fn execute_fcvtmu_x_from_d(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f64::from_bits(read_v_scalar_f64(state, rn));
    let converted = source.floor() as u64;
    state.write_x(rd, converted);
}

fn execute_mov_x_from_v_d(state: &mut CpuState, rd: u8, vn: u8, index: u8) {
    let (lo, hi) = state.read_v128(vn);
    let value = if index == 0 { lo } else { hi };
    state.write_x(rd, value);
}

fn execute_mov_v_d1_from_x(state: &mut CpuState, vd: u8, rn: u8) {
    let (lo, _) = state.read_v128(vd);
    let hi = state.read_x(rn);
    state.write_v128(vd, lo, hi);
}

fn execute_dup_v16b_from_w(state: &mut CpuState, vd: u8, rn: u8) {
    let byte = state.read_x(rn) as u8;
    let repeated = u64::from_le_bytes([byte; 8]);
    state.write_v128(vd, repeated, repeated);
}

fn execute_dup_v4s_from_w(state: &mut CpuState, vd: u8, rn: u8) {
    let word = state.read_x(rn) as u32;
    let repeated64 = (u64::from(word) << 32) | u64::from(word);
    state.write_v128(vd, repeated64, repeated64);
}

fn execute_dc_zva<M>(state: &CpuState, memory: &mut M, rt: u8) -> Result<(), ()>
where
    M: MemoryInterface,
{
    let dczid = DCZID_EL0_VALUE as u32;
    if (dczid & (1 << 4)) != 0 {
        return Ok(());
    }

    let block_size = 4u64 << (dczid & 0xf);
    let address = state.read_x(rt);
    let block_start = address & !(block_size - 1);

    let mut offset = 0u64;
    while offset < block_size {
        if !memory.write_u8(block_start.wrapping_add(offset), 0) {
            return Err(());
        }
        offset = offset.saturating_add(1);
    }

    Ok(())
}

fn execute_movi_v2d_zero(state: &mut CpuState, vd: u8) {
    state.write_v128(vd, 0, 0);
}

fn execute_add_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let lhs = (vn_bits >> shift) & lane_mask;
        let rhs = (vm_bits >> shift) & lane_mask;
        let lane_sum = lhs.wrapping_add(rhs) & lane_mask;
        result |= lane_sum << shift;
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_addp_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let first_idx = lane * 2;
        let second_idx = first_idx + 1;

        let lhs = if first_idx < lane_count {
            (vn_bits >> (first_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((first_idx - lane_count) * lane_bits)) & lane_mask
        };

        let rhs = if second_idx < lane_count {
            (vn_bits >> (second_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((second_idx - lane_count) * lane_bits)) & lane_mask
        };

        let lane_sum = lhs.wrapping_add(rhs) & lane_mask;
        result |= lane_sum << (lane * lane_bits);
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_uminp_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let first_idx = lane * 2;
        let second_idx = first_idx + 1;

        let lhs = if first_idx < lane_count {
            (vn_bits >> (first_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((first_idx - lane_count) * lane_bits)) & lane_mask
        };

        let rhs = if second_idx < lane_count {
            (vn_bits >> (second_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((second_idx - lane_count) * lane_bits)) & lane_mask
        };

        let lane_min = lhs.min(rhs);
        result |= lane_min << (lane * lane_bits);
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_umaxp_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let first_idx = lane * 2;
        let second_idx = first_idx + 1;

        let lhs = if first_idx < lane_count {
            (vn_bits >> (first_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((first_idx - lane_count) * lane_bits)) & lane_mask
        };

        let rhs = if second_idx < lane_count {
            (vn_bits >> (second_idx * lane_bits)) & lane_mask
        } else {
            (vm_bits >> ((second_idx - lane_count) * lane_bits)) & lane_mask
        };

        let lane_max = lhs.max(rhs);
        result |= lane_max << (lane * lane_bits);
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_shrn_vector(state: &mut CpuState, vd: u8, vn: u8, q: bool, size: u8, shift: u8) {
    let source_bits = 8u32 << u32::from(size);
    let dest_bits = source_bits / 2;
    let lane_count = 128 / source_bits;

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);

    let source_mask = (1u128 << source_bits) - 1;
    let dest_mask = (1u128 << dest_bits) - 1;

    let mut narrowed_half = 0u64;
    for lane in 0..lane_count {
        let source = (vn_bits >> (lane * source_bits)) & source_mask;
        let narrowed = ((source >> u32::from(shift)) & dest_mask) as u64;
        narrowed_half |= narrowed << (lane * dest_bits);
    }

    if q {
        let (old_lo, _) = state.read_v128(vd);
        state.write_v128(vd, old_lo, narrowed_half);
    } else {
        state.write_v128(vd, narrowed_half, 0);
    }
}

fn execute_and_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let lo = vn_lo & vm_lo;
    let hi = if q { vn_hi & vm_hi } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_orr_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let lo = vn_lo | vm_lo;
    let hi = if q { vn_hi | vm_hi } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_cmeq_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let lhs = (vn_bits >> shift) & lane_mask;
        let rhs = (vm_bits >> shift) & lane_mask;
        if lhs == rhs {
            result |= lane_mask << shift;
        }
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_cmhs_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let lhs = (vn_bits >> shift) & lane_mask;
        let rhs = (vm_bits >> shift) & lane_mask;
        if lhs >= rhs {
            result |= lane_mask << shift;
        }
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_cmhi_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let lhs = (vn_bits >> shift) & lane_mask;
        let rhs = (vm_bits >> shift) & lane_mask;
        if lhs > rhs {
            result |= lane_mask << shift;
        }
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_ushl_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let vm_bits = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let lane_mask = (1u128 << lane_bits) - 1;
    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let val = (vn_bits >> shift) & lane_mask;
        // The shift amount is the low byte of the Vm element, treated as signed
        let amt_raw = ((vm_bits >> shift) & 0xff) as u8;
        let amt = amt_raw as i8;
        let shifted = if amt >= 0 {
            let a = amt as u32;
            if a >= lane_bits {
                0u128
            } else {
                (val << a) & lane_mask
            }
        } else {
            let a = (-(amt as i32)) as u32;
            if a >= lane_bits {
                0u128
            } else {
                val >> a
            }
        };
        result |= shifted << shift;
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_bif_vector(state: &mut CpuState, vd: u8, vn: u8, vm: u8, q: bool) {
    let (vd_lo, vd_hi) = state.read_v128(vd);
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);

    let old_dest = (u128::from(vd_hi) << 64) | u128::from(vd_lo);
    let source = (u128::from(vn_hi) << 64) | u128::from(vn_lo);
    let mask = (u128::from(vm_hi) << 64) | u128::from(vm_lo);

    let active_mask = if q { u128::MAX } else { (1u128 << 64) - 1 };
    let old_dest = old_dest & active_mask;
    let source = source & active_mask;
    let mask = mask & active_mask;

    let result = (old_dest & mask) | (source & !mask);
    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

fn execute_cmeq_vector_zero(state: &mut CpuState, vd: u8, vn: u8, q: bool, size: u8) {
    let lane_bits = 8u32 << u32::from(size);
    let lane_count = if q { 128 / lane_bits } else { 64 / lane_bits };

    let (vn_lo, vn_hi) = state.read_v128(vn);
    let vn_bits = (u128::from(vn_hi) << 64) | u128::from(vn_lo);

    let lane_mask = if lane_bits == 128 {
        u128::MAX
    } else {
        (1u128 << lane_bits) - 1
    };

    let mut result = 0u128;
    for lane in 0..lane_count {
        let shift = lane * lane_bits;
        let value = (vn_bits >> shift) & lane_mask;
        if value == 0 {
            result |= lane_mask << shift;
        }
    }

    let lo = result as u64;
    let hi = if q { (result >> 64) as u64 } else { 0 };
    state.write_v128(vd, lo, hi);
}

/// CNT - Population count per byte in a vector register.
/// REV64 - Reverse elements within 64-bit doublewords.
fn execute_rev64_vector(state: &mut CpuState, vd: u8, vn: u8, q: bool, size: u8) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let byte_count = if q { 16 } else { 8 };

    let vn_bytes = {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&vn_lo.to_le_bytes());
        buf[8..16].copy_from_slice(&vn_hi.to_le_bytes());
        buf
    };

    let mut result = [0u8; 16];
    let elem_size = 1usize << size; // 1, 2, or 4 bytes per element
                                    // For each 8-byte group, reverse elements within it
    for group_start in (0..byte_count).step_by(8) {
        let elems_per_group = 8 / elem_size;
        for i in 0..elems_per_group {
            let src_offset = group_start + i * elem_size;
            let dst_offset = group_start + (elems_per_group - 1 - i) * elem_size;
            result[dst_offset..dst_offset + elem_size]
                .copy_from_slice(&vn_bytes[src_offset..src_offset + elem_size]);
        }
    }

    let lo = u64::from_le_bytes(result[0..8].try_into().unwrap());
    let hi = if q {
        u64::from_le_bytes(result[8..16].try_into().unwrap())
    } else {
        0
    };
    state.write_v128(vd, lo, hi);
}

fn execute_cnt_vector(state: &mut CpuState, vd: u8, vn: u8, q: bool) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let byte_count = if q { 16 } else { 8 };

    let vn_bytes = {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&vn_lo.to_le_bytes());
        buf[8..16].copy_from_slice(&vn_hi.to_le_bytes());
        buf
    };

    let mut result = [0u8; 16];
    for i in 0..byte_count {
        result[i] = vn_bytes[i].count_ones() as u8;
    }

    let lo = u64::from_le_bytes(result[0..8].try_into().unwrap());
    let hi = if q {
        u64::from_le_bytes(result[8..16].try_into().unwrap())
    } else {
        0
    };
    state.write_v128(vd, lo, hi);
}

/// UADDLV - Unsigned Add Long Across Vector.
/// Sums all elements as unsigned, stores result in scalar of double element width.
fn execute_uaddlv_vector(state: &mut CpuState, vd: u8, vn: u8, q: bool, size: u8) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let vn_bytes = {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&vn_lo.to_le_bytes());
        buf[8..16].copy_from_slice(&vn_hi.to_le_bytes());
        buf
    };

    let byte_count = if q { 16 } else { 8 };
    let sum: u64 = match size {
        0 => {
            // 8-bit elements → sum into 16-bit result
            let mut s = 0u64;
            for i in 0..byte_count {
                s += vn_bytes[i] as u64;
            }
            s
        }
        1 => {
            // 16-bit elements → sum into 32-bit result
            let elem_count = byte_count / 2;
            let mut s = 0u64;
            for i in 0..elem_count {
                let val = u16::from_le_bytes([vn_bytes[i * 2], vn_bytes[i * 2 + 1]]);
                s += val as u64;
            }
            s
        }
        2 => {
            // 32-bit elements (Q=1 only, 4S) → sum into 64-bit result
            let elem_count = byte_count / 4;
            let mut s = 0u64;
            for i in 0..elem_count {
                let val = u32::from_le_bytes([
                    vn_bytes[i * 4],
                    vn_bytes[i * 4 + 1],
                    vn_bytes[i * 4 + 2],
                    vn_bytes[i * 4 + 3],
                ]);
                s += val as u64;
            }
            s
        }
        _ => 0,
    };

    // Result goes into scalar register (zeroed upper bits).
    state.write_v128(vd, sum, 0);
}

/// SADDLV - Signed Add Long Across Vector.
fn execute_saddlv_vector(state: &mut CpuState, vd: u8, vn: u8, q: bool, size: u8) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let vn_bytes = {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&vn_lo.to_le_bytes());
        buf[8..16].copy_from_slice(&vn_hi.to_le_bytes());
        buf
    };

    let byte_count = if q { 16 } else { 8 };
    let sum: i64 = match size {
        0 => {
            let mut s = 0i64;
            for i in 0..byte_count {
                s += (vn_bytes[i] as i8) as i64;
            }
            s
        }
        1 => {
            let elem_count = byte_count / 2;
            let mut s = 0i64;
            for i in 0..elem_count {
                let val = i16::from_le_bytes([vn_bytes[i * 2], vn_bytes[i * 2 + 1]]);
                s += val as i64;
            }
            s
        }
        2 => {
            let elem_count = byte_count / 4;
            let mut s = 0i64;
            for i in 0..elem_count {
                let val = i32::from_le_bytes([
                    vn_bytes[i * 4],
                    vn_bytes[i * 4 + 1],
                    vn_bytes[i * 4 + 2],
                    vn_bytes[i * 4 + 3],
                ]);
                s += val as i64;
            }
            s
        }
        _ => 0,
    };

    state.write_v128(vd, sum as u64, 0);
}

fn execute_fcmp_s(state: &mut CpuState, rn: u8, rm: u8) {
    let lhs = f32::from_bits(read_v_scalar_f32(state, rn));
    let rhs = f32::from_bits(read_v_scalar_f32(state, rm));

    let (n, z, c, v) = if lhs.is_nan() || rhs.is_nan() {
        (0u32, 0u32, 1u32, 1u32)
    } else if lhs < rhs {
        (1u32, 0u32, 0u32, 0u32)
    } else if lhs > rhs {
        (0u32, 0u32, 1u32, 0u32)
    } else {
        (0u32, 1u32, 1u32, 0u32)
    };

    state.nzcv = (n << 31) | (z << 30) | (c << 29) | (v << 28);
}

fn execute_fcmp_d(state: &mut CpuState, rn: u8, rm: u8) {
    let lhs = f64::from_bits(read_v_scalar_f64(state, rn));
    let rhs = f64::from_bits(read_v_scalar_f64(state, rm));

    let (n, z, c, v) = if lhs.is_nan() || rhs.is_nan() {
        (0u32, 0u32, 1u32, 1u32)
    } else if lhs < rhs {
        (1u32, 0u32, 0u32, 0u32)
    } else if lhs > rhs {
        (0u32, 0u32, 1u32, 0u32)
    } else {
        (0u32, 1u32, 1u32, 0u32)
    };

    state.nzcv = (n << 31) | (z << 30) | (c << 29) | (v << 28);
}

fn execute_frintm_s(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f32::from_bits(read_v_scalar_f32(state, rn));
    let rounded = source.floor().to_bits();
    write_v_scalar_f32(state, rd, rounded);
}

fn execute_frintm_d(state: &mut CpuState, rd: u8, rn: u8) {
    let source = f64::from_bits(read_v_scalar_f64(state, rn));
    let rounded = source.floor().to_bits();
    write_v_scalar_f64(state, rd, rounded);
}

/// Local tag enum for FP vector three-same binary operations (replaces decoder::FpVecOp).
#[derive(Copy, Clone, PartialEq, Eq)]
enum FpVecOp {
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fmax,
    Fmin,
    Fmaxnm,
    Fminnm,
    Fmulx,
    Fabd,
    Faddp,
    Fmaxp,
    Fminp,
    Frecps,
    Frsqrts,
    Fcmeq,
    Fcmge,
    Fcmgt,
    Facge,
    Facgt,
    Fmla,
    Fmls,
}

/// Local tag enum for FP vector unary operations (replaces decoder::FpVecUnaryOp).
#[derive(Copy, Clone, PartialEq, Eq)]
enum FpVecUnaryOp {
    Frecpe,
    Frsqrte,
    Fabs,
    Fneg,
    Fsqrt,
    Fcvtns,
    Fcvtnu,
    Fcvtms,
    Fcvtmu,
    Fcvtas,
    Fcvtau,
    Fcvtps,
    Fcvtpu,
    Fcvtzs,
    Fcvtzu,
    Fcmeq0,
    Fcmge0,
    Fcmgt0,
    Fcmle0,
    Fcmlt0,
    Scvtf,
    Ucvtf,
    Frint32x,
    Frintn,
    Frintm,
    Frintp,
    Frintz,
    Frinta,
    Frintx,
    Frinti,
}

/// Local tag enum for FP vector by-element operations (replaces decoder::FpVecElemOp).
#[derive(Copy, Clone, PartialEq, Eq)]
enum FpVecElemOp {
    Fmul,
    Fmla,
    Fmls,
    Fmulx,
}

/// Execute AdvSIMD FP three-same (vector) operation.
fn execute_fp_vector_three_same(
    state: &mut CpuState,
    vd: u8,
    vn: u8,
    vm: u8,
    q: bool,
    sz: u8,
    op: FpVecOp,
) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);
    let (vd_lo, vd_hi) = state.read_v128(vd);

    let (res_lo, res_hi) = if sz == 0 {
        // F32 lanes
        let lane_count = if q { 4usize } else { 2 };
        let vn_f32: [f32; 4] = [
            f32::from_bits(vn_lo as u32),
            f32::from_bits((vn_lo >> 32) as u32),
            f32::from_bits(vn_hi as u32),
            f32::from_bits((vn_hi >> 32) as u32),
        ];
        let vm_f32: [f32; 4] = [
            f32::from_bits(vm_lo as u32),
            f32::from_bits((vm_lo >> 32) as u32),
            f32::from_bits(vm_hi as u32),
            f32::from_bits((vm_hi >> 32) as u32),
        ];
        let vd_f32: [f32; 4] = [
            f32::from_bits(vd_lo as u32),
            f32::from_bits((vd_lo >> 32) as u32),
            f32::from_bits(vd_hi as u32),
            f32::from_bits((vd_hi >> 32) as u32),
        ];

        let mut out = [0u32; 4];
        match op {
            FpVecOp::Fmla => {
                for i in 0..lane_count {
                    out[i] = (vd_f32[i] + vn_f32[i] * vm_f32[i]).to_bits();
                }
            }
            FpVecOp::Fmls => {
                for i in 0..lane_count {
                    out[i] = (vd_f32[i] - vn_f32[i] * vm_f32[i]).to_bits();
                }
            }
            FpVecOp::Faddp | FpVecOp::Fmaxp | FpVecOp::Fminp => {
                let mut concat = [0f32; 8];
                for i in 0..lane_count {
                    concat[i] = vn_f32[i];
                }
                for i in 0..lane_count {
                    concat[lane_count + i] = vm_f32[i];
                }
                for i in 0..lane_count {
                    let a = concat[2 * i];
                    let b = concat[2 * i + 1];
                    out[i] = match op {
                        FpVecOp::Faddp => (a + b).to_bits(),
                        FpVecOp::Fmaxp => a.max(b).to_bits(),
                        FpVecOp::Fminp => a.min(b).to_bits(),
                        _ => unreachable!(),
                    };
                }
            }
            _ => {
                for i in 0..lane_count {
                    out[i] = fp_vec_binop_f32(vn_f32[i], vm_f32[i], op);
                }
            }
        }
        let lo = (out[0] as u64) | ((out[1] as u64) << 32);
        let hi = if q {
            (out[2] as u64) | ((out[3] as u64) << 32)
        } else {
            0
        };
        (lo, hi)
    } else {
        // F64 lanes (Q must be true, lane_count = 2)
        let vn_f64 = [f64::from_bits(vn_lo), f64::from_bits(vn_hi)];
        let vm_f64 = [f64::from_bits(vm_lo), f64::from_bits(vm_hi)];
        let vd_f64 = [f64::from_bits(vd_lo), f64::from_bits(vd_hi)];

        let mut out = [0u64; 2];
        match op {
            FpVecOp::Fmla => {
                for i in 0..2 {
                    out[i] = (vd_f64[i] + vn_f64[i] * vm_f64[i]).to_bits();
                }
            }
            FpVecOp::Fmls => {
                for i in 0..2 {
                    out[i] = (vd_f64[i] - vn_f64[i] * vm_f64[i]).to_bits();
                }
            }
            FpVecOp::Faddp | FpVecOp::Fmaxp | FpVecOp::Fminp => {
                let pairs = [(vn_f64[0], vn_f64[1]), (vm_f64[0], vm_f64[1])];
                for i in 0..2 {
                    let (a, b) = pairs[i];
                    out[i] = match op {
                        FpVecOp::Faddp => (a + b).to_bits(),
                        FpVecOp::Fmaxp => a.max(b).to_bits(),
                        FpVecOp::Fminp => a.min(b).to_bits(),
                        _ => unreachable!(),
                    };
                }
            }
            _ => {
                for i in 0..2 {
                    out[i] = fp_vec_binop_f64(vn_f64[i], vm_f64[i], op);
                }
            }
        }
        (out[0], out[1])
    };

    state.write_v128(vd, res_lo, res_hi);
}

/// Execute AdvSIMD FP two-register miscellaneous (vector unary) operation.
fn execute_fp_vector_unary(
    state: &mut CpuState,
    vd: u8,
    vn: u8,
    q: bool,
    sz: u8,
    op: FpVecUnaryOp,
) {
    let (vn_lo, vn_hi) = state.read_v128(vn);

    let (res_lo, res_hi) = if sz == 0 {
        let lane_count = if q { 4usize } else { 2 };
        let vn_f32: [f32; 4] = [
            f32::from_bits(vn_lo as u32),
            f32::from_bits((vn_lo >> 32) as u32),
            f32::from_bits(vn_hi as u32),
            f32::from_bits((vn_hi >> 32) as u32),
        ];
        let mut out = [0u32; 4];
        for i in 0..lane_count {
            out[i] = fp_vec_unary_f32(vn_f32[i], op);
        }
        let lo = (out[0] as u64) | ((out[1] as u64) << 32);
        let hi = if q {
            (out[2] as u64) | ((out[3] as u64) << 32)
        } else {
            0
        };
        (lo, hi)
    } else {
        let vn_f64 = [f64::from_bits(vn_lo), f64::from_bits(vn_hi)];
        let mut out = [0u64; 2];
        for i in 0..2 {
            out[i] = fp_vec_unary_f64(vn_f64[i], op);
        }
        (out[0], out[1])
    };

    state.write_v128(vd, res_lo, res_hi);
}

/// Execute AdvSIMD FP vector × indexed element operation.
fn execute_fp_vector_by_element(
    state: &mut CpuState,
    vd: u8,
    vn: u8,
    vm: u8,
    index: u8,
    q: bool,
    sz: u8,
    op: FpVecElemOp,
) {
    let (vn_lo, vn_hi) = state.read_v128(vn);
    let (vm_lo, vm_hi) = state.read_v128(vm);

    let (res_lo, res_hi) = if sz == 0 {
        // f32 lanes
        let vn_f32 = [
            f32::from_bits(vn_lo as u32),
            f32::from_bits((vn_lo >> 32) as u32),
            f32::from_bits(vn_hi as u32),
            f32::from_bits((vn_hi >> 32) as u32),
        ];
        // Extract the indexed element from Vm
        let elem_bits = match index {
            0 => vm_lo as u32,
            1 => (vm_lo >> 32) as u32,
            2 => vm_hi as u32,
            3 => (vm_hi >> 32) as u32,
            _ => unreachable!(),
        };
        let elem = f32::from_bits(elem_bits);

        let lanes = if q { 4 } else { 2 };
        let mut out = [0u32; 4];

        // For FMLA/FMLS read accumulator
        let (vd_lo, vd_hi) = if matches!(op, FpVecElemOp::Fmla | FpVecElemOp::Fmls) {
            state.read_v128(vd)
        } else {
            (0u64, 0u64)
        };
        let vd_f32 = [
            f32::from_bits(vd_lo as u32),
            f32::from_bits((vd_lo >> 32) as u32),
            f32::from_bits(vd_hi as u32),
            f32::from_bits((vd_hi >> 32) as u32),
        ];

        for i in 0..lanes {
            out[i] = match op {
                FpVecElemOp::Fmul | FpVecElemOp::Fmulx => (vn_f32[i] * elem).to_bits(),
                FpVecElemOp::Fmla => (vd_f32[i] + vn_f32[i] * elem).to_bits(),
                FpVecElemOp::Fmls => (vd_f32[i] - vn_f32[i] * elem).to_bits(),
            };
        }
        let lo = (out[0] as u64) | ((out[1] as u64) << 32);
        let hi = if q {
            (out[2] as u64) | ((out[3] as u64) << 32)
        } else {
            0
        };
        (lo, hi)
    } else {
        // f64 lanes
        let vn_f64 = [f64::from_bits(vn_lo), f64::from_bits(vn_hi)];
        let elem = f64::from_bits(if index == 0 { vm_lo } else { vm_hi });

        let (vd_lo, vd_hi) = if matches!(op, FpVecElemOp::Fmla | FpVecElemOp::Fmls) {
            state.read_v128(vd)
        } else {
            (0u64, 0u64)
        };
        let vd_f64 = [f64::from_bits(vd_lo), f64::from_bits(vd_hi)];

        let mut out = [0u64; 2];
        for i in 0..2 {
            out[i] = match op {
                FpVecElemOp::Fmul | FpVecElemOp::Fmulx => (vn_f64[i] * elem).to_bits(),
                FpVecElemOp::Fmla => (vd_f64[i] + vn_f64[i] * elem).to_bits(),
                FpVecElemOp::Fmls => (vd_f64[i] - vn_f64[i] * elem).to_bits(),
            };
        }
        (out[0], out[1])
    };

    state.write_v128(vd, res_lo, res_hi);
}

fn fp_vec_unary_f32(a: f32, op: FpVecUnaryOp) -> u32 {
    let all_ones = 0xFFFF_FFFFu32;
    match op {
        FpVecUnaryOp::Frecpe => (1.0f32 / a).to_bits(),
        FpVecUnaryOp::Frsqrte => (1.0f32 / a.sqrt()).to_bits(),
        FpVecUnaryOp::Fabs => a.abs().to_bits(),
        FpVecUnaryOp::Fneg => (-a).to_bits(),
        FpVecUnaryOp::Fsqrt => a.sqrt().to_bits(),
        FpVecUnaryOp::Scvtf => {
            let ival = (a.to_bits() as i32) as f32;
            ival.to_bits()
        }
        FpVecUnaryOp::Ucvtf => {
            let ival = a.to_bits() as f32;
            ival.to_bits()
        }
        FpVecUnaryOp::Fcvtns => {
            // Round to nearest, ties to even → i32 → f32 bits reinterpreted
            let rounded = a.round_ties_even();
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            clamped as u32
        }
        FpVecUnaryOp::Fcvtnu => {
            let rounded = a.round_ties_even();
            let clamped = if rounded < 0.0 {
                0u32
            } else {
                rounded.min(u32::MAX as f32) as u32
            };
            clamped
        }
        FpVecUnaryOp::Fcvtms => {
            let rounded = a.floor();
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            clamped as u32
        }
        FpVecUnaryOp::Fcvtmu => {
            let rounded = a.floor();
            let clamped = if rounded < 0.0 {
                0u32
            } else {
                rounded.min(u32::MAX as f32) as u32
            };
            clamped
        }
        FpVecUnaryOp::Fcvtas => {
            let rounded = if a >= 0.0 {
                (a + 0.5).floor()
            } else {
                (a - 0.5).ceil()
            };
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            clamped as u32
        }
        FpVecUnaryOp::Fcvtau => {
            let rounded = (a + 0.5).floor();
            let clamped = if rounded < 0.0 {
                0u32
            } else {
                rounded.min(u32::MAX as f32) as u32
            };
            clamped
        }
        FpVecUnaryOp::Fcvtps => {
            let rounded = a.ceil();
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            clamped as u32
        }
        FpVecUnaryOp::Fcvtpu => {
            let rounded = a.ceil();
            let clamped = if rounded < 0.0 {
                0u32
            } else {
                rounded.min(u32::MAX as f32) as u32
            };
            clamped
        }
        FpVecUnaryOp::Fcvtzs => {
            let clamped = a.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            clamped as u32
        }
        FpVecUnaryOp::Fcvtzu => {
            let clamped = if a < 0.0 {
                0u32
            } else {
                a.min(u32::MAX as f32) as u32
            };
            clamped
        }
        FpVecUnaryOp::Fcmeq0 => {
            if a == 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmge0 => {
            if a >= 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmgt0 => {
            if a > 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmle0 => {
            if a <= 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmlt0 => {
            if a < 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Frintn => a.round_ties_even().to_bits(),
        FpVecUnaryOp::Frintm => a.floor().to_bits(),
        FpVecUnaryOp::Frintp => a.ceil().to_bits(),
        FpVecUnaryOp::Frintz => a.trunc().to_bits(),
        FpVecUnaryOp::Frinta => {
            if a >= 0.0 {
                (a + 0.5).floor().to_bits()
            } else {
                (a - 0.5).ceil().to_bits()
            }
        }
        FpVecUnaryOp::Frintx | FpVecUnaryOp::Frinti => a.round_ties_even().to_bits(),
        FpVecUnaryOp::Frint32x => a.round_ties_even().to_bits(),
    }
}

fn fp_vec_unary_f64(a: f64, op: FpVecUnaryOp) -> u64 {
    let all_ones = 0xFFFF_FFFF_FFFF_FFFFu64;
    match op {
        FpVecUnaryOp::Frecpe => (1.0f64 / a).to_bits(),
        FpVecUnaryOp::Frsqrte => (1.0f64 / a.sqrt()).to_bits(),
        FpVecUnaryOp::Fabs => a.abs().to_bits(),
        FpVecUnaryOp::Fneg => (-a).to_bits(),
        FpVecUnaryOp::Fsqrt => a.sqrt().to_bits(),
        FpVecUnaryOp::Scvtf => {
            let ival = (a.to_bits() as i64) as f64;
            ival.to_bits()
        }
        FpVecUnaryOp::Ucvtf => {
            let ival = a.to_bits() as f64;
            ival.to_bits()
        }
        FpVecUnaryOp::Fcvtns => {
            let rounded = a.round_ties_even();
            let clamped = rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            clamped as u64
        }
        FpVecUnaryOp::Fcvtnu => {
            let rounded = a.round_ties_even();
            let clamped = if rounded < 0.0 {
                0u64
            } else {
                rounded.min(u64::MAX as f64) as u64
            };
            clamped
        }
        FpVecUnaryOp::Fcvtms => {
            let rounded = a.floor();
            let clamped = rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            clamped as u64
        }
        FpVecUnaryOp::Fcvtmu => {
            let rounded = a.floor();
            let clamped = if rounded < 0.0 {
                0u64
            } else {
                rounded.min(u64::MAX as f64) as u64
            };
            clamped
        }
        FpVecUnaryOp::Fcvtas => {
            let rounded = if a >= 0.0 {
                (a + 0.5).floor()
            } else {
                (a - 0.5).ceil()
            };
            let clamped = rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            clamped as u64
        }
        FpVecUnaryOp::Fcvtau => {
            let rounded = (a + 0.5).floor();
            let clamped = if rounded < 0.0 {
                0u64
            } else {
                rounded.min(u64::MAX as f64) as u64
            };
            clamped
        }
        FpVecUnaryOp::Fcvtps => {
            let rounded = a.ceil();
            let clamped = rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            clamped as u64
        }
        FpVecUnaryOp::Fcvtpu => {
            let rounded = a.ceil();
            let clamped = if rounded < 0.0 {
                0u64
            } else {
                rounded.min(u64::MAX as f64) as u64
            };
            clamped
        }
        FpVecUnaryOp::Fcvtzs => {
            let clamped = a.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            clamped as u64
        }
        FpVecUnaryOp::Fcvtzu => {
            let clamped = if a < 0.0 {
                0u64
            } else {
                a.min(u64::MAX as f64) as u64
            };
            clamped
        }
        FpVecUnaryOp::Fcmeq0 => {
            if a == 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmge0 => {
            if a >= 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmgt0 => {
            if a > 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmle0 => {
            if a <= 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Fcmlt0 => {
            if a < 0.0 {
                all_ones
            } else {
                0
            }
        }
        FpVecUnaryOp::Frintn => a.round_ties_even().to_bits(),
        FpVecUnaryOp::Frintm => a.floor().to_bits(),
        FpVecUnaryOp::Frintp => a.ceil().to_bits(),
        FpVecUnaryOp::Frintz => a.trunc().to_bits(),
        FpVecUnaryOp::Frinta => {
            if a >= 0.0 {
                (a + 0.5).floor().to_bits()
            } else {
                (a - 0.5).ceil().to_bits()
            }
        }
        FpVecUnaryOp::Frintx | FpVecUnaryOp::Frinti => a.round_ties_even().to_bits(),
        FpVecUnaryOp::Frint32x => a.round_ties_even().to_bits(),
    }
}

fn fp_vec_binop_f32(a: f32, b: f32, op: FpVecOp) -> u32 {
    let all_ones = 0xFFFF_FFFFu32;
    match op {
        FpVecOp::Fadd => (a + b).to_bits(),
        FpVecOp::Fsub => (a - b).to_bits(),
        FpVecOp::Fmul => (a * b).to_bits(),
        FpVecOp::Fdiv => (a / b).to_bits(),
        FpVecOp::Fmax => a.max(b).to_bits(),
        FpVecOp::Fmin => a.min(b).to_bits(),
        FpVecOp::Fmaxnm => {
            if a.is_nan() && b.is_nan() {
                f32::NAN.to_bits()
            } else if a.is_nan() {
                b.to_bits()
            } else if b.is_nan() {
                a.to_bits()
            } else {
                a.max(b).to_bits()
            }
        }
        FpVecOp::Fminnm => {
            if a.is_nan() && b.is_nan() {
                f32::NAN.to_bits()
            } else if a.is_nan() {
                b.to_bits()
            } else if b.is_nan() {
                a.to_bits()
            } else {
                a.min(b).to_bits()
            }
        }
        FpVecOp::Fmulx => {
            // FMULX: like FMUL but 0 * Inf = +2.0
            if (a == 0.0 && b.is_infinite()) || (a.is_infinite() && b == 0.0) {
                2.0f32.to_bits()
            } else {
                (a * b).to_bits()
            }
        }
        FpVecOp::Fabd => (a - b).abs().to_bits(),
        FpVecOp::Frecps => (2.0 - a * b).to_bits(),
        FpVecOp::Frsqrts => ((3.0 - a * b) / 2.0).to_bits(),
        FpVecOp::Fcmeq => {
            if a == b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Fcmge => {
            if a >= b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Fcmgt => {
            if a > b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Facge => {
            if a.abs() >= b.abs() {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Facgt => {
            if a.abs() > b.abs() {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Faddp | FpVecOp::Fmaxp | FpVecOp::Fminp | FpVecOp::Fmla | FpVecOp::Fmls => {
            unreachable!()
        }
    }
}

fn fp_vec_binop_f64(a: f64, b: f64, op: FpVecOp) -> u64 {
    let all_ones = 0xFFFF_FFFF_FFFF_FFFFu64;
    match op {
        FpVecOp::Fadd => (a + b).to_bits(),
        FpVecOp::Fsub => (a - b).to_bits(),
        FpVecOp::Fmul => (a * b).to_bits(),
        FpVecOp::Fdiv => (a / b).to_bits(),
        FpVecOp::Fmax => a.max(b).to_bits(),
        FpVecOp::Fmin => a.min(b).to_bits(),
        FpVecOp::Fmaxnm => {
            if a.is_nan() && b.is_nan() {
                f64::NAN.to_bits()
            } else if a.is_nan() {
                b.to_bits()
            } else if b.is_nan() {
                a.to_bits()
            } else {
                a.max(b).to_bits()
            }
        }
        FpVecOp::Fminnm => {
            if a.is_nan() && b.is_nan() {
                f64::NAN.to_bits()
            } else if a.is_nan() {
                b.to_bits()
            } else if b.is_nan() {
                a.to_bits()
            } else {
                a.min(b).to_bits()
            }
        }
        FpVecOp::Fmulx => {
            if (a == 0.0 && b.is_infinite()) || (a.is_infinite() && b == 0.0) {
                2.0f64.to_bits()
            } else {
                (a * b).to_bits()
            }
        }
        FpVecOp::Fabd => (a - b).abs().to_bits(),
        FpVecOp::Frecps => (2.0 - a * b).to_bits(),
        FpVecOp::Frsqrts => ((3.0 - a * b) / 2.0).to_bits(),
        FpVecOp::Fcmeq => {
            if a == b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Fcmge => {
            if a >= b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Fcmgt => {
            if a > b {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Facge => {
            if a.abs() >= b.abs() {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Facgt => {
            if a.abs() > b.abs() {
                all_ones
            } else {
                0
            }
        }
        FpVecOp::Faddp | FpVecOp::Fmaxp | FpVecOp::Fminp | FpVecOp::Fmla | FpVecOp::Fmls => {
            unreachable!()
        }
    }
}

#[derive(Copy, Clone)]
enum SimdWidenOp {
    Saddl,
    Uaddl,
    Saddw,
    Uaddw,
    Ssubl,
    Usubl,
    Ssubw,
    Usubw,
    Sabdl,
    Uabdl,
    Smlal,
    Umlal,
    Smlsl,
    Umlsl,
    Smull,
    Umull,
}

/// AdvSIMD three-different (widening) operations.
/// Q=false operates on lower halves, Q=true on upper halves (xxx2 variant).
/// Size determines element width: 0=8→16, 1=16→32, 2=32→64.
fn execute_simd_three_diff(
    state: &mut CpuState,
    q: bool,
    op: SimdWidenOp,
    size: u8,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    let (vn_lo, vn_hi) = state.read_v128(rn);
    let (vm_lo, vm_hi) = state.read_v128(rm);
    let (vd_lo, vd_hi) = state.read_v128(rd);

    // Source half: Q=0 uses low 64 bits, Q=1 uses high 64 bits
    let src_n = if q { vn_hi } else { vn_lo };
    let src_m = if q { vm_hi } else { vm_lo };

    let elem_bits = 8u32 << size;
    let elem_count = 64 / elem_bits;
    let wide_bits = elem_bits * 2;

    // Read source elements (narrow)
    let read_elem = |src: u64, idx: u32| -> u64 {
        let shift = idx * elem_bits;
        (src >> shift) & ((1u64 << elem_bits) - 1)
    };

    // Read wide element from a 128-bit register pair
    let read_wide = |lo: u64, hi: u64, idx: u32| -> u64 {
        let shift = idx * wide_bits;
        if shift < 64 {
            (lo >> shift) & ((1u64 << wide_bits) - 1)
        } else {
            (hi >> (shift - 64)) & ((1u64 << wide_bits) - 1)
        }
    };

    // Sign-extend for signed operations
    let sign_extend = |val: u64, bits: u32| -> i64 {
        let shift = 64 - bits;
        ((val as i64) << shift) >> shift
    };

    let mut result_lo = 0u64;
    let mut result_hi = 0u64;

    for i in 0..elem_count {
        let a = read_elem(src_n, i);
        let b = read_elem(src_m, i);

        let wide_result: u64 = match op {
            SimdWidenOp::Saddl => {
                (sign_extend(a, elem_bits).wrapping_add(sign_extend(b, elem_bits))) as u64
            }
            SimdWidenOp::Uaddl => a.wrapping_add(b),
            SimdWidenOp::Ssubl => {
                (sign_extend(a, elem_bits).wrapping_sub(sign_extend(b, elem_bits))) as u64
            }
            SimdWidenOp::Usubl => a.wrapping_sub(b),
            SimdWidenOp::Sabdl => {
                (sign_extend(a, elem_bits) - sign_extend(b, elem_bits)).unsigned_abs()
            }
            SimdWidenOp::Uabdl => {
                if a >= b {
                    a - b
                } else {
                    b - a
                }
            }
            SimdWidenOp::Smlal => {
                let acc = read_wide(vd_lo, vd_hi, i);
                let product =
                    (sign_extend(a, elem_bits).wrapping_mul(sign_extend(b, elem_bits))) as u64;
                acc.wrapping_add(product)
            }
            SimdWidenOp::Umlal => {
                let acc = read_wide(vd_lo, vd_hi, i);
                acc.wrapping_add(a.wrapping_mul(b))
            }
            SimdWidenOp::Smlsl => {
                let acc = read_wide(vd_lo, vd_hi, i);
                let product =
                    (sign_extend(a, elem_bits).wrapping_mul(sign_extend(b, elem_bits))) as u64;
                acc.wrapping_sub(product)
            }
            SimdWidenOp::Umlsl => {
                let acc = read_wide(vd_lo, vd_hi, i);
                acc.wrapping_sub(a.wrapping_mul(b))
            }
            SimdWidenOp::Smull => {
                (sign_extend(a, elem_bits).wrapping_mul(sign_extend(b, elem_bits))) as u64
            }
            SimdWidenOp::Umull => a.wrapping_mul(b),
            // SADDW / UADDW: Vn is wide (128-bit), Vm is narrow
            SimdWidenOp::Saddw => {
                let wide_n = read_wide(vn_lo, vn_hi, i);
                (sign_extend(wide_n, wide_bits).wrapping_add(sign_extend(b, elem_bits))) as u64
            }
            SimdWidenOp::Uaddw => {
                let wide_n = read_wide(vn_lo, vn_hi, i);
                wide_n.wrapping_add(b)
            }
            // SSUBW / USUBW: Vn is wide (128-bit), Vm is narrow
            SimdWidenOp::Ssubw => {
                let wide_n = read_wide(vn_lo, vn_hi, i);
                (sign_extend(wide_n, wide_bits).wrapping_sub(sign_extend(b, elem_bits))) as u64
            }
            SimdWidenOp::Usubw => {
                let wide_n = read_wide(vn_lo, vn_hi, i);
                wide_n.wrapping_sub(b)
            }
        };

        let masked = wide_result & ((1u128 << wide_bits) - 1) as u64;
        let bit_offset = i * wide_bits;
        if bit_offset < 64 {
            result_lo |= masked << bit_offset;
        } else {
            result_hi |= masked << (bit_offset - 64);
        }
    }

    state.write_v128(rd, result_lo, result_hi);
}

/// AdvSIMD shift-by-immediate (widening): SSHLL / USHLL
/// Takes narrow elements from the source, sign/zero extends to double width,
/// then shifts left.
/// Q=false: source from lower 64 bits.  Q=true: source from upper 64 bits.
fn execute_simd_shift_long(
    state: &mut CpuState,
    q: bool,
    signed: bool,
    size: u8,
    shift: u8,
    rd: u8,
    rn: u8,
) {
    let (vn_lo, vn_hi) = state.read_v128(rn);
    let src = if q { vn_hi } else { vn_lo };

    let elem_bits = 8u32 << size;
    let elem_count = 64 / elem_bits;
    let wide_bits = elem_bits * 2;
    let elem_mask = if elem_bits == 64 {
        u64::MAX
    } else {
        (1u64 << elem_bits) - 1
    };
    let wide_mask = if wide_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << wide_bits) - 1
    };

    let mut result_lo = 0u64;
    let mut result_hi = 0u64;

    for i in 0..elem_count {
        let raw = (src >> (i * elem_bits)) & elem_mask;
        let widened = if signed {
            let sext = ((raw as i64) << (64 - elem_bits)) >> (64 - elem_bits);
            (sext as u64) & wide_mask
        } else {
            raw
        };
        let shifted = (widened << shift) & wide_mask;
        let bit_offset = i * wide_bits;
        if bit_offset < 64 {
            result_lo |= shifted << bit_offset;
        } else {
            result_hi |= shifted << (bit_offset - 64);
        }
    }

    state.write_v128(rd, result_lo, result_hi);
}

#[derive(Copy, Clone)]
enum SimdShiftOp {
    Sshr,
    Ushr,
    Shl,
}

/// AdvSIMD shift by immediate: SSHR / USHR / SHL
fn execute_simd_shift_imm(
    state: &mut CpuState,
    q: bool,
    op: SimdShiftOp,
    size: u8,
    shift: u8,
    rd: u8,
    rn: u8,
) {
    let (vn_lo, vn_hi) = state.read_v128(rn);
    let elem_bits = 8u32 << size;
    let elem_mask = if elem_bits == 64 {
        u64::MAX
    } else {
        (1u64 << elem_bits) - 1
    };

    // Q=false: 64 bits (low), Q=true: 128 bits
    let total_bits = if q { 128u32 } else { 64 };
    let elem_count = total_bits / elem_bits;

    let mut result_lo = 0u64;
    let mut result_hi = 0u64;

    for i in 0..elem_count {
        let bit_offset = i * elem_bits;
        let raw = if bit_offset < 64 {
            (vn_lo >> bit_offset) & elem_mask
        } else {
            (vn_hi >> (bit_offset - 64)) & elem_mask
        };

        let result = match op {
            SimdShiftOp::Sshr => {
                let sext = ((raw as i64) << (64 - elem_bits)) >> (64 - elem_bits);
                ((sext >> shift) as u64) & elem_mask
            }
            SimdShiftOp::Ushr => (raw >> shift) & elem_mask,
            SimdShiftOp::Shl => (raw << shift) & elem_mask,
        };

        if bit_offset < 64 {
            result_lo |= result << bit_offset;
        } else {
            result_hi |= result << (bit_offset - 64);
        }
    }

    // Q=false: zero upper 64 bits
    if !q {
        result_hi = 0;
    }

    state.write_v128(rd, result_lo, result_hi);
}

/// FP to integer conversion with rounding mode.
/// rmode: 0=nearest (ties to even), 1=+inf, 2=-inf, 4=nearest (ties away from zero)
fn execute_fcvt_round_from_float(
    state: &mut CpuState,
    rd: u8,
    rn: u8,
    sf: bool,
    ftype: u8,
    rmode: u8,
    is_unsigned: bool,
) {
    // Read the FP source value as f64 regardless of precision.
    let fp_val: f64 = match ftype {
        0 => f32::from_bits(read_v_scalar_f32(state, rn)) as f64,
        1 => f64::from_bits(read_v_scalar_f64(state, rn)),
        _ => 0.0,
    };

    // Apply rounding
    let rounded = match rmode {
        0 => {
            // Round to nearest, ties to even (banker's rounding)
            let r = fp_val.round();
            // Check for tie: if fractional part is exactly 0.5, round to even
            if (fp_val - fp_val.floor()).abs() == 0.5 {
                let fl = fp_val.floor();
                let ce = fp_val.ceil();
                if (fl as i64) % 2 == 0 {
                    fl
                } else {
                    ce
                }
            } else {
                r
            }
        }
        1 => fp_val.ceil(),  // Round toward +infinity
        2 => fp_val.floor(), // Round toward -infinity
        4 => fp_val.round(), // Round to nearest, ties away from zero
        _ => fp_val.trunc(), // Fallback
    };

    // Convert to integer and write to destination register
    if sf {
        // 64-bit destination
        let result = if is_unsigned {
            if rounded.is_nan() || rounded < 0.0 {
                0u64
            } else if rounded > u64::MAX as f64 {
                u64::MAX
            } else {
                rounded as u64
            }
        } else {
            let v = if rounded.is_nan() {
                0i64
            } else if rounded < i64::MIN as f64 {
                i64::MIN
            } else if rounded > i64::MAX as f64 {
                i64::MAX
            } else {
                rounded as i64
            };
            v as u64
        };
        state.write_x(rd, result);
    } else {
        // 32-bit destination
        let result = if is_unsigned {
            if rounded.is_nan() || rounded < 0.0 {
                0u64
            } else if rounded > u32::MAX as f64 {
                u32::MAX as u64
            } else {
                rounded as u32 as u64
            }
        } else {
            let v = if rounded.is_nan() {
                0i32
            } else if rounded < i32::MIN as f64 {
                i32::MIN
            } else if rounded > i32::MAX as f64 {
                i32::MAX
            } else {
                rounded as i32
            };
            v as u32 as u64
        };
        state.write_x(rd, result);
    }
}

#[derive(Copy, Clone)]
enum SimdNarrowOp {
    Addhn,
    Raddhn,
    Subhn,
    Rsubhn,
}

/// AdvSIMD three-different (narrowing) operations: ADDHN/RADDHN/SUBHN/RSUBHN.
/// Reads wide elements from Vn and Vm, produces narrow result.
/// Q=false: writes lower half of Vd, zeros upper.
/// Q=true: preserves lower half of Vd, writes to upper half (xxxHN2).
fn execute_simd_narrow(
    state: &mut CpuState,
    q: bool,
    op: SimdNarrowOp,
    size: u8,
    rd: u8,
    rn: u8,
    rm: u8,
) {
    let (vn_lo, vn_hi) = state.read_v128(rn);
    let (vm_lo, vm_hi) = state.read_v128(rm);
    let (vd_lo, vd_hi) = state.read_v128(rd);

    // Source elements are wide (size determines source width: 0→16bit, 1→32bit, 2→64bit)
    let wide_bits = 16u32 << size; // source element width
    let narrow_bits = wide_bits / 2; // result element width
    let elem_count = 128 / wide_bits; // number of wide elements across 128 bits
    let round_const: u64 = 1u64 << (narrow_bits - 1);

    let read_wide = |lo: u64, hi: u64, idx: u32| -> u64 {
        let shift = idx * wide_bits;
        if wide_bits == 64 {
            if idx == 0 {
                lo
            } else {
                hi
            }
        } else if shift < 64 {
            (lo >> shift) & ((1u64 << wide_bits) - 1)
        } else {
            (hi >> (shift - 64)) & ((1u64 << wide_bits) - 1)
        }
    };

    let mut result_half = 0u64;
    for i in 0..elem_count {
        let a = read_wide(vn_lo, vn_hi, i);
        let b = read_wide(vm_lo, vm_hi, i);

        let wide_result = match op {
            SimdNarrowOp::Addhn => a.wrapping_add(b),
            SimdNarrowOp::Raddhn => a.wrapping_add(b).wrapping_add(round_const),
            SimdNarrowOp::Subhn => a.wrapping_sub(b),
            SimdNarrowOp::Rsubhn => a.wrapping_sub(b).wrapping_add(round_const),
        };

        // Take the high narrow_bits of the wide result
        let narrow_val = (wide_result >> narrow_bits) & ((1u64 << narrow_bits) - 1);
        result_half |= narrow_val << (i * narrow_bits);
    }

    if q {
        // xxxHN2: result goes to upper half, lower preserved
        state.write_v128(rd, vd_lo, result_half);
    } else {
        // xxxHN: result goes to lower half, upper zeroed
        state.write_v128(rd, result_half, 0);
    }
}

fn expand_fmov_imm8_f32(imm8: u8) -> u32 {
    let sign = u32::from((imm8 >> 7) & 1);
    let exp_group = (imm8 >> 4) & 0x3;
    let high_exp = if ((imm8 >> 6) & 1) != 0 { 124 } else { 128 };
    let exponent = u32::from(high_exp + exp_group);
    let mantissa = u32::from(imm8 & 0x0f) << 19;
    (sign << 31) | (exponent << 23) | mantissa
}

fn expand_fmov_imm8_f64(imm8: u8) -> u64 {
    let sign = u64::from((imm8 >> 7) & 1);
    let exp_group = u16::from((imm8 >> 4) & 0x3);
    let high_exp: u16 = if ((imm8 >> 6) & 1) != 0 { 1020 } else { 1024 };
    let exponent = u64::from(high_exp + exp_group);
    let mantissa = u64::from(imm8 & 0x0f) << 48;
    (sign << 63) | (exponent << 52) | mantissa
}

fn read_v_scalar_f32(state: &CpuState, reg: u8) -> u32 {
    let (lo, _) = state.read_v128(reg);
    lo as u32
}

fn read_v_scalar_f64(state: &CpuState, reg: u8) -> u64 {
    let (lo, _) = state.read_v128(reg);
    lo
}

fn write_v_scalar_f32(state: &mut CpuState, reg: u8, bits: u32) {
    state.write_v128(reg, u64::from(bits), 0);
}

fn write_v_scalar_f64(state: &mut CpuState, reg: u8, bits: u64) {
    state.write_v128(reg, bits, 0);
}

fn add_signed_offset(base: u64, offset: i64) -> u64 {
    if offset >= 0 {
        base.wrapping_add(offset as u64)
    } else {
        base.wrapping_sub((-offset) as u64)
    }
}

fn resolve_ls_reg_offset(
    state: &CpuState,
    rm: u8,
    option: u8,
    shift: bool,
    scale_shift: u8,
) -> Option<u64> {
    let raw = state.read_x(rm);
    let base = match option {
        0b010 => u64::from(raw as u32),               // UXTW
        0b011 => raw,                                 // LSL
        0b110 => ((raw as u32 as i32) as i64) as u64, // SXTW
        0b111 => raw,                                 // SXTX
        _ => return None,
    };

    Some(if shift {
        base.wrapping_shl(u32::from(scale_shift))
    } else {
        base
    })
}

fn pair_address(base: u64, offset: i64, mode: PairAddressMode) -> (u64, Option<u64>) {
    match mode {
        PairAddressMode::SignedOffset => (add_signed_offset(base, offset), None),
        PairAddressMode::PreIndex => {
            let updated = add_signed_offset(base, offset);
            (updated, Some(updated))
        }
        PairAddressMode::PostIndex => (base, Some(add_signed_offset(base, offset))),
    }
}

fn apply_shift(value: u64, shift: ShiftType, amount: u8) -> u64 {
    if amount == 0 {
        return value;
    }

    match shift {
        ShiftType::Lsl => value.wrapping_shl(u32::from(amount)),
        ShiftType::Lsr => value.wrapping_shr(u32::from(amount)),
        ShiftType::Asr => ((value as i64) >> amount) as u64,
        ShiftType::Ror => value.rotate_right(u32::from(amount)),
    }
}

fn apply_extend(value: u64, extend: ExtendType, amount: u8) -> u64 {
    let extended = match extend {
        ExtendType::Uxtb => u64::from(value as u8),
        ExtendType::Uxth => u64::from(value as u16),
        ExtendType::Uxtw => u64::from(value as u32),
        ExtendType::Uxtx => value,
        ExtendType::Sxtb => (value as i8) as i64 as u64,
        ExtendType::Sxth => (value as i16) as i64 as u64,
        ExtendType::Sxtw => (value as i32) as i64 as u64,
        ExtendType::Sxtx => value,
    };

    if amount == 0 {
        extended
    } else {
        extended.wrapping_shl(u32::from(amount))
    }
}

fn ubfm64(value: u64, immr: u8, imms: u8) -> u64 {
    if immr <= imms {
        let width = u32::from(imms - immr) + 1;
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        (value >> u32::from(immr)) & mask
    } else {
        let width = u32::from(imms) + 1;
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let lsb = 64u32 - u32::from(immr);
        (value & mask).wrapping_shl(lsb)
    }
}

fn ubfm32(value: u32, immr: u8, imms: u8) -> u32 {
    if immr <= imms {
        let width = u32::from(imms - immr) + 1;
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        (value >> u32::from(immr)) & mask
    } else {
        let width = u32::from(imms) + 1;
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let lsb = 32u32 - u32::from(immr);
        (value & mask).wrapping_shl(lsb)
    }
}

fn sbfm64(value: u64, immr: u8, imms: u8) -> u64 {
    let result = ubfm64(value, immr, imms);
    let sign_bit = if immr <= imms {
        u32::from(imms - immr)
    } else {
        (64u32 - u32::from(immr)) + u32::from(imms)
    };
    sign_extend_u64(result, sign_bit)
}

fn sbfm32(value: u32, immr: u8, imms: u8) -> u32 {
    let result = ubfm32(value, immr, imms);
    let sign_bit = if immr <= imms {
        u32::from(imms - immr)
    } else {
        (32u32 - u32::from(immr)) + u32::from(imms)
    };
    sign_extend_u32(result, sign_bit)
}

fn sign_extend_u64(value: u64, sign_bit: u32) -> u64 {
    if sign_bit >= 63 {
        return value;
    }

    let sign_mask = 1u64 << sign_bit;
    if value & sign_mask == 0 {
        value
    } else {
        value | (!0u64 << (sign_bit + 1))
    }
}

fn sign_extend_u32(value: u32, sign_bit: u32) -> u32 {
    if sign_bit >= 31 {
        return value;
    }

    let sign_mask = 1u32 << sign_bit;
    if value & sign_mask == 0 {
        value
    } else {
        value | (!0u32 << (sign_bit + 1))
    }
}

fn bfm64(dst: u64, src: u64, immr: u8, imms: u8) -> u64 {
    if immr <= imms {
        let width = u32::from(imms - immr) + 1;
        let low_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let insert_mask = low_mask.wrapping_shl(u32::from(immr));
        let insert_value = (src & low_mask).wrapping_shl(u32::from(immr));
        (dst & !insert_mask) | (insert_value & insert_mask)
    } else {
        let width = u32::from(imms) + 1;
        let low_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let lsb = 64u32 - u32::from(immr);
        let insert_mask = low_mask.wrapping_shl(lsb);
        let insert_value = (src & low_mask).wrapping_shl(lsb);
        (dst & !insert_mask) | (insert_value & insert_mask)
    }
}

fn bfm32(dst: u32, src: u32, immr: u8, imms: u8) -> u32 {
    if immr <= imms {
        let width = u32::from(imms - immr) + 1;
        let low_mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let insert_mask = low_mask.wrapping_shl(u32::from(immr));
        let insert_value = (src & low_mask).wrapping_shl(u32::from(immr));
        (dst & !insert_mask) | (insert_value & insert_mask)
    } else {
        let width = u32::from(imms) + 1;
        let low_mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let lsb = 32u32 - u32::from(immr);
        let insert_mask = low_mask.wrapping_shl(lsb);
        let insert_value = (src & low_mask).wrapping_shl(lsb);
        (dst & !insert_mask) | (insert_value & insert_mask)
    }
}

fn apply_shift32(value: u32, shift: ShiftType, amount: u8) -> u32 {
    if amount == 0 {
        return value;
    }

    match shift {
        ShiftType::Lsl => value.wrapping_shl(u32::from(amount)),
        ShiftType::Lsr => value.wrapping_shr(u32::from(amount)),
        ShiftType::Asr => ((value as i32) >> amount) as u32,
        ShiftType::Ror => value.rotate_right(u32::from(amount)),
    }
}

fn apply_extend32(value: u64, extend: ExtendType, amount: u8) -> u32 {
    let extended = match extend {
        ExtendType::Uxtb => u32::from(value as u8),
        ExtendType::Uxth => u32::from(value as u16),
        ExtendType::Uxtw | ExtendType::Uxtx => value as u32,
        ExtendType::Sxtb => (value as i8) as i32 as u32,
        ExtendType::Sxth => (value as i16) as i32 as u32,
        ExtendType::Sxtw | ExtendType::Sxtx => value as u32,
    };

    if amount == 0 {
        extended
    } else {
        extended.wrapping_shl(u32::from(amount))
    }
}

fn mark_decode_error(state: &mut CpuState, err: DecodeError) {
    match err {
        DecodeError::MissingInstruction { .. } => state.set_memory_fault(),
        DecodeError::UnsupportedOpcode { .. } => state.set_decode_fault(),
    }
}

#[cfg(test)]
#[allow(unused_mut)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{
        classify_bound_signal_fault, return_to_dispatcher_from_signal, CachedBlock, DynarecEngine,
        DynarecOptions, FastmemRegion, MemoryInterface, SignalFaultRecord, StepResult,
        TLS_ACTIVE_DISPATCH_FRAME,
    };
    use crate::decoder::SystemRegister;
    use crate::memory::{rw_flags, FaultClass, MemoryAccessKind, PteRegion, PteTable};
    use crate::runtime::{CpuState, ExceptionKind, ExceptionLevel, ExitReason};

    const BASE: u64 = 0x8000_0000;

    #[derive(Default)]
    struct TestMemory {
        inst: HashMap<u64, u32>,
        data: HashMap<u64, u64>,
    }

    impl MemoryInterface for TestMemory {
        fn fetch_u32(&mut self, address: u64) -> Option<u32> {
            self.inst.get(&address).copied()
        }

        fn read_u8(&mut self, address: u64) -> Option<u8> {
            self.data.get(&address).copied().map(|v| v as u8)
        }

        fn write_u8(&mut self, address: u64, value: u8) -> bool {
            self.data.insert(address, u64::from(value));
            true
        }

        fn read_u16(&mut self, address: u64) -> Option<u16> {
            self.data.get(&address).copied().map(|v| v as u16)
        }

        fn write_u16(&mut self, address: u64, value: u16) -> bool {
            self.data.insert(address, u64::from(value));
            true
        }

        fn read_u32(&mut self, address: u64) -> Option<u32> {
            self.data.get(&address).copied().map(|v| v as u32)
        }

        fn write_u32(&mut self, address: u64, value: u32) -> bool {
            self.data.insert(address, u64::from(value));
            true
        }

        fn read_u64(&mut self, address: u64) -> Option<u64> {
            self.data.get(&address).copied()
        }

        fn write_u64(&mut self, address: u64, value: u64) -> bool {
            self.data.insert(address, value);
            true
        }
    }

    struct FastmemOnlyMemory {
        inst: HashMap<u64, u32>,
        base: u64,
        bytes: Vec<u8>,
    }

    impl FastmemOnlyMemory {
        fn new(base: u64, size: usize) -> Self {
            Self {
                inst: HashMap::new(),
                base,
                bytes: vec![0; size],
            }
        }
    }

    impl MemoryInterface for FastmemOnlyMemory {
        fn fetch_u32(&mut self, address: u64) -> Option<u32> {
            self.inst.get(&address).copied()
        }

        fn fastmem_region(&mut self) -> Option<FastmemRegion> {
            Some(FastmemRegion {
                guest_base: self.base,
                host_ptr: self.bytes.as_mut_ptr(),
                size: self.bytes.len() as u64,
            })
        }

        fn read_u64(&mut self, _address: u64) -> Option<u64> {
            panic!("slow-path read_u64 should not be used")
        }

        fn write_u64(&mut self, _address: u64, _value: u64) -> bool {
            panic!("slow-path write_u64 should not be used")
        }
    }

    #[derive(Default)]
    struct FaultingReadMemory {
        inst: HashMap<u64, u32>,
    }

    impl MemoryInterface for FaultingReadMemory {
        fn fetch_u32(&mut self, address: u64) -> Option<u32> {
            self.inst.get(&address).copied()
        }

        fn read_u64(&mut self, _address: u64) -> Option<u64> {
            None
        }
    }

    struct ClassifyingMemory {
        inst: HashMap<u64, u32>,
        guest_base: u64,
        bytes: Vec<u8>,
        pte_table: PteTable,
    }

    impl ClassifyingMemory {
        fn new(guest_base: u64, size: usize) -> Self {
            let pte_table = PteTable::new(guest_base, size as u64);
            Self {
                inst: HashMap::new(),
                guest_base,
                bytes: vec![0; size],
                pte_table,
            }
        }

        fn allow_rw(&mut self) {
            let _ = self.pte_table.map_range(
                self.guest_base,
                self.bytes.as_mut_ptr(),
                self.bytes.len() as u64,
                rw_flags(),
            );
        }

        fn fault_record(&mut self, guest_offset: usize, is_write: bool) -> SignalFaultRecord {
            let fastmem = self.fastmem_region().expect("fastmem region should exist");
            SignalFaultRecord {
                fault_address: fastmem.host_ptr as usize + guest_offset,
                is_write,
                kind: FaultClass::MemoryFault,
            }
        }
    }

    struct PteOnlyMemory {
        inst: HashMap<u64, u32>,
        guest_base: u64,
        bytes: Vec<u8>,
        pte_table: PteTable,
    }

    impl PteOnlyMemory {
        fn new(guest_base: u64, size: usize) -> Self {
            let mut bytes = vec![0; size];
            let pte_table = PteTable::new(guest_base, size as u64);
            let _ = pte_table.map_range(guest_base, bytes.as_mut_ptr(), size as u64, rw_flags());
            Self {
                inst: HashMap::new(),
                guest_base,
                bytes,
                pte_table,
            }
        }
    }

    impl MemoryInterface for PteOnlyMemory {
        fn fetch_u32(&mut self, address: u64) -> Option<u32> {
            self.inst.get(&address).copied()
        }

        fn pte_region(&mut self) -> Option<PteRegion> {
            Some(self.pte_table.region())
        }

        fn classify_guest_access(&mut self, address: u64, kind: MemoryAccessKind) -> FaultClass {
            self.pte_table.region().classify_access(address, kind)
        }

        fn read_u64(&mut self, address: u64) -> Option<u64> {
            let offset = address.checked_sub(self.guest_base)? as usize;
            let bytes = self.bytes.get(offset..offset + 8)?;
            Some(u64::from_le_bytes(bytes.try_into().ok()?))
        }
    }

    impl MemoryInterface for ClassifyingMemory {
        fn fetch_u32(&mut self, address: u64) -> Option<u32> {
            self.inst.get(&address).copied()
        }

        fn fastmem_region(&mut self) -> Option<FastmemRegion> {
            Some(FastmemRegion {
                guest_base: self.guest_base,
                host_ptr: self.bytes.as_mut_ptr(),
                size: self.bytes.len() as u64,
            })
        }

        fn pte_region(&mut self) -> Option<PteRegion> {
            Some(self.pte_table.region())
        }

        fn classify_guest_access(&mut self, address: u64, kind: MemoryAccessKind) -> FaultClass {
            self.pte_table.region().classify_access(address, kind)
        }
    }

    fn encode_mrs(rt: u8, reg: SystemRegister) -> u32 {
        0xd530_0000 | (u32::from(reg.encoding()) << 5) | u32::from(rt)
    }

    fn encode_msr(reg: SystemRegister, rt: u8) -> u32 {
        0xd510_0000 | (u32::from(reg.encoding()) << 5) | u32::from(rt)
    }

    fn encode_ccmp_reg(sf: bool, rn: u8, rm: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xfa40_0000 } else { 0x7a40_0000 };
        base | (u32::from(rm & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmp_imm(sf: bool, rn: u8, imm5: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xfa40_0800 } else { 0x7a40_0800 };
        base | (u32::from(imm5 & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmn_reg(sf: bool, rn: u8, rm: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xba40_0000 } else { 0x3a40_0000 };
        base | (u32::from(rm & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmn_imm(sf: bool, rn: u8, imm5: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xba40_0800 } else { 0x3a40_0800 };
        base | (u32::from(imm5 & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    #[test]
    fn executes_cbnz_loop_until_breakpoint() {
        let words = [
            0xd280_0060, // movz x0, #3
            0xf100_0400, // subs x0, x0, #1
            0xb5ff_ffe0, // cbnz x0, -4
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 0);
        assert_eq!(state.pc, BASE + 16);
    }

    #[test]
    fn supports_parallel_execution_with_shared_dynarec() {
        let words = Arc::new([
            0xd280_0060, // movz x0, #3
            0xf100_0400, // subs x0, x0, #1
            0xb5ff_ffe0, // cbnz x0, -4
            0xd420_0020, // brk #1
        ]);

        let engine = Arc::new(
            DynarecEngine::new(DynarecOptions {
                force_interpreter: false,
                block_max_instructions: 16,
            })
            .expect("dynarec engine should initialize"),
        );

        let mut handles = Vec::new();

        for _ in 0..4 {
            let engine = Arc::clone(&engine);
            let words = Arc::clone(&words);

            handles.push(std::thread::spawn(move || {
                let mut state = CpuState::default();
                state.pc = BASE;

                let mut fetch = |pc: u64| {
                    let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
                    words.get(idx).copied()
                };

                for _ in 0..16 {
                    match engine
                        .step(&mut state, &mut fetch)
                        .expect("step should succeed")
                    {
                        StepResult::Advanced { .. } => continue,
                        StepResult::Trap { reason, .. } => return (reason, state.read_x(0)),
                    }
                }

                panic!("expected breakpoint trap");
            }));
        }

        for handle in handles {
            let (reason, x0) = handle.join().expect("thread should finish successfully");
            assert_eq!(reason, ExitReason::Breakpoint(1));
            assert_eq!(x0, 0);
        }
    }

    #[test]
    fn late_linking_compiles_target_block_on_first_edge() {
        let words = [
            0x1400_0002, // b +8
            0xd503_201f, // nop
            0xd280_0020, // movz x0, #1
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let first = engine
            .step(&mut state, &mut fetch)
            .expect("first step should succeed");
        match first {
            StepResult::Advanced { start_pc, end_pc } => {
                assert_eq!(start_pc, BASE);
                assert_eq!(end_pc, BASE + 8);
            }
            StepResult::Trap { reason, .. } => {
                panic!("expected advanced result on first edge, got trap {reason:?}")
            }
        }

        let links = engine
            .block_return_links(BASE)
            .expect("source block should expose return-link entries");
        let dispatch_target = engine
            .block_dispatch_target(BASE)
            .expect("source block should expose dispatcher target");

        let mut found_edge = false;
        for (expected_next_pc, linked_target) in links {
            if expected_next_pc == BASE + 8 {
                assert_ne!(linked_target, 0);
                assert_ne!(linked_target, dispatch_target);
                found_edge = true;
                break;
            }
        }

        assert!(found_edge, "expected late-link entry for target block edge");
    }

    #[test]
    fn eager_linking_chains_precompiled_direct_branch_on_first_execution() {
        let words = [
            0x1400_0002, // b +8
            0xd503_201f, // nop padding
            0xd420_0020, // brk #1
        ];

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        // Compile target block first so source compilation can eagerly chain to it.
        let mut target_state = CpuState::default();
        target_state.pc = BASE + 8;
        let target_result = engine
            .step(&mut target_state, &mut fetch)
            .expect("target step should succeed");
        assert!(matches!(
            target_result,
            StepResult::Trap {
                reason: ExitReason::Breakpoint(1),
                ..
            }
        ));

        let mut source_state = CpuState::default();
        source_state.pc = BASE;
        let source_result = engine
            .step(&mut source_state, &mut fetch)
            .expect("source step should succeed");
        assert!(matches!(
            source_result,
            StepResult::Trap {
                reason: ExitReason::Breakpoint(1),
                ..
            }
        ));

        let links = engine
            .block_return_links(BASE)
            .expect("source block should expose return-link entries");
        assert!(
            links
                .iter()
                .any(|(expected, linked)| *expected == BASE + 8 && *linked != 0),
            "expected eager direct-branch edge to be initialized"
        );
    }

    #[test]
    fn eager_linking_initializes_both_conditional_edges_when_precompiled() {
        let words = [
            0x3400_0040, // cbz w0, +8
            0xd420_0040, // brk #2 (fallthrough)
            0xd420_0020, // brk #1 (taken)
        ];

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        // Precompile both conditional destinations before compiling the source block.
        let mut fallthrough_state = CpuState::default();
        fallthrough_state.pc = BASE + 4;
        let fallthrough_result = engine
            .step(&mut fallthrough_state, &mut fetch)
            .expect("fallthrough step should succeed");
        assert!(matches!(
            fallthrough_result,
            StepResult::Trap {
                reason: ExitReason::Breakpoint(2),
                ..
            }
        ));

        let mut taken_state = CpuState::default();
        taken_state.pc = BASE + 8;
        let taken_result = engine
            .step(&mut taken_state, &mut fetch)
            .expect("taken step should succeed");
        assert!(matches!(
            taken_result,
            StepResult::Trap {
                reason: ExitReason::Breakpoint(1),
                ..
            }
        ));

        let mut source_state = CpuState::default();
        source_state.pc = BASE;
        source_state.write_x(0, 0);
        let source_result = engine
            .step(&mut source_state, &mut fetch)
            .expect("source step should succeed");
        assert!(matches!(
            source_result,
            StepResult::Trap {
                reason: ExitReason::Breakpoint(1),
                ..
            }
        ));

        let links = engine
            .block_return_links(BASE)
            .expect("source block should expose return-link entries");
        let has_taken = links
            .iter()
            .any(|(expected, linked)| *expected == BASE + 8 && *linked != 0);
        let has_fallthrough = links
            .iter()
            .any(|(expected, linked)| *expected == BASE + 4 && *linked != 0);

        assert!(has_taken, "expected eager direct link for taken edge");
        assert!(
            has_fallthrough,
            "expected eager direct link for fallthrough edge"
        );
    }

    #[test]
    fn late_linking_records_direct_miss_source_edge() {
        let words = [
            0x1400_0002, // b +8 (block1 -> block2)
            0xd503_201f, // nop padding
            0x1400_0002, // b +8 (block2 -> block3)
            0xd503_201f, // nop padding
            0x1400_0000, // b +0 (block3 -> block3)
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let first = engine
            .step(&mut state, &mut fetch)
            .expect("first step should succeed");
        assert!(matches!(
            first,
            StepResult::Advanced {
                start_pc: BASE,
                end_pc: pc
            } if pc == BASE + 8
        ));

        state.pc = BASE;
        let second = engine
            .step(&mut state, &mut fetch)
            .expect("second step should succeed");
        assert!(matches!(
            second,
            StepResult::Advanced {
                start_pc: BASE,
                end_pc: pc
            } if pc == BASE + 16
        ));

        let block1_links = engine
            .block_return_links(BASE)
            .expect("block1 should expose return-link entries");
        let block2_links = engine
            .block_return_links(BASE + 8)
            .expect("block2 should expose return-link entries");

        let block1_has_block2 = block1_links
            .iter()
            .any(|(expected, linked)| *expected == BASE + 8 && *linked != 0);
        let block1_has_block3 = block1_links
            .iter()
            .any(|(expected, linked)| *expected == BASE + 16 && *linked != 0);
        let block2_has_block3 = block2_links
            .iter()
            .any(|(expected, linked)| *expected == BASE + 16 && *linked != 0);

        assert!(
            block1_has_block2,
            "expected block1->block2 direct edge link"
        );
        assert!(!block1_has_block3, "block1 should not be patched to block3");
        assert!(
            block2_has_block3,
            "expected block2->block3 direct edge link"
        );
    }

    #[test]
    fn late_linking_skips_pte_backed_targets() {
        let words = [
            0x1400_0002, // b +8
            0xd503_201f, // nop
            0xf940_0020, // ldr x0, [x1]
            0xd420_0020, // brk #1
        ];

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut memory = PteOnlyMemory::new(BASE, 0x1000);
        for (index, word) in words.into_iter().enumerate() {
            memory.inst.insert(BASE + (index as u64) * 4, word);
        }

        let data_address = BASE + 0x100;
        memory.bytes[0x100..0x108].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_x(1, data_address);

        let result = engine
            .step_with_memory(&mut state, &mut memory)
            .expect("late-link source step should succeed");
        assert!(matches!(
            result,
            StepResult::Advanced {
                start_pc: BASE,
                end_pc: pc
            } if pc == BASE + 8
        ));

        let cache = engine
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        assert!(
            !cache.contains_key(&(BASE + 8)),
            "pte-backed memory should not late-link compile the target block"
        );
    }

    #[test]
    fn late_link_slot_keeps_multiple_edges() {
        let words = [
            0xd280_0001, // movz x1, #0
            0x3400_0041, // cbz w1, +8
            0xd420_0020, // brk #1
            0xd420_0040, // brk #2
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let _ = engine
            .step(&mut state, &mut fetch)
            .expect("step should succeed");

        assert!(engine.patch_block_return_target(BASE, BASE + 8, 0x1111));
        assert!(engine.patch_block_return_target(BASE, BASE + 12, 0x2222));

        assert_eq!(
            engine
                .block_return_link_entry_count(BASE)
                .expect("source block should have return-link metadata"),
            2,
        );

        let links = engine
            .block_return_links(BASE)
            .expect("source block should expose return-link entries");

        let mut saw_first = false;
        let mut saw_second = false;
        for (expected, linked) in links {
            if expected == BASE + 8 {
                saw_first = linked == 0x1111;
            } else if expected == BASE + 12 {
                saw_second = linked == 0x2222;
            }
        }

        assert!(saw_first, "expected first edge in return-link table");
        assert!(saw_second, "expected second edge in return-link table");
    }

    #[test]
    fn svc_exit_thread_traps_as_thread_exit() {
        let words = [
            0xd400_0141, // svc #0xa (ExitThread)
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let result = engine
            .step(&mut state, &mut fetch)
            .expect("step should succeed");

        match result {
            StepResult::Trap { reason, .. } => {
                assert_eq!(reason, ExitReason::SupervisorCall(0xa));
            }
            StepResult::Advanced { .. } => panic!("expected supervisor-call trap"),
        }

        assert_eq!(state.pc, BASE + 4);
        assert_eq!(state.current_el(), ExceptionLevel::El1);
    }

    #[test]
    fn dynarec_engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DynarecEngine>();
    }

    #[test]
    fn executes_wide_and_narrow_compare_branches() {
        let words = [
            0xd280_0000, // movz x0, #0
            0xd280_0001, // movz x1, #0
            0xf2e0_0021, // movk x1, #1, lsl #48
            0x3400_0041, // cbz w1, +8
            0xd280_0020, // movz x0, #1
            0xd280_0022, // movz x2, #1
            0x3500_0042, // cbnz w2, +8
            0xd280_0040, // movz x0, #2
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 0);
        assert_eq!(state.read_x(1), 0x0001_0000_0000_0000);
    }

    #[test]
    fn executes_u32_move_wide_instructions() {
        let words = [
            0x5280_0001, // movz w1, #0
            0x12a2_4685, // movn w5, #0x1234, lsl #16
            0x7280_00a5, // movk w5, #5
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 0);
        assert_eq!(state.read_x(5), 0x0000_0000_edcb_0005);
    }

    #[test]
    fn executes_logical_immediate_instructions() {
        let words = [
            0x1280_0002, // movn w2, #0
            0x1200_1c43, // and w3, w2, #0xff
            0x3200_1c64, // orr w4, w3, #0xff
            0x5200_0c85, // eor w5, w4, #0x0f
            0x7200_0ca6, // ands w6, w5, #0x0f
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 0x0000_0000_0000_00ff);
        assert_eq!(state.read_x(4), 0x0000_0000_0000_00ff);
        assert_eq!(state.read_x(5), 0x0000_0000_0000_00f0);
        assert_eq!(state.read_x(6), 0x0000_0000_0000_0000);
        assert_eq!(state.nzcv, 0x4000_0000);
    }

    #[test]
    fn executes_logical_not_register_variants() {
        let words = [
            0xd282_4681, // movz x1, #0x1234
            0xd280_1fe2, // movz x2, #0xff
            0x8a22_0023, // bic x3, x1, x2
            0xaa22_0024, // orn x4, x1, x2
            0xca22_0025, // eon x5, x1, x2
            0xea22_0026, // bics x6, x1, x2
            0xd282_4687, // movz x7, #0x1234
            0xd280_1fe8, // movz x8, #0xff
            0x0a28_00e9, // bic w9, w7, w8
            0x2a28_00ea, // orn w10, w7, w8
            0x4a28_00eb, // eon w11, w7, w8
            0x6a28_00ec, // bics w12, w7, w8
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..20 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 0x0000_0000_0000_1200);
        assert_eq!(state.read_x(4), 0xffff_ffff_ffff_ff34);
        assert_eq!(state.read_x(5), 0xffff_ffff_ffff_ed34);
        assert_eq!(state.read_x(6), 0x0000_0000_0000_1200);
        assert_eq!(state.read_x(9), 0x0000_0000_0000_1200);
        assert_eq!(state.read_x(10), 0x0000_0000_ffff_ff34);
        assert_eq!(state.read_x(11), 0x0000_0000_ffff_ed34);
        assert_eq!(state.read_x(12), 0x0000_0000_0000_1200);
        assert_eq!(state.nzcv, 0);
    }

    #[test]
    fn compiled_add_sub_immediate_respects_sp_and_wsp() {
        let words = [
            0xd280_0020, // movz x0, #1
            0x9100_141f, // add sp, x0, #5
            0xd100_0be1, // sub x1, sp, #2
            0x1100_043f, // add wsp, w1, #1
            0x5100_07e2, // sub w2, wsp, #1
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.sp, 5);
        assert_eq!(state.read_x(1), 4);
        assert_eq!(state.read_x(2), 4);
    }

    #[test]
    fn executes_bitfield_shift_aliases() {
        let words = [
            0xd282_4689, // movz x9, #0x1234
            0xd343_fd29, // lsr x9, x9, #3
            0xd280_00a8, // movz x8, #5
            0xd37c_ed08, // lsl x8, x8, #4
            0x92a2_468a, // movn x10, #0x1234, lsl #16
            0x9345_fd4a, // asr x10, x10, #5
            0xd2c0_0028, // movz x8, #1, lsl #32
            0xd360_a50b, // ubfm x11, x8, #32, #41
            0xd295_79aa, // movz x10, #0xabcd
            0xb360_7d4b, // bfm x11, x10, #32, #31
            0x9280_0000, // movn x0, #0
            0x9340_7c00, // sbfm x0, x0, #0, #31 (sxtw x0, w0)
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(9), 0x246);
        assert_eq!(state.read_x(8), 0x1_0000_0000);
        assert_eq!(state.read_x(10), 0xabcd);
        assert_eq!(state.read_x(11), 0x0000_abcd_0000_0001);
        assert_eq!(state.read_x(0), u64::MAX);
    }

    #[test]
    fn executes_bl_and_br_return_path() {
        let words = [
            0x9400_0002, // bl +8
            0xd420_0020, // brk #1
            0xd280_0060, // movz x0, #3
            0xd61f_03c0, // br x30
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 3);
        assert_eq!(state.read_x(30), BASE + 4);
    }

    #[test]
    fn executes_adr_and_blr_path() {
        let words = [
            0x1000_0042, // adr x2, +8
            0xd63f_0040, // blr x2
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(30), BASE + 8);
    }

    #[test]
    fn executes_mul_madd_and_safe_divide() {
        let target = BASE + 0x20;
        let words = [
            (BASE, 0xd280_00c1),        // movz x1, #6
            (BASE + 4, 0xd280_00e2),    // movz x2, #7
            (BASE + 8, 0x9b02_7c20),    // mul x0, x1, x2
            (BASE + 12, 0x9b02_0023),   // madd x3, x1, x2, x0
            (BASE + 16, 0x9b02_8024),   // msub x4, x1, x2, x0
            (BASE + 20, 0x1400_0003),   // b +12 -> BASE + 0x20
            (target, 0x9ac1_0865),      // udiv x5, x3, x1
            (target + 4, 0x9ac1_0c66),  // sdiv x6, x3, x1
            (target + 8, 0x9adf_0867),  // udiv x7, x3, xzr
            (target + 12, 0x9adf_0c68), // sdiv x8, x3, xzr
            (target + 16, 0x9ba2_0029), // umaddl x9, w1, w2, x0
            (target + 20, 0x9ba2_802a), // umsubl x10, w1, w2, x0
            (target + 24, 0x9b22_002b), // smaddl x11, w1, w2, x0
            (target + 28, 0x9b22_802c), // smsubl x12, w1, w2, x0
            (target + 32, 0x9280_000d), // movn x13, #0
            (target + 36, 0xd280_004e), // movz x14, #2
            (target + 40, 0x9b4e_7daf), // smulh x15, x13, x14
            (target + 44, 0x9bce_7db0), // umulh x16, x13, x14
            (target + 48, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..20 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 42);
        assert_eq!(state.read_x(3), 84);
        assert_eq!(state.read_x(4), 0);
        assert_eq!(state.read_x(5), 14);
        assert_eq!(state.read_x(6), 14);
        assert_eq!(state.read_x(7), 0);
        assert_eq!(state.read_x(8), 0);
        assert_eq!(state.read_x(9), 84);
        assert_eq!(state.read_x(10), 0);
        assert_eq!(state.read_x(11), 84);
        assert_eq!(state.read_x(12), 0);
        assert_eq!(state.read_x(15), u64::MAX);
        assert_eq!(state.read_x(16), 1);
    }

    #[test]
    fn executes_sdiv_min_overflow_behavior() {
        let words = [
            (BASE, 0xd280_0000),      // movz x0, #0
            (BASE + 4, 0xf2f0_0000),  // movk x0, #0x8000, lsl #48
            (BASE + 8, 0x9280_0001),  // movn x1, #0
            (BASE + 12, 0x9ac1_0c02), // sdiv x2, x0, x1
            (BASE + 16, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x8000_0000_0000_0000);
    }

    #[test]
    fn executes_sdiv32_min_overflow_behavior() {
        let words = [
            (BASE, 0xd2b0_0000),      // movz x0, #0x8000, lsl #16
            (BASE + 4, 0x9280_0001),  // movn x1, #0
            (BASE + 8, 0x1ac1_0c02),  // sdiv w2, w0, w1
            (BASE + 12, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x0000_0000_8000_0000);
    }

    #[test]
    fn executes_conditional_select_and_variable_shifts() {
        let words = [
            (BASE, 0xd280_00a1),      // movz x1, #5
            (BASE + 4, 0xd280_00e2),  // movz x2, #7
            (BASE + 8, 0xf100_143f),  // subs xzr, x1, #5
            (BASE + 12, 0x9a82_0023), // csel x3, x1, x2, eq
            (BASE + 16, 0x9a82_1424), // csinc x4, x1, x2, ne
            (BASE + 20, 0xda82_0025), // csinv x5, x1, x2, eq
            (BASE + 24, 0xda82_1426), // csneg x6, x1, x2, ne
            (BASE + 28, 0xd280_0027), // movz x7, #1
            (BASE + 32, 0xd280_0088), // movz x8, #4
            (BASE + 36, 0x9ac8_2029), // lslv x9, x1, x8
            (BASE + 40, 0x9ac8_252a), // lsrv x10, x9, x8
            (BASE + 44, 0x9ac7_28cb), // asrv x11, x6, x7
            (BASE + 48, 0x9ac8_2c2c), // rorv x12, x1, x8
            (BASE + 52, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 5);
        assert_eq!(state.read_x(4), 8);
        assert_eq!(state.read_x(5), 5);
        assert_eq!(state.read_x(6), (!7u64).wrapping_add(1));
        assert_eq!(state.read_x(9), 80);
        assert_eq!(state.read_x(10), 5);
        assert_eq!(state.read_x(11), ((-7i64) >> 1) as u64);
        assert_eq!(state.read_x(12), 5u64.rotate_right(4));
    }

    #[test]
    fn executes_extr_and_ror_immediate_alias() {
        let words = [
            (BASE, 0xd28e_f100),      // movz x0, #0x7788
            (BASE + 4, 0xf2aa_acc0),  // movk x0, #0x5566, lsl #16
            (BASE + 8, 0xf2c6_6880),  // movk x0, #0x3344, lsl #32
            (BASE + 12, 0xf2e2_2440), // movk x0, #0x1122, lsl #48
            (BASE + 16, 0xd29f_e001), // movz x1, #0xff00
            (BASE + 20, 0xf2bb_bdc1), // movk x1, #0xddee, lsl #16
            (BASE + 24, 0xf2d7_7981), // movk x1, #0xbbcc, lsl #32
            (BASE + 28, 0xf2f3_3541), // movk x1, #0x99aa, lsl #48
            (BASE + 32, 0x5298_7a84), // movz w4, #0xc3d4
            (BASE + 36, 0x72b4_3644), // movk w4, #0xa1b2, lsl #16
            (BASE + 40, 0x5286_6885), // movz w5, #0x3344
            (BASE + 44, 0x72a2_2445), // movk w5, #0x1122, lsl #16
            (BASE + 48, 0x93c1_2002), // extr x2, x0, x1, #8
            (BASE + 52, 0x1385_3083), // extr w3, w4, w5, #12
            (BASE + 56, 0x93c0_4006), // ror x6, x0, #16 (alias)
            (BASE + 60, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..20 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x8899_aabb_ccdd_eeff);
        assert_eq!(state.read_x(3), 0x0000_0000_3d41_1223);
        assert_eq!(state.read_x(6), 0x7788_1122_3344_5566);
    }

    #[test]
    fn executes_compiled_conditional_compare_variants() {
        let words = [
            (BASE, 0xd280_00a1),                                // movz x1, #5
            (BASE + 4, 0xd280_00a2),                            // movz x2, #5
            (BASE + 8, encode_ccmp_reg(true, 1, 2, 0xf, 0x0)),  // ccmp x1, x2, #0xf, eq
            (BASE + 12, 0xf100_143f),                           // subs xzr, x1, #5
            (BASE + 16, encode_ccmp_imm(true, 1, 5, 0x2, 0x0)), // ccmp x1, #5, #0x2, eq
            (BASE + 20, encode_ccmn_reg(true, 1, 2, 0x1, 0x1)), // ccmn x1, x2, #0x1, ne
            (BASE + 24, 0xf100_103f),                           // subs xzr, x1, #4
            (BASE + 28, encode_ccmn_imm(true, 1, 1, 0xa, 0x1)), // ccmn x1, #1, #0xa, ne
            (BASE + 32, 0xd420_0020),                           // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.nzcv, 0x0000_0000);
    }

    #[test]
    fn executes_interpreted_conditional_compare_variants() {
        let words = [
            0xd280_0021,                            // movz x1, #1
            0xd280_0022,                            // movz x2, #1
            0x7100_043f,                            // subs wzr, w1, #1
            encode_ccmp_reg(false, 1, 2, 0xe, 0x0), // ccmp w1, w2, #0xe, eq
            encode_ccmn_imm(false, 1, 1, 0x5, 0x1), // ccmn w1, #1, #0x5, ne
            0xd400_0001,                            // svc #0
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let result = engine
            .step(&mut state, &mut fetch)
            .expect("step should succeed");

        match result {
            StepResult::Trap {
                reason: ExitReason::SupervisorCall(imm),
                ..
            } => {
                assert_eq!(imm, 0);
            }
            StepResult::Trap { reason, .. } => {
                panic!("expected supervisor-call trap, got {reason:?}")
            }
            StepResult::Advanced { .. } => panic!("expected supervisor-call trap"),
        }

        assert_eq!(state.nzcv, 0x5000_0000);
    }

    #[test]
    fn executes_u32_conditional_shift_and_mul_div() {
        let words = [
            (BASE, 0xd280_00a1),      // movz x1, #5
            (BASE + 4, 0xd280_00e2),  // movz x2, #7
            (BASE + 8, 0x7100_143f),  // subs wzr, w1, #5
            (BASE + 12, 0x1a82_0023), // csel w3, w1, w2, eq
            (BASE + 16, 0x1a82_1424), // csinc w4, w1, w2, ne
            (BASE + 20, 0x5a82_0025), // csinv w5, w1, w2, eq
            (BASE + 24, 0x5a82_1426), // csneg w6, w1, w2, ne
            (BASE + 28, 0xd280_0027), // movz x7, #1
            (BASE + 32, 0xd280_0088), // movz x8, #4
            (BASE + 36, 0x1ac8_2029), // lslv w9, w1, w8
            (BASE + 40, 0x1ac8_252a), // lsrv w10, w9, w8
            (BASE + 44, 0x1ac7_28cb), // asrv w11, w6, w7
            (BASE + 48, 0x1ac8_2c2c), // rorv w12, w1, w8
            (BASE + 52, 0x1b02_7c2d), // mul w13, w1, w2
            (BASE + 56, 0x1b02_342e), // madd w14, w1, w2, w13
            (BASE + 60, 0x1b02_b42f), // msub w15, w1, w2, w13
            (BASE + 64, 0x1ac1_09d0), // udiv w16, w14, w1
            (BASE + 68, 0x1ac1_0dd1), // sdiv w17, w14, w1
            (BASE + 72, 0x1adf_09d2), // udiv w18, w14, wzr
            (BASE + 76, 0x1adf_0dd3), // sdiv w19, w14, wzr
            (BASE + 80, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 32,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..24 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 5);
        assert_eq!(state.read_x(4), 8);
        assert_eq!(state.read_x(5), 5);
        assert_eq!(state.read_x(6), 0x0000_0000_ffff_fff9);
        assert_eq!(state.read_x(9), 80);
        assert_eq!(state.read_x(10), 5);
        assert_eq!(state.read_x(11), 0x0000_0000_ffff_fffc);
        assert_eq!(state.read_x(12), 0x0000_0000_5000_0000);
        assert_eq!(state.read_x(13), 35);
        assert_eq!(state.read_x(14), 70);
        assert_eq!(state.read_x(15), 0);
        assert_eq!(state.read_x(16), 14);
        assert_eq!(state.read_x(17), 14);
        assert_eq!(state.read_x(18), 0);
        assert_eq!(state.read_x(19), 0);
        assert_eq!(state.nzcv, 0x6000_0000);
    }

    #[test]
    fn executes_rbit_instructions() {
        let words = [
            0xd280_00a1, // movz x1, #5
            0xdac0_0022, // rbit x2, x1
            0x5ac0_0034, // rbit w20, w1
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0xa000_0000_0000_0000);
        assert_eq!(state.read_x(20), 0x0000_0000_a000_0000);
    }

    #[test]
    fn executes_clz_instructions() {
        let words = [
            0xd280_0020, // movz x0, #1
            0xdac0_1001, // clz x1, x0
            0x5ac0_1002, // clz w2, w0
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 63);
        assert_eq!(state.read_x(2), 31);
    }

    #[test]
    fn executes_rev_instructions() {
        let words = [
            0xd29b_de00, // movz x0, #0xdef0
            0xf2b3_57a0, // movk x0, #0x9abd, lsl #16
            0xf2ca_cf00, // movk x0, #0x5678, lsl #32
            0xf2e2_4680, // movk x0, #0x1234, lsl #48
            0xdac0_0c01, // rev x1, x0
            0x5ac0_0802, // rev w2, w0
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..10 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 0x1234_5678_9abd_def0);
        assert_eq!(state.read_x(1), 0xf0de_bd9a_7856_3412);
        assert_eq!(state.read_x(2), 0x0000_0000_f0de_bd9a);
    }

    #[test]
    fn executes_ins_general_vector_instructions() {
        let words = [
            0x6f00_e400, // movi v0.2d, #0
            0xd286_6881, // movz x1, #0x3344
            0xf2a2_2441, // movk x1, #0x1122, lsl #16
            0x4e0c_1c20, // mov v0.s[1], w1
            0xd297_dde2, // movz x2, #0xbeef
            0xf2bb_d5a2, // movk x2, #0xdead, lsl #16
            0xf2ca_cf02, // movk x2, #0x5678, lsl #32
            0xf2e2_4682, // movk x2, #0x1234, lsl #48
            0x4e18_1c40, // mov v0.d[1], x2
            0x4e08_3c03, // mov x3, v0.d[0]
            0x4e18_3c04, // mov x4, v0.d[1]
            0xd420_0020, // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            let idx = ((pc.wrapping_sub(BASE)) / 4) as usize;
            words.get(idx).copied()
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 0x1122_3344_0000_0000);
        assert_eq!(state.read_x(4), 0x1234_5678_dead_beef);
    }

    #[test]
    fn executes_u32_alu_zero_extend_and_logical_flags() {
        let words = [
            (BASE, 0x9280_0001),      // movn x1, #0
            (BASE + 4, 0x1100_0420),  // add w0, w1, #1
            (BASE + 8, 0xd280_0003),  // movz x3, #0
            (BASE + 12, 0x5100_0462), // sub w2, w3, #1
            (BASE + 16, 0xd280_0029), // movz x9, #1
            (BASE + 20, 0xf2e0_0029), // movk x9, #1, lsl #48
            (BASE + 24, 0xd280_004a), // movz x10, #2
            (BASE + 28, 0x0b0a_0528), // add w8, w9, w10, lsl #1
            (BASE + 32, 0x0b2a_0129), // add w9, w9, w10, uxtb
            (BASE + 36, 0xd2b0_0018), // movz x24, #0x8000, lsl #16
            (BASE + 40, 0x9280_0019), // movn x25, #0
            (BASE + 44, 0x6a19_0317), // ands w23, w24, w25
            (BASE + 48, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), u64::MAX);
        assert_eq!(state.read_x(0), 0);
        assert_eq!(state.read_x(2), 0x0000_0000_ffff_ffff);
        assert_eq!(state.read_x(8), 5);
        assert_eq!(state.read_x(9), 3);
        assert_eq!(state.read_x(23), 0x0000_0000_8000_0000);
        assert_eq!(state.nzcv, 0x8000_0000);
    }

    #[test]
    fn executes_u32_adds_subs_flag_updates() {
        let words = [
            (BASE, 0xd280_00a5),      // movz x5, #5
            (BASE + 4, 0x3100_1ca4),  // adds w4, w5, #7
            (BASE + 8, 0x7100_3086),  // subs w6, w4, #12
            (BASE + 12, 0xd280_0289), // movz x9, #20
            (BASE + 16, 0xd280_006a), // movz x10, #3
            (BASE + 20, 0x2b0a_052b), // adds w11, w9, w10, lsl #1
            (BASE + 24, 0x6b0a_056c), // subs w12, w11, w10, lsl #1
            (BASE + 28, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(4), 12);
        assert_eq!(state.read_x(6), 0);
        assert_eq!(state.read_x(11), 26);
        assert_eq!(state.read_x(12), 20);
        assert_eq!(state.nzcv, 0x2000_0000);
    }

    #[test]
    fn interpreted_subreg32_uses_wzr_not_wsp() {
        let words = [
            (BASE, 0xd280_00a1),      // movz x1, #5
            (BASE + 4, 0x4b01_03e2),  // sub w2, wzr, w1
            (BASE + 8, 0x9340_7c43),  // sbfm x3, x2, #0, #31 (force interpreter)
            (BASE + 12, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;
        state.sp = 0x1000;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x0000_0000_ffff_fffb);
        assert_eq!(state.read_x(3), 0xffff_ffff_ffff_fffb);
        assert_eq!(state.sp, 0x1000);
    }

    #[test]
    fn executes_byte_halfword_and_signed_memory_ops() {
        let mut memory = TestMemory::default();
        memory.data.insert(268, 0x8000_0001);
        memory.data.insert(252, 0x8000_0002);
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd280_0fe0); // movz x0, #127
        memory.inst.insert(BASE + 8, 0x3900_2020); // strb w0, [x1, #8]
        memory.inst.insert(BASE + 12, 0x3940_2022); // ldrb w2, [x1, #8]
        memory.inst.insert(BASE + 16, 0xd282_4683); // movz x3, #0x1234
        memory.inst.insert(BASE + 20, 0x7900_0c23); // strh w3, [x1, #6]
        memory.inst.insert(BASE + 24, 0x7940_0c24); // ldrh w4, [x1, #6]
        memory.inst.insert(BASE + 28, 0xd280_1545); // movz x5, #0xaa
        memory.inst.insert(BASE + 32, 0x381f_f025); // sturb w5, [x1, #-1]
        memory.inst.insert(BASE + 36, 0x385f_f026); // ldurb w6, [x1, #-1]
        memory.inst.insert(BASE + 40, 0xd297_dde7); // movz x7, #0xbeef
        memory.inst.insert(BASE + 44, 0x781f_e027); // sturh w7, [x1, #-2]
        memory.inst.insert(BASE + 48, 0x785f_e028); // ldurh w8, [x1, #-2]
        memory.inst.insert(BASE + 52, 0xd280_1000); // movz x0, #128
        memory.inst.insert(BASE + 56, 0x3900_2020); // strb w0, [x1, #8]
        memory.inst.insert(BASE + 60, 0xd290_0023); // movz x3, #0x8001
        memory.inst.insert(BASE + 64, 0x7900_0c23); // strh w3, [x1, #6]
        memory.inst.insert(BASE + 68, 0x3980_2029); // ldrsb x9, [x1, #8]
        memory.inst.insert(BASE + 72, 0x39c0_202a); // ldrsb w10, [x1, #8]
        memory.inst.insert(BASE + 76, 0x7980_0c2b); // ldrsh x11, [x1, #6]
        memory.inst.insert(BASE + 80, 0x79c0_0c2c); // ldrsh w12, [x1, #6]
        memory.inst.insert(BASE + 84, 0xb980_0c2d); // ldrsw x13, [x1, #12]
        memory.inst.insert(BASE + 88, 0x389f_f02e); // ldursb x14, [x1, #-1]
        memory.inst.insert(BASE + 92, 0x38df_f02f); // ldursb w15, [x1, #-1]
        memory.inst.insert(BASE + 96, 0x789f_e030); // ldursh x16, [x1, #-2]
        memory.inst.insert(BASE + 100, 0x78df_e031); // ldursh w17, [x1, #-2]
        memory.inst.insert(BASE + 104, 0xb89f_c032); // ldursw x18, [x1, #-4]
        memory.inst.insert(BASE + 108, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 64,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..48 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(memory.data.get(&264).copied(), Some(0x80));
        assert_eq!(memory.data.get(&262).copied(), Some(0x8001));
        assert_eq!(memory.data.get(&255).copied(), Some(0xaa));
        assert_eq!(memory.data.get(&254).copied(), Some(0xbeef));
        assert_eq!(state.read_x(2), 0x7f);
        assert_eq!(state.read_x(4), 0x1234);
        assert_eq!(state.read_x(6), 0xaa);
        assert_eq!(state.read_x(8), 0xbeef);
        assert_eq!(state.read_x(9), 0xffff_ffff_ffff_ff80);
        assert_eq!(state.read_x(10), 0x0000_0000_ffff_ff80);
        assert_eq!(state.read_x(11), 0xffff_ffff_ffff_8001);
        assert_eq!(state.read_x(12), 0x0000_0000_ffff_8001);
        assert_eq!(state.read_x(13), 0xffff_ffff_8000_0001);
        assert_eq!(state.read_x(14), 0xffff_ffff_ffff_ffaa);
        assert_eq!(state.read_x(15), 0x0000_0000_ffff_ffaa);
        assert_eq!(state.read_x(16), 0xffff_ffff_ffff_beef);
        assert_eq!(state.read_x(17), 0x0000_0000_ffff_beef);
        assert_eq!(state.read_x(18), 0xffff_ffff_8000_0002);
    }

    #[test]
    fn executes_str_and_ldr_roundtrip() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_0aa0); // movz x0, #85
        memory.inst.insert(BASE + 4, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 8, 0xf900_0020); // str x0, [x1]
        memory.inst.insert(BASE + 12, 0xf940_0022); // ldr x2, [x1]
        memory.inst.insert(BASE + 16, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 85);
        assert_eq!(memory.data.get(&256).copied(), Some(85));
    }

    #[test]
    fn memory_fault_fallback_resumes_at_faulting_pc() {
        let mut memory = FaultingReadMemory::default();
        memory.inst.insert(BASE, 0x9100_0400); // add x0, x0, #1
        memory.inst.insert(BASE + 4, 0xf940_0041); // ldr x1, [x2]
        memory.inst.insert(BASE + 8, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_x(0, 1);
        state.write_x(2, 0x1000);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let result = engine
            .step_with_memory(&mut state, &mut memory)
            .expect("step should succeed");

        match result {
            StepResult::Trap { pc, reason } => {
                assert_eq!(pc, BASE + 4);
                assert_eq!(reason, ExitReason::MemoryFault);
            }
            StepResult::Advanced { .. } => panic!("expected memory-fault trap"),
        }

        // Fallback starts from the mapped faulting PC, so prior instructions are not replayed.
        assert_eq!(state.read_x(0), 2);

        let cache = engine
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        assert!(cache.get(&BASE).is_none());
        let fallback_cached = cache
            .get(&(BASE + 4))
            .expect("faulting instruction block should be cached");
        assert!(matches!(
            fallback_cached.as_ref(),
            CachedBlock::Interpreted(_)
        ));
    }

    #[test]
    fn uses_fastmem_for_compiled_u64_load_store() {
        let mut memory = FastmemOnlyMemory::new(0, 0x400);
        memory.inst.insert(BASE, 0xd280_0aa0); // movz x0, #85
        memory.inst.insert(BASE + 4, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 8, 0xf900_0020); // str x0, [x1]
        memory.inst.insert(BASE + 12, 0xf940_0022); // ldr x2, [x1]
        memory.inst.insert(BASE + 16, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 85);
        assert_eq!(memory.bytes[0x100], 85);
    }

    #[test]
    fn executes_unscaled_and_32bit_memory_ops() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd282_4680); // movz x0, #0x1234
        memory.inst.insert(BASE + 8, 0xf81f_8020); // stur x0, [x1, #-8]
        memory.inst.insert(BASE + 12, 0xf85f_8022); // ldur x2, [x1, #-8]
        memory.inst.insert(BASE + 16, 0xd28a_cf03); // movz x3, #0x5678
        memory.inst.insert(BASE + 20, 0xb81f_c023); // stur w3, [x1, #-4]
        memory.inst.insert(BASE + 24, 0xb85f_c024); // ldur w4, [x1, #-4]
        memory.inst.insert(BASE + 28, 0xb900_0820); // str w0, [x1, #8]
        memory.inst.insert(BASE + 32, 0xb940_0825); // ldr w5, [x1, #8]
        memory.inst.insert(BASE + 36, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x1234);
        assert_eq!(state.read_x(4), 0x5678);
        assert_eq!(state.read_x(5), 0x1234);
        assert_eq!(memory.data.get(&0x00f8).copied(), Some(0x1234));
        assert_eq!(memory.data.get(&0x00fc).copied(), Some(0x5678));
        assert_eq!(memory.data.get(&0x0108).copied(), Some(0x1234));
    }

    #[test]
    fn executes_pre_post_index_unscaled_memory_ops() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd282_4683); // movz x3, #0x1234
        memory.inst.insert(BASE + 8, 0xf800_8423); // str x3, [x1], #8
        memory.inst.insert(BASE + 12, 0xf85f_8c24); // ldr x4, [x1, #-8]!
        memory.inst.insert(BASE + 16, 0xd28a_cf05); // movz x5, #0x5678
        memory.inst.insert(BASE + 20, 0xb800_4425); // str w5, [x1], #4
        memory.inst.insert(BASE + 24, 0xb85f_cc26); // ldr w6, [x1, #-4]!
        memory.inst.insert(BASE + 28, 0xd280_1547); // movz x7, #0xaa
        memory.inst.insert(BASE + 32, 0x3800_1427); // strb w7, [x1], #1
        memory.inst.insert(BASE + 36, 0x385f_fc28); // ldrb w8, [x1, #-1]!
        memory.inst.insert(BASE + 40, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(4), 0x1234);
        assert_eq!(state.read_x(6), 0x5678);
        assert_eq!(state.read_x(8), 0xaa);
        assert_eq!(state.read_x(1), 256);
        assert_eq!(memory.data.get(&256).copied(), Some(0xaa));
    }

    #[test]
    fn executes_register_offset_memory_ops() {
        let mut memory = TestMemory::default();
        memory.data.insert(0x0103, 0x55aa);
        memory.data.insert(0x0106, 0x8001);
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd280_0062); // movz x2, #3
        memory.inst.insert(BASE + 8, 0xd282_4683); // movz x3, #0x1234
        memory.inst.insert(BASE + 12, 0xf822_7823); // str x3, [x1, x2, lsl #3]
        memory.inst.insert(BASE + 16, 0xf862_7824); // ldr x4, [x1, x2, lsl #3]
        memory.inst.insert(BASE + 20, 0x9280_0005); // movn x5, #0
        memory.inst.insert(BASE + 24, 0xb822_7825); // str w5, [x1, x2, lsl #2]
        memory.inst.insert(BASE + 28, 0xb862_7826); // ldr w6, [x1, x2, lsl #2]
        memory.inst.insert(BASE + 32, 0xf862_4827); // ldr x7, [x1, w2, uxtw]
        memory.inst.insert(BASE + 36, 0x3862_682d); // ldrb w13, [x1, x2]
        memory.inst.insert(BASE + 40, 0x38a2_682f); // ldrsb x15, [x1, x2]
        memory.inst.insert(BASE + 44, 0x38e2_6830); // ldrsb w16, [x1, x2]
        memory.inst.insert(BASE + 48, 0x7862_7831); // ldrh w17, [x1, x2, lsl #1]
        memory.inst.insert(BASE + 52, 0x78a2_7832); // ldrsh x18, [x1, x2, lsl #1]
        memory.inst.insert(BASE + 56, 0x78e2_7833); // ldrsh w19, [x1, x2, lsl #1]
        memory.inst.insert(BASE + 60, 0xb8a2_7834); // ldrsw x20, [x1, x2, lsl #2]
        memory.inst.insert(BASE + 64, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 32,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..24 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(4), 0x1234);
        assert_eq!(state.read_x(6), 0xffff_ffff);
        assert_eq!(state.read_x(7), 0x55aa);
        assert_eq!(state.read_x(13), 0x0000_0000_0000_00aa);
        assert_eq!(state.read_x(15), 0xffff_ffff_ffff_ffaa);
        assert_eq!(state.read_x(16), 0x0000_0000_ffff_ffaa);
        assert_eq!(state.read_x(17), 0x0000_0000_0000_8001);
        assert_eq!(state.read_x(18), 0xffff_ffff_ffff_8001);
        assert_eq!(state.read_x(19), 0x0000_0000_ffff_8001);
        assert_eq!(state.read_x(20), 0xffff_ffff_ffff_ffff);
        assert_eq!(memory.data.get(&0x0118).copied(), Some(0x1234));
        assert_eq!(memory.data.get(&0x010c).copied(), Some(0xffff_ffff));
    }

    #[test]
    fn executes_pair_load_store_modes() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd280_0220); // movz x0, #17
        memory.inst.insert(BASE + 8, 0xd280_0442); // movz x2, #34
        memory.inst.insert(BASE + 12, 0xa901_0820); // stp x0, x2, [x1, #16]
        memory.inst.insert(BASE + 16, 0xa941_1023); // ldp x3, x4, [x1, #16]
        memory.inst.insert(BASE + 20, 0xd280_0665); // movz x5, #51
        memory.inst.insert(BASE + 24, 0xd280_0886); // movz x6, #68
        memory.inst.insert(BASE + 28, 0xa9bf_1825); // stp x5, x6, [x1, #-16]!
        memory.inst.insert(BASE + 32, 0xa8c1_2027); // ldp x7, x8, [x1], #16
        memory.inst.insert(BASE + 36, 0xd282_4689); // movz x9, #0x1234
        memory.inst.insert(BASE + 40, 0xd28a_cf0a); // movz x10, #0x5678
        memory.inst.insert(BASE + 44, 0x2901_2829); // stp w9, w10, [x1, #8]
        memory.inst.insert(BASE + 48, 0x2941_302b); // ldp w11, w12, [x1, #8]
        memory.inst.insert(BASE + 52, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 32,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..24 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 17);
        assert_eq!(state.read_x(4), 34);
        assert_eq!(state.read_x(7), 51);
        assert_eq!(state.read_x(8), 68);
        assert_eq!(state.read_x(11), 0x1234);
        assert_eq!(state.read_x(12), 0x5678);
        assert_eq!(state.read_x(1), 256);
    }

    #[test]
    fn executes_guest_framebuffer_addressing_sequence() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x9140_0acb); // add x11, x22, #2, lsl #12
        memory.inst.insert(BASE + 4, 0xd140_0966); // sub x6, x11, #2, lsl #12
        memory.inst.insert(BASE + 8, 0xa881_352c); // stp x12, x13, [x9], #16
        memory.inst.insert(BASE + 12, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_x(22, 0x0000_0000_0001_0000);
        state.write_x(9, 0x0000_0000_0000_0200);
        state.write_x(12, 0x1122_3344_5566_7788);
        state.write_x(13, 0x99aa_bbcc_ddee_ff00);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(11), 0x0000_0000_0001_2000);
        assert_eq!(state.read_x(6), 0x0000_0000_0001_0000);
        assert_eq!(state.read_x(9), 0x0000_0000_0000_0210);
        assert_eq!(
            memory.data.get(&0x0200).copied(),
            Some(0x1122_3344_5566_7788)
        );
        assert_eq!(
            memory.data.get(&0x0208).copied(),
            Some(0x99aa_bbcc_ddee_ff00)
        );
    }

    #[test]
    fn executes_simd_pair_load_store_modes() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xad00_0420); // stp q0, q1, [x1]
        memory.inst.insert(BASE + 8, 0xad40_0c22); // ldp q2, q3, [x1]
        memory.inst.insert(BASE + 12, 0xadbf_1424); // stp q4, q5, [x1, #-32]!
        memory.inst.insert(BASE + 16, 0xacc1_1c26); // ldp q6, q7, [x1], #32
        memory.inst.insert(BASE + 20, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(0, 0x1111_1111_1111_1111, 0x2222_2222_2222_2222);
        state.write_v128(1, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444);
        state.write_v128(4, 0xaaaa_aaaa_aaaa_aaaa, 0xbbbb_bbbb_bbbb_bbbb);
        state.write_v128(5, 0xcccc_cccc_cccc_cccc, 0xdddd_dddd_dddd_dddd);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 256);
        assert_eq!(state.read_v128(2), state.read_v128(0));
        assert_eq!(state.read_v128(3), state.read_v128(1));
        assert_eq!(state.read_v128(6), state.read_v128(4));
        assert_eq!(state.read_v128(7), state.read_v128(5));
        assert_eq!(memory.data.get(&256).copied(), Some(0x1111_1111_1111_1111));
        assert_eq!(memory.data.get(&264).copied(), Some(0x2222_2222_2222_2222));
        assert_eq!(memory.data.get(&272).copied(), Some(0x3333_3333_3333_3333));
        assert_eq!(memory.data.get(&280).copied(), Some(0x4444_4444_4444_4444));
        assert_eq!(memory.data.get(&224).copied(), Some(0xaaaa_aaaa_aaaa_aaaa));
        assert_eq!(memory.data.get(&232).copied(), Some(0xbbbb_bbbb_bbbb_bbbb));
        assert_eq!(memory.data.get(&240).copied(), Some(0xcccc_cccc_cccc_cccc));
        assert_eq!(memory.data.get(&248).copied(), Some(0xdddd_dddd_dddd_dddd));
    }

    #[test]
    fn executes_movi_v2d_zero() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6f00_e400); // movi v0.2d, #0
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(0, 0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(0), (0, 0));
    }

    #[test]
    fn executes_movi_v4s_zero() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4f00_041f); // movi v31.4s, #0
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(31, 0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(31), (0, 0));
    }

    #[test]
    fn executes_add_vector_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4ebe_87fe); // add v30.4s, v31.4s, v30.4s
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(31, 0x0000_0001_ffff_fffe, 0x7fff_ffff_8000_0000);
        state.write_v128(30, 0xffff_ffff_0000_0005, 0x0000_0002_0000_0001);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(30),
            (0x0000_0000_0000_0003, 0x8000_0001_8000_0001)
        );
    }

    #[test]
    fn executes_add_vector_4h_and_clears_upper_half() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x0e7e_87fe); // add v30.4h, v31.4h, v30.4h
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(31, 0x7fff_8000_0001_fffe, 0x1111_2222_3333_4444);
        state.write_v128(30, 0x0002_0001_ffff_0005, 0xdead_beef_dead_beef);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(30), (0x8001_8001_0000_0003, 0));
    }

    #[test]
    fn executes_addp_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4e32_be31); // addp v17.16b, v17.16b, v18.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(17, 0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908);
        state.write_v128(18, 0x1716_1514_1312_1110, 0x1f1e_1d1c_1b1a_1918);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(17),
            (0x1d19_1511_0d09_0501, 0x3d39_3531_2d29_2521)
        );
    }

    #[test]
    fn executes_uminp_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6e22_ac20); // uminp v0.16b, v1.16b, v2.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(1, 0x0308_0104_0902_0500, 0x0f0b_0d0e_060c_070a);
        state.write_v128(2, 0x1317_1216_1115_1014, 0x1d1b_1c1a_1f19_1e18);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(0),
            (0x0b0d_0607_0301_0200, 0x1b1a_1918_1312_1110)
        );
    }

    #[test]
    fn executes_umaxp_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6e22_a420); // umaxp v0.16b, v1.16b, v2.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(1, 0x0308_0104_0902_0500, 0x0f0b_0d0e_060c_070a);
        state.write_v128(2, 0x1317_1216_1115_1014, 0x1d1b_1c1a_1f19_1e18);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(0),
            (0x0f0e_0c0a_0804_0905, 0x1d1c_1f1e_1716_1514)
        );
    }

    #[test]
    fn executes_shrn_vector_and_shrn2_preserves_low_half() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x0f0c_8400); // shrn v0.8b, v0.8h, #4
        memory.inst.insert(BASE + 4, 0x4f0c_8422); // shrn2 v2.16b, v1.8h, #4
        memory.inst.insert(BASE + 8, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(0, 0x0308_0104_0902_0500, 0x0f0b_0d0e_060c_070a);
        state.write_v128(1, 0x0308_0104_0902_0500, 0x0f0b_0d0e_060c_070a);
        state.write_v128(2, 0x8877_6655_4433_2211, 0xfeed_face_dead_beef);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(0), (0xf0d0_6070_3010_9050, 0));
        assert_eq!(
            state.read_v128(2),
            (0x8877_6655_4433_2211, 0xf0d0_6070_3010_9050)
        );
    }

    #[test]
    fn executes_cmeq_vector_zero_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4e20_9823); // cmeq v3.16b, v1.16b, #0
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(1, 0, 1);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(3), (u64::MAX, 0xffff_ffff_ffff_ff00));
    }

    #[test]
    fn executes_cmeq_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6e20_8c25); // cmeq v5.16b, v1.16b, v0.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(1, 0x0200_0200_0100_0100, 0x0908_0706_0504_0302);
        state.write_v128(0, 0x0300_0200_0400_0100, 0x0900_0706_0004_0302);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(5),
            (0x00ff_ffff_00ff_ffff, 0xff00_ffff_00ff_ffff)
        );
    }

    #[test]
    fn executes_cmhs_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6e21_3ca3); // cmhs v3.16b, v5.16b, v1.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(5, 0x0807_0605_0403_0201, 0x5046_3c32_281e_140a);
        state.write_v128(1, 0x0907_0701_0404_0200, 0x5145_3c37_271e_1509);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(3),
            (0x00ff_00ff_ff00_ffff, 0x00ff_ff00_ffff_00ff)
        );
    }

    #[test]
    fn executes_bif_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6ee7_1ca3); // bif v3.16b, v5.16b, v7.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(3, 0x1111_1111_1111_1111, 0x2222_2222_2222_2222);
        state.write_v128(5, 0xaaaa_aaaa_aaaa_aaaa, 0xbbbb_bbbb_bbbb_bbbb);
        state.write_v128(7, 0xff00_ff00_ff00_ff00, 0x00ff_00ff_00ff_00ff);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(3),
            (0x11aa_11aa_11aa_11aa, 0xbb22_bb22_bb22_bb22)
        );
    }

    #[test]
    fn executes_and_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4e30_1c71); // and v17.16b, v3.16b, v16.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(3, 0xffff_0000_ffff_0000, 0x1234_5678_9abc_def0);
        state.write_v128(16, 0x0f0f_f0f0_00ff_00ff, 0xffff_ffff_0000_0000);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(17),
            (0x0f0f_0000_00ff_0000, 0x1234_5678_0000_0000)
        );
    }

    #[test]
    fn executes_orr_vector_16b() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4ea4_1c71); // orr v17.16b, v3.16b, v4.16b
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(3, 0x00ff_0000_00ff_0000, 0x1234_0000_9abc_0000);
        state.write_v128(4, 0xff00_00ff_0000_00ff, 0x0000_5678_0000_def0);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(17),
            (0xffff_00ff_00ff_00ff, 0x1234_5678_9abc_def0)
        );
    }

    #[test]
    fn executes_dup_v4s_from_w() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4e04_0c90); // dup v16.4s, w4
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_x(4, 0xaaaa_bbbb_cccc_dddd);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(16),
            (0xcccc_dddd_cccc_dddd, 0xcccc_dddd_cccc_dddd)
        );
    }

    #[test]
    fn executes_ld1_two_v16b_post_index() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2002); // movz x2, #256
        memory.inst.insert(BASE + 4, 0x4cdf_a041); // ld1 {v1.16b, v2.16b}, [x2], #32
        memory.inst.insert(BASE + 8, 0xd420_0020); // brk #1

        memory.data.insert(256, 0x1111_1111_1111_1111);
        memory.data.insert(264, 0x2222_2222_2222_2222);
        memory.data.insert(272, 0x3333_3333_3333_3333);
        memory.data.insert(280, 0x4444_4444_4444_4444);

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 8,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..6 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(
            state.read_v128(1),
            (0x1111_1111_1111_1111, 0x2222_2222_2222_2222)
        );
        assert_eq!(
            state.read_v128(2),
            (0x3333_3333_3333_3333, 0x4444_4444_4444_4444)
        );
        assert_eq!(state.read_x(2), 288);
    }

    #[test]
    fn executes_fp_simd_post_pre_index_q_memory_ops() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0x3c81_0420); // str q0, [x1], #16
        memory.inst.insert(BASE + 8, 0x3c81_0c21); // str q1, [x1, #16]!
        memory.inst.insert(BASE + 12, 0x3cc1_0422); // ldr q2, [x1], #16
        memory.inst.insert(BASE + 16, 0x3cdd_0c23); // ldr q3, [x1, #-48]!
        memory.inst.insert(BASE + 20, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(0, 0x1111_1111_1111_1111, 0x2222_2222_2222_2222);
        state.write_v128(1, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..10 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 256);
        assert_eq!(state.read_v128(2), state.read_v128(1));
        assert_eq!(state.read_v128(3), state.read_v128(0));
        assert_eq!(memory.data.get(&256).copied(), Some(0x1111_1111_1111_1111));
        assert_eq!(memory.data.get(&264).copied(), Some(0x2222_2222_2222_2222));
        assert_eq!(memory.data.get(&288).copied(), Some(0x3333_3333_3333_3333));
        assert_eq!(memory.data.get(&296).copied(), Some(0x4444_4444_4444_4444));
    }

    #[test]
    fn executes_fp_simd_unscaled_immediate_memory_ops() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0x3c01_f020); // stur b0, [x1, #31]
        memory.inst.insert(BASE + 8, 0x3c41_f022); // ldur b2, [x1, #31]
        memory.inst.insert(BASE + 12, 0x7c1f_a024); // stur h4, [x1, #-6]
        memory.inst.insert(BASE + 16, 0x7c5f_a026); // ldur h6, [x1, #-6]
        memory.inst.insert(BASE + 20, 0xbc02_4028); // stur s8, [x1, #36]
        memory.inst.insert(BASE + 24, 0xbc42_402a); // ldur s10, [x1, #36]
        memory.inst.insert(BASE + 28, 0xfc1f_c02c); // stur d12, [x1, #-4]
        memory.inst.insert(BASE + 32, 0xfc5f_c02e); // ldur d14, [x1, #-4]
        memory.inst.insert(BASE + 36, 0x3c84_8020); // stur q0, [x1, #72]
        memory.inst.insert(BASE + 40, 0x3cc4_8022); // ldur q2, [x1, #72]
        memory.inst.insert(BASE + 44, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        state.write_v128(0, 0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);
        state.write_v128(4, 0x0000_0000_0000_beef, 0xffff_ffff_ffff_ffff);
        state.write_v128(8, 0x0000_0000_a1b2_c3d4, 0xffff_ffff_ffff_ffff);
        state.write_v128(12, 0x1234_5678_9abc_def0, 0xffff_ffff_ffff_ffff);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..20 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_v128(2), state.read_v128(0));
        assert_eq!(state.read_v128(6), (0x0000_0000_0000_beef, 0));
        assert_eq!(state.read_v128(10), (0x0000_0000_a1b2_c3d4, 0));
        assert_eq!(state.read_v128(14), (0x1234_5678_9abc_def0, 0));
        assert_eq!(memory.data.get(&287).copied(), Some(0x88));
        assert_eq!(memory.data.get(&250).copied(), Some(0xbeef));
        assert_eq!(memory.data.get(&292).copied(), Some(0xa1b2_c3d4));
        assert_eq!(memory.data.get(&252).copied(), Some(0x1234_5678_9abc_def0));
        assert_eq!(memory.data.get(&328).copied(), Some(0x1122_3344_5566_7788));
        assert_eq!(memory.data.get(&336).copied(), Some(0x99aa_bbcc_ddee_ff00));
    }

    #[test]
    fn executes_exclusive_load_store_and_clrex() {
        let mut memory = TestMemory::default();
        memory.data.insert(256, 0x99);
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xc85f_7c22); // ldxr x2, [x1]
        memory.inst.insert(BASE + 8, 0xd280_0aa3); // movz x3, #85
        memory.inst.insert(BASE + 12, 0xc804_7c23); // stxr w4, x3, [x1]
        memory.inst.insert(BASE + 16, 0xd503_305f); // clrex
        memory.inst.insert(BASE + 20, 0xc805_7c22); // stxr w5, x2, [x1]
        memory.inst.insert(BASE + 24, 0xc85f_7c26); // ldxr x6, [x1]
        memory.inst.insert(BASE + 28, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        let cache = engine
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        let cached = cache
            .get(&BASE)
            .expect("exclusive-load/store block should be cached");
        assert!(matches!(cached.as_ref(), CachedBlock::Compiled(_)));
        assert_eq!(state.read_x(2), 0x99);
        assert_eq!(state.read_x(4), 0);
        assert_eq!(state.read_x(5), 1);
        assert_eq!(state.read_x(6), 0x55);
        assert_eq!(memory.data.get(&256).copied(), Some(0x55));
    }

    #[test]
    fn executes_pair_exclusive_variants() {
        let mut memory = TestMemory::default();
        memory.data.insert(256, 0x1111_2222_3333_4444);
        memory.data.insert(264, 0x5555_6666_7777_8888);
        memory.data.insert(512, 0xaabb_ccdd);
        memory.data.insert(516, 0xeeff_0011);
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xc87f_0c22); // ldxp x2, x3, [x1]
        memory.inst.insert(BASE + 8, 0xd280_1324); // movz x4, #153
        memory.inst.insert(BASE + 12, 0xd280_0ee5); // movz x5, #119
        memory.inst.insert(BASE + 16, 0xc826_1424); // stxp w6, x4, x5, [x1]
        memory.inst.insert(BASE + 20, 0xd280_4007); // movz x7, #512
        memory.inst.insert(BASE + 24, 0x887f_a4e8); // ldaxp w8, w9, [x7]
        memory.inst.insert(BASE + 28, 0xd282_468a); // movz x10, #0x1234
        memory.inst.insert(BASE + 32, 0xd28a_cf0b); // movz x11, #0x5678
        memory.inst.insert(BASE + 36, 0x882c_acea); // stlxp w12, w10, w11, [x7]
        memory.inst.insert(BASE + 40, 0xd503_3f5f); // clrex
        memory.inst.insert(BASE + 44, 0xc82d_1424); // stxp w13, x4, x5, [x1]
        memory.inst.insert(BASE + 48, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 32,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..24 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x1111_2222_3333_4444);
        assert_eq!(state.read_x(3), 0x5555_6666_7777_8888);
        assert_eq!(state.read_x(6), 0);
        assert_eq!(state.read_x(8), 0xaabb_ccdd);
        assert_eq!(state.read_x(9), 0xeeff_0011);
        assert_eq!(state.read_x(12), 0);
        assert_eq!(state.read_x(13), 1);
        assert_eq!(memory.data.get(&256).copied(), Some(153));
        assert_eq!(memory.data.get(&264).copied(), Some(119));
        assert_eq!(memory.data.get(&512).copied(), Some(0x1234));
        assert_eq!(memory.data.get(&516).copied(), Some(0x5678));
    }

    #[test]
    fn executes_tbz_tbnz_and_acquire_release_exclusives() {
        let mut memory = TestMemory::default();
        memory.data.insert(256, 0x1122_3344_5566_7788);
        memory.data.insert(512, 0xaabb_ccdd);
        memory.inst.insert(BASE, 0xd280_0403); // movz x3, #32
        memory.inst.insert(BASE + 4, 0x3628_0043); // tbz w3, #5, +8
        memory.inst.insert(BASE + 8, 0xd280_0021); // movz x1, #1
        memory.inst.insert(BASE + 12, 0xd280_1004); // movz x4, #128
        memory.inst.insert(BASE + 16, 0x3738_0044); // tbnz w4, #7, +8
        memory.inst.insert(BASE + 20, 0xd280_0041); // movz x1, #2
        memory.inst.insert(BASE + 24, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 28, 0xd280_4002); // movz x2, #512
        memory.inst.insert(BASE + 32, 0xc85f_fc23); // ldaxr x3, [x1]
        memory.inst.insert(BASE + 36, 0xd280_1324); // movz x4, #153
        memory.inst.insert(BASE + 40, 0xc805_fc24); // stlxr w5, x4, [x1]
        memory.inst.insert(BASE + 44, 0x885f_fc46); // ldaxr w6, [x2]
        memory.inst.insert(BASE + 48, 0xd282_4687); // movz x7, #0x1234
        memory.inst.insert(BASE + 52, 0x8808_fc47); // stlxr w8, w7, [x2]
        memory.inst.insert(BASE + 56, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 32,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..24 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 256);
        assert_eq!(state.read_x(3), 0x1122_3344_5566_7788);
        assert_eq!(state.read_x(5), 0);
        assert_eq!(state.read_x(6), 0xaabb_ccdd);
        assert_eq!(state.read_x(8), 0);
        assert_eq!(memory.data.get(&256).copied(), Some(153));
        assert_eq!(memory.data.get(&512).copied(), Some(0x1234));
    }

    #[test]
    fn executes_ldar_stlr_roundtrip() {
        let mut memory = TestMemory::default();
        memory.data.insert(256, 0x1122_3344_5566_7788);
        memory.data.insert(512, 0xaabb_ccdd);
        memory.data.insert(768, 0x5a);
        memory.data.insert(1024, 0xbeef);
        memory.inst.insert(BASE, 0xd280_2001); // movz x1, #256
        memory.inst.insert(BASE + 4, 0xd280_4002); // movz x2, #512
        memory.inst.insert(BASE + 8, 0xc8df_fc23); // ldar x3, [x1]
        memory.inst.insert(BASE + 12, 0x88df_fc44); // ldar w4, [x2]
        memory.inst.insert(BASE + 16, 0xd280_1325); // movz x5, #153
        memory.inst.insert(BASE + 20, 0xc89f_fc25); // stlr x5, [x1]
        memory.inst.insert(BASE + 24, 0xd282_4686); // movz x6, #0x1234
        memory.inst.insert(BASE + 28, 0x889f_fc46); // stlr w6, [x2]
        memory.inst.insert(BASE + 32, 0xd280_6007); // movz x7, #768
        memory.inst.insert(BASE + 36, 0xd280_8008); // movz x8, #1024
        memory.inst.insert(BASE + 40, 0x08df_fce9); // ldarb w9, [x7]
        memory.inst.insert(BASE + 44, 0x48df_fd0a); // ldarh w10, [x8]
        memory.inst.insert(BASE + 48, 0xd280_154b); // movz x11, #0xaa
        memory.inst.insert(BASE + 52, 0x089f_fceb); // stlrb w11, [x7]
        memory.inst.insert(BASE + 56, 0xd288_642c); // movz x12, #0x4321
        memory.inst.insert(BASE + 60, 0x489f_fd0c); // stlrh w12, [x8]
        memory.inst.insert(BASE + 64, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(&mut state, &mut memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        let cache = engine
            .block_cache
            .read()
            .expect("dynarec block cache poisoned");
        let cached = cache.get(&BASE).expect("ldar/stlr block should be cached");
        assert!(matches!(cached.as_ref(), CachedBlock::Compiled(_)));
        assert_eq!(state.read_x(3), 0x1122_3344_5566_7788);
        assert_eq!(state.read_x(4), 0xaabb_ccdd);
        assert_eq!(state.read_x(9), 0x5a);
        assert_eq!(state.read_x(10), 0xbeef);
        assert_eq!(memory.data.get(&256).copied(), Some(153));
        assert_eq!(memory.data.get(&512).copied(), Some(0x1234));
        assert_eq!(memory.data.get(&768).copied(), Some(0xaa));
        assert_eq!(memory.data.get(&1024).copied(), Some(0x4321));
    }

    #[test]
    fn executes_dc_civac_as_noop() {
        let words = [
            (BASE, 0xd280_246a),     // movz x10, #291
            (BASE + 4, 0xd50b_7e2a), // dc civac, x10
            (BASE + 8, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(10), 291);
    }

    #[test]
    fn svc_transitions_to_el1_and_eret_returns_to_el0() {
        let handler_base = BASE + 0x100;
        let words = [
            (BASE, 0xd400_0001),                 // svc #0
            (BASE + 4, 0xd420_0020),             // brk #1
            (handler_base + 0x400, 0xd280_0aa0), // movz x0, #85
            (handler_base + 0x404, 0xd69f_03e0), // eret
        ];

        let mut state = CpuState::default();
        state.pc = BASE;
        state.set_current_el(ExceptionLevel::El0);
        state.vbar_el1 = handler_base;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap {
                    reason: ExitReason::SupervisorCall(imm),
                    ..
                } => {
                    assert_eq!(imm, 0);
                    state.clear_exit();
                    state
                        .take_exception(ExceptionKind::Svc)
                        .expect("svc exception entry should succeed");
                }
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 85);
        assert_eq!(state.current_el(), ExceptionLevel::El0);
        assert_eq!(state.pc, BASE + 8);
    }

    #[test]
    fn executes_fmov_immediate_d_variant() {
        let words = [
            (BASE, 0x1e64_d01f),     // fmov d31, #11.0
            (BASE + 4, 0x9e66_03e0), // fmov x0, d31
            (BASE + 8, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(0), 0x4026_0000_0000_0000);
        assert_eq!(state.read_v128(31), (0x4026_0000_0000_0000, 0));
    }

    #[test]
    fn executes_ucvtf_scvtf_d_reg_variants() {
        let words = [
            (BASE, 0xd280_0541),      // movz x1, #42
            (BASE + 4, 0x9e67_0022),  // fmov d2, x1
            (BASE + 8, 0x7e61_d840),  // ucvtf d0, d2
            (BASE + 12, 0x9e66_0003), // fmov x3, d0
            (BASE + 16, 0x9280_0041), // movn x1, #2
            (BASE + 20, 0x9e67_0026), // fmov d6, x1
            (BASE + 24, 0x5e61_d8c4), // scvtf d4, d6
            (BASE + 28, 0x9e66_0085), // fmov x5, d4
            (BASE + 32, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(3), 0x4045_0000_0000_0000);
        assert_eq!(state.read_x(5), 0xc008_0000_0000_0000);
        assert_eq!(state.read_v128(0), (0x4045_0000_0000_0000, 0));
        assert_eq!(state.read_v128(4), (0xc008_0000_0000_0000, 0));
    }

    #[test]
    fn executes_ucvtf_scvtf_d_from_w_variants() {
        let words = [
            (BASE, 0xd280_0540),      // movz x0, #42
            (BASE + 4, 0x1e62_0001),  // scvtf d1, w0
            (BASE + 8, 0x9e66_0022),  // fmov x2, d1
            (BASE + 12, 0x9280_0040), // movn x0, #2 (w0 = -3)
            (BASE + 16, 0x1e62_0003), // scvtf d3, w0
            (BASE + 20, 0x9e66_0064), // fmov x4, d3
            (BASE + 24, 0xd280_0540), // movz x0, #42
            (BASE + 28, 0x1e63_0005), // ucvtf d5, w0
            (BASE + 32, 0x9e66_00a6), // fmov x6, d5
            (BASE + 36, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x4045_0000_0000_0000);
        assert_eq!(state.read_x(4), 0xc008_0000_0000_0000);
        assert_eq!(state.read_x(6), 0x4045_0000_0000_0000);
    }

    #[test]
    fn executes_fcvt_between_s_and_d_variants() {
        let words = [
            (BASE, 0x1e2e_1000),      // fmov s0, #1.0
            (BASE + 4, 0x1e22_c001),  // fcvt d1, s0
            (BASE + 8, 0x9e66_0022),  // fmov x2, d1
            (BASE + 12, 0x1e64_d003), // fmov d3, #11.0
            (BASE + 16, 0x1e62_4064), // fcvt s4, d3
            (BASE + 20, 0x1e26_0085), // fmov w5, s4
            (BASE + 24, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..10 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(2), 0x3ff0_0000_0000_0000);
        assert_eq!(state.read_x(5), 0x0000_0000_4130_0000);
        assert_eq!(state.read_v128(1), (0x3ff0_0000_0000_0000, 0));
        assert_eq!(state.read_v128(4), (0x0000_0000_4130_0000, 0));
    }

    #[test]
    fn executes_fcvtzu_x_from_d_variant() {
        let words = [
            (BASE, 0x1e64_d000),      // fmov d0, #11.0
            (BASE + 4, 0x9e79_0001),  // fcvtzu x1, d0
            (BASE + 8, 0x1e60_1002),  // fmov d2, #2.0
            (BASE + 12, 0x1e62_1803), // fdiv d3, d0, d2 -> 5.5
            (BASE + 16, 0x9e79_0064), // fcvtzu x4, d3 -> 5
            (BASE + 20, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..10 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 11);
        assert_eq!(state.read_x(4), 5);
    }

    #[test]
    fn executes_fcvtmu_x_from_d_variant() {
        let words = [
            (BASE, 0x1e64_d000),      // fmov d0, #11.0
            (BASE + 4, 0x9e71_0001),  // fcvtmu x1, d0
            (BASE + 8, 0x1e60_1002),  // fmov d2, #2.0
            (BASE + 12, 0x1e62_1803), // fdiv d3, d0, d2 -> 5.5
            (BASE + 16, 0x9e71_0064), // fcvtmu x4, d3 -> 5
            (BASE + 20, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..10 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(1), 11);
        assert_eq!(state.read_x(4), 5);
    }

    #[test]
    fn executes_f64_arithmetic_variants() {
        let words = [
            (BASE, 0x1e64_d000),      // fmov d0, #11.0
            (BASE + 4, 0x1e60_1001),  // fmov d1, #2.0
            (BASE + 8, 0x1e61_2802),  // fadd d2, d0, d1
            (BASE + 12, 0x1e61_3803), // fsub d3, d0, d1
            (BASE + 16, 0x1e61_0804), // fmul d4, d0, d1
            (BASE + 20, 0x1e61_1805), // fdiv d5, d0, d1
            (BASE + 24, 0x1e61_4806), // fmax d6, d0, d1
            (BASE + 28, 0x1e61_5807), // fmin d7, d0, d1
            (BASE + 32, 0x9e66_004a), // fmov x10, d2
            (BASE + 36, 0x9e66_006b), // fmov x11, d3
            (BASE + 40, 0x9e66_008c), // fmov x12, d4
            (BASE + 44, 0x9e66_00ad), // fmov x13, d5
            (BASE + 48, 0x9e66_00ce), // fmov x14, d6
            (BASE + 52, 0x9e66_00ef), // fmov x15, d7
            (BASE + 56, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(10), 0x402a_0000_0000_0000);
        assert_eq!(state.read_x(11), 0x4022_0000_0000_0000);
        assert_eq!(state.read_x(12), 0x4036_0000_0000_0000);
        assert_eq!(state.read_x(13), 0x4016_0000_0000_0000);
        assert_eq!(state.read_x(14), 0x4026_0000_0000_0000);
        assert_eq!(state.read_x(15), 0x4000_0000_0000_0000);
    }

    #[test]
    fn executes_fcmpe_d_variant() {
        let words = [
            (BASE, 0x1e64_d000),      // fmov d0, #11.0
            (BASE + 4, 0x1e60_1001),  // fmov d1, #2.0
            (BASE + 8, 0x1e61_2010),  // fcmpe d0, d1
            (BASE + 12, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..8 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.nzcv, 0x2000_0000);
    }

    #[test]
    fn executes_frintm_d_variant() {
        let words = [
            (BASE, 0x1e64_d000),      // fmov d0, #11.0
            (BASE + 4, 0x1e60_1001),  // fmov d1, #2.0
            (BASE + 8, 0x1e61_1802),  // fdiv d2, d0, d1 -> 5.5
            (BASE + 12, 0x1e74_d004), // fmov d4, #-11.0
            (BASE + 16, 0x1e61_1885), // fdiv d5, d4, d1 -> -5.5
            (BASE + 20, 0x1e65_4046), // frintm d6, d2 -> 5.0
            (BASE + 24, 0x1e65_40a7), // frintm d7, d5 -> -6.0
            (BASE + 28, 0x9e66_00c8), // fmov x8, d6
            (BASE + 32, 0x9e66_00e9), // fmov x9, d7
            (BASE + 36, 0xd420_0020), // brk #1
        ];

        let mut state = CpuState::default();
        state.pc = BASE;

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 24,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let mut trap = None;
        for _ in 0..12 {
            match engine
                .step(&mut state, &mut fetch)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }

        assert_eq!(trap, Some(ExitReason::Breakpoint(1)));
        assert_eq!(state.read_x(8), 0x4014_0000_0000_0000);
        assert_eq!(state.read_x(9), 0xc018_0000_0000_0000);
    }

    #[test]
    fn msr_vbar_el1_traps_at_el0() {
        let words = [
            (BASE, encode_mrs(0, SystemRegister::CurrentEl)),
            (BASE + 4, encode_msr(SystemRegister::VbarEl1, 0)),
            (BASE + 8, 0xd420_0020),
        ];

        let mut state = CpuState::default();
        state.pc = BASE;
        state.set_current_el(ExceptionLevel::El0);

        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut fetch = |pc: u64| {
            words
                .iter()
                .find(|(addr, _)| *addr == pc)
                .map(|(_, op)| *op)
        };

        let result = engine
            .step(&mut state, &mut fetch)
            .expect("step should succeed");

        match result {
            StepResult::Trap { reason, .. } => {
                assert_eq!(reason, ExitReason::PrivilegeFault);
            }
            StepResult::Advanced { .. } => panic!("expected privilege trap"),
        }
    }

    // =====================================================================
    // FP SIMD vector instruction regression tests
    // =====================================================================

    /// Helper: run a short sequence via TestMemory, returning the trap reason.
    fn run_vec_test(memory: &mut TestMemory, state: &mut CpuState) -> Option<ExitReason> {
        let mut engine = DynarecEngine::new(DynarecOptions {
            force_interpreter: false,
            block_max_instructions: 16,
        })
        .expect("dynarec engine should initialize");

        let mut trap = None;
        for _ in 0..16 {
            match engine
                .step_with_memory(state, memory)
                .expect("step should succeed")
            {
                StepResult::Advanced { .. } => continue,
                StepResult::Trap { reason, .. } => {
                    trap = Some(reason);
                    break;
                }
            }
        }
        trap
    }

    // 1. FADD V0.4S, V1.4S, V2.4S
    //    Encoding (FP three-same): Q=1 U=0 s1=0 sz=0 opcode3=010 → 0x4E22_D420
    #[test]
    fn test_vec_fadd_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4E22_D420); // fadd v0.4s, v1.4s, v2.4s
        memory.inst.insert(BASE + 4, 0xd420_0020); // brk #1

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [1.0, 2.0, 3.0, 4.0]
        state.write_v128(1, 0x4000_0000_3F80_0000, 0x4080_0000_4040_0000);
        // V2 = [5.0, 6.0, 7.0, 8.0]
        state.write_v128(2, 0x40C0_0000_40A0_0000, 0x4100_0000_40E0_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [6.0, 8.0, 10.0, 12.0]
        assert_eq!(
            state.read_v128(0),
            (0x4100_0000_40C0_0000, 0x4140_0000_4120_0000)
        );
    }

    // 2. FSUB V0.4S, V1.4S, V2.4S
    //    Encoding: Q=1 U=0 s1=1 sz=0 opcode3=010 → 0x4EA2_D420
    #[test]
    fn test_vec_fsub_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4EA2_D420); // fsub v0.4s, v1.4s, v2.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [5.0, 8.0, 10.0, 16.0]
        state.write_v128(1, 0x4100_0000_40A0_0000, 0x4180_0000_4120_0000);
        // V2 = [1.0, 2.0, 3.0, 4.0]
        state.write_v128(2, 0x4000_0000_3F80_0000, 0x4080_0000_4040_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [4.0, 6.0, 7.0, 12.0]
        assert_eq!(
            state.read_v128(0),
            (0x40C0_0000_4080_0000, 0x4140_0000_40E0_0000)
        );
    }

    // 3. FMUL V0.4S, V1.4S, V2.4S
    //    Encoding: Q=1 U=1 s1=0 sz=0 opcode3=011 → 0x6E22_DC20
    #[test]
    fn test_vec_fmul_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6E22_DC20); // fmul v0.4s, v1.4s, v2.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [2.0, 3.0, 4.0, 5.0]
        state.write_v128(1, 0x4040_0000_4000_0000, 0x40A0_0000_4080_0000);
        // V2 = [3.0, 2.0, 0.5, 2.0]
        state.write_v128(2, 0x4000_0000_4040_0000, 0x4000_0000_3F00_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [6.0, 6.0, 2.0, 10.0]
        assert_eq!(
            state.read_v128(0),
            (0x40C0_0000_40C0_0000, 0x4120_0000_4000_0000)
        );
    }

    // 4. FABS V0.4S, V1.4S
    //    Encoding (FP two-reg misc): Q=1 U=0 size=10 opcode5=01111 → 0x4EA0_F820
    #[test]
    fn test_vec_fabs_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4EA0_F820); // fabs v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [-1.0, 2.0, -3.0, 4.0]
        state.write_v128(1, 0x4000_0000_BF80_0000, 0x4080_0000_C040_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [1.0, 2.0, 3.0, 4.0]
        assert_eq!(
            state.read_v128(0),
            (0x4000_0000_3F80_0000, 0x4080_0000_4040_0000)
        );
    }

    // 5. FNEG V0.4S, V1.4S
    //    Encoding: Q=1 U=1 size=10 opcode5=01111 → 0x6EA0_F820
    #[test]
    fn test_vec_fneg_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6EA0_F820); // fneg v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [1.0, -2.0, 3.0, -4.0]
        state.write_v128(1, 0xC000_0000_3F80_0000, 0xC080_0000_4040_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [-1.0, 2.0, -3.0, 4.0]
        assert_eq!(
            state.read_v128(0),
            (0x4000_0000_BF80_0000, 0x4080_0000_C040_0000)
        );
    }

    // 6. FSQRT V0.4S, V1.4S
    //    Encoding: Q=1 U=1 size=10 opcode5=11111 → 0x6EA1_F820
    #[test]
    fn test_vec_fsqrt_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x6EA1_F820); // fsqrt v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [4.0, 16.0, 1.0, 9.0]
        state.write_v128(1, 0x4180_0000_4080_0000, 0x4110_0000_3F80_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [2.0, 4.0, 1.0, 3.0]
        assert_eq!(
            state.read_v128(0),
            (0x4080_0000_4000_0000, 0x4040_0000_3F80_0000)
        );
    }

    // 7. FCMEQ V0.4S, V1.4S, #0
    //    Encoding: Q=1 U=0 size=10 opcode5=01101 → 0x4EA0_D820
    #[test]
    fn test_vec_fcmeq_zero_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4EA0_D820); // fcmeq v0.4s, v1.4s, #0
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [0.0, 1.0, 0.0, -0.0]
        //  -0.0 == 0.0 in IEEE 754, so element 3 should also match
        state.write_v128(1, 0x3F80_0000_0000_0000, 0x8000_0000_0000_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF]
        assert_eq!(
            state.read_v128(0),
            (0x0000_0000_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF)
        );
    }

    // 8. FCVTZS V0.4S, V1.4S  (FP → signed int, round toward zero)
    //    Encoding: Q=1 U=0 size=10 opcode5=11011 → 0x4EA1_B820
    #[test]
    fn test_vec_fcvtzs_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4EA1_B820); // fcvtzs v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [1.5, -2.5, 3.75, 0.0]
        //  1.5f32 = 0x3FC00000, -2.5f32 = 0xC0200000
        //  3.75f32 = 0x40700000, 0.0f32 = 0x00000000
        state.write_v128(1, 0xC020_0000_3FC0_0000, 0x0000_0000_4070_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected (truncated): [1, -2, 3, 0]
        //  1  = 0x00000001, -2 = 0xFFFFFFFE, 3  = 0x00000003, 0 = 0x00000000
        assert_eq!(
            state.read_v128(0),
            (0xFFFF_FFFE_0000_0001, 0x0000_0000_0000_0003)
        );
    }

    // 9. SCVTF V0.4S, V1.4S  (signed int → FP)
    //    Encoding: Q=1 U=0 size=00 opcode5=11101 → 0x4E21_D820
    #[test]
    fn test_vec_scvtf_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4E21_D820); // scvtf v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 contains signed 32-bit integers: [1, -2, 3, 0]
        //  As u32: 0x00000001, 0xFFFFFFFE, 0x00000003, 0x00000000
        state.write_v128(1, 0xFFFF_FFFE_0000_0001, 0x0000_0000_0000_0003);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [1.0, -2.0, 3.0, 0.0]
        assert_eq!(
            state.read_v128(0),
            (0xC000_0000_3F80_0000, 0x0000_0000_4040_0000)
        );
    }

    // 10. FRINTN V0.4S, V1.4S  (round to nearest, ties to even)
    //     Encoding: Q=1 U=0 size=00 opcode5=11000 → 0x4E21_8820
    #[test]
    fn test_vec_frintn_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4E21_8820); // frintn v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [1.5, 2.5, 3.5, 4.5]
        //  1.5 = 0x3FC00000, 2.5 = 0x40200000
        //  3.5 = 0x40600000, 4.5 = 0x40900000
        state.write_v128(1, 0x4020_0000_3FC0_0000, 0x4090_0000_4060_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected (ties to even): [2.0, 2.0, 4.0, 4.0]
        assert_eq!(
            state.read_v128(0),
            (0x4000_0000_4000_0000, 0x4080_0000_4080_0000)
        );
    }

    // 11. FRECPE V0.4S, V1.4S  (reciprocal estimate — this impl uses exact 1.0/x)
    //     Encoding: Q=1 U=0 size=10 opcode5=11101 → 0x4EA1_D820
    #[test]
    fn test_vec_frecpe_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4EA1_D820); // frecpe v0.4s, v1.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 = [1.0, 2.0, 4.0, 0.5]
        state.write_v128(1, 0x4000_0000_3F80_0000, 0x3F00_0000_4080_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [1.0, 0.5, 0.25, 2.0]
        assert_eq!(
            state.read_v128(0),
            (0x3F00_0000_3F80_0000, 0x4000_0000_3E80_0000)
        );
    }

    // 12. FMLA V0.4S, V1.4S, V2.4S  (Vd += Vn * Vm)
    //     Encoding (FP three-same): Q=1 U=0 s1=0 sz=0 opcode3=001 → 0x4E22_CC20
    #[test]
    fn test_vec_fmla_4s() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x4E22_CC20); // fmla v0.4s, v1.4s, v2.4s
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V0 (accumulator) = [1.0, 2.0, 3.0, 4.0]
        state.write_v128(0, 0x4000_0000_3F80_0000, 0x4080_0000_4040_0000);
        // V1 = [2.0, 3.0, 4.0, 5.0]
        state.write_v128(1, 0x4040_0000_4000_0000, 0x40A0_0000_4080_0000);
        // V2 = [3.0, 2.0, 1.0, 0.5]
        state.write_v128(2, 0x4000_0000_4040_0000, 0x3F00_0000_3F80_0000);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Expected: [1+2*3=7.0, 2+3*2=8.0, 3+4*1=7.0, 4+5*0.5=6.5]
        //  7.0 = 0x40E00000, 8.0 = 0x41000000, 6.5 = 0x40D00000
        assert_eq!(
            state.read_v128(0),
            (0x4100_0000_40E0_0000, 0x40D0_0000_40E0_0000)
        );
    }

    // 13. ADDHN V0.8B, V1.8H, V2.8H  (add-and-narrow-high, 16→8 bit)
    //     Encoding (three-different): Q=0 U=0 size=00 opcode=0100 → 0x0E02_4020
    #[test]
    fn test_simd_addhn_8h() {
        let mut memory = TestMemory::default();
        memory.inst.insert(BASE, 0x0E02_4020); // addhn v0.8b, v1.8h, v2.8h
        memory.inst.insert(BASE + 4, 0xd420_0020);

        let mut state = CpuState::default();
        state.pc = BASE;
        // V1 (8 halfwords): [0x0100, 0x0200, 0x0300, 0x0400, 0x0500, 0x0600, 0x0700, 0x0800]
        state.write_v128(1, 0x0400_0300_0200_0100, 0x0800_0700_0600_0500);
        // V2 (8 halfwords): all 0x0100
        state.write_v128(2, 0x0100_0100_0100_0100, 0x0100_0100_0100_0100);

        assert_eq!(
            run_vec_test(&mut memory, &mut state),
            Some(ExitReason::Breakpoint(1))
        );
        // Each 16-bit sum's high byte: [0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]
        // Q=0 → result in lower 64 bits, upper zeroed
        assert_eq!(state.read_v128(0), (0x0908_0706_0504_0302, 0));
    }

    #[test]
    fn signal_return_marks_dispatcher_interrupt() {
        let mut state = CpuState::default();
        let mut frame =
            crate::arch::JitDispatchFrame::new(&mut state, BASE, 0, 0, std::ptr::null(), 0);
        frame.landing_pc = 1;
        frame.landing_sp = 2;

        TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.set((&mut frame) as *mut _));
        let redirected = unsafe { return_to_dispatcher_from_signal(Some(BASE + 4)) };
        TLS_ACTIVE_DISPATCH_FRAME.with(|slot| slot.set(std::ptr::null_mut()));

        assert!(redirected);
        assert_eq!(state.pc, BASE + 4);
        assert_eq!(state.exit_reason(), ExitReason::DispatcherInterrupt);
    }

    #[test]
    fn signal_fault_classifies_true_memory_fault() {
        let mut memory = ClassifyingMemory::new(0x9000_0000, 0x1000);
        let record = memory.fault_record(0x80, false);
        let fastmem_region = memory.fastmem_region();

        assert_eq!(
            classify_bound_signal_fault(&mut memory, fastmem_region, record),
            FaultClass::MemoryFault
        );
    }

    #[test]
    fn signal_fault_classifies_granularity_mismatch() {
        let mut memory = ClassifyingMemory::new(0x9100_0000, 0x1000);
        memory.allow_rw();
        let record = memory.fault_record(0x40, false);
        let fastmem_region = memory.fastmem_region();

        assert_eq!(
            classify_bound_signal_fault(&mut memory, fastmem_region, record),
            FaultClass::GranularityMismatch
        );
    }
}
