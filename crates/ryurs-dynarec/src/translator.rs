use anyhow::{anyhow, Context, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, AbiParam, BlockArg, ConstantData, InstBuilder, MemFlags, SourceLoc, Value,
};
use cranelift_codegen::ir::{AliasRegion, ArgumentPurpose, Endianness, SigRef};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use std::cell::Cell;
use std::mem;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::decoder::{
    BranchCondition, DecodedBlock, DecodedInstruction, ExtendType, PairAddressMode, ShiftType,
    SystemRegister,
};
use crate::memory::{
    MemoryStrategy, GUEST_PAGE_BITS, GUEST_PAGE_MASK, PTE_FLAG_READ, PTE_FLAG_SPECIAL,
    PTE_FLAG_VALID, PTE_FLAG_WRITE,
};
use crate::runtime::{
    CpuState, ExecutionContext, CPUSTATE_EXCLUSIVE_ADDRESS_OFFSET, CPUSTATE_EXCLUSIVE_SIZE_OFFSET,
    CPUSTATE_EXCLUSIVE_VALID_OFFSET, CPUSTATE_FASTMEM_HOST_PTR_OFFSET,
    CPUSTATE_FASTMEM_SIZE_OFFSET, CPUSTATE_PTE_GUEST_BASE_OFFSET, CPUSTATE_PTE_PAGE_COUNT_OFFSET,
    CPUSTATE_PTE_PTR_OFFSET, CPUSTATE_VREG_BASE_OFFSET, CPUSTATE_VREG_STRIDE,
};

pub type BlockEntry = unsafe extern "C" fn(*mut ExecutionContext, u64) -> u64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HostPcMapEntry {
    pub host_offset_start: u32,
    pub host_offset_end: u32,
    pub guest_pc: u64,
}

#[derive(Clone)]
pub struct CompiledBlock {
    pub entry: BlockEntry,
    pub code_start: usize,
    pub code_size: u32,
    pub host_pc_map: Arc<[HostPcMapEntry]>,
    pub strategy: MemoryStrategy,
    return_address_slot: usize,
    return_link_entry_count: usize,
}

#[repr(C)]
struct ReturnLinkEntry {
    expected_next_pc: AtomicU64,
    linked_target: AtomicUsize,
}

const RETURN_LINK_DISABLED_EXPECTED_PC: u64 = u64::MAX;
const RETURN_LINK_ENTRY_COUNT_SINGLE: usize = 1;
const RETURN_LINK_ENTRY_COUNT_CONDITIONAL: usize = 2;
const RETURN_LINK_ENTRY_COUNT_DYNAMIC: usize = 4;
const RETURN_LINK_ENTRY_COUNT_MAX_OVERRIDE: usize = 16;

impl CompiledBlock {
    fn return_link_entries(&self) -> &[ReturnLinkEntry] {
        // SAFETY: `return_address_slot` is finalized JIT data for this block and
        // outlives the block cache entry.
        unsafe {
            std::slice::from_raw_parts(
                self.return_address_slot as *const ReturnLinkEntry,
                self.return_link_entry_count,
            )
        }
    }

    fn dispatch_target_ptr(&self) -> *const usize {
        let dispatch_offset = self.return_link_entry_count * mem::size_of::<ReturnLinkEntry>();
        (self.return_address_slot + dispatch_offset) as *const usize
    }

    pub fn entry_ptr(&self) -> usize {
        self.entry as *const () as usize
    }

    pub fn return_address_slot_ptr(&self) -> usize {
        self.return_address_slot
    }

    pub fn return_link_entry_count(&self) -> usize {
        self.return_link_entry_count
    }

    pub fn dispatch_target(&self) -> usize {
        // SAFETY: dispatch target is stored in finalized data for this block.
        unsafe { self.dispatch_target_ptr().read() }
    }

    pub fn snapshot_return_links(&self) -> Vec<(u64, usize)> {
        self.return_link_entries()
            .iter()
            .map(|entry| {
                (
                    entry.expected_next_pc.load(Ordering::Acquire),
                    entry.linked_target.load(Ordering::Acquire),
                )
            })
            .collect()
    }

    pub fn initialize_return_link(&self, expected_next_pc: u64, target: usize) {
        if target == 0 {
            return;
        }

        if self.find_entry_by_expected(expected_next_pc).is_some() {
            return;
        }

        let Some(entry) = self.try_claim_empty_entry(expected_next_pc) else {
            return;
        };
        entry.linked_target.store(target, Ordering::Release);
    }

    pub fn patch_return_link(&self, expected_next_pc: u64, target: usize) {
        if target == 0 {
            self.clear_return_link();
            return;
        }

        if let Some(entry) = self.find_entry_by_expected(expected_next_pc) {
            Self::write_link_entry(entry, expected_next_pc, target);
            return;
        }

        if let Some(entry) = self.try_claim_empty_entry(expected_next_pc) {
            entry.linked_target.store(target, Ordering::Release);
            return;
        }

        let entries = self.return_link_entries();
        let evict_index = (expected_next_pc as usize) % self.return_link_entry_count;
        let entry = &entries[evict_index];
        Self::write_link_entry(entry, expected_next_pc, target);
    }

    pub fn clear_return_link(&self) {
        for entry in self.return_link_entries() {
            entry
                .expected_next_pc
                .store(RETURN_LINK_DISABLED_EXPECTED_PC, Ordering::Release);
            entry.linked_target.store(0, Ordering::Release);
        }
    }

    fn find_entry_by_expected(&self, expected_next_pc: u64) -> Option<&ReturnLinkEntry> {
        self.return_link_entries()
            .iter()
            .find(|entry| entry.expected_next_pc.load(Ordering::Acquire) == expected_next_pc)
    }

    fn try_claim_empty_entry(&self, expected_next_pc: u64) -> Option<&ReturnLinkEntry> {
        for entry in self.return_link_entries() {
            match entry.expected_next_pc.compare_exchange(
                RETURN_LINK_DISABLED_EXPECTED_PC,
                expected_next_pc,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(entry),
                Err(v) if v == expected_next_pc => return Some(entry),
                Err(_) => continue,
            }
        }

        None
    }

    fn write_link_entry(entry: &ReturnLinkEntry, expected_next_pc: u64, target: usize) {
        // Disable matching while rewriting this entry.
        entry
            .expected_next_pc
            .store(RETURN_LINK_DISABLED_EXPECTED_PC, Ordering::Release);
        entry.linked_target.store(target, Ordering::Release);
        entry
            .expected_next_pc
            .store(expected_next_pc, Ordering::Release);
    }
}

const REG_FILE_BASE_OFFSET: i32 = 0;
const REG_FILE_STRIDE: i32 = 8;
const SP_OFFSET: i32 = 248;
const PC_OFFSET: i32 = mem::offset_of!(CpuState, pc) as i32;
const NZCV_OFFSET: i32 = 264;
const EXIT_CODE_OFFSET: i32 = mem::offset_of!(CpuState, exit_code) as i32;
const TRAP_IMM_OFFSET: i32 = mem::offset_of!(CpuState, trap_imm) as i32;
const FASTMEM_SIZE_OFFSET: i32 = CPUSTATE_FASTMEM_SIZE_OFFSET;
const FASTMEM_HOST_PTR_OFFSET: i32 = CPUSTATE_FASTMEM_HOST_PTR_OFFSET;
const EXCLUSIVE_VALID_OFFSET: i32 = CPUSTATE_EXCLUSIVE_VALID_OFFSET;
const EXCLUSIVE_SIZE_OFFSET: i32 = CPUSTATE_EXCLUSIVE_SIZE_OFFSET;
const EXCLUSIVE_ADDRESS_OFFSET: i32 = CPUSTATE_EXCLUSIVE_ADDRESS_OFFSET;
const LINK_REGISTER: u8 = 30;
const EXIT_BREAKPOINT: i64 = 1;
const EXIT_MEMORY_FAULT: i64 = 3;
const EXIT_PRIVILEGE_FAULT: i64 = 4;
const EXIT_SVC: i64 = 5;
const EXIT_HVC: i64 = 6;
const EXIT_SMC: i64 = 7;
const EXIT_ERET: i64 = 8;
const EXIT_GRANULARITY_FAULT: i64 = 10;
const RETURN_LINK_ENTRY_STRIDE: i32 = mem::size_of::<ReturnLinkEntry>() as i32;
const RETURN_LINK_ENTRY_EXPECTED_OFFSET: i32 =
    mem::offset_of!(ReturnLinkEntry, expected_next_pc) as i32;
const RETURN_LINK_ENTRY_TARGET_OFFSET: i32 = mem::offset_of!(ReturnLinkEntry, linked_target) as i32;
const NZCV_MASK_N: u8 = 1 << 0;
const NZCV_MASK_Z: u8 = 1 << 1;
const NZCV_MASK_C: u8 = 1 << 2;
const NZCV_MASK_V: u8 = 1 << 3;
const NZCV_MASK_ALL: u8 = NZCV_MASK_N | NZCV_MASK_Z | NZCV_MASK_C | NZCV_MASK_V;

#[derive(Clone, Copy, PartialEq)]
struct HelperRef {
    sig_ref: SigRef,
    raw_ptr: u64,
}

impl HelperRef {
    fn call(self, builder: &mut FunctionBuilder, args: &[Value]) -> cranelift_codegen::ir::Inst {
        let pointer_ty = builder.func.dfg.value_type(args[0]);
        let callee = builder.ins().iconst(pointer_ty, self.raw_ptr as i64);
        builder.ins().call_indirect(self.sig_ref, callee, args)
    }
}

struct MemoryHelperRefs {
    read_u8: HelperRef,
    read_u16: HelperRef,
    read_u32: HelperRef,
    read_u64: HelperRef,
    write_u8: HelperRef,
    write_u16: HelperRef,
    write_u32: HelperRef,
    write_u64: HelperRef,
    record_link_miss: HelperRef,
    state_clrex: HelperRef,
    mrs: HelperRef,
    msr: HelperRef,
}

#[derive(Clone, Copy)]
struct FastmemHelperRefs {
    read_u8: HelperRef,
    read_u16: HelperRef,
    read_u32: HelperRef,
    read_u64: HelperRef,
    write_u8: HelperRef,
    write_u16: HelperRef,
    write_u32: HelperRef,
    write_u64: HelperRef,
}

#[derive(Clone, Copy)]
struct FastmemStateValues {
    pointer_ty: cranelift_codegen::ir::Type,
    guest_base: Value,
    size: Value,
    host_ptr_u64: Value,
}

#[derive(Clone, Copy)]
struct PteStateValues {
    pointer_ty: cranelift_codegen::ir::Type,
    guest_base: Value,
    page_count: Value,
    pte_ptr_u64: Value,
}

#[derive(Clone, Copy)]
enum PendingNzcvRecipe {
    Values {
        n: Value,
        z: Value,
        c: Value,
        v: Value,
    },
    Logic64 {
        result: Value,
    },
    Logic32 {
        result: Value,
    },
    Add64 {
        lhs: Value,
        rhs: Value,
        result: Value,
    },
    Add32 {
        lhs: Value,
        rhs: Value,
        result: Value,
    },
    Sub64 {
        lhs: Value,
        rhs: Value,
        result: Value,
    },
    Sub32 {
        lhs: Value,
        rhs: Value,
        result: Value,
    },
}

#[derive(Clone, Copy)]
struct ReturnLinkLayout {
    entry_count: usize,
    dispatch_target_offset: i32,
    source_pc: u64,
    record_link_miss: HelperRef,
}

type EagerDirectTarget = (u64, usize);
type EagerDirectTargets = [Option<EagerDirectTarget>; 2];

#[derive(Clone, Copy)]
enum FastmemAccessWidth {
    U8,
    U16,
    U32,
    U64,
}

thread_local! {
    static TLS_FASTMEM_HELPERS: Cell<Option<FastmemHelperRefs>> = const { Cell::new(None) };
    static TLS_FASTMEM_STATE: Cell<Option<FastmemStateValues>> = const { Cell::new(None) };
    static TLS_PTE_STATE: Cell<Option<PteStateValues>> = const { Cell::new(None) };
    static TLS_PENDING_NZCV: Cell<Option<PendingNzcvRecipe>> = const { Cell::new(None) };
    static TLS_MEMORY_STRATEGY: Cell<MemoryStrategy> = const { Cell::new(MemoryStrategy::Fastmem) };
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    ))]
    static TLS_RETURN_TO_DISPATCHER_TAIL_SIG: Cell<Option<SigRef>> = const { Cell::new(None) };
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    ))]
    static TLS_RETURN_TARGET_SLOT_GLOBAL: Cell<Option<u64>> = const { Cell::new(None) };
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    ))]
    static TLS_RETURN_LINK_LAYOUT: Cell<Option<ReturnLinkLayout>> = const { Cell::new(None) };
    static TLS_CURRENT_GUEST_PC: Cell<Option<u64>> = const { Cell::new(None) };
    static TLS_ACTIVE_REG_CACHE: Cell<*mut JitRegCache> = const { Cell::new(std::ptr::null_mut()) };
}

fn set_current_guest_pc(pc: Option<u64>) {
    TLS_CURRENT_GUEST_PC.with(|slot| slot.set(pc));
}

fn current_guest_srcloc() -> Option<SourceLoc> {
    let guest_pc = TLS_CURRENT_GUEST_PC.with(|slot| slot.get())?;
    let bits = u32::try_from(guest_pc).ok()?;
    Some(SourceLoc::new(bits))
}

fn current_guest_pc() -> Option<u64> {
    TLS_CURRENT_GUEST_PC.with(|slot| slot.get())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedMovWide {
    U64 { rd: u8, value: u64, consumed: usize },
    U32 { rd: u8, value: u32, consumed: usize },
}

fn try_fuse_mov_wide_sequence(
    instructions: &[DecodedInstruction],
    start: usize,
) -> Option<FusedMovWide> {
    let initial = instructions.get(start)?;

    let mut consumed = 1usize;

    match initial {
        DecodedInstruction::MovZ { rd, imm16, shift } => {
            if *shift >= 64 {
                return None;
            }

            let mut value = (u64::from(*imm16)) << *shift;

            for instruction in instructions.iter().skip(start + 1).take(3) {
                let DecodedInstruction::MovK {
                    rd: next_rd,
                    imm16: next_imm16,
                    shift: next_shift,
                } = instruction
                else {
                    break;
                };

                if next_rd != rd || *next_shift >= 64 {
                    break;
                }

                let clear_mask = !((0xffffu64) << *next_shift);
                value = (value & clear_mask) | ((u64::from(*next_imm16)) << *next_shift);
                consumed += 1;
            }

            (consumed >= 2).then_some(FusedMovWide::U64 {
                rd: *rd,
                value,
                consumed,
            })
        }
        DecodedInstruction::MovN { rd, imm16, shift } => {
            if *shift >= 64 {
                return None;
            }

            let mut value = !((u64::from(*imm16)) << *shift);

            for instruction in instructions.iter().skip(start + 1).take(3) {
                let DecodedInstruction::MovK {
                    rd: next_rd,
                    imm16: next_imm16,
                    shift: next_shift,
                } = instruction
                else {
                    break;
                };

                if next_rd != rd || *next_shift >= 64 {
                    break;
                }

                let clear_mask = !((0xffffu64) << *next_shift);
                value = (value & clear_mask) | ((u64::from(*next_imm16)) << *next_shift);
                consumed += 1;
            }

            (consumed >= 2).then_some(FusedMovWide::U64 {
                rd: *rd,
                value,
                consumed,
            })
        }
        DecodedInstruction::MovZ32 { rd, imm16, shift } => {
            if *shift >= 32 {
                return None;
            }

            let mut value = (u32::from(*imm16)) << *shift;

            for instruction in instructions.iter().skip(start + 1).take(1) {
                let DecodedInstruction::MovK32 {
                    rd: next_rd,
                    imm16: next_imm16,
                    shift: next_shift,
                } = instruction
                else {
                    break;
                };

                if next_rd != rd || *next_shift >= 32 {
                    break;
                }

                let clear_mask = !((0xffffu32) << *next_shift);
                value = (value & clear_mask) | ((u32::from(*next_imm16)) << *next_shift);
                consumed += 1;
            }

            (consumed >= 2).then_some(FusedMovWide::U32 {
                rd: *rd,
                value,
                consumed,
            })
        }
        DecodedInstruction::MovN32 { rd, imm16, shift } => {
            if *shift >= 32 {
                return None;
            }

            let mut value = !((u32::from(*imm16)) << *shift);

            for instruction in instructions.iter().skip(start + 1).take(1) {
                let DecodedInstruction::MovK32 {
                    rd: next_rd,
                    imm16: next_imm16,
                    shift: next_shift,
                } = instruction
                else {
                    break;
                };

                if next_rd != rd || *next_shift >= 32 {
                    break;
                }

                let clear_mask = !((0xffffu32) << *next_shift);
                value = (value & clear_mask) | ((u32::from(*next_imm16)) << *next_shift);
                consumed += 1;
            }

            (consumed >= 2).then_some(FusedMovWide::U32 {
                rd: *rd,
                value,
                consumed,
            })
        }
        _ => None,
    }
}

struct FastmemHelperBinding;

struct FastmemStateBinding;
struct ActiveRegCacheBinding;
struct PteStateBinding;
struct MemoryStrategyBinding;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
struct ReturnTargetSlotBinding;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
struct ReturnLinkLayoutBinding;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
struct ReturnToDispatcherTailSigBinding;

impl FastmemHelperBinding {
    fn bind(memory_helpers: &MemoryHelperRefs) -> Self {
        let refs = FastmemHelperRefs {
            read_u8: memory_helpers.read_u8,
            read_u16: memory_helpers.read_u16,
            read_u32: memory_helpers.read_u32,
            read_u64: memory_helpers.read_u64,
            write_u8: memory_helpers.write_u8,
            write_u16: memory_helpers.write_u16,
            write_u32: memory_helpers.write_u32,
            write_u64: memory_helpers.write_u64,
        };

        TLS_FASTMEM_HELPERS.with(|slot| slot.set(Some(refs)));
        Self
    }
}

impl Drop for FastmemHelperBinding {
    fn drop(&mut self) {
        TLS_FASTMEM_HELPERS.with(|slot| slot.set(None));
    }
}

impl FastmemStateBinding {
    fn bind(values: FastmemStateValues) -> Self {
        TLS_FASTMEM_STATE.with(|slot| slot.set(Some(values)));
        Self
    }
}

impl Drop for FastmemStateBinding {
    fn drop(&mut self) {
        TLS_FASTMEM_STATE.with(|slot| slot.set(None));
    }
}

impl ActiveRegCacheBinding {
    fn bind(reg_cache: &mut JitRegCache) -> Self {
        TLS_ACTIVE_REG_CACHE.with(|slot| slot.set(reg_cache as *mut JitRegCache));
        Self
    }
}

impl Drop for ActiveRegCacheBinding {
    fn drop(&mut self) {
        TLS_ACTIVE_REG_CACHE.with(|slot| slot.set(std::ptr::null_mut()));
    }
}

impl PteStateBinding {
    fn bind(values: PteStateValues) -> Self {
        TLS_PTE_STATE.with(|slot| slot.set(Some(values)));
        Self
    }
}

impl Drop for PteStateBinding {
    fn drop(&mut self) {
        TLS_PTE_STATE.with(|slot| slot.set(None));
    }
}

impl MemoryStrategyBinding {
    fn bind(strategy: MemoryStrategy) -> Self {
        TLS_MEMORY_STRATEGY.with(|slot| slot.set(strategy));
        Self
    }
}

impl Drop for MemoryStrategyBinding {
    fn drop(&mut self) {
        TLS_MEMORY_STRATEGY.with(|slot| slot.set(MemoryStrategy::Fastmem));
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl ReturnTargetSlotBinding {
    fn bind(slot_addr: u64) -> Self {
        TLS_RETURN_TARGET_SLOT_GLOBAL.with(|slot| slot.set(Some(slot_addr)));
        Self
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl Drop for ReturnTargetSlotBinding {
    fn drop(&mut self) {
        TLS_RETURN_TARGET_SLOT_GLOBAL.with(|slot| slot.set(None));
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl ReturnLinkLayoutBinding {
    fn bind(layout: ReturnLinkLayout) -> Self {
        TLS_RETURN_LINK_LAYOUT.with(|slot| slot.set(Some(layout)));
        Self
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl Drop for ReturnLinkLayoutBinding {
    fn drop(&mut self) {
        TLS_RETURN_LINK_LAYOUT.with(|slot| slot.set(None));
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl ReturnToDispatcherTailSigBinding {
    fn bind(sig_ref: SigRef) -> Self {
        TLS_RETURN_TO_DISPATCHER_TAIL_SIG.with(|slot| slot.set(Some(sig_ref)));
        Self
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
impl Drop for ReturnToDispatcherTailSigBinding {
    fn drop(&mut self) {
        TLS_RETURN_TO_DISPATCHER_TAIL_SIG.with(|slot| slot.set(None));
    }
}

fn current_fastmem_state() -> Option<FastmemStateValues> {
    TLS_FASTMEM_STATE.with(|slot| slot.get())
}

fn current_pte_state() -> Option<PteStateValues> {
    TLS_PTE_STATE.with(|slot| slot.get())
}

fn current_memory_strategy() -> MemoryStrategy {
    TLS_MEMORY_STRATEGY.with(|slot| slot.get())
}

fn with_active_reg_cache<R>(f: impl FnOnce(&mut JitRegCache) -> R) -> Option<R> {
    TLS_ACTIVE_REG_CACHE.with(|slot| {
        let ptr = slot.get();
        if ptr.is_null() {
            None
        } else {
            // SAFETY: bound for the current compile invocation only.
            Some(f(unsafe { &mut *ptr }))
        }
    })
}

fn current_pending_nzcv() -> Option<PendingNzcvRecipe> {
    TLS_PENDING_NZCV.with(|slot| slot.get())
}

fn set_pending_nzcv(recipe: PendingNzcvRecipe) {
    TLS_PENDING_NZCV.with(|slot| slot.set(Some(recipe)));
}

fn clear_pending_nzcv() {
    TLS_PENDING_NZCV.with(|slot| slot.set(None));
}

fn unsafe_direct_memory_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RYURS_UNSAFE_DIRECT_MEMORY")
            .map(|value| {
                let lowered = value.trim().to_ascii_lowercase();
                lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on"
            })
            .unwrap_or(false)
    })
}

fn parse_return_link_entry_count_override(value: &str) -> Option<usize> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return None;
    }

    match trimmed.parse::<usize>() {
        Ok(count) if (1..=RETURN_LINK_ENTRY_COUNT_MAX_OVERRIDE).contains(&count) => Some(count),
        _ => None,
    }
}

fn return_link_entry_count_override() -> Option<usize> {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("RYURS_RETURN_LINK_ENTRY_COUNT")
            .ok()
            .and_then(|value| parse_return_link_entry_count_override(&value))
    })
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
fn current_return_to_dispatcher_tail_sig() -> Option<SigRef> {
    TLS_RETURN_TO_DISPATCHER_TAIL_SIG.with(|slot| slot.get())
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
fn current_return_target_slot_addr() -> Option<u64> {
    TLS_RETURN_TARGET_SLOT_GLOBAL.with(|slot| slot.get())
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
fn current_return_link_layout() -> Option<ReturnLinkLayout> {
    TLS_RETURN_LINK_LAYOUT.with(|slot| slot.get())
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
fn import_return_to_dispatcher_tail_signature(
    func: &mut cranelift_codegen::ir::Function,
    pointer_ty: cranelift_codegen::ir::Type,
) -> SigRef {
    let mut signature = cranelift_codegen::ir::Signature::new(CallConv::Tail);
    signature.params.push(AbiParam::new(pointer_ty));
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    func.import_signature(signature)
}

fn fastmem_read_width(helper: HelperRef) -> Option<FastmemAccessWidth> {
    TLS_FASTMEM_HELPERS.with(|slot| {
        let refs = slot.get()?;
        if helper == refs.read_u8 {
            Some(FastmemAccessWidth::U8)
        } else if helper == refs.read_u16 {
            Some(FastmemAccessWidth::U16)
        } else if helper == refs.read_u32 {
            Some(FastmemAccessWidth::U32)
        } else if helper == refs.read_u64 {
            Some(FastmemAccessWidth::U64)
        } else {
            None
        }
    })
}

fn fastmem_write_width(helper: HelperRef) -> Option<FastmemAccessWidth> {
    TLS_FASTMEM_HELPERS.with(|slot| {
        let refs = slot.get()?;
        if helper == refs.write_u8 {
            Some(FastmemAccessWidth::U8)
        } else if helper == refs.write_u16 {
            Some(FastmemAccessWidth::U16)
        } else if helper == refs.write_u32 {
            Some(FastmemAccessWidth::U32)
        } else if helper == refs.write_u64 {
            Some(FastmemAccessWidth::U64)
        } else {
            None
        }
    })
}

fn fastmem_access_bytes(width: FastmemAccessWidth) -> i64 {
    match width {
        FastmemAccessWidth::U8 => 1,
        FastmemAccessWidth::U16 => 2,
        FastmemAccessWidth::U32 => 4,
        FastmemAccessWidth::U64 => 8,
    }
}

fn return_link_entry_count_for_block(block: &DecodedBlock) -> usize {
    if let Some(forced) = return_link_entry_count_override() {
        return forced;
    }

    match block.instructions.last() {
        Some(
            DecodedInstruction::Br { .. }
            | DecodedInstruction::Blr { .. }
            | DecodedInstruction::Ret { .. },
        ) => RETURN_LINK_ENTRY_COUNT_DYNAMIC,
        Some(
            DecodedInstruction::BCond { .. }
            | DecodedInstruction::Cbz { .. }
            | DecodedInstruction::Cbnz { .. }
            | DecodedInstruction::Cbz32 { .. }
            | DecodedInstruction::Cbnz32 { .. }
            | DecodedInstruction::Tbz { .. }
            | DecodedInstruction::Tbnz { .. },
        ) => RETURN_LINK_ENTRY_COUNT_CONDITIONAL,
        Some(DecodedInstruction::B { .. } | DecodedInstruction::Bl { .. }) => {
            RETURN_LINK_ENTRY_COUNT_SINGLE
        }
        _ => RETURN_LINK_ENTRY_COUNT_DYNAMIC,
    }
}

pub struct BlockCompiler {
    isa: Arc<dyn cranelift_codegen::isa::TargetIsa>,
    arena: Arc<crate::jit_arena::JitArena>,
    ctx: CodegenContext,
    fb_ctx: FunctionBuilderContext,
}

impl BlockCompiler {
    pub fn new(arena: Arc<crate::jit_arena::JitArena>) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .context("failed to configure use_colocated_libcalls")?;
        flag_builder
            .set("is_pic", "false")
            .context("failed to configure is_pic")?;
        flag_builder
            .set("opt_level", "speed")
            .context("failed to configure opt_level")?;
        #[cfg(target_arch = "x86_64")]
        flag_builder
            .set("preserve_frame_pointers", "true")
            .context("failed to configure preserve_frame_pointers")?;
        #[cfg(not(debug_assertions))]
        flag_builder
            .set("enable_verifier", "false")
            .context("failed to configure enable_verifier")?;

        let isa_builder =
            cranelift_native::builder().map_err(|msg| anyhow!("unsupported host ISA: {msg}"))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .context("failed to build target ISA")?;

        Ok(Self {
            isa,
            arena,
            ctx: CodegenContext::new(),
            fb_ctx: FunctionBuilderContext::new(),
        })
    }

    pub fn compile(
        &mut self,
        block: &DecodedBlock,
        eager_direct_targets: EagerDirectTargets,
        strategy: MemoryStrategy,
    ) -> Result<CompiledBlock> {
        self.ctx.clear();
        self.ctx.func.signature.call_conv = self.isa.default_call_conv();
        clear_pending_nzcv();

        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        {
            // Required so return_call_indirect can tail-transfer to the dispatcher landing.
            self.ctx.func.signature.call_conv = CallConv::Tail;
        }

        let pointer_ty = self.isa.pointer_type();
        let return_link_entry_count = return_link_entry_count_for_block(block);
        let return_link_dispatch_offset =
            i32::try_from(return_link_entry_count * mem::size_of::<ReturnLinkEntry>())
                .context("return-link dispatch offset exceeds i32 range")?;

        // Pre-allocate the return-link slot in the arena's data region.
        let slot_size =
            return_link_entry_count * mem::size_of::<ReturnLinkEntry>() + mem::size_of::<usize>();
        let slot_align = mem::align_of::<ReturnLinkEntry>().max(mem::align_of::<usize>());
        let return_slot_ptr = self
            .arena
            .alloc_data(slot_size, slot_align)
            .context("failed to allocate return-link slot in JIT arena")?;

        // Initialize the slot contents.
        {
            let landing_target = crate::arch::dispatcher_landing_target();
            let mut offset = 0usize;
            for _ in 0..return_link_entry_count {
                unsafe {
                    std::ptr::write_unaligned(
                        return_slot_ptr.add(offset) as *mut u64,
                        RETURN_LINK_DISABLED_EXPECTED_PC,
                    );
                    offset += mem::size_of::<u64>();
                    std::ptr::write_unaligned(return_slot_ptr.add(offset) as *mut usize, 0usize);
                    offset += mem::size_of::<usize>();
                }
            }
            unsafe {
                std::ptr::write_unaligned(
                    return_slot_ptr.add(offset) as *mut usize,
                    landing_target,
                );
            }
        }
        let return_slot_addr = return_slot_ptr as u64;

        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        let _return_target_slot_binding = ReturnTargetSlotBinding::bind(return_slot_addr);
        let memory_helpers = self.declare_memory_helpers(pointer_ty)?;
        let _fastmem_helper_binding = FastmemHelperBinding::bind(&memory_helpers);
        let _memory_strategy_binding = MemoryStrategyBinding::bind(strategy);
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        let _return_link_layout_binding = ReturnLinkLayoutBinding::bind(ReturnLinkLayout {
            entry_count: return_link_entry_count,
            dispatch_target_offset: return_link_dispatch_offset,
            source_pc: block.start_pc,
            record_link_miss: memory_helpers.record_link_miss,
        });
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        let _return_to_dispatcher_tail_sig_binding = ReturnToDispatcherTailSigBinding::bind(
            import_return_to_dispatcher_tail_signature(&mut self.ctx.func, pointer_ty),
        );
        let mut cpu_state_param = AbiParam::new(pointer_ty);
        cpu_state_param.purpose = ArgumentPurpose::VMContext;
        self.ctx.func.signature.params.push(cpu_state_param);
        self.ctx
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64));
        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(types::I64));

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.fb_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let state_ptr = builder.block_params(entry_block)[0];
            let fastmem_guest_base = builder.block_params(entry_block)[1];
            let fastmem_size =
                builder
                    .ins()
                    .load(types::I64, MemFlags::new(), state_ptr, FASTMEM_SIZE_OFFSET);
            let fastmem_host_ptr_u64 = builder.ins().load(
                types::I64,
                MemFlags::new(),
                state_ptr,
                FASTMEM_HOST_PTR_OFFSET,
            );
            let pte_guest_base = builder.ins().load(
                types::I64,
                MemFlags::new(),
                state_ptr,
                CPUSTATE_PTE_GUEST_BASE_OFFSET,
            );
            let pte_page_count = builder.ins().load(
                types::I64,
                MemFlags::new(),
                state_ptr,
                CPUSTATE_PTE_PAGE_COUNT_OFFSET,
            );
            let pte_ptr_u64 = builder.ins().load(
                types::I64,
                MemFlags::new(),
                state_ptr,
                CPUSTATE_PTE_PTR_OFFSET,
            );
            let _fastmem_state_binding = FastmemStateBinding::bind(FastmemStateValues {
                pointer_ty,
                guest_base: fastmem_guest_base,
                size: fastmem_size,
                host_ptr_u64: fastmem_host_ptr_u64,
            });
            let _pte_state_binding = PteStateBinding::bind(PteStateValues {
                pointer_ty,
                guest_base: pte_guest_base,
                page_count: pte_page_count,
                pte_ptr_u64,
            });

            let mut reg_cache = JitRegCache::new();
            let _active_reg_cache_binding = ActiveRegCacheBinding::bind(&mut reg_cache);
            let mut terminated = false;

            let mut index = 0usize;
            while index < block.instructions.len() {
                if let Some(fused) = try_fuse_mov_wide_sequence(&block.instructions, index) {
                    match fused {
                        FusedMovWide::U64 {
                            rd,
                            value,
                            consumed,
                        } => {
                            let folded = builder.ins().iconst(types::I64, value as i64);
                            reg_cache.write(rd, folded);
                            index += consumed;
                            continue;
                        }
                        FusedMovWide::U32 {
                            rd,
                            value,
                            consumed,
                        } => {
                            let folded32 =
                                builder.ins().iconst(types::I32, i64::from(value as i32));
                            let folded = builder.ins().uextend(types::I64, folded32);
                            reg_cache.write(rd, folded);
                            index += consumed;
                            continue;
                        }
                    }
                }

                let instruction = &block.instructions[index];
                let instr_pc = block.start_pc.wrapping_add((index as u64) * 4);
                set_current_guest_pc(Some(instr_pc));

                match instruction {
                    // ── NOP / Hints / Barriers ──
                    DecodedInstruction::Nop
                    | DecodedInstruction::Yield
                    | DecodedInstruction::Wfe
                    | DecodedInstruction::Wfi
                    | DecodedInstruction::Sev
                    | DecodedInstruction::Sevl
                    | DecodedInstruction::Xpaclri
                    | DecodedInstruction::Bti { .. }
                    | DecodedInstruction::DcZva { .. }
                    | DecodedInstruction::DcCivac { .. }
                    | DecodedInstruction::Dmb { .. }
                    | DecodedInstruction::Dsb { .. }
                    | DecodedInstruction::Isb { .. } => {}

                    // ── FP Scalar Conversions (int → float) ──
                    DecodedInstruction::ScvtfS32 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        let converted = builder.ins().fcvt_from_sint(types::F32, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfS32 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        let converted = builder.ins().fcvt_from_uint(types::F32, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfD32 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        let converted = builder.ins().fcvt_from_uint(types::F64, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::ScvtfD32 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        let converted = builder.ins().fcvt_from_sint(types::F64, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfSReg32 { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_uint(types::F32, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfDReg64 { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_uint(types::F64, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::ScvtfSReg32 { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_sint(types::F32, source32);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::ScvtfDReg64 { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_sint(types::F64, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }
                    // 64-bit integer → float conversions (SCVTF/UCVTF with X register source)
                    DecodedInstruction::ScvtfS64 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_sint(types::F32, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::ScvtfD64 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_sint(types::F64, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfS64 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_uint(types::F32, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, converted_bits);
                    }
                    DecodedInstruction::UcvtfD64 { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let converted = builder.ins().fcvt_from_uint(types::F64, source64);
                        let converted_bits = bitcast_little(&mut builder, converted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, converted_bits);
                    }

                    // ── FP Vector Three-Same (22 flattened variants) ──
                    DecodedInstruction::VecFadd { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fadd,
                        );
                    }
                    DecodedInstruction::VecFsub { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fsub,
                        );
                    }
                    DecodedInstruction::VecFmul { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmul,
                        );
                    }
                    DecodedInstruction::VecFdiv { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fdiv,
                        );
                    }
                    DecodedInstruction::VecFmax { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmax,
                        );
                    }
                    DecodedInstruction::VecFmin { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmin,
                        );
                    }
                    DecodedInstruction::VecFmaxnm { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmaxnm,
                        );
                    }
                    DecodedInstruction::VecFminnm { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fminnm,
                        );
                    }
                    DecodedInstruction::VecFmulx { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmulx,
                        );
                    }
                    DecodedInstruction::VecFabd { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fabd,
                        );
                    }
                    DecodedInstruction::VecFaddp { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Faddp,
                        );
                    }
                    DecodedInstruction::VecFmaxp { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmaxp,
                        );
                    }
                    DecodedInstruction::VecFminp { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fminp,
                        );
                    }
                    DecodedInstruction::VecFrecps { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Frecps,
                        );
                    }
                    DecodedInstruction::VecFrsqrts { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Frsqrts,
                        );
                    }
                    DecodedInstruction::VecFcmeq { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fcmeq,
                        );
                    }
                    DecodedInstruction::VecFcmge { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fcmge,
                        );
                    }
                    DecodedInstruction::VecFcmgt { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fcmgt,
                        );
                    }
                    DecodedInstruction::VecFacge { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Facge,
                        );
                    }
                    DecodedInstruction::VecFacgt { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Facgt,
                        );
                    }
                    DecodedInstruction::VecFmla { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmla,
                        );
                    }
                    DecodedInstruction::VecFmls { vd, vn, vm, q, sz } => {
                        emit_fp_vector_three_same(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *q,
                            *sz,
                            FpVecOp::Fmls,
                        );
                    }

                    // ── FP Vector × Indexed Element (4 flattened variants) ──
                    DecodedInstruction::VecFmulElem {
                        vd,
                        vn,
                        vm,
                        index,
                        q,
                        sz,
                    } => {
                        emit_fp_vector_by_element(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *index,
                            *q,
                            *sz,
                            FpVecElemOp::Fmul,
                        );
                    }
                    DecodedInstruction::VecFmlaElem {
                        vd,
                        vn,
                        vm,
                        index,
                        q,
                        sz,
                    } => {
                        emit_fp_vector_by_element(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *index,
                            *q,
                            *sz,
                            FpVecElemOp::Fmla,
                        );
                    }
                    DecodedInstruction::VecFmlsElem {
                        vd,
                        vn,
                        vm,
                        index,
                        q,
                        sz,
                    } => {
                        emit_fp_vector_by_element(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *index,
                            *q,
                            *sz,
                            FpVecElemOp::Fmls,
                        );
                    }
                    DecodedInstruction::VecFmulxElem {
                        vd,
                        vn,
                        vm,
                        index,
                        q,
                        sz,
                    } => {
                        emit_fp_vector_by_element(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *vm,
                            *index,
                            *q,
                            *sz,
                            FpVecElemOp::Fmulx,
                        );
                    }

                    // ── FP Binary Scalar Ops ──
                    DecodedInstruction::FaddS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Add,
                            false,
                        );
                    }
                    DecodedInstruction::FsubS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Sub,
                            false,
                        );
                    }
                    DecodedInstruction::FmulS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Mul,
                            false,
                        );
                    }
                    DecodedInstruction::FmaxS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Max,
                            false,
                        );
                    }
                    DecodedInstruction::FminS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Min,
                            false,
                        );
                    }
                    DecodedInstruction::FmaxnmS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Maxnm,
                            false,
                        );
                    }
                    DecodedInstruction::FminnmS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Minnm,
                            false,
                        );
                    }
                    DecodedInstruction::FdivS { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Div,
                            false,
                        );
                    }
                    DecodedInstruction::FaddD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Add,
                            true,
                        );
                    }
                    DecodedInstruction::FsubD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Sub,
                            true,
                        );
                    }
                    DecodedInstruction::FmulD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Mul,
                            true,
                        );
                    }
                    DecodedInstruction::FmaxD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Max,
                            true,
                        );
                    }
                    DecodedInstruction::FminD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Min,
                            true,
                        );
                    }
                    DecodedInstruction::FmaxnmD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Maxnm,
                            true,
                        );
                    }
                    DecodedInstruction::FminnmD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Minnm,
                            true,
                        );
                    }
                    DecodedInstruction::FdivD { rd, rn, rm } => {
                        emit_fp_binop(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *rd,
                            *rn,
                            *rm,
                            FpBinOp::Div,
                            true,
                        );
                    }
                    DecodedInstruction::FnmulS { rd, rn, rm } => {
                        // FNMUL: -(Rn * Rm)
                        let lhs_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let rhs_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rm);
                        let lhs = bitcast_little(&mut builder, lhs_bits, types::F32);
                        let rhs = bitcast_little(&mut builder, rhs_bits, types::F32);
                        let product = builder.ins().fmul(lhs, rhs);
                        let negated = builder.ins().fneg(product);
                        let result_bits = bitcast_little(&mut builder, negated, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FnmulD { rd, rn, rm } => {
                        let lhs_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let rhs_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rm);
                        let lhs = bitcast_little(&mut builder, lhs_bits, types::F64);
                        let rhs = bitcast_little(&mut builder, rhs_bits, types::F64);
                        let product = builder.ins().fmul(lhs, rhs);
                        let negated = builder.ins().fneg(product);
                        let result_bits = bitcast_little(&mut builder, negated, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }

                    // ── FP Moves & Immediates ──
                    DecodedInstruction::FmovImmS { rd, imm8 } => {
                        let imm_bits = expand_fmov_imm8_f32(*imm8);
                        let value = builder.ins().iconst(types::I32, i64::from(imm_bits as i32));
                        reg_cache.write_v_scalar32(&mut builder, *rd, value);
                    }
                    DecodedInstruction::FmovImmD { rd, imm8 } => {
                        let imm_bits = expand_fmov_imm8_f64(*imm8);
                        let value = builder.ins().iconst(types::I64, imm_bits as i64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, value);
                    }
                    DecodedInstruction::FmovSFromW { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        reg_cache.write_v_scalar32(&mut builder, *rd, source32);
                    }
                    DecodedInstruction::FmovWFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source64 = builder.ins().uextend(types::I64, source32);
                        reg_cache.write(*rd, source64);
                    }
                    DecodedInstruction::FmovDFromX { rd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        reg_cache.write_v_scalar64(&mut builder, *rd, source64);
                    }
                    DecodedInstruction::FmovXFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        reg_cache.write(*rd, source64);
                    }

                    // ── FP Conversions Between Sizes ──
                    DecodedInstruction::FcvtDFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let promoted = builder.ins().fpromote(types::F64, source_f32);
                        let result_bits = bitcast_little(&mut builder, promoted, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FcvtSFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let demoted = builder.ins().fdemote(types::F32, source_f64);
                        let result_bits = bitcast_little(&mut builder, demoted, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FcvtzuXFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let result = builder.ins().fcvt_to_uint_sat(types::I64, source_f64);
                        reg_cache.write(*rd, result);
                    }
                    // FCVTZS (scalar integer) - FP to signed int, round toward zero
                    DecodedInstruction::FcvtzsWFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let result32 = builder.ins().fcvt_to_sint_sat(types::I32, source_f32);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsXFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let promoted = builder.ins().fpromote(types::F64, source_f32);
                        let result = builder.ins().fcvt_to_sint_sat(types::I64, promoted);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsWFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let result32 = builder.ins().fcvt_to_sint_sat(types::I32, source_f64);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsXFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let result = builder.ins().fcvt_to_sint_sat(types::I64, source_f64);
                        reg_cache.write(*rd, result);
                    }
                    // FCVTZU (scalar integer) - FP to unsigned int, round toward zero
                    DecodedInstruction::FcvtzuWFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let result32 = builder.ins().fcvt_to_uint_sat(types::I32, source_f32);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuXFromS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let promoted = builder.ins().fpromote(types::F64, source_f32);
                        let result = builder.ins().fcvt_to_uint_sat(types::I64, promoted);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuWFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let result32 = builder.ins().fcvt_to_uint_sat(types::I32, source_f64);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsFixedWFromS { rd, rn, fbits } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let scale = builder.ins().f32const((1u64 << *fbits) as f32);
                        let scaled = builder.ins().fmul(source_f32, scale);
                        let result32 = builder.ins().fcvt_to_sint_sat(types::I32, scaled);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsFixedXFromS { rd, rn, fbits } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let promoted = builder.ins().fpromote(types::F64, source_f32);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(promoted, scale);
                        let result = builder.ins().fcvt_to_sint_sat(types::I64, scaled);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsFixedWFromD { rd, rn, fbits } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(source_f64, scale);
                        let result32 = builder.ins().fcvt_to_sint_sat(types::I32, scaled);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzsFixedXFromD { rd, rn, fbits } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(source_f64, scale);
                        let result = builder.ins().fcvt_to_sint_sat(types::I64, scaled);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuFixedWFromS { rd, rn, fbits } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let scale = builder.ins().f32const((1u64 << *fbits) as f32);
                        let scaled = builder.ins().fmul(source_f32, scale);
                        let result32 = builder.ins().fcvt_to_uint_sat(types::I32, scaled);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuFixedXFromS { rd, rn, fbits } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let promoted = builder.ins().fpromote(types::F64, source_f32);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(promoted, scale);
                        let result = builder.ins().fcvt_to_uint_sat(types::I64, scaled);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuFixedWFromD { rd, rn, fbits } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(source_f64, scale);
                        let result32 = builder.ins().fcvt_to_uint_sat(types::I32, scaled);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtzuFixedXFromD { rd, rn, fbits } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let scale = builder.ins().f64const((1u64 << *fbits) as f64);
                        let scaled = builder.ins().fmul(source_f64, scale);
                        let result = builder.ins().fcvt_to_uint_sat(types::I64, scaled);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::FcvtmuXFromD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let floored = builder.ins().floor(source_f64);
                        let result = builder.ins().fcvt_to_uint_sat(types::I64, floored);
                        reg_cache.write(*rd, result);
                    }

                    // ── Vector Moves / Duplicates ──
                    DecodedInstruction::MovXFromVD { rd, vn, index } => {
                        let vector = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let lane = builder.ins().extractlane(vector, *index);
                        reg_cache.write(*rd, lane);
                    }
                    DecodedInstruction::MovVD1FromX { vd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let vector = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let updated = builder.ins().insertlane(vector, source64, 1);
                        reg_cache.write_v(*vd, updated);
                    }
                    DecodedInstruction::InsGeneralVector {
                        vd,
                        rn,
                        dst_index,
                        size,
                    } => {
                        let vec_ty = match size {
                            0 => types::I8X16,
                            1 => types::I16X8,
                            2 => types::I32X4,
                            3 => types::I64X2,
                            _ => unreachable!(),
                        };
                        let elem_ty = match size {
                            0 => types::I8,
                            1 => types::I16,
                            2 => types::I32,
                            3 => types::I64,
                            _ => unreachable!(),
                        };
                        let src_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let elem = if *size == 3 {
                            src_raw
                        } else {
                            builder.ins().ireduce(elem_ty, src_raw)
                        };
                        let dst_vec = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let dst_typed = bitcast_little(&mut builder, dst_vec, vec_ty);
                        let updated = builder.ins().insertlane(dst_typed, elem, *dst_index);
                        let result = bitcast_little(&mut builder, updated, types::I64X2);
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::InsElement {
                        vd,
                        vn,
                        dst_index,
                        src_index,
                        size,
                    } => {
                        let vec_ty = match size {
                            0 => types::I8X16,
                            1 => types::I16X8,
                            2 => types::I32X4,
                            3 => types::I64X2,
                            _ => unreachable!(),
                        };
                        let src_vec = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let src_typed = bitcast_little(&mut builder, src_vec, vec_ty);
                        let elem = builder.ins().extractlane(src_typed, *src_index);
                        let dst_vec = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let dst_typed = bitcast_little(&mut builder, dst_vec, vec_ty);
                        let updated = builder.ins().insertlane(dst_typed, elem, *dst_index);
                        let result = bitcast_little(&mut builder, updated, types::I64X2);
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::DupV16BFromW { vd, rn } => {
                        let source = reg_cache.read(&mut builder, state_ptr, *rn);
                        let byte = builder.ins().ireduce(types::I8, source);
                        let byte64 = builder.ins().uextend(types::I64, byte);
                        let shifted8 = builder.ins().ishl_imm(byte64, 8);
                        let repeated16 = builder.ins().bor(byte64, shifted8);
                        let shifted16 = builder.ins().ishl_imm(repeated16, 16);
                        let repeated32 = builder.ins().bor(repeated16, shifted16);
                        let shifted32 = builder.ins().ishl_imm(repeated32, 32);
                        let repeated64 = builder.ins().bor(repeated32, shifted32);
                        let vector = builder.ins().splat(types::I64X2, repeated64);
                        reg_cache.write_v(*vd, vector);
                    }
                    DecodedInstruction::DupV4SFromW { vd, rn } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source32 = builder.ins().ireduce(types::I32, source64);
                        let word64 = builder.ins().uextend(types::I64, source32);
                        let shifted32 = builder.ins().ishl_imm(word64, 32);
                        let repeated64 = builder.ins().bor(word64, shifted32);
                        let vector = builder.ins().splat(types::I64X2, repeated64);
                        reg_cache.write_v(*vd, vector);
                    }
                    DecodedInstruction::MoviV2DZero { vd }
                    | DecodedInstruction::MoviV4SZero { vd } => {
                        let zero = builder.ins().iconst(types::I64, 0);
                        let vector = builder.ins().splat(types::I64X2, zero);
                        reg_cache.write_v(*vd, vector);
                    }
                    DecodedInstruction::MoviVector { vd, imm_lo, imm_hi }
                    | DecodedInstruction::MvniVector { vd, imm_lo, imm_hi } => {
                        let lo = builder.ins().iconst(types::I64, *imm_lo as i64);
                        let hi = builder.ins().iconst(types::I64, *imm_hi as i64);
                        let vec = builder.ins().scalar_to_vector(types::I64X2, lo);
                        let vec = builder.ins().insertlane(vec, hi, 1);
                        reg_cache.write_v(*vd, vec);
                    }
                    DecodedInstruction::OrrImmVector { vd, imm_lo, imm_hi } => {
                        let old = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let lo = builder.ins().iconst(types::I64, *imm_lo as i64);
                        let hi = builder.ins().iconst(types::I64, *imm_hi as i64);
                        let imm_vec = builder.ins().scalar_to_vector(types::I64X2, lo);
                        let imm_vec = builder.ins().insertlane(imm_vec, hi, 1);
                        let result = builder.ins().bor(old, imm_vec);
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::BicImmVector { vd, imm_lo, imm_hi } => {
                        let old = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let lo = builder.ins().iconst(types::I64, *imm_lo as i64);
                        let hi = builder.ins().iconst(types::I64, *imm_hi as i64);
                        let imm_vec = builder.ins().scalar_to_vector(types::I64X2, lo);
                        let imm_vec = builder.ins().insertlane(imm_vec, hi, 1);
                        let not_imm = builder.ins().bnot(imm_vec);
                        let result = builder.ins().band(old, not_imm);
                        reg_cache.write_v(*vd, result);
                    }

                    // ── Vector Integer / FP Operations ──
                    DecodedInstruction::AddVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let vm = bitcast_little(&mut builder, vm, ty);
                        let mut result = builder.ins().iadd(vn, vm);

                        if !*q {
                            // result is 64-bit
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::AddpVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let vm = bitcast_little(&mut builder, vm, ty);

                        let result = builder.ins().iadd_pairwise(vn, vm);
                        let mut result = bitcast_little(&mut builder, result, types::I64X2);

                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::UminpVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        if *size == 0b11 {
                            unreachable!("uminp with size=0b11 is UNDEFINED");
                        }
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm_val = reg_cache.read_v(&mut builder, state_ptr, *vm);
                        let vn_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                        let vm_bytes = bitcast_little(&mut builder, vm_val, types::I8X16);

                        let elem_bytes = 1u8 << *size;
                        let (even_mask, odd_mask) = build_pairwise_shuffle_masks(elem_bytes, *q);
                        let even_imm = builder
                            .func
                            .dfg
                            .immediates
                            .push(ConstantData::from(&even_mask[..]));
                        let odd_imm = builder
                            .func
                            .dfg
                            .immediates
                            .push(ConstantData::from(&odd_mask[..]));

                        let vec_a = builder.ins().shuffle(vn_bytes, vm_bytes, even_imm);
                        let vec_b = builder.ins().shuffle(vn_bytes, vm_bytes, odd_imm);

                        let ty = vector_op_type(true, *size);
                        let vec_a = bitcast_little(&mut builder, vec_a, ty);
                        let vec_b = bitcast_little(&mut builder, vec_b, ty);

                        let mut result = builder.ins().umin(vec_a, vec_b);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::UmaxpVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        if *size == 0b11 {
                            unreachable!("umaxp with size=0b11 is UNDEFINED");
                        }
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm_val = reg_cache.read_v(&mut builder, state_ptr, *vm);
                        let vn_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                        let vm_bytes = bitcast_little(&mut builder, vm_val, types::I8X16);

                        let elem_bytes = 1u8 << *size;
                        let (even_mask, odd_mask) = build_pairwise_shuffle_masks(elem_bytes, *q);
                        let even_imm = builder
                            .func
                            .dfg
                            .immediates
                            .push(ConstantData::from(&even_mask[..]));
                        let odd_imm = builder
                            .func
                            .dfg
                            .immediates
                            .push(ConstantData::from(&odd_mask[..]));

                        let vec_a = builder.ins().shuffle(vn_bytes, vm_bytes, even_imm);
                        let vec_b = builder.ins().shuffle(vn_bytes, vm_bytes, odd_imm);

                        let ty = vector_op_type(true, *size);
                        let vec_a = bitcast_little(&mut builder, vec_a, ty);
                        let vec_b = bitcast_little(&mut builder, vec_b, ty);

                        let mut result = builder.ins().umax(vec_a, vec_b);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::ShrnVector {
                        vd,
                        vn,
                        q,
                        size,
                        shift,
                    } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);

                        let source_ty = match *size {
                            1 => types::I16X8,
                            2 => types::I32X4,
                            3 => types::I64X2,
                            _ => unreachable!("shrn with size={} is invalid", *size),
                        };
                        let vn_typed = bitcast_little(&mut builder, vn_val, source_ty);
                        let shifted = builder.ins().ushr_imm(vn_typed, i64::from(*shift));

                        let shifted_bytes = bitcast_little(&mut builder, shifted, types::I8X16);
                        let narrow_mask = build_narrow_shuffle_mask(*size);
                        let mask_imm = builder
                            .func
                            .dfg
                            .immediates
                            .push(ConstantData::from(&narrow_mask[..]));
                        let narrowed =
                            builder
                                .ins()
                                .shuffle(shifted_bytes, shifted_bytes, mask_imm);

                        let narrowed = bitcast_little(&mut builder, narrowed, types::I64X2);
                        let narrowed_lo = builder.ins().extractlane(narrowed, 0);

                        if *q {
                            // SHRN2: preserve lower half, put narrowed in upper half
                            let existing_vd = reg_cache.read_v(&mut builder, state_ptr, *vd);
                            let result = builder.ins().insertlane(existing_vd, narrowed_lo, 1);
                            reg_cache.write_v(*vd, result);
                        } else {
                            // SHRN: narrowed in lower half, upper zeroed
                            let result = builder.ins().scalar_to_vector(types::I64X2, narrowed_lo);
                            reg_cache.write_v(*vd, result);
                        }
                    }
                    DecodedInstruction::AndVector { vd, vn, vm, q } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let vn = bitcast_little(&mut builder, vn, types::I8X16);
                        let vm = bitcast_little(&mut builder, vm, types::I8X16);

                        let result = builder.ins().band(vn, vm);
                        let mut result = bitcast_little(&mut builder, result, types::I64X2);

                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::OrrVector { vd, vn, vm, q } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let vn = bitcast_little(&mut builder, vn, types::I8X16);
                        let vm = bitcast_little(&mut builder, vm, types::I8X16);

                        let result = builder.ins().bor(vn, vm);
                        let mut result = bitcast_little(&mut builder, result, types::I64X2);

                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::CmeqVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let vm = bitcast_little(&mut builder, vm, ty);

                        let mut result = builder.ins().icmp(IntCC::Equal, vn, vm);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::CmhsVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let vm = bitcast_little(&mut builder, vm, ty);

                        let mut result =
                            builder
                                .ins()
                                .icmp(IntCC::UnsignedGreaterThanOrEqual, vn, vm);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::CmhiVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let vm = bitcast_little(&mut builder, vm, ty);

                        let mut result = builder.ins().icmp(IntCC::UnsignedGreaterThan, vn, vm);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::UshlVector {
                        vd,
                        vn,
                        vm,
                        q,
                        size,
                    } => {
                        // Per-lane variable shift: positive = left, negative = right (unsigned).
                        // Implemented via lane extraction since Cranelift lacks per-lane variable shifts.
                        let ty = vector_op_type(true, *size);
                        let lane_ty = match *size {
                            0 => types::I8,
                            1 => types::I16,
                            2 => types::I32,
                            3 => types::I64,
                            _ => unreachable!(),
                        };
                        let elem_bits = (8u32 << *size) as i64;
                        let lane_count = if *q {
                            128 / elem_bits as u32
                        } else {
                            64 / elem_bits as u32
                        };

                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm_val = reg_cache.read_v(&mut builder, state_ptr, *vm);
                        let vn_typed = bitcast_little(&mut builder, vn_val, ty);
                        let vm_typed = bitcast_little(&mut builder, vm_val, ty);

                        // Start with zero vector
                        let zero_scalar = builder.ins().iconst(lane_ty, 0);
                        let mut result = builder.ins().splat(ty, zero_scalar);

                        for lane in 0..lane_count as u8 {
                            let val = builder.ins().extractlane(vn_typed, lane);
                            let amt_full = builder.ins().extractlane(vm_typed, lane);
                            // Get the low 8 bits as signed shift amount
                            let amt_i8 = if lane_ty == types::I8 {
                                amt_full
                            } else {
                                builder.ins().ireduce(types::I8, amt_full)
                            };
                            let amt_signed = builder.ins().sextend(types::I32, amt_i8);
                            let zero_i32 = builder.ins().iconst(types::I32, 0);
                            let is_neg =
                                builder
                                    .ins()
                                    .icmp(IntCC::SignedLessThan, amt_signed, zero_i32);

                            // Left shift amount (when positive)
                            let left_amt = if lane_ty == types::I32 {
                                amt_signed
                            } else {
                                // For I8/I16/I64, need to convert
                                if lane_ty.bits() < 32 {
                                    builder.ins().ireduce(lane_ty, amt_signed)
                                } else {
                                    builder.ins().sextend(lane_ty, amt_signed)
                                }
                            };
                            let left_result = builder.ins().ishl(val, left_amt);

                            // Right shift amount (when negative, negate)
                            let neg_amt = builder.ins().ineg(amt_signed);
                            let right_amt = if lane_ty == types::I32 {
                                neg_amt
                            } else if lane_ty.bits() < 32 {
                                builder.ins().ireduce(lane_ty, neg_amt)
                            } else {
                                builder.ins().sextend(lane_ty, neg_amt)
                            };
                            let right_result = builder.ins().ushr(val, right_amt);

                            let shifted = builder.ins().select(is_neg, right_result, left_result);

                            // Clamp: if |amt| >= elem_bits, result is 0
                            let abs_amt = builder.ins().select(is_neg, neg_amt, amt_signed);
                            let bits_const = builder.ins().iconst(types::I32, elem_bits);
                            let overflow = builder.ins().icmp(
                                IntCC::UnsignedGreaterThanOrEqual,
                                abs_amt,
                                bits_const,
                            );
                            let zero_lane = builder.ins().iconst(lane_ty, 0);
                            let clamped = builder.ins().select(overflow, zero_lane, shifted);

                            result = builder.ins().insertlane(result, clamped, lane);
                        }

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::BifVector { vd, vn, vm, q } => {
                        let vd_old = reg_cache.read_v(&mut builder, state_ptr, *vd);
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let vm = reg_cache.read_v(&mut builder, state_ptr, *vm);

                        let vd_old = bitcast_little(&mut builder, vd_old, types::I8X16);
                        let vn = bitcast_little(&mut builder, vn, types::I8X16);
                        let vm = bitcast_little(&mut builder, vm, types::I8X16);

                        let result = builder.ins().bitselect(vm, vd_old, vn);
                        let mut result = bitcast_little(&mut builder, result, types::I64X2);

                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::CmeqVectorZero { vd, vn, q, size } => {
                        let vn = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let zero = builder.ins().scalar_to_vector(types::I64X2, zero);

                        let ty = vector_op_type(true, *size);

                        let vn = bitcast_little(&mut builder, vn, ty);
                        let zero = bitcast_little(&mut builder, zero, ty);

                        let mut result = builder.ins().icmp(IntCC::Equal, vn, zero);

                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }

                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::NegVector { vd, vn, q, size } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let ty = vector_op_type(true, *size);
                        let vn_typed = bitcast_little(&mut builder, vn_val, ty);
                        let mut result = builder.ins().ineg(vn_typed);
                        if !*q {
                            result = bitcast_little(&mut builder, result, types::I64X2);
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::DupGeneralVector { vd, rn, q, size } => {
                        let gpr_val = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = match size {
                            0 => {
                                let byte = builder.ins().ireduce(types::I8, gpr_val);
                                let value = builder.ins().splat(types::I8X16, byte);
                                bitcast_little(&mut builder, value, types::I64X2)
                            }
                            1 => {
                                let half = builder.ins().ireduce(types::I16, gpr_val);
                                let bcast = builder.ins().splat(types::I16X8, half);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            2 => {
                                let word = builder.ins().ireduce(types::I32, gpr_val);
                                let bcast = builder.ins().splat(types::I32X4, word);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            3 => {
                                let bcast = builder.ins().splat(types::I64X2, gpr_val);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            _ => unreachable!("Invalid size for DupGeneralVector"),
                        };
                        let result = if !*q {
                            // Clear upper 64 bits for 64-bit arrangement
                            let lo = builder.ins().extractlane(result, 0);
                            builder.ins().scalar_to_vector(types::I64X2, lo)
                        } else {
                            result
                        };
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::DupElementVector {
                        vd,
                        vn,
                        q,
                        size,
                        index,
                    } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let result = match *size {
                            0 => {
                                let vn_val = bitcast_little(&mut builder, vn_val, types::I8X16);
                                let elem = builder.ins().extractlane(vn_val, *index); // i8
                                let bcast = builder.ins().splat(types::I8X16, elem);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            1 => {
                                let vn_val = bitcast_little(&mut builder, vn_val, types::I16X8);
                                let elem = builder.ins().extractlane(vn_val, *index); // i16
                                let bcast = builder.ins().splat(types::I16X8, elem);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            2 => {
                                let vn_val = bitcast_little(&mut builder, vn_val, types::I32X4);
                                let elem = builder.ins().extractlane(vn_val, *index); // i32
                                let bcast = builder.ins().splat(types::I32X4, elem);
                                bitcast_little(&mut builder, bcast, types::I64X2)
                            }
                            3 => {
                                let elem = builder.ins().extractlane(vn_val, *index); // i64
                                builder.ins().splat(types::I64X2, elem)
                            }
                            _ => unreachable!("Invalid size for DupElementVector"),
                        };

                        let result = if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            builder.ins().scalar_to_vector(types::I64X2, lo)
                        } else {
                            result
                        };
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::Rev64Vector { vd, vn, q, size } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        // Build a shuffle mask to reverse elements within each 8-byte group.
                        // size=0 (bytes): [7,6,5,4,3,2,1,0, 15,14,13,12,11,10,9,8]
                        // size=1 (halfwords): [6,7,4,5,2,3,0,1, 14,15,12,13,10,11,8,9]
                        // size=2 (words): [4,5,6,7,0,1,2,3, 12,13,14,15,8,9,10,11]
                        let elem_size = 1usize << *size;
                        let mut mask_bytes = [0u8; 16];
                        for group_start in (0..16usize).step_by(8) {
                            let elems_per_group = 8 / elem_size;
                            for i in 0..elems_per_group {
                                let dst_elem = group_start + i * elem_size;
                                let src_elem = group_start + (elems_per_group - 1 - i) * elem_size;
                                for b in 0..elem_size {
                                    mask_bytes[dst_elem + b] = (src_elem + b) as u8;
                                }
                            }
                        }
                        let mask_lo = u64::from_le_bytes(mask_bytes[0..8].try_into().unwrap());
                        let mask_hi = u64::from_le_bytes(mask_bytes[8..16].try_into().unwrap());
                        let as_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                        let mask_lo_v = builder.ins().iconst(types::I64, mask_lo as i64);
                        let mask_hi_v = builder.ins().iconst(types::I64, mask_hi as i64);
                        let mask_vec = builder.ins().scalar_to_vector(types::I64X2, mask_lo_v);
                        let mask_vec = builder.ins().insertlane(mask_vec, mask_hi_v, 1);
                        let mask_i8 = bitcast_little(&mut builder, mask_vec, types::I8X16);
                        let shuffled = builder.ins().swizzle(as_bytes, mask_i8);
                        let mut result = bitcast_little(&mut builder, shuffled, types::I64X2);
                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::CntVector { vd, vn, q } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let as_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                        let mut result = builder.ins().popcnt(as_bytes);
                        result = bitcast_little(&mut builder, result, types::I64X2);
                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::NotVector { vd, vn, q } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let mut result = builder.ins().bnot(vn_val);
                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::RbitVector { vd, vn, q } => {
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);
                        let as_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                        let mut result = builder.ins().bitrev(as_bytes);
                        result = bitcast_little(&mut builder, result, types::I64X2);
                        if !*q {
                            let lo = builder.ins().extractlane(result, 0);
                            result = builder.ins().scalar_to_vector(types::I64X2, lo);
                        }
                        reg_cache.write_v(*vd, result);
                    }
                    DecodedInstruction::UaddlvVector { vd, vn, q, size }
                    | DecodedInstruction::SaddlvVector { vd, vn, q, size } => {
                        let is_signed =
                            matches!(instruction, DecodedInstruction::SaddlvVector { .. });
                        let vn_val = reg_cache.read_v(&mut builder, state_ptr, *vn);

                        let sum = match *size {
                            0 => {
                                // 8-bit elements: sum bytes
                                let as_bytes = bitcast_little(&mut builder, vn_val, types::I8X16);
                                let lane_count = if *q { 16 } else { 8 };
                                let mut acc = if is_signed {
                                    let b = builder.ins().extractlane(as_bytes, 0);
                                    builder.ins().sextend(types::I64, b)
                                } else {
                                    let b = builder.ins().extractlane(as_bytes, 0);
                                    builder.ins().uextend(types::I64, b)
                                };
                                for i in 1..lane_count {
                                    let b = builder.ins().extractlane(as_bytes, i);
                                    let ext = if is_signed {
                                        builder.ins().sextend(types::I64, b)
                                    } else {
                                        builder.ins().uextend(types::I64, b)
                                    };
                                    acc = builder.ins().iadd(acc, ext);
                                }
                                acc
                            }
                            1 => {
                                // 16-bit elements
                                let as_i16 = bitcast_little(&mut builder, vn_val, types::I16X8);
                                let lane_count: u8 = if *q { 8 } else { 4 };
                                let mut acc = if is_signed {
                                    let b = builder.ins().extractlane(as_i16, 0);
                                    builder.ins().sextend(types::I64, b)
                                } else {
                                    let b = builder.ins().extractlane(as_i16, 0);
                                    builder.ins().uextend(types::I64, b)
                                };
                                for i in 1..lane_count {
                                    let b = builder.ins().extractlane(as_i16, i);
                                    let ext = if is_signed {
                                        builder.ins().sextend(types::I64, b)
                                    } else {
                                        builder.ins().uextend(types::I64, b)
                                    };
                                    acc = builder.ins().iadd(acc, ext);
                                }
                                acc
                            }
                            2 => {
                                // 32-bit elements (Q=1 only: 4S)
                                let as_i32 = bitcast_little(&mut builder, vn_val, types::I32X4);
                                let lane_count: u8 = if *q { 4 } else { 2 };
                                let mut acc = if is_signed {
                                    let b = builder.ins().extractlane(as_i32, 0);
                                    builder.ins().sextend(types::I64, b)
                                } else {
                                    let b = builder.ins().extractlane(as_i32, 0);
                                    builder.ins().uextend(types::I64, b)
                                };
                                for i in 1..lane_count {
                                    let b = builder.ins().extractlane(as_i32, i);
                                    let ext = if is_signed {
                                        builder.ins().sextend(types::I64, b)
                                    } else {
                                        builder.ins().uextend(types::I64, b)
                                    };
                                    acc = builder.ins().iadd(acc, ext);
                                }
                                acc
                            }
                            _ => builder.ins().iconst(types::I64, 0),
                        };

                        // Store result as a scalar in the lower bits, clear upper.
                        let result = builder.ins().scalar_to_vector(types::I64X2, sum);
                        reg_cache.write_v(*vd, result);
                    }

                    // ── FP/SIMD 128-bit Load/Store Pre/Post & Ld1 ──
                    DecodedInstruction::LdrPostFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            base,
                        );
                        let addr_8 = builder.ins().iadd_imm(base, 8);
                        let hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            addr_8,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cleared = builder.ins().splat(types::I64X2, zero);
                        let with_lo = builder.ins().insertlane(cleared, lo, 0);
                        let vec = builder.ins().insertlane(with_lo, hi, 1);
                        reg_cache.write_v(*rt, vec);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let addr_8 = builder.ins().iadd_imm(address, 8);
                        let hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            addr_8,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cleared = builder.ins().splat(types::I64X2, zero);
                        let with_lo = builder.ins().insertlane(cleared, lo, 0);
                        let vec = builder.ins().insertlane(with_lo, hi, 1);
                        reg_cache.write_v(*rt, vec);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let vec = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let lo = builder.ins().extractlane(vec, 0);
                        let hi = builder.ins().extractlane(vec, 1);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            base,
                            lo,
                        );
                        let addr_8 = builder.ins().iadd_imm(base, 8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            addr_8,
                            hi,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let vec = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let lo = builder.ins().extractlane(vec, 0);
                        let hi = builder.ins().extractlane(vec, 1);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            lo,
                        );
                        let addr_8 = builder.ins().iadd_imm(address, 8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            addr_8,
                            hi,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }

                    // ── FP/SIMD 64-bit Load/Store Pre/Post ──
                    DecodedInstruction::LdrPostFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            base,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            base,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }

                    // ── FP/SIMD 32-bit Load/Store Pre/Post ──
                    DecodedInstruction::LdrPostFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            base,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            base,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }

                    // ── AdvSIMD Load Multiple Structures ──
                    DecodedInstruction::SimdLdMulti {
                        q,
                        nregs,
                        interleave,
                        size,
                        rn,
                        vt,
                        post_index,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);

                        let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                        let total_bytes = bytes_per_reg * u64::from(*nregs);
                        let elem_size: u64 = 1u64 << *size;
                        let elems_per_reg = bytes_per_reg / elem_size;
                        let n = *nregs as u64;
                        let num_u64s = total_bytes / 8;

                        // Load all data as u64 chunks
                        let mut chunks = Vec::new();
                        for i in 0..num_u64s {
                            let addr = if i == 0 {
                                base
                            } else {
                                builder.ins().iadd_imm(base, (i * 8) as i64)
                            };
                            let v = emit_call_mem_read(
                                &mut builder,
                                memory_helpers.read_u64,
                                state_ptr,
                                addr,
                            );
                            chunks.push(v);
                        }
                        emit_fault_check(&mut builder, state_ptr, instr_pc);

                        if !*interleave {
                            // Non-interleaved: each register gets consecutive u64s
                            let u64s_per_reg = bytes_per_reg / 8;
                            for reg_idx in 0..n {
                                let base_chunk = (reg_idx * u64s_per_reg) as usize;
                                let lo = chunks[base_chunk];
                                let vec = if u64s_per_reg == 2 {
                                    let hi = chunks[base_chunk + 1];
                                    let v = builder.ins().scalar_to_vector(types::I64X2, lo);
                                    builder.ins().insertlane(v, hi, 1)
                                } else {
                                    builder.ins().scalar_to_vector(types::I64X2, lo)
                                };
                                let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                                reg_cache.write_v(vreg, vec);
                            }
                        } else {
                            // Interleaved: element-by-element deinterleaving
                            let elem_bits = elem_size * 8;
                            let elem_mask_val: i64 = if elem_bits == 64 {
                                -1i64
                            } else {
                                ((1u64 << elem_bits) - 1) as i64
                            };

                            for reg_idx in 0..n {
                                let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                                let mut lo_val = builder.ins().iconst(types::I64, 0);
                                let mut hi_val = builder.ins().iconst(types::I64, 0);

                                for lane in 0..elems_per_reg {
                                    let mem_byte_offset = (lane * n + reg_idx) * elem_size;
                                    let chunk_idx = (mem_byte_offset / 8) as usize;
                                    let bit_shift = ((mem_byte_offset % 8) * 8) as i64;

                                    let chunk = chunks[chunk_idx];
                                    let elem = if bit_shift == 0 && elem_bits == 64 {
                                        chunk
                                    } else if bit_shift == 0 {
                                        builder.ins().band_imm(chunk, elem_mask_val)
                                    } else if elem_bits == 64 {
                                        builder.ins().ushr_imm(chunk, bit_shift)
                                    } else {
                                        let shifted = builder.ins().ushr_imm(chunk, bit_shift);
                                        builder.ins().band_imm(shifted, elem_mask_val)
                                    };

                                    let bit_in_reg = (lane * elem_bits) as i64;
                                    if bit_in_reg < 64 {
                                        if bit_in_reg == 0 {
                                            lo_val = elem;
                                        } else {
                                            let placed = builder.ins().ishl_imm(elem, bit_in_reg);
                                            lo_val = builder.ins().bor(lo_val, placed);
                                        }
                                    } else {
                                        let off = bit_in_reg - 64;
                                        if off == 0 {
                                            hi_val = elem;
                                        } else {
                                            let placed = builder.ins().ishl_imm(elem, off);
                                            hi_val = builder.ins().bor(hi_val, placed);
                                        }
                                    }
                                }

                                let vec = builder.ins().scalar_to_vector(types::I64X2, lo_val);
                                let vec = builder.ins().insertlane(vec, hi_val, 1);
                                reg_cache.write_v(vreg, vec);
                            }
                        }

                        // Post-index writeback
                        if let Some(rm) = post_index {
                            let offset = if *rm == 31 {
                                builder.ins().iconst(types::I64, total_bytes as i64)
                            } else {
                                reg_cache.read(&mut builder, state_ptr, *rm)
                            };
                            let new_base = builder.ins().iadd(base, offset);
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── AdvSIMD Store Multiple Structures ──
                    DecodedInstruction::SimdStMulti {
                        q,
                        nregs,
                        interleave,
                        size,
                        rn,
                        vt,
                        post_index,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);

                        let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                        let total_bytes = bytes_per_reg * u64::from(*nregs);
                        let elem_size: u64 = 1u64 << *size;
                        let elems_per_reg = bytes_per_reg / elem_size;
                        let n = *nregs as u64;
                        let num_u64s = total_bytes / 8;

                        if !*interleave {
                            // Non-interleaved: each register provides consecutive u64s
                            let u64s_per_reg = bytes_per_reg / 8;
                            for reg_idx in 0..n {
                                let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                                let vec = emit_read_vreg128(&mut builder, state_ptr, vreg);
                                let lo = builder.ins().extractlane(vec, 0u8);
                                let addr_lo = if reg_idx == 0 {
                                    base
                                } else {
                                    builder
                                        .ins()
                                        .iadd_imm(base, (reg_idx * u64s_per_reg * 8) as i64)
                                };
                                emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u64,
                                    state_ptr,
                                    addr_lo,
                                    lo,
                                );
                                if u64s_per_reg == 2 {
                                    let hi = builder.ins().extractlane(vec, 1u8);
                                    let addr_hi = builder
                                        .ins()
                                        .iadd_imm(base, (reg_idx * u64s_per_reg * 8 + 8) as i64);
                                    emit_call_mem_write(
                                        &mut builder,
                                        memory_helpers.write_u64,
                                        state_ptr,
                                        addr_hi,
                                        hi,
                                    );
                                }
                            }
                        } else {
                            // Interleaved: build memory u64 chunks from register elements
                            let elem_bits = elem_size * 8;
                            let elem_mask_val: i64 = if elem_bits == 64 {
                                -1i64
                            } else {
                                ((1u64 << elem_bits) - 1) as i64
                            };

                            let mut chunk_values: Vec<Value> = (0..num_u64s)
                                .map(|_| builder.ins().iconst(types::I64, 0))
                                .collect();

                            for reg_idx in 0..n {
                                let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                                let vec = emit_read_vreg128(&mut builder, state_ptr, vreg);
                                let reg_lo = builder.ins().extractlane(vec, 0u8);
                                let reg_hi = builder.ins().extractlane(vec, 1u8);

                                for lane in 0..elems_per_reg {
                                    let bit_in_reg = (lane * elem_bits) as i64;
                                    let elem = if bit_in_reg < 64 {
                                        if bit_in_reg == 0 && elem_bits == 64 {
                                            reg_lo
                                        } else if bit_in_reg == 0 {
                                            builder.ins().band_imm(reg_lo, elem_mask_val)
                                        } else {
                                            let shifted =
                                                builder.ins().ushr_imm(reg_lo, bit_in_reg);
                                            if elem_bits == 64 {
                                                shifted
                                            } else {
                                                builder.ins().band_imm(shifted, elem_mask_val)
                                            }
                                        }
                                    } else {
                                        let off = bit_in_reg - 64;
                                        if off == 0 && elem_bits == 64 {
                                            reg_hi
                                        } else if off == 0 {
                                            builder.ins().band_imm(reg_hi, elem_mask_val)
                                        } else {
                                            let shifted = builder.ins().ushr_imm(reg_hi, off);
                                            if elem_bits == 64 {
                                                shifted
                                            } else {
                                                builder.ins().band_imm(shifted, elem_mask_val)
                                            }
                                        }
                                    };

                                    let mem_byte_offset = (lane * n + reg_idx) * elem_size;
                                    let chunk_idx = (mem_byte_offset / 8) as usize;
                                    let bit_in_chunk = ((mem_byte_offset % 8) * 8) as i64;

                                    if bit_in_chunk == 0 && elem_bits == 64 {
                                        chunk_values[chunk_idx] = elem;
                                    } else if bit_in_chunk == 0 {
                                        chunk_values[chunk_idx] =
                                            builder.ins().bor(chunk_values[chunk_idx], elem);
                                    } else {
                                        let placed = builder.ins().ishl_imm(elem, bit_in_chunk);
                                        chunk_values[chunk_idx] =
                                            builder.ins().bor(chunk_values[chunk_idx], placed);
                                    }
                                }
                            }

                            // Write all chunks to memory
                            for i in 0..num_u64s as usize {
                                let addr = if i == 0 {
                                    base
                                } else {
                                    builder.ins().iadd_imm(base, (i as i64) * 8)
                                };
                                emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u64,
                                    state_ptr,
                                    addr,
                                    chunk_values[i],
                                );
                            }
                        }

                        emit_fault_check(&mut builder, state_ptr, instr_pc);

                        // Post-index writeback
                        if let Some(rm) = post_index {
                            let offset = if *rm == 31 {
                                builder.ins().iconst(types::I64, total_bytes as i64)
                            } else {
                                reg_cache.read(&mut builder, state_ptr, *rm)
                            };
                            let new_base = builder.ins().iadd(base, offset);
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── AdvSIMD Load Single Structure and Replicate ──
                    DecodedInstruction::SimdLdReplicate {
                        q,
                        vt,
                        rn,
                        size,
                        nregs,
                        post_index,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let elem_size: u64 = 1u64 << *size;
                        let bytes_per_reg: u64 = if *q { 16 } else { 8 };
                        let elems_per_reg = bytes_per_reg / elem_size;
                        let elem_bits = elem_size * 8;

                        for reg_idx in 0..(*nregs as u64) {
                            let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                            let addr = if reg_idx == 0 {
                                base
                            } else {
                                builder.ins().iadd_imm(base, (reg_idx * elem_size) as i64)
                            };

                            // Read one element
                            let elem = match *size {
                                0 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u8,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xff)
                                }
                                1 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u16,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xffff)
                                }
                                2 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u32,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xffff_ffff_i64)
                                }
                                3 => emit_call_mem_read(
                                    &mut builder,
                                    memory_helpers.read_u64,
                                    state_ptr,
                                    addr,
                                ),
                                _ => unreachable!(),
                            };

                            // Replicate: build lo and hi u64 values
                            let mut lo_val = builder.ins().iconst(types::I64, 0);
                            let mut hi_val = builder.ins().iconst(types::I64, 0);
                            for lane in 0..elems_per_reg {
                                let bit_offset = lane * elem_bits;
                                if bit_offset < 64 {
                                    if bit_offset == 0 {
                                        lo_val = elem;
                                    } else {
                                        let placed =
                                            builder.ins().ishl_imm(elem, bit_offset as i64);
                                        lo_val = builder.ins().bor(lo_val, placed);
                                    }
                                } else {
                                    let off = (bit_offset - 64) as i64;
                                    if off == 0 {
                                        hi_val = elem;
                                    } else {
                                        let placed = builder.ins().ishl_imm(elem, off);
                                        hi_val = builder.ins().bor(hi_val, placed);
                                    }
                                }
                            }

                            let vec = builder.ins().scalar_to_vector(types::I64X2, lo_val);
                            let vec = builder.ins().insertlane(vec, hi_val, 1);
                            reg_cache.write_v(vreg, vec);
                        }
                        emit_fault_check(&mut builder, state_ptr, instr_pc);

                        if let Some(rm) = post_index {
                            let offset = if *rm == 31 {
                                builder
                                    .ins()
                                    .iconst(types::I64, (elem_size * u64::from(*nregs)) as i64)
                            } else {
                                reg_cache.read(&mut builder, state_ptr, *rm)
                            };
                            let new_base = builder.ins().iadd(base, offset);
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── AdvSIMD Load Single Element to Lane ──
                    DecodedInstruction::SimdLdSingle {
                        q: _,
                        vt,
                        rn,
                        size,
                        index,
                        nregs,
                        post_index,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let elem_size: u64 = 1u64 << *size;
                        let elem_bits = elem_size * 8;
                        let elem_mask: i64 = if elem_bits == 64 {
                            -1i64
                        } else {
                            ((1u64 << elem_bits) - 1) as i64
                        };

                        for reg_idx in 0..(*nregs as u64) {
                            let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                            let addr = if reg_idx == 0 {
                                base
                            } else {
                                builder.ins().iadd_imm(base, (reg_idx * elem_size) as i64)
                            };

                            let elem = match *size {
                                0 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u8,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xff)
                                }
                                1 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u16,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xffff)
                                }
                                2 => {
                                    let v = emit_call_mem_read(
                                        &mut builder,
                                        memory_helpers.read_u32,
                                        state_ptr,
                                        addr,
                                    );
                                    builder.ins().band_imm(v, 0xffff_ffff_i64)
                                }
                                3 => emit_call_mem_read(
                                    &mut builder,
                                    memory_helpers.read_u64,
                                    state_ptr,
                                    addr,
                                ),
                                _ => unreachable!(),
                            };

                            // Insert into existing vector at [index]
                            let vec = emit_read_vreg128(&mut builder, state_ptr, vreg);
                            let reg_lo = builder.ins().extractlane(vec, 0u8);
                            let reg_hi = builder.ins().extractlane(vec, 1u8);

                            let bit_offset = u64::from(*index) * elem_bits;
                            let (new_lo, new_hi) = if bit_offset < 64 {
                                let off = bit_offset as i64;
                                let clear_mask = !(elem_mask << off);
                                let cleared = builder.ins().band_imm(reg_lo, clear_mask);
                                let placed = if off == 0 {
                                    builder.ins().band_imm(elem, elem_mask)
                                } else {
                                    let masked = builder.ins().band_imm(elem, elem_mask);
                                    builder.ins().ishl_imm(masked, off)
                                };
                                let new_lo = builder.ins().bor(cleared, placed);
                                (new_lo, reg_hi)
                            } else {
                                let off = (bit_offset - 64) as i64;
                                let clear_mask = !(elem_mask << off);
                                let cleared = builder.ins().band_imm(reg_hi, clear_mask);
                                let placed = if off == 0 {
                                    builder.ins().band_imm(elem, elem_mask)
                                } else {
                                    let masked = builder.ins().band_imm(elem, elem_mask);
                                    builder.ins().ishl_imm(masked, off)
                                };
                                let new_hi = builder.ins().bor(cleared, placed);
                                (reg_lo, new_hi)
                            };

                            let new_vec = builder.ins().scalar_to_vector(types::I64X2, new_lo);
                            let new_vec = builder.ins().insertlane(new_vec, new_hi, 1);
                            reg_cache.write_v(vreg, new_vec);
                        }
                        emit_fault_check(&mut builder, state_ptr, instr_pc);

                        if let Some(rm) = post_index {
                            let offset = if *rm == 31 {
                                builder
                                    .ins()
                                    .iconst(types::I64, (elem_size * u64::from(*nregs)) as i64)
                            } else {
                                reg_cache.read(&mut builder, state_ptr, *rm)
                            };
                            let new_base = builder.ins().iadd(base, offset);
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── AdvSIMD Store Single Element from Lane ──
                    DecodedInstruction::SimdStSingle {
                        q: _,
                        vt,
                        rn,
                        size,
                        index,
                        nregs,
                        post_index,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let elem_size: u64 = 1u64 << *size;
                        let elem_bits = elem_size * 8;
                        let elem_mask: i64 = if elem_bits == 64 {
                            -1i64
                        } else {
                            ((1u64 << elem_bits) - 1) as i64
                        };

                        for reg_idx in 0..(*nregs as u64) {
                            let vreg = ((u64::from(*vt) + reg_idx) & 31) as u8;
                            let vec = emit_read_vreg128(&mut builder, state_ptr, vreg);
                            let reg_lo = builder.ins().extractlane(vec, 0u8);
                            let reg_hi = builder.ins().extractlane(vec, 1u8);

                            let bit_offset = u64::from(*index) * elem_bits;
                            let elem = if bit_offset < 64 {
                                let shifted = if bit_offset == 0 {
                                    reg_lo
                                } else {
                                    builder.ins().ushr_imm(reg_lo, bit_offset as i64)
                                };
                                builder.ins().band_imm(shifted, elem_mask)
                            } else {
                                let off = (bit_offset - 64) as i64;
                                let shifted = if off == 0 {
                                    reg_hi
                                } else {
                                    builder.ins().ushr_imm(reg_hi, off)
                                };
                                builder.ins().band_imm(shifted, elem_mask)
                            };

                            let addr = if reg_idx == 0 {
                                base
                            } else {
                                builder.ins().iadd_imm(base, (reg_idx * elem_size) as i64)
                            };

                            match *size {
                                0 => emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u8,
                                    state_ptr,
                                    addr,
                                    elem,
                                ),
                                1 => emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u16,
                                    state_ptr,
                                    addr,
                                    elem,
                                ),
                                2 => emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u32,
                                    state_ptr,
                                    addr,
                                    elem,
                                ),
                                3 => emit_call_mem_write(
                                    &mut builder,
                                    memory_helpers.write_u64,
                                    state_ptr,
                                    addr,
                                    elem,
                                ),
                                _ => unreachable!(),
                            }
                        }
                        emit_fault_check(&mut builder, state_ptr, instr_pc);

                        if let Some(rm) = post_index {
                            let offset = if *rm == 31 {
                                builder
                                    .ins()
                                    .iconst(types::I64, (elem_size * u64::from(*nregs)) as i64)
                            } else {
                                reg_cache.read(&mut builder, state_ptr, *rm)
                            };
                            let new_base = builder.ins().iadd(base, offset);
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── AdvSIMD Three-Different (Widening) ──
                    DecodedInstruction::SimdSaddl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Addl,
                        );
                    }
                    DecodedInstruction::SimdUaddl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Addl,
                        );
                    }
                    DecodedInstruction::SimdSaddw {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Addw,
                        );
                    }
                    DecodedInstruction::SimdUaddw {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Addw,
                        );
                    }
                    DecodedInstruction::SimdSsubl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Subl,
                        );
                    }
                    DecodedInstruction::SimdUsubl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Subl,
                        );
                    }
                    DecodedInstruction::SimdSsubw {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Subw,
                        );
                    }
                    DecodedInstruction::SimdUsubw {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Subw,
                        );
                    }
                    DecodedInstruction::SimdSabdl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Abdl,
                        );
                    }
                    DecodedInstruction::SimdUabdl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Abdl,
                        );
                    }
                    DecodedInstruction::SimdSmlal {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mlal,
                        );
                    }
                    DecodedInstruction::SimdUmlal {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mlal,
                        );
                    }
                    DecodedInstruction::SimdSmlsl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mlsl,
                        );
                    }
                    DecodedInstruction::SimdUmlsl {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mlsl,
                        );
                    }
                    DecodedInstruction::SimdSmull {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            true,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mull,
                        );
                    }
                    DecodedInstruction::SimdUmull {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_widen_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            false,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdWidenJitOp::Mull,
                        );
                    }

                    // ── Narrowing three-different: ADDHN/RADDHN/SUBHN/RSUBHN ──
                    DecodedInstruction::SimdAddhn {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_narrow_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdNarrowOp::Addhn,
                        );
                    }
                    DecodedInstruction::SimdRaddhn {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_narrow_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdNarrowOp::Raddhn,
                        );
                    }
                    DecodedInstruction::SimdSubhn {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_narrow_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdNarrowOp::Subhn,
                        );
                    }
                    DecodedInstruction::SimdRsubhn {
                        q,
                        size,
                        rd,
                        rn,
                        rm,
                    } => {
                        emit_simd_narrow_jit(
                            &mut builder,
                            state_ptr,
                            &mut reg_cache,
                            *q,
                            *size,
                            *rd,
                            *rn,
                            *rm,
                            SimdNarrowOp::Rsubhn,
                        );
                    }

                    // ── FP vector unary operations ──
                    DecodedInstruction::VecFabs { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fabs,
                        );
                    }
                    DecodedInstruction::VecFneg { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fneg,
                        );
                    }
                    DecodedInstruction::VecFsqrt { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fsqrt,
                        );
                    }
                    DecodedInstruction::VecFrecpe { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frecpe,
                        );
                    }
                    DecodedInstruction::VecFrsqrte { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frsqrte,
                        );
                    }
                    DecodedInstruction::VecFrintn { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frintn,
                        );
                    }
                    DecodedInstruction::VecFrintm { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frintm,
                        );
                    }
                    DecodedInstruction::VecFrintp { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frintp,
                        );
                    }
                    DecodedInstruction::VecFrintz { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frintz,
                        );
                    }
                    DecodedInstruction::VecFrinta { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frinta,
                        );
                    }
                    DecodedInstruction::VecFrintx { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frintx,
                        );
                    }
                    DecodedInstruction::VecFrinti { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frinti,
                        );
                    }
                    DecodedInstruction::VecFrint32x { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Frint32x,
                        );
                    }
                    DecodedInstruction::VecFcvtns { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtns,
                        );
                    }
                    DecodedInstruction::VecFcvtnu { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtnu,
                        );
                    }
                    DecodedInstruction::VecFcvtms { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtms,
                        );
                    }
                    DecodedInstruction::VecFcvtmu { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtmu,
                        );
                    }
                    DecodedInstruction::VecFcvtas { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtas,
                        );
                    }
                    DecodedInstruction::VecFcvtau { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtau,
                        );
                    }
                    DecodedInstruction::VecFcvtps { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtps,
                        );
                    }
                    DecodedInstruction::VecFcvtpu { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtpu,
                        );
                    }
                    DecodedInstruction::VecFcvtzs { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtzs,
                        );
                    }
                    DecodedInstruction::VecFcvtzu { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcvtzu,
                        );
                    }
                    DecodedInstruction::VecScvtf { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Scvtf,
                        );
                    }
                    DecodedInstruction::VecUcvtf { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Ucvtf,
                        );
                    }
                    DecodedInstruction::VecFcmeq0 { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcmeq0,
                        );
                    }
                    DecodedInstruction::VecFcmge0 { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcmge0,
                        );
                    }
                    DecodedInstruction::VecFcmgt0 { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcmgt0,
                        );
                    }
                    DecodedInstruction::VecFcmle0 { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcmle0,
                        );
                    }
                    DecodedInstruction::VecFcmlt0 { vd, vn, q, sz } => {
                        emit_fp_vector_unary(
                            &mut builder,
                            &mut reg_cache,
                            state_ptr,
                            *vd,
                            *vn,
                            *q,
                            *sz,
                            FpVecUnaryOp::Fcmlt0,
                        );
                    }

                    // ── FP Conditional Select ──
                    DecodedInstruction::FcselS {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let false_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rm);
                        let result = builder.ins().select(cond, true_bits, false_bits);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result);
                    }
                    DecodedInstruction::FcselD {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let false_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rm);
                        let result = builder.ins().select(cond, true_bits, false_bits);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result);
                    }

                    // ── FP Compare ──
                    DecodedInstruction::FcmpS { rn, rm } => {
                        let lhs_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let rhs_bits = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rm);
                        let lhs = bitcast_little(&mut builder, lhs_bits, types::F32);
                        let rhs = bitcast_little(&mut builder, rhs_bits, types::F32);

                        let less_than = builder.ins().fcmp(FloatCC::LessThan, lhs, rhs);
                        let equal = builder.ins().fcmp(FloatCC::Equal, lhs, rhs);
                        let unordered = builder.ins().fcmp(FloatCC::Unordered, lhs, rhs);

                        let n = emit_bool_to_i32(&mut builder, less_than);
                        let z = emit_bool_to_i32(&mut builder, equal);
                        let not_less_than = emit_bool_not(&mut builder, less_than);
                        let c = emit_bool_to_i32(&mut builder, not_less_than);
                        let v = emit_bool_to_i32(&mut builder, unordered);

                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }
                    DecodedInstruction::FcmpD { rn, rm } => {
                        let lhs_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let rhs_bits = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rm);
                        let lhs = bitcast_little(&mut builder, lhs_bits, types::F64);
                        let rhs = bitcast_little(&mut builder, rhs_bits, types::F64);

                        let less_than = builder.ins().fcmp(FloatCC::LessThan, lhs, rhs);
                        let equal = builder.ins().fcmp(FloatCC::Equal, lhs, rhs);
                        let unordered = builder.ins().fcmp(FloatCC::Unordered, lhs, rhs);

                        let n = emit_bool_to_i32(&mut builder, less_than);
                        let z = emit_bool_to_i32(&mut builder, equal);
                        let not_less_than = emit_bool_not(&mut builder, less_than);
                        let c = emit_bool_to_i32(&mut builder, not_less_than);
                        let v = emit_bool_to_i32(&mut builder, unordered);

                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }

                    // ── FP Rounding ──
                    DecodedInstruction::FrintmS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let floored = builder.ins().floor(source_f32);
                        let result_bits = bitcast_little(&mut builder, floored, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FrintmD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let floored = builder.ins().floor(source_f64);
                        let result_bits = bitcast_little(&mut builder, floored, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FnegS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let negated = builder.ins().fneg(source_f32);
                        let result_bits = bitcast_little(&mut builder, negated, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FnegD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let negated = builder.ins().fneg(source_f64);
                        let result_bits = bitcast_little(&mut builder, negated, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FabsS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let absval = builder.ins().fabs(source_f32);
                        let result_bits = bitcast_little(&mut builder, absval, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FabsD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let absval = builder.ins().fabs(source_f64);
                        let result_bits = bitcast_little(&mut builder, absval, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FsqrtS { rd, rn } => {
                        let source32 = reg_cache.read_v_scalar32(&mut builder, state_ptr, *rn);
                        let source_f32 = bitcast_little(&mut builder, source32, types::F32);
                        let result = builder.ins().sqrt(source_f32);
                        let result_bits = bitcast_little(&mut builder, result, types::I32);
                        reg_cache.write_v_scalar32(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::FsqrtD { rd, rn } => {
                        let source64 = reg_cache.read_v_scalar64(&mut builder, state_ptr, *rn);
                        let source_f64 = bitcast_little(&mut builder, source64, types::F64);
                        let result = builder.ins().sqrt(source_f64);
                        let result_bits = bitcast_little(&mut builder, result, types::I64);
                        reg_cache.write_v_scalar64(&mut builder, *rd, result_bits);
                    }
                    DecodedInstruction::Clrex { .. } => {
                        let _ = memory_helpers.state_clrex.call(&mut builder, &[state_ptr]);
                    }

                    // ── Integer Move-Wide ──
                    DecodedInstruction::MovZ { rd, imm16, shift } => {
                        let literal = (u64::from(*imm16)) << *shift;
                        let value = builder.ins().iconst(types::I64, literal as i64);
                        reg_cache.write(*rd, value);
                    }
                    DecodedInstruction::MovZ32 { rd, imm16, shift } => {
                        let literal32 = (u32::from(*imm16)) << *shift;
                        let value32 = builder
                            .ins()
                            .iconst(types::I32, i64::from(literal32 as i32));
                        let value = builder.ins().uextend(types::I64, value32);
                        reg_cache.write(*rd, value);
                    }
                    DecodedInstruction::MovN { rd, imm16, shift } => {
                        let literal = !((u64::from(*imm16)) << *shift);
                        let value = builder.ins().iconst(types::I64, literal as i64);
                        reg_cache.write(*rd, value);
                    }
                    DecodedInstruction::MovN32 { rd, imm16, shift } => {
                        let literal32 = !((u32::from(*imm16)) << *shift);
                        let value32 = builder
                            .ins()
                            .iconst(types::I32, i64::from(literal32 as i32));
                        let value = builder.ins().uextend(types::I64, value32);
                        reg_cache.write(*rd, value);
                    }
                    DecodedInstruction::MovK { rd, imm16, shift } => {
                        let base = reg_cache.read(&mut builder, state_ptr, *rd);
                        let clear_mask = !((0xffffu64) << *shift);
                        let clear_mask_val = builder.ins().iconst(types::I64, clear_mask as i64);
                        let cleared = builder.ins().band(base, clear_mask_val);
                        let literal = (u64::from(*imm16)) << *shift;
                        let literal_val = builder.ins().iconst(types::I64, literal as i64);
                        let updated = builder.ins().bor(cleared, literal_val);
                        reg_cache.write(*rd, updated);
                    }
                    DecodedInstruction::MovK32 { rd, imm16, shift } => {
                        let base64 = reg_cache.read(&mut builder, state_ptr, *rd);
                        let base32 = builder.ins().ireduce(types::I32, base64);
                        let clear_mask = !((0xffffu32) << *shift);
                        let clear_mask_val = builder
                            .ins()
                            .iconst(types::I32, i64::from(clear_mask as i32));
                        let cleared = builder.ins().band(base32, clear_mask_val);
                        let literal32 = (u32::from(*imm16)) << *shift;
                        let literal_val = builder
                            .ins()
                            .iconst(types::I32, i64::from(literal32 as i32));
                        let updated32 = builder.ins().bor(cleared, literal_val);
                        let updated = builder.ins().uextend(types::I64, updated32);
                        reg_cache.write(*rd, updated);
                    }

                    // ── Integer Logic Immediate ──
                    DecodedInstruction::AndImm { rd, rn, imm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, *imm as i64);
                        let result = builder.ins().band(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AndImm32 { rd, rn, imm } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm as i32));
                        let result32 = builder.ins().band(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrrImm { rd, rn, imm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, *imm as i64);
                        let result = builder.ins().bor(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrrImm32 { rd, rn, imm } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm as i32));
                        let result32 = builder.ins().bor(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EorImm { rd, rn, imm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, *imm as i64);
                        let result = builder.ins().bxor(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EorImm32 { rd, rn, imm } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm as i32));
                        let result32 = builder.ins().bxor(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AndsImm { rd, rn, imm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, *imm as i64);
                        let result = builder.ins().band(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic(&mut builder, state_ptr, result);
                    }
                    DecodedInstruction::AndsImm32 { rd, rn, imm } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm as i32));
                        let result32 = builder.ins().band(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic32(&mut builder, state_ptr, result32);
                    }

                    // ── Integer Shift Immediate ──
                    DecodedInstruction::LslImm { rd, rn, amount } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().ishl_imm(lhs, i64::from(*amount));
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::LslImm32 { rd, rn, amount } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let result32 = builder.ins().ishl_imm(lhs, i64::from(*amount));
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::LsrImm { rd, rn, amount } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().ushr_imm(lhs, i64::from(*amount));
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::LsrImm32 { rd, rn, amount } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let result32 = builder.ins().ushr_imm(lhs, i64::from(*amount));
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AsrImm { rd, rn, amount } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().sshr_imm(lhs, i64::from(*amount));
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AsrImm32 { rd, rn, amount } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let result32 = builder.ins().sshr_imm(lhs, i64::from(*amount));
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Integer Add/Sub Immediate ──
                    DecodedInstruction::AddImm {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let src = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let imm = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let dst = builder.ins().iadd_imm(src, imm as i64);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rd, dst);
                    }
                    DecodedInstruction::AddImm32 {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let src32 = read_w_or_wsp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let imm = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I32, i64::from(imm as i32));
                        let dst32 = builder.ins().iadd(src32, rhs);
                        write_w_or_wsp(&mut builder, state_ptr, &mut reg_cache, *rd, dst32);
                    }
                    DecodedInstruction::SubImm {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let src = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let imm = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let dst = builder.ins().iadd_imm(src, -(imm as i64));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rd, dst);
                    }
                    DecodedInstruction::SubImm32 {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let src32 = read_w_or_wsp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let imm = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I32, i64::from(imm as i32));
                        let dst32 = builder.ins().isub(src32, rhs);
                        write_w_or_wsp(&mut builder, state_ptr, &mut reg_cache, *rd, dst32);
                    }
                    DecodedInstruction::AddsImm {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_imm = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I64, rhs_imm as i64);
                        let result = builder.ins().iadd(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::AddsImm32 {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs_imm = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I32, i64::from(rhs_imm as i32));
                        let result32 = builder.ins().iadd(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add32(&mut builder, state_ptr, lhs, rhs, result32);
                    }
                    DecodedInstruction::SubsImm {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_imm = u64::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I64, rhs_imm as i64);
                        let result = builder.ins().isub(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::SubsImm32 {
                        rd,
                        rn,
                        imm12,
                        shift12,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs_imm = u32::from(*imm12) << if *shift12 { 12 } else { 0 };
                        let rhs = builder.ins().iconst(types::I32, i64::from(rhs_imm as i32));
                        let result32 = builder.ins().isub(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub32(&mut builder, state_ptr, lhs, rhs, result32);
                    }

                    // ── Integer Add/Sub Register ──
                    DecodedInstruction::AddReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().iadd(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AddReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().iadd(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::SubReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().isub(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::SubReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().isub(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AddsReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().iadd(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::AddsReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().iadd(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add32(&mut builder, state_ptr, lhs, rhs, result32);
                    }
                    DecodedInstruction::SubsReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().isub(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::SubsReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().isub(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub32(&mut builder, state_ptr, lhs, rhs, result32);
                    }

                    // ── Integer Add/Sub Extended Register ──
                    DecodedInstruction::AddExt {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        let lhs = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_extend(&mut builder, rhs_raw, *extend, *amount);
                        let result = builder.ins().iadd(lhs, rhs);
                        if *rd == 31 {
                            reg_cache.write_sp(result);
                        } else {
                            reg_cache.write(*rd, result);
                        }
                    }
                    DecodedInstruction::AddExt32 {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        let lhs64 = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_extend32(&mut builder, rhs_raw, *extend, *amount);
                        let result32 = builder.ins().iadd(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        if *rd == 31 {
                            reg_cache.write_sp(result);
                        } else {
                            reg_cache.write(*rd, result);
                        }
                    }
                    DecodedInstruction::SubExt {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        let lhs = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_extend(&mut builder, rhs_raw, *extend, *amount);
                        let result = builder.ins().isub(lhs, rhs);
                        if *rd == 31 {
                            reg_cache.write_sp(result);
                        } else {
                            reg_cache.write(*rd, result);
                        }
                    }
                    DecodedInstruction::SubExt32 {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        let lhs64 = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_extend32(&mut builder, rhs_raw, *extend, *amount);
                        let result32 = builder.ins().isub(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        if *rd == 31 {
                            reg_cache.write_sp(result);
                        } else {
                            reg_cache.write(*rd, result);
                        }
                    }
                    DecodedInstruction::AddsExt {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        // For ADDS ext, Rn=31 refers to SP but Rd=31 refers to XZR
                        let lhs = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_extend(&mut builder, rhs_raw, *extend, *amount);
                        let result = builder.ins().iadd(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::AddsExt32 {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        // For ADDS ext, Rn=31 refers to SP but Rd=31 refers to XZR
                        let lhs64 = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_extend32(&mut builder, rhs_raw, *extend, *amount);
                        let result32 = builder.ins().iadd(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_add32(&mut builder, state_ptr, lhs, rhs, result32);
                    }
                    DecodedInstruction::SubsExt {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        // For SUBS ext, Rn=31 refers to SP but Rd=31 refers to XZR
                        let lhs = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_extend(&mut builder, rhs_raw, *extend, *amount);
                        let result = builder.ins().isub(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub(&mut builder, state_ptr, lhs, rhs, result);
                    }
                    DecodedInstruction::SubsExt32 {
                        rd,
                        rn,
                        rm,
                        extend,
                        amount,
                    } => {
                        // For SUBS ext, Rn=31 refers to SP but Rd=31 refers to XZR
                        let lhs64 = if *rn == 31 {
                            reg_cache.read_sp(&mut builder, state_ptr)
                        } else {
                            reg_cache.read(&mut builder, state_ptr, *rn)
                        };
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_extend32(&mut builder, rhs_raw, *extend, *amount);
                        let result32 = builder.ins().isub(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_sub32(&mut builder, state_ptr, lhs, rhs, result32);
                    }

                    // ── Integer Logic Register ──
                    DecodedInstruction::AndReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().band(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AndReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().band(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::BicReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result = builder.ins().band(lhs, inv_rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::BicReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result32 = builder.ins().band(lhs, inv_rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrrReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().bor(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrrReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().bor(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrnReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result = builder.ins().bor(lhs, inv_rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::OrnReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result32 = builder.ins().bor(lhs, inv_rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EorReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().bxor(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EorReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().bxor(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EonReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result = builder.ins().bxor(lhs, inv_rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::EonReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result32 = builder.ins().bxor(lhs, inv_rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::AndsReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let result = builder.ins().band(lhs, rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic(&mut builder, state_ptr, result);
                    }
                    DecodedInstruction::AndsReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let result32 = builder.ins().band(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic32(&mut builder, state_ptr, result32);
                    }
                    DecodedInstruction::BicsReg {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let rhs = emit_apply_shift(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result = builder.ins().band(lhs, inv_rhs);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic(&mut builder, state_ptr, result);
                    }
                    DecodedInstruction::BicsReg32 {
                        rd,
                        rn,
                        rm,
                        shift,
                        amount,
                    } => {
                        let lhs64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs64);
                        let rhs = emit_apply_shift32(&mut builder, rhs_raw, *shift, *amount);
                        let inv_rhs = builder.ins().bnot(rhs);
                        let result32 = builder.ins().band(lhs, inv_rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                        emit_update_nzcv_logic32(&mut builder, state_ptr, result32);
                    }

                    // ── Conditional Select ──
                    DecodedInstruction::Csel {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_val = reg_cache.read(&mut builder, state_ptr, *rn);
                        let false_val = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = builder.ins().select(cond, true_val, false_val);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csel32 {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let false_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let true_val = builder.ins().ireduce(types::I32, true_raw);
                        let false_val = builder.ins().ireduce(types::I32, false_raw);
                        let result32 = builder.ins().select(cond, true_val, false_val);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csinc {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_val = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_val = reg_cache.read(&mut builder, state_ptr, *rm);
                        let false_val = builder.ins().iadd_imm(rm_val, 1);
                        let result = builder.ins().select(cond, true_val, false_val);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csinc32 {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let true_val = builder.ins().ireduce(types::I32, true_raw);
                        let rm_val = builder.ins().ireduce(types::I32, rm_raw);
                        let false_val = builder.ins().iadd_imm(rm_val, 1);
                        let result32 = builder.ins().select(cond, true_val, false_val);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csinv {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_val = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_val = reg_cache.read(&mut builder, state_ptr, *rm);
                        let all_ones = builder.ins().iconst(types::I64, -1);
                        let false_val = builder.ins().bxor(rm_val, all_ones);
                        let result = builder.ins().select(cond, true_val, false_val);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csinv32 {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let true_val = builder.ins().ireduce(types::I32, true_raw);
                        let rm_val = builder.ins().ireduce(types::I32, rm_raw);
                        let all_ones = builder.ins().iconst(types::I32, -1);
                        let false_val = builder.ins().bxor(rm_val, all_ones);
                        let result32 = builder.ins().select(cond, true_val, false_val);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csneg {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_val = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_val = reg_cache.read(&mut builder, state_ptr, *rm);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let false_val = builder.ins().isub(zero, rm_val);
                        let result = builder.ins().select(cond, true_val, false_val);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Csneg32 {
                        rd,
                        rn,
                        rm,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let true_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rm_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let true_val = builder.ins().ireduce(types::I32, true_raw);
                        let rm_val = builder.ins().ireduce(types::I32, rm_raw);
                        let zero = builder.ins().iconst(types::I32, 0);
                        let false_val = builder.ins().isub(zero, rm_val);
                        let result32 = builder.ins().select(cond, true_val, false_val);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Conditional Compare ──
                    DecodedInstruction::CcmpReg {
                        rn,
                        rm,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = builder.ins().isub(lhs, rhs);
                        let true_values = emit_compute_nzcv_sub64(&mut builder, lhs, rhs, result);
                        let false_values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        let (n, z, c, v) =
                            emit_select_nzcv_values(&mut builder, cond, true_values, false_values);
                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }
                    DecodedInstruction::CcmpReg32 {
                        rn,
                        rm,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let result = builder.ins().isub(lhs, rhs);
                        let true_values = emit_compute_nzcv_sub32(&mut builder, lhs, rhs, result);
                        let false_values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        let (n, z, c, v) =
                            emit_select_nzcv_values(&mut builder, cond, true_values, false_values);
                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }
                    DecodedInstruction::CcmpImm {
                        rn,
                        imm5,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, i64::from(*imm5));
                        let result = builder.ins().isub(lhs, rhs);
                        let true_values = emit_compute_nzcv_sub64(&mut builder, lhs, rhs, result);
                        let false_values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        let (n, z, c, v) =
                            emit_select_nzcv_values(&mut builder, cond, true_values, false_values);
                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }
                    DecodedInstruction::CcmpImm32 {
                        rn,
                        imm5,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm5 as i32));
                        let result = builder.ins().isub(lhs, rhs);
                        let true_values = emit_compute_nzcv_sub32(&mut builder, lhs, rhs, result);
                        let false_values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        let (n, z, c, v) =
                            emit_select_nzcv_values(&mut builder, cond, true_values, false_values);
                        set_pending_nzcv(PendingNzcvRecipe::Values { n, z, c, v });
                    }
                    DecodedInstruction::CcmnReg {
                        rn,
                        rm,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let true_block = builder.create_block();
                        let false_block = builder.create_block();
                        let done_block = builder.create_block();
                        emit_append_nzcv_block_params(&mut builder, done_block);
                        builder.ins().brif(cond, true_block, &[], false_block, &[]);
                        builder.seal_block(true_block);
                        builder.seal_block(false_block);

                        builder.switch_to_block(true_block);
                        let result = builder.ins().iadd(lhs, rhs);
                        let values = emit_compute_nzcv_add64(&mut builder, lhs, rhs, result);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.switch_to_block(false_block);
                        let values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.seal_block(done_block);
                        builder.switch_to_block(done_block);
                        emit_set_pending_nzcv_from_block_params(&mut builder, done_block);
                    }
                    DecodedInstruction::CcmnReg32 {
                        rn,
                        rm,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let true_block = builder.create_block();
                        let false_block = builder.create_block();
                        let done_block = builder.create_block();
                        emit_append_nzcv_block_params(&mut builder, done_block);
                        builder.ins().brif(cond, true_block, &[], false_block, &[]);
                        builder.seal_block(true_block);
                        builder.seal_block(false_block);

                        builder.switch_to_block(true_block);
                        let result = builder.ins().iadd(lhs, rhs);
                        let values = emit_compute_nzcv_add32(&mut builder, lhs, rhs, result);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.switch_to_block(false_block);
                        let values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.seal_block(done_block);
                        builder.switch_to_block(done_block);
                        emit_set_pending_nzcv_from_block_params(&mut builder, done_block);
                    }
                    DecodedInstruction::CcmnImm {
                        rn,
                        imm5,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = builder.ins().iconst(types::I64, i64::from(*imm5));
                        let true_block = builder.create_block();
                        let false_block = builder.create_block();
                        let done_block = builder.create_block();
                        emit_append_nzcv_block_params(&mut builder, done_block);
                        builder.ins().brif(cond, true_block, &[], false_block, &[]);
                        builder.seal_block(true_block);
                        builder.seal_block(false_block);

                        builder.switch_to_block(true_block);
                        let result = builder.ins().iadd(lhs, rhs);
                        let values = emit_compute_nzcv_add64(&mut builder, lhs, rhs, result);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.switch_to_block(false_block);
                        let values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.seal_block(done_block);
                        builder.switch_to_block(done_block);
                        emit_set_pending_nzcv_from_block_params(&mut builder, done_block);
                    }
                    DecodedInstruction::CcmnImm32 {
                        rn,
                        imm5,
                        nzcv,
                        condition,
                    } => {
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().iconst(types::I32, i64::from(*imm5 as i32));
                        let true_block = builder.create_block();
                        let false_block = builder.create_block();
                        let done_block = builder.create_block();
                        emit_append_nzcv_block_params(&mut builder, done_block);
                        builder.ins().brif(cond, true_block, &[], false_block, &[]);
                        builder.seal_block(true_block);
                        builder.seal_block(false_block);

                        builder.switch_to_block(true_block);
                        let result = builder.ins().iadd(lhs, rhs);
                        let values = emit_compute_nzcv_add32(&mut builder, lhs, rhs, result);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.switch_to_block(false_block);
                        let values = emit_nzcv_values_from_immediate(&mut builder, *nzcv);
                        emit_jump_with_nzcv_values(&mut builder, done_block, values);

                        builder.seal_block(done_block);
                        builder.switch_to_block(done_block);
                        emit_set_pending_nzcv_from_block_params(&mut builder, done_block);
                    }

                    // ── Variable Shifts ──
                    DecodedInstruction::Lslv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount = reg_cache.read(&mut builder, state_ptr, *rm);
                        let masked = builder.ins().band_imm(amount, 63);
                        let result = builder.ins().ishl(lhs, masked);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Lslv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let amount = builder.ins().ireduce(types::I32, amount_raw);
                        let masked = builder.ins().band_imm(amount, 31);
                        let result32 = builder.ins().ishl(lhs, masked);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Lsrv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount = reg_cache.read(&mut builder, state_ptr, *rm);
                        let masked = builder.ins().band_imm(amount, 63);
                        let result = builder.ins().ushr(lhs, masked);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Lsrv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let amount = builder.ins().ireduce(types::I32, amount_raw);
                        let masked = builder.ins().band_imm(amount, 31);
                        let result32 = builder.ins().ushr(lhs, masked);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Asrv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount = reg_cache.read(&mut builder, state_ptr, *rm);
                        let masked = builder.ins().band_imm(amount, 63);
                        let result = builder.ins().sshr(lhs, masked);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Asrv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let amount = builder.ins().ireduce(types::I32, amount_raw);
                        let masked = builder.ins().band_imm(amount, 31);
                        let result32 = builder.ins().sshr(lhs, masked);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Rorv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount = reg_cache.read(&mut builder, state_ptr, *rm);
                        let masked = builder.ins().band_imm(amount, 63);
                        let inv_raw = builder.ins().irsub_imm(masked, 64);
                        let inv = builder.ins().band_imm(inv_raw, 63);
                        let right = builder.ins().ushr(lhs, masked);
                        let left = builder.ins().ishl(lhs, inv);
                        let result = builder.ins().bor(left, right);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Rorv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let amount_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let amount = builder.ins().ireduce(types::I32, amount_raw);
                        let masked = builder.ins().band_imm(amount, 31);
                        let inv_raw = builder.ins().irsub_imm(masked, 32);
                        let inv = builder.ins().band_imm(inv_raw, 31);
                        let right = builder.ins().ushr(lhs, masked);
                        let left = builder.ins().ishl(lhs, inv);
                        let result32 = builder.ins().bor(left, right);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Extract ──
                    DecodedInstruction::Extr { rd, rn, rm, lsb } => {
                        let low = reg_cache.read(&mut builder, state_ptr, *rm);
                        let high = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = if *lsb == 0 {
                            low
                        } else {
                            let right = builder.ins().ushr_imm(low, i64::from(*lsb));
                            let left = builder.ins().ishl_imm(high, i64::from(64 - *lsb));
                            builder.ins().bor(left, right)
                        };
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Extr32 { rd, rn, rm, lsb } => {
                        let low64 = reg_cache.read(&mut builder, state_ptr, *rm);
                        let high64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let low = builder.ins().ireduce(types::I32, low64);
                        let high = builder.ins().ireduce(types::I32, high64);
                        let result32 = if *lsb == 0 {
                            low
                        } else {
                            let right = builder.ins().ushr_imm(low, i64::from(*lsb));
                            let left = builder.ins().ishl_imm(high, i64::from(32 - *lsb));
                            builder.ins().bor(left, right)
                        };
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Bit Manipulation ──
                    DecodedInstruction::Rbit { rd, rn } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().bitrev(value);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Rbit32 { rd, rn } => {
                        let value_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let value = builder.ins().ireduce(types::I32, value_raw);
                        let result32 = builder.ins().bitrev(value);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Clz { rd, rn } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().clz(value);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Clz32 { rd, rn } => {
                        let value_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let value = builder.ins().ireduce(types::I32, value_raw);
                        let result32 = builder.ins().clz(value);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Rev { rd, rn } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = builder.ins().bswap(value);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Rev32 { rd, rn } => {
                        let value_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let value = builder.ins().ireduce(types::I32, value_raw);
                        let result32 = builder.ins().bswap(value);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Multiply / Divide ──
                    DecodedInstruction::Mul { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = builder.ins().imul(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Mul32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let result32 = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Madd { rd, rn, rm, ra } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let addend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().iadd(product, addend);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Madd32 { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let addend_raw = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let addend = builder.ins().ireduce(types::I32, addend_raw);
                        let product = builder.ins().imul(lhs, rhs);
                        let result32 = builder.ins().iadd(product, addend);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Msub { rd, rn, rm, ra } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let minuend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().isub(minuend, product);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Msub32 { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let minuend_raw = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let minuend = builder.ins().ireduce(types::I32, minuend_raw);
                        let product = builder.ins().imul(lhs, rhs);
                        let result32 = builder.ins().isub(minuend, product);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Smaddl { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let addend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs32 = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs32 = builder.ins().ireduce(types::I32, rhs_raw);
                        let lhs = builder.ins().sextend(types::I64, lhs32);
                        let rhs = builder.ins().sextend(types::I64, rhs32);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().iadd(product, addend);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Smsubl { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let minuend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs32 = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs32 = builder.ins().ireduce(types::I32, rhs_raw);
                        let lhs = builder.ins().sextend(types::I64, lhs32);
                        let rhs = builder.ins().sextend(types::I64, rhs32);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().isub(minuend, product);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Umaddl { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let addend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs32 = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs32 = builder.ins().ireduce(types::I32, rhs_raw);
                        let lhs = builder.ins().uextend(types::I64, lhs32);
                        let rhs = builder.ins().uextend(types::I64, rhs32);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().iadd(product, addend);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Umsubl { rd, rn, rm, ra } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let minuend = reg_cache.read(&mut builder, state_ptr, *ra);
                        let lhs32 = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs32 = builder.ins().ireduce(types::I32, rhs_raw);
                        let lhs = builder.ins().uextend(types::I64, lhs32);
                        let rhs = builder.ins().uextend(types::I64, rhs32);
                        let product = builder.ins().imul(lhs, rhs);
                        let result = builder.ins().isub(minuend, product);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Smulh { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = builder.ins().smulhi(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Umulh { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = builder.ins().umulhi(lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Udiv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = emit_safe_udiv(&mut builder, lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Udiv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let result32 = emit_safe_udiv32(&mut builder, lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Sdiv { rd, rn, rm } => {
                        let lhs = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs = reg_cache.read(&mut builder, state_ptr, *rm);
                        let result = emit_safe_sdiv(&mut builder, lhs, rhs);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Sdiv32 { rd, rn, rm } => {
                        let lhs_raw = reg_cache.read(&mut builder, state_ptr, *rn);
                        let rhs_raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let lhs = builder.ins().ireduce(types::I32, lhs_raw);
                        let rhs = builder.ins().ireduce(types::I32, rhs_raw);
                        let result32 = emit_safe_sdiv32(&mut builder, lhs, rhs);
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── Branches ──
                    DecodedInstruction::B { target } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_return_to_pc_with_optional_eager(
                            &mut builder,
                            state_ptr,
                            *target,
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Bl { target } => {
                        let link = instr_pc.wrapping_add(4);
                        let link_value = builder.ins().iconst(types::I64, link as i64);
                        reg_cache.write(LINK_REGISTER, link_value);
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_return_to_pc_with_optional_eager(
                            &mut builder,
                            state_ptr,
                            *target,
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::BCond { target, condition } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let cond = emit_eval_condition(&mut builder, state_ptr, *condition);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Cbz { rt, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        reg_cache.flush(&mut builder, state_ptr);
                        let cond = builder.ins().icmp_imm(IntCC::Equal, value, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Cbnz { rt, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        reg_cache.flush(&mut builder, state_ptr);
                        let cond = builder.ins().icmp_imm(IntCC::NotEqual, value, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Cbz32 { rt, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        let narrow = builder.ins().band_imm(value, 0xffff_ffff);
                        reg_cache.flush(&mut builder, state_ptr);
                        let cond = builder.ins().icmp_imm(IntCC::Equal, narrow, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Cbnz32 { rt, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        let narrow = builder.ins().band_imm(value, 0xffff_ffff);
                        reg_cache.flush(&mut builder, state_ptr);
                        let cond = builder.ins().icmp_imm(IntCC::NotEqual, narrow, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Tbz { rt, bit, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        reg_cache.flush(&mut builder, state_ptr);
                        let shifted = builder.ins().ushr_imm(value, i64::from(*bit));
                        let bit_value = builder.ins().band_imm(shifted, 1);
                        let cond = builder.ins().icmp_imm(IntCC::Equal, bit_value, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Tbnz { rt, bit, target } => {
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        reg_cache.flush(&mut builder, state_ptr);
                        let shifted = builder.ins().ushr_imm(value, i64::from(*bit));
                        let bit_value = builder.ins().band_imm(shifted, 1);
                        let cond = builder.ins().icmp_imm(IntCC::NotEqual, bit_value, 0);
                        emit_conditional_return(
                            &mut builder,
                            state_ptr,
                            cond,
                            *target,
                            instr_pc.wrapping_add(4),
                            &eager_direct_targets,
                        );
                        terminated = true;
                    }

                    // ── Address Literals ──
                    DecodedInstruction::Adr { rd, target }
                    | DecodedInstruction::Adrp { rd, target } => {
                        let value = builder.ins().iconst(types::I64, *target as i64);
                        reg_cache.write(*rd, value);
                    }

                    // ── Indirect Branches ──
                    DecodedInstruction::Blr { rn } => {
                        let link = instr_pc.wrapping_add(4);
                        let link_value = builder.ins().iconst(types::I64, link as i64);
                        reg_cache.write(LINK_REGISTER, link_value);

                        let next = reg_cache.read(&mut builder, state_ptr, *rn);
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_return_to_dispatcher(&mut builder, state_ptr, next);
                        terminated = true;
                    }
                    DecodedInstruction::Br { rn } | DecodedInstruction::Ret { rn } => {
                        let next = reg_cache.read(&mut builder, state_ptr, *rn);
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_return_to_dispatcher(&mut builder, state_ptr, next);
                        terminated = true;
                    }

                    // ── Exceptions ──
                    DecodedInstruction::Brk { imm16 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_BREAKPOINT,
                            *imm16,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Svc { imm16 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_SVC,
                            *imm16,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Hvc { imm16 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_HVC,
                            *imm16,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Smc { imm16 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_SMC,
                            *imm16,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }
                    DecodedInstruction::Eret => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_ERET,
                            0,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }

                    // ── Integer Load Unsigned Immediate ──
                    DecodedInstruction::LdrImmU64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 3) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                    }
                    DecodedInstruction::LdrImmU32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 2) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdrImmU16 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdrImmU8 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }

                    // ── Integer Store Unsigned Immediate ──
                    DecodedInstruction::StrImmU64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 3) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrImmU32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 2) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrImmU16 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrImmU8 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }

                    // ── Integer Load Sign-Extending ──
                    DecodedInstruction::LdrsbImm64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value8);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdrsbImm32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value8);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdrshImm64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value16);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdrshImm32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value16);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdrswImm64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 2) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value32 = builder.ins().ireduce(types::I32, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value32);
                        reg_cache.write(*rt, signed64);
                    }

                    // ── Integer Load/Store Unscaled ──
                    DecodedInstruction::LdurI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                    }
                    DecodedInstruction::LdurI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdurI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdurI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::SturI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdursbI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value8);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdursbI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value8);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdurshI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value16);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdurshI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value16);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdurswI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value32 = builder.ins().ireduce(types::I32, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value32);
                        reg_cache.write(*rt, signed64);
                    }

                    // ── Integer Load/Store Register-Offset ──
                    DecodedInstruction::LdrRegU64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 3)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                    }
                    DecodedInstruction::StrRegU64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 3)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdrRegU32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 2)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::StrRegU32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 2)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdrRegU16 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 1)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::StrRegU16 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 1)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdrRegU8 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 0)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::StrRegU8 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 0)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdrsbReg64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 0)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value8);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdrsbReg32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 0)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value8 = builder.ins().ireduce(types::I8, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value8);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdrshReg64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 1)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value16);
                        reg_cache.write(*rt, signed64);
                    }
                    DecodedInstruction::LdrshReg32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 1)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value16 = builder.ins().ireduce(types::I16, loaded64);
                        let signed32 = builder.ins().sextend(types::I32, value16);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, signed32));
                    }
                    DecodedInstruction::LdrswReg64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 2)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let value32 = builder.ins().ireduce(types::I32, loaded64);
                        let signed64 = builder.ins().sextend(types::I64, value32);
                        reg_cache.write(*rt, signed64);
                    }

                    // ── Integer Load/Store Pre/Post-Index ──
                    DecodedInstruction::LdrPostI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreI64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::LdrPostI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreI32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::LdrPostI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded8 = builder.ins().ireduce(types::I8, loaded64);
                        let loaded32 = builder.ins().uextend(types::I32, loaded8);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded8 = builder.ins().ireduce(types::I8, loaded64);
                        let loaded32 = builder.ins().uextend(types::I32, loaded8);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreI8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::LdrPostI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded16 = builder.ins().ireduce(types::I16, loaded64);
                        let loaded32 = builder.ins().uextend(types::I32, loaded16);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::LdrPreI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded16 = builder.ins().ireduce(types::I16, loaded64);
                        let loaded32 = builder.ins().uextend(types::I32, loaded16);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }
                    DecodedInstruction::StrPostI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = base;
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let updated = builder.ins().iadd_imm(base, i64::from(*imm9));
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, updated);
                    }
                    DecodedInstruction::StrPreI16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, address);
                    }

                    // ── Integer Load/Store Pair ──
                    DecodedInstruction::LdpI64 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 8;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 8);
                        let first = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let second = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            second_address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, first);
                        reg_cache.write(*rt2, second);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::StpI64 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 8;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 8);
                        let first = reg_cache.read(&mut builder, state_ptr, *rt);
                        let second = reg_cache.read(&mut builder, state_ptr, *rt2);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            first,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            second_address,
                            second,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::LdpI32 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 4;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 4);
                        let first64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        let second64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            second_address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let first32 = builder.ins().ireduce(types::I32, first64);
                        let second32 = builder.ins().ireduce(types::I32, second64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, first32));
                        reg_cache.write(*rt2, builder.ins().uextend(types::I64, second32));
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::Ldpsw {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 4;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 4);
                        let first64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        let second64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            second_address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let first32 = builder.ins().ireduce(types::I32, first64);
                        let second32 = builder.ins().ireduce(types::I32, second64);
                        let first = builder.ins().sextend(types::I64, first32);
                        let second = builder.ins().sextend(types::I64, second32);
                        reg_cache.write(*rt, first);
                        reg_cache.write(*rt2, second);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::StpI32 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 4;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 4);
                        let first64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let second64 = reg_cache.read(&mut builder, state_ptr, *rt2);
                        let first32 = builder.ins().ireduce(types::I32, first64);
                        let second32 = builder.ins().ireduce(types::I32, second64);
                        let first = builder.ins().uextend(types::I64, first32);
                        let second = builder.ins().uextend(types::I64, second32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            first,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            second_address,
                            second,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── FP/SIMD Load/Store Pair ──
                    DecodedInstruction::LdpFp32 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 4;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 4);
                        let first64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );
                        let second64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            second_address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, first64);
                        reg_cache.write_v_scalar64(&mut builder, *rt2, second64);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::StpFp32 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 4;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 4);
                        let first64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let second64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt2);
                        let first32 = builder.ins().ireduce(types::I32, first64);
                        let second32 = builder.ins().ireduce(types::I32, second64);
                        let first = builder.ins().uextend(types::I64, first32);
                        let second = builder.ins().uextend(types::I64, second32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            first,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            second_address,
                            second,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::LdpFp64 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 8;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 8);
                        let first64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let second64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            second_address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, first64);
                        reg_cache.write_v_scalar64(&mut builder, *rt2, second64);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::StpFp64 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 8;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let second_address = builder.ins().iadd_imm(address, 8);
                        let first = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let second = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt2);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            first,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            second_address,
                            second,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::LdpV128 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 16;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let address_16 = builder.ins().iadd_imm(address, 16);
                        let address_24 = builder.ins().iadd_imm(address, 24);

                        let first_lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let first_hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_8,
                        );
                        let second_lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_16,
                        );
                        let second_hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_24,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let first_existing = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let first_lanes = builder.ins().insertlane(first_existing, first_lo, 0);
                        let first_vector = builder.ins().insertlane(first_lanes, first_hi, 1);
                        reg_cache.write_v(*rt, first_vector);

                        let second_existing = emit_read_vreg128(&mut builder, state_ptr, *rt2);
                        let second_lanes = builder.ins().insertlane(second_existing, second_lo, 0);
                        let second_vector = builder.ins().insertlane(second_lanes, second_hi, 1);
                        reg_cache.write_v(*rt2, second_vector);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }
                    DecodedInstruction::StpV128 {
                        rt,
                        rt2,
                        rn,
                        imm7,
                        mode,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = i64::from(*imm7) * 16;
                        let (address, writeback) =
                            emit_pair_address(&mut builder, base, offset, *mode);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let address_16 = builder.ins().iadd_imm(address, 16);
                        let address_24 = builder.ins().iadd_imm(address, 24);

                        let first_vector = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let first_lo = builder.ins().extractlane(first_vector, 0);
                        let first_hi = builder.ins().extractlane(first_vector, 1);

                        let second_vector = emit_read_vreg128(&mut builder, state_ptr, *rt2);
                        let second_lo = builder.ins().extractlane(second_vector, 0);
                        let second_hi = builder.ins().extractlane(second_vector, 1);

                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            first_lo,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_8,
                            first_hi,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_16,
                            second_lo,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_24,
                            second_hi,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        if let Some(new_base) = writeback {
                            write_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn, new_base);
                        }
                    }

                    // ── FP/SIMD Unscaled Load/Store ──
                    DecodedInstruction::LdurFp8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdurFp16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdurFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdurFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdurFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_8,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cleared = builder.ins().splat(types::I64X2, zero);
                        let with_lo = builder.ins().insertlane(cleared, lo, 0);
                        let vec = builder.ins().insertlane(with_lo, hi, 1);
                        reg_cache.write_v(*rt, vec);
                    }
                    DecodedInstruction::SturFp8 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturFp16 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturFp32 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturFp64 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let value = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::SturFp128 { rt, rn, imm9 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm9));
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let vec = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let lo = builder.ins().extractlane(vec, 0);
                        let hi = builder.ins().extractlane(vec, 1);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            lo,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_8,
                            hi,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }

                    // ── FP/SIMD Register-Offset Load/Store ──
                    DecodedInstruction::LdrFpReg128 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 4)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_8,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cleared = builder.ins().splat(types::I64X2, zero);
                        let with_lo = builder.ins().insertlane(cleared, lo, 0);
                        let vec = builder.ins().insertlane(with_lo, hi, 1);
                        reg_cache.write_v(*rt, vec);
                    }
                    DecodedInstruction::StrFpReg128 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 4)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let vec = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let lo = builder.ins().extractlane(vec, 0);
                        let hi = builder.ins().extractlane(vec, 1);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            lo,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_8,
                            hi,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::LdrFpReg32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 2)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdrFpReg64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 3)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::StrFpReg32 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 2)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrFpReg64 {
                        rt,
                        rn,
                        rm,
                        option,
                        shift,
                    } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let raw = reg_cache.read(&mut builder, state_ptr, *rm);
                        let Some(offset) =
                            emit_resolve_ls_reg_offset(&mut builder, raw, *option, *shift, 3)
                        else {
                            emit_exit_with_imm(
                                &mut builder,
                                state_ptr,
                                EXIT_PRIVILEGE_FAULT,
                                0,
                                instr_pc.wrapping_add(4),
                            );
                            terminated = true;
                            continue;
                        };

                        let address = builder.ins().iadd(base, offset);
                        let value = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }

                    // ── FP/SIMD Unsigned-Imm Load/Store ──
                    DecodedInstruction::LdrFpImm8 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdrFpImm16 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdrFpImm32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 2) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdrFpImm64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 3) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write_v_scalar64(&mut builder, *rt, loaded64);
                    }
                    DecodedInstruction::LdrFpImm128 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 4) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let lo = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );
                        let hi = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address_8,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cleared = builder.ins().splat(types::I64X2, zero);
                        let with_lo = builder.ins().insertlane(cleared, lo, 0);
                        let vec = builder.ins().insertlane(with_lo, hi, 1);
                        reg_cache.write_v(*rt, vec);
                    }
                    DecodedInstruction::StrFpImm8 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let address = builder.ins().iadd_imm(base, i64::from(*imm12));
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrFpImm16 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 1) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrFpImm32 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 2) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value64 = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrFpImm64 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 3) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let value = emit_read_v_scalar_bits64(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StrFpImm128 { rt, rn, imm12 } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let base = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let offset = (u64::from(*imm12) << 4) as i64;
                        let address = builder.ins().iadd_imm(base, offset);
                        let address_8 = builder.ins().iadd_imm(address, 8);
                        let vec = emit_read_vreg128(&mut builder, state_ptr, *rt);
                        let lo = builder.ins().extractlane(vec, 0);
                        let hi = builder.ins().extractlane(vec, 1);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            lo,
                        );
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address_8,
                            hi,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }

                    // ── Exclusive Load/Store ──
                    DecodedInstruction::LdxrU64 { rt, rn }
                    | DecodedInstruction::LdaxrU64 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        emit_state_set_exclusive_reservation(&mut builder, state_ptr, address, 8);
                        reg_cache.write(*rt, loaded);
                    }
                    DecodedInstruction::LdxrU32 { rt, rn }
                    | DecodedInstruction::LdaxrU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        emit_state_set_exclusive_reservation(&mut builder, state_ptr, address, 4);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::StxrU64 { rs, rt, rn }
                    | DecodedInstruction::StlxrU64 { rs, rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        let reservation_ok = emit_state_check_exclusive_reservation(
                            &mut builder,
                            state_ptr,
                            address,
                            8,
                        );
                        let store_block = builder.create_block();
                        let mismatch_block = builder.create_block();
                        let done_store_block = builder.create_block();
                        builder.append_block_param(done_store_block, types::I64);
                        builder
                            .ins()
                            .brif(reservation_ok, store_block, &[], mismatch_block, &[]);
                        builder.seal_block(store_block);
                        builder.seal_block(mismatch_block);

                        builder.switch_to_block(store_block);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        let success_status = builder.ins().iconst(types::I64, 0);
                        let success_arg = [BlockArg::from(success_status)];
                        builder.ins().jump(done_store_block, &success_arg);

                        builder.switch_to_block(mismatch_block);
                        let fail_status = builder.ins().iconst(types::I64, 1);
                        let fail_arg = [BlockArg::from(fail_status)];
                        builder.ins().jump(done_store_block, &fail_arg);

                        builder.seal_block(done_store_block);
                        builder.switch_to_block(done_store_block);
                        let status64 = builder.block_params(done_store_block)[0];
                        emit_state_clear_exclusive_reservation(&mut builder, state_ptr);

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let status32 = builder.ins().ireduce(types::I32, status64);
                        reg_cache.write(*rs, builder.ins().uextend(types::I64, status32));
                    }
                    DecodedInstruction::StxrU32 { rs, rt, rn }
                    | DecodedInstruction::StlxrU32 { rs, rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        let reservation_ok = emit_state_check_exclusive_reservation(
                            &mut builder,
                            state_ptr,
                            address,
                            4,
                        );
                        let store_block = builder.create_block();
                        let mismatch_block = builder.create_block();
                        let done_store_block = builder.create_block();
                        builder.append_block_param(done_store_block, types::I64);
                        builder
                            .ins()
                            .brif(reservation_ok, store_block, &[], mismatch_block, &[]);
                        builder.seal_block(store_block);
                        builder.seal_block(mismatch_block);

                        builder.switch_to_block(store_block);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        let success_status = builder.ins().iconst(types::I64, 0);
                        let success_arg = [BlockArg::from(success_status)];
                        builder.ins().jump(done_store_block, &success_arg);

                        builder.switch_to_block(mismatch_block);
                        let fail_status = builder.ins().iconst(types::I64, 1);
                        let fail_arg = [BlockArg::from(fail_status)];
                        builder.ins().jump(done_store_block, &fail_arg);

                        builder.seal_block(done_store_block);
                        builder.switch_to_block(done_store_block);
                        let status64 = builder.block_params(done_store_block)[0];
                        emit_state_clear_exclusive_reservation(&mut builder, state_ptr);

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let status32 = builder.ins().ireduce(types::I32, status64);
                        reg_cache.write(*rs, builder.ins().uextend(types::I64, status32));
                    }

                    // ── Ordered (Acquire/Release) Load/Store ──
                    DecodedInstruction::LdarU64 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u64,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        reg_cache.write(*rt, loaded);
                    }
                    DecodedInstruction::LdarU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u32,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdarbU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u8,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::LdarhU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let loaded64 = emit_call_mem_read(
                            &mut builder,
                            memory_helpers.read_u16,
                            state_ptr,
                            address,
                        );

                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                        let loaded32 = builder.ins().ireduce(types::I32, loaded64);
                        reg_cache.write(*rt, builder.ins().uextend(types::I64, loaded32));
                    }
                    DecodedInstruction::StlrU64 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value = reg_cache.read(&mut builder, state_ptr, *rt);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u64,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StlrU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value32 = builder.ins().ireduce(types::I32, value64);
                        let value = builder.ins().uextend(types::I64, value32);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u32,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StlrbU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value8 = builder.ins().ireduce(types::I8, value64);
                        let value = builder.ins().uextend(types::I64, value8);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u8,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }
                    DecodedInstruction::StlrhU32 { rt, rn } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        let address = read_gp_or_sp(&mut builder, state_ptr, &mut reg_cache, *rn);
                        let value64 = reg_cache.read(&mut builder, state_ptr, *rt);
                        let value16 = builder.ins().ireduce(types::I16, value64);
                        let value = builder.ins().uextend(types::I64, value16);
                        emit_call_mem_write(
                            &mut builder,
                            memory_helpers.write_u16,
                            state_ptr,
                            address,
                            value,
                        );
                        emit_fault_check(&mut builder, state_ptr, instr_pc);
                    }

                    // ── Bitfield Operations ──
                    DecodedInstruction::Ubfm { rd, rn, immr, imms } => {
                        let source = reg_cache.read(&mut builder, state_ptr, *rn);
                        let result = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let shifted = builder.ins().ushr_imm(source, i64::from(*immr));
                            builder.ins().band_imm(shifted, mask as i64)
                        } else {
                            let width = u32::from(*imms) + 1;
                            let mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let lsb = 64u32 - u32::from(*immr);
                            let masked = builder.ins().band_imm(source, mask as i64);
                            builder.ins().ishl_imm(masked, i64::from(lsb))
                        };
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Ubfm32 { rd, rn, immr, imms } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source = builder.ins().ireduce(types::I32, source64);
                        let result32 = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let shifted = builder.ins().ushr_imm(source, i64::from(*immr));
                            builder.ins().band_imm(shifted, i64::from(mask))
                        } else {
                            let width = u32::from(*imms) + 1;
                            let mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let lsb = 32u32 - u32::from(*immr);
                            let masked = builder.ins().band_imm(source, i64::from(mask));
                            builder.ins().ishl_imm(masked, i64::from(lsb))
                        };
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Sbfm { rd, rn, immr, imms } => {
                        let source = reg_cache.read(&mut builder, state_ptr, *rn);
                        // First compute UBFM result
                        let ubfm_result = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let shifted = builder.ins().ushr_imm(source, i64::from(*immr));
                            builder.ins().band_imm(shifted, mask as i64)
                        } else {
                            let width = u32::from(*imms) + 1;
                            let mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let lsb = 64u32 - u32::from(*immr);
                            let masked = builder.ins().band_imm(source, mask as i64);
                            builder.ins().ishl_imm(masked, i64::from(lsb))
                        };
                        // Sign extend from the sign_bit position
                        let sign_bit = if *immr <= *imms {
                            u32::from(*imms - *immr)
                        } else {
                            (64u32 - u32::from(*immr)) + u32::from(*imms)
                        };
                        let result = if sign_bit >= 63 {
                            ubfm_result
                        } else {
                            let shift_amount = 63 - sign_bit;
                            let shifted_left =
                                builder.ins().ishl_imm(ubfm_result, i64::from(shift_amount));
                            builder
                                .ins()
                                .sshr_imm(shifted_left, i64::from(shift_amount))
                        };
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Sbfm32 { rd, rn, immr, imms } => {
                        let source64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let source = builder.ins().ireduce(types::I32, source64);
                        let ubfm_result = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let shifted = builder.ins().ushr_imm(source, i64::from(*immr));
                            builder.ins().band_imm(shifted, i64::from(mask))
                        } else {
                            let width = u32::from(*imms) + 1;
                            let mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let lsb = 32u32 - u32::from(*immr);
                            let masked = builder.ins().band_imm(source, i64::from(mask));
                            builder.ins().ishl_imm(masked, i64::from(lsb))
                        };
                        let sign_bit = if *immr <= *imms {
                            u32::from(*imms - *immr)
                        } else {
                            (32u32 - u32::from(*immr)) + u32::from(*imms)
                        };
                        let result32 = if sign_bit >= 31 {
                            ubfm_result
                        } else {
                            let shift_amount = 31 - sign_bit;
                            let shifted_left =
                                builder.ins().ishl_imm(ubfm_result, i64::from(shift_amount));
                            builder
                                .ins()
                                .sshr_imm(shifted_left, i64::from(shift_amount))
                        };
                        // SBFM32 result is sign-extended to 64-bit
                        let result = builder.ins().sextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Bfm { rd, rn, immr, imms } => {
                        let src = reg_cache.read(&mut builder, state_ptr, *rn);
                        let dst = reg_cache.read(&mut builder, state_ptr, *rd);
                        let result = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let low_mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let insert_mask = low_mask.wrapping_shl(u32::from(*immr));
                            let src_masked = builder.ins().band_imm(src, low_mask as i64);
                            let src_shifted = builder.ins().ishl_imm(src_masked, i64::from(*immr));
                            let dst_preserved = builder.ins().band_imm(dst, !insert_mask as i64);
                            let src_inserted =
                                builder.ins().band_imm(src_shifted, insert_mask as i64);
                            builder.ins().bor(dst_preserved, src_inserted)
                        } else {
                            let width = u32::from(*imms) + 1;
                            let low_mask = if width >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << width) - 1
                            };
                            let lsb = 64u32 - u32::from(*immr);
                            let insert_mask = low_mask.wrapping_shl(lsb);
                            let src_masked = builder.ins().band_imm(src, low_mask as i64);
                            let src_shifted = builder.ins().ishl_imm(src_masked, i64::from(lsb));
                            let dst_preserved = builder.ins().band_imm(dst, !insert_mask as i64);
                            let src_inserted =
                                builder.ins().band_imm(src_shifted, insert_mask as i64);
                            builder.ins().bor(dst_preserved, src_inserted)
                        };
                        reg_cache.write(*rd, result);
                    }
                    DecodedInstruction::Bfm32 { rd, rn, immr, imms } => {
                        let src64 = reg_cache.read(&mut builder, state_ptr, *rn);
                        let src = builder.ins().ireduce(types::I32, src64);
                        let dst64 = reg_cache.read(&mut builder, state_ptr, *rd);
                        let dst = builder.ins().ireduce(types::I32, dst64);
                        let result32 = if *immr <= *imms {
                            let width = u32::from(*imms - *immr) + 1;
                            let low_mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let insert_mask = low_mask.wrapping_shl(u32::from(*immr));
                            let src_masked = builder.ins().band_imm(src, i64::from(low_mask));
                            let src_shifted = builder.ins().ishl_imm(src_masked, i64::from(*immr));
                            let dst_preserved =
                                builder.ins().band_imm(dst, i64::from(!insert_mask));
                            let src_inserted =
                                builder.ins().band_imm(src_shifted, i64::from(insert_mask));
                            builder.ins().bor(dst_preserved, src_inserted)
                        } else {
                            let width = u32::from(*imms) + 1;
                            let low_mask = if width >= 32 {
                                u32::MAX
                            } else {
                                (1u32 << width) - 1
                            };
                            let lsb = 32u32 - u32::from(*immr);
                            let insert_mask = low_mask.wrapping_shl(lsb);
                            let src_masked = builder.ins().band_imm(src, i64::from(low_mask));
                            let src_shifted = builder.ins().ishl_imm(src_masked, i64::from(lsb));
                            let dst_preserved =
                                builder.ins().band_imm(dst, i64::from(!insert_mask));
                            let src_inserted =
                                builder.ins().band_imm(src_shifted, i64::from(insert_mask));
                            builder.ins().bor(dst_preserved, src_inserted)
                        };
                        let result = builder.ins().uextend(types::I64, result32);
                        reg_cache.write(*rd, result);
                    }

                    // ── System Registers ──
                    DecodedInstruction::Mrs { rt, reg } => {
                        // Flush pending NZCV if reading NZCV register
                        if *reg == SystemRegister::Nzcv {
                            emit_flush_pending_nzcv(&mut builder, state_ptr);
                        }
                        reg_cache.flush(&mut builder, state_ptr);
                        let reg_idx = crate::engine::sys_reg_to_index(reg);
                        let reg_idx_val = builder.ins().iconst(types::I64, reg_idx as i64);
                        let inst = memory_helpers
                            .mrs
                            .call(&mut builder, &[state_ptr, reg_idx_val]);
                        let result = builder.inst_results(inst)[0];
                        // Check for privilege fault
                        let exit_code = builder.ins().load(
                            types::I32,
                            MemFlags::new(),
                            state_ptr,
                            EXIT_CODE_OFFSET,
                        );
                        let has_exit = builder.ins().icmp_imm(IntCC::NotEqual, exit_code, 0);
                        let fault_block = builder.create_block();
                        let success_block = builder.create_block();
                        builder
                            .ins()
                            .brif(has_exit, fault_block, &[], success_block, &[]);
                        builder.seal_block(fault_block);
                        builder.seal_block(success_block);

                        builder.switch_to_block(fault_block);
                        let fault_pc = builder
                            .ins()
                            .iconst(types::I64, instr_pc.wrapping_add(4) as i64);
                        emit_return_to_dispatcher(&mut builder, state_ptr, fault_pc);

                        builder.switch_to_block(success_block);
                        reg_cache.write(*rt, result);
                    }
                    DecodedInstruction::Msr { reg, rt } => {
                        // Flush pending NZCV before any MSR
                        emit_flush_pending_nzcv(&mut builder, state_ptr);
                        reg_cache.flush(&mut builder, state_ptr);
                        // Read the GP register value directly from CpuState (already flushed)
                        let value = if *rt == 31 {
                            builder.ins().iconst(types::I64, 0)
                        } else {
                            let offset = REG_FILE_BASE_OFFSET + i32::from(*rt) * REG_FILE_STRIDE;
                            builder
                                .ins()
                                .load(types::I64, MemFlags::new(), state_ptr, offset)
                        };
                        let reg_idx = crate::engine::sys_reg_to_index(reg);
                        let reg_idx_val = builder.ins().iconst(types::I64, reg_idx as i64);
                        let _ = memory_helpers
                            .msr
                            .call(&mut builder, &[state_ptr, reg_idx_val, value]);
                        // Check for privilege fault
                        let exit_code = builder.ins().load(
                            types::I32,
                            MemFlags::new(),
                            state_ptr,
                            EXIT_CODE_OFFSET,
                        );
                        let has_exit = builder.ins().icmp_imm(IntCC::NotEqual, exit_code, 0);
                        let fault_block = builder.create_block();
                        let success_block = builder.create_block();
                        builder
                            .ins()
                            .brif(has_exit, fault_block, &[], success_block, &[]);
                        builder.seal_block(fault_block);
                        builder.seal_block(success_block);

                        builder.switch_to_block(fault_block);
                        let fault_pc = builder
                            .ins()
                            .iconst(types::I64, instr_pc.wrapping_add(4) as i64);
                        emit_return_to_dispatcher(&mut builder, state_ptr, fault_pc);

                        builder.switch_to_block(success_block);
                        // Clear pending NZCV if we wrote NZCV
                        if *reg == SystemRegister::Nzcv {
                            clear_pending_nzcv();
                        }
                    }

                    // ── Exclusive Pair Fallback (interpreter) ──
                    DecodedInstruction::LdxpU32 { .. }
                    | DecodedInstruction::LdxpU64 { .. }
                    | DecodedInstruction::LdaxpU32 { .. }
                    | DecodedInstruction::LdaxpU64 { .. }
                    | DecodedInstruction::StxpU32 { .. }
                    | DecodedInstruction::StxpU64 { .. }
                    | DecodedInstruction::StlxpU32 { .. }
                    | DecodedInstruction::StlxpU64 { .. }
                    | DecodedInstruction::SimdSshll { .. }
                    | DecodedInstruction::SimdUshll { .. }
                    | DecodedInstruction::SimdSshr { .. }
                    | DecodedInstruction::SimdUshr { .. }
                    | DecodedInstruction::SimdShl { .. }
                    | DecodedInstruction::FcvtRoundFromFloat { .. }
                    | DecodedInstruction::LdrshPostI32 { .. }
                    | DecodedInstruction::LdrshPostI64 { .. }
                    | DecodedInstruction::LdrshPreI32 { .. }
                    | DecodedInstruction::LdrshPreI64 { .. }
                    | DecodedInstruction::LdrsbPostI32 { .. }
                    | DecodedInstruction::LdrsbPostI64 { .. }
                    | DecodedInstruction::LdrsbPreI32 { .. }
                    | DecodedInstruction::LdrsbPreI64 { .. }
                    | DecodedInstruction::LdrswPostI64 { .. }
                    | DecodedInstruction::LdrswPreI64 { .. } => {
                        reg_cache.flush(&mut builder, state_ptr);
                        emit_exit_with_imm(
                            &mut builder,
                            state_ptr,
                            EXIT_PRIVILEGE_FAULT,
                            0,
                            instr_pc.wrapping_add(4),
                        );
                        terminated = true;
                    }
                }

                if terminated {
                    break;
                }

                index += 1;
            }

            if !terminated {
                reg_cache.flush(&mut builder, state_ptr);
                let fallthrough = builder
                    .ins()
                    .iconst(types::I64, block.fallthrough_pc as i64);
                emit_return_to_dispatcher(&mut builder, state_ptr, fallthrough);
            }

            set_current_guest_pc(None);
            builder.finalize();
        }

        let func_ir = self.ctx.func.display().to_string();
        if let Err(errors) = cranelift_codegen::verify_function(&self.ctx.func, &*self.isa) {
            // verify now for debugging, remove later.
            return Err(anyhow!(
                "cranelift verifier failed for block @ 0x{start_pc:016x}:\n{errors}\n{func_ir}, {block:?}",
                start_pc = block.start_pc,
            ));
        }
        let mut ctrl_plane = cranelift_codegen::control::ControlPlane::default();
        let compiled = self.ctx.compile(&*self.isa, &mut ctrl_plane).map_err(|e| {
            anyhow!(
                "cranelift compilation failed for block @ 0x{start_pc:016x}: {}\n{}",
                e.inner,
                func_ir,
                start_pc = block.start_pc,
            )
        })?;

        let code_bytes = compiled.code_buffer();
        let code_align = (compiled.buffer.alignment as u64)
            .max(self.isa.function_alignment().minimum as u64)
            .max(self.isa.symbol_alignment());

        let (code_size, host_pc_map) = {
            let entries = compiled
                .buffer
                .get_srclocs_sorted()
                .iter()
                .filter_map(|srcloc| {
                    if srcloc.loc.is_default() || srcloc.start >= srcloc.end {
                        return None;
                    }

                    Some(HostPcMapEntry {
                        host_offset_start: srcloc.start,
                        host_offset_end: srcloc.end,
                        guest_pc: u64::from(srcloc.loc.bits()),
                    })
                })
                .collect::<Vec<_>>();

            let code_size = u32::try_from(code_bytes.len())
                .context("compiled block code size exceeds u32 range")?;

            (code_size, Arc::<[HostPcMapEntry]>::from(entries))
        };

        let code_ptr = self
            .arena
            .alloc_code(code_bytes.len(), code_align as usize)
            .context("failed to allocate code in JIT arena")?;

        // `pthread_jit_write_protect_np` required on macOS.
        self.arena.write_unprotect();

        unsafe {
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), code_ptr, code_bytes.len());
        }

        for reloc in compiled.buffer.relocs() {
            let at = unsafe { code_ptr.add(reloc.offset as usize) };
            match reloc.kind {
                cranelift_codegen::binemit::Reloc::Abs8 => {
                    // FinalizedRelocTarget::Func(offset)
                    let target_addr = match &reloc.target {
                        cranelift_codegen::FinalizedRelocTarget::Func(offset) => unsafe {
                            code_ptr.add(*offset as usize) as u64
                        },
                        _ => {
                            return Err(anyhow!(
                                "unexpected Abs8 reloc target: {:?}",
                                reloc.target
                            ));
                        }
                    };
                    let what = target_addr.wrapping_add(reloc.addend as u64);
                    unsafe {
                        std::ptr::write_unaligned(at as *mut u64, what);
                    }
                }
                #[cfg(target_arch = "x86_64")]
                cranelift_codegen::binemit::Reloc::X86PCRel4
                | cranelift_codegen::binemit::Reloc::X86CallPCRel4 => {
                    let target_addr = match &reloc.target {
                        cranelift_codegen::FinalizedRelocTarget::Func(offset) => unsafe {
                            code_ptr.add(*offset as usize) as isize
                        },
                        _ => {
                            return Err(anyhow!(
                                "unexpected X86PCRel4 reloc target: {:?}",
                                reloc.target
                            ));
                        }
                    };
                    let what = target_addr.wrapping_add(reloc.addend as isize);
                    let pcrel = i32::try_from(what - at as isize)
                        .context("rel32 overflow in JIT relocation")?;
                    unsafe {
                        std::ptr::write_unaligned(at as *mut i32, pcrel);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                cranelift_codegen::binemit::Reloc::Arm64Call => {
                    let target_addr = match &reloc.target {
                        cranelift_codegen::FinalizedRelocTarget::Func(offset) => unsafe {
                            code_ptr.add(*offset as usize) as isize
                        },
                        _ => {
                            return Err(anyhow!(
                                "unexpected Arm64Call reloc target: {:?}",
                                reloc.target
                            ));
                        }
                    };
                    let what = target_addr.wrapping_add(reloc.addend as isize);
                    let diff = (what - at as isize) >> 2;
                    if !((diff >> 25 == -1) || (diff >> 25 == 0)) {
                        return Err(anyhow!("Arm64Call reloc out of range"));
                    }
                    let chop = 32 - 26;
                    let imm26 = (diff as u32) << chop >> chop;
                    let iptr = at as *mut u32;
                    unsafe {
                        let inst = iptr.read_unaligned();
                        iptr.write_unaligned(inst | imm26);
                    }
                }
                other => {
                    return Err(anyhow!("unhandled relocation kind: {:?}", other));
                }
            }
        }

        // icache invalidation (required on aarch64).
        // SAFETY: `code_ptr` and `code_bytes.len()` are guaranteed to be valid.
        unsafe {
            crate::sys::icache_invalidate(code_ptr, code_bytes.len());
        }

        // Switch back to execute mode (macOS MAP_JIT W^X).
        self.arena.write_protect();

        let return_address_slot = return_slot_ptr as usize;

        // SAFETY: Cranelift generated code uses the exact signature declared above.
        let entry = unsafe { std::mem::transmute::<*const u8, BlockEntry>(code_ptr as *const u8) };

        Ok(CompiledBlock {
            entry,
            code_start: code_ptr as usize,
            code_size,
            host_pc_map,
            strategy,
            return_address_slot,
            return_link_entry_count,
        })
    }

    fn declare_memory_helpers(
        &mut self,
        pointer_ty: cranelift_codegen::ir::Type,
    ) -> Result<MemoryHelperRefs> {
        let cc = self.isa.default_call_conv();
        let read_u8 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_read_u8 as *const u8,
            &[pointer_ty, types::I64],
            &[types::I64],
        );
        let read_u16 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_read_u16 as *const u8,
            &[pointer_ty, types::I64],
            &[types::I64],
        );
        let read_u32 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_read_u32 as *const u8,
            &[pointer_ty, types::I64],
            &[types::I64],
        );
        let read_u64 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_read_u64 as *const u8,
            &[pointer_ty, types::I64],
            &[types::I64],
        );
        let write_u8 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_write_u8 as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[types::I32],
        );
        let write_u16 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_write_u16 as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[types::I32],
        );
        let write_u32 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_write_u32 as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[types::I32],
        );
        let write_u64 = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mem_write_u64 as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[types::I32],
        );
        let record_link_miss = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_record_link_miss as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[],
        );
        let state_clrex = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_state_clrex as *const u8,
            &[pointer_ty],
            &[],
        );
        let mrs = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_mrs as *const u8,
            &[pointer_ty, types::I64],
            &[types::I64],
        );
        let msr = declare_helper(
            &mut self.ctx.func,
            cc,
            crate::engine::ryurs_jit_msr as *const u8,
            &[pointer_ty, types::I64, types::I64],
            &[],
        );
        Ok(MemoryHelperRefs {
            read_u8,
            read_u16,
            read_u32,
            read_u64,
            write_u8,
            write_u16,
            write_u32,
            write_u64,
            record_link_miss,
            state_clrex,
            mrs,
            msr,
        })
    }
}

fn declare_helper(
    func: &mut cranelift_codegen::ir::Function,
    call_conv: cranelift_codegen::isa::CallConv,
    raw_ptr: *const u8,
    params: &[cranelift_codegen::ir::Type],
    returns: &[cranelift_codegen::ir::Type],
) -> HelperRef {
    let mut signature = cranelift_codegen::ir::Signature::new(call_conv);
    for param in params {
        signature.params.push(AbiParam::new(*param));
    }
    for ret in returns {
        signature.returns.push(AbiParam::new(*ret));
    }
    let sig_ref = func.import_signature(signature);
    HelperRef {
        sig_ref,
        raw_ptr: raw_ptr as u64,
    }
}

struct JitRegCache {
    values: [Option<Value>; 31],
    dirty: [bool; 31],
    sp_value: Option<Value>,
    sp_dirty: bool,
    vec_values: [Option<Value>; 32],
    vec_dirty: [bool; 32],
    trusted: bool, // Make sure the first access to cpu state can trap so that the safepoint trap will be always working.
}

impl JitRegCache {
    fn new() -> Self {
        Self {
            values: [None; 31],
            dirty: [false; 31],
            vec_values: [None; 32],
            vec_dirty: [false; 32],
            sp_value: None,
            sp_dirty: false,
            trusted: false,
        }
    }

    fn mem_flag(&mut self) -> MemFlags {
        let flag = if self.trusted {
            MemFlags::trusted().with_can_move()
        } else {
            self.trusted = true;
            MemFlags::new()
        };

        flag.with_alias_region(Some(AliasRegion::Vmctx))
    }

    fn read(&mut self, builder: &mut FunctionBuilder, state_ptr: Value, reg: u8) -> Value {
        if reg == 31 {
            return builder.ins().iconst(types::I64, 0);
        }

        if reg >= 32 {
            unreachable!("register number should be within [0, 31], found {}", reg);
        }

        let idx = usize::from(reg);
        if let Some(value) = self.values[idx] {
            return value;
        }

        let offset = REG_FILE_BASE_OFFSET + i32::from(reg) * REG_FILE_STRIDE;
        let value = builder
            .ins()
            .load(types::I64, self.mem_flag(), state_ptr, offset);
        self.values[idx] = Some(value);
        value
    }

    fn read_sp(&mut self, builder: &mut FunctionBuilder, state_ptr: Value) -> Value {
        if let Some(value) = self.sp_value {
            return value;
        }

        let value = builder
            .ins()
            .load(types::I64, self.mem_flag(), state_ptr, SP_OFFSET);
        self.sp_value = Some(value);
        value
    }

    fn read_v(&mut self, builder: &mut FunctionBuilder, state_ptr: Value, reg: u8) -> Value {
        if reg >= 32 {
            unreachable!(
                "vector register number should be within [0, 31], found {}",
                reg
            );
        }

        let idx = usize::from(reg);
        if let Some(value) = self.vec_values[idx] {
            return value;
        }

        let offset = CPUSTATE_VREG_BASE_OFFSET + i32::from(reg) * CPUSTATE_VREG_STRIDE;
        let value = builder
            .ins()
            .load(types::I64X2, self.mem_flag(), state_ptr, offset);
        self.vec_values[idx] = Some(value);
        value
    }

    fn write(&mut self, reg: u8, value: Value) {
        if reg == 31 {
            // zero register
            return;
        }

        if reg >= 32 {
            unreachable!("register number should be within [0, 31], found {}", reg);
        }

        let idx = usize::from(reg);
        self.values[idx] = Some(value);
        self.dirty[idx] = true;
    }

    fn write_sp(&mut self, value: Value) {
        self.sp_value = Some(value);
        self.sp_dirty = true;
    }

    fn write_v(&mut self, reg: u8, value: Value) {
        if reg >= 32 {
            unreachable!(
                "vector register number should be within [0, 31], found {}",
                reg
            );
        }

        let idx = usize::from(reg);
        self.vec_values[idx] = Some(value);
        self.vec_dirty[idx] = true;
    }

    /// Read scalar 32-bit value from the low lane of a cached vector register.
    fn read_v_scalar32(
        &mut self,
        builder: &mut FunctionBuilder,
        state_ptr: Value,
        reg: u8,
    ) -> Value {
        let vec = self.read_v(builder, state_ptr, reg);
        let vec32 = bitcast_little(builder, vec, types::I32X4);
        builder.ins().extractlane(vec32, 0)
    }

    /// Read scalar 64-bit value from the low lane of a cached vector register.
    fn read_v_scalar64(
        &mut self,
        builder: &mut FunctionBuilder,
        state_ptr: Value,
        reg: u8,
    ) -> Value {
        let vec = self.read_v(builder, state_ptr, reg);
        builder.ins().extractlane(vec, 0)
    }

    /// Write a 32-bit scalar into a vector register (upper bits zeroed), through the cache.
    fn write_v_scalar32(&mut self, builder: &mut FunctionBuilder, reg: u8, value_bits: Value) {
        let value64 = builder.ins().uextend(types::I64, value_bits);
        let updated = builder.ins().scalar_to_vector(types::I64X2, value64);
        self.write_v(reg, updated);
    }

    /// Write a 64-bit scalar into a vector register (upper 64 bits zeroed), through the cache.
    fn write_v_scalar64(&mut self, builder: &mut FunctionBuilder, reg: u8, value_bits: Value) {
        let updated = builder.ins().scalar_to_vector(types::I64X2, value_bits);
        self.write_v(reg, updated);
    }

    fn flush(&mut self, builder: &mut FunctionBuilder, state_ptr: Value) {
        for reg in 0..self.values.len() as u8 {
            let idx = usize::from(reg);
            if !self.dirty[idx] {
                continue;
            }

            if let Some(value) = self.values[idx] {
                let offset = REG_FILE_BASE_OFFSET + i32::from(reg) * REG_FILE_STRIDE;
                builder
                    .ins()
                    .store(self.mem_flag(), value, state_ptr, offset);
            }

            self.dirty[idx] = false;
        }

        if self.sp_dirty {
            if let Some(value) = self.sp_value {
                builder
                    .ins()
                    .store(self.mem_flag(), value, state_ptr, SP_OFFSET);
            }
            self.sp_dirty = false;
        }

        for reg in 0..self.vec_values.len() as u8 {
            let idx = usize::from(reg);
            if !self.vec_dirty[idx] {
                continue;
            }

            if let Some(value) = self.vec_values[idx] {
                let offset = CPUSTATE_VREG_BASE_OFFSET + i32::from(reg) * CPUSTATE_VREG_STRIDE;
                builder
                    .ins()
                    .store(self.mem_flag(), value, state_ptr, offset);
            }

            self.vec_dirty[idx] = false;
        }
    }
}

fn read_gp_or_sp(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    reg_cache: &mut JitRegCache,
    reg: u8,
) -> Value {
    if reg == 31 {
        reg_cache.read_sp(builder, state_ptr)
    } else {
        reg_cache.read(builder, state_ptr, reg)
    }
}

fn write_gp_or_sp(
    _builder: &mut FunctionBuilder,
    _state_ptr: Value,
    reg_cache: &mut JitRegCache,
    reg: u8,
    value: Value,
) {
    if reg == 31 {
        reg_cache.write_sp(value);
    } else {
        reg_cache.write(reg, value);
    }
}

fn read_w_or_wsp(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    reg_cache: &mut JitRegCache,
    reg: u8,
) -> Value {
    let value64 = read_gp_or_sp(builder, state_ptr, reg_cache, reg);
    builder.ins().ireduce(types::I32, value64)
}

fn write_w_or_wsp(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    reg_cache: &mut JitRegCache,
    reg: u8,
    value32: Value,
) {
    let value64 = builder.ins().uextend(types::I64, value32);
    write_gp_or_sp(builder, state_ptr, reg_cache, reg, value64);
}

trait BitcastLittleEmitter {
    fn emit_bitcast_little(&mut self, value: Value, to_ty: types::Type) -> Value;
}

impl<'a> BitcastLittleEmitter for FunctionBuilder<'a> {
    fn emit_bitcast_little(&mut self, value: Value, to_ty: types::Type) -> Value {
        self.ins().bitcast(
            to_ty,
            MemFlags::new().with_endianness(Endianness::Little),
            value,
        )
    }
}

impl<'a> BitcastLittleEmitter for &mut FunctionBuilder<'a> {
    fn emit_bitcast_little(&mut self, value: Value, to_ty: types::Type) -> Value {
        (**self).ins().bitcast(
            to_ty,
            MemFlags::new().with_endianness(Endianness::Little),
            value,
        )
    }
}

#[inline]
fn bitcast_little(
    builder: &mut impl BitcastLittleEmitter,
    value: Value,
    to_ty: types::Type,
) -> Value {
    builder.emit_bitcast_little(value, to_ty)
}

fn vector_op_type(q: bool, size: u8) -> types::Type {
    match (q, size) {
        (true, 0b00) => types::I8X16,
        (true, 0b01) => types::I16X8,
        (true, 0b10) => types::I32X4,
        (true, 0b11) => types::I64X2,
        (false, 0b00) => types::I8X8,
        (false, 0b01) => types::I16X4,
        (false, 0b10) => types::I32X2,
        // (false, 0b11) => types::I64, // if size:Q == '110' then UNDEFINED; https://developer.arm.com/documentation/ddi0596/2021-03/SIMD-FP-Instructions/ADD--vector---Add--vector--
        _ => unreachable!("vector operation with q={q:?} and size={size:?} is invalid in AArch64"),
    }
}

// ── AdvSIMD three-different (widening) JIT helper ─────────────────────────────

#[derive(Copy, Clone, PartialEq)]
enum SimdWidenJitOp {
    Addl,
    Subl,
    Abdl,
    Mull,
    Mlal,
    Mlsl,
    Addw,
    Subw,
}

fn emit_simd_widen_jit(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    reg_cache: &mut JitRegCache,
    q: bool,
    signed: bool,
    size: u8,
    rd: u8,
    rn: u8,
    rm: u8,
    op: SimdWidenJitOp,
) {
    let (narrow_vec_ty, wide_vec_ty, wide_lane_ty, elem_count) = match size {
        0 => (types::I8X16, types::I16X8, types::I16, 8u8),
        1 => (types::I16X8, types::I32X4, types::I32, 4u8),
        2 => (types::I32X4, types::I64X2, types::I64, 2u8),
        _ => unreachable!(),
    };
    let q_lane: u8 = if q { 1 } else { 0 };

    let vn_val = reg_cache.read_v(builder, state_ptr, rn);
    let vm_val = reg_cache.read_v(builder, state_ptr, rm);

    // Narrow source halves from Vn and Vm (low or high 64 bits based on Q)
    let m64 = builder.ins().extractlane(vm_val, q_lane);
    let m_vec = builder.ins().scalar_to_vector(types::I64X2, m64);
    let src_m = bitcast_little(builder, m_vec, narrow_vec_ty);

    let n64 = builder.ins().extractlane(vn_val, q_lane);
    let n_vec = builder.ins().scalar_to_vector(types::I64X2, n64);
    let src_n = bitcast_little(builder, n_vec, narrow_vec_ty);

    // Wide Vn for ADDW/SUBW
    let vn_wide = bitcast_little(builder, vn_val, wide_vec_ty);

    // Accumulator Vd for MLAL/MLSL
    let vd_val = reg_cache.read_v(builder, state_ptr, rd);
    let vd_wide = bitcast_little(builder, vd_val, wide_vec_ty);

    let zero_wide = builder.ins().iconst(wide_lane_ty, 0);
    let mut result = builder.ins().splat(wide_vec_ty, zero_wide);

    for i in 0..elem_count {
        let m_elem = builder.ins().extractlane(src_m, i);
        let m_wide = if signed {
            builder.ins().sextend(wide_lane_ty, m_elem)
        } else {
            builder.ins().uextend(wide_lane_ty, m_elem)
        };

        let lane_result = match op {
            SimdWidenJitOp::Addw => {
                let n_elem = builder.ins().extractlane(vn_wide, i);
                builder.ins().iadd(n_elem, m_wide)
            }
            SimdWidenJitOp::Subw => {
                let n_elem = builder.ins().extractlane(vn_wide, i);
                builder.ins().isub(n_elem, m_wide)
            }
            _ => {
                let n_elem = builder.ins().extractlane(src_n, i);
                let n_wide = if signed {
                    builder.ins().sextend(wide_lane_ty, n_elem)
                } else {
                    builder.ins().uextend(wide_lane_ty, n_elem)
                };
                match op {
                    SimdWidenJitOp::Addl => builder.ins().iadd(n_wide, m_wide),
                    SimdWidenJitOp::Subl => builder.ins().isub(n_wide, m_wide),
                    SimdWidenJitOp::Abdl => {
                        let diff = builder.ins().isub(n_wide, m_wide);
                        let neg = builder.ins().ineg(diff);
                        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, diff, zero_wide);
                        builder.ins().select(is_neg, neg, diff)
                    }
                    SimdWidenJitOp::Mull => builder.ins().imul(n_wide, m_wide),
                    SimdWidenJitOp::Mlal => {
                        let product = builder.ins().imul(n_wide, m_wide);
                        let acc = builder.ins().extractlane(vd_wide, i);
                        builder.ins().iadd(acc, product)
                    }
                    SimdWidenJitOp::Mlsl => {
                        let product = builder.ins().imul(n_wide, m_wide);
                        let acc = builder.ins().extractlane(vd_wide, i);
                        builder.ins().isub(acc, product)
                    }
                    _ => unreachable!(),
                }
            }
        };

        result = builder.ins().insertlane(result, lane_result, i);
    }

    let result = bitcast_little(builder, result, types::I64X2);
    reg_cache.write_v(rd, result);
}

// ── Pairwise / narrowing shuffle mask helpers ─────────────────────────────────

/// Build the two shuffle masks (even/odd) for pairwise vector operations
/// (UMINP, UMAXP). `elem_bytes` is the element size in bytes (1, 2, or 4).
fn build_pairwise_shuffle_masks(elem_bytes: u8, q: bool) -> ([u8; 16], [u8; 16]) {
    let mut even = [0u8; 16];
    let mut odd = [0u8; 16];
    let elems_per_reg: u8 = if q { 16 / elem_bytes } else { 8 / elem_bytes };

    for out_elem in 0..elems_per_reg {
        let first_idx = out_elem * 2;
        let second_idx = first_idx + 1;

        // 0-15 = bytes from first vector (vn), 16-31 = bytes from second vector (vm)
        let first_byte_base = if first_idx < elems_per_reg {
            first_idx * elem_bytes
        } else {
            16 + (first_idx - elems_per_reg) * elem_bytes
        };
        let second_byte_base = if second_idx < elems_per_reg {
            second_idx * elem_bytes
        } else {
            16 + (second_idx - elems_per_reg) * elem_bytes
        };

        for b in 0..elem_bytes {
            let pos = usize::from(out_elem * elem_bytes + b);
            even[pos] = first_byte_base + b;
            odd[pos] = second_byte_base + b;
        }
    }

    (even, odd)
}

/// Build the shuffle mask that narrows vector elements (extracts the lower half
/// of each lane). `size` encodes the *source* width: 1 → I16→I8, 2 → I32→I16,
/// 3 → I64→I32.
fn build_narrow_shuffle_mask(size: u8) -> [u8; 16] {
    let mut mask = [0u8; 16];
    let (src_bytes, dst_bytes, lane_count): (u8, u8, u8) = match size {
        1 => (2, 1, 8),
        2 => (4, 2, 4),
        3 => (8, 4, 2),
        _ => unreachable!(),
    };
    for lane in 0..lane_count {
        for b in 0..dst_bytes {
            mask[usize::from(lane * dst_bytes + b)] = lane * src_bytes + b;
        }
    }
    mask
}

// ── FP scalar binary operation helper ──────────────────────────────────────────

enum FpBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Maxnm,
    Minnm,
}

/// Emit a scalar FP binary operation (single or double precision) through the register cache.
fn emit_fp_binop(
    builder: &mut FunctionBuilder,
    reg_cache: &mut JitRegCache,
    state_ptr: Value,
    rd: u8,
    rn: u8,
    rm: u8,
    op: FpBinOp,
    double: bool,
) {
    let (fp_ty, int_ty) = if double {
        (types::F64, types::I64)
    } else {
        (types::F32, types::I32)
    };

    let lhs_bits = if double {
        reg_cache.read_v_scalar64(builder, state_ptr, rn)
    } else {
        reg_cache.read_v_scalar32(builder, state_ptr, rn)
    };
    let rhs_bits = if double {
        reg_cache.read_v_scalar64(builder, state_ptr, rm)
    } else {
        reg_cache.read_v_scalar32(builder, state_ptr, rm)
    };

    let lhs = builder.ins().bitcast(fp_ty, MemFlags::new(), lhs_bits);
    let rhs = builder.ins().bitcast(fp_ty, MemFlags::new(), rhs_bits);

    let result = match op {
        FpBinOp::Add => builder.ins().fadd(lhs, rhs),
        FpBinOp::Sub => builder.ins().fsub(lhs, rhs),
        FpBinOp::Mul => builder.ins().fmul(lhs, rhs),
        FpBinOp::Div => builder.ins().fdiv(lhs, rhs),
        FpBinOp::Max => builder.ins().fmax(lhs, rhs),
        FpBinOp::Min => builder.ins().fmin(lhs, rhs),
        FpBinOp::Maxnm => {
            // FMAXNM: if one NaN, return other; if both NaN, return NaN
            let lhs_nan = builder.ins().fcmp(FloatCC::Unordered, lhs, lhs);
            let rhs_nan = builder.ins().fcmp(FloatCC::Unordered, rhs, rhs);
            let normal_max = builder.ins().fmax(lhs, rhs);
            let pick_lhs = builder.ins().select(rhs_nan, lhs, normal_max);
            builder.ins().select(lhs_nan, rhs, pick_lhs)
        }
        FpBinOp::Minnm => {
            let lhs_nan = builder.ins().fcmp(FloatCC::Unordered, lhs, lhs);
            let rhs_nan = builder.ins().fcmp(FloatCC::Unordered, rhs, rhs);
            let normal_min = builder.ins().fmin(lhs, rhs);
            let pick_lhs = builder.ins().select(rhs_nan, lhs, normal_min);
            builder.ins().select(lhs_nan, rhs, pick_lhs)
        }
    };

    let result_bits = bitcast_little(builder, result, int_ty);
    if double {
        reg_cache.write_v_scalar64(builder, rd, result_bits);
    } else {
        reg_cache.write_v_scalar32(builder, rd, result_bits);
    }
}

// ── FP vector three-same JIT helper ────────────────────────────────────────────

fn fp_vec_type(q: bool, sz: u8) -> types::Type {
    match (q, sz) {
        (true, 0) => types::F32X4,
        (false, 0) => types::F32X2,
        (true, 1) => types::F64X2,
        _ => unreachable!("Q=0,sz=1 is UNDEFINED for FP vector"),
    }
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

/// Local tag enum for FP vector by-element operations (replaces decoder::FpVecElemOp).
#[derive(Copy, Clone, PartialEq, Eq)]
enum FpVecElemOp {
    Fmul,
    Fmla,
    Fmls,
    Fmulx,
}

/// Emit an AdvSIMD FP vector three-same operation.
fn emit_fp_vector_three_same(
    builder: &mut FunctionBuilder,
    reg_cache: &mut JitRegCache,
    state_ptr: Value,
    vd: u8,
    vn: u8,
    vm: u8,
    q: bool,
    sz: u8,
    op: FpVecOp,
) {
    // Always use 128-bit SIMD types to avoid 64-bit vector lowering issues.
    let fp_ty = if sz == 0 { types::F32X4 } else { types::F64X2 };

    let vn_raw = reg_cache.read_v(builder, state_ptr, vn);
    let vm_raw = reg_cache.read_v(builder, state_ptr, vm);

    let vn_fp = bitcast_little(builder, vn_raw, fp_ty);
    let vm_fp = bitcast_little(builder, vm_raw, fp_ty);

    let mut result = match op {
        FpVecOp::Fadd => builder.ins().fadd(vn_fp, vm_fp),
        FpVecOp::Fsub => builder.ins().fsub(vn_fp, vm_fp),
        FpVecOp::Fmul => builder.ins().fmul(vn_fp, vm_fp),
        FpVecOp::Fdiv => builder.ins().fdiv(vn_fp, vm_fp),
        FpVecOp::Fmax => builder.ins().fmax(vn_fp, vm_fp),
        FpVecOp::Fmin => builder.ins().fmin(vn_fp, vm_fp),
        FpVecOp::Fmla => {
            // FMLA: Vd + Vn * Vm
            let vd_raw = reg_cache.read_v(builder, state_ptr, vd);
            let vd_fp = bitcast_little(builder, vd_raw, fp_ty);
            builder.ins().fma(vn_fp, vm_fp, vd_fp)
        }
        FpVecOp::Fmls => {
            // FMLS: Vd - Vn * Vm = Vd + (-Vn) * Vm
            let vd_raw = reg_cache.read_v(builder, state_ptr, vd);
            let vd_fp = bitcast_little(builder, vd_raw, fp_ty);
            let neg_vn = builder.ins().fneg(vn_fp);
            builder.ins().fma(neg_vn, vm_fp, vd_fp)
        }
        FpVecOp::Fmaxnm | FpVecOp::Fminnm => {
            // Lane-wise MAXNM/MINNM: if one NaN return the other
            let lhs_nan = builder.ins().fcmp(FloatCC::Unordered, vn_fp, vn_fp);
            let rhs_nan = builder.ins().fcmp(FloatCC::Unordered, vm_fp, vm_fp);
            let normal = if matches!(op, FpVecOp::Fmaxnm) {
                builder.ins().fmax(vn_fp, vm_fp)
            } else {
                builder.ins().fmin(vn_fp, vm_fp)
            };
            let pick_lhs = builder.ins().select(rhs_nan, vn_fp, normal);
            builder.ins().select(lhs_nan, vm_fp, pick_lhs)
        }
        FpVecOp::Fmulx => {
            // FMULX: like FMUL but 0*Inf = 2.0. Use fmul for simple path.
            builder.ins().fmul(vn_fp, vm_fp)
        }
        FpVecOp::Fabd => {
            let diff = builder.ins().fsub(vn_fp, vm_fp);
            builder.ins().fabs(diff)
        }
        FpVecOp::Frecps => {
            // FRECPS: result = 2.0 - Vn * Vm
            let prod = builder.ins().fmul(vn_fp, vm_fp);
            let scalar_ty = if sz == 0 { types::F32 } else { types::F64 };
            let two = builder.ins().f64const(2.0);
            let two = if scalar_ty == types::F32 {
                builder.ins().fdemote(types::F32, two)
            } else {
                two
            };
            let two_vec = builder.ins().splat(fp_ty, two);
            builder.ins().fsub(two_vec, prod)
        }
        FpVecOp::Frsqrts => {
            // FRSQRTS: result = (3.0 - Vn * Vm) / 2.0
            let prod = builder.ins().fmul(vn_fp, vm_fp);
            let scalar_ty = if sz == 0 { types::F32 } else { types::F64 };
            let three = builder.ins().f64const(3.0);
            let three = if scalar_ty == types::F32 {
                builder.ins().fdemote(types::F32, three)
            } else {
                three
            };
            let two = builder.ins().f64const(2.0);
            let two = if scalar_ty == types::F32 {
                builder.ins().fdemote(types::F32, two)
            } else {
                two
            };
            let three_vec = builder.ins().splat(fp_ty, three);
            let two_vec = builder.ins().splat(fp_ty, two);
            let numer = builder.ins().fsub(three_vec, prod);
            builder.ins().fdiv(numer, two_vec)
        }
        FpVecOp::Fcmeq => builder.ins().fcmp(FloatCC::Equal, vn_fp, vm_fp),
        FpVecOp::Fcmge => builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, vn_fp, vm_fp),
        FpVecOp::Fcmgt => builder.ins().fcmp(FloatCC::GreaterThan, vn_fp, vm_fp),
        FpVecOp::Facge => {
            let a = builder.ins().fabs(vn_fp);
            let b = builder.ins().fabs(vm_fp);
            builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b)
        }
        FpVecOp::Facgt => {
            let a = builder.ins().fabs(vn_fp);
            let b = builder.ins().fabs(vm_fp);
            builder.ins().fcmp(FloatCC::GreaterThan, a, b)
        }
        FpVecOp::Faddp => {
            // Pairwise add: use Cranelift fadd_pairwise if available, else lane-level
            emit_fp_pairwise_binop(builder, reg_cache, state_ptr, vn, vm, q, sz, |b, a, c| {
                b.ins().fadd(a, c)
            })
        }
        FpVecOp::Fmaxp => {
            emit_fp_pairwise_binop(builder, reg_cache, state_ptr, vn, vm, q, sz, |b, a, c| {
                b.ins().fmax(a, c)
            })
        }
        FpVecOp::Fminp => {
            emit_fp_pairwise_binop(builder, reg_cache, state_ptr, vn, vm, q, sz, |b, a, c| {
                b.ins().fmin(a, c)
            })
        }
    };

    // Bitcast result to I64X2 for storage
    result = bitcast_little(builder, result, types::I64X2);

    if !q {
        // 64-bit: zero the upper 64 bits
        let lo = builder.ins().extractlane(result, 0u8);
        result = builder.ins().scalar_to_vector(types::I64X2, lo);
    }

    reg_cache.write_v(vd, result);
}

/// Emit an AdvSIMD FP vector × indexed element operation.
fn emit_fp_vector_by_element(
    builder: &mut FunctionBuilder,
    reg_cache: &mut JitRegCache,
    state_ptr: Value,
    vd: u8,
    vn: u8,
    vm: u8,
    index: u8,
    q: bool,
    sz: u8,
    op: FpVecElemOp,
) {
    // Always use 128-bit types to avoid 64-bit vector lowering issues.
    let fp_ty = if sz == 0 { types::F32X4 } else { types::F64X2 };

    let vn_raw = reg_cache.read_v(builder, state_ptr, vn);
    let vm_raw = reg_cache.read_v(builder, state_ptr, vm);

    let vn_fp = bitcast_little(builder, vn_raw, fp_ty);

    // Extract the indexed element from Vm and splat to all lanes
    let vm_full = bitcast_little(builder, vm_raw, fp_ty);
    let elem_scalar = builder.ins().extractlane(vm_full, index);
    let elem_splat = builder.ins().splat(fp_ty, elem_scalar);

    let mut result = match op {
        FpVecElemOp::Fmul | FpVecElemOp::Fmulx => builder.ins().fmul(vn_fp, elem_splat),
        FpVecElemOp::Fmla => {
            let vd_raw = reg_cache.read_v(builder, state_ptr, vd);
            let vd_fp = bitcast_little(builder, vd_raw, fp_ty);
            builder.ins().fma(vn_fp, elem_splat, vd_fp)
        }
        FpVecElemOp::Fmls => {
            let vd_raw = reg_cache.read_v(builder, state_ptr, vd);
            let vd_fp = bitcast_little(builder, vd_raw, fp_ty);
            let neg_vn = builder.ins().fneg(vn_fp);
            builder.ins().fma(neg_vn, elem_splat, vd_fp)
        }
    };

    // Convert back to I64X2
    result = bitcast_little(builder, result, types::I64X2);
    if !q {
        let lo = builder.ins().extractlane(result, 0);
        result = builder.ins().scalar_to_vector(types::I64X2, lo);
    }

    reg_cache.write_v(vd, result);
}

/// Helper for FP pairwise operations (FADDP, FMAXP, FMINP).
fn emit_fp_pairwise_binop(
    builder: &mut FunctionBuilder,
    reg_cache: &mut JitRegCache,
    state_ptr: Value,
    vn: u8,
    vm: u8,
    q: bool,
    sz: u8,
    op: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> Value {
    // Always use 128-bit types to avoid lowering issues
    let fp_ty = if sz == 0 { types::F32X4 } else { types::F64X2 };
    let vn_raw = reg_cache.read_v(builder, state_ptr, vn);
    let vm_raw = reg_cache.read_v(builder, state_ptr, vm);
    let vn_fp = bitcast_little(builder, vn_raw, fp_ty);
    let vm_fp = bitcast_little(builder, vm_raw, fp_ty);

    if sz == 0 {
        // F32 lanes
        if q {
            // 4 lanes: concat 8 f32 values, pair them → 4 results
            let n0 = builder.ins().extractlane(vn_fp, 0);
            let n1 = builder.ins().extractlane(vn_fp, 1);
            let n2 = builder.ins().extractlane(vn_fp, 2);
            let n3 = builder.ins().extractlane(vn_fp, 3);
            let m0 = builder.ins().extractlane(vm_fp, 0);
            let m1 = builder.ins().extractlane(vm_fp, 1);
            let m2 = builder.ins().extractlane(vm_fp, 2);
            let m3 = builder.ins().extractlane(vm_fp, 3);

            let r0 = op(builder, n0, n1);
            let r1 = op(builder, n2, n3);
            let r2 = op(builder, m0, m1);
            let r3 = op(builder, m2, m3);

            let mut result = builder.ins().splat(types::F32X4, r0);
            result = builder.ins().insertlane(result, r1, 1);
            result = builder.ins().insertlane(result, r2, 2);
            result = builder.ins().insertlane(result, r3, 3);
            result
        } else {
            // 2 lanes: result[0] = op(Vn[0], Vn[1]), result[1] = op(Vm[0], Vm[1])
            // Use F32X4 (128-bit) to avoid 64-bit vector issues; upper lanes will be zeroed.
            let n0 = builder.ins().extractlane(vn_fp, 0);
            let n1 = builder.ins().extractlane(vn_fp, 1);
            let m0 = builder.ins().extractlane(vm_fp, 0);
            let m1 = builder.ins().extractlane(vm_fp, 1);

            let r0 = op(builder, n0, n1);
            let r1 = op(builder, m0, m1);

            let mut result = builder.ins().splat(types::F32X4, r0);
            result = builder.ins().insertlane(result, r1, 1);
            result
        }
    } else {
        // F64 lanes, Q=1 always, 2 lanes
        // result[0] = op(Vn[0], Vn[1]), result[1] = op(Vm[0], Vm[1])
        let n0 = builder.ins().extractlane(vn_fp, 0);
        let n1 = builder.ins().extractlane(vn_fp, 1);
        let m0 = builder.ins().extractlane(vm_fp, 0);
        let m1 = builder.ins().extractlane(vm_fp, 1);

        let r0 = op(builder, n0, n1);
        let r1 = op(builder, m0, m1);

        let mut result = builder.ins().splat(types::F64X2, r0);
        result = builder.ins().insertlane(result, r1, 1);
        result
    }
}

// ── FP vector unary JIT helper ─────────────────────────────────────────────────

/// Local tag enum for FP vector two-register miscellaneous (unary) operations.
#[derive(Copy, Clone, PartialEq, Eq)]
enum FpVecUnaryOp {
    Fabs,
    Fneg,
    Fsqrt,
    Frecpe,
    Frsqrte,
    Frintn,
    Frintm,
    Frintp,
    Frintz,
    Frinta,
    Frintx,
    Frinti,
    Frint32x,
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
    Scvtf,
    Ucvtf,
    Fcmeq0,
    Fcmge0,
    Fcmgt0,
    Fcmle0,
    Fcmlt0,
}

/// Create a vectorised FP constant of the appropriate precision.
fn make_fp_splat_const(
    builder: &mut FunctionBuilder,
    sz: u8,
    fp_ty: types::Type,
    val: f64,
) -> Value {
    let scalar = if sz == 0 {
        let c = builder.ins().f64const(val);
        builder.ins().fdemote(types::F32, c)
    } else {
        builder.ins().f64const(val)
    };
    builder.ins().splat(fp_ty, scalar)
}

/// Per-lane FP→int conversion (signed or unsigned) with saturation.
fn emit_fp_to_int_per_lane(
    builder: &mut FunctionBuilder,
    src_fp: Value,
    int_vec_ty: types::Type,
    scalar_int: types::Type,
    lane_count: u8,
    signed: bool,
) -> Value {
    let zero = builder.ins().iconst(scalar_int, 0);
    let mut acc = builder.ins().splat(int_vec_ty, zero);
    for i in 0..lane_count {
        let lane_fp = builder.ins().extractlane(src_fp, i);
        let lane_int = if signed {
            builder.ins().fcvt_to_sint_sat(scalar_int, lane_fp)
        } else {
            builder.ins().fcvt_to_uint_sat(scalar_int, lane_fp)
        };
        acc = builder.ins().insertlane(acc, lane_int, i);
    }
    acc
}

/// Per-lane int→FP conversion (signed or unsigned).
fn emit_int_to_fp_per_lane(
    builder: &mut FunctionBuilder,
    src_int: Value,
    fp_ty: types::Type,
    scalar_fp: types::Type,
    lane_count: u8,
    signed: bool,
) -> Value {
    let first_int = builder.ins().extractlane(src_int, 0u8);
    let first_fp = if signed {
        builder.ins().fcvt_from_sint(scalar_fp, first_int)
    } else {
        builder.ins().fcvt_from_uint(scalar_fp, first_int)
    };
    let mut acc = builder.ins().splat(fp_ty, first_fp);
    for i in 1..lane_count {
        let int_lane = builder.ins().extractlane(src_int, i);
        let fp_lane = if signed {
            builder.ins().fcvt_from_sint(scalar_fp, int_lane)
        } else {
            builder.ins().fcvt_from_uint(scalar_fp, int_lane)
        };
        acc = builder.ins().insertlane(acc, fp_lane, i);
    }
    acc
}

/// Emit an AdvSIMD FP vector unary operation.
fn emit_fp_vector_unary(
    builder: &mut FunctionBuilder,
    reg_cache: &mut JitRegCache,
    state_ptr: Value,
    vd: u8,
    vn: u8,
    q: bool,
    sz: u8,
    op: FpVecUnaryOp,
) {
    // Always use 128-bit SIMD types for computation to avoid 64-bit vector
    // type issues with Cranelift's bitcast/scalar_to_vector lowering.
    let fp_ty_128 = if sz == 0 { types::F32X4 } else { types::F64X2 };
    let scalar_fp = if sz == 0 { types::F32 } else { types::F64 };
    let scalar_int = if sz == 0 { types::I32 } else { types::I64 };
    let int_vec_ty_128 = if sz == 0 { types::I32X4 } else { types::I64X2 };
    let lane_count: u8 = match (q, sz) {
        (true, 0) => 4,
        (false, 0) => 2,
        (true, 1) | (false, 1) => 2,
        _ => unreachable!(),
    };

    let vn_raw = reg_cache.read_v(builder, state_ptr, vn);
    let vn_fp = bitcast_little(builder, vn_raw, fp_ty_128);

    let mut result = match op {
        // ── Simple FP unary ──
        FpVecUnaryOp::Fabs => builder.ins().fabs(vn_fp),
        FpVecUnaryOp::Fneg => builder.ins().fneg(vn_fp),
        FpVecUnaryOp::Fsqrt => builder.ins().sqrt(vn_fp),

        // ── Rounding ──
        FpVecUnaryOp::Frintn
        | FpVecUnaryOp::Frinta
        | FpVecUnaryOp::Frintx
        | FpVecUnaryOp::Frinti => builder.ins().nearest(vn_fp),
        FpVecUnaryOp::Frintm => builder.ins().floor(vn_fp),
        FpVecUnaryOp::Frintp => builder.ins().ceil(vn_fp),
        FpVecUnaryOp::Frintz => builder.ins().trunc(vn_fp),

        // FRINT32X: round to nearest, clamp to i32 range, convert back
        FpVecUnaryOp::Frint32x => {
            let rounded = builder.ins().nearest(vn_fp);
            let first = builder.ins().extractlane(rounded, 0u8);
            let as_int = builder.ins().fcvt_to_sint_sat(types::I32, first);
            let as_fp = builder.ins().fcvt_from_sint(scalar_fp, as_int);
            let mut v = builder.ins().splat(fp_ty_128, as_fp);
            for i in 1..lane_count {
                let lane = builder.ins().extractlane(rounded, i);
                let li = builder.ins().fcvt_to_sint_sat(types::I32, lane);
                let lf = builder.ins().fcvt_from_sint(scalar_fp, li);
                v = builder.ins().insertlane(v, lf, i);
            }
            v
        }

        // ── Reciprocal estimates ──
        FpVecUnaryOp::Frecpe => {
            let one = make_fp_splat_const(builder, sz, fp_ty_128, 1.0);
            builder.ins().fdiv(one, vn_fp)
        }
        FpVecUnaryOp::Frsqrte => {
            let sq = builder.ins().sqrt(vn_fp);
            let one = make_fp_splat_const(builder, sz, fp_ty_128, 1.0);
            builder.ins().fdiv(one, sq)
        }

        // ── Compare to zero ──
        FpVecUnaryOp::Fcmeq0 => {
            let zero = make_fp_splat_const(builder, sz, fp_ty_128, 0.0);
            builder.ins().fcmp(FloatCC::Equal, vn_fp, zero)
        }
        FpVecUnaryOp::Fcmge0 => {
            let zero = make_fp_splat_const(builder, sz, fp_ty_128, 0.0);
            builder.ins().fcmp(FloatCC::GreaterThanOrEqual, vn_fp, zero)
        }
        FpVecUnaryOp::Fcmgt0 => {
            let zero = make_fp_splat_const(builder, sz, fp_ty_128, 0.0);
            builder.ins().fcmp(FloatCC::GreaterThan, vn_fp, zero)
        }
        FpVecUnaryOp::Fcmle0 => {
            let zero = make_fp_splat_const(builder, sz, fp_ty_128, 0.0);
            builder.ins().fcmp(FloatCC::GreaterThanOrEqual, zero, vn_fp)
        }
        FpVecUnaryOp::Fcmlt0 => {
            let zero = make_fp_splat_const(builder, sz, fp_ty_128, 0.0);
            builder.ins().fcmp(FloatCC::GreaterThan, zero, vn_fp)
        }

        // ── FP-to-signed-int conversions ──
        FpVecUnaryOp::Fcvtns | FpVecUnaryOp::Fcvtas => {
            let rounded = builder.ins().nearest(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                true,
            )
        }
        FpVecUnaryOp::Fcvtms => {
            let rounded = builder.ins().floor(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                true,
            )
        }
        FpVecUnaryOp::Fcvtps => {
            let rounded = builder.ins().ceil(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                true,
            )
        }
        FpVecUnaryOp::Fcvtzs => {
            emit_fp_to_int_per_lane(builder, vn_fp, int_vec_ty_128, scalar_int, lane_count, true)
        }

        // ── FP-to-unsigned-int conversions ──
        FpVecUnaryOp::Fcvtnu | FpVecUnaryOp::Fcvtau => {
            let rounded = builder.ins().nearest(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                false,
            )
        }
        FpVecUnaryOp::Fcvtmu => {
            let rounded = builder.ins().floor(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                false,
            )
        }
        FpVecUnaryOp::Fcvtpu => {
            let rounded = builder.ins().ceil(vn_fp);
            emit_fp_to_int_per_lane(
                builder,
                rounded,
                int_vec_ty_128,
                scalar_int,
                lane_count,
                false,
            )
        }
        FpVecUnaryOp::Fcvtzu => emit_fp_to_int_per_lane(
            builder,
            vn_fp,
            int_vec_ty_128,
            scalar_int,
            lane_count,
            false,
        ),

        // ── Int-to-FP conversions ──
        FpVecUnaryOp::Scvtf => {
            let vn_int = bitcast_little(builder, vn_raw, int_vec_ty_128);
            emit_int_to_fp_per_lane(builder, vn_int, fp_ty_128, scalar_fp, lane_count, true)
        }
        FpVecUnaryOp::Ucvtf => {
            let vn_int = bitcast_little(builder, vn_raw, int_vec_ty_128);
            emit_int_to_fp_per_lane(builder, vn_int, fp_ty_128, scalar_fp, lane_count, false)
        }
    };

    // All results are in 128-bit SIMD types. Bitcast to I64X2 for storage.
    result = bitcast_little(builder, result, types::I64X2);

    if !q {
        // 64-bit operation: zero the upper 64 bits
        let lo = builder.ins().extractlane(result, 0u8);
        result = builder.ins().scalar_to_vector(types::I64X2, lo);
    }

    reg_cache.write_v(vd, result);
}

// ── SIMD narrowing three-different JIT helper ──────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq)]
enum SimdNarrowOp {
    Addhn,
    Raddhn,
    Subhn,
    Rsubhn,
}

fn emit_simd_narrow_jit(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    reg_cache: &mut JitRegCache,
    q: bool,
    size: u8,
    rd: u8,
    rn: u8,
    rm: u8,
    op: SimdNarrowOp,
) {
    // size determines the source (wide) element width:
    //   0 → I16 sources → I8 results  (shift 8)
    //   1 → I32 sources → I16 results (shift 16)
    //   2 → I64 sources → I32 results (shift 32)
    let (wide_vec_ty, wide_lane_ty, narrow_bits, lane_count): (types::Type, types::Type, u32, u8) =
        match size {
            0 => (types::I16X8, types::I16, 8, 8),
            1 => (types::I32X4, types::I32, 16, 4),
            2 => (types::I64X2, types::I64, 32, 2),
            _ => unreachable!(),
        };

    let vn_raw = reg_cache.read_v(builder, state_ptr, rn);
    let vm_raw = reg_cache.read_v(builder, state_ptr, rm);
    let vn_wide = bitcast_little(builder, vn_raw, wide_vec_ty);
    let vm_wide = bitcast_little(builder, vm_raw, wide_vec_ty);

    // Per-lane: add/sub, optional rounding, shift right, pack into 64-bit scalar
    let mut packed = builder.ins().iconst(types::I64, 0);
    for i in 0..lane_count {
        let n_lane = builder.ins().extractlane(vn_wide, i);
        let m_lane = builder.ins().extractlane(vm_wide, i);

        let mut wide_res = match op {
            SimdNarrowOp::Addhn | SimdNarrowOp::Raddhn => builder.ins().iadd(n_lane, m_lane),
            SimdNarrowOp::Subhn | SimdNarrowOp::Rsubhn => builder.ins().isub(n_lane, m_lane),
        };

        // Rounding: add (1 << (shift - 1)) before shifting
        if matches!(op, SimdNarrowOp::Raddhn | SimdNarrowOp::Rsubhn) {
            let round_val = 1i64 << (narrow_bits - 1);
            let round_const = builder.ins().iconst(wide_lane_ty, round_val);
            wide_res = builder.ins().iadd(wide_res, round_const);
        }

        // Shift right to get the high half
        let shift_const = builder.ins().iconst(wide_lane_ty, narrow_bits as i64);
        let shifted = builder.ins().ushr(wide_res, shift_const);

        // Extend to I64 and mask to narrow_bits
        let as_i64 = if wide_lane_ty == types::I64 {
            shifted
        } else {
            builder.ins().uextend(types::I64, shifted)
        };
        let mask = (1i64 << narrow_bits) - 1;
        let mask_val = builder.ins().iconst(types::I64, mask);
        let masked = builder.ins().band(as_i64, mask_val);

        // Pack at the correct bit position
        let bit_pos = i as u32 * narrow_bits;
        if bit_pos > 0 {
            let shift_pack = builder.ins().iconst(types::I64, bit_pos as i64);
            let shifted_val = builder.ins().ishl(masked, shift_pack);
            packed = builder.ins().bor(packed, shifted_val);
        } else {
            packed = builder.ins().bor(packed, masked);
        }
    }

    if q {
        // Write to upper 64 bits, preserving lower
        let vd_raw = reg_cache.read_v(builder, state_ptr, rd);
        let vd_lo = builder.ins().extractlane(vd_raw, 0u8);
        let mut result = builder.ins().scalar_to_vector(types::I64X2, vd_lo);
        result = builder.ins().insertlane(result, packed, 1u8);
        reg_cache.write_v(rd, result);
    } else {
        // Write to lower 64 bits, zero upper
        let result = builder.ins().scalar_to_vector(types::I64X2, packed);
        reg_cache.write_v(rd, result);
    }
}

// ── Memory fault check helper ──────────────────────────────────────────────────

/// Check `state.exit_code` after a memory helper call. If non-zero, return to
/// dispatcher with `instr_pc`. Otherwise, continue execution in the returned
/// "success" block.
fn emit_fault_check(builder: &mut FunctionBuilder, state_ptr: Value, instr_pc: u64) {
    let exit_code = builder
        .ins()
        .load(types::I32, MemFlags::new(), state_ptr, EXIT_CODE_OFFSET);
    let has_exit = builder.ins().icmp_imm(IntCC::NotEqual, exit_code, 0);
    let fault_block = builder.create_block();
    let cont_block = builder.create_block();
    builder
        .ins()
        .brif(has_exit, fault_block, &[], cont_block, &[]);
    builder.seal_block(fault_block);
    builder.seal_block(cont_block);

    builder.switch_to_block(fault_block);
    let fault_pc = builder.ins().iconst(types::I64, instr_pc as i64);
    emit_return_to_dispatcher(builder, state_ptr, fault_pc);

    builder.switch_to_block(cont_block);
}

fn emit_memory_barrier_flush(builder: &mut FunctionBuilder, state_ptr: Value) {
    emit_flush_pending_nzcv(builder, state_ptr);
    let _ = with_active_reg_cache(|reg_cache| reg_cache.flush(builder, state_ptr));
}

fn emit_exit_to_dispatcher(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    next_pc: Value,
    exit_code: i64,
) {
    let exit_value = builder.ins().iconst(types::I32, exit_code);
    builder
        .ins()
        .store(MemFlags::new(), exit_value, state_ptr, EXIT_CODE_OFFSET);
    emit_return_to_dispatcher(builder, state_ptr, next_pc);
}

fn emit_memory_slowpath_return(builder: &mut FunctionBuilder, state_ptr: Value, exit_code: i64) {
    emit_memory_barrier_flush(builder, state_ptr);
    let fault_pc = builder
        .ins()
        .iconst(types::I64, current_guest_pc().unwrap_or(0) as i64);
    emit_exit_to_dispatcher(builder, state_ptr, fault_pc, exit_code);
}

fn vreg_offset(reg: u8) -> i32 {
    CPUSTATE_VREG_BASE_OFFSET + i32::from(reg) * CPUSTATE_VREG_STRIDE
}

fn emit_read_vreg128(builder: &mut FunctionBuilder, state_ptr: Value, reg: u8) -> Value {
    builder
        .ins()
        .load(types::I64X2, MemFlags::new(), state_ptr, vreg_offset(reg))
}

#[allow(dead_code)]
fn emit_read_v_scalar_bits32(builder: &mut FunctionBuilder, state_ptr: Value, reg: u8) -> Value {
    let vec = emit_read_vreg128(builder, state_ptr, reg);
    //let lo64 = builder.ins().extractlane(vec, 0);
    //builder.ins().ireduce(types::I32, lo64)
    let vec = bitcast_little(builder, vec, types::I32X4);
    builder.ins().extractlane(vec, 0)
}

fn emit_read_v_scalar_bits64(builder: &mut FunctionBuilder, state_ptr: Value, reg: u8) -> Value {
    let vec = emit_read_vreg128(builder, state_ptr, reg);
    builder.ins().extractlane(vec, 0)
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

fn emit_state_set_exclusive_reservation(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    address: Value,
    size: i64,
) {
    let valid = builder.ins().iconst(types::I8, 1);
    let size = builder.ins().iconst(types::I8, size);
    builder
        .ins()
        .store(MemFlags::new(), valid, state_ptr, EXCLUSIVE_VALID_OFFSET);
    builder
        .ins()
        .store(MemFlags::new(), size, state_ptr, EXCLUSIVE_SIZE_OFFSET);
    builder.ins().store(
        MemFlags::new(),
        address,
        state_ptr,
        EXCLUSIVE_ADDRESS_OFFSET,
    );
}

fn emit_state_clear_exclusive_reservation(builder: &mut FunctionBuilder, state_ptr: Value) {
    let zero8 = builder.ins().iconst(types::I8, 0);
    let zero64 = builder.ins().iconst(types::I64, 0);
    builder
        .ins()
        .store(MemFlags::new(), zero8, state_ptr, EXCLUSIVE_VALID_OFFSET);
    builder
        .ins()
        .store(MemFlags::new(), zero8, state_ptr, EXCLUSIVE_SIZE_OFFSET);
    builder
        .ins()
        .store(MemFlags::new(), zero64, state_ptr, EXCLUSIVE_ADDRESS_OFFSET);
}

fn emit_state_check_exclusive_reservation(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    address: Value,
    size: i64,
) -> Value {
    let valid = builder.ins().load(
        types::I8,
        MemFlags::new(),
        state_ptr,
        EXCLUSIVE_VALID_OFFSET,
    );
    let valid_set = builder.ins().icmp_imm(IntCC::NotEqual, valid, 0);

    let expected_size = builder.ins().iconst(types::I8, size);
    let reservation_size =
        builder
            .ins()
            .load(types::I8, MemFlags::new(), state_ptr, EXCLUSIVE_SIZE_OFFSET);
    let size_matches = builder
        .ins()
        .icmp(IntCC::Equal, reservation_size, expected_size);

    let reservation_address = builder.ins().load(
        types::I64,
        MemFlags::new(),
        state_ptr,
        EXCLUSIVE_ADDRESS_OFFSET,
    );
    let address_matches = builder
        .ins()
        .icmp(IntCC::Equal, reservation_address, address);

    let one = builder.ins().iconst(types::I8, 1);
    let zero = builder.ins().iconst(types::I8, 0);
    let valid_int = builder.ins().select(valid_set, one, zero);
    let size_int = builder.ins().select(size_matches, one, zero);
    let addr_int = builder.ins().select(address_matches, one, zero);
    let valid_and_size = builder.ins().band(valid_int, size_int);
    let all = builder.ins().band(valid_and_size, addr_int);
    builder.ins().icmp_imm(IntCC::NotEqual, all, 0)
}

fn enter_trap_srcloc(builder: &mut FunctionBuilder) {
    if let Some(srcloc) = current_guest_srcloc() {
        builder.set_srcloc(srcloc);
    }
}

fn leave_trap_srcloc(builder: &mut FunctionBuilder) {
    builder.set_srcloc(SourceLoc::default());
}

fn emit_call_mem_read(
    builder: &mut FunctionBuilder,
    helper: HelperRef,
    state_ptr: Value,
    address: Value,
) -> Value {
    enter_trap_srcloc(builder);

    match current_memory_strategy() {
        MemoryStrategy::Fastmem => {
            let Some(width) = fastmem_read_width(helper) else {
                let call = helper.call(builder, &[state_ptr, address]);
                let result = builder.inst_results(call)[0];
                leave_trap_srcloc(builder);
                return result;
            };
            let Some(fastmem_state) = current_fastmem_state() else {
                let call = helper.call(builder, &[state_ptr, address]);
                let result = builder.inst_results(call)[0];
                leave_trap_srcloc(builder);
                return result;
            };

            emit_memory_barrier_flush(builder, state_ptr);

            let pointer_ty = fastmem_state.pointer_ty;
            let access_size = fastmem_access_bytes(width);
            let guest_base = fastmem_state.guest_base;
            let region_size = fastmem_state.size;
            let host_ptr_u64 = fastmem_state.host_ptr_u64;

            let miss_block = builder.create_block();
            let check_base_block = builder.create_block();
            let check_wrap_block = builder.create_block();
            let check_limit_block = builder.create_block();
            let fastmem_block = builder.create_block();
            let done_block = builder.create_block();
            builder.append_block_param(done_block, types::I64);

            let host_ptr_missing = builder.ins().icmp_imm(IntCC::Equal, host_ptr_u64, 0);
            builder
                .ins()
                .brif(host_ptr_missing, miss_block, &[], check_base_block, &[]);
            builder.seal_block(check_base_block);

            builder.switch_to_block(check_base_block);
            let before_base = builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, address, guest_base);
            builder
                .ins()
                .brif(before_base, miss_block, &[], check_wrap_block, &[]);
            builder.seal_block(check_wrap_block);

            builder.switch_to_block(check_wrap_block);
            let offset = builder.ins().isub(address, guest_base);
            let end = builder.ins().iadd_imm(offset, access_size);
            let wrapped = builder.ins().icmp(IntCC::UnsignedLessThan, end, offset);
            builder
                .ins()
                .brif(wrapped, miss_block, &[], check_limit_block, &[]);
            builder.seal_block(check_limit_block);

            builder.switch_to_block(check_limit_block);
            let out_of_range = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, end, region_size);
            builder
                .ins()
                .brif(out_of_range, miss_block, &[], fastmem_block, &[]);
            builder.seal_block(fastmem_block);
            builder.seal_block(miss_block);

            builder.switch_to_block(fastmem_block);
            let host_addr_u64 = builder.ins().iadd(host_ptr_u64, offset);
            let host_addr = if pointer_ty == types::I64 {
                host_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, host_addr_u64)
            };
            let loaded = match width {
                FastmemAccessWidth::U8 => {
                    let raw = builder.ins().load(types::I8, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U16 => {
                    let raw = builder
                        .ins()
                        .load(types::I16, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U32 => {
                    let raw = builder
                        .ins()
                        .load(types::I32, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U64 => {
                    builder
                        .ins()
                        .load(types::I64, MemFlags::new(), host_addr, 0)
                }
            };
            builder.ins().jump(done_block, &[BlockArg::from(loaded)]);

            builder.switch_to_block(miss_block);
            emit_memory_slowpath_return(builder, state_ptr, EXIT_MEMORY_FAULT);

            builder.seal_block(done_block);
            builder.switch_to_block(done_block);
            let result = builder.block_params(done_block)[0];
            leave_trap_srcloc(builder);
            result
        }
        MemoryStrategy::PteChecked => {
            let Some(width) = fastmem_read_width(helper) else {
                let call = helper.call(builder, &[state_ptr, address]);
                let result = builder.inst_results(call)[0];
                leave_trap_srcloc(builder);
                return result;
            };
            let Some(pte_state) = current_pte_state() else {
                let call = helper.call(builder, &[state_ptr, address]);
                let result = builder.inst_results(call)[0];
                leave_trap_srcloc(builder);
                return result;
            };

            let pointer_ty = pte_state.pointer_ty;
            let miss_block = builder.create_block();
            let check_base_block = builder.create_block();
            let check_limit_block = builder.create_block();
            let check_perm_block = builder.create_block();
            let fastmem_block = builder.create_block();
            let done_block = builder.create_block();
            builder.append_block_param(done_block, types::I64);

            let pte_missing = builder
                .ins()
                .icmp_imm(IntCC::Equal, pte_state.pte_ptr_u64, 0);
            builder
                .ins()
                .brif(pte_missing, miss_block, &[], check_base_block, &[]);
            builder.seal_block(check_base_block);

            builder.switch_to_block(check_base_block);
            let before_base =
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, address, pte_state.guest_base);
            builder
                .ins()
                .brif(before_base, miss_block, &[], check_limit_block, &[]);
            builder.seal_block(check_limit_block);

            builder.switch_to_block(check_limit_block);
            let offset = builder.ins().isub(address, pte_state.guest_base);
            let page_index = builder.ins().ushr_imm(offset, GUEST_PAGE_BITS as i64);
            let out_of_range = builder.ins().icmp(
                IntCC::UnsignedGreaterThanOrEqual,
                page_index,
                pte_state.page_count,
            );
            builder
                .ins()
                .brif(out_of_range, miss_block, &[], check_perm_block, &[]);
            builder.seal_block(check_perm_block);

            builder.switch_to_block(check_perm_block);
            let pte_offset = builder.ins().ishl_imm(page_index, 3);
            let pte_addr_u64 = builder.ins().iadd(pte_state.pte_ptr_u64, pte_offset);
            let pte_addr = if pointer_ty == types::I64 {
                pte_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, pte_addr_u64)
            };
            let pte = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), pte_addr, 0);
            let required = builder
                .ins()
                .iconst(types::I64, (PTE_FLAG_VALID | PTE_FLAG_READ) as i64);
            let access_bits = builder
                .ins()
                .band_imm(pte, (PTE_FLAG_VALID | PTE_FLAG_READ) as i64);
            let special = builder.ins().band_imm(pte, PTE_FLAG_SPECIAL as i64);
            let missing_flags = builder.ins().icmp(IntCC::NotEqual, access_bits, required);
            let has_special = builder.ins().icmp_imm(IntCC::NotEqual, special, 0);
            let slow = builder.ins().bor(missing_flags, has_special);
            builder
                .ins()
                .brif(slow, miss_block, &[], fastmem_block, &[]);
            builder.seal_block(fastmem_block);
            builder.seal_block(miss_block);

            builder.switch_to_block(fastmem_block);
            let host_page = builder.ins().band_imm(pte, !(GUEST_PAGE_MASK as i64));
            let host_addr_u64 = builder.ins().iadd(host_page, address);
            let host_addr = if pointer_ty == types::I64 {
                host_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, host_addr_u64)
            };
            let loaded = match width {
                FastmemAccessWidth::U8 => {
                    let raw = builder.ins().load(types::I8, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U16 => {
                    let raw = builder
                        .ins()
                        .load(types::I16, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U32 => {
                    let raw = builder
                        .ins()
                        .load(types::I32, MemFlags::new(), host_addr, 0);
                    builder.ins().uextend(types::I64, raw)
                }
                FastmemAccessWidth::U64 => {
                    builder
                        .ins()
                        .load(types::I64, MemFlags::new(), host_addr, 0)
                }
            };
            builder.ins().jump(done_block, &[BlockArg::from(loaded)]);

            builder.switch_to_block(miss_block);
            emit_memory_slowpath_return(builder, state_ptr, EXIT_MEMORY_FAULT);

            builder.seal_block(done_block);
            builder.switch_to_block(done_block);
            let result = builder.block_params(done_block)[0];
            leave_trap_srcloc(builder);
            result
        }
        _ => {
            let call = helper.call(builder, &[state_ptr, address]);
            let result = builder.inst_results(call)[0];
            leave_trap_srcloc(builder);
            result
        }
    }
}

fn emit_call_mem_write(
    builder: &mut FunctionBuilder,
    helper: HelperRef,
    state_ptr: Value,
    address: Value,
    value: Value,
) {
    enter_trap_srcloc(builder);

    match current_memory_strategy() {
        MemoryStrategy::Fastmem => {
            let Some(width) = fastmem_write_width(helper) else {
                let _ = helper.call(builder, &[state_ptr, address, value]);
                leave_trap_srcloc(builder);
                return;
            };
            let Some(fastmem_state) = current_fastmem_state() else {
                let _ = helper.call(builder, &[state_ptr, address, value]);
                leave_trap_srcloc(builder);
                return;
            };

            emit_memory_barrier_flush(builder, state_ptr);

            let pointer_ty = fastmem_state.pointer_ty;
            let access_size = fastmem_access_bytes(width);
            let guest_base = fastmem_state.guest_base;
            let region_size = fastmem_state.size;
            let host_ptr_u64 = fastmem_state.host_ptr_u64;

            let miss_block = builder.create_block();
            let check_base_block = builder.create_block();
            let check_wrap_block = builder.create_block();
            let check_limit_block = builder.create_block();
            let fastmem_block = builder.create_block();
            let done_block = builder.create_block();

            let host_ptr_missing = builder.ins().icmp_imm(IntCC::Equal, host_ptr_u64, 0);
            builder
                .ins()
                .brif(host_ptr_missing, miss_block, &[], check_base_block, &[]);
            builder.seal_block(check_base_block);

            builder.switch_to_block(check_base_block);
            let before_base = builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, address, guest_base);
            builder
                .ins()
                .brif(before_base, miss_block, &[], check_wrap_block, &[]);
            builder.seal_block(check_wrap_block);

            builder.switch_to_block(check_wrap_block);
            let offset = builder.ins().isub(address, guest_base);
            let end = builder.ins().iadd_imm(offset, access_size);
            let wrapped = builder.ins().icmp(IntCC::UnsignedLessThan, end, offset);
            builder
                .ins()
                .brif(wrapped, miss_block, &[], check_limit_block, &[]);
            builder.seal_block(check_limit_block);

            builder.switch_to_block(check_limit_block);
            let out_of_range = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, end, region_size);
            builder
                .ins()
                .brif(out_of_range, miss_block, &[], fastmem_block, &[]);
            builder.seal_block(fastmem_block);
            builder.seal_block(miss_block);

            builder.switch_to_block(fastmem_block);
            let host_addr_u64 = builder.ins().iadd(host_ptr_u64, offset);
            let host_addr = if pointer_ty == types::I64 {
                host_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, host_addr_u64)
            };
            match width {
                FastmemAccessWidth::U8 => {
                    let value8 = builder.ins().ireduce(types::I8, value);
                    builder.ins().store(MemFlags::new(), value8, host_addr, 0);
                }
                FastmemAccessWidth::U16 => {
                    let value16 = builder.ins().ireduce(types::I16, value);
                    builder.ins().store(MemFlags::new(), value16, host_addr, 0);
                }
                FastmemAccessWidth::U32 => {
                    let value32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::new(), value32, host_addr, 0);
                }
                FastmemAccessWidth::U64 => {
                    builder.ins().store(MemFlags::new(), value, host_addr, 0);
                }
            }
            builder.ins().jump(done_block, &[]);

            builder.switch_to_block(miss_block);
            emit_memory_slowpath_return(builder, state_ptr, EXIT_MEMORY_FAULT);

            builder.seal_block(done_block);
            builder.switch_to_block(done_block);
            leave_trap_srcloc(builder);
        }
        MemoryStrategy::PteChecked => {
            let Some(width) = fastmem_write_width(helper) else {
                let _ = helper.call(builder, &[state_ptr, address, value]);
                leave_trap_srcloc(builder);
                return;
            };
            let Some(pte_state) = current_pte_state() else {
                let _ = helper.call(builder, &[state_ptr, address, value]);
                leave_trap_srcloc(builder);
                return;
            };

            let pointer_ty = pte_state.pointer_ty;
            let miss_block = builder.create_block();
            let check_base_block = builder.create_block();
            let check_limit_block = builder.create_block();
            let check_perm_block = builder.create_block();
            let fastmem_block = builder.create_block();
            let done_block = builder.create_block();

            let pte_missing = builder
                .ins()
                .icmp_imm(IntCC::Equal, pte_state.pte_ptr_u64, 0);
            builder
                .ins()
                .brif(pte_missing, miss_block, &[], check_base_block, &[]);
            builder.seal_block(check_base_block);

            builder.switch_to_block(check_base_block);
            let before_base =
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, address, pte_state.guest_base);
            builder
                .ins()
                .brif(before_base, miss_block, &[], check_limit_block, &[]);
            builder.seal_block(check_limit_block);

            builder.switch_to_block(check_limit_block);
            let offset = builder.ins().isub(address, pte_state.guest_base);
            let page_index = builder.ins().ushr_imm(offset, GUEST_PAGE_BITS as i64);
            let out_of_range = builder.ins().icmp(
                IntCC::UnsignedGreaterThanOrEqual,
                page_index,
                pte_state.page_count,
            );
            builder
                .ins()
                .brif(out_of_range, miss_block, &[], check_perm_block, &[]);
            builder.seal_block(check_perm_block);

            builder.switch_to_block(check_perm_block);
            let pte_offset = builder.ins().ishl_imm(page_index, 3);
            let pte_addr_u64 = builder.ins().iadd(pte_state.pte_ptr_u64, pte_offset);
            let pte_addr = if pointer_ty == types::I64 {
                pte_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, pte_addr_u64)
            };
            let pte = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), pte_addr, 0);
            let required = builder
                .ins()
                .iconst(types::I64, (PTE_FLAG_VALID | PTE_FLAG_WRITE) as i64);
            let access_bits = builder
                .ins()
                .band_imm(pte, (PTE_FLAG_VALID | PTE_FLAG_WRITE) as i64);
            let special = builder.ins().band_imm(pte, PTE_FLAG_SPECIAL as i64);
            let missing_flags = builder.ins().icmp(IntCC::NotEqual, access_bits, required);
            let has_special = builder.ins().icmp_imm(IntCC::NotEqual, special, 0);
            let slow = builder.ins().bor(missing_flags, has_special);
            builder
                .ins()
                .brif(slow, miss_block, &[], fastmem_block, &[]);
            builder.seal_block(fastmem_block);
            builder.seal_block(miss_block);

            builder.switch_to_block(fastmem_block);
            let host_page = builder.ins().band_imm(pte, !(GUEST_PAGE_MASK as i64));
            let host_addr_u64 = builder.ins().iadd(host_page, address);
            let host_addr = if pointer_ty == types::I64 {
                host_addr_u64
            } else {
                builder.ins().ireduce(pointer_ty, host_addr_u64)
            };
            match width {
                FastmemAccessWidth::U8 => {
                    let value8 = builder.ins().ireduce(types::I8, value);
                    builder.ins().store(MemFlags::new(), value8, host_addr, 0);
                }
                FastmemAccessWidth::U16 => {
                    let value16 = builder.ins().ireduce(types::I16, value);
                    builder.ins().store(MemFlags::new(), value16, host_addr, 0);
                }
                FastmemAccessWidth::U32 => {
                    let value32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::new(), value32, host_addr, 0);
                }
                FastmemAccessWidth::U64 => {
                    builder.ins().store(MemFlags::new(), value, host_addr, 0);
                }
            }
            builder.ins().jump(done_block, &[]);

            builder.switch_to_block(miss_block);
            emit_memory_slowpath_return(builder, state_ptr, EXIT_MEMORY_FAULT);

            builder.seal_block(done_block);
            builder.switch_to_block(done_block);
            leave_trap_srcloc(builder);
        }
        _ => {
            let _ = helper.call(builder, &[state_ptr, address, value]);
            leave_trap_srcloc(builder);
        }
    }
}

fn eager_target_entry_for_pc(
    eager_direct_targets: &EagerDirectTargets,
    target_pc: u64,
) -> Option<usize> {
    eager_direct_targets
        .iter()
        .flatten()
        .find_map(|(expected_pc, target_entry)| {
            (*expected_pc == target_pc).then_some(*target_entry)
        })
}

fn emit_return_to_pc_with_optional_eager(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    target_pc: u64,
    eager_direct_targets: &EagerDirectTargets,
) {
    let next = builder.ins().iconst(types::I64, target_pc as i64);
    let eager_linked =
        eager_target_entry_for_pc(eager_direct_targets, target_pc).is_some_and(|target_entry| {
            emit_return_to_eager_target(builder, state_ptr, next, target_entry)
        });

    if !eager_linked {
        emit_return_to_dispatcher(builder, state_ptr, next);
    }
}

fn emit_return_to_eager_target(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    next_pc: Value,
    target_entry: usize,
) -> bool {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    ))]
    {
        if let (Some(sig_ref), Some(fastmem_state)) = (
            current_return_to_dispatcher_tail_sig(),
            current_fastmem_state(),
        ) {
            emit_flush_pending_nzcv(builder, state_ptr);

            // Keep architectural PC observable while chaining directly to a known target.
            builder
                .ins()
                .store(MemFlags::new(), next_pc, state_ptr, PC_OFFSET);

            let pointer_ty = builder.func.dfg.value_type(state_ptr);
            let target_ptr = builder.ins().iconst(pointer_ty, target_entry as i64);
            builder.ins().return_call_indirect(
                sig_ref,
                target_ptr,
                &[state_ptr, fastmem_state.guest_base],
            );
            return true;
        }
    }

    let _ = builder;
    let _ = state_ptr;
    let _ = next_pc;
    let _ = target_entry;
    false
}

fn emit_return_to_dispatcher(builder: &mut FunctionBuilder, state_ptr: Value, next_pc: Value) {
    emit_flush_pending_nzcv(builder, state_ptr);

    if let (Some(sig_ref), Some(return_slot_addr), Some(fastmem_state), Some(layout)) = (
        current_return_to_dispatcher_tail_sig(),
        current_return_target_slot_addr(),
        current_fastmem_state(),
        current_return_link_layout(),
    ) {
        let pointer_ty = builder.func.dfg.value_type(state_ptr);

        // Keep architectural PC observable for interrupts and non-linked exits.
        builder
            .ins()
            .store(MemFlags::new(), next_pc, state_ptr, PC_OFFSET);

        let return_slot_ptr = builder.ins().iconst(pointer_ty, return_slot_addr as i64);

        let dispatch_block = builder.create_block();
        let mut check_block = builder.create_block();
        builder.ins().jump(check_block, &[]);

        for index in 0..layout.entry_count {
            builder.switch_to_block(check_block);

            let miss_block = if index + 1 < layout.entry_count {
                builder.create_block()
            } else {
                dispatch_block
            };
            let linked_entry_block = builder.create_block();
            let linked_call_block = builder.create_block();

            let entry_base_offset = (index as i32) * RETURN_LINK_ENTRY_STRIDE;
            let expected_next_pc = builder.ins().load(
                types::I64,
                MemFlags::new(),
                return_slot_ptr,
                entry_base_offset + RETURN_LINK_ENTRY_EXPECTED_OFFSET,
            );
            let edge_matches = builder.ins().icmp(IntCC::Equal, next_pc, expected_next_pc);
            builder
                .ins()
                .brif(edge_matches, linked_entry_block, &[], miss_block, &[]);
            builder.seal_block(check_block);

            builder.switch_to_block(linked_entry_block);
            let linked_target = builder.ins().load(
                pointer_ty,
                MemFlags::new(),
                return_slot_ptr,
                entry_base_offset + RETURN_LINK_ENTRY_TARGET_OFFSET,
            );
            let has_linked_target = builder.ins().icmp_imm(IntCC::NotEqual, linked_target, 0);
            builder.ins().brif(
                has_linked_target,
                linked_call_block,
                &[],
                dispatch_block,
                &[],
            );
            builder.seal_block(linked_entry_block);

            builder.switch_to_block(linked_call_block);
            builder.ins().return_call_indirect(
                sig_ref,
                linked_target,
                &[state_ptr, fastmem_state.guest_base],
            );
            builder.seal_block(linked_call_block);

            check_block = miss_block;
        }

        builder.switch_to_block(dispatch_block);
        let source_pc = builder.ins().iconst(types::I64, layout.source_pc as i64);
        let _ = layout
            .record_link_miss
            .call(builder, &[state_ptr, source_pc, next_pc]);
        let dispatch_target = builder.ins().load(
            pointer_ty,
            MemFlags::new(),
            return_slot_ptr,
            layout.dispatch_target_offset,
        );
        builder.ins().return_call_indirect(
            sig_ref,
            dispatch_target,
            &[state_ptr, fastmem_state.guest_base],
        );
        builder.seal_block(dispatch_block);
        return;
    }

    let _ = state_ptr;
    builder.ins().return_(&[next_pc]);
}

fn emit_apply_extend(
    builder: &mut FunctionBuilder,
    value: Value,
    extend: ExtendType,
    amount: u8,
) -> Value {
    let extended = match extend {
        ExtendType::Uxtb => {
            let value8 = builder.ins().ireduce(types::I8, value);
            builder.ins().uextend(types::I64, value8)
        }
        ExtendType::Uxth => {
            let value16 = builder.ins().ireduce(types::I16, value);
            builder.ins().uextend(types::I64, value16)
        }
        ExtendType::Uxtw => {
            let value32 = builder.ins().ireduce(types::I32, value);
            builder.ins().uextend(types::I64, value32)
        }
        ExtendType::Uxtx => value,
        ExtendType::Sxtb => {
            let value8 = builder.ins().ireduce(types::I8, value);
            builder.ins().sextend(types::I64, value8)
        }
        ExtendType::Sxth => {
            let value16 = builder.ins().ireduce(types::I16, value);
            builder.ins().sextend(types::I64, value16)
        }
        ExtendType::Sxtw => {
            let value32 = builder.ins().ireduce(types::I32, value);
            builder.ins().sextend(types::I64, value32)
        }
        ExtendType::Sxtx => value,
    };

    if amount == 0 {
        extended
    } else {
        builder.ins().ishl_imm(extended, i64::from(amount))
    }
}

fn emit_apply_extend32(
    builder: &mut FunctionBuilder,
    value: Value,
    extend: ExtendType,
    amount: u8,
) -> Value {
    let extended = match extend {
        ExtendType::Uxtb => {
            let value8 = builder.ins().ireduce(types::I8, value);
            builder.ins().uextend(types::I32, value8)
        }
        ExtendType::Uxth => {
            let value16 = builder.ins().ireduce(types::I16, value);
            builder.ins().uextend(types::I32, value16)
        }
        ExtendType::Uxtw | ExtendType::Uxtx => builder.ins().ireduce(types::I32, value),
        ExtendType::Sxtb => {
            let value8 = builder.ins().ireduce(types::I8, value);
            builder.ins().sextend(types::I32, value8)
        }
        ExtendType::Sxth => {
            let value16 = builder.ins().ireduce(types::I16, value);
            builder.ins().sextend(types::I32, value16)
        }
        ExtendType::Sxtw | ExtendType::Sxtx => builder.ins().ireduce(types::I32, value),
    };

    if amount == 0 {
        extended
    } else {
        builder.ins().ishl_imm(extended, i64::from(amount))
    }
}

fn emit_apply_shift(
    builder: &mut FunctionBuilder,
    value: Value,
    shift: ShiftType,
    amount: u8,
) -> Value {
    if amount == 0 {
        return value;
    }

    match shift {
        ShiftType::Lsl => builder.ins().ishl_imm(value, i64::from(amount)),
        ShiftType::Lsr => builder.ins().ushr_imm(value, i64::from(amount)),
        ShiftType::Asr => builder.ins().sshr_imm(value, i64::from(amount)),
        ShiftType::Ror => builder.ins().rotr_imm(value, i64::from(amount)),
    }
}

fn emit_apply_shift32(
    builder: &mut FunctionBuilder,
    value: Value,
    shift: ShiftType,
    amount: u8,
) -> Value {
    let value32 = builder.ins().ireduce(types::I32, value);
    if amount == 0 {
        return value32;
    }

    match shift {
        ShiftType::Lsl => builder.ins().ishl_imm(value32, i64::from(amount)),
        ShiftType::Lsr => builder.ins().ushr_imm(value32, i64::from(amount)),
        ShiftType::Asr => builder.ins().sshr_imm(value32, i64::from(amount)),
        ShiftType::Ror => builder.ins().rotr_imm(value32, i64::from(amount)),
    }
}

fn emit_resolve_ls_reg_offset(
    builder: &mut FunctionBuilder,
    raw: Value,
    option: u8,
    shift: bool,
    scale_shift: u8,
) -> Option<Value> {
    let base = match option {
        0b010 => {
            let w = builder.ins().ireduce(types::I32, raw);
            builder.ins().uextend(types::I64, w)
        }
        0b011 => raw,
        0b110 => {
            let w = builder.ins().ireduce(types::I32, raw);
            builder.ins().sextend(types::I64, w)
        }
        0b111 => raw,
        _ => return None,
    };

    Some(if shift {
        builder.ins().ishl_imm(base, i64::from(scale_shift))
    } else {
        base
    })
}

fn emit_pair_address(
    builder: &mut FunctionBuilder,
    base: Value,
    offset: i64,
    mode: PairAddressMode,
) -> (Value, Option<Value>) {
    match mode {
        PairAddressMode::SignedOffset => {
            let address = builder.ins().iadd_imm(base, offset);
            (address, None)
        }
        PairAddressMode::PreIndex => {
            let updated = builder.ins().iadd_imm(base, offset);
            (updated, Some(updated))
        }
        PairAddressMode::PostIndex => {
            let updated = builder.ins().iadd_imm(base, offset);
            (base, Some(updated))
        }
    }
}

fn emit_safe_udiv(builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
    let zero_block = builder.create_block();
    let div_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, types::I64);

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);
    builder
        .ins()
        .brif(rhs_is_zero, zero_block, &[], div_block, &[]);
    builder.seal_block(zero_block);
    builder.seal_block(div_block);

    builder.switch_to_block(zero_block);
    let zero = builder.ins().iconst(types::I64, 0);
    let zero_arg = [BlockArg::from(zero)];
    builder.ins().jump(merge_block, &zero_arg);

    builder.switch_to_block(div_block);
    let result = builder.ins().udiv(lhs, rhs);
    let result_arg = [BlockArg::from(result)];
    builder.ins().jump(merge_block, &result_arg);

    builder.seal_block(merge_block);
    builder.switch_to_block(merge_block);
    builder.block_params(merge_block)[0]
}

fn emit_safe_sdiv(builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
    let zero_block = builder.create_block();
    let nonzero_block = builder.create_block();
    let rhs_neg_one_block = builder.create_block();
    let overflow_block = builder.create_block();
    let div_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, types::I64);

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);
    builder
        .ins()
        .brif(rhs_is_zero, zero_block, &[], nonzero_block, &[]);
    builder.seal_block(zero_block);
    builder.seal_block(nonzero_block);

    builder.switch_to_block(zero_block);
    let zero = builder.ins().iconst(types::I64, 0);
    let zero_arg = [BlockArg::from(zero)];
    builder.ins().jump(merge_block, &zero_arg);

    builder.switch_to_block(nonzero_block);
    let rhs_is_neg_one = builder.ins().icmp_imm(IntCC::Equal, rhs, -1);
    builder
        .ins()
        .brif(rhs_is_neg_one, rhs_neg_one_block, &[], div_block, &[]);
    builder.seal_block(rhs_neg_one_block);

    builder.switch_to_block(rhs_neg_one_block);
    let lhs_is_min = builder.ins().icmp_imm(IntCC::Equal, lhs, i64::MIN);
    builder
        .ins()
        .brif(lhs_is_min, overflow_block, &[], div_block, &[]);

    builder.seal_block(overflow_block);
    builder.seal_block(div_block);

    builder.switch_to_block(overflow_block);
    let min_value = builder.ins().iconst(types::I64, i64::MIN);
    let overflow_arg = [BlockArg::from(min_value)];
    builder.ins().jump(merge_block, &overflow_arg);

    builder.switch_to_block(div_block);
    let result = builder.ins().sdiv(lhs, rhs);
    let result_arg = [BlockArg::from(result)];
    builder.ins().jump(merge_block, &result_arg);

    builder.seal_block(merge_block);
    builder.switch_to_block(merge_block);
    builder.block_params(merge_block)[0]
}

fn emit_safe_udiv32(builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
    let zero_block = builder.create_block();
    let div_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, types::I32);

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);
    builder
        .ins()
        .brif(rhs_is_zero, zero_block, &[], div_block, &[]);
    builder.seal_block(zero_block);
    builder.seal_block(div_block);

    builder.switch_to_block(zero_block);
    let zero = builder.ins().iconst(types::I32, 0);
    let zero_arg = [BlockArg::from(zero)];
    builder.ins().jump(merge_block, &zero_arg);

    builder.switch_to_block(div_block);
    let result = builder.ins().udiv(lhs, rhs);
    let result_arg = [BlockArg::from(result)];
    builder.ins().jump(merge_block, &result_arg);

    builder.seal_block(merge_block);
    builder.switch_to_block(merge_block);
    builder.block_params(merge_block)[0]
}

fn emit_safe_sdiv32(builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
    let zero_block = builder.create_block();
    let nonzero_block = builder.create_block();
    let rhs_neg_one_block = builder.create_block();
    let overflow_block = builder.create_block();
    let div_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, types::I32);

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);
    builder
        .ins()
        .brif(rhs_is_zero, zero_block, &[], nonzero_block, &[]);
    builder.seal_block(zero_block);
    builder.seal_block(nonzero_block);

    builder.switch_to_block(zero_block);
    let zero = builder.ins().iconst(types::I32, 0);
    let zero_arg = [BlockArg::from(zero)];
    builder.ins().jump(merge_block, &zero_arg);

    builder.switch_to_block(nonzero_block);
    let rhs_is_neg_one = builder.ins().icmp_imm(IntCC::Equal, rhs, -1);
    builder
        .ins()
        .brif(rhs_is_neg_one, rhs_neg_one_block, &[], div_block, &[]);
    builder.seal_block(rhs_neg_one_block);

    builder.switch_to_block(rhs_neg_one_block);
    let lhs_is_min = builder
        .ins()
        .icmp_imm(IntCC::Equal, lhs, i64::from(i32::MIN));
    builder
        .ins()
        .brif(lhs_is_min, overflow_block, &[], div_block, &[]);

    builder.seal_block(overflow_block);
    builder.seal_block(div_block);

    builder.switch_to_block(overflow_block);
    let min_value = builder.ins().iconst(types::I32, i64::from(i32::MIN));
    let overflow_arg = [BlockArg::from(min_value)];
    builder.ins().jump(merge_block, &overflow_arg);

    builder.switch_to_block(div_block);
    let result = builder.ins().sdiv(lhs, rhs);
    let result_arg = [BlockArg::from(result)];
    builder.ins().jump(merge_block, &result_arg);

    builder.seal_block(merge_block);
    builder.switch_to_block(merge_block);
    builder.block_params(merge_block)[0]
}

fn emit_conditional_return(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    condition: Value,
    taken_pc: u64,
    fallthrough_pc: u64,
    eager_direct_targets: &EagerDirectTargets,
) {
    let taken_block = builder.create_block();
    let fallthrough_block = builder.create_block();
    builder
        .ins()
        .brif(condition, taken_block, &[], fallthrough_block, &[]);
    builder.seal_block(taken_block);
    builder.seal_block(fallthrough_block);

    builder.switch_to_block(taken_block);
    emit_return_to_pc_with_optional_eager(builder, state_ptr, taken_pc, eager_direct_targets);

    builder.switch_to_block(fallthrough_block);
    emit_return_to_pc_with_optional_eager(builder, state_ptr, fallthrough_pc, eager_direct_targets);
}

fn emit_eval_condition(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    condition: BranchCondition,
) -> Value {
    match condition {
        BranchCondition::Al => return emit_bool_true(builder),
        BranchCondition::Nv => return emit_bool_false(builder),
        _ => {}
    }

    let mask = required_nzcv_mask(condition);
    let (n, z, c, v) = emit_materialize_nzcv_bits(builder, state_ptr, mask);

    match condition {
        BranchCondition::Eq => emit_bool_from_i32(builder, required_nzcv_bit(z, "z")),
        BranchCondition::Ne => {
            let z_set = emit_bool_from_i32(builder, required_nzcv_bit(z, "z"));
            emit_bool_not(builder, z_set)
        }
        BranchCondition::Cs => emit_bool_from_i32(builder, required_nzcv_bit(c, "c")),
        BranchCondition::Cc => {
            let c_set = emit_bool_from_i32(builder, required_nzcv_bit(c, "c"));
            emit_bool_not(builder, c_set)
        }
        BranchCondition::Mi => emit_bool_from_i32(builder, required_nzcv_bit(n, "n")),
        BranchCondition::Pl => {
            let n_set = emit_bool_from_i32(builder, required_nzcv_bit(n, "n"));
            emit_bool_not(builder, n_set)
        }
        BranchCondition::Vs => emit_bool_from_i32(builder, required_nzcv_bit(v, "v")),
        BranchCondition::Vc => {
            let v_set = emit_bool_from_i32(builder, required_nzcv_bit(v, "v"));
            emit_bool_not(builder, v_set)
        }
        BranchCondition::Hi => {
            let c_set = emit_bool_from_i32(builder, required_nzcv_bit(c, "c"));
            let z_set = emit_bool_from_i32(builder, required_nzcv_bit(z, "z"));
            let not_z = emit_bool_not(builder, z_set);
            emit_bool_and(builder, c_set, not_z)
        }
        BranchCondition::Ls => {
            let c_set = emit_bool_from_i32(builder, required_nzcv_bit(c, "c"));
            let z_set = emit_bool_from_i32(builder, required_nzcv_bit(z, "z"));
            let not_z = emit_bool_not(builder, z_set);
            let c_and_not_z = emit_bool_and(builder, c_set, not_z);
            emit_bool_not(builder, c_and_not_z)
        }
        BranchCondition::Ge => {
            let n = required_nzcv_bit(n, "n");
            let v = required_nzcv_bit(v, "v");
            builder.ins().icmp(IntCC::Equal, n, v)
        }
        BranchCondition::Lt => {
            let n = required_nzcv_bit(n, "n");
            let v = required_nzcv_bit(v, "v");
            builder.ins().icmp(IntCC::NotEqual, n, v)
        }
        BranchCondition::Gt => {
            let z_set = emit_bool_from_i32(builder, required_nzcv_bit(z, "z"));
            let not_z = emit_bool_not(builder, z_set);
            let n = required_nzcv_bit(n, "n");
            let v = required_nzcv_bit(v, "v");
            let n_eq_v = builder.ins().icmp(IntCC::Equal, n, v);
            emit_bool_and(builder, not_z, n_eq_v)
        }
        BranchCondition::Le => {
            let z_set = emit_bool_from_i32(builder, required_nzcv_bit(z, "z"));
            let not_z = emit_bool_not(builder, z_set);
            let n = required_nzcv_bit(n, "n");
            let v = required_nzcv_bit(v, "v");
            let n_eq_v = builder.ins().icmp(IntCC::Equal, n, v);
            let not_z_and_n_eq_v = emit_bool_and(builder, not_z, n_eq_v);
            emit_bool_not(builder, not_z_and_n_eq_v)
        }
        BranchCondition::Al | BranchCondition::Nv => unreachable!("handled early"),
    }
}

fn required_nzcv_mask(condition: BranchCondition) -> u8 {
    match condition {
        BranchCondition::Eq | BranchCondition::Ne => NZCV_MASK_Z,
        BranchCondition::Cs | BranchCondition::Cc => NZCV_MASK_C,
        BranchCondition::Mi | BranchCondition::Pl => NZCV_MASK_N,
        BranchCondition::Vs | BranchCondition::Vc => NZCV_MASK_V,
        BranchCondition::Hi | BranchCondition::Ls => NZCV_MASK_C | NZCV_MASK_Z,
        BranchCondition::Ge | BranchCondition::Lt => NZCV_MASK_N | NZCV_MASK_V,
        BranchCondition::Gt | BranchCondition::Le => NZCV_MASK_N | NZCV_MASK_Z | NZCV_MASK_V,
        BranchCondition::Al | BranchCondition::Nv => 0,
    }
}

fn required_nzcv_bit(value: Option<Value>, bit: &str) -> Value {
    value.unwrap_or_else(|| panic!("missing required NZCV bit: {bit}"))
}

fn emit_materialize_nzcv_bits(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    mask: u8,
) -> (Option<Value>, Option<Value>, Option<Value>, Option<Value>) {
    if let Some(pending) = current_pending_nzcv() {
        emit_materialize_pending_nzcv_bits(builder, pending, mask)
    } else {
        emit_materialize_stored_nzcv_bits(builder, state_ptr, mask)
    }
}

fn emit_materialize_stored_nzcv_bits(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    mask: u8,
) -> (Option<Value>, Option<Value>, Option<Value>, Option<Value>) {
    let nzcv = builder
        .ins()
        .load(types::I32, MemFlags::new(), state_ptr, NZCV_OFFSET);
    (
        if (mask & NZCV_MASK_N) != 0 {
            Some(emit_extract_nzcv_bit(builder, nzcv, 31))
        } else {
            None
        },
        if (mask & NZCV_MASK_Z) != 0 {
            Some(emit_extract_nzcv_bit(builder, nzcv, 30))
        } else {
            None
        },
        if (mask & NZCV_MASK_C) != 0 {
            Some(emit_extract_nzcv_bit(builder, nzcv, 29))
        } else {
            None
        },
        if (mask & NZCV_MASK_V) != 0 {
            Some(emit_extract_nzcv_bit(builder, nzcv, 28))
        } else {
            None
        },
    )
}

fn emit_materialize_pending_nzcv_bits(
    builder: &mut FunctionBuilder,
    pending: PendingNzcvRecipe,
    mask: u8,
) -> (Option<Value>, Option<Value>, Option<Value>, Option<Value>) {
    match pending {
        PendingNzcvRecipe::Values { n, z, c, v } => (
            if (mask & NZCV_MASK_N) != 0 {
                Some(n)
            } else {
                None
            },
            if (mask & NZCV_MASK_Z) != 0 {
                Some(z)
            } else {
                None
            },
            if (mask & NZCV_MASK_C) != 0 {
                Some(c)
            } else {
                None
            },
            if (mask & NZCV_MASK_V) != 0 {
                Some(v)
            } else {
                None
            },
        ),
        PendingNzcvRecipe::Logic64 { result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                let n64 = builder.ins().ushr_imm(result, 63);
                Some(builder.ins().ireduce(types::I32, n64))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                Some(builder.ins().iconst(types::I32, 0))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                Some(builder.ins().iconst(types::I32, 0))
            } else {
                None
            };
            (n, z, c, v)
        }
        PendingNzcvRecipe::Logic32 { result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                Some(builder.ins().ushr_imm(result, 31))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                Some(builder.ins().iconst(types::I32, 0))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                Some(builder.ins().iconst(types::I32, 0))
            } else {
                None
            };
            (n, z, c, v)
        }
        PendingNzcvRecipe::Add64 { lhs, rhs, result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                let n64 = builder.ins().ushr_imm(result, 63);
                Some(builder.ins().ireduce(types::I32, n64))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                let c_flag = builder.ins().icmp(IntCC::UnsignedLessThan, result, lhs);
                Some(emit_bool_to_i32(builder, c_flag))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                let lhs_xor_result = builder.ins().bxor(lhs, result);
                let rhs_xor_result = builder.ins().bxor(rhs, result);
                let overflow_mask = builder.ins().band(lhs_xor_result, rhs_xor_result);
                let v64 = builder.ins().ushr_imm(overflow_mask, 63);
                Some(builder.ins().ireduce(types::I32, v64))
            } else {
                None
            };
            (n, z, c, v)
        }
        PendingNzcvRecipe::Add32 { lhs, rhs, result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                Some(builder.ins().ushr_imm(result, 31))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                let c_flag = builder.ins().icmp(IntCC::UnsignedLessThan, result, lhs);
                Some(emit_bool_to_i32(builder, c_flag))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                let lhs_xor_result = builder.ins().bxor(lhs, result);
                let rhs_xor_result = builder.ins().bxor(rhs, result);
                let overflow_mask = builder.ins().band(lhs_xor_result, rhs_xor_result);
                Some(builder.ins().ushr_imm(overflow_mask, 31))
            } else {
                None
            };
            (n, z, c, v)
        }
        PendingNzcvRecipe::Sub64 { lhs, rhs, result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                let n64 = builder.ins().ushr_imm(result, 63);
                Some(builder.ins().ireduce(types::I32, n64))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                let c_flag = builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);
                Some(emit_bool_to_i32(builder, c_flag))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                let lhs_xor_rhs = builder.ins().bxor(lhs, rhs);
                let lhs_xor_result = builder.ins().bxor(lhs, result);
                let overflow_mask = builder.ins().band(lhs_xor_rhs, lhs_xor_result);
                let v64 = builder.ins().ushr_imm(overflow_mask, 63);
                Some(builder.ins().ireduce(types::I32, v64))
            } else {
                None
            };
            (n, z, c, v)
        }
        PendingNzcvRecipe::Sub32 { lhs, rhs, result } => {
            let n = if (mask & NZCV_MASK_N) != 0 {
                Some(builder.ins().ushr_imm(result, 31))
            } else {
                None
            };
            let z = if (mask & NZCV_MASK_Z) != 0 {
                let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
                Some(emit_bool_to_i32(builder, z_flag))
            } else {
                None
            };
            let c = if (mask & NZCV_MASK_C) != 0 {
                let c_flag = builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);
                Some(emit_bool_to_i32(builder, c_flag))
            } else {
                None
            };
            let v = if (mask & NZCV_MASK_V) != 0 {
                let lhs_xor_rhs = builder.ins().bxor(lhs, rhs);
                let lhs_xor_result = builder.ins().bxor(lhs, result);
                let overflow_mask = builder.ins().band(lhs_xor_rhs, lhs_xor_result);
                Some(builder.ins().ushr_imm(overflow_mask, 31))
            } else {
                None
            };
            (n, z, c, v)
        }
    }
}

fn emit_extract_nzcv_bit(builder: &mut FunctionBuilder, nzcv: Value, bit: u8) -> Value {
    let shifted = builder.ins().ushr_imm(nzcv, i64::from(bit));
    builder.ins().band_imm(shifted, 1)
}

fn emit_bool_true(builder: &mut FunctionBuilder) -> Value {
    let one = builder.ins().iconst(types::I32, 1);
    emit_bool_from_i32(builder, one)
}

fn emit_bool_false(builder: &mut FunctionBuilder) -> Value {
    let zero = builder.ins().iconst(types::I32, 0);
    emit_bool_from_i32(builder, zero)
}

fn emit_bool_from_i32(builder: &mut FunctionBuilder, value: Value) -> Value {
    builder.ins().icmp_imm(IntCC::NotEqual, value, 0)
}

fn emit_bool_not(builder: &mut FunctionBuilder, value: Value) -> Value {
    let value_i32 = emit_bool_to_i32(builder, value);
    builder.ins().icmp_imm(IntCC::Equal, value_i32, 0)
}

fn emit_bool_and(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
    let left_i32 = emit_bool_to_i32(builder, left);
    let right_i32 = emit_bool_to_i32(builder, right);
    let value = builder.ins().band(left_i32, right_i32);
    emit_bool_from_i32(builder, value)
}

fn emit_bool_to_i32(builder: &mut FunctionBuilder, value: Value) -> Value {
    let one = builder.ins().iconst(types::I32, 1);
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().select(value, one, zero)
}

fn emit_compute_nzcv_add64(
    builder: &mut FunctionBuilder,
    lhs: Value,
    rhs: Value,
    result: Value,
) -> (Value, Value, Value, Value) {
    let n64 = builder.ins().ushr_imm(result, 63);
    let n = builder.ins().ireduce(types::I32, n64);
    let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
    let z = emit_bool_to_i32(builder, z_flag);
    let c_flag = builder.ins().icmp(IntCC::UnsignedLessThan, result, lhs);
    let c = emit_bool_to_i32(builder, c_flag);

    let lhs_xor_result = builder.ins().bxor(lhs, result);
    let rhs_xor_result = builder.ins().bxor(rhs, result);
    let overflow_mask = builder.ins().band(lhs_xor_result, rhs_xor_result);
    let v64 = builder.ins().ushr_imm(overflow_mask, 63);
    let v = builder.ins().ireduce(types::I32, v64);

    (n, z, c, v)
}

fn emit_compute_nzcv_add32(
    builder: &mut FunctionBuilder,
    lhs: Value,
    rhs: Value,
    result: Value,
) -> (Value, Value, Value, Value) {
    let n = builder.ins().ushr_imm(result, 31);
    let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
    let z = emit_bool_to_i32(builder, z_flag);
    let c_flag = builder.ins().icmp(IntCC::UnsignedLessThan, result, lhs);
    let c = emit_bool_to_i32(builder, c_flag);

    let lhs_xor_result = builder.ins().bxor(lhs, result);
    let rhs_xor_result = builder.ins().bxor(rhs, result);
    let overflow_mask = builder.ins().band(lhs_xor_result, rhs_xor_result);
    let v = builder.ins().ushr_imm(overflow_mask, 31);

    (n, z, c, v)
}

fn emit_compute_nzcv_sub64(
    builder: &mut FunctionBuilder,
    lhs: Value,
    rhs: Value,
    result: Value,
) -> (Value, Value, Value, Value) {
    let n64 = builder.ins().ushr_imm(result, 63);
    let n = builder.ins().ireduce(types::I32, n64);
    let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
    let z = emit_bool_to_i32(builder, z_flag);
    let c_flag = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);
    let c = emit_bool_to_i32(builder, c_flag);

    let lhs_xor_rhs = builder.ins().bxor(lhs, rhs);
    let lhs_xor_result = builder.ins().bxor(lhs, result);
    let overflow_mask = builder.ins().band(lhs_xor_rhs, lhs_xor_result);
    let v64 = builder.ins().ushr_imm(overflow_mask, 63);
    let v = builder.ins().ireduce(types::I32, v64);

    (n, z, c, v)
}

fn emit_compute_nzcv_sub32(
    builder: &mut FunctionBuilder,
    lhs: Value,
    rhs: Value,
    result: Value,
) -> (Value, Value, Value, Value) {
    let n = builder.ins().ushr_imm(result, 31);
    let z_flag = builder.ins().icmp_imm(IntCC::Equal, result, 0);
    let z = emit_bool_to_i32(builder, z_flag);
    let c_flag = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);
    let c = emit_bool_to_i32(builder, c_flag);

    let lhs_xor_rhs = builder.ins().bxor(lhs, rhs);
    let lhs_xor_result = builder.ins().bxor(lhs, result);
    let overflow_mask = builder.ins().band(lhs_xor_rhs, lhs_xor_result);
    let v = builder.ins().ushr_imm(overflow_mask, 31);

    (n, z, c, v)
}

fn emit_nzcv_values_from_immediate(
    builder: &mut FunctionBuilder,
    nzcv: u8,
) -> (Value, Value, Value, Value) {
    let n = builder.ins().iconst(types::I32, i64::from((nzcv >> 3) & 1));
    let z = builder.ins().iconst(types::I32, i64::from((nzcv >> 2) & 1));
    let c = builder.ins().iconst(types::I32, i64::from((nzcv >> 1) & 1));
    let v = builder.ins().iconst(types::I32, i64::from(nzcv & 1));
    (n, z, c, v)
}

fn emit_select_nzcv_values(
    builder: &mut FunctionBuilder,
    cond: Value,
    true_values: (Value, Value, Value, Value),
    false_values: (Value, Value, Value, Value),
) -> (Value, Value, Value, Value) {
    let (true_n, true_z, true_c, true_v) = true_values;
    let (false_n, false_z, false_c, false_v) = false_values;
    (
        builder.ins().select(cond, true_n, false_n),
        builder.ins().select(cond, true_z, false_z),
        builder.ins().select(cond, true_c, false_c),
        builder.ins().select(cond, true_v, false_v),
    )
}

fn emit_append_nzcv_block_params(
    builder: &mut FunctionBuilder,
    block: cranelift_codegen::ir::Block,
) {
    builder.append_block_param(block, types::I32);
    builder.append_block_param(block, types::I32);
    builder.append_block_param(block, types::I32);
    builder.append_block_param(block, types::I32);
}

fn emit_jump_with_nzcv_values(
    builder: &mut FunctionBuilder,
    block: cranelift_codegen::ir::Block,
    values: (Value, Value, Value, Value),
) {
    let (n, z, c, v) = values;
    let args = [
        BlockArg::from(n),
        BlockArg::from(z),
        BlockArg::from(c),
        BlockArg::from(v),
    ];
    builder.ins().jump(block, &args);
}

fn emit_set_pending_nzcv_from_block_params(
    builder: &mut FunctionBuilder,
    block: cranelift_codegen::ir::Block,
) {
    let params = builder.block_params(block);
    set_pending_nzcv(PendingNzcvRecipe::Values {
        n: params[0],
        z: params[1],
        c: params[2],
        v: params[3],
    });
}

fn emit_update_nzcv_logic(builder: &mut FunctionBuilder, state_ptr: Value, result: Value) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Logic64 { result });
}

fn emit_update_nzcv_logic32(builder: &mut FunctionBuilder, state_ptr: Value, result: Value) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Logic32 { result });
}

fn emit_update_nzcv_add(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    lhs: Value,
    rhs: Value,
    result: Value,
) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Add64 { lhs, rhs, result });
}

fn emit_update_nzcv_add32(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    lhs: Value,
    rhs: Value,
    result: Value,
) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Add32 { lhs, rhs, result });
}

fn emit_update_nzcv_sub(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    lhs: Value,
    rhs: Value,
    result: Value,
) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Sub64 { lhs, rhs, result });
}

fn emit_update_nzcv_sub32(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    lhs: Value,
    rhs: Value,
    result: Value,
) {
    let _ = (builder, state_ptr);
    set_pending_nzcv(PendingNzcvRecipe::Sub32 { lhs, rhs, result });
}

fn emit_flush_pending_nzcv(builder: &mut FunctionBuilder, state_ptr: Value) {
    if let Some(pending) = current_pending_nzcv() {
        let (n, z, c, v) = emit_materialize_pending_nzcv_bits(builder, pending, NZCV_MASK_ALL);
        emit_store_nzcv(
            builder,
            state_ptr,
            required_nzcv_bit(n, "n"),
            required_nzcv_bit(z, "z"),
            required_nzcv_bit(c, "c"),
            required_nzcv_bit(v, "v"),
        );
    }
}

fn emit_store_nzcv(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    n: Value,
    z: Value,
    c: Value,
    v: Value,
) {
    let n_shifted = builder.ins().ishl_imm(n, 31);
    let z_shifted = builder.ins().ishl_imm(z, 30);
    let c_shifted = builder.ins().ishl_imm(c, 29);
    let v_shifted = builder.ins().ishl_imm(v, 28);

    let nz = builder.ins().bor(n_shifted, z_shifted);
    let nzc = builder.ins().bor(nz, c_shifted);
    let nzcv = builder.ins().bor(nzc, v_shifted);

    builder
        .ins()
        .store(MemFlags::new(), nzcv, state_ptr, NZCV_OFFSET);
}

fn emit_exit_with_imm(
    builder: &mut FunctionBuilder,
    state_ptr: Value,
    exit_code: i64,
    imm16: u16,
    next_pc: u64,
) {
    let exit = builder.ins().iconst(types::I32, exit_code);
    builder
        .ins()
        .store(MemFlags::new(), exit, state_ptr, EXIT_CODE_OFFSET);

    let imm = builder.ins().iconst(types::I16, i64::from(imm16 as i16));
    builder
        .ins()
        .store(MemFlags::new(), imm, state_ptr, TRAP_IMM_OFFSET);

    let next = builder.ins().iconst(types::I64, next_pc as i64);
    emit_return_to_dispatcher(builder, state_ptr, next);
}

#[cfg(test)]
mod tests {
    use super::{
        parse_return_link_entry_count_override, required_nzcv_mask, try_fuse_mov_wide_sequence,
        BranchCondition, DecodedInstruction, FusedMovWide, NZCV_MASK_C, NZCV_MASK_N, NZCV_MASK_V,
        NZCV_MASK_Z,
    };

    #[test]
    fn parses_return_link_entry_count_override() {
        assert_eq!(parse_return_link_entry_count_override(""), None);
        assert_eq!(parse_return_link_entry_count_override(" auto "), None);
        assert_eq!(parse_return_link_entry_count_override("1"), Some(1));
        assert_eq!(parse_return_link_entry_count_override("4"), Some(4));
        assert_eq!(parse_return_link_entry_count_override("16"), Some(16));
        assert_eq!(parse_return_link_entry_count_override("0"), None);
        assert_eq!(parse_return_link_entry_count_override("17"), None);
        assert_eq!(parse_return_link_entry_count_override("invalid"), None);
    }

    #[test]
    fn condition_masks_only_include_required_nzcv_bits() {
        assert_eq!(required_nzcv_mask(BranchCondition::Eq), NZCV_MASK_Z);
        assert_eq!(required_nzcv_mask(BranchCondition::Ne), NZCV_MASK_Z);
        assert_eq!(required_nzcv_mask(BranchCondition::Cs), NZCV_MASK_C);
        assert_eq!(required_nzcv_mask(BranchCondition::Cc), NZCV_MASK_C);
        assert_eq!(required_nzcv_mask(BranchCondition::Mi), NZCV_MASK_N);
        assert_eq!(required_nzcv_mask(BranchCondition::Pl), NZCV_MASK_N);
        assert_eq!(required_nzcv_mask(BranchCondition::Vs), NZCV_MASK_V);
        assert_eq!(required_nzcv_mask(BranchCondition::Vc), NZCV_MASK_V);
        assert_eq!(
            required_nzcv_mask(BranchCondition::Hi),
            NZCV_MASK_C | NZCV_MASK_Z
        );
        assert_eq!(
            required_nzcv_mask(BranchCondition::Ls),
            NZCV_MASK_C | NZCV_MASK_Z
        );
        assert_eq!(
            required_nzcv_mask(BranchCondition::Ge),
            NZCV_MASK_N | NZCV_MASK_V
        );
        assert_eq!(
            required_nzcv_mask(BranchCondition::Lt),
            NZCV_MASK_N | NZCV_MASK_V
        );
        assert_eq!(
            required_nzcv_mask(BranchCondition::Gt),
            NZCV_MASK_N | NZCV_MASK_Z | NZCV_MASK_V
        );
        assert_eq!(
            required_nzcv_mask(BranchCondition::Le),
            NZCV_MASK_N | NZCV_MASK_Z | NZCV_MASK_V
        );
        assert_eq!(required_nzcv_mask(BranchCondition::Al), 0);
        assert_eq!(required_nzcv_mask(BranchCondition::Nv), 0);
    }

    #[test]
    fn fuses_64bit_mov_wide_chain() {
        let instructions = vec![
            DecodedInstruction::MovZ {
                rd: 5,
                imm16: 0x1234,
                shift: 48,
            },
            DecodedInstruction::MovK {
                rd: 5,
                imm16: 0xabcd,
                shift: 16,
            },
            DecodedInstruction::MovK {
                rd: 5,
                imm16: 0x5678,
                shift: 0,
            },
            DecodedInstruction::MovK {
                rd: 5,
                imm16: 0xffff,
                shift: 32,
            },
        ];

        let fused = try_fuse_mov_wide_sequence(&instructions, 0);
        assert_eq!(
            fused,
            Some(FusedMovWide::U64 {
                rd: 5,
                value: 0x1234_ffff_abcd_5678,
                consumed: 4,
            })
        );
    }

    #[test]
    fn fuses_32bit_mov_wide_chain() {
        let instructions = vec![
            DecodedInstruction::MovN32 {
                rd: 3,
                imm16: 0x1234,
                shift: 16,
            },
            DecodedInstruction::MovK32 {
                rd: 3,
                imm16: 0x0055,
                shift: 0,
            },
        ];

        let fused = try_fuse_mov_wide_sequence(&instructions, 0);
        assert_eq!(
            fused,
            Some(FusedMovWide::U32 {
                rd: 3,
                value: 0xedcb_0055,
                consumed: 2,
            })
        );
    }

    #[test]
    fn does_not_fuse_single_move_wide() {
        let instructions = vec![DecodedInstruction::MovZ {
            rd: 1,
            imm16: 0x0042,
            shift: 0,
        }];

        assert_eq!(try_fuse_mov_wide_sequence(&instructions, 0), None);
    }

    #[test]
    fn does_not_cross_destination_registers() {
        let instructions = vec![
            DecodedInstruction::MovZ {
                rd: 1,
                imm16: 0x1111,
                shift: 0,
            },
            DecodedInstruction::MovK {
                rd: 2,
                imm16: 0x2222,
                shift: 16,
            },
        ];

        assert_eq!(try_fuse_mov_wide_sequence(&instructions, 0), None);
    }
}
