use super::JitDispatchFrame;
use crate::runtime::{CpuState, ExecutionContext};
use crate::translator::BlockEntry;
use core::arch::global_asm;
use std::mem;

const LANDING_PC_OFFSET: usize = mem::offset_of!(JitDispatchFrame, landing_pc);
const LANDING_SP_OFFSET: usize = mem::offset_of!(JitDispatchFrame, landing_sp);
const RETURN_ADDRESS_OFFSET: usize = mem::offset_of!(ExecutionContext, return_address);
const CONTEXT_PC_OFFSET: usize =
    mem::offset_of!(ExecutionContext, context) + mem::offset_of!(CpuState, pc);

global_asm!(
    include_str!("asm/x86_64_sysv64.s"),
    landing_pc_off = const LANDING_PC_OFFSET,
    landing_sp_off = const LANDING_SP_OFFSET,
    return_addr_off = const RETURN_ADDRESS_OFFSET,
    context_pc_off = const CONTEXT_PC_OFFSET,
);

// We use `sysv64` here to avoid annoying windows CC issues.
unsafe extern "sysv64" {
    fn _RNvNtNtC13ryurs_dynarec4arch3asm13execute_block(
        entry: usize,
        execution_context_ptr: *mut ExecutionContext,
        guest_base: u64,
        frame_ptr: *mut JitDispatchFrame,
    ) -> u64;

    fn _RNvNtNtC13ryurs_dynarec4arch3asm14landing_target();
}

pub(crate) fn dispatcher_landing_target() -> usize {
    _RNvNtNtC13ryurs_dynarec4arch3asm14landing_target as *const () as usize
}

pub(crate) unsafe fn execute_compiled_block_with_dispatcher_landing(
    entry: BlockEntry,
    execution_context_ptr: *mut ExecutionContext,
    guest_base: u64,
    frame_ptr: *mut JitDispatchFrame,
) -> u64 {
    // SAFETY: assembly shim follows the declared ABI and returns JIT next-pc.
    unsafe {
        _RNvNtNtC13ryurs_dynarec4arch3asm13execute_block(
            entry as usize,
            execution_context_ptr,
            guest_base,
            frame_ptr,
        )
    }
}
