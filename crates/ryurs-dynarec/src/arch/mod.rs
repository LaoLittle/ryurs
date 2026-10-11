use crate::runtime::CpuState;
use crate::translator::HostPcMapEntry;
use cfg_if::cfg_if;

#[repr(C, align(16))]
pub(crate) struct JitDispatchFrame {
    pub(crate) landing_pc: usize,
    pub(crate) landing_sp: usize,
    pub(crate) host_sp: usize,
    pub(crate) state_ptr: *mut CpuState,
    pub(crate) block_pc: u64,
    pub(crate) host_code_start: usize,
    pub(crate) host_code_size: u32,
    pub(crate) host_pc_map_ptr: *const HostPcMapEntry,
    pub(crate) host_pc_map_len: usize,
}

impl JitDispatchFrame {
    pub(crate) fn new(
        state_ptr: *mut CpuState,
        block_pc: u64,
        host_code_start: usize,
        host_code_size: u32,
        host_pc_map_ptr: *const HostPcMapEntry,
        host_pc_map_len: usize,
    ) -> Self {
        Self {
            landing_pc: 0,
            landing_sp: 0,
            host_sp: 0,
            state_ptr,
            block_pc,
            host_code_start,
            host_code_size,
            host_pc_map_ptr,
            host_pc_map_len,
        }
    }
}

cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        mod x86_64;
        pub(crate) use x86_64::dispatcher_landing_target;
        pub(crate) use x86_64::execute_compiled_block_with_dispatcher_landing;
    } else if #[cfg(target_arch = "aarch64")] {
        mod aarch64;
        pub(crate) use aarch64::dispatcher_landing_target;
        pub(crate) use aarch64::execute_compiled_block_with_dispatcher_landing;
    } else if #[cfg(target_arch = "riscv64")] {
        mod riscv64;
        pub(crate) use riscv64::dispatcher_landing_target;
        pub(crate) use riscv64::execute_compiled_block_with_dispatcher_landing;
    } else {
        compile_error!("Unsupported host architecture");
    }
}
