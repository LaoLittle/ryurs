#[cfg(target_arch = "riscv64")]
mod arch {
    const SYSCALL_RISCV_FLUSH_ICACHE: isize = 259;

    pub unsafe fn icache_invalidate_arch(start: *const u8, len: usize) {
        let mut a0: isize = start as isize;
        let end = a0 + len as isize;
        unsafe {
            std::arch::asm!(
                "ecall",
                inlateout("a0") a0,
                in("a1") end,
                in("a2") 0, // flags (0 = all)
                in("a7") SYSCALL_RISCV_FLUSH_ICACHE,
                options(nostack)
            );
        }
    }
}

#[inline(always)]
pub unsafe fn icache_invalidate_impl_sys(start: *const u8, len: usize) {
    // Safety: Caller must ensure that the specified range is valid.
    #[cfg(target_arch = "riscv64")]
    unsafe {
        icache_invalidate_arch(start, len);
    }
    // do nothing here for other arch, aarch64 implementation is in `super::arch`.
}
