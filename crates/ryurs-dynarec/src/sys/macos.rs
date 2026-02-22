unsafe extern "C" {
    fn sys_icache_invalidate(start: *const u8, len: usize);
}

pub unsafe fn icache_invalidate_impl_sys(start: *const u8, len: usize) {
    // Safety: Caller must ensure that the specified range is valid.
    unsafe {
        sys_icache_invalidate(start, len);
    }
}
