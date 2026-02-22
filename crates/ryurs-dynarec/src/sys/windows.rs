unsafe extern "system" {
    fn GetCurrentProcess() -> *mut core::ffi::c_void;
    fn FlushInstructionCache(
        process: *mut core::ffi::c_void,
        base_address: *const u8,
        size: usize,
    ) -> i32;
}

pub unsafe fn icache_invalidate_impl_sys(start: *const u8, len: usize) {
    // SAFETY: Caller must ensure that the specified range is valid.
    unsafe {
        FlushInstructionCache(GetCurrentProcess(), start, len);
    }
}
