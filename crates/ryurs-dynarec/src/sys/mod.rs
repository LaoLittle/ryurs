cfg_if::cfg_if! {
    if #[cfg(windows)] {
        mod windows;
    } else if #[cfg(target_os = "linux")] {
        mod linux;
    } else if #[cfg(target_os = "macos")] {
        mod macos;
    }
}

pub unsafe fn icache_invalidate(start: *const u8, len: usize) {
    // Safety: Caller must ensure that the specified range is valid.
    unsafe {
        arch::icache_invalidate_impl(start, len);
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    cfg_if::cfg_if! {
        if #[cfg(windows)] {
            use super::windows::icache_invalidate_impl_sys;
        } else if #[cfg(target_os = "macos")] {
            use super::macos::icache_invalidate_impl_sys;
        }
    }

    #[cfg(any(windows, target_os = "macos"))]
    pub unsafe fn icache_invalidate_impl(start: *const u8, len: usize) {
        // Safety: Caller must ensure that the specified range is valid.
        unsafe {
            icache_invalidate_impl_sys(start, len);
        }
    }

    #[cfg(not(any(windows, target_os = "macos")))]
    pub unsafe fn icache_invalidate_impl(start: *const u8, len: usize) {
        unsafe {
            let start = start as usize;
            let end = start + len;
            let cache_line = 64usize; // conservative
            let mut addr = start & !(cache_line - 1);
            while addr < end {
                std::arch::asm!("dc cvau, {}", in(reg) addr);
                addr += cache_line;
            }
            std::arch::asm!("dsb ish");
            addr = start & !(cache_line - 1);
            while addr < end {
                std::arch::asm!("ic ivau, {}", in(reg) addr);
                addr += cache_line;
            }
            std::arch::asm!("dsb ish");
            std::arch::asm!("isb");
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    pub unsafe fn icache_invalidate_impl(start: *const u8, len: usize) {}
}

#[cfg(target_arch = "riscv64")]
mod arch {
    pub unsafe fn icache_invalidate_impl(start: *const u8, len: usize) {}
}
