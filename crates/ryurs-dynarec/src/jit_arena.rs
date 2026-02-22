//! Lock-free JIT memory arena.
//!
//! Reserves a large contiguous virtual address region (default 4 GiB) up front.
//! Memory is committed in 256 MiB *chunks* on demand.  Each chunk is split into
//! a small **data region** (RW, for return-link slots and similar metadata) and
//! a large **code region** (RWX, for compiled machine code).
//!
//! All allocations within a chunk use atomic bump pointers, making them
//! lock-free and safe for concurrent use from multiple compilation threads.
//!
//! Because every chunk lives inside the same contiguous reservation, any two
//! addresses in the arena are guaranteed to be within 4 GiB of each other,
//! which keeps `rel32` PC-relative relocations safe.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Total virtual address space reserved (4 GiB).
const ARENA_RESERVE_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// Size of each committed chunk (256 MiB).
const CHUNK_SIZE: usize = 256 * 1024 * 1024;

/// Data region at the start of each chunk (1 MiB).
const CHUNK_DATA_CAPACITY: usize = 1 * 1024 * 1024;

/// Maximum number of chunks that fit in the reservation.
const MAX_CHUNKS: usize = ARENA_RESERVE_SIZE / CHUNK_SIZE;

const CHUNK_NOT_COMMITTED: u8 = 0;
const CHUNK_COMMITTED: u8 = 1;
const CHUNK_FAILED_COMMIT: u8 = 2;

/// A single 256 MiB chunk within the arena.
///
/// Layout:
/// ```text
/// [  data (1 MiB, RW)  |  code (255 MiB, RWX)  ]
///  ^                     ^
///  base                  base + CHUNK_DATA_CAPACITY
/// ```
struct JitChunk {
    base: *mut u8,
    /// Bump pointer for the data region (offset from `base`).
    data_pos: AtomicUsize,
    /// Bump pointer for the code region (offset from `base`).
    code_pos: AtomicUsize,
    committed: AtomicU8,
}

// The raw pointer is to mmap'd/VirtualAlloc'd memory owned by the arena.
unsafe impl Send for JitChunk {}
unsafe impl Sync for JitChunk {}

impl JitChunk {
    /// Allocate `size` bytes from the data region with the given alignment.
    fn alloc_data(&self, size: usize, align: usize) -> Option<*mut u8> {
        let mut pos = self.data_pos.load(Ordering::Relaxed);
        loop {
            let aligned = (pos + align - 1) & !(align - 1);
            let new_pos = aligned + size;
            if new_pos > CHUNK_DATA_CAPACITY {
                return None;
            }
            match self.data_pos.compare_exchange_weak(
                pos,
                new_pos,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(unsafe { self.base.add(aligned) }),
                Err(new_value) => {
                    pos = new_value;
                    continue;
                }
            }
        }
    }

    /// Allocate `size` bytes from the code region with the given alignment.
    fn alloc_code(&self, size: usize, align: usize) -> Option<*mut u8> {
        let mut pos = self.code_pos.load(Ordering::Relaxed);

        loop {
            let aligned = (pos + align - 1) & !(align - 1);
            let new_pos = aligned + size;
            if new_pos > CHUNK_SIZE {
                return None;
            }
            match self.code_pos.compare_exchange_weak(
                pos,
                new_pos,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(unsafe { self.base.add(aligned) }),
                Err(new_value) => {
                    pos = new_value;
                    continue;
                }
            }
        }
    }
}

/// A lock-free JIT memory arena.
///
/// Thread-safety: all public methods are `&self` and use atomic operations
/// internally.  Multiple threads may allocate concurrently without any mutex.
pub struct JitArena {
    /// Base of the reserved region.
    base: *mut u8,
    /// Committed chunks (indices 0..chunk_count).
    chunks: Box<[JitChunk; MAX_CHUNKS]>,
    /// Number of chunks committed so far (monotonically increasing).
    chunk_count: AtomicUsize,
}

unsafe impl Send for JitArena {}
unsafe impl Sync for JitArena {}

// ── Platform-specific helpers ───────────────────────────────────────────

#[cfg(unix)]
mod platform {
    use std::ptr;

    pub(super) fn reserve(size: usize) -> std::io::Result<*mut u8> {
        let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        // macOS requires MAP_JIT for pages that will later become executable.
        #[cfg(target_os = "macos")]
        {
            flags |= libc::MAP_JIT;
        }
        let base = unsafe { libc::mmap(ptr::null_mut(), size, libc::PROT_NONE, flags, -1, 0) };
        if base == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(base as *mut u8)
    }

    pub(super) fn release(base: *mut u8, size: usize) {
        unsafe {
            libc::munmap(base as *mut libc::c_void, size);
        }
    }

    /// Commit the data region of a chunk as RW.
    pub(super) fn commit_data(base: *mut u8, size: usize) -> bool {
        unsafe {
            libc::mprotect(
                base as *mut libc::c_void,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
            ) == 0
        }
    }

    /// Commit the code region of a chunk as RWX.
    pub(super) fn commit_code(base: *mut u8, size: usize) -> bool {
        unsafe {
            libc::mprotect(
                base as *mut libc::c_void,
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            ) == 0
        }
    }

    /// Switch the current thread to JIT write mode (macOS MAP_JIT W^X).
    ///
    /// After this call, MAP_JIT pages can be written but not executed.
    /// Call [`jit_write_protect`] to switch back to execute mode.
    #[cfg(target_os = "macos")]
    pub(super) fn jit_write_unprotect() {
        unsafe extern "C" {
            fn pthread_jit_write_protect_np(enabled: libc::c_int);
        }
        unsafe {
            pthread_jit_write_protect_np(0);
        }
    }

    /// Switch the current thread back to JIT execute mode (macOS MAP_JIT W^X).
    #[cfg(target_os = "macos")]
    pub(super) fn jit_write_protect() {
        unsafe extern "C" {
            fn pthread_jit_write_protect_np(enabled: libc::c_int);
        }
        unsafe {
            pthread_jit_write_protect_np(1);
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn jit_write_unprotect() {}

    #[cfg(not(target_os = "macos"))]
    pub(super) fn jit_write_protect() {}
}

#[cfg(windows)]
mod platform {
    use std::ptr;

    // Minimal FFI for the Windows APIs we need.
    #[allow(non_snake_case)]
    extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut u8,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut u8;
        fn VirtualFree(lpAddress: *mut u8, dwSize: usize, dwFreeType: u32) -> i32;
        fn VirtualProtect(
            lpAddress: *mut u8,
            dwSize: usize,
            flNewProtect: u32,
            lpflOldProtect: *mut u32,
        ) -> i32;
    }

    const MEM_RESERVE: u32 = 0x00002000;
    const MEM_COMMIT: u32 = 0x00001000;
    const MEM_RELEASE: u32 = 0x00008000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;

    pub(super) fn reserve(size: usize) -> std::io::Result<*mut u8> {
        let base = unsafe { VirtualAlloc(ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
        if base.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(base)
    }

    pub(super) fn release(base: *mut u8, _size: usize) {
        unsafe {
            VirtualFree(base, 0, MEM_RELEASE);
        }
    }

    pub(super) fn commit_data(base: *mut u8, size: usize) -> bool {
        let ptr = unsafe { VirtualAlloc(base, size, MEM_COMMIT, PAGE_READWRITE) };
        !ptr.is_null()
    }

    pub(super) fn commit_code(base: *mut u8, size: usize) -> bool {
        let ptr = unsafe { VirtualAlloc(base, size, MEM_COMMIT, PAGE_EXECUTE_READWRITE) };
        !ptr.is_null()
    }

    pub(super) fn jit_write_unprotect() {}

    pub(super) fn jit_write_protect() {}
}

// ── JitArena implementation ─────────────────────────────────────────────

impl JitArena {
    /// Create a new arena, reserving virtual address space.
    ///
    /// No physical memory is consumed until chunks are committed.
    pub fn new() -> std::io::Result<Self> {
        let base = platform::reserve(ARENA_RESERVE_SIZE)?;

        // Initialize chunk descriptors (all pointing into the reserved region,
        // but none committed yet).
        let chunks = {
            let mut boxed: Box<[std::mem::MaybeUninit<JitChunk>; MAX_CHUNKS]> =
                Box::new(unsafe { std::mem::MaybeUninit::uninit().assume_init() });
            for i in 0..MAX_CHUNKS {
                let chunk_base = unsafe { base.add(i * CHUNK_SIZE) };
                boxed[i].write(JitChunk {
                    base: chunk_base,
                    data_pos: AtomicUsize::new(0),
                    code_pos: AtomicUsize::new(CHUNK_DATA_CAPACITY),
                    committed: AtomicU8::new(CHUNK_NOT_COMMITTED),
                });
            }
            // SAFETY: all elements initialised.
            unsafe { Box::from_raw(Box::into_raw(boxed) as *mut [JitChunk; MAX_CHUNKS]) }
        };

        Ok(Self {
            base,
            chunks,
            chunk_count: AtomicUsize::new(0),
        })
    }

    /// Base address of the entire reserved region.
    pub fn base(&self) -> *const u8 {
        self.base
    }

    /// Allocate `size` bytes in the data region (RW) with the given alignment.
    pub fn alloc_data(&self, size: usize, align: usize) -> Option<*mut u8> {
        let count = self.chunk_count.load(Ordering::Acquire);
        if count > 0 {
            if let Some(ptr) = self.chunks[count - 1].alloc_data(size, align) {
                return Some(ptr);
            }
        }
        self.alloc_new_chunk_then(size, align, true)
    }

    /// Allocate `size` bytes in the code region (RWX) with the given alignment.
    pub fn alloc_code(&self, size: usize, align: usize) -> Option<*mut u8> {
        let count = self.chunk_count.load(Ordering::Acquire);
        if count > 0 {
            if let Some(ptr) = self.chunks[count - 1].alloc_code(size, align) {
                return Some(ptr);
            }
        }
        self.alloc_new_chunk_then(size, align, false)
    }

    /// Switch the current thread to JIT write mode.
    ///
    /// On macOS with MAP_JIT, this calls `pthread_jit_write_protect_np(0)` so
    /// that code pages can be written.  On other platforms this is a no-op.
    ///
    /// Must be paired with [`write_protect`] before executing JIT code.
    pub fn write_unprotect(&self) {
        platform::jit_write_unprotect();
    }

    /// Switch the current thread back to JIT execute mode.
    ///
    /// On macOS with MAP_JIT, this calls `pthread_jit_write_protect_np(1)`.
    /// On other platforms this is a no-op.
    pub fn write_protect(&self) {
        platform::jit_write_protect();
    }

    /// Check whether `addr` is within the arena's reserved region.
    pub fn contains(&self, addr: *const u8) -> bool {
        let base = self.base as usize;
        let a = addr as usize;
        a >= base && a < base + ARENA_RESERVE_SIZE
    }

    /// Returns `true` if `a` and `b` are close enough for a rel32 relocation
    /// (within ±2 GiB).
    pub fn rel32_reachable(a: *const u8, b: *const u8) -> bool {
        let diff = (a as isize).wrapping_sub(b as isize);
        diff >= i32::MIN as isize && diff <= i32::MAX as isize
    }

    // -- private --

    /// Commit a new chunk and allocate from it.
    /// `is_data` selects data vs code region.
    fn alloc_new_chunk_then(&self, size: usize, align: usize, is_data: bool) -> Option<*mut u8> {
        let mut count = self.chunk_count.load(Ordering::Acquire);
        loop {
            if count >= MAX_CHUNKS {
                return None;
            }

            match self.chunk_count.compare_exchange_weak(
                count,
                count + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if !self.commit_chunk(count) {
                        self.chunks[count]
                            .committed
                            .store(CHUNK_FAILED_COMMIT, Ordering::Release);
                        return None;
                    }

                    self.chunks[count]
                        .committed
                        .store(CHUNK_COMMITTED, Ordering::Release);
                    let chunk = &self.chunks[count];
                    return if is_data {
                        chunk.alloc_data(size, align)
                    } else {
                        chunk.alloc_code(size, align)
                    };
                }
                Err(new_count) => {
                    // Another thread committed a chunk — try it.
                    if new_count > count {
                        let chunk = &self.chunks[new_count - 1];

                        loop {
                            let commit_status = chunk.committed.load(Ordering::Acquire);
                            match commit_status {
                                CHUNK_COMMITTED => break,
                                CHUNK_FAILED_COMMIT => return None,
                                CHUNK_NOT_COMMITTED => {
                                    // Still committing — wait and retry.
                                }
                                _ => unreachable!("invalid commit status"),
                            }

                            std::hint::spin_loop(); // Another thread is committing, should be fast, so we busy-wait.
                        }

                        let result = if is_data {
                            chunk.alloc_data(size, align)
                        } else {
                            chunk.alloc_code(size, align)
                        };
                        if result.is_some() {
                            return result;
                        }
                    }

                    count = new_count;
                }
            }
        }
    }

    fn commit_chunk(&self, index: usize) -> bool {
        if index >= MAX_CHUNKS {
            return false;
        }
        let chunk_base = self.chunks[index].base;
        // Data region: RW only.
        if !platform::commit_data(chunk_base, CHUNK_DATA_CAPACITY) {
            return false;
        }
        // Code region: RWX.
        let code_base = unsafe { chunk_base.add(CHUNK_DATA_CAPACITY) };
        let code_size = CHUNK_SIZE - CHUNK_DATA_CAPACITY;
        platform::commit_code(code_base, code_size)
    }
}

impl Drop for JitArena {
    fn drop(&mut self) {
        platform::release(self.base, ARENA_RESERVE_SIZE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_data_and_code() {
        let arena = JitArena::new().expect("failed to create arena");
        let d = arena.alloc_data(64, 8).expect("data alloc failed");
        let c = arena.alloc_code(256, 16).expect("code alloc failed");
        assert!(arena.contains(d));
        assert!(arena.contains(c));
        // data is at the start, code is after data region
        assert!(d < c);
        // Both within rel32 range of each other
        assert!(JitArena::rel32_reachable(d, c));
    }

    #[test]
    fn chunk_overflow_commits_new_chunk() {
        let arena = JitArena::new().expect("failed to create arena");
        // Fill the data region of the first chunk
        let big = arena
            .alloc_data(CHUNK_DATA_CAPACITY, 1)
            .expect("big data alloc failed");
        // Next alloc should trigger a new chunk
        let next = arena.alloc_data(64, 8).expect("overflow data alloc failed");
        assert!(arena.contains(big));
        assert!(arena.contains(next));
        // They should be in different chunks
        let big_chunk = (big as usize - arena.base as usize) / CHUNK_SIZE;
        let next_chunk = (next as usize - arena.base as usize) / CHUNK_SIZE;
        assert_ne!(big_chunk, next_chunk);
    }

    #[test]
    fn data_and_code_in_same_chunk_are_rel32_reachable() {
        let arena = JitArena::new().expect("failed to create arena");
        let d = arena.alloc_data(128, 8).expect("data alloc");
        let c = arena.alloc_code(512, 16).expect("code alloc");
        assert!(JitArena::rel32_reachable(d, c));
        assert!(JitArena::rel32_reachable(c, d));
    }
}
