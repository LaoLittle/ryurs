use anyhow::{Result, anyhow};
use ryurs_dynarec::{FastmemRegion, MemoryInterface};
use std::collections::HashMap;
#[cfg(windows)]
use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

const LOW_PAGE_SIZE: usize = 0x1000;
const LOW_PAGE_MASK: u64 = !(LOW_PAGE_SIZE as u64 - 1);
const NO_FAULT_ADDRESS: u64 = u64::MAX;
const RESERVED_GUEST_BYTES: usize = 8 * 1024 * 1024 * 1024;

const MEMORY_STATE_UNMAPPED: u32 = 0;
const MEMORY_STATE_HEAP: u32 = 0x5;
const MEMORY_PERMISSION_NONE: u32 = 0;
const MEMORY_PERMISSION_READ_WRITE: u32 = 0x3;

static MEMORY_WATCH_PC: AtomicU64 = AtomicU64::new(NO_FAULT_ADDRESS);

pub fn set_memory_watch_pc(pc: u64) {
    MEMORY_WATCH_PC.store(pc, Ordering::Relaxed);
}

fn trace_memory_watch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RYURS_TRACE_MEMORY_WATCH")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy)]
struct MemoryWatchRange {
    start: u64,
    end: u64,
}

fn parse_watch_value(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn trace_memory_watch_ranges() -> &'static [MemoryWatchRange] {
    static RANGES: OnceLock<Vec<MemoryWatchRange>> = OnceLock::new();
    RANGES.get_or_init(|| {
        std::env::var("RYURS_TRACE_MEMORY_RANGE")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|entry| {
                        let (start, end) = entry.split_once(':')?;
                        let start = parse_watch_value(start)?;
                        let end = parse_watch_value(end)?;
                        (start < end).then_some(MemoryWatchRange { start, end })
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

#[derive(Debug, Clone, Copy)]
struct HighMemoryRegion {
    start: u64,
    end: u64,
    state: u32,
    permission: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryQueryInfo {
    pub address: u64,
    pub size: u64,
    pub state: u32,
    pub attribute: u32,
    pub permission: u32,
    pub ipc_ref_count: u32,
    pub device_ref_count: u32,
}

impl MemoryQueryInfo {
    fn mapped(address: u64, size: u64) -> Self {
        Self::mapped_with(
            address,
            size,
            MEMORY_STATE_HEAP,
            MEMORY_PERMISSION_READ_WRITE,
        )
    }

    fn mapped_with(address: u64, size: u64, state: u32, permission: u32) -> Self {
        Self {
            address,
            size,
            state,
            attribute: 0,
            permission,
            ipc_ref_count: 0,
            device_ref_count: 0,
        }
    }

    fn unmapped(address: u64, size: u64) -> Self {
        Self {
            address,
            size,
            state: MEMORY_STATE_UNMAPPED,
            attribute: 0,
            permission: MEMORY_PERMISSION_NONE,
            ipc_ref_count: 0,
            device_ref_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAliasMode {
    SingleMapping,
    DoubleMapping,
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(
        lp_address: *mut c_void,
        dw_size: usize,
        fl_allocation_type: u32,
        fl_protect: u32,
    ) -> *mut c_void;
    fn VirtualFree(lp_address: *mut c_void, dw_size: usize, dw_free_type: u32) -> i32;
    fn VirtualProtect(
        lp_address: *mut c_void,
        dw_size: usize,
        fl_new_protect: u32,
        lpfl_old_protect: *mut u32,
    ) -> i32;
}

#[cfg(windows)]
const MEM_COMMIT: u32 = 0x0000_1000;
#[cfg(windows)]
const MEM_RESERVE: u32 = 0x0000_2000;
#[cfg(windows)]
const MEM_RELEASE: u32 = 0x0000_8000;
#[cfg(windows)]
const PAGE_NOACCESS: u32 = 0x01;
#[cfg(windows)]
const PAGE_READWRITE: u32 = 0x04;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostPageProtection {
    NoAccess,
    ReadWrite,
}

#[cfg(unix)]
fn set_host_page_protection(
    page: *mut u8,
    page_size: usize,
    protection: HostPageProtection,
) -> bool {
    let prot = match protection {
        HostPageProtection::NoAccess => libc::PROT_NONE,
        HostPageProtection::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
    };
    // SAFETY: caller provides a valid page-aligned pointer into a live mapping.
    unsafe { libc::mprotect(page.cast(), page_size, prot) == 0 }
}

#[cfg(windows)]
fn set_host_page_protection(
    page: *mut u8,
    page_size: usize,
    protection: HostPageProtection,
) -> bool {
    let mut previous_protect = 0u32;
    let new_protect = match protection {
        HostPageProtection::NoAccess => PAGE_NOACCESS,
        HostPageProtection::ReadWrite => PAGE_READWRITE,
    };
    // SAFETY: caller provides a valid page pointer and a writable output slot.
    unsafe {
        VirtualProtect(
            page.cast(),
            page_size,
            new_protect,
            &mut previous_protect,
        ) != 0
    }
}

#[cfg(all(not(unix), not(windows)))]
fn set_host_page_protection(
    _page: *mut u8,
    _page_size: usize,
    _protection: HostPageProtection,
) -> bool {
    false
}

#[derive(Debug)]
struct HostReservation {
    reservation_base: *mut u8,
    reservation_size: usize,
    guest_base: *mut u8,
    guest_span_size: usize,
    safepoint_page_size: usize,
}

impl HostReservation {
    fn new(guest_span_size: usize, host_page_size: usize) -> Self {
        reserve_host_span(guest_span_size, host_page_size).unwrap_or_else(|| {
            panic!(
                "failed to reserve guest memory span via host VM APIs (span=0x{guest_span_size:016x})"
            )
        })
    }
}

impl Drop for HostReservation {
    fn drop(&mut self) {
        if self.reservation_base.is_null() || self.reservation_size == 0 {
            return;
        }

        // SAFETY: this object owns the reservation and never releases it elsewhere.
        unsafe {
            release_host_span(self.reservation_base, self.reservation_size);
        }
    }
}

#[cfg(unix)]
fn reserve_host_span(guest_span_size: usize, host_page_size: usize) -> Option<HostReservation> {
    let reservation_size = host_page_size.checked_add(guest_span_size)?;
    // SAFETY: `mmap` is called with a null hint and checked return value.
    let reservation = unsafe {
        libc::mmap(
            ptr::null_mut(),
            reservation_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if reservation == libc::MAP_FAILED {
        return None;
    }

    if !set_host_page_protection(
        reservation.cast::<u8>(),
        host_page_size,
        HostPageProtection::NoAccess,
    ) {
        // SAFETY: reservation came from `mmap` above.
        unsafe {
            let _ = libc::munmap(reservation, reservation_size);
        }
        return None;
    }

    let reservation_base = reservation.cast::<u8>();
    // SAFETY: `host_page_size` is strictly smaller than `reservation_size`.
    let guest_base = unsafe { reservation_base.add(host_page_size) };
    Some(HostReservation {
        reservation_base,
        reservation_size,
        guest_base,
        guest_span_size,
        safepoint_page_size: host_page_size,
    })
}

#[cfg(windows)]
fn reserve_host_span(guest_span_size: usize, host_page_size: usize) -> Option<HostReservation> {
    let reservation_size = host_page_size.checked_add(guest_span_size)?;
    // SAFETY: `VirtualAlloc` return value is checked for null.
    let reservation = unsafe {
        VirtualAlloc(
            ptr::null_mut(),
            reservation_size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    if reservation.is_null() {
        return None;
    }

    if !set_host_page_protection(
        reservation.cast::<u8>(),
        host_page_size,
        HostPageProtection::NoAccess,
    ) {
        // SAFETY: reservation came from `VirtualAlloc` above.
        unsafe {
            let _ = VirtualFree(reservation, 0, MEM_RELEASE);
        }
        return None;
    }

    let reservation_base = reservation.cast::<u8>();
    // SAFETY: `host_page_size` is strictly smaller than `reservation_size`.
    let guest_base = unsafe { reservation_base.add(host_page_size) };
    Some(HostReservation {
        reservation_base,
        reservation_size,
        guest_base,
        guest_span_size,
        safepoint_page_size: host_page_size,
    })
}

#[cfg(all(not(unix), not(windows)))]
fn reserve_host_span(_guest_span_size: usize, _host_page_size: usize) -> Option<HostReservation> {
    None
}

#[cfg(unix)]
unsafe fn release_host_span(reservation_base: *mut u8, reservation_size: usize) {
    // SAFETY: caller guarantees this pair was returned by `mmap` and is still owned.
    let _ = unsafe { libc::munmap(reservation_base.cast::<libc::c_void>(), reservation_size) };
}

#[cfg(windows)]
unsafe fn release_host_span(reservation_base: *mut u8, _reservation_size: usize) {
    // SAFETY: caller guarantees pointer came from `VirtualAlloc` and is still owned.
    let _ = unsafe { VirtualFree(reservation_base.cast::<c_void>(), 0, MEM_RELEASE) };
}

#[cfg(all(not(unix), not(windows)))]
unsafe fn release_host_span(_reservation_base: *mut u8, _reservation_size: usize) {}

#[derive(Debug, Clone)]
struct MemoryMaps {
    high_regions: Vec<HighMemoryRegion>,
    high_sparse_pages: HashMap<u64, Box<[u8; LOW_PAGE_SIZE]>>,
    low_regions: Vec<(u64, u64)>,
    low_pages: HashMap<u64, Box<[u8; LOW_PAGE_SIZE]>>,
}

impl Default for MemoryMaps {
    fn default() -> Self {
        Self {
            high_regions: Vec::new(),
            high_sparse_pages: HashMap::new(),
            low_regions: Vec::new(),
            low_pages: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct GuestMemory {
    base: u64,
    size: usize,
    reservation: HostReservation,
    maps: RwLock<MemoryMaps>,
    last_fault_address: AtomicU64,
    host_page_size: usize,
    alias_mode: MemoryAliasMode,
}

// Guest memory is intentionally shared between worker threads with lock-free raw-pointer access.
// Synchronization and ordering are enforced by the emulated runtime model and host VM protections.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl Clone for GuestMemory {
    fn clone(&self) -> Self {
        let reservation =
            HostReservation::new(self.reservation.guest_span_size, self.host_page_size);

        // SAFETY: both source and destination guest spans are valid for `self.size` bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                self.reservation.guest_base,
                reservation.guest_base,
                self.size,
            );
        }

        let maps = self
            .maps
            .read()
            .expect("guest memory maps lock poisoned")
            .clone();

        Self {
            base: self.base,
            size: self.size,
            reservation,
            maps: RwLock::new(maps),
            last_fault_address: AtomicU64::new(self.last_fault_address.load(Ordering::Relaxed)),
            host_page_size: self.host_page_size,
            alias_mode: self.alias_mode,
        }
    }
}

impl GuestMemory {
    pub fn new(base: u64, size: usize) -> Self {
        let host_page_size = page_size::get();
        let alias_mode = if host_page_size <= LOW_PAGE_SIZE {
            MemoryAliasMode::SingleMapping
        } else {
            MemoryAliasMode::DoubleMapping
        };
        let guest_span_size = RESERVED_GUEST_BYTES.max(size);
        let reservation = HostReservation::new(guest_span_size, host_page_size);

        Self {
            base,
            size,
            reservation,
            maps: RwLock::new(MemoryMaps::default()),
            last_fault_address: AtomicU64::new(NO_FAULT_ADDRESS),
            host_page_size,
            alias_mode,
        }
    }

    pub fn base_address(&self) -> u64 {
        self.base
    }

    pub fn size_bytes(&self) -> u64 {
        self.size as u64
    }

    /// Return a raw byte slice covering the entire guest memory region.
    ///
    /// # Safety
    /// The caller must ensure no concurrent writes are occurring.
    pub unsafe fn guest_memory_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.reservation.guest_base, self.size) }
    }

    pub fn host_page_size(&self) -> usize {
        self.host_page_size
    }

    pub fn alias_mode(&self) -> MemoryAliasMode {
        self.alias_mode
    }

    pub fn clear_high_regions(&self) {
        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        maps.high_regions.clear();
        maps.high_sparse_pages.clear();
    }

    /// Return all sparse pages (both high and low) as `(page_address, &[u8; 4096])` pairs.
    /// Useful for syncing with an external emulator.
    pub fn sparse_pages(&self) -> Vec<(u64, Vec<u8>)> {
        let maps = self.maps.read().expect("guest memory maps lock poisoned");
        let mut pages: Vec<(u64, Vec<u8>)> = Vec::new();
        for (&addr, data) in &maps.high_sparse_pages {
            pages.push((addr, data.to_vec()));
        }
        for (&addr, data) in &maps.low_pages {
            pages.push((addr, data.to_vec()));
        }
        pages
    }

    pub fn clear_low_regions(&self) {
        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        maps.low_regions.clear();
        maps.low_pages.clear();
    }

    pub fn map_low_region(&self, start: u64, size: u64) -> bool {
        if size == 0 {
            return true;
        }

        let Some(end) = start.checked_add(size) else {
            return false;
        };

        if start >= self.base || end > self.base {
            return false;
        }

        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        maps.low_regions.push((start, end));
        true
    }

    pub fn map_high_region(&self, start: u64, size: u64, state: u32, permission: u32) -> bool {
        if size == 0 {
            return true;
        }

        let Some(end) = start.checked_add(size) else {
            return false;
        };

        if start < self.base {
            return false;
        }

        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        maps.high_regions.push(HighMemoryRegion {
            start,
            end,
            state,
            permission,
        });
        maps.high_regions.sort_by_key(|region| region.start);

        let mut merged: Vec<HighMemoryRegion> = Vec::with_capacity(maps.high_regions.len());
        for region in maps.high_regions.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.state == region.state
                    && last.permission == region.permission
                    && region.start <= last.end
                {
                    last.end = last.end.max(region.end);
                    continue;
                }
            }

            merged.push(region);
        }
        maps.high_regions = merged;
        true
    }

    pub fn clear_last_fault(&self) {
        self.last_fault_address
            .store(NO_FAULT_ADDRESS, Ordering::Relaxed);
    }

    pub fn last_fault_address(&self) -> Option<u64> {
        let address = self.last_fault_address.load(Ordering::Relaxed);
        (address != NO_FAULT_ADDRESS).then_some(address)
    }

    fn mark_fault(&self, address: u64) {
        self.last_fault_address.store(address, Ordering::Relaxed);
    }

    fn checked_offset(&self, address: u64) -> Option<usize> {
        let offset = address.checked_sub(self.base)?;
        let offset = usize::try_from(offset).ok()?;
        (offset < self.size).then_some(offset)
    }

    fn low_region_contains(
        low_regions: &[(u64, u64)],
        base: u64,
        address: u64,
        size: usize,
    ) -> bool {
        if address >= base {
            return false;
        }

        let Some(end) = address.checked_add(size as u64) else {
            return false;
        };

        if end > base {
            return false;
        }

        low_regions
            .iter()
            .any(|(start, region_end)| address >= *start && end <= *region_end)
    }

    fn low_page_parts(address: u64) -> (u64, usize) {
        let page = address & LOW_PAGE_MASK;
        let offset = (address as usize) & (LOW_PAGE_SIZE - 1);
        (page, offset)
    }

    fn high_region_contains(high_regions: &[HighMemoryRegion], address: u64, size: usize) -> bool {
        let Some(end) = address.checked_add(size as u64) else {
            return false;
        };

        high_regions
            .iter()
            .any(|region| address >= region.start && end <= region.end)
    }

    fn read_low_bytes<const N: usize>(&self, address: u64) -> Option<[u8; N]> {
        let maps = self.maps.read().expect("guest memory maps lock poisoned");
        if !Self::low_region_contains(&maps.low_regions, self.base, address, N) {
            return None;
        }

        let mut out = [0u8; N];

        for (index, byte) in out.iter_mut().enumerate() {
            let addr = address.wrapping_add(index as u64);
            let (page, offset) = Self::low_page_parts(addr);
            if let Some(data) = maps.low_pages.get(&page) {
                *byte = data[offset];
            }
        }

        Some(out)
    }

    fn write_low_bytes<const N: usize>(&self, address: u64, bytes: [u8; N]) -> bool {
        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        if !Self::low_region_contains(&maps.low_regions, self.base, address, N) {
            return false;
        }

        for (index, value) in bytes.into_iter().enumerate() {
            let addr = address.wrapping_add(index as u64);
            let (page, offset) = Self::low_page_parts(addr);
            let data = maps
                .low_pages
                .entry(page)
                .or_insert_with(|| Box::new([0; LOW_PAGE_SIZE]));
            data[offset] = value;
        }

        true
    }

    fn read_high_bytes<const N: usize>(&self, address: u64) -> Option<[u8; N]> {
        if N != 0 && address.checked_add((N - 1) as u64).is_none() {
            return None;
        }

        let maps = self.maps.read().expect("guest memory maps lock poisoned");
        let mut out = [0u8; N];

        for (index, slot) in out.iter_mut().enumerate() {
            let byte_address = address.wrapping_add(index as u64);
            if let Some(offset) = self.checked_offset(byte_address) {
                // SAFETY: the offset is in-bounds for the reserved guest span.
                unsafe {
                    *slot = *self.reservation.guest_base.add(offset);
                }
                continue;
            }

            if !Self::high_region_contains(&maps.high_regions, byte_address, 1) {
                return None;
            }

            let (page, page_offset) = Self::low_page_parts(byte_address);
            *slot = maps
                .high_sparse_pages
                .get(&page)
                .map(|data| data[page_offset])
                .unwrap_or(0);
        }

        Some(out)
    }

    fn write_high_bytes<const N: usize>(&self, address: u64, bytes: [u8; N]) -> bool {
        if N != 0 && address.checked_add((N - 1) as u64).is_none() {
            return false;
        }

        let mut maps = self.maps.write().expect("guest memory maps lock poisoned");
        for index in 0..N {
            let byte_address = address.wrapping_add(index as u64);
            if self.checked_offset(byte_address).is_none()
                && !Self::high_region_contains(&maps.high_regions, byte_address, 1)
            {
                return false;
            }
        }

        for (index, value) in bytes.into_iter().enumerate() {
            let byte_address = address.wrapping_add(index as u64);
            if let Some(offset) = self.checked_offset(byte_address) {
                // SAFETY: the offset is in-bounds for the reserved guest span.
                unsafe {
                    *self.reservation.guest_base.add(offset) = value;
                }
            } else {
                let (page, page_offset) = Self::low_page_parts(byte_address);
                let page_data = maps
                    .high_sparse_pages
                    .entry(page)
                    .or_insert_with(|| Box::new([0; LOW_PAGE_SIZE]));
                page_data[page_offset] = value;
            }
        }

        true
    }

    fn read_bytes<const N: usize>(&self, address: u64) -> Option<[u8; N]> {
        if address < self.base {
            self.read_low_bytes::<N>(address)
        } else {
            self.read_high_bytes::<N>(address)
        }
    }

    fn write_bytes<const N: usize>(&self, address: u64, bytes: [u8; N]) -> bool {
        if trace_memory_watch_enabled() {
            let write_end = address.saturating_add(N as u64);
            let matched = trace_memory_watch_ranges()
                .iter()
                .copied()
                .find(|range| address < range.end && write_end > range.start);
            if let Some(range) = matched {
                let mut packed = [0u8; 8];
                let copy_len = N.min(8);
                packed[..copy_len].copy_from_slice(&bytes[..copy_len]);
                eprintln!(
                    "[mem-watch] write addr=0x{address:016x} size={} value=0x{:016x} watch=0x{:016x}..0x{:016x} block_pc=0x{:016x}",
                    N,
                    u64::from_le_bytes(packed),
                    range.start,
                    range.end,
                    MEMORY_WATCH_PC.load(Ordering::Relaxed),
                );
            }
        }

        if address < self.base {
            self.write_low_bytes::<N>(address, bytes)
        } else {
            self.write_high_bytes::<N>(address, bytes)
        }
    }

    pub fn load(&self, address: u64, data: &[u8]) -> Result<()> {
        let offset = address
            .checked_sub(self.base)
            .ok_or_else(|| anyhow!("address 0x{address:016x} is below memory base"))?;
        let offset = usize::try_from(offset)
            .map_err(|_| anyhow!("address 0x{address:016x} is outside addressable host space"))?;
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("program image overflows guest memory"))?;

        if end > self.size {
            return Err(anyhow!(
                "program image exceeds guest memory (end=0x{end:016x}, size=0x{:016x})",
                self.size
            ));
        }

        // SAFETY: bounds are checked above and destination stays within the fixed guest span.
        unsafe {
            let target = self.reservation.guest_base.add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), target, data.len());
        }

        let _ = self.map_high_region(
            address,
            data.len() as u64,
            MEMORY_STATE_HEAP,
            MEMORY_PERMISSION_READ_WRITE,
        );
        Ok(())
    }

    pub fn query_memory(&self, address: u64) -> MemoryQueryInfo {
        let maps = self.maps.read().expect("guest memory maps lock poisoned");

        if address >= self.base {
            if !maps.high_regions.is_empty() {
                if let Some(region) = maps
                    .high_regions
                    .iter()
                    .copied()
                    .find(|region| address >= region.start && address < region.end)
                {
                    return MemoryQueryInfo::mapped_with(
                        region.start,
                        region.end.saturating_sub(region.start),
                        region.state,
                        region.permission,
                    );
                }

                let next_region_start = maps
                    .high_regions
                    .iter()
                    .filter_map(|region| (region.start > address).then_some(region.start))
                    .min();

                let size = next_region_start
                    .map(|next| next.saturating_sub(address).max(1))
                    .unwrap_or_else(|| u64::MAX.saturating_sub(address));

                return MemoryQueryInfo::unmapped(address, size);
            }

            let high_end = self.base.saturating_add(self.size as u64);
            if address < high_end {
                return MemoryQueryInfo::mapped(self.base, high_end.saturating_sub(self.base));
            }

            return MemoryQueryInfo::unmapped(address, u64::MAX.saturating_sub(address));
        }

        if let Some((start, end)) = maps
            .low_regions
            .iter()
            .copied()
            .find(|(start, end)| address >= *start && address < *end)
        {
            return MemoryQueryInfo::unmapped(start, end.saturating_sub(start));
        }

        let next_region_start = maps
            .low_regions
            .iter()
            .filter_map(|(start, _)| (*start > address).then_some(*start))
            .min()
            .unwrap_or(self.base);
        let size = next_region_start.saturating_sub(address).max(1);

        MemoryQueryInfo::unmapped(address, size)
    }

    pub fn fastmem_region(&self) -> FastmemRegion {
        FastmemRegion {
            guest_base: self.base,
            host_ptr: self.reservation.guest_base,
            size: self.size as u64,
        }
    }

    #[allow(dead_code)]
    pub fn safepoint_page_host_ptr(&self) -> *mut u8 {
        self.reservation.reservation_base
    }

    #[allow(dead_code)]
    pub fn arm_safepoint_page(&self) -> bool {
        set_host_page_protection(
            self.reservation.reservation_base,
            self.reservation.safepoint_page_size,
            HostPageProtection::NoAccess,
        )
    }

    #[allow(dead_code)]
    pub fn disarm_safepoint_page(&self) -> bool {
        set_host_page_protection(
            self.reservation.reservation_base,
            self.reservation.safepoint_page_size,
            HostPageProtection::ReadWrite,
        )
    }

    pub fn fetch_u32(&self, address: u64) -> Option<u32> {
        let bytes = self.read_bytes::<4>(address)?;
        Some(u32::from_le_bytes(bytes))
    }

    pub fn read_u8(&self, address: u64) -> Option<u8> {
        let bytes = self.read_bytes::<1>(address)?;
        Some(bytes[0])
    }

    pub fn write_u8(&self, address: u64, value: u8) -> bool {
        self.write_bytes::<1>(address, [value])
    }

    pub fn read_u16(&self, address: u64) -> Option<u16> {
        let bytes = self.read_bytes::<2>(address)?;
        Some(u16::from_le_bytes(bytes))
    }

    pub fn write_u16(&self, address: u64, value: u16) -> bool {
        self.write_bytes::<2>(address, value.to_le_bytes())
    }

    pub fn read_u32(&self, address: u64) -> Option<u32> {
        let bytes = self.read_bytes::<4>(address)?;
        Some(u32::from_le_bytes(bytes))
    }

    pub fn write_u32(&self, address: u64, value: u32) -> bool {
        self.write_bytes::<4>(address, value.to_le_bytes())
    }

    pub fn read_u64(&self, address: u64) -> Option<u64> {
        let bytes = self.read_bytes::<8>(address)?;
        Some(u64::from_le_bytes(bytes))
    }

    pub fn write_u64(&self, address: u64, value: u64) -> bool {
        self.write_bytes::<8>(address, value.to_le_bytes())
    }
}

impl MemoryInterface for GuestMemory {
    fn fastmem_region(&mut self) -> Option<FastmemRegion> {
        Some(GuestMemory::fastmem_region(self))
    }

    fn fetch_u32(&mut self, address: u64) -> Option<u32> {
        let value = GuestMemory::fetch_u32(self, address);
        if value.is_none() {
            self.mark_fault(address);
        }
        value
    }

    fn read_u8(&mut self, address: u64) -> Option<u8> {
        let value = GuestMemory::read_u8(self, address);
        if value.is_none() {
            self.mark_fault(address);
        }
        value
    }

    fn write_u8(&mut self, address: u64, value: u8) -> bool {
        let ok = GuestMemory::write_u8(self, address, value);
        if !ok {
            self.mark_fault(address);
        }
        ok
    }

    fn read_u16(&mut self, address: u64) -> Option<u16> {
        let value = GuestMemory::read_u16(self, address);
        if value.is_none() {
            self.mark_fault(address);
        }
        value
    }

    fn write_u16(&mut self, address: u64, value: u16) -> bool {
        let ok = GuestMemory::write_u16(self, address, value);
        if !ok {
            self.mark_fault(address);
        }
        ok
    }

    fn read_u32(&mut self, address: u64) -> Option<u32> {
        let value = GuestMemory::read_u32(self, address);
        if value.is_none() {
            self.mark_fault(address);
        }
        value
    }

    fn write_u32(&mut self, address: u64, value: u32) -> bool {
        let ok = GuestMemory::write_u32(self, address, value);
        if !ok {
            self.mark_fault(address);
        }
        ok
    }

    fn read_u64(&mut self, address: u64) -> Option<u64> {
        let value = GuestMemory::read_u64(self, address);
        if value.is_none() {
            self.mark_fault(address);
        }
        value
    }

    fn write_u64(&mut self, address: u64, value: u64) -> bool {
        let ok = GuestMemory::write_u64(self, address, value);
        if !ok {
            self.mark_fault(address);
        }
        ok
    }
}

#[derive(Clone, Debug)]
pub struct SharedGuestMemory {
    inner: Arc<GuestMemory>,
}

pub struct LockedSharedMemoryView {
    memory: SharedGuestMemory,
}

impl SharedGuestMemory {
    pub fn new(base: u64, size: usize) -> Self {
        Self {
            inner: Arc::new(GuestMemory::new(base, size)),
        }
    }

    pub fn locked_view(&self) -> LockedSharedMemoryView {
        LockedSharedMemoryView {
            memory: self.clone(),
        }
    }

    pub fn base_address(&self) -> u64 {
        self.inner.base_address()
    }

    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes()
    }

    pub fn host_page_size(&self) -> usize {
        self.inner.host_page_size()
    }

    pub fn alias_mode(&self) -> MemoryAliasMode {
        self.inner.alias_mode()
    }

    #[allow(dead_code)]
    pub fn safepoint_page_host_ptr(&self) -> *mut u8 {
        self.inner.safepoint_page_host_ptr()
    }

    #[allow(dead_code)]
    pub fn arm_safepoint_page(&self) -> bool {
        self.inner.arm_safepoint_page()
    }

    #[allow(dead_code)]
    pub fn disarm_safepoint_page(&self) -> bool {
        self.inner.disarm_safepoint_page()
    }

    pub fn clear_low_regions(&self) {
        self.inner.clear_low_regions();
    }

    pub fn clear_high_regions(&self) {
        self.inner.clear_high_regions();
    }

    pub fn map_high_region(&self, start: u64, size: u64, state: u32, permission: u32) -> bool {
        self.inner.map_high_region(start, size, state, permission)
    }

    pub fn map_low_region(&self, start: u64, size: u64) -> bool {
        self.inner.map_low_region(start, size)
    }

    pub fn clear_last_fault(&self) {
        self.inner.clear_last_fault();
    }

    pub fn last_fault_address(&self) -> Option<u64> {
        self.inner.last_fault_address()
    }

    pub fn load(&self, address: u64, data: &[u8]) -> Result<()> {
        self.inner.load(address, data)
    }

    pub fn fetch_u32(&self, address: u64) -> Option<u32> {
        self.inner.fetch_u32(address)
    }

    pub fn read_u32(&self, address: u64) -> Option<u32> {
        self.inner.read_u32(address)
    }

    #[allow(dead_code)]
    pub fn read_u64(&self, address: u64) -> Option<u64> {
        self.inner.read_u64(address)
    }

    pub fn write_u32(&self, address: u64, value: u32) -> bool {
        self.inner.write_u32(address, value)
    }

    pub fn write_u64(&self, address: u64, value: u64) -> bool {
        self.inner.write_u64(address, value)
    }

    pub fn guest_base_address(&self) -> u64 {
        self.inner.base
    }

    pub fn guest_memory_size(&self) -> usize {
        self.inner.size
    }

    /// Return a raw byte slice covering the entire guest memory region.
    ///
    /// # Safety
    /// The caller must ensure no concurrent writes are occurring.
    pub unsafe fn guest_memory_slice(&self) -> &[u8] {
        unsafe { self.inner.guest_memory_slice() }
    }

    pub fn sparse_pages(&self) -> Vec<(u64, Vec<u8>)> {
        self.inner.sparse_pages()
    }

    pub fn query_memory(&self, address: u64) -> MemoryQueryInfo {
        self.inner.query_memory(address)
    }

    pub fn fork(&self) -> Self {
        Self {
            inner: Arc::new(self.inner.as_ref().clone()),
        }
    }
}

impl MemoryInterface for LockedSharedMemoryView {
    fn fastmem_region(&mut self) -> Option<FastmemRegion> {
        Some(self.memory.inner.fastmem_region())
    }

    fn fetch_u32(&mut self, address: u64) -> Option<u32> {
        let value = self.memory.inner.fetch_u32(address);
        if value.is_none() {
            self.memory.inner.mark_fault(address);
        }
        value
    }

    fn read_u8(&mut self, address: u64) -> Option<u8> {
        let value = self.memory.inner.read_u8(address);
        if value.is_none() {
            self.memory.inner.mark_fault(address);
        }
        value
    }

    fn write_u8(&mut self, address: u64, value: u8) -> bool {
        let ok = self.memory.inner.write_u8(address, value);
        if !ok {
            self.memory.inner.mark_fault(address);
        }
        ok
    }

    fn read_u16(&mut self, address: u64) -> Option<u16> {
        let value = self.memory.inner.read_u16(address);
        if value.is_none() {
            self.memory.inner.mark_fault(address);
        }
        value
    }

    fn write_u16(&mut self, address: u64, value: u16) -> bool {
        let ok = self.memory.inner.write_u16(address, value);
        if !ok {
            self.memory.inner.mark_fault(address);
        }
        ok
    }

    fn read_u32(&mut self, address: u64) -> Option<u32> {
        let value = self.memory.inner.read_u32(address);
        if value.is_none() {
            self.memory.inner.mark_fault(address);
        }
        value
    }

    fn write_u32(&mut self, address: u64, value: u32) -> bool {
        let ok = self.memory.inner.write_u32(address, value);
        if !ok {
            self.memory.inner.mark_fault(address);
        }
        ok
    }

    fn read_u64(&mut self, address: u64) -> Option<u64> {
        let value = self.memory.inner.read_u64(address);
        if value.is_none() {
            self.memory.inner.mark_fault(address);
        }
        value
    }

    fn write_u64(&mut self, address: u64, value: u64) -> bool {
        let ok = self.memory.inner.write_u64(address, value);
        if !ok {
            self.memory.inner.mark_fault(address);
        }
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::{GuestMemory, MEMORY_PERMISSION_READ_WRITE, MEMORY_STATE_HEAP};

    #[test]
    fn query_memory_uses_registered_high_regions() {
        let memory = GuestMemory::new(0x8000_0000, 0x10_0000);
        memory.clear_high_regions();

        assert!(memory.map_high_region(0x8000_0000, 0x1000, 0x3, 0x5));
        assert!(memory.map_high_region(0x8000_2000, 0x1000, 0x5, 0x3));

        let first = memory.query_memory(0x8000_0000);
        assert_eq!(first.address, 0x8000_0000);
        assert_eq!(first.size, 0x1000);
        assert_eq!(first.state, 0x3);
        assert_eq!(first.permission, 0x5);

        let gap = memory.query_memory(0x8000_1000);
        assert_eq!(gap.address, 0x8000_1000);
        assert_eq!(gap.size, 0x1000);
        assert_eq!(gap.state, 0);
        assert_eq!(gap.permission, 0);

        let second = memory.query_memory(0x8000_2000);
        assert_eq!(second.address, 0x8000_2000);
        assert_eq!(second.size, 0x1000);
        assert_eq!(second.state, 0x5);
        assert_eq!(second.permission, 0x3);
    }

    #[test]
    fn load_registers_default_heap_region() {
        let memory = GuestMemory::new(0x8000_0000, 0x10_0000);
        memory.clear_high_regions();

        memory
            .load(0x8000_4000, &[1, 2, 3, 4])
            .expect("load should succeed");

        let info = memory.query_memory(0x8000_4000);
        assert_eq!(info.address, 0x8000_4000);
        assert_eq!(info.size, 4);
        assert_eq!(info.state, MEMORY_STATE_HEAP);
        assert_eq!(info.permission, MEMORY_PERMISSION_READ_WRITE);
    }

    #[test]
    fn safepoint_page_toggles_between_rw_and_noaccess() {
        let memory = GuestMemory::new(0x8000_0000, 0x10_0000);
        assert!(memory.disarm_safepoint_page());
        assert!(memory.arm_safepoint_page());
    }

    #[test]
    fn high_sparse_pages_back_mapped_regions_outside_fastmem_span() {
        let memory = GuestMemory::new(0x8000_0000, 0x10_0000);
        memory.clear_high_regions();

        let address = 0xba60_0000;
        assert!(!memory.write_u32(address, 0x1122_3344));

        assert!(memory.map_high_region(
            0x8000_0000,
            0xc000_0000,
            MEMORY_STATE_HEAP,
            MEMORY_PERMISSION_READ_WRITE,
        ));
        assert_eq!(memory.read_u32(address), Some(0));
        assert!(memory.write_u32(address, 0x1122_3344));
        assert_eq!(memory.read_u32(address), Some(0x1122_3344));

        let cloned = memory.clone();
        assert_eq!(cloned.read_u32(address), Some(0x1122_3344));
    }
}
