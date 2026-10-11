use std::sync::atomic::{AtomicU64, Ordering};

pub const GUEST_PAGE_BITS: u64 = 12;
pub const GUEST_PAGE_SIZE: u64 = 1 << GUEST_PAGE_BITS;
pub const GUEST_PAGE_MASK: u64 = GUEST_PAGE_SIZE - 1;
pub const PTE_PTR_MASK: u64 = !GUEST_PAGE_MASK;
pub const PTE_FLAG_VALID: u64 = 1 << 0;
pub const PTE_FLAG_READ: u64 = 1 << 1;
pub const PTE_FLAG_WRITE: u64 = 1 << 2;
pub const PTE_FLAG_EXECUTE: u64 = 1 << 3;
pub const PTE_FLAG_TRACKED: u64 = 1 << 4;
pub const PTE_FLAG_SPECIAL: u64 = 1 << 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    Fastmem,
    PteChecked,
    Memhelper,
    Interpret,
}

impl MemoryStrategy {
    pub fn downgrade(self) -> Self {
        match self {
            MemoryStrategy::Fastmem => MemoryStrategy::PteChecked,
            MemoryStrategy::PteChecked => MemoryStrategy::Memhelper,
            MemoryStrategy::Memhelper => MemoryStrategy::Interpret,
            MemoryStrategy::Interpret => MemoryStrategy::Interpret,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessKind {
    Read,
    Write,
    Execute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultClass {
    Polling,
    MemoryFault,
    GranularityMismatch,
    HostFault,
}

#[derive(Debug, Clone, Copy)]
pub struct PteRegion {
    pub guest_base: u64,
    pub page_count: u64,
    pub pte_ptr: *const AtomicU64,
    pub epoch_ptr: *const AtomicU64,
}

impl PteRegion {
    pub fn contains(&self, address: u64) -> bool {
        address >= self.guest_base
            && ((address - self.guest_base) >> GUEST_PAGE_BITS) < self.page_count
    }

    pub fn page_index(&self, address: u64) -> Option<usize> {
        self.contains(address)
            .then_some(((address - self.guest_base) >> GUEST_PAGE_BITS) as usize)
    }

    pub fn load_pte(&self, address: u64) -> Option<u64> {
        let index = self.page_index(address)?;
        // SAFETY: `pte_ptr` references a page table owned by the memory implementation
        // and stays live for the duration of the bound step.
        Some(unsafe { (*self.pte_ptr.add(index)).load(Ordering::Acquire) })
    }

    pub fn classify_access(&self, address: u64, kind: MemoryAccessKind) -> FaultClass {
        let Some(pte) = self.load_pte(address) else {
            return FaultClass::MemoryFault;
        };

        if !pte_is_valid(pte) {
            return FaultClass::MemoryFault;
        }

        if pte_allows_access(pte, kind) {
            FaultClass::GranularityMismatch
        } else {
            FaultClass::MemoryFault
        }
    }

    pub fn load_epoch(&self) -> u64 {
        if self.epoch_ptr.is_null() {
            return 0;
        }

        // SAFETY: the epoch pointer shares the same lifetime guarantees as `pte_ptr`.
        unsafe { (*self.epoch_ptr).load(Ordering::Acquire) }
    }
}

#[derive(Debug)]
pub struct PteTable {
    guest_base: u64,
    page_count: usize,
    entries: Vec<AtomicU64>,
    epoch: AtomicU64,
}

impl PteTable {
    pub fn new(guest_base: u64, guest_size: u64) -> Self {
        let page_count = guest_size.div_ceil(GUEST_PAGE_SIZE) as usize;
        let mut entries = Vec::with_capacity(page_count);
        entries.resize_with(page_count, || AtomicU64::new(0));
        Self {
            guest_base,
            page_count,
            entries,
            epoch: AtomicU64::new(0),
        }
    }

    pub fn region(&self) -> PteRegion {
        PteRegion {
            guest_base: self.guest_base,
            page_count: self.page_count as u64,
            pte_ptr: self.entries.as_ptr(),
            epoch_ptr: &self.epoch,
        }
    }

    pub fn clear(&self) {
        for entry in &self.entries {
            entry.store(0, Ordering::Release);
        }
        self.bump_epoch();
    }

    pub fn map_range(&self, guest_address: u64, host_base: *mut u8, size: u64, flags: u64) -> bool {
        let Some(start_page) = self.page_index(guest_address) else {
            return false;
        };
        let page_count = size.div_ceil(GUEST_PAGE_SIZE) as usize;
        if start_page.saturating_add(page_count) > self.page_count {
            return false;
        }

        let host_base_addr = host_base as usize as u64;
        for page in 0..page_count {
            let guest_page_addr = guest_address + (page as u64 * GUEST_PAGE_SIZE);
            let host_page_addr = host_base_addr + (page as u64 * GUEST_PAGE_SIZE);
            let entry = (host_page_addr.wrapping_sub(guest_page_addr) & PTE_PTR_MASK)
                | (flags | PTE_FLAG_VALID);
            self.entries[start_page + page].store(entry, Ordering::Release);
        }
        self.bump_epoch();
        true
    }

    pub fn unmap_range(&self, guest_address: u64, size: u64) -> bool {
        let Some(start_page) = self.page_index(guest_address) else {
            return false;
        };
        let page_count = size.div_ceil(GUEST_PAGE_SIZE) as usize;
        if start_page.saturating_add(page_count) > self.page_count {
            return false;
        }

        for page in 0..page_count {
            self.entries[start_page + page].store(0, Ordering::Release);
        }
        self.bump_epoch();
        true
    }

    pub fn reprotect_range(&self, guest_address: u64, size: u64, flags: u64) -> bool {
        let Some(start_page) = self.page_index(guest_address) else {
            return false;
        };
        let page_count = size.div_ceil(GUEST_PAGE_SIZE) as usize;
        if start_page.saturating_add(page_count) > self.page_count {
            return false;
        }

        for page in 0..page_count {
            let old = self.entries[start_page + page].load(Ordering::Acquire);
            let new_entry = (old & PTE_PTR_MASK) | (flags & !PTE_PTR_MASK);
            self.entries[start_page + page].store(new_entry, Ordering::Release);
        }
        self.bump_epoch();
        true
    }

    fn page_index(&self, address: u64) -> Option<usize> {
        if address < self.guest_base {
            return None;
        }
        let page = ((address - self.guest_base) >> GUEST_PAGE_BITS) as usize;
        (page < self.page_count).then_some(page)
    }

    fn bump_epoch(&self) {
        let _ = self.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn pte_allows_access(pte: u64, kind: MemoryAccessKind) -> bool {
    if !pte_is_valid(pte) {
        return false;
    }

    match kind {
        MemoryAccessKind::Read => pte & PTE_FLAG_READ != 0,
        MemoryAccessKind::Write => pte & PTE_FLAG_WRITE != 0,
        MemoryAccessKind::Execute => pte & PTE_FLAG_EXECUTE != 0,
    }
}

pub fn pte_is_valid(pte: u64) -> bool {
    pte & PTE_FLAG_VALID != 0
}

pub fn decode_host_pointer(pte: u64, address: u64) -> *mut u8 {
    ((pte & PTE_PTR_MASK).wrapping_add(address)) as usize as *mut u8
}

pub fn rw_flags() -> u64 {
    PTE_FLAG_READ | PTE_FLAG_WRITE
}
