use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

pub const NVDRV_RESULT_SUCCESS: u32 = 0;
pub const NVDRV_RESULT_INVALID_STATE: u32 = 1;
pub const NVDRV_RESULT_BAD_VALUE: u32 = 4;

const PAGE_SIZE: u32 = 0x1000;
const NV_IOCTL_NVMAP_CUSTOM_MAGIC: u32 = 0x01;
const NV_IOCTL_NVGPU_MAGIC: u32 = 0x47;
const NV_IOCTL_NVHOST_MAGIC: u32 = 0x48;
const NV_IOCTL_AS_MAGIC: u32 = 0x41;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvdrvIoctlKind {
    Ioctl,
    Ioctl2,
    Ioctl3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NvIoctl {
    raw: u32,
}

impl NvIoctl {
    fn new(raw: u32) -> Self {
        Self { raw }
    }

    fn number(self) -> u32 {
        self.raw & 0xff
    }

    fn ty(self) -> u32 {
        (self.raw >> 8) & 0xff
    }

    fn size(self) -> u32 {
        (self.raw >> 16) & 0x3fff
    }

    fn direction(self) -> u32 {
        (self.raw >> 30) & 0x3
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NvDeviceKind {
    NvMap,
    NvHostCtrl,
    NvHostCtrlGpu,
    NvHostAsGpu,
    NvHostGpu,
    NvHostNvdec,
    NvHostVic,
    NvHostDbgGpu,
    NvHostProfGpu,
}

impl NvDeviceKind {
    fn from_path(path: &str) -> Option<Self> {
        match path {
            "/dev/nvmap" => Some(Self::NvMap),
            "/dev/nvhost-ctrl" => Some(Self::NvHostCtrl),
            "/dev/nvhost-ctrl-gpu" => Some(Self::NvHostCtrlGpu),
            "/dev/nvhost-as-gpu" => Some(Self::NvHostAsGpu),
            "/dev/nvhost-gpu" => Some(Self::NvHostGpu),
            "/dev/nvhost-nvdec" => Some(Self::NvHostNvdec),
            "/dev/nvhost-vic" => Some(Self::NvHostVic),
            "/dev/nvhost-dbg-gpu" => Some(Self::NvHostDbgGpu),
            "/dev/nvhost-prof-gpu" => Some(Self::NvHostProfGpu),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct NvOpenFd {
    path: String,
    kind: NvDeviceKind,
}

#[derive(Debug, Clone)]
struct NvMapHandle {
    size: u32,
    align: i32,
    kind: i32,
    address: u64,
    allocated: bool,
    refcount: i64,
}

/// Tracks a VA region allocated by AllocSpace.
#[derive(Debug, Clone)]
struct AsAllocation {
    size: u64,
    page_size: u32,
    sparse: bool,
    big_pages: bool,
}

/// Tracks a buffer mapping created by MapBufferEx.
#[derive(Debug, Clone)]
struct AsMapping {
    handle: i32,
    offset: u64,
    offset_in_handle: u64,
    size: u64,
    fixed: bool,
    big_page: bool,
    sparse_alloc: bool,
}

/// Simple bump allocator for GPU virtual address pages.
#[derive(Debug, Clone)]
struct SimpleAllocator {
    /// Start of allocatable region (in pages).
    start: u32,
    /// End of allocatable region (in pages).
    end: u32,
    /// Next page to allocate from (bump pointer).
    next: u32,
    /// Freed regions that can be reused, keyed by page offset.
    free_list: BTreeMap<u32, u32>,
}

impl SimpleAllocator {
    fn new(start: u32, limit: u32) -> Self {
        Self {
            start,
            end: start.saturating_add(limit),
            next: start,
            free_list: BTreeMap::new(),
        }
    }

    fn allocate(&mut self, pages: u32) -> u32 {
        // Try to find a free region first.
        let mut chosen = None;
        for (&offset, &count) in &self.free_list {
            if count >= pages {
                chosen = Some(offset);
                break;
            }
        }
        if let Some(offset) = chosen {
            let count = self.free_list.remove(&offset).unwrap();
            if count > pages {
                self.free_list.insert(offset + pages, count - pages);
            }
            return offset;
        }
        // Bump allocate.
        if self.next + pages > self.end {
            return 0; // OOM
        }
        let result = self.next;
        self.next += pages;
        result
    }

    fn allocate_fixed(&mut self, offset: u32, pages: u32) {
        // Just ensure our bump pointer is past this region.
        let region_end = offset.saturating_add(pages);
        if region_end > self.next {
            self.next = region_end;
        }
        // Remove any free list overlap.
        self.free_list.remove(&offset);
    }

    fn free(&mut self, offset: u32, pages: u32) {
        self.free_list.insert(offset, pages);
    }

    fn va_start(&self) -> u32 {
        self.start
    }

    fn va_limit(&self) -> u32 {
        self.end
    }
}

#[derive(Debug, Clone)]
pub struct NvdrvService {
    next_fd: i32,
    open_fds: HashMap<i32, NvOpenFd>,
    next_nvmap_handle: i32,
    nvmap_handles: HashMap<i32, NvMapHandle>,
    transfer_memory_initialized: bool,
    client_pid: Option<u64>,
    gpu_boot_time: Instant,

    // GPU address space state (nvhost-as-gpu)
    as_initialized: bool,
    big_page_size: u32,
    big_page_size_bits: u32,
    va_range_start: u64,
    va_range_split: u64,
    va_range_end: u64,
    small_page_allocator: Option<SimpleAllocator>,
    big_page_allocator: Option<SimpleAllocator>,
    allocation_map: BTreeMap<u64, AsAllocation>,
    mapping_map: BTreeMap<u64, AsMapping>,

    // GPU channel state (nvhost-gpu)
    gpu_nvmap_fd: i32,
    gpu_user_data: u64,
    gpu_channel_syncpoint: u32,
    gpu_fence_value: u32,
    gpu_channel_initialized: bool,
    gpu_channel_priority: u32,
    gpu_channel_timeslice: u32,

    // Persistent Maxwell GPU engine state (survives across GpFifo entries/batches)
    /// subchannel → engine class binding (SET_OBJECT, method 0)
    gpu_subchannel_class: [u32; 8],
    /// Per-engine query/semaphore registers {addr_high, addr_low, sequence}
    gpu_query_3d: [u32; 3],
    gpu_query_2d: [u32; 3],
    gpu_query_compute: [u32; 3],
    gpu_query_dma: [u32; 3],

    // Puller/host-class semaphore & syncpoint registers (methods < 0x40)
    gpu_puller_semaphore: [u32; 3], // [addr_high, addr_low, payload]
    gpu_puller_syncpoint_payload: u32,
}

const DEFAULT_BIG_PAGE_SIZE: u32 = 0x20000;
const SMALL_PAGE_SIZE: u32 = 0x1000;
const SMALL_PAGE_SIZE_BITS: u32 = 12;
const SUPPORTED_BIG_PAGE_SIZES: u32 = 0x30000;
const VA_START_SHIFT: u32 = 10;
const DEFAULT_VA_SPLIT: u64 = 1u64 << 34;
const DEFAULT_VA_RANGE: u64 = 1u64 << 37;

impl Default for NvdrvService {
    fn default() -> Self {
        Self {
            next_fd: 1,
            open_fds: HashMap::new(),
            next_nvmap_handle: 1,
            nvmap_handles: HashMap::new(),
            transfer_memory_initialized: false,
            client_pid: None,
            gpu_boot_time: Instant::now(),

            as_initialized: false,
            big_page_size: DEFAULT_BIG_PAGE_SIZE,
            big_page_size_bits: DEFAULT_BIG_PAGE_SIZE.trailing_zeros(),
            va_range_start: (DEFAULT_BIG_PAGE_SIZE as u64) << VA_START_SHIFT,
            va_range_split: DEFAULT_VA_SPLIT,
            va_range_end: DEFAULT_VA_RANGE,
            small_page_allocator: None,
            big_page_allocator: None,
            allocation_map: BTreeMap::new(),
            mapping_map: BTreeMap::new(),

            gpu_nvmap_fd: -1,
            gpu_user_data: 0,
            gpu_channel_syncpoint: 0,
            gpu_fence_value: 0,
            gpu_channel_initialized: false,
            gpu_channel_priority: 0,
            gpu_channel_timeslice: 0,

            gpu_subchannel_class: [0; 8],
            gpu_query_3d: [0; 3],
            gpu_query_2d: [0; 3],
            gpu_query_compute: [0; 3],
            gpu_query_dma: [0; 3],
            gpu_puller_semaphore: [0; 3],
            gpu_puller_syncpoint_payload: 0,
        }
    }
}

impl NvdrvService {
    pub fn open_device(&mut self, path: &str) -> Option<i32> {
        let kind = NvDeviceKind::from_path(path)?;

        let mut fd = self.next_fd.max(1);
        while self.open_fds.contains_key(&fd) {
            fd = fd.checked_add(1)?;
        }

        self.open_fds.insert(
            fd,
            NvOpenFd {
                path: path.to_string(),
                kind,
            },
        );
        self.next_fd = fd.checked_add(1).unwrap_or(1);
        Some(fd)
    }

    pub fn close_device(&mut self, fd: i32) -> bool {
        self.open_fds.remove(&fd).is_some()
    }

    pub fn device_path(&self, fd: i32) -> Option<&str> {
        self.open_fds.get(&fd).map(|entry| entry.path.as_str())
    }

    pub fn ioctl_result(
        &mut self,
        kind: NvdrvIoctlKind,
        fd: i32,
        ioctl_raw: u32,
        args: &mut [u8],
        inline_buffer: &mut [u8],
    ) -> u32 {
        let Some(open_fd) = self.open_fds.get(&fd).cloned() else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let ioctl = NvIoctl::new(ioctl_raw);
        match open_fd.kind {
            NvDeviceKind::NvMap => self.handle_nvmap_ioctl(kind, ioctl, args),
            NvDeviceKind::NvHostCtrl => self.handle_nvhost_ctrl_ioctl(kind, ioctl, args),
            NvDeviceKind::NvHostCtrlGpu => {
                self.handle_nvhost_ctrl_gpu_ioctl(kind, ioctl, args, inline_buffer)
            }
            NvDeviceKind::NvHostAsGpu => {
                self.handle_nvhost_as_gpu_ioctl(kind, ioctl, args, inline_buffer)
            }
            NvDeviceKind::NvHostGpu => {
                self.handle_nvhost_gpu_ioctl(kind, ioctl, args, inline_buffer)
            }
            NvDeviceKind::NvHostNvdec
            | NvDeviceKind::NvHostVic
            | NvDeviceKind::NvHostDbgGpu
            | NvDeviceKind::NvHostProfGpu => {
                unimplemented!(
                    "nvdrv {:?} ioctl 0x{ioctl_raw:08x} on {} (fd={fd}) is not implemented",
                    kind,
                    open_fd.path
                )
            }
        }
    }

    pub fn initialize_payload(
        &mut self,
        _gpu_backend_available: bool,
        _adapter_name: Option<&str>,
    ) -> Vec<u8> {
        self.transfer_memory_initialized = true;

        // Ryujinx returns only nv_result (u32 = 0 for success) in the
        // Initialize response.  The SDK reads exactly one u32 from the
        // response payload and interprets it as the NV result code.
        // Returning anything other than 0 can cause games to skip
        // GPU-related initialization leading to crashes.
        let mut payload = Vec::new();
        payload.extend_from_slice(&NVDRV_RESULT_SUCCESS.to_le_bytes());
        payload
    }

    pub fn query_event_command(&self, fd: i32, event_id: u32) -> Option<u32> {
        let open_fd = self.open_fds.get(&fd)?;

        match open_fd.kind {
            NvDeviceKind::NvHostCtrlGpu => match event_id {
                1 | 2 => Some(4 | ((event_id & 0xffff) << 16)),
                _ => None,
            },
            NvDeviceKind::NvHostCtrl => {
                // libnx passes event_id as `0x10000000 | slot` for nvhost-ctrl
                // QueryEvent calls.  Strip the flag to get the actual slot index.
                let slot = event_id & 0x0FFF_FFFF;
                // Return event handles for syncpoint events (up to 64 slots).
                if slot < 64 {
                    Some(4 | ((slot & 0xffff) << 16))
                } else {
                    None
                }
            }
            NvDeviceKind::NvHostGpu => {
                // nvhost-gpu events:
                //   1 = sm_exception_breakpoint_int_report_event
                //   2 = sm_exception_breakpoint_pause_report_event
                //   3 = error_notifier_event
                match event_id {
                    1 | 2 | 3 => Some(4 | ((event_id & 0xffff) << 16)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Returns true if the event from QueryEvent should be pre-signaled.
    /// Syncpoint events are signaled immediately (we pretend GPU work is done).
    /// Error/exception notifier events must NOT be signaled — the game polls
    /// them to check for GPU errors and will crash if they appear signaled.
    pub fn should_auto_signal_event(&self, fd: i32, event_id: u32) -> bool {
        let Some(open_fd) = self.open_fds.get(&fd) else {
            return false;
        };
        match open_fd.kind {
            // GPU ctrl events (error, thermal) — auto-signal for now
            NvDeviceKind::NvHostCtrlGpu => true,
            // Syncpoint wait events — auto-signal (pretend done)
            NvDeviceKind::NvHostCtrl => true,
            // nvhost-gpu: error notifier and SM exception events — do NOT signal
            NvDeviceKind::NvHostGpu => false,
            _ => false,
        }
    }

    pub fn map_shared_memory(&mut self, payload: &[u8]) -> u32 {
        let fd = read_i32(payload, 0).unwrap_or(-1);
        let Some(path) = self.device_path(fd) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        unimplemented!("nvdrv MapSharedMemory for {path} (fd={fd}) is not implemented")
    }

    pub fn get_status_payload(&self, _payload: &[u8]) -> Vec<u8> {
        if !self.transfer_memory_initialized {
            return NVDRV_RESULT_INVALID_STATE.to_le_bytes().to_vec();
        }

        let mut payload = vec![0; 0x20];
        payload.extend_from_slice(&NVDRV_RESULT_SUCCESS.to_le_bytes());
        payload
    }

    pub fn set_client_pid(&mut self, payload: &[u8]) -> u32 {
        self.client_pid = read_u64(payload, 0);
        NVDRV_RESULT_SUCCESS
    }

    pub fn finish_initialize(&mut self, _payload: &[u8]) -> u32 {
        NVDRV_RESULT_SUCCESS
    }

    // =====================================================================
    // nvhost-as-gpu ioctls (type 'A' = 0x41)
    // =====================================================================

    fn handle_nvhost_as_gpu_ioctl(
        &mut self,
        kind: NvdrvIoctlKind,
        ioctl: NvIoctl,
        args: &mut [u8],
        inline_buffer: &mut [u8],
    ) -> u32 {
        if ioctl.ty() != NV_IOCTL_AS_MAGIC {
            unimplemented!(
                "nvdrv {:?} as-gpu ioctl 0x{:08x} has unsupported type 0x{:02x}",
                kind,
                ioctl.raw,
                ioctl.ty()
            );
        }

        match (kind, ioctl.number()) {
            (_, 0x01) => self.as_gpu_bind_channel(args),
            (_, 0x02) => self.as_gpu_alloc_space(args),
            (_, 0x03) => self.as_gpu_free_space(args),
            (_, 0x05) => self.as_gpu_unmap_buffer(args),
            (_, 0x06) => self.as_gpu_map_buffer_ex(args),
            (NvdrvIoctlKind::Ioctl3, 0x08) => self.as_gpu_get_va_regions(args, Some(inline_buffer)),
            (_, 0x08) => self.as_gpu_get_va_regions(args, None),
            (_, 0x09) => self.as_gpu_alloc_as_ex(args),
            (_, 0x14) => self.as_gpu_remap(args),
            _ => unimplemented!(
                "nvdrv {:?} as-gpu ioctl number 0x{:02x} (raw=0x{:08x}) is not implemented",
                kind,
                ioctl.number(),
                ioctl.raw
            ),
        }
    }

    /// AllocAsEx (0x09): Initialize GPU address space.
    /// Struct (40 bytes): { u32 flags, i32 as_fd, u32 big_page_size, u32 reserved,
    ///                       u64 va_range_start, u64 va_range_end, u64 va_range_split }
    fn as_gpu_alloc_as_ex(&mut self, args: &mut [u8]) -> u32 {
        if self.as_initialized {
            eprintln!("nvdrv: AllocAsEx called but address space already initialized");
            return NVDRV_RESULT_BAD_VALUE;
        }

        let flags = read_u32(args, 0).unwrap_or(0);
        let _as_fd = read_i32(args, 4).unwrap_or(0);
        let big_page_size_param = read_u32(args, 8).unwrap_or(0);
        let _reserved = read_u32(args, 12).unwrap_or(0);
        let va_range_start_param = read_u64(args, 16).unwrap_or(0);
        let va_range_end_param = read_u64(args, 24).unwrap_or(0);
        let va_range_split_param = read_u64(args, 32).unwrap_or(0);

        eprintln!(
            "nvdrv: AllocAsEx flags=0x{:x} big_page_size=0x{:x} va_range_start=0x{:x} \
             va_range_end=0x{:x} va_range_split=0x{:x}",
            flags,
            big_page_size_param,
            va_range_start_param,
            va_range_end_param,
            va_range_split_param
        );

        // Apply big page size if specified.
        if big_page_size_param != 0 {
            if !is_power_of_two(big_page_size_param) {
                return NVDRV_RESULT_BAD_VALUE;
            }
            if (big_page_size_param & SUPPORTED_BIG_PAGE_SIZES) == 0 {
                return NVDRV_RESULT_BAD_VALUE;
            }
            self.big_page_size = big_page_size_param;
            self.big_page_size_bits = big_page_size_param.trailing_zeros();
            self.va_range_start = (big_page_size_param as u64) << VA_START_SHIFT;
        }

        // Apply custom VA range if specified.
        if va_range_start_param != 0 {
            self.va_range_start = va_range_start_param;
            self.va_range_split = va_range_split_param;
            self.va_range_end = va_range_end_param;
        }

        // Create small page allocator: covers [va_range_start, va_range_split) in 4K pages.
        let small_start_pages = (self.va_range_start >> SMALL_PAGE_SIZE_BITS) as u32;
        let small_end_pages = (self.va_range_split >> SMALL_PAGE_SIZE_BITS) as u32;
        let small_limit = small_end_pages.saturating_sub(small_start_pages);
        self.small_page_allocator = Some(SimpleAllocator::new(small_start_pages, small_limit));

        // Create big page allocator: covers [va_range_split, va_range_end) in big pages.
        let big_start_pages = (self.va_range_split >> self.big_page_size_bits) as u32;
        let big_total_pages =
            ((self.va_range_end - self.va_range_split) >> self.big_page_size_bits) as u32;
        self.big_page_allocator = Some(SimpleAllocator::new(big_start_pages, big_total_pages));

        self.as_initialized = true;
        NVDRV_RESULT_SUCCESS
    }

    /// BindChannel (0x01): Bind a GPU channel to this address space.
    /// Struct (4 bytes): { i32 fd }
    fn as_gpu_bind_channel(&mut self, args: &mut [u8]) -> u32 {
        let fd = read_i32(args, 0).unwrap_or(-1);
        eprintln!("nvdrv: AsGpu BindChannel fd={}", fd);
        // Stub: In a full implementation this would link the channel's memory manager.
        NVDRV_RESULT_SUCCESS
    }

    /// AllocSpace (0x02): Allocate a region in the GPU virtual address space.
    /// Struct (24 bytes): { u32 pages, u32 page_size, u32 flags, u32 pad, u64 offset/align }
    fn as_gpu_alloc_space(&mut self, args: &mut [u8]) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let pages = read_u32(args, 0).unwrap_or(0);
        let page_size = read_u32(args, 4).unwrap_or(0);
        let flags = read_u32(args, 8).unwrap_or(0);
        let offset_or_align = read_u64(args, 16).unwrap_or(0);

        if page_size != SMALL_PAGE_SIZE && page_size != self.big_page_size {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let is_big = page_size != SMALL_PAGE_SIZE;
        let page_size_bits = if is_big {
            self.big_page_size_bits
        } else {
            SMALL_PAGE_SIZE_BITS
        };

        let is_fixed = (flags & 1) != 0;
        let is_sparse = (flags & 2) != 0;

        let allocator = if is_big {
            self.big_page_allocator.as_mut()
        } else {
            self.small_page_allocator.as_mut()
        };
        let Some(allocator) = allocator else {
            return NVDRV_RESULT_BAD_VALUE;
        };

        let offset = if is_fixed {
            let page_offset = (offset_or_align >> page_size_bits) as u32;
            allocator.allocate_fixed(page_offset, pages);
            offset_or_align
        } else {
            let page_offset = allocator.allocate(pages);
            if page_offset == 0 && pages > 0 {
                return NVDRV_RESULT_BAD_VALUE;
            }
            (page_offset as u64) << page_size_bits
        };

        // Write back the offset to the args buffer.
        write_u64(args, 16, offset);

        let size = (pages as u64) * (page_size as u64);
        self.allocation_map.insert(
            offset,
            AsAllocation {
                size,
                page_size,
                sparse: is_sparse,
                big_pages: is_big,
            },
        );

        eprintln!(
            "nvdrv: AsGpu AllocSpace pages={} page_size=0x{:x} flags=0x{:x} -> offset=0x{:x}",
            pages, page_size, flags, offset
        );

        NVDRV_RESULT_SUCCESS
    }

    /// FreeSpace (0x03): Free a previously allocated VA region.
    /// Struct (16 bytes): { u64 offset, u32 pages, u32 page_size }
    fn as_gpu_free_space(&mut self, args: &mut [u8]) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let offset = read_u64(args, 0).unwrap_or(0);
        let pages = read_u32(args, 8).unwrap_or(0);
        let page_size = read_u32(args, 12).unwrap_or(0);

        let alloc = match self.allocation_map.remove(&offset) {
            Some(a) => a,
            None => return NVDRV_RESULT_BAD_VALUE,
        };

        if alloc.page_size != page_size || alloc.size != (pages as u64) * (page_size as u64) {
            // Put it back.
            self.allocation_map.insert(offset, alloc);
            return NVDRV_RESULT_BAD_VALUE;
        }

        let is_big = page_size != SMALL_PAGE_SIZE;
        let page_size_bits = if is_big {
            self.big_page_size_bits
        } else {
            SMALL_PAGE_SIZE_BITS
        };
        let allocator = if is_big {
            self.big_page_allocator.as_mut()
        } else {
            self.small_page_allocator.as_mut()
        };
        if let Some(allocator) = allocator {
            allocator.free((offset >> page_size_bits) as u32, pages);
        }

        eprintln!(
            "nvdrv: AsGpu FreeSpace offset=0x{:x} pages={} page_size=0x{:x}",
            offset, pages, page_size
        );

        NVDRV_RESULT_SUCCESS
    }

    /// UnmapBuffer (0x05): Unmap a mapped buffer.
    /// Struct (8 bytes): { i64 offset }
    fn as_gpu_unmap_buffer(&mut self, args: &mut [u8]) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let offset = read_u64(args, 0).unwrap_or(0) as u64;

        if let Some(mapping) = self.mapping_map.remove(&offset) {
            if !mapping.fixed {
                let is_big = mapping.big_page;
                let page_size_bits = if is_big {
                    self.big_page_size_bits
                } else {
                    SMALL_PAGE_SIZE_BITS
                };
                let page_size = if is_big {
                    self.big_page_size
                } else {
                    SMALL_PAGE_SIZE
                };
                let aligned_size = align_up_u64(mapping.size, page_size as u64);
                let allocator = if is_big {
                    self.big_page_allocator.as_mut()
                } else {
                    self.small_page_allocator.as_mut()
                };
                if let Some(allocator) = allocator {
                    allocator.free(
                        (mapping.offset >> page_size_bits) as u32,
                        (aligned_size >> page_size_bits) as u32,
                    );
                }
            }
            eprintln!("nvdrv: AsGpu UnmapBuffer offset=0x{:x}", offset);
        } else {
            eprintln!("nvdrv: AsGpu UnmapBuffer: no mapping at 0x{:x}", offset);
        }

        NVDRV_RESULT_SUCCESS
    }

    /// MapBufferEx (0x06): Map an nvmap handle into GPU virtual address space.
    /// Struct (40 bytes): { u32 flags, u32 kind, i32 handle, u32 page_size,
    ///                       i64 buffer_offset, u64 mapping_size, i64 offset }
    fn as_gpu_map_buffer_ex(&mut self, args: &mut [u8]) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let flags = read_u32(args, 0).unwrap_or(0);
        let _kind = read_u32(args, 4).unwrap_or(0);
        let handle = read_i32(args, 8).unwrap_or(0);
        let _page_size = read_u32(args, 12).unwrap_or(0);
        let buffer_offset = read_i64(args, 16).unwrap_or(0);
        let mapping_size = read_u64(args, 24).unwrap_or(0);
        let offset_param = read_i64(args, 32).unwrap_or(0);

        let is_fixed = (flags & 1) != 0;
        let is_remap = (flags & 0x100) != 0;

        // Handle remap case.
        if is_remap {
            if self.mapping_map.contains_key(&(offset_param as u64)) {
                eprintln!("nvdrv: AsGpu MapBufferEx remap at 0x{:x}", offset_param);
                return NVDRV_RESULT_SUCCESS;
            } else {
                eprintln!(
                    "nvdrv: AsGpu MapBufferEx remap: no mapping at 0x{:x}",
                    offset_param
                );
                return NVDRV_RESULT_BAD_VALUE;
            }
        }

        // Get nvmap handle to determine size.
        let handle_size = self
            .nvmap_handles
            .get(&handle)
            .map(|h| h.size as u64)
            .unwrap_or(0);
        let size = if mapping_size != 0 {
            mapping_size
        } else {
            handle_size
        };

        // Determine if we should use big or small pages based on handle alignment.
        let handle_align = self
            .nvmap_handles
            .get(&handle)
            .map(|h| h.align as u32)
            .unwrap_or(SMALL_PAGE_SIZE);
        let big_page = handle_align >= self.big_page_size
            || (size >= self.big_page_size as u64 && handle_align >= self.big_page_size);
        // More conservative: use big pages only if aligned.
        let big_page = (handle_align % self.big_page_size) == 0;

        let offset;
        if is_fixed {
            offset = offset_param as u64;
            // Verify it falls within an allocation.
            let valid = self
                .allocation_map
                .range(..=offset)
                .next_back()
                .map(|(&alloc_offset, alloc)| {
                    offset >= alloc_offset && (offset - alloc_offset) + size <= alloc.size
                })
                .unwrap_or(false);
            if !valid {
                eprintln!(
                    "nvdrv: AsGpu MapBufferEx fixed mapping at 0x{:x} not in any allocation",
                    offset
                );
                return NVDRV_RESULT_BAD_VALUE;
            }
        } else {
            // Dynamic allocation.
            let (allocator, page_size, page_size_bits) = if big_page {
                (
                    self.big_page_allocator.as_mut(),
                    self.big_page_size,
                    self.big_page_size_bits,
                )
            } else {
                (
                    self.small_page_allocator.as_mut(),
                    SMALL_PAGE_SIZE,
                    SMALL_PAGE_SIZE_BITS,
                )
            };
            let Some(allocator) = allocator else {
                return NVDRV_RESULT_BAD_VALUE;
            };
            let aligned_size = align_up_u64(size, page_size as u64);
            let pages = (aligned_size >> page_size_bits) as u32;
            let page_offset = allocator.allocate(pages);
            if page_offset == 0 && pages > 0 {
                return NVDRV_RESULT_BAD_VALUE;
            }
            offset = (page_offset as u64) << page_size_bits;
        }

        // Write offset back.
        write_i64(args, 32, offset as i64);

        self.mapping_map.insert(
            offset,
            AsMapping {
                handle,
                offset,
                offset_in_handle: buffer_offset.max(0) as u64,
                size,
                fixed: is_fixed,
                big_page,
                sparse_alloc: false,
            },
        );

        eprintln!(
            "nvdrv: AsGpu MapBufferEx handle={} size=0x{:x} flags=0x{:x} -> offset=0x{:x}",
            handle, size, flags, offset
        );

        NVDRV_RESULT_SUCCESS
    }

    /// GetVaRegions (0x08): Get virtual address region info.
    /// Struct (64 bytes): { u64 buf_addr, u32 buf_size, u32 reserved,
    ///                       VaRegion[2] { u64 offset, u32 page_size, u32 pad, u64 pages } }
    fn as_gpu_get_va_regions(&self, args: &mut [u8], inline_out: Option<&mut [u8]>) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let small_alloc = self.small_page_allocator.as_ref().unwrap();
        let big_alloc = self.big_page_allocator.as_ref().unwrap();

        // buf_size = 2 * sizeof(VaRegion) = 2 * 0x18 = 0x30
        write_u32(args, 8, 0x30);

        // Region 0: small pages
        let region0_offset = 16;
        let small_va_start = (small_alloc.va_start() as u64) << SMALL_PAGE_SIZE_BITS;
        let small_va_pages = (small_alloc.va_limit() as u64) - (small_alloc.va_start() as u64);
        write_u64(args, region0_offset, small_va_start);
        write_u32(args, region0_offset + 8, SMALL_PAGE_SIZE);
        write_u32(args, region0_offset + 12, 0); // pad
        write_u64(args, region0_offset + 16, small_va_pages);

        // Region 1: big pages
        let region1_offset = region0_offset + 24;
        let big_va_start = (big_alloc.va_start() as u64) << self.big_page_size_bits;
        let big_va_pages = (big_alloc.va_limit() as u64) - (big_alloc.va_start() as u64);
        write_u64(args, region1_offset, big_va_start);
        write_u32(args, region1_offset + 8, self.big_page_size);
        write_u32(args, region1_offset + 12, 0); // pad
        write_u64(args, region1_offset + 16, big_va_pages);

        // Also write to inline output for Ioctl3
        if let Some(buf) = inline_out {
            if buf.len() >= 0x30 {
                write_u64(buf, 0, small_va_start);
                write_u32(buf, 8, SMALL_PAGE_SIZE);
                write_u32(buf, 12, 0);
                write_u64(buf, 16, small_va_pages);
                write_u64(buf, 24, big_va_start);
                write_u32(buf, 32, self.big_page_size);
                write_u32(buf, 36, 0);
                write_u64(buf, 40, big_va_pages);
            }
        }

        NVDRV_RESULT_SUCCESS
    }

    /// Remap (0x14): Remap entries in a sparse allocation.
    /// Variable-length array of 20-byte entries:
    /// { u16 flags, u16 kind, u32 handle, u32 handle_offset_big_pages,
    ///   u32 as_offset_big_pages, u32 big_pages }
    fn as_gpu_remap(&mut self, args: &mut [u8]) -> u32 {
        if !self.as_initialized {
            return NVDRV_RESULT_BAD_VALUE;
        }

        let entry_size = 20;
        let num_entries = args.len() / entry_size;

        for i in 0..num_entries {
            let base = i * entry_size;
            let _flags = read_u16(args, base).unwrap_or(0);
            let _kind = read_u16(args, base + 2).unwrap_or(0);
            let _handle = read_u32(args, base + 4).unwrap_or(0);
            let _handle_offset = read_u32(args, base + 8).unwrap_or(0);
            let as_offset_big_pages = read_u32(args, base + 12).unwrap_or(0);
            let big_pages = read_u32(args, base + 16).unwrap_or(0);

            let _virtual_address = (as_offset_big_pages as u64) << self.big_page_size_bits;
            let _size = (big_pages as u64) << self.big_page_size_bits;

            // Stub: we accept the remap but don't actually change GPU page tables
            // since we don't have a real GPU memory manager yet.
        }

        eprintln!("nvdrv: AsGpu Remap {} entries (stubbed)", num_entries);
        NVDRV_RESULT_SUCCESS
    }

    // =====================================================================
    // nvhost-gpu ioctls
    // =====================================================================

    fn handle_nvhost_gpu_ioctl(
        &mut self,
        kind: NvdrvIoctlKind,
        ioctl: NvIoctl,
        args: &mut [u8],
        _inline_buffer: &mut [u8],
    ) -> u32 {
        let group = ioctl.ty();
        let cmd = ioctl.number();

        match (group, cmd) {
            // Group 0x00 - NvHost
            (0x00, 0x02) => {
                // GetSyncpoint: { u32 id } -> { u32 syncpoint }
                // Return our allocated syncpoint id.
                write_u32(args, 4, self.gpu_channel_syncpoint);
                NVDRV_RESULT_SUCCESS
            }
            (0x00, 0x03) => {
                // GetWaitbase: { u32 unknown, u32 value }
                write_u32(args, 4, 0);
                NVDRV_RESULT_SUCCESS
            }

            // Group 0x48 'H' - NvHostGpu
            (NV_IOCTL_NVHOST_MAGIC, 0x01) => {
                // SetNvMapFd: { i32 nvmap_fd }
                self.gpu_nvmap_fd = read_i32(args, 0).unwrap_or(-1);
                eprintln!("nvdrv: NvHostGpu SetNvMapFd fd={}", self.gpu_nvmap_fd);
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x03) => {
                // ChannelSetTimeout: { u32 timeout }
                eprintln!(
                    "nvdrv: NvHostGpu ChannelSetTimeout timeout={}",
                    read_u32(args, 0).unwrap_or(0)
                );
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x08) => {
                // SubmitGpfifo (inline entries)
                self.gpu_submit_gpfifo(args, false)
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x09) => {
                // AllocateObjectContext: { u32 class_num, u32 flags, u64 obj_id }
                let class_num = read_u32(args, 0).unwrap_or(0);
                eprintln!(
                    "nvdrv: NvHostGpu AllocObjCtx class=0x{:x} (stubbed)",
                    class_num
                );
                write_u64(args, 8, 0); // obj_id = 0
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x0b) => {
                // ZCullBind: { u64 gpu_va, u32 mode, u32 pad }
                eprintln!(
                    "nvdrv: NvHostGpu ZCullBind gpu_va=0x{:x} mode={} (stubbed)",
                    read_u64(args, 0).unwrap_or(0),
                    read_u32(args, 8).unwrap_or(0)
                );
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x0c) => {
                // SetErrorNotifier: { u64 offset, u64 size, u32 mem, u32 pad }
                eprintln!("nvdrv: NvHostGpu SetErrorNotifier (stubbed)");
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x0d) => {
                // SetChannelPriority: { u32 priority }
                self.gpu_channel_priority = read_u32(args, 0).unwrap_or(0);
                eprintln!(
                    "nvdrv: NvHostGpu SetPriority priority=0x{:x}",
                    self.gpu_channel_priority
                );
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x1a) => {
                // AllocGpfifoEx2: { u32 num_entries, u32 flags, u32 unk0,
                //                    NvFence { u32 id, u32 value }, u32 unk1, u32 unk2, u32 unk3 }
                self.gpu_alloc_gpfifo_ex2(args)
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x1b) => {
                // SubmitGpfifoEx (kickoff from guest memory)
                self.gpu_submit_gpfifo(args, true)
            }
            (NV_IOCTL_NVHOST_MAGIC, 0x1d) => {
                // ChannelSetTimeslice: { u32 timeslice }
                self.gpu_channel_timeslice = read_u32(args, 0).unwrap_or(0);
                eprintln!(
                    "nvdrv: NvHostGpu SetTimeslice timeslice=0x{:x}",
                    self.gpu_channel_timeslice
                );
                NVDRV_RESULT_SUCCESS
            }

            // Group 0x47 'G' - NvGpu specific
            (NV_IOCTL_NVGPU_MAGIC, 0x14) => {
                // SetClientData: { u64 data }
                self.gpu_user_data = read_u64(args, 0).unwrap_or(0);
                eprintln!(
                    "nvdrv: NvHostGpu SetClientData data=0x{:x}",
                    self.gpu_user_data
                );
                NVDRV_RESULT_SUCCESS
            }
            (NV_IOCTL_NVGPU_MAGIC, 0x15) => {
                // GetClientData: { u64 data }
                write_u64(args, 0, self.gpu_user_data);
                NVDRV_RESULT_SUCCESS
            }

            _ => {
                unimplemented!(
                    "nvdrv {:?} nvhost-gpu ioctl group=0x{:02x} cmd=0x{:02x} (raw=0x{:08x}) is not implemented",
                    kind,
                    group,
                    cmd,
                    ioctl.raw
                )
            }
        }
    }

    /// AllocGpfifoEx2: Allocate a GPU FIFO channel.
    /// Struct (32 bytes): { u32 num_entries, u32 flags, u32 unk0,
    ///                       u32 fence_id (out), u32 fence_value (out),
    ///                       u32 unk1, u32 unk2, u32 unk3 }
    fn gpu_alloc_gpfifo_ex2(&mut self, args: &mut [u8]) -> u32 {
        let num_entries = read_u32(args, 0).unwrap_or(0);
        let flags = read_u32(args, 4).unwrap_or(0);

        if self.gpu_channel_initialized {
            eprintln!("nvdrv: AllocGpfifoEx2 called but channel already initialized");
            return NVDRV_RESULT_BAD_VALUE;
        }

        // Allocate a syncpoint for this channel.
        self.gpu_channel_syncpoint = 1; // Simple fixed syncpoint id.
        self.gpu_channel_initialized = true;

        // Write fence output: { u32 id, u32 value } at offset 12.
        write_u32(args, 12, self.gpu_channel_syncpoint); // fence.id
        write_u32(args, 16, 0); // fence.value

        eprintln!(
            "nvdrv: NvHostGpu AllocGpfifoEx2 num_entries={} flags=0x{:x} -> syncpoint={}",
            num_entries, flags, self.gpu_channel_syncpoint
        );

        NVDRV_RESULT_SUCCESS
    }

    /// SubmitGpfifo: Submit GPU command lists.
    /// Struct (24 bytes): { u64 address, u32 num_entries, u32 flags, u32 fence_id, u32 fence_value }
    /// Followed by inline entries (if not kickoff mode).
    fn gpu_submit_gpfifo(&mut self, args: &mut [u8], _kickoff: bool) -> u32 {
        let address = read_u64(args, 0).unwrap_or(0);
        let num_entries = read_u32(args, 8).unwrap_or(0);
        let flags = read_u32(args, 12).unwrap_or(0);

        eprintln!(
            "nvdrv: NvHostGpu SubmitGpfifo addr=0x{:x} entries={} flags=0x{:x}",
            address, num_entries, flags
        );

        // If FenceIncrement (bit 2) is set, increment the syncpoint value.
        if flags & 0x4 != 0 {
            self.gpu_fence_value = self.gpu_fence_value.wrapping_add(1);
        }

        // Return the current fence as the output.
        write_u32(args, 16, self.gpu_channel_syncpoint); // fence.id
        write_u32(args, 20, self.gpu_fence_value); // fence.value

        eprintln!(
            "nvdrv: SubmitGpfifo -> fence_id={} fence_value={}",
            self.gpu_channel_syncpoint, self.gpu_fence_value
        );

        // Clear flags to indicate success.
        write_u32(args, 12, 0);

        NVDRV_RESULT_SUCCESS
    }

    /// Translate a GPU virtual address to a guest (CPU) virtual address.
    /// Uses the nvhost-as-gpu mapping table to find which NvMap handle is
    /// mapped at the given GPU VA and computes the corresponding CPU address.
    pub fn translate_gpu_va(&self, gpu_va: u64) -> Option<u64> {
        // Find the mapping that contains this GPU VA.  mapping_map is keyed by
        // the mapping's starting GPU VA.
        let (&map_start, mapping) = self.mapping_map.range(..=gpu_va).next_back()?;
        let offset_in_mapping = gpu_va.checked_sub(map_start)?;
        if offset_in_mapping >= mapping.size {
            return None; // outside this mapping
        }
        let handle = self.nvmap_handles.get(&mapping.handle)?;
        if !handle.allocated || handle.address == 0 {
            return None;
        }
        Some(handle.address + mapping.offset_in_handle + offset_in_mapping)
    }

    /// Process GpFifo entries after a SubmitGpfifo ioctl.
    ///
    /// Parses each 8-byte GpFifo entry, translates the GPU VA of the command
    /// buffer to a guest VA, reads the command words, and scans for semaphore
    /// release / query write commands.  When found, the semaphore value is
    /// written to guest memory via `memory_writer`.
    pub fn process_gpfifo_entries(
        &mut self,
        inline_buffer: &[u8],
        memory_reader: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
        memory_writer: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if inline_buffer.is_empty() {
            return;
        }

        let entry_count = inline_buffer.len() / 8;
        eprintln!(
            "[gpfifo] processing {} entries ({} bytes)",
            entry_count,
            inline_buffer.len()
        );
        for i in 0..entry_count {
            let entry = u64::from_le_bytes(
                inline_buffer[i * 8..(i + 1) * 8]
                    .try_into()
                    .unwrap_or([0; 8]),
            );

            // GpFifo entry format (Entry0 + Entry1 as u64):
            //   [0]     = fetch (conditional flag)
            //   [1]     = reserved
            //   [39:2]  = GPU VA (40-bit, 4-byte aligned)
            //   [40]    = priv
            //   [41]    = level (0=main, 1=subroutine)
            //   [62:42] = length in dwords (21 bits)
            //   [63]    = sync
            let gpu_va = entry & 0xFF_FFFF_FFFC;
            let length_dwords = ((entry >> 42) & 0x1FFFFF) as usize;
            eprintln!(
                "  entry[{}]: raw=0x{:016x} gpu_va=0x{:x} len={} dwords",
                i, entry, gpu_va, length_dwords
            );
            let length_dwords = ((entry >> 42) & 0xFFFFF) as usize;
            if length_dwords == 0 || gpu_va == 0 {
                continue;
            }

            let Some(guest_va) = self.translate_gpu_va(gpu_va) else {
                eprintln!(
                    "  gpfifo[{}]: gpu_va=0x{:x} len={} dwords — UNTRANSLATABLE",
                    i, gpu_va, length_dwords
                );
                continue;
            };

            let byte_len = length_dwords * 4;
            let Some(cmd_bytes) = memory_reader(guest_va, byte_len) else {
                eprintln!(
                    "  gpfifo[{}]: gpu_va=0x{:x} guest_va=0x{:x} len={} — UNREADABLE",
                    i, gpu_va, guest_va, byte_len
                );
                continue;
            };

            eprintln!(
                "  gpfifo[{}]: gpu_va=0x{:x} guest_va=0x{:x} len={} dwords",
                i, gpu_va, guest_va, length_dwords
            );

            // Parse the command stream and execute semaphore writes.
            self.process_gpu_commands(&cmd_bytes, memory_writer);
        }
    }

    /// Parse a buffer of GPU command words and execute semaphore/query writes.
    ///
    /// The Maxwell command stream uses submission headers:
    ///   [12:0]  = method (register offset)
    ///   [15:13] = subchannel
    ///   [28:16] = arg count or inline data
    ///   [31:29] = submission mode
    ///     0 = IncreasingOld (method, method+1, ...)
    ///     1 = Increasing    (method, method+1, ...)
    ///     2 = NonIncreasingOld (same method N times)
    ///     3 = NonIncreasing (same method N times)
    ///     4 = Inline (data in bits [28:16], no extra words)
    ///     5 = IncreaseOnce (method, method+1, method+1, ...)
    fn process_gpu_commands(
        &mut self,
        cmd_bytes: &[u8],
        memory_writer: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let dword_count = cmd_bytes.len() / 4;
        let dwords: Vec<u32> = (0..dword_count)
            .map(|i| u32::from_le_bytes(cmd_bytes[i * 4..(i + 1) * 4].try_into().unwrap_or([0; 4])))
            .collect();

        let mut pos = 0;
        while pos < dwords.len() {
            let header = dwords[pos];
            pos += 1;

            let method = header & 0x1FFF;
            let subchannel = ((header >> 13) & 0x7) as usize;
            let arg_count_or_data = (header >> 16) & 0x1FFF;
            let mode = (header >> 29) & 0x7;

            match mode {
                4 => {
                    // Inline: data is bits [28:16] of the header, no trailing words.
                    let data = arg_count_or_data;
                    if method == 0 {
                        if data != 0 {
                            self.gpu_subchannel_class[subchannel] = data;
                        }
                        eprintln!(
                            "  [gpu] SET_OBJECT subchannel={} class=0x{:x}",
                            subchannel, data
                        );
                    }
                    self.handle_gpu_method(subchannel, method, data, memory_writer);
                }
                0 | 1 => {
                    // Increasing / IncreasingOld: method, method+1, method+2, ...
                    let count = arg_count_or_data as usize;
                    for j in 0..count {
                        if pos >= dwords.len() {
                            break;
                        }
                        let m = method + j as u32;
                        let d = dwords[pos];
                        if m == 0 {
                            if d != 0 {
                                self.gpu_subchannel_class[subchannel] = d;
                            }
                            eprintln!(
                                "  [gpu] SET_OBJECT subchannel={} class=0x{:x}",
                                subchannel, d
                            );
                        }
                        self.handle_gpu_method(subchannel, m, d, memory_writer);
                        pos += 1;
                    }
                }
                2 | 3 => {
                    // NonIncreasing / NonIncreasingOld: all args go to the same method.
                    let count = arg_count_or_data as usize;
                    for _ in 0..count {
                        if pos >= dwords.len() {
                            break;
                        }
                        let d = dwords[pos];
                        if method == 0 {
                            if d != 0 {
                                self.gpu_subchannel_class[subchannel] = d;
                            }
                            eprintln!(
                                "  [gpu] SET_OBJECT subchannel={} class=0x{:x}",
                                subchannel, d
                            );
                        }
                        self.handle_gpu_method(subchannel, method, d, memory_writer);
                        pos += 1;
                    }
                }
                5 => {
                    // IncreaseOnce: first arg → method, rest → method+1.
                    let count = arg_count_or_data as usize;
                    for j in 0..count {
                        if pos >= dwords.len() {
                            break;
                        }
                        let m = if j == 0 { method } else { method + 1 };
                        let d = dwords[pos];
                        if m == 0 {
                            if d != 0 {
                                self.gpu_subchannel_class[subchannel] = d;
                            }
                            eprintln!(
                                "  [gpu] SET_OBJECT subchannel={} class=0x{:x}",
                                subchannel, d
                            );
                        }
                        self.handle_gpu_method(subchannel, m, d, memory_writer);
                        pos += 1;
                    }
                }
                _ => {
                    // Unknown mode — skip remaining args.
                    let count = arg_count_or_data as usize;
                    pos += count.min(dwords.len().saturating_sub(pos));
                }
            }
        }
    }

    /// Handle a single GPU method write.  Uses persistent subchannel→class
    /// bindings and query register state from `self`.
    fn handle_gpu_method(
        &mut self,
        subchannel: usize,
        method: u32,
        data: u32,
        memory_writer: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if method < 0x40 && method != 0 {
            match method {
                // 0x04: SemaphoreAddressHigh
                0x04 => {
                    self.gpu_puller_semaphore[0] = data;
                }
                // 0x05: SemaphoreAddressLow
                0x05 => {
                    self.gpu_puller_semaphore[1] = data;
                }
                // 0x06: SemaphoreSequencePayload
                0x06 => {
                    self.gpu_puller_semaphore[2] = data;
                }
                // 0x07: SemaphoreOperation (trigger)
                0x07 => {
                    let op = data & 0xF;
                    let gpu_addr = ((self.gpu_puller_semaphore[0] as u64) << 32)
                        | (self.gpu_puller_semaphore[1] as u64);
                    if op == 1 {
                        // WriteLong: write {payload, 0, timestamp_lo, timestamp_hi}
                        if gpu_addr != 0 {
                            if let Some(addr) = self.translate_gpu_va(gpu_addr) {
                                let payload = self.gpu_puller_semaphore[2];
                                let mut buf = [0u8; 16];
                                buf[0..4].copy_from_slice(&payload.to_le_bytes());
                                eprintln!(
                                    "  [gpu:puller] SEMAPHORE WRITE-LONG payload=0x{:x} to guest=0x{:x}",
                                    payload, addr
                                );
                                memory_writer(addr, &buf);
                            }
                        }
                    }
                    // Other ops (AcquireEqual=4, AcquireGequal=8, AcquireMask) are GPU-side waits; ignore.
                }
                // 0x1B: SemaphoreRelease — write payload to semaphore address
                0x1B => {
                    let gpu_addr = ((self.gpu_puller_semaphore[0] as u64) << 32)
                        | (self.gpu_puller_semaphore[1] as u64);
                    if gpu_addr != 0 {
                        if let Some(addr) = self.translate_gpu_va(gpu_addr) {
                            eprintln!(
                                "  [gpu:puller] SEMAPHORE RELEASE data=0x{:x} to guest=0x{:x}",
                                data, addr
                            );
                            memory_writer(addr, &data.to_le_bytes());
                        }
                    }
                }
                // 0x1C: SyncpointPayload
                0x1C => {
                    self.gpu_puller_syncpoint_payload = data;
                }
                // 0x1D: SyncpointOperation (FenceAction)
                0x1D => {
                    let op = data & 1; // 0=Acquire, 1=Increment
                    let _syncpoint_id = (data >> 4) & 0xFFF;
                    if op == 1 {
                        // Increment — just bump the fence value
                        self.gpu_fence_value = self.gpu_fence_value.wrapping_add(1);
                        eprintln!(
                            "  [gpu:puller] SYNCPOINT INCREMENT -> value={}",
                            self.gpu_fence_value
                        );
                    }
                }
                // Everything else (Nop, MemOp, WrcacheFlush, etc.) — ignore
                _ => {}
            }
            return;
        }

        let engine_class = self.gpu_subchannel_class[subchannel];

        // Maxwell 3D classes: 0xB197 (Maxwell B), 0xB097 (Maxwell A)
        let is_3d = engine_class == 0xB197 || engine_class == 0xB097;
        // Compute class: 0xB1C0 (Maxwell Compute A)
        let is_compute = engine_class == 0xB1C0;
        // 2D class: 0x902D or 0xB02D
        let is_2d = engine_class == 0x902D || engine_class == 0xB02D;
        // DMA Copy: 0xB0B5
        let is_dma = engine_class == 0xB0B5;
        // Inline2Memory: 0xA140
        let is_inline2mem = engine_class == 0xA140;

        // Log non-trivial methods on DMA/Inline2Mem engines (they write to memory).
        if is_dma && method != 0 {
            eprintln!("  [gpu:dma] method=0x{:04x} data=0x{:08x}", method, data);
        }
        if is_inline2mem && method != 0 {
            eprintln!("  [gpu:i2m] method=0x{:04x} data=0x{:08x}", method, data);
        }

        // Maxwell3D Query/Semaphore registers (only on 3D engine)
        if is_3d {
            match method {
                0x06C0 => {
                    self.gpu_query_3d[0] = data;
                }
                0x06C1 => {
                    self.gpu_query_3d[1] = data;
                }
                0x06C2 => {
                    self.gpu_query_3d[2] = data;
                }
                0x06C3 => {
                    // QUERY_GET — trigger the semaphore write.
                    let mode = data & 0x3;
                    let gpu_addr =
                        ((self.gpu_query_3d[0] as u64) << 32) | (self.gpu_query_3d[1] as u64);
                    if gpu_addr == 0 {
                        return;
                    }
                    let guest_addr = self.translate_gpu_va(gpu_addr);
                    if let Some(addr) = guest_addr {
                        let seq = self.gpu_query_3d[2];
                        match mode {
                            0 => {
                                eprintln!(
                                    "  [gpu:3d] QUERY WRITE seq=0x{:x} to gpu_va=0x{:x} guest=0x{:x}",
                                    seq, gpu_addr, addr
                                );
                                memory_writer(addr, &seq.to_le_bytes());
                            }
                            2 => {
                                eprintln!(
                                    "  [gpu:3d] QUERY WRITE-LONG seq=0x{:x} to gpu_va=0x{:x} guest=0x{:x}",
                                    seq, gpu_addr, addr
                                );
                                let mut payload = [0u8; 16];
                                payload[0..4].copy_from_slice(&seq.to_le_bytes());
                                memory_writer(addr, &payload);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // 2D engine semaphore (class 0x902D/0xB02D)
        if is_2d {
            match method {
                0x0060 => {
                    self.gpu_query_2d[0] = data;
                }
                0x0061 => {
                    self.gpu_query_2d[1] = data;
                }
                0x0064 => {
                    self.gpu_query_2d[2] = data;
                }
                0x0065 => {
                    // Semaphore trigger
                    let op = data & 0x3; // 1=release
                    if op == 1 {
                        let gpu_addr =
                            ((self.gpu_query_2d[0] as u64) << 32) | (self.gpu_query_2d[1] as u64);
                        if gpu_addr != 0 {
                            if let Some(addr) = self.translate_gpu_va(gpu_addr) {
                                let seq = self.gpu_query_2d[2];
                                eprintln!(
                                    "  [gpu:2d] SEMAPHORE RELEASE seq=0x{:x} to guest=0x{:x}",
                                    seq, addr
                                );
                                memory_writer(addr, &seq.to_le_bytes());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Compute engine semaphore (class 0xB1C0)
        if is_compute {
            match method {
                0x0110 => {
                    self.gpu_query_compute[0] = data;
                }
                0x0111 => {
                    self.gpu_query_compute[1] = data;
                }
                0x0112 => {
                    self.gpu_query_compute[2] = data;
                }
                0x0113 => {
                    let op = data & 0x3;
                    if op == 1 {
                        let gpu_addr = ((self.gpu_query_compute[0] as u64) << 32)
                            | (self.gpu_query_compute[1] as u64);
                        if gpu_addr != 0 {
                            if let Some(addr) = self.translate_gpu_va(gpu_addr) {
                                let seq = self.gpu_query_compute[2];
                                eprintln!(
                                    "  [gpu:compute] SEMAPHORE RELEASE seq=0x{:x} to guest=0x{:x}",
                                    seq, addr
                                );
                                memory_writer(addr, &seq.to_le_bytes());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // DMA copy engine semaphore (class 0xB0B5)
        if is_dma {
            match method {
                0x0240 => {
                    self.gpu_query_dma[0] = data;
                }
                0x0241 => {
                    self.gpu_query_dma[1] = data;
                }
                0x0244 => {
                    self.gpu_query_dma[2] = data;
                }
                0x0245 => {
                    let gpu_addr =
                        ((self.gpu_query_dma[0] as u64) << 32) | (self.gpu_query_dma[1] as u64);
                    if gpu_addr != 0 {
                        if let Some(addr) = self.translate_gpu_va(gpu_addr) {
                            let seq = self.gpu_query_dma[2];
                            eprintln!(
                                "  [gpu:dma] SEMAPHORE RELEASE seq=0x{:x} to guest=0x{:x}",
                                seq, addr
                            );
                            memory_writer(addr, &seq.to_le_bytes());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // =====================================================================
    // nvmap ioctls
    // =====================================================================

    fn handle_nvmap_ioctl(&mut self, kind: NvdrvIoctlKind, ioctl: NvIoctl, args: &mut [u8]) -> u32 {
        if ioctl.ty() != NV_IOCTL_NVMAP_CUSTOM_MAGIC {
            unimplemented!(
                "nvdrv {:?} ioctl 0x{:08x} has unsupported nvmap type 0x{:02x}",
                kind,
                ioctl.raw,
                ioctl.ty()
            );
        }

        match ioctl.number() {
            0x01 => self.nvmap_create(ioctl, args),
            0x03 => self.nvmap_from_id(args),
            0x04 => self.nvmap_alloc(args),
            0x05 => self.nvmap_free(args),
            0x09 => self.nvmap_param(args),
            0x0e => self.nvmap_get_id(args),
            _ => unimplemented!(
                "nvdrv {:?} nvmap ioctl number 0x{:02x} (raw=0x{:08x}, dir={}, size=0x{:x}) is not implemented",
                kind,
                ioctl.number(),
                ioctl.raw,
                ioctl.direction(),
                ioctl.size()
            ),
        }
    }

    fn handle_nvhost_ctrl_ioctl(
        &self,
        kind: NvdrvIoctlKind,
        ioctl: NvIoctl,
        args: &mut [u8],
    ) -> u32 {
        // nvhost-ctrl uses type 0x00 (NvHostCustomMagic) for its ioctls,
        // or type 0x48 (NvHostMagic) for some older paths.
        match ioctl.number() {
            // EventWait (0x1d, allocation mode) / EventWaitAsync (0x1e)
            // Struct: { u32 fence_id, u32 fence_value, u32 timeout, u32 value }
            // Stub: pretend syncpoint is already reached → return Success immediately.
            0x1d | 0x1e => {
                // Return the fence value as the current syncpoint value in the
                // value field (offset 12) to indicate the fence is satisfied.
                if args.len() >= 16 {
                    let fence_value = read_u32(args, 4).unwrap_or(0);
                    write_u32(args, 12, fence_value);
                }
                NVDRV_RESULT_SUCCESS
            }
            // EventSignal / ClearEventWait (0x1c)
            // Struct: { u32 event_id }
            0x1c => NVDRV_RESULT_SUCCESS,
            // EventRegister (0x1f): { u32 event_id }
            0x1f => NVDRV_RESULT_SUCCESS,
            // EventUnregister (0x20): { u32 event_id }
            0x20 => NVDRV_RESULT_SUCCESS,
            // EventUnregisterBatch (0x21): { u64 bitmask }
            0x21 => NVDRV_RESULT_SUCCESS,
            // GetConfig (0x1b): returns error in production mode
            0x1b => NVDRV_RESULT_INVALID_STATE,
            _ => {
                unimplemented!(
                    "nvdrv {:?} nvhost-ctrl ioctl number 0x{:02x} (raw=0x{:08x}) is not implemented",
                    kind,
                    ioctl.number(),
                    ioctl.raw
                )
            }
        }
    }

    fn handle_nvhost_ctrl_gpu_ioctl(
        &self,
        kind: NvdrvIoctlKind,
        ioctl: NvIoctl,
        args: &mut [u8],
        inline_buffer: &mut [u8],
    ) -> u32 {
        if ioctl.ty() != NV_IOCTL_NVGPU_MAGIC {
            unimplemented!(
                "nvdrv {:?} ioctl 0x{:08x} has unsupported nvgpu type 0x{:02x}",
                kind,
                ioctl.raw,
                ioctl.ty()
            );
        }

        match (kind, ioctl.number()) {
            (_, 0x01) => self.zcull_get_ctx_size(args),
            (_, 0x02) => self.zcull_get_info(args),
            (_, 0x03) => NVDRV_RESULT_SUCCESS,
            (NvdrvIoctlKind::Ioctl3, 0x05) => self.get_characteristics(args, Some(inline_buffer)),
            (_, 0x05) => self.get_characteristics(args, None),
            (NvdrvIoctlKind::Ioctl3, 0x06) => self.get_tpc_masks(args, Some(inline_buffer)),
            (_, 0x06) => self.get_tpc_masks(args, None),
            (_, 0x12) => self.num_vsms(args),
            (_, 0x13) => self.vsms_mapping(args),
            (_, 0x14) => self.get_active_slot_mask(args),
            (_, 0x1c) => self.get_gpu_time(args),
            _ => unimplemented!(
                "nvdrv {:?} nvhost-ctrl-gpu ioctl number 0x{:02x} (raw=0x{:08x}) is not implemented",
                kind,
                ioctl.number(),
                ioctl.raw
            ),
        }
    }

    fn nvmap_create(&mut self, ioctl: NvIoctl, args: &mut [u8]) -> u32 {
        let Some(size) = read_u32(args, 0).or_else(|| Some(ioctl.size())) else {
            return NVDRV_RESULT_INVALID_STATE;
        };
        if size == 0 {
            return NVDRV_RESULT_INVALID_STATE;
        }

        let aligned_size = align_up(size, PAGE_SIZE);
        let handle = self.allocate_nvmap_handle(aligned_size);
        if !write_i32(args, 4, handle) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn nvmap_from_id(&mut self, args: &mut [u8]) -> u32 {
        let Some(id) = read_i32(args, 0) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let Some(map) = self.nvmap_handles.get_mut(&id) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        map.refcount = map.refcount.saturating_add(1);
        if !write_i32(args, 4, id) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn nvmap_alloc(&mut self, args: &mut [u8]) -> u32 {
        let Some(handle) = read_i32(args, 0) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let Some(map) = self.nvmap_handles.get_mut(&handle) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let Some(mut align) = read_i32(args, 12) else {
            return NVDRV_RESULT_INVALID_STATE;
        };
        if align <= 0 || !is_power_of_two(align as u32) {
            return NVDRV_RESULT_INVALID_STATE;
        }
        if (align as u32) < PAGE_SIZE {
            align = PAGE_SIZE as i32;
        }

        if !map.allocated {
            map.allocated = true;
            map.align = align;
            map.kind = read_i64(args, 16).unwrap_or(0) as i32;

            let Some(address) = read_u64(args, 24) else {
                return NVDRV_RESULT_INVALID_STATE;
            };
            if address == 0 {
                return NVDRV_RESULT_INVALID_STATE;
            }

            map.address = address;
            map.size = align_up(map.size, PAGE_SIZE);
        }

        NVDRV_RESULT_SUCCESS
    }

    fn nvmap_free(&mut self, args: &mut [u8]) -> u32 {
        let Some(handle) = read_i32(args, 0) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let Some(existing) = self.nvmap_handles.get(&handle).cloned() else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let next_refcount = existing.refcount.saturating_sub(1);
        if !write_u32(args, 16, existing.size) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        if next_refcount <= 0 {
            if !write_u64(args, 8, existing.address) || !write_i32(args, 20, 0) {
                return NVDRV_RESULT_INVALID_STATE;
            }
            self.nvmap_handles.remove(&handle);
        } else {
            if let Some(map) = self.nvmap_handles.get_mut(&handle) {
                map.refcount = next_refcount;
            }
            if !write_u64(args, 8, 0) || !write_i32(args, 20, 1) {
                return NVDRV_RESULT_INVALID_STATE;
            }
        }

        NVDRV_RESULT_SUCCESS
    }

    fn nvmap_param(&self, args: &mut [u8]) -> u32 {
        let Some(handle) = read_i32(args, 0) else {
            return NVDRV_RESULT_INVALID_STATE;
        };
        let Some(param) = read_u32(args, 4) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let Some(map) = self.nvmap_handles.get(&handle) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        let value = match param {
            1 => map.size as i32,
            2 => map.align,
            4 => 0x4000_0000u32 as i32,
            5 => map.kind,
            6 => 0,
            _ => return NVDRV_RESULT_INVALID_STATE,
        };
        if !write_i32(args, 8, value) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn nvmap_get_id(&self, args: &mut [u8]) -> u32 {
        let Some(handle) = read_i32(args, 4) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        if !self.nvmap_handles.contains_key(&handle) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        if !write_i32(args, 0, handle) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn allocate_nvmap_handle(&mut self, size: u32) -> i32 {
        let mut handle = self.next_nvmap_handle.max(1);
        while self.nvmap_handles.contains_key(&handle) {
            handle = handle.saturating_add(1);
        }

        self.nvmap_handles.insert(
            handle,
            NvMapHandle {
                size,
                align: PAGE_SIZE as i32,
                kind: 0,
                address: 0,
                allocated: false,
                refcount: 1,
            },
        );
        self.next_nvmap_handle = handle.saturating_add(1);
        handle
    }

    fn zcull_get_ctx_size(&self, args: &mut [u8]) -> u32 {
        if !write_i32(args, 0, 1) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn zcull_get_info(&self, args: &mut [u8]) -> u32 {
        const ZCULL_VALUES: [i32; 10] =
            [0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10];
        for (index, value) in ZCULL_VALUES.iter().enumerate() {
            if !write_i32(args, index * 4, *value) {
                return NVDRV_RESULT_INVALID_STATE;
            }
        }

        NVDRV_RESULT_SUCCESS
    }

    fn get_characteristics(&self, args: &mut [u8], inline_out: Option<&mut [u8]>) -> u32 {
        if !write_i64(args, 0, 0xa0) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        if !write_gpu_characteristics_at(args, 16) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        if let Some(buffer) = inline_out
            && !buffer.is_empty()
            && !write_gpu_characteristics_at(buffer, 0)
        {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn get_tpc_masks(&self, args: &mut [u8], inline_out: Option<&mut [u8]>) -> u32 {
        let Some(mask_buffer_size) = read_i32(args, 0) else {
            return NVDRV_RESULT_INVALID_STATE;
        };

        if mask_buffer_size != 0 {
            if !write_i32(args, 16, 3) {
                return NVDRV_RESULT_INVALID_STATE;
            }
            if let Some(buffer) = inline_out
                && !buffer.is_empty()
                && !write_i32(buffer, 0, 3)
            {
                return NVDRV_RESULT_INVALID_STATE;
            }
        }

        NVDRV_RESULT_SUCCESS
    }

    fn num_vsms(&self, args: &mut [u8]) -> u32 {
        if !write_u32(args, 0, 2) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn vsms_mapping(&self, args: &mut [u8]) -> u32 {
        if !write_u8(args, 0, 0)
            || !write_u8(args, 1, 0)
            || !write_u8(args, 2, 0)
            || !write_u8(args, 3, 1)
        {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn get_active_slot_mask(&self, args: &mut [u8]) -> u32 {
        if !write_i32(args, 0, 0x07) || !write_i32(args, 4, 0x01) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    fn get_gpu_time(&self, args: &mut [u8]) -> u32 {
        let elapsed = self.gpu_boot_time.elapsed().as_nanos();
        let timestamp = (elapsed as u64) & 0x00ff_ffff_ffff_ffff;
        if !write_u64(args, 0, timestamp) {
            return NVDRV_RESULT_INVALID_STATE;
        }

        NVDRV_RESULT_SUCCESS
    }

    /// Returns all allocated NvMap handles that could be framebuffers.
    /// Returns Vec<(address, size)> of candidates sorted by size descending.
    pub fn framebuffer_candidates(&self) -> Vec<(u64, u32)> {
        let mut candidates: Vec<(u64, u32)> = self
            .nvmap_handles
            .values()
            .filter(|h| h.allocated && h.address != 0 && h.size >= 1280 * 720 * 4)
            .map(|h| (h.address, h.size))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates
    }

    pub fn nvmap_info(&self, handle: i32) -> Option<(u64, u32, i32)> {
        let map = self.nvmap_handles.get(&handle)?;
        if !map.allocated || map.address == 0 {
            return None;
        }

        Some((map.address, map.size, map.kind))
    }
}

fn write_gpu_characteristics_at(buffer: &mut [u8], base: usize) -> bool {
    // NvHostCtrlGpuDeviceFile.GetCharacteristics
    write_i32(buffer, base, 0x120)
        && write_i32(buffer, base + 0x04, 0xb)
        && write_i32(buffer, base + 0x08, 0xa1)
        && write_i32(buffer, base + 0x0c, 0x1)
        && write_i64(buffer, base + 0x10, 0x40000)
        && write_i64(buffer, base + 0x18, 0)
        && write_i32(buffer, base + 0x20, 0x2)
        && write_i32(buffer, base + 0x24, 0x20)
        && write_i32(buffer, base + 0x28, 0x20000)
        && write_i32(buffer, base + 0x2c, 0x20000)
        && write_i32(buffer, base + 0x30, 0x1b)
        && write_i32(buffer, base + 0x34, 0x30000)
        && write_i32(buffer, base + 0x38, 0x1)
        && write_i32(buffer, base + 0x3c, 0x503)
        && write_i32(buffer, base + 0x40, 0x503)
        && write_i32(buffer, base + 0x44, 0x80)
        && write_i32(buffer, base + 0x48, 0x28)
        && write_i32(buffer, base + 0x4c, 0)
        && write_i64(buffer, base + 0x50, 0x55)
        && write_i32(buffer, base + 0x58, 0x902d)
        && write_i32(buffer, base + 0x5c, 0xb197)
        && write_i32(buffer, base + 0x60, 0xb1c0)
        && write_i32(buffer, base + 0x64, 0xb06f)
        && write_i32(buffer, base + 0x68, 0xa140)
        && write_i32(buffer, base + 0x6c, 0xb0b5)
        && write_i32(buffer, base + 0x70, 0x1)
        && write_i32(buffer, base + 0x74, 0x0)
        && write_i32(buffer, base + 0x78, 0x2)
        && write_i32(buffer, base + 0x7c, 0x1)
        && write_i32(buffer, base + 0x80, 0x0)
        && write_i32(buffer, base + 0x84, 0x1)
        && write_i32(buffer, base + 0x88, 0x21d70)
        && write_i32(buffer, base + 0x8c, 0x0)
        && write_i64(buffer, base + 0x90, 0x6230_326d_67)
        && write_i64(buffer, base + 0x98, 0)
}

fn read_u16(payload: &[u8], offset: usize) -> Option<u16> {
    let bytes = payload.get(offset..offset.checked_add(2)?)?;
    let packed: [u8; 2] = bytes.try_into().ok()?;
    Some(u16::from_le_bytes(packed))
}

fn read_u32(payload: &[u8], offset: usize) -> Option<u32> {
    let bytes = payload.get(offset..offset.checked_add(4)?)?;
    let packed: [u8; 4] = bytes.try_into().ok()?;
    Some(u32::from_le_bytes(packed))
}

fn read_i32(payload: &[u8], offset: usize) -> Option<i32> {
    Some(read_u32(payload, offset)? as i32)
}

fn read_u64(payload: &[u8], offset: usize) -> Option<u64> {
    let bytes = payload.get(offset..offset.checked_add(8)?)?;
    let packed: [u8; 8] = bytes.try_into().ok()?;
    Some(u64::from_le_bytes(packed))
}

fn read_i64(payload: &[u8], offset: usize) -> Option<i64> {
    Some(read_u64(payload, offset)? as i64)
}

fn write_u8(payload: &mut [u8], offset: usize, value: u8) -> bool {
    let Some(slot) = payload.get_mut(offset) else {
        return false;
    };
    *slot = value;
    true
}

fn write_u32(payload: &mut [u8], offset: usize, value: u32) -> bool {
    let Some(bytes) = payload.get_mut(offset..offset.saturating_add(4)) else {
        return false;
    };
    bytes.copy_from_slice(&value.to_le_bytes());
    true
}

fn write_i32(payload: &mut [u8], offset: usize, value: i32) -> bool {
    write_u32(payload, offset, value as u32)
}

fn write_u64(payload: &mut [u8], offset: usize, value: u64) -> bool {
    let Some(bytes) = payload.get_mut(offset..offset.saturating_add(8)) else {
        return false;
    };
    bytes.copy_from_slice(&value.to_le_bytes());
    true
}

fn write_i64(payload: &mut [u8], offset: usize, value: i64) -> bool {
    write_u64(payload, offset, value as u64)
}

fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }

    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value.saturating_add(alignment.saturating_sub(remainder))
    }
}

fn align_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value.saturating_add(alignment.saturating_sub(remainder))
    }
}

fn is_power_of_two(value: u32) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

#[cfg(test)]
mod tests {
    use super::{
        NV_IOCTL_NVGPU_MAGIC, NV_IOCTL_NVMAP_CUSTOM_MAGIC, NVDRV_RESULT_INVALID_STATE,
        NVDRV_RESULT_SUCCESS, NvdrvIoctlKind, NvdrvService, read_i32, read_u32,
    };

    fn encode_ioctl(direction: u32, size: u32, ty: u32, number: u32) -> u32 {
        ((direction & 0x3) << 30) | ((size & 0x3fff) << 16) | ((ty & 0xff) << 8) | (number & 0xff)
    }

    #[test]
    fn open_close_and_query_event_track_fd_lifecycle() {
        let mut nvdrv = NvdrvService::default();
        let fd = nvdrv
            .open_device("/dev/nvhost-ctrl-gpu")
            .expect("open should allocate an fd");
        assert!(fd > 0);
        assert_eq!(
            nvdrv.query_event_command(fd, 1),
            Some(4 | (1 << 16)),
            "valid event id should produce query-event command key"
        );

        assert!(nvdrv.close_device(fd));
        assert_eq!(nvdrv.query_event_command(fd, 1), None);
    }

    #[test]
    fn unknown_path_is_rejected() {
        let mut nvdrv = NvdrvService::default();
        assert_eq!(nvdrv.open_device("/dev/unknown"), None);
    }

    #[test]
    fn invalid_fd_ioctl_reports_invalid_state() {
        let mut nvdrv = NvdrvService::default();
        assert_eq!(
            nvdrv.ioctl_result(NvdrvIoctlKind::Ioctl, 99, 0xc008_0101, &mut [], &mut []),
            NVDRV_RESULT_INVALID_STATE
        );
    }

    #[test]
    fn nvmap_create_alloc_and_get_id_follow_ryujinx_layout() {
        let mut nvdrv = NvdrvService::default();
        let fd = nvdrv
            .open_device("/dev/nvmap")
            .expect("nvmap should open successfully");

        let create_ioctl = encode_ioctl(0b11, 8, NV_IOCTL_NVMAP_CUSTOM_MAGIC, 0x01);
        let mut create_args = [0u8; 8];
        create_args[..4].copy_from_slice(&0x2000u32.to_le_bytes());
        assert_eq!(
            nvdrv.ioctl_result(
                NvdrvIoctlKind::Ioctl,
                fd,
                create_ioctl,
                &mut create_args,
                &mut []
            ),
            NVDRV_RESULT_SUCCESS
        );
        let handle = read_i32(&create_args, 4).expect("create should write handle");
        assert!(handle > 0);

        let alloc_ioctl = encode_ioctl(0b11, 0x20, NV_IOCTL_NVMAP_CUSTOM_MAGIC, 0x04);
        let mut alloc_args = [0u8; 0x20];
        alloc_args[..4].copy_from_slice(&(handle as u32).to_le_bytes());
        alloc_args[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
        alloc_args[24..32].copy_from_slice(&0x817d_1870u64.to_le_bytes());
        assert_eq!(
            nvdrv.ioctl_result(
                NvdrvIoctlKind::Ioctl,
                fd,
                alloc_ioctl,
                &mut alloc_args,
                &mut []
            ),
            NVDRV_RESULT_SUCCESS
        );

        let get_id_ioctl = encode_ioctl(0b11, 8, NV_IOCTL_NVMAP_CUSTOM_MAGIC, 0x0e);
        let mut get_id_args = [0u8; 8];
        get_id_args[4..8].copy_from_slice(&(handle as u32).to_le_bytes());
        assert_eq!(
            nvdrv.ioctl_result(
                NvdrvIoctlKind::Ioctl,
                fd,
                get_id_ioctl,
                &mut get_id_args,
                &mut []
            ),
            NVDRV_RESULT_SUCCESS
        );
        assert_eq!(
            read_i32(&get_id_args, 0).expect("get_id should write id"),
            handle
        );
    }

    #[test]
    fn nvhost_ctrl_gpu_get_characteristics_matches_ryujinx_constants() {
        let mut nvdrv = NvdrvService::default();
        let fd = nvdrv
            .open_device("/dev/nvhost-ctrl-gpu")
            .expect("ctrl-gpu should open successfully");

        let ioctl = encode_ioctl(0b11, 0xb0, NV_IOCTL_NVGPU_MAGIC, 0x05);
        let mut args = [0u8; 0xb0];
        let mut inline_out = [0u8; 0xa0];
        assert_eq!(
            nvdrv.ioctl_result(
                NvdrvIoctlKind::Ioctl3,
                fd,
                ioctl,
                &mut args,
                &mut inline_out
            ),
            NVDRV_RESULT_SUCCESS
        );

        assert_eq!(
            read_u32(&args, 0).expect("header low word should exist"),
            0xa0,
            "buffer size should be 0xa0"
        );
        assert_eq!(
            read_i32(&inline_out, 0).expect("arch should exist"),
            0x120,
            "arch constant should match Ryujinx"
        );
        assert_eq!(
            read_i32(&inline_out, 4).expect("impl should exist"),
            0xb,
            "impl constant should match Ryujinx"
        );
    }
}
