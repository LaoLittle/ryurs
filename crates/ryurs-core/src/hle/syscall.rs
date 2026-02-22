use ryurs_dynarec::{CpuState, MemoryInterface, virtual_counter_ticks};
use ryurs_gpu_wgpu::nvdrv::{NVDRV_RESULT_INVALID_STATE, NVDRV_RESULT_SUCCESS, NvdrvIoctlKind};
use ryurs_macros::svc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::kernel::{
    DomainObjectKind, KernelState, MAIN_PROCESS_HANDLE, MAIN_THREAD_HANDLE, SELF_PROCESS_HANDLE,
    SELF_THREAD_HANDLE, ServiceSessionKind, ThreadCoreMaskError, ViSessionRole,
};
use super::service_manager::ServiceManager;
use crate::emulator::DisplayFrame;
use crate::memory::SharedGuestMemory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallDisposition {
    Continue,
    ThreadExit,
    Unhandled,
}

pub type SvcResult = SyscallDisposition;
pub type SvcHandler = for<'a> fn(&mut SvcContext<'a>) -> SvcResult;

#[derive(Clone, Copy)]
pub struct SvcService {
    pub svc_number: u16,
    pub name: &'static str,
    pub handler: SvcHandler,
}

pub struct SvcContext<'a> {
    pub kernel: &'a mut KernelState,
    pub current_thread_handle: u32,
    pub state: &'a mut CpuState,
    pub memory: &'a SharedGuestMemory,
}

const RESULT_SUCCESS: u64 = 0;
const RESULT_INVALID_HANDLE: u64 = 0x1;
const RESULT_INVALID_MEMORY_STATE: u64 = 0x2;
const RESULT_INVALID_COMBINATION: u64 = 0x3;
const RESULT_INVALID_SIZE: u64 = 0xca01;
const RESULT_OUT_OF_MEMORY: u64 = 0xd001;
const RESULT_TIMED_OUT: u64 = 0xea01;
const RESULT_NOT_FOUND: u64 = 0xf201;
const RESULT_AM_NO_MESSAGES: u32 = 0x680;
const MAX_DEBUG_STRING_BYTES: usize = 0x10000;
const MAX_SERVICE_PAYLOAD_BYTES: u64 = 0x400;
const DEFAULT_STACK_REGION_SIZE: u64 = 0x10_0000;
const PAGE_SIZE: u64 = 0x1000;
const MEMORY_STATE_SHARED: u32 = 0x6;
const KERNEL_ASLR_REGION_SIZE_64BIT: u64 = 0x7ff8_0000_00;
const HIPC_TYPE_CMIF_CLOSE: u16 = 2;
const HIPC_TYPE_CMIF_REQUEST: u16 = 4;
const HIPC_TYPE_CMIF_CONTROL: u16 = 5;
const HIPC_TYPE_CMIF_REQUEST_WITH_CONTEXT: u16 = 6;
const HIPC_TYPE_CMIF_CONTROL_WITH_CONTEXT: u16 = 7;
const CMIF_IN_HEADER_MAGIC: u32 = 0x4943_4653;
const CMIF_OUT_HEADER_MAGIC: u32 = 0x4f43_4653;
const CMIF_HEADER_SIZE: u64 = 16;
const DOMAIN_MESSAGE_HEADER_SIZE: u64 = 16;
const SM_POINTER_BUFFER_SIZE: u16 = 0x8000;
const MAX_SERVICES: usize = 0x100;
const HANDLE_WAIT_MASK: u32 = 0x4000_0000;
const REPORTED_TOTAL_MEMORY_SIZE: u64 = 0xc000_0000;
const HID_SHARED_MEMORY_SIZE: u64 = 0x40_000;
const HID_SHARED_MEMORY_PERMISSION: u32 = 1;
const TIME_SHARED_MEMORY_SIZE: u64 = 0x1000;
const TIME_SHARED_MEMORY_PERMISSION: u32 = 1;

fn create_hid_shared_memory_handle(
    kernel: &mut KernelState,
    current_thread_handle: u32,
) -> Option<u32> {
    kernel.create_shared_memory_handle(
        current_thread_handle,
        HID_SHARED_MEMORY_SIZE,
        HID_SHARED_MEMORY_PERMISSION,
    )
}

fn create_time_shared_memory_handle(
    kernel: &mut KernelState,
    current_thread_handle: u32,
) -> Option<u32> {
    kernel.create_shared_memory_handle(
        current_thread_handle,
        TIME_SHARED_MEMORY_SIZE,
        TIME_SHARED_MEMORY_PERMISSION,
    )
}

fn write_query_memory_info(
    memory: &SharedGuestMemory,
    info_ptr: u64,
    page_info_ptr: u64,
    query_address: u64,
) -> bool {
    let info = memory.query_memory(query_address);

    memory.write_u64(info_ptr, info.address)
        && memory.write_u64(info_ptr.wrapping_add(8), info.size)
        && memory.write_u32(info_ptr.wrapping_add(16), info.state)
        && memory.write_u32(info_ptr.wrapping_add(20), info.attribute)
        && memory.write_u32(info_ptr.wrapping_add(24), info.permission)
        && memory.write_u32(info_ptr.wrapping_add(28), info.ipc_ref_count)
        && memory.write_u32(info_ptr.wrapping_add(32), info.device_ref_count)
        && memory.write_u32(info_ptr.wrapping_add(36), 0)
        && memory.write_u32(page_info_ptr, 0)
}

fn map_memory_bytes(
    memory: &SharedGuestMemory,
    dst_address: u64,
    src_address: u64,
    size: u64,
) -> bool {
    let mut view = memory.locked_view();
    let mut offset = 0u64;
    while offset < size {
        let src = src_address.wrapping_add(offset);
        let dst = dst_address.wrapping_add(offset);
        let Some(value) = view.read_u8(src) else {
            return false;
        };
        if !view.write_u8(dst, value) {
            return false;
        }
        offset = offset.saturating_add(1);
    }
    true
}

fn query_memory_info(
    memory: &SharedGuestMemory,
    query_address: u64,
) -> crate::memory::MemoryQueryInfo {
    memory.query_memory(query_address)
}

fn process_used_memory_size(
    memory: &SharedGuestMemory,
    address_space_base: u64,
    address_space_size: u64,
) -> u64 {
    let address_space_end = address_space_base.saturating_add(address_space_size);
    let mut cursor = address_space_base;
    let mut used = 0u64;

    while cursor < address_space_end {
        let info = query_memory_info(memory, cursor);
        let region_start = info.address.max(cursor);
        let region_end = info
            .address
            .saturating_add(info.size)
            .min(address_space_end);

        if region_end <= region_start {
            let next = cursor.saturating_add(PAGE_SIZE);
            if next <= cursor {
                break;
            }
            cursor = next;
            continue;
        }

        if info.state != 0 {
            used = used.saturating_add(region_end.saturating_sub(region_start));
        }

        cursor = region_end;
    }

    used
}

fn reported_total_memory_size(address_space_size: u64) -> u64 {
    address_space_size.max(REPORTED_TOTAL_MEMORY_SIZE)
}

fn release_mutex_to_next_waiter(
    kernel: &mut KernelState,
    current_thread_handle: u32,
    memory: &SharedGuestMemory,
    mutex_address: u64,
) -> u64 {
    let next_owner = kernel
        .take_waiters_for_address(current_thread_handle, mutex_address, 1)
        .into_iter()
        .next();

    let mut next_value = 0u32;
    if let Some(waiter_thread_handle) = next_owner
        && let Some((_, waiter_tag)) = kernel.thread_address_key(waiter_thread_handle)
    {
        next_value = waiter_tag;
        if kernel.waiter_count_for_address(current_thread_handle, mutex_address) > 0 {
            next_value |= HANDLE_WAIT_MASK;
        }
    } else if kernel.main_thread_yield_mutex_address() == Some(mutex_address) {
        // The main thread is yield-waiting on this exact mutex.
        // Transfer ownership directly to avoid the race where the
        // releasing thread re-acquires before the yield-poll sees 0.
        //
        // Write 0 (free) so the guest's mutexLock CAS
        //   `__sync_val_compare_and_swap(m, 0, self)` succeeds on the
        // first retry.  Writing MAIN_THREAD_HANDLE would leave a
        // non-zero word that the CAS can never match, causing the main
        // thread to spin forever in its mutexLock loop.
        next_value = 0;
        kernel.clear_main_thread_yield();
        eprintln!(
            "[mutex_transfer] transferred mutex 0x{mutex_address:016x} to main thread (was held by handle {current_thread_handle})"
        );
    }

    if !memory.write_u32(mutex_address, next_value) {
        if let Some(waiter_thread_handle) = next_owner {
            let _ = kernel.wake_thread(waiter_thread_handle);
        }
        return RESULT_INVALID_MEMORY_STATE;
    }

    if let Some(waiter_thread_handle) = next_owner {
        let _ = kernel.wake_thread(waiter_thread_handle);
    }

    RESULT_SUCCESS
}

fn read_guest_buffer(memory: &SharedGuestMemory, address: u64, size: u64) -> Option<Vec<u8>> {
    let size = usize::try_from(size).ok()?;
    if size > MAX_DEBUG_STRING_BYTES {
        return None;
    }

    let mut view = memory.locked_view();
    let mut bytes = Vec::with_capacity(size);
    for offset in 0..size {
        bytes.push(view.read_u8(address.wrapping_add(offset as u64))?);
    }

    Some(bytes)
}

fn read_guest_c_string(memory: &SharedGuestMemory, address: u64, max_len: usize) -> Option<String> {
    let bytes = read_guest_buffer(memory, address, max_len as u64)?;
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    let ascii = &bytes[..end];
    if !ascii.is_ascii() {
        return None;
    }

    Some(String::from_utf8_lossy(ascii).to_string())
}

#[derive(Debug, Clone, Copy)]
struct HipcRequestInfo {
    message_type: u16,
    data_start: u64,
    data_end: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HipcBufferDescriptor {
    address: u64,
    size: u64,
}

#[derive(Debug, Clone, Default)]
struct HipcBufferLayout {
    send_statics: Vec<HipcBufferDescriptor>,
    send_buffers: Vec<HipcBufferDescriptor>,
    recv_buffers: Vec<HipcBufferDescriptor>,
    exch_buffers: Vec<HipcBufferDescriptor>,
    /// C descriptors (receive-list entries) parsed from after the data words.
    recv_statics: Vec<HipcBufferDescriptor>,
}

fn align_up_16(value: u64) -> u64 {
    (value + 15) & !15
}

fn parse_hipc_request(memory: &SharedGuestMemory, tls: u64) -> Option<HipcRequestInfo> {
    let word0 = memory.read_u32(tls)?;
    let word1 = memory.read_u32(tls.wrapping_add(4))?;

    let message_type = (word0 & 0xffff) as u16;
    let num_send_statics = ((word0 >> 16) & 0xf) as u64;
    let num_send_buffers = ((word0 >> 20) & 0xf) as u64;
    let num_recv_buffers = ((word0 >> 24) & 0xf) as u64;
    let num_exch_buffers = ((word0 >> 28) & 0xf) as u64;
    let num_data_words = (word1 & 0x3ff) as u64;
    let has_special_header = ((word1 >> 31) & 1) != 0;

    let mut cursor = tls.wrapping_add(8);
    if has_special_header {
        let special = memory.read_u32(cursor)?;
        cursor = cursor.wrapping_add(4);

        if (special & 1) != 0 {
            cursor = cursor.wrapping_add(8);
        }

        let num_copy_handles = ((special >> 1) & 0xf) as u64;
        let num_move_handles = ((special >> 5) & 0xf) as u64;
        cursor = cursor.wrapping_add((num_copy_handles + num_move_handles).saturating_mul(4));
    }

    let descriptor_bytes = num_send_statics.saturating_mul(8)
        + (num_send_buffers + num_recv_buffers + num_exch_buffers).saturating_mul(12);
    let data_words_address = cursor.wrapping_add(descriptor_bytes);
    let data_end = data_words_address.wrapping_add(num_data_words.saturating_mul(4));
    let data_start = if num_data_words == 0 {
        data_words_address
    } else {
        let aligned = align_up_16(data_words_address);
        if aligned > data_end {
            return None;
        }
        aligned
    };

    Some(HipcRequestInfo {
        message_type,
        data_start,
        data_end,
    })
}

fn decode_hipc_static_descriptor(word0: u32, word1: u32) -> HipcBufferDescriptor {
    let word0 = u64::from(word0);
    let word1 = u64::from(word1);
    let mut address = word1;
    address |= (word0 << 20) & 0x0f00_0000_00;
    address |= (word0 << 30) & 0x7000_0000_00;

    HipcBufferDescriptor {
        address,
        size: (word0 >> 16) & 0xffff,
    }
}

fn decode_hipc_buffer_descriptor(word0: u32, word1: u32, word2: u32) -> HipcBufferDescriptor {
    let word0 = u64::from(word0);
    let word1 = u64::from(word1);
    let word2 = u64::from(word2);
    let mut address = word1;
    address |= (word2 << 4) & 0x0f00_0000_00;
    address |= (word2 << 34) & 0x7000_0000_00;

    HipcBufferDescriptor {
        address,
        size: word0 | ((word2 << 8) & 0x0f00_0000_00),
    }
}

fn parse_hipc_buffer_layout(memory: &SharedGuestMemory, tls: u64) -> Option<HipcBufferLayout> {
    let word0 = memory.read_u32(tls)?;
    let word1 = memory.read_u32(tls.wrapping_add(4))?;

    let num_send_statics = ((word0 >> 16) & 0xf) as usize;
    let num_send_buffers = ((word0 >> 20) & 0xf) as usize;
    let num_recv_buffers = ((word0 >> 24) & 0xf) as usize;
    let num_exch_buffers = ((word0 >> 28) & 0xf) as usize;
    let num_data_words = (word1 & 0x3ff) as u64;
    let recv_static_mode = ((word1 >> 10) & 0xf) as usize;
    let has_special_header = ((word1 >> 31) & 1) != 0;

    // Decode recv_static_mode → actual C descriptor count.
    // Mode 0 = none, mode 2 = 1 (auto/single), mode > 2 = mode - 2.
    let num_recv_statics = match recv_static_mode {
        0 | 1 => 0,
        2 => 1,
        n => n - 2,
    };

    let mut cursor = tls.wrapping_add(8);
    if has_special_header {
        let special = memory.read_u32(cursor)?;
        cursor = cursor.wrapping_add(4);

        if (special & 1) != 0 {
            cursor = cursor.wrapping_add(8);
        }

        let num_copy_handles = ((special >> 1) & 0xf) as u64;
        let num_move_handles = ((special >> 5) & 0xf) as u64;
        cursor = cursor.wrapping_add((num_copy_handles + num_move_handles).saturating_mul(4));
    }

    let mut layout = HipcBufferLayout::default();
    layout.send_statics.reserve(num_send_statics);
    layout.send_buffers.reserve(num_send_buffers);
    layout.recv_buffers.reserve(num_recv_buffers);
    layout.exch_buffers.reserve(num_exch_buffers);
    layout.recv_statics.reserve(num_recv_statics);

    for _ in 0..num_send_statics {
        let word0 = memory.read_u32(cursor)?;
        let word1 = memory.read_u32(cursor.wrapping_add(4))?;
        cursor = cursor.wrapping_add(8);
        layout
            .send_statics
            .push(decode_hipc_static_descriptor(word0, word1));
    }

    for _ in 0..num_send_buffers {
        let word0 = memory.read_u32(cursor)?;
        let word1 = memory.read_u32(cursor.wrapping_add(4))?;
        let word2 = memory.read_u32(cursor.wrapping_add(8))?;
        cursor = cursor.wrapping_add(12);
        layout
            .send_buffers
            .push(decode_hipc_buffer_descriptor(word0, word1, word2));
    }

    for _ in 0..num_recv_buffers {
        let word0 = memory.read_u32(cursor)?;
        let word1 = memory.read_u32(cursor.wrapping_add(4))?;
        let word2 = memory.read_u32(cursor.wrapping_add(8))?;
        cursor = cursor.wrapping_add(12);
        layout
            .recv_buffers
            .push(decode_hipc_buffer_descriptor(word0, word1, word2));
    }

    for _ in 0..num_exch_buffers {
        let word0 = memory.read_u32(cursor)?;
        let word1 = memory.read_u32(cursor.wrapping_add(4))?;
        let word2 = memory.read_u32(cursor.wrapping_add(8))?;
        cursor = cursor.wrapping_add(12);
        layout
            .exch_buffers
            .push(decode_hipc_buffer_descriptor(word0, word1, word2));
    }

    // C descriptors (receive-list entries) appear immediately after data words.
    // Skip past the data words section to reach them.
    let data_words_address = cursor;
    let recv_statics_start = data_words_address.wrapping_add(num_data_words.saturating_mul(4));
    for i in 0..num_recv_statics {
        let entry_addr = recv_statics_start.wrapping_add((i as u64).saturating_mul(8));
        let packed = memory.read_u64(entry_addr)?;
        let address = packed & 0x0000_ffff_ffff_ffff; // lower 48 bits
        let size = packed >> 48; // upper 16 bits
        layout
            .recv_statics
            .push(HipcBufferDescriptor { address, size });
    }

    Some(layout)
}

fn descriptor_has_data(descriptor: HipcBufferDescriptor) -> bool {
    descriptor.address != 0 && descriptor.size != 0
}

fn read_descriptor_bytes(
    memory: &SharedGuestMemory,
    descriptor: HipcBufferDescriptor,
) -> Option<Vec<u8>> {
    if !descriptor_has_data(descriptor) {
        return None;
    }

    let size = descriptor.size.min(MAX_DEBUG_STRING_BYTES as u64);
    read_guest_buffer(memory, descriptor.address, size)
}

fn write_guest_buffer(memory: &SharedGuestMemory, address: u64, bytes: &[u8]) -> bool {
    let mut view = memory.locked_view();
    for (index, byte) in bytes.iter().enumerate() {
        let offset = index as u64;
        if !view.write_u8(address.wrapping_add(offset), *byte) {
            return false;
        }
    }

    true
}

fn parse_cmif_command_id(memory: &SharedGuestMemory, request: HipcRequestInfo) -> Option<u32> {
    if request.data_start.wrapping_add(CMIF_HEADER_SIZE) <= request.data_end
        && memory.read_u32(request.data_start)? == CMIF_IN_HEADER_MAGIC
    {
        return memory.read_u32(request.data_start.wrapping_add(8));
    }

    let domain_payload_start = request.data_start.wrapping_add(DOMAIN_MESSAGE_HEADER_SIZE);
    if domain_payload_start.wrapping_add(CMIF_HEADER_SIZE) <= request.data_end
        && memory.read_u32(domain_payload_start)? == CMIF_IN_HEADER_MAGIC
    {
        return memory.read_u32(domain_payload_start.wrapping_add(8));
    }

    None
}

fn has_domain_message_header(memory: &SharedGuestMemory, request: HipcRequestInfo) -> bool {
    if request.data_start.wrapping_add(DOMAIN_MESSAGE_HEADER_SIZE) > request.data_end {
        return false;
    }

    // Domain close uses only the 8-byte domain header (no CMIF in-header magic).
    if let Some(header) = memory.read_u32(request.data_start)
        && ((header & 0xff) as u8) == 2
    {
        return true;
    }

    let domain_payload_start = request.data_start.wrapping_add(DOMAIN_MESSAGE_HEADER_SIZE);
    domain_payload_start.wrapping_add(CMIF_HEADER_SIZE) <= request.data_end
        && memory.read_u32(domain_payload_start) == Some(CMIF_IN_HEADER_MAGIC)
}

fn parse_domain_header(memory: &SharedGuestMemory, request: HipcRequestInfo) -> Option<(u8, u32)> {
    if !has_domain_message_header(memory, request) {
        return None;
    }

    let header = memory.read_u32(request.data_start)?;
    let object_id = memory.read_u32(request.data_start.wrapping_add(4))?;
    Some(((header & 0xff) as u8, object_id))
}

fn read_service_name_8(memory: &SharedGuestMemory, address: u64) -> Option<String> {
    let bytes = read_guest_buffer(memory, address, 8)?;

    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(8);
    let ascii = &bytes[..end];
    if !ascii.is_ascii() {
        return None;
    }

    Some(String::from_utf8_lossy(ascii).to_string())
}

fn cmif_payload_start(request: HipcRequestInfo, is_domain_message: bool) -> Option<u64> {
    let cmif_header_start = if is_domain_message {
        request.data_start.wrapping_add(DOMAIN_MESSAGE_HEADER_SIZE)
    } else {
        request.data_start
    };
    let payload_start = cmif_header_start.wrapping_add(CMIF_HEADER_SIZE);
    (payload_start <= request.data_end).then_some(payload_start)
}

fn read_cmif_payload_bytes(
    memory: &SharedGuestMemory,
    request: HipcRequestInfo,
    is_domain_message: bool,
) -> Vec<u8> {
    let Some(payload_start) = cmif_payload_start(request, is_domain_message) else {
        return Vec::new();
    };
    let payload_size = request
        .data_end
        .saturating_sub(payload_start)
        .min(MAX_SERVICE_PAYLOAD_BYTES);

    read_guest_buffer(memory, payload_start, payload_size).unwrap_or_default()
}

fn consume_cmif_payload(
    memory: &SharedGuestMemory,
    request: HipcRequestInfo,
    is_domain_message: bool,
) {
    let _ = read_cmif_payload_bytes(memory, request, is_domain_message);
}

fn read_payload_u32(payload: &[u8], offset: usize) -> Option<u32> {
    let bytes = payload.get(offset..offset.checked_add(4)?)?;
    let packed: [u8; 4] = bytes.try_into().ok()?;
    Some(u32::from_le_bytes(packed))
}

fn read_payload_i32(payload: &[u8], offset: usize) -> Option<i32> {
    Some(read_payload_u32(payload, offset)? as i32)
}

fn read_payload_u64(payload: &[u8], offset: usize) -> Option<u64> {
    let bytes = payload.get(offset..offset.checked_add(8)?)?;
    let packed: [u8; 8] = bytes.try_into().ok()?;
    Some(u64::from_le_bytes(packed))
}

fn read_ascii_cstr_at(payload: &[u8], start: usize) -> Option<String> {
    let tail = payload.get(start..)?;
    let end = tail
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(tail.len());
    let candidate = tail.get(..end)?;
    if candidate.is_empty() || !candidate.is_ascii() {
        return None;
    }

    Some(String::from_utf8_lossy(candidate).to_string())
}

fn nvdrv_device_path_from_payload(payload: &[u8]) -> String {
    for start in [0usize, 4, 8, 16] {
        if let Some(path) = read_ascii_cstr_at(payload, start)
            && !path.is_empty()
        {
            return path;
        }
    }

    "nvdrv:unknown".to_string()
}

fn nvdrv_device_path_from_descriptor(
    memory: &SharedGuestMemory,
    descriptor: HipcBufferDescriptor,
) -> Option<String> {
    if !descriptor_has_data(descriptor) {
        return None;
    }

    let max_len = usize::try_from(descriptor.size.min(MAX_DEBUG_STRING_BYTES as u64)).ok()?;
    let raw = descriptor.address;
    let candidates = [
        raw,
        raw & !0x0100_0000,
        raw & !0x8000_0000,
        raw & !0x8100_0000,
        raw | 0x8000_0000,
    ];

    for address in candidates {
        if let Some(path) = read_guest_c_string(memory, address, max_len)
            && path.starts_with("/dev/")
        {
            return Some(path);
        }
    }

    None
}

fn format_u32_words(memory: &SharedGuestMemory, address: u64, words: usize) -> String {
    (0..words)
        .map(|index| {
            format!(
                "{:08x}",
                memory
                    .read_u32(address.wrapping_add((index as u64) * 4))
                    .unwrap_or(u32::MAX)
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_hex_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Clone)]
struct HipcSpecialHeaderDump {
    raw: u32,
    pid: Option<u64>,
    copy_handles: Vec<u32>,
    move_handles: Vec<u32>,
}

fn parse_hipc_special_header(
    memory: &SharedGuestMemory,
    tls: u64,
) -> Option<HipcSpecialHeaderDump> {
    let word1 = memory.read_u32(tls.wrapping_add(4))?;
    if ((word1 >> 31) & 1) == 0 {
        return None;
    }

    let raw = memory.read_u32(tls.wrapping_add(8))?;
    let mut cursor = tls.wrapping_add(12);

    let pid = if (raw & 1) != 0 {
        let lo = memory.read_u32(cursor)? as u64;
        let hi = memory.read_u32(cursor.wrapping_add(4))? as u64;
        cursor = cursor.wrapping_add(8);
        Some(lo | (hi << 32))
    } else {
        None
    };

    let copy_count = ((raw >> 1) & 0xf) as usize;
    let move_count = ((raw >> 5) & 0xf) as usize;

    let mut copy_handles = Vec::with_capacity(copy_count);
    for _ in 0..copy_count {
        copy_handles.push(memory.read_u32(cursor).unwrap_or(u32::MAX));
        cursor = cursor.wrapping_add(4);
    }

    let mut move_handles = Vec::with_capacity(move_count);
    for _ in 0..move_count {
        move_handles.push(memory.read_u32(cursor).unwrap_or(u32::MAX));
        cursor = cursor.wrapping_add(4);
    }

    Some(HipcSpecialHeaderDump {
        raw,
        pid,
        copy_handles,
        move_handles,
    })
}

fn should_trace_domain_command(session_kind: ServiceSessionKind, command_id: u32) -> bool {
    if !trace_service_traffic_enabled() {
        return false;
    }

    match session_kind {
        ServiceSessionKind::AppletOE => {
            matches!(
                command_id,
                0x00 | 0x01 | 0x0b | 0x14 | 0x16 | 0x384 | 0x5b | 0x82
            )
        }
        // Friend flow is central to the current crash triage, so trace all domain commands.
        ServiceSessionKind::Friend => true,
        // Keep bootstrap services verbose while triaging early-failure paths.
        ServiceSessionKind::Pctl | ServiceSessionKind::AocU | ServiceSessionKind::EctxAw => true,
        _ => false,
    }
}

fn should_trace_non_domain_command(session_kind: ServiceSessionKind, _command_id: u32) -> bool {
    if !trace_service_traffic_enabled() {
        return false;
    }

    matches!(
        session_kind,
        ServiceSessionKind::Friend
            | ServiceSessionKind::AppletOE
            | ServiceSessionKind::ViU
            | ServiceSessionKind::Nvdrv
    )
}

fn unsupported_control_command(session_kind: ServiceSessionKind, command_id: u32) -> u32 {
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}] STUB unsupported control command id=0x{command_id:08x}"
    );
    0x0001
}

fn unsupported_domain_command(
    session_kind: ServiceSessionKind,
    object_kind: DomainObjectKind,
    command_id: u32,
) -> u32 {
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}] STUB unsupported domain command object={object_kind:?} id=0x{command_id:08x}"
    );
    // Return a generic "not supported" error instead of panicking so the
    // guest can potentially handle the failure gracefully.
    0x0001 // ResultNotImplemented-ish; many HOS services accept this.
}

fn unsupported_non_domain_command(session_kind: ServiceSessionKind, command_id: u32) -> u32 {
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}] STUB unsupported non-domain command id=0x{command_id:08x}"
    );
    0x0001
}

fn fsp_global_access_log_mode() -> &'static AtomicU32 {
    static MODE: OnceLock<AtomicU32> = OnceLock::new();
    MODE.get_or_init(|| AtomicU32::new(0))
}

fn vi_next_stray_layer_id() -> u64 {
    static NEXT_LAYER_ID: OnceLock<AtomicU64> = OnceLock::new();
    NEXT_LAYER_ID
        .get_or_init(|| AtomicU64::new(1))
        .fetch_add(1, Ordering::AcqRel)
}

fn vi_next_binder_id() -> u32 {
    static NEXT_BINDER_ID: OnceLock<AtomicU32> = OnceLock::new();
    NEXT_BINDER_ID
        .get_or_init(|| AtomicU32::new(1))
        .fetch_add(1, Ordering::AcqRel)
}

fn vi_write_parcel_to_out_buffer(
    memory: &SharedGuestMemory,
    tls: u64,
    payload: &[u8],
) -> Option<u64> {
    let descriptor_layout = parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
    let out_descriptor = descriptor_layout
        .recv_buffers
        .iter()
        .copied()
        .find(|descriptor| descriptor_has_data(*descriptor))
        .or_else(|| {
            descriptor_layout
                .exch_buffers
                .iter()
                .copied()
                .find(|descriptor| descriptor_has_data(*descriptor))
        })
        .or_else(|| {
            // In pointer mode, B descriptors are empty; fall back to C descriptors.
            descriptor_layout
                .recv_statics
                .iter()
                .copied()
                .find(|d| descriptor_has_data(*d))
        });

    let descriptor = out_descriptor?;
    let descriptor_size = usize::try_from(descriptor.size).ok()?;
    let total_size = 16usize.checked_add(payload.len())?;
    if descriptor_size < total_size {
        return None;
    }

    let payload_size_u32 = u32::try_from(payload.len()).ok()?;
    let total_size_u32 = u32::try_from(total_size).ok()?;

    let mut parcel = vec![0u8; total_size];
    // ParcelHeader { payload_size, payload_off, objects_size, objects_off }
    parcel[0..4].copy_from_slice(&payload_size_u32.to_le_bytes());
    parcel[4..8].copy_from_slice(&16u32.to_le_bytes());
    parcel[8..12].copy_from_slice(&0u32.to_le_bytes());
    parcel[12..16].copy_from_slice(&total_size_u32.to_le_bytes());
    parcel[16..].copy_from_slice(payload);

    if !write_guest_buffer(memory, descriptor.address, &parcel) {
        return None;
    }

    Some(total_size as u64)
}

/// Read the incoming binder parcel payload from the send buffer/statics.
/// Returns the raw data portion of the parcel (after the 16-byte header).
fn vi_read_binder_parcel(memory: &SharedGuestMemory, tls: u64) -> Option<Vec<u8>> {
    let descriptor_layout = parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
    let in_descriptor = descriptor_layout
        .send_buffers
        .iter()
        .copied()
        .find(|d| descriptor_has_data(*d))
        .or_else(|| {
            descriptor_layout
                .exch_buffers
                .iter()
                .copied()
                .find(|d| descriptor_has_data(*d))
        })
        .or_else(|| {
            descriptor_layout
                .send_statics
                .iter()
                .copied()
                .find(|d| descriptor_has_data(*d))
        })?;

    let raw = read_guest_buffer(memory, in_descriptor.address, in_descriptor.size)?;
    if raw.len() < 16 {
        return None;
    }
    // Parcel header: data_size(u32), data_offset(u32), objects_size(u32), objects_offset(u32)
    let data_size = u32::from_le_bytes(raw[0..4].try_into().ok()?) as usize;
    let data_offset = u32::from_le_bytes(raw[4..8].try_into().ok()?) as usize;
    if data_offset.checked_add(data_size)? > raw.len() {
        return None;
    }
    Some(raw[data_offset..data_offset + data_size].to_vec())
}

fn vi_write_native_window_parcel(
    memory: &SharedGuestMemory,
    tls: u64,
    binder_id: u32,
) -> Option<u64> {
    // Full FlatBinderObject / NativeWindow format (0x28 = 40 bytes):
    //   [0x00] u32 type  = 2 (FLAT_BINDER_TYPE_BINDER)
    //   [0x04] u32 flags = 0
    //   [0x08] u64 binder_id
    //   [0x10] u64 cookie = 0
    //   [0x18] u8[8] service_name = "dispdrv\0"
    //   [0x20] u8[8] padding = 0
    //
    // The parcel also needs an objects section (4 bytes of zeroes) so that
    // nn::vi SDK parses the flattened binder correctly.  vi_write_parcel_to_out_buffer
    // only writes payload data, so we build the full 60-byte parcel here.
    let mut payload = [0u8; 40];
    payload[0..4].copy_from_slice(&2u32.to_le_bytes()); // type = FLAT_BINDER_TYPE_BINDER
    // flags = 0 at [4..8]
    payload[8..12].copy_from_slice(&binder_id.to_le_bytes()); // binder_id low 32 bits
    // binder_id high 32 bits = 0 at [12..16]
    // cookie = 0 at [16..24]
    payload[24..32].copy_from_slice(b"dispdrv\0"); // service name
    // padding = 0 at [32..40]

    // Build the full parcel with objects section (4 bytes).
    let payload_size: u32 = 40;
    let parcel_header_size: u32 = 16;
    let objects_size: u32 = 4;
    let total_size = parcel_header_size + payload_size + objects_size; // 60

    let mut parcel = vec![0u8; total_size as usize];
    // Parcel header
    parcel[0..4].copy_from_slice(&payload_size.to_le_bytes()); // data_size
    parcel[4..8].copy_from_slice(&parcel_header_size.to_le_bytes()); // data_offset
    parcel[8..12].copy_from_slice(&objects_size.to_le_bytes()); // objects_size
    let objects_offset = parcel_header_size + payload_size;
    parcel[12..16].copy_from_slice(&objects_offset.to_le_bytes()); // objects_offset
    // Payload
    parcel[16..56].copy_from_slice(&payload);
    // Objects section: single u32 = 0
    // (already zeroed)

    // Write directly to the output buffer descriptor.
    let descriptor_layout = parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
    let out_descriptor = descriptor_layout
        .recv_buffers
        .iter()
        .copied()
        .find(|d| descriptor_has_data(*d))
        .or_else(|| {
            descriptor_layout
                .exch_buffers
                .iter()
                .copied()
                .find(|d| descriptor_has_data(*d))
        })
        .or_else(|| {
            // In pointer mode, B descriptors are empty; fall back to C descriptors.
            descriptor_layout
                .recv_statics
                .iter()
                .copied()
                .find(|d| descriptor_has_data(*d))
        })?;

    if (out_descriptor.size as usize) < parcel.len() {
        return None;
    }

    if !write_guest_buffer(memory, out_descriptor.address, &parcel) {
        return None;
    }

    Some(total_size as u64)
}

fn vi_binder_reply_payload_for_transaction(transaction_id: u32) -> Vec<u8> {
    const STATUS_OK: u32 = 0;
    const DEFAULT_WIDTH: u32 = 1280;
    const DEFAULT_HEIGHT: u32 = 720;

    let mut payload = Vec::new();
    match transaction_id {
        // IGraphicBufferProducer::RequestBuffer
        1 => {
            payload.extend_from_slice(&0u32.to_le_bytes());
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
        // IGraphicBufferProducer::DequeueBuffer
        // Format: slot(i32), has_fence(i32=1), flattened MultiFence, status(i32=0)
        3 => {
            payload.extend_from_slice(&0i32.to_le_bytes()); // slot 0
            payload.extend_from_slice(&1i32.to_le_bytes()); // has_fence = true
            // flattened object header
            payload.extend_from_slice(&36u32.to_le_bytes()); // flat_size
            payload.extend_from_slice(&0u32.to_le_bytes());  // fd_count
            // NvMultiFence: numFences=0, 4 fences zeroed
            payload.extend_from_slice(&0u32.to_le_bytes());
            for _ in 0..4 {
                payload.extend_from_slice(&u32::MAX.to_le_bytes());
                payload.extend_from_slice(&0u32.to_le_bytes());
            }
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
        // IGraphicBufferProducer::{DetachBuffer,Disconnect}
        4 | 11 => {
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
        // IGraphicBufferProducer::QueueBuffer
        7
        // IGraphicBufferProducer::Connect
        | 10 => {
            payload.extend_from_slice(&DEFAULT_WIDTH.to_le_bytes());
            payload.extend_from_slice(&DEFAULT_HEIGHT.to_le_bytes());
            payload.extend_from_slice(&0u32.to_le_bytes());
            payload.extend_from_slice(&0u32.to_le_bytes());
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
        // libnx treats these as success with an empty reply parcel.
        // bqCancelBuffer()/bqSetPreallocatedBuffer() do not parse reply payload.
        8 | 14 => {}
        // IGraphicBufferProducer::Query
        9 => {
            payload.extend_from_slice(&0u32.to_le_bytes());
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
        // Other binder transactions currently return only android::status_t.
        _ => {
            payload.extend_from_slice(&STATUS_OK.to_le_bytes());
        }
    }
    payload
}

/// Skip the Android Binder InterfaceToken at the start of transaction data.
/// Format: u32 strict_mode_policy, String16(interface_name)
/// String16: u32 char_count, UTF-16LE chars[char_count+1] (with null), aligned to 4.
fn skip_binder_interface_token(data: &[u8]) -> &[u8] {
    if data.len() < 8 {
        return &[];
    }
    let _strict_mode = u32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
    let str_len = u32::from_le_bytes(data[4..8].try_into().unwrap_or([0; 4])) as usize;
    // UTF-16 bytes: (str_len+1) chars × 2 bytes, aligned up to 4
    let str_bytes = ((str_len + 1) * 2 + 3) & !3;
    let offset = 8 + str_bytes;
    if offset > data.len() {
        return &[];
    }
    &data[offset..]
}

/// Handle SetPreallocatedBuffer (transaction 14).
/// Parcel data format: InterfaceToken, i32 slot, i32 hasBuffer, [GraphicBuffer data if hasBuffer!=0]
fn vi_handle_set_preallocated_buffer(kernel: &mut KernelState, parcel_data: &[u8]) {
    // Skip the InterfaceToken at the beginning of the binder parcel data.
    let data = skip_binder_interface_token(parcel_data);
    if data.len() < 8 {
        eprintln!(
            "[binder] SetPreallocatedBuffer: data too short after token ({} bytes)",
            data.len()
        );
        return;
    }
    let slot = i32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
    let has_buffer = i32::from_le_bytes(data[4..8].try_into().unwrap_or([0; 4]));
    eprintln!(
        "[binder] SetPreallocatedBuffer slot={} hasBuffer={} data_len={}",
        slot,
        has_buffer,
        data.len()
    );
    if has_buffer != 0 && data.len() > 8 {
        // Store everything after the slot+hasBuffer as the GraphicBuffer blob.
        let gb_data = data[8..].to_vec();
        eprintln!(
            "[binder] Storing GraphicBuffer for slot {} ({} bytes)",
            slot,
            gb_data.len()
        );
        kernel.set_preallocated_buffer(slot as u32, gb_data);
    }
}

/// Handle RequestBuffer (transaction 1).
/// Request parcel: InterfaceToken, i32 slot
/// Reply: i32 hasBuffer, [GraphicBuffer if hasBuffer!=0], i32 status
fn vi_handle_request_buffer(
    kernel: &mut KernelState,
    memory: &SharedGuestMemory,
    tls: u64,
) -> Vec<u8> {
    const STATUS_OK: u32 = 0;
    let mut payload = Vec::new();

    // Read the requested slot from the incoming parcel (after InterfaceToken).
    let slot = vi_read_binder_parcel(memory, tls)
        .and_then(|data| {
            let after_token = skip_binder_interface_token(&data);
            if after_token.len() >= 4 {
                Some(i32::from_le_bytes(after_token[0..4].try_into().ok()?))
            } else {
                None
            }
        })
        .unwrap_or(0);

    eprintln!("[binder] RequestBuffer slot={}", slot);

    if let Some(gb_data) = kernel.get_preallocated_buffer(slot as u32) {
        eprintln!(
            "[binder] RequestBuffer returning GraphicBuffer for slot {} ({} bytes)",
            slot,
            gb_data.len()
        );
        // hasBuffer = 1
        payload.extend_from_slice(&1i32.to_le_bytes());
        // The stored GraphicBuffer blob (from SetPreallocatedBuffer)
        payload.extend_from_slice(gb_data);
        payload.extend_from_slice(&STATUS_OK.to_le_bytes());
    } else {
        eprintln!("[binder] RequestBuffer: no buffer for slot {}", slot);
        // hasBuffer = 0
        payload.extend_from_slice(&0i32.to_le_bytes());
        payload.extend_from_slice(&STATUS_OK.to_le_bytes());
    }
    payload
}

/// Handle DequeueBuffer (transaction 3).
/// Request parcel: InterfaceToken, i32 async, u32 width, u32 height, i32 format, u32 usage
/// Reply (libnx format): i32 slot, i32 has_fence, [flattened NvMultiFence if has_fence], i32 status
fn vi_handle_dequeue_buffer(
    kernel: &mut KernelState,
    _memory: &SharedGuestMemory,
    _tls: u64,
) -> Vec<u8> {
    const STATUS_OK: i32 = 0;
    let slot = kernel.dequeue_buffer_slot();
    eprintln!("[binder] DequeueBuffer -> slot={}", slot);

    let mut payload = Vec::new();
    // i32 slot
    payload.extend_from_slice(&(slot as i32).to_le_bytes());
    // i32 has_fence = 1  (must be true so libnx reads the flattened object
    // and properly initializes the local NvMultiFence variable; if 0, the
    // stack variable stays uninitialized and nvMultiFenceWait reads garbage)
    payload.extend_from_slice(&1i32.to_le_bytes());
    // Flattened object header for NvMultiFence (36 bytes)
    payload.extend_from_slice(&36u32.to_le_bytes()); // flat_size = sizeof(NvMultiFence)
    payload.extend_from_slice(&0u32.to_le_bytes()); // fd_count = 0
    // NvMultiFence { num_fences = 0, fences[4] = zeroed }
    payload.extend_from_slice(&0u32.to_le_bytes()); // num_fences = 0
    for _ in 0..4 {
        payload.extend_from_slice(&u32::MAX.to_le_bytes()); // syncpt_id = 0xFFFFFFFF (invalid)
        payload.extend_from_slice(&0u32.to_le_bytes()); // syncpt_value = 0
    }
    // i32 status — success
    payload.extend_from_slice(&STATUS_OK.to_le_bytes());
    payload
}

/// Handle Query (transaction 9).
/// Request parcel: InterfaceToken, i32 what
/// Reply: i32 value, i32 status
fn vi_handle_query(memory: &SharedGuestMemory, tls: u64) -> Vec<u8> {
    const STATUS_OK: u32 = 0;

    let what = vi_read_binder_parcel(memory, tls)
        .and_then(|data| {
            let after_token = skip_binder_interface_token(&data);
            if after_token.len() >= 4 {
                Some(i32::from_le_bytes(after_token[0..4].try_into().ok()?))
            } else {
                None
            }
        })
        .unwrap_or(0);

    // NATIVE_WINDOW_* query values (from Android's window.h)
    let value: i32 = match what {
        0 => 1280,  // NATIVE_WINDOW_WIDTH
        1 => 720,   // NATIVE_WINDOW_HEIGHT
        2 => 1,     // NATIVE_WINDOW_FORMAT (HAL_PIXEL_FORMAT_RGBA_8888 = 1)
        3 => 0,     // NATIVE_WINDOW_MIN_UNDEQUEUED_BUFFERS
        5 => 1280,  // NATIVE_WINDOW_DEFAULT_WIDTH
        6 => 720,   // NATIVE_WINDOW_DEFAULT_HEIGHT
        8 => 0x100, // NATIVE_WINDOW_TRANSFORM_HINT
        17 => 1280, // NATIVE_WINDOW_DEFAULT_DATASPACE (or width)
        _ => 0,
    };

    eprintln!("[binder] Query what={} -> value={}", what, value);

    let mut payload = Vec::new();
    payload.extend_from_slice(&value.to_le_bytes());
    payload.extend_from_slice(&STATUS_OK.to_le_bytes());
    payload
}

/// Handle QueueBuffer (transaction 7).
/// Request parcel: InterfaceToken, i32 slot, QueueBufferInput{...}
/// Reply: u32 width, u32 height, u32 transformHint, u32 numPendingBuffers, i32 status
fn vi_handle_queue_buffer(
    kernel: &mut KernelState,
    memory: &SharedGuestMemory,
    tls: u64,
) -> Vec<u8> {
    const STATUS_OK: u32 = 0;
    const DEFAULT_WIDTH: u32 = 1280;
    const DEFAULT_HEIGHT: u32 = 720;

    let parcel_data = vi_read_binder_parcel(memory, tls).unwrap_or_default();
    let after_token = skip_binder_interface_token(&parcel_data);
    let slot = if after_token.len() >= 4 {
        i32::from_le_bytes(after_token[0..4].try_into().unwrap_or([0; 4]))
    } else {
        0
    };
    let queue_input = after_token.get(4..).unwrap_or(&[]);
    let async_mode = read_payload_i32(queue_input, 40).unwrap_or(0) != 0;
    let swap_interval = read_payload_i32(queue_input, 44).unwrap_or(1);
    let fence_count = read_payload_i32(queue_input, 48).unwrap_or(0).max(0) as u32;

    let frame = kernel.increment_frame_count();
    eprintln!(
        "[binder] QueueBuffer slot={} frame={} async={} swap_interval={} fence_count={}",
        slot, frame, async_mode, swap_interval, fence_count,
    );
    let pending_buffers = if slot >= 0 {
        kernel.queue_binder_buffer(slot as u32, frame, async_mode) as u32
    } else {
        0
    };
    if slot >= 0 {
        kernel.set_queued_buffer_slot(slot as u32);
    }
    kernel.set_binder_pending_frame(frame);

    if std::env::var("RYURS_TRACE_2048_BOARD").ok().as_deref() == Some("1") {
        const BOARD_BASE: u64 = 0x8014_7180;
        let mut cells = [0u32; 16];
        let mut ok = true;
        for (index, cell) in cells.iter_mut().enumerate() {
            let addr = BOARD_BASE + (index as u64) * 4;
            match memory.fetch_u32(addr) {
                Some(value) => *cell = value,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            eprintln!(
                "[2048-board] frame={} {:4} {:4} {:4} {:4} | {:4} {:4} {:4} {:4} | {:4} {:4} {:4} {:4} | {:4} {:4} {:4} {:4}",
                frame,
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                cells[4],
                cells[5],
                cells[6],
                cells[7],
                cells[8],
                cells[9],
                cells[10],
                cells[11],
                cells[12],
                cells[13],
                cells[14],
                cells[15],
            );
        }
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&DEFAULT_WIDTH.to_le_bytes());
    payload.extend_from_slice(&DEFAULT_HEIGHT.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes()); // transformHint
    payload.extend_from_slice(&pending_buffers.to_le_bytes()); // numPendingBuffers
    payload.extend_from_slice(&STATUS_OK.to_le_bytes());
    payload
}

#[derive(Debug, Clone, Copy)]
struct QueuedGraphicBufferInfo {
    width: u32,
    height: u32,
    stride: u32,
    pixel_format: u32,
    color_format: u64,
    layout: i32,
    pitch: u32,
    nvmap_handle: i32,
    offset: u64,
    kind: i32,
    block_height_log2: u32,
    size: u64,
}

fn read_blob_u32(blob: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let bytes: [u8; 4] = blob.get(offset..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn read_blob_i32(blob: &[u8], offset: usize) -> Option<i32> {
    Some(read_blob_u32(blob, offset)? as i32)
}

fn read_blob_u64(blob: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let bytes: [u8; 8] = blob.get(offset..end)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn parse_queued_graphic_buffer(blob: &[u8]) -> Option<QueuedGraphicBufferInfo> {
    const HEADER_SIZE: usize = 0x28;
    const BUFFER_SIZE: usize = 0x144;
    const FLATTENED_SIZE_PREFIX: usize = 8;
    let base = if blob.len() >= FLATTENED_SIZE_PREFIX + HEADER_SIZE + BUFFER_SIZE {
        let flattened_size = read_blob_u64(blob, 0)? as usize;
        if flattened_size == HEADER_SIZE + BUFFER_SIZE {
            FLATTENED_SIZE_PREFIX
        } else {
            0
        }
    } else {
        0
    };

    const SURFACE0_REL_OFFSET: usize = HEADER_SIZE + 0x34;
    if blob.len() < base + HEADER_SIZE + BUFFER_SIZE {
        return None;
    }
    let surface0_offset = base + SURFACE0_REL_OFFSET;

    let header_width = read_blob_i32(blob, base + 0x04)?;
    let header_height = read_blob_i32(blob, base + 0x08)?;
    let header_stride = read_blob_i32(blob, base + 0x0c)?;
    let pixel_format = read_blob_u32(blob, base + 0x10)?;
    let nvmap_id = read_blob_i32(blob, base + HEADER_SIZE + 0x04)?;

    let surface_width = read_blob_u32(blob, surface0_offset + 0x00)?;
    let surface_height = read_blob_u32(blob, surface0_offset + 0x04)?;
    let color_format = read_blob_u64(blob, surface0_offset + 0x08)?;
    let layout = read_blob_i32(blob, surface0_offset + 0x10)?;
    let pitch = read_blob_i32(blob, surface0_offset + 0x14)?;
    let surface_nvmap_handle = read_blob_i32(blob, surface0_offset + 0x18)?;
    let offset = read_blob_i32(blob, surface0_offset + 0x1c)? as i64;
    let kind = read_blob_i32(blob, surface0_offset + 0x20)?;
    let block_height_log2 = read_blob_i32(blob, surface0_offset + 0x24)?;
    let size = read_blob_u64(blob, surface0_offset + 0x38)?;

    let width = if surface_width != 0 {
        surface_width
    } else {
        u32::try_from(header_width).ok()?
    };
    let height = if surface_height != 0 {
        surface_height
    } else {
        u32::try_from(header_height).ok()?
    };
    let stride = if header_stride > 0 {
        header_stride as u32
    } else {
        width
    };
    let pitch = if pitch > 0 {
        pitch as u32
    } else {
        stride.saturating_mul(4)
    };
    let nvmap_handle = if surface_nvmap_handle != 0 {
        surface_nvmap_handle
    } else {
        nvmap_id
    };
    let offset = u64::try_from(offset).ok()?;
    let block_height_log2 = u32::try_from(block_height_log2).ok()?;

    if width == 0 || height == 0 || nvmap_handle == 0 {
        return None;
    }

    Some(QueuedGraphicBufferInfo {
        width,
        height,
        stride,
        pixel_format,
        color_format,
        layout,
        pitch,
        nvmap_handle,
        offset,
        kind,
        block_height_log2,
        size,
    })
}

fn read_guest_memory_span(
    memory: &SharedGuestMemory,
    address: u64,
    size: usize,
) -> Option<Vec<u8>> {
    let guest_base = memory.guest_base_address();
    let guest_size = memory.guest_memory_size() as u64;

    if address >= guest_base
        && address.saturating_add(size as u64) <= guest_base.saturating_add(guest_size)
    {
        let slice = unsafe { memory.guest_memory_slice() };
        let offset = (address - guest_base) as usize;
        return Some(slice.get(offset..offset + size)?.to_vec());
    }

    let mut view = memory.locked_view();
    let mut bytes = Vec::with_capacity(size);
    for offset in 0..size {
        bytes.push(view.read_u8(address.wrapping_add(offset as u64))?);
    }
    Some(bytes)
}

fn div_round_up(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

fn align_up(value: usize, alignment: usize) -> usize {
    div_round_up(value, alignment) * alignment
}

fn block_linear_offset(
    x: usize,
    y: usize,
    width: usize,
    gob_blocks_in_y: usize,
    bpp: usize,
) -> usize {
    const GOB_STRIDE: usize = 64;
    const GOB_HEIGHT: usize = 8;
    const GOB_SIZE: usize = 512;

    let width_in_gobs = div_round_up(width * bpp, GOB_STRIDE);
    let rob_size = GOB_SIZE * gob_blocks_in_y * width_in_gobs;
    let bh_mask = gob_blocks_in_y - 1;
    let bh_shift = gob_blocks_in_y.trailing_zeros() as usize;
    let x_shift = (GOB_SIZE * gob_blocks_in_y).trailing_zeros() as usize;

    let x_bytes = x * bpp;
    let yh = y / GOB_HEIGHT;

    let mut offset = (yh >> bh_shift) * rob_size;
    offset += (x_bytes / GOB_STRIDE) << x_shift;
    offset += (yh & bh_mask) * GOB_SIZE;
    offset += ((x_bytes & 0x3f) >> 5) << 8;
    offset += ((y & 0x07) >> 1) << 6;
    offset += ((x_bytes & 0x1f) >> 4) << 5;
    offset += (y & 0x01) << 4;
    offset += x_bytes & 0x0f;
    offset
}

fn bytes_per_pixel_from_color_format(color_format: u64) -> Option<usize> {
    let bpp = ((color_format >> 3) & 0x1f) as usize;
    match bpp {
        1 | 2 | 4 | 8 | 16 => Some(bpp),
        _ => None,
    }
}

fn decode_gob_16bx2(dst: &mut [u8], dst_base: usize, stride: usize, src_gob: &[u8]) -> Option<()> {
    for i in 0..32usize {
        let y = ((i >> 1) & 0x06) | (i & 0x01);
        let x = ((i << 3) & 0x10) | ((i << 1) & 0x20);
        let src_offset = i * 16;
        let dst_offset = dst_base
            .checked_add(y.checked_mul(stride)?)?
            .checked_add(x)?;
        let src = src_gob.get(src_offset..src_offset + 16)?;
        let dst = dst.get_mut(dst_offset..dst_offset + 16)?;
        dst.copy_from_slice(src);
    }

    Some(())
}

fn trace_blocklinear_probe_samples(
    address: u64,
    info: QueuedGraphicBufferInfo,
    source: &[u8],
    decoded_rgba: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
) {
    if std::env::var_os("RYURS_TRACE_FB_PROBES").is_none() {
        return;
    }

    let gob_blocks_in_y = 1usize << info.block_height_log2.min(5);
    let aligned_width = align_up(width, 64 / bpp.max(1));
    let probes = [
        (0usize, 170usize),
        (64, 170),
        (128, 170),
        (0, 200),
        (64, 200),
        (128, 200),
        (0, 250),
        (64, 250),
        (128, 250),
    ];

    for (x, y) in probes {
        if x >= width || y >= height {
            continue;
        }

        let src_offset = block_linear_offset(x, y, aligned_width, gob_blocks_in_y, bpp);
        let Some(src_pixel) = source.get(src_offset..src_offset + bpp) else {
            continue;
        };
        let dst_offset = (y * width + x) * 4;
        let Some(rgba) = decoded_rgba.get(dst_offset..dst_offset + 4) else {
            continue;
        };
        let src_bytes = src_pixel
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join("");

        eprintln!(
            "[fb-probe] addr=0x{:016x} x={} y={} src_off=0x{:x} src={} rgba=({}, {}, {}, {})",
            address + src_offset as u64,
            x,
            y,
            src_offset,
            src_bytes,
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
        );
    }
}

fn decode_blocklinear_16bx2_to_linear(
    source: &[u8],
    pitch: usize,
    height: usize,
    block_height_log2: u32,
) -> Option<Vec<u8>> {
    let block_height_gobs = 1usize << block_height_log2.min(5);
    let block_height_px = 8usize.checked_mul(block_height_gobs)?;
    let width_blocks = div_round_up(pitch, 64);
    let height_blocks = div_round_up(height, block_height_px);
    let output_len = pitch.checked_mul(height)?;
    let mut output = vec![0u8; output_len];
    let mut source_offset = 0usize;

    for block_y in 0..height_blocks {
        for block_x in 0..width_blocks {
            for gob_y in 0..block_height_gobs {
                let gob = source.get(source_offset..source_offset + 512)?;
                let x = block_x * 64;
                let y = block_y
                    .checked_mul(block_height_px)?
                    .checked_add(gob_y.checked_mul(8)?)?;

                if y < height {
                    let dst_base = y.checked_mul(pitch)?.checked_add(x)?;
                    decode_gob_16bx2(&mut output, dst_base, pitch, gob)?;
                }

                source_offset = source_offset.checked_add(512)?;
            }
        }
    }

    Some(output)
}

fn convert_pixel_to_rgba(pixel: &[u8], bpp: usize, pixel_format: u32, out: &mut [u8]) {
    const PIXEL_FORMAT_RGBA_8888: u32 = 1;
    const PIXEL_FORMAT_RGBX_8888: u32 = 2;
    const PIXEL_FORMAT_RGB_565: u32 = 4;
    const PIXEL_FORMAT_BGRA_8888: u32 = 5;
    const PIXEL_FORMAT_RGBA_5551: u32 = 6;
    const PIXEL_FORMAT_RGBA_4444: u32 = 7;

    match (bpp, pixel_format) {
        (4, PIXEL_FORMAT_BGRA_8888) => {
            out[0] = pixel[2];
            out[1] = pixel[1];
            out[2] = pixel[0];
            out[3] = pixel[3];
        }
        (4, PIXEL_FORMAT_RGBX_8888) => {
            out[0] = pixel[0];
            out[1] = pixel[1];
            out[2] = pixel[2];
            out[3] = 0xff;
        }
        (2, PIXEL_FORMAT_RGB_565) => {
            let raw = u16::from_le_bytes([pixel[0], pixel[1]]);
            let r = ((raw >> 11) & 0x1f) as u8;
            let g = ((raw >> 5) & 0x3f) as u8;
            let b = (raw & 0x1f) as u8;
            out[0] = (r << 3) | (r >> 2);
            out[1] = (g << 2) | (g >> 4);
            out[2] = (b << 3) | (b >> 2);
            out[3] = 0xff;
        }
        (2, PIXEL_FORMAT_RGBA_5551) => {
            let raw = u16::from_le_bytes([pixel[0], pixel[1]]);
            let r = ((raw >> 11) & 0x1f) as u8;
            let g = ((raw >> 6) & 0x1f) as u8;
            let b = ((raw >> 1) & 0x1f) as u8;
            out[0] = (r << 3) | (r >> 2);
            out[1] = (g << 3) | (g >> 2);
            out[2] = (b << 3) | (b >> 2);
            out[3] = if (raw & 1) != 0 { 0xff } else { 0x00 };
        }
        (2, PIXEL_FORMAT_RGBA_4444) => {
            let raw = u16::from_le_bytes([pixel[0], pixel[1]]);
            out[0] = (((raw >> 12) & 0x0f) as u8) * 0x11;
            out[1] = (((raw >> 8) & 0x0f) as u8) * 0x11;
            out[2] = (((raw >> 4) & 0x0f) as u8) * 0x11;
            out[3] = ((raw & 0x0f) as u8) * 0x11;
        }
        (4, PIXEL_FORMAT_RGBA_8888) | (4, _) => {
            out.copy_from_slice(&pixel[..4]);
        }
        _ => {
            out.fill(0);
        }
    }
}

fn decode_graphic_buffer(
    memory: &SharedGuestMemory,
    address: u64,
    info: QueuedGraphicBufferInfo,
) -> Option<Vec<u8>> {
    let width = info.width as usize;
    let height = info.height as usize;
    let pitch = info.pitch as usize;
    let inferred_bpp = (pitch / width.max(1)).max(1);
    let format_bpp = bytes_per_pixel_from_color_format(info.color_format);
    let bpp = match inferred_bpp {
        2 | 4 => inferred_bpp,
        _ => format_bpp.unwrap_or(4),
    };
    let linear_row_bytes = width.checked_mul(bpp)?;
    let output_len = width.checked_mul(height)?.checked_mul(4)?;
    let mut output = vec![0u8; output_len];

    let force_linear = std::env::var_os("RYURS_FORCE_LINEAR_FB").is_some();
    let is_block_linear = !force_linear
        && (info.block_height_log2 > 0 || info.size > (pitch as u64).saturating_mul(height as u64));

    if std::env::var_os("RYURS_TRACE_FB_SOURCE").is_some() {
        eprintln!(
            "[fb] decode address=0x{address:016x} size=0x{:x} {}x{} pitch={} bpp={} kind={} block_h_log2={} force_linear={} block_linear={}",
            info.size,
            width,
            height,
            pitch,
            bpp,
            info.kind,
            info.block_height_log2,
            force_linear,
            is_block_linear,
        );
    }

    if !is_block_linear {
        let total_size = pitch.checked_mul(height)?;
        let source = read_guest_memory_span(memory, address, total_size)?;
        for y in 0..height {
            let src_row = &source[y * pitch..y * pitch + linear_row_bytes];
            let dst_row = &mut output[y * width * 4..(y + 1) * width * 4];
            for x in 0..width {
                let src_pixel = &src_row[x * bpp..x * bpp + bpp];
                let dst_pixel = &mut dst_row[x * 4..x * 4 + 4];
                convert_pixel_to_rgba(src_pixel, bpp, info.pixel_format, dst_pixel);
            }
        }
        return Some(output);
    }

    let source_size = if info.size != 0 {
        usize::try_from(info.size).ok()?
    } else {
        pitch.checked_mul(height)?
    };
    let Some(source) = read_guest_memory_span(memory, address, source_size) else {
        if std::env::var_os("RYURS_TRACE_FB_PROBES").is_some() {
            eprintln!(
                "[fb-probe] failed to read source span addr=0x{address:016x} size=0x{source_size:x}"
            );
        }
        return None;
    };

    if info.kind == 254 && std::env::var_os("RYURS_DISABLE_16BX2").is_none() {
        let Some(linear) =
            decode_blocklinear_16bx2_to_linear(&source, pitch, height, info.block_height_log2)
        else {
            if std::env::var_os("RYURS_TRACE_FB_PROBES").is_some() {
                eprintln!(
                    "[fb-probe] 16bx2 decode failed addr=0x{address:016x} pitch={} height={} block_h={} source_len=0x{:x}",
                    pitch,
                    height,
                    info.block_height_log2,
                    source.len(),
                );
            }
            return None;
        };
        for y in 0..height {
            let src_row = &linear[y * pitch..y * pitch + linear_row_bytes];
            let dst_row = &mut output[y * width * 4..(y + 1) * width * 4];
            for x in 0..width {
                let src_pixel = &src_row[x * bpp..x * bpp + bpp];
                let dst_pixel = &mut dst_row[x * 4..x * 4 + 4];
                convert_pixel_to_rgba(src_pixel, bpp, info.pixel_format, dst_pixel);
            }
        }
        if std::env::var_os("RYURS_TRACE_FB_PROBES").is_some() {
            eprintln!("[fb-probe] 16bx2 decode succeeded addr=0x{address:016x}");
        }
        trace_blocklinear_probe_samples(address, info, &source, &output, width, height, bpp);
        return Some(output);
    }

    let gob_blocks_in_y = 1usize << info.block_height_log2.min(5);
    let aligned_width = align_up(width, 64 / bpp.max(1));

    for y in 0..height {
        for x in 0..width {
            let src_offset = block_linear_offset(x, y, aligned_width, gob_blocks_in_y, bpp);
            let src_pixel = source.get(src_offset..src_offset + bpp)?;
            let dst_offset = (y * width + x) * 4;
            convert_pixel_to_rgba(
                src_pixel,
                bpp,
                info.pixel_format,
                &mut output[dst_offset..dst_offset + 4],
            );
        }
    }

    trace_blocklinear_probe_samples(address, info, &source, &output, width, height, bpp);

    Some(output)
}

fn decode_queued_graphic_buffer_slot(
    kernel: &KernelState,
    memory: &SharedGuestMemory,
    slot: u32,
) -> Option<(QueuedGraphicBufferInfo, u64, u32, Vec<u8>)> {
    let blob = kernel.get_preallocated_buffer(slot)?;
    let info = parse_queued_graphic_buffer(blob)?;
    let (base_address, map_size, _map_kind) = kernel.nvmap_info(info.nvmap_handle)?;
    let address = base_address.checked_add(info.offset)?;
    let info = QueuedGraphicBufferInfo {
        size: if info.size != 0 {
            info.size
        } else {
            u64::from(map_size).saturating_sub(info.offset)
        },
        ..info
    };

    let frame = decode_graphic_buffer(memory, address, info)?;
    Some((info, address, map_size, frame))
}

fn read_queued_graphic_buffer(
    kernel: &KernelState,
    memory: &SharedGuestMemory,
    slot_override: Option<u32>,
) -> Option<DisplayFrame> {
    let slot = slot_override.or_else(|| kernel.queued_buffer_slot())?;
    let (info, _, _, frame) = decode_queued_graphic_buffer_slot(kernel, memory, slot)?;
    Some(DisplayFrame {
        width: info.width,
        height: info.height,
        pixels: frame,
    })
}

/// Read the guest framebuffer pixels from memory.
///
/// Returns `None` if no framebuffer candidate is found or the memory read fails.
fn read_guest_framebuffer(
    kernel: &KernelState,
    memory: &SharedGuestMemory,
    slot_override: Option<u32>,
) -> Option<DisplayFrame> {
    const FB_SIZE: usize = 1280 * 720 * 4;

    if let Ok(raw) = std::env::var("RYURS_FB_ADDRESS") {
        let raw = raw
            .trim()
            .strip_prefix("0x")
            .or_else(|| raw.trim().strip_prefix("0X"))
            .unwrap_or(raw.trim());
        let address = u64::from_str_radix(raw, 16).ok()?;
        let size = 1280usize.checked_mul(720)?.checked_mul(4)?;
        if std::env::var_os("RYURS_TRACE_FB_SOURCE").is_some() {
            eprintln!("[fb] source=override-address address=0x{address:016x} size=0x{size:x}");
        }
        return Some(DisplayFrame {
            width: 1280,
            height: 720,
            pixels: read_guest_buffer(memory, address, size as u64)?,
        });
    }

    if let Some(frame) = read_queued_graphic_buffer(kernel, memory, slot_override) {
        if std::env::var_os("RYURS_TRACE_FB_SOURCE").is_some() {
            eprintln!(
                "[fb] source=queued-graphic-buffer {}x{} size={}",
                frame.width,
                frame.height,
                frame.pixels.len(),
            );
        }
        return Some(frame);
    }

    let candidates = kernel.framebuffer_candidates();
    let (address, _size) = candidates.first()?;
    let address = *address;

    let guest_base = memory.guest_base_address();
    let guest_size = memory.guest_memory_size() as u64;

    let mut fb_bytes = vec![0u8; FB_SIZE];

    if address >= guest_base
        && address.saturating_add(FB_SIZE as u64) <= guest_base.saturating_add(guest_size)
    {
        let slice = unsafe { memory.guest_memory_slice() };
        let offset = (address - guest_base) as usize;
        if offset + FB_SIZE <= slice.len() {
            fb_bytes.copy_from_slice(&slice[offset..offset + FB_SIZE]);
        } else {
            return None;
        }
    } else {
        for i in (0..FB_SIZE).step_by(4) {
            if let Some(word) = memory.read_u32(address + i as u64) {
                fb_bytes[i..i + 4].copy_from_slice(&word.to_le_bytes());
            } else {
                return None;
            }
        }
    }

    Some(DisplayFrame {
        width: 1280,
        height: 720,
        pixels: fb_bytes,
    })
}

/// Send the current guest framebuffer to the display renderer via the frame channel.
fn send_frame_to_display(
    kernel: &KernelState,
    memory: &SharedGuestMemory,
    slot_override: Option<u32>,
) {
    let Some(tx) = kernel.frame_sender() else {
        return;
    };
    let Some(frame) = read_guest_framebuffer(kernel, memory, slot_override) else {
        return;
    };
    // Non-blocking send — drop the frame if the channel is full.
    let _ = tx.try_send(frame);
}

fn flush_pending_binder_frame(kernel: &mut KernelState, memory: &SharedGuestMemory) {
    let Some(_frame) = kernel.take_binder_pending_frame() else {
        return;
    };

    let Some(item) = kernel.acquire_next_binder_buffer() else {
        return;
    };

    if kernel.frame_sender().is_some() {
        send_frame_to_display(kernel, memory, Some(item.slot));
    }

    if item.frame > 0 && item.frame % 30 == 0 {
        try_capture_frame(kernel, memory, item.frame, Some(item.slot));
    }

    kernel.release_binder_buffer(item.slot);
}

/// Attempt to capture a framebuffer from guest memory and save it as a PNG.
///
/// Captures the currently selected framebuffer and writes it to `frames/frame_{n}.png`.
fn try_capture_frame(
    kernel: &KernelState,
    memory: &SharedGuestMemory,
    frame: u64,
    slot_override: Option<u32>,
) {
    if let Some(slot) = slot_override.or_else(|| kernel.queued_buffer_slot()) {
        if let Some(blob) = kernel.get_preallocated_buffer(slot) {
            if let Some(info) = parse_queued_graphic_buffer(blob) {
                let (base_address, map_size) = kernel
                    .nvmap_info(info.nvmap_handle)
                    .map(|(base_address, map_size, _)| (base_address, map_size))
                    .unwrap_or((0, 0));
                let frame_address = base_address.saturating_add(info.offset);
                eprintln!(
                    "[frame-capture] frame {frame}: slot={} {}x{} stride={} fmt={} color_fmt=0x{:x} layout={} pitch={} nvmap={} base=0x{:x} addr=0x{:x} map_size=0x{:x} offset=0x{:x} kind={} block_h={} size=0x{:x}",
                    slot,
                    info.width,
                    info.height,
                    info.stride,
                    info.pixel_format,
                    info.color_format,
                    info.layout,
                    info.pitch,
                    info.nvmap_handle,
                    base_address,
                    frame_address,
                    map_size,
                    info.offset,
                    info.kind,
                    info.block_height_log2,
                    info.size,
                );
            } else {
                eprintln!(
                    "[frame-capture] frame {frame}: failed to parse queued GraphicBuffer for slot {slot}"
                );
            }
        }
    }

    let Some(display_frame) = read_guest_framebuffer(kernel, memory, slot_override) else {
        eprintln!("[frame-capture] frame {frame}: failed to decode framebuffer");
        return;
    };

    // Save as PNG.
    if let Err(err) = std::fs::create_dir_all("frames") {
        eprintln!("[frame-capture] frame {frame}: cannot create frames/ directory: {err}");
        return;
    }

    let path = format!("frames/frame_{frame}.png");
    match image::save_buffer(
        &path,
        &display_frame.pixels,
        display_frame.width,
        display_frame.height,
        image::ColorType::Rgba8,
    ) {
        Ok(()) => eprintln!("[frame-capture] frame {frame}: saved {path}"),
        Err(err) => eprintln!("[frame-capture] frame {frame}: failed to save PNG: {err}"),
    }
}

fn apm_current_performance_mode() -> &'static AtomicU32 {
    static MODE: OnceLock<AtomicU32> = OnceLock::new();
    MODE.get_or_init(|| AtomicU32::new(0))
}

fn apm_performance_configuration_slots() -> &'static [AtomicU32; 2] {
    static CONFIGS: OnceLock<[AtomicU32; 2]> = OnceLock::new();
    CONFIGS.get_or_init(|| [AtomicU32::new(0), AtomicU32::new(0)])
}

fn apm_read_performance_configuration(mode: u32) -> u32 {
    apm_performance_configuration_slots()
        .get(mode as usize)
        .map(|slot| slot.load(Ordering::Acquire))
        .unwrap_or(0)
}

fn apm_write_performance_configuration(mode: u32, value: u32) {
    if let Some(slot) = apm_performance_configuration_slots().get(mode as usize) {
        slot.store(value, Ordering::Release);
    }
}

fn trace_domain_request(
    session_kind: ServiceSessionKind,
    memory: &SharedGuestMemory,
    tls: u64,
    request: HipcRequestInfo,
    command_id: u32,
    object_id: u32,
    object_kind: DomainObjectKind,
) {
    let data_size = request
        .data_end
        .saturating_sub(request.data_start)
        .min(0x80);
    let request_bytes =
        read_guest_buffer(memory, request.data_start, data_size).unwrap_or_default();

    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] cmd=0x{command_id:08x} object_id={object_id} object_kind={object_kind:?} type={} data_start=0x{:016x} data_end=0x{:016x}",
        request.message_type, request.data_start, request.data_end,
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] tls_words={}",
        format_u32_words(memory, tls, 24),
    );
    if let Some(special) = parse_hipc_special_header(memory, tls) {
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:req] special=0x{:08x} pid={:?} copy={:?} move={:?}",
            special.raw, special.pid, special.copy_handles, special.move_handles,
        );
    }
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] data_bytes={}",
        format_hex_bytes(&request_bytes),
    );
}

fn trace_domain_response(
    session_kind: ServiceSessionKind,
    memory: &SharedGuestMemory,
    tls: u64,
    request: HipcRequestInfo,
    command_id: u32,
    object_id: u32,
    object_kind: DomainObjectKind,
    cmif_result: u32,
    payload: &[u8],
    copy_handle: Option<u32>,
    move_handle: Option<u32>,
    domain_objects: &[u32],
    write_ok: bool,
) {
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] cmd=0x{command_id:08x} object_id={object_id} object_kind={object_kind:?} type={} cmif_result=0x{cmif_result:08x} write_ok={write_ok} copy_handle={copy_handle:?} move_handle={move_handle:?} domain_objects={domain_objects:?}",
        request.message_type,
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] payload_bytes={}",
        format_hex_bytes(payload),
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] tls_words={}",
        format_u32_words(memory, tls, 24),
    );
    if let Some(special) = parse_hipc_special_header(memory, tls) {
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:resp] special=0x{:08x} pid={:?} copy={:?} move={:?}",
            special.raw, special.pid, special.copy_handles, special.move_handles,
        );
    }

    if let Some(response) = parse_hipc_request(memory, tls) {
        let data_size = response
            .data_end
            .saturating_sub(response.data_start)
            .min(0x80);
        let response_bytes =
            read_guest_buffer(memory, response.data_start, data_size).unwrap_or_default();
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:resp] data_start=0x{:016x} data_end=0x{:016x} data_bytes={}",
            response.data_start,
            response.data_end,
            format_hex_bytes(&response_bytes),
        );
    } else {
        eprintln!("[svc::SendSyncRequest:{session_kind:?}:trace:resp] failed to parse response");
    }
}

fn trace_non_domain_request(
    session_kind: ServiceSessionKind,
    memory: &SharedGuestMemory,
    tls: u64,
    request: HipcRequestInfo,
    command_id: u32,
) {
    let data_size = request
        .data_end
        .saturating_sub(request.data_start)
        .min(0x80);
    let request_bytes =
        read_guest_buffer(memory, request.data_start, data_size).unwrap_or_default();

    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] cmd=0x{command_id:08x} type={} data_start=0x{:016x} data_end=0x{:016x}",
        request.message_type, request.data_start, request.data_end,
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] tls_words={}",
        format_u32_words(memory, tls, 24),
    );
    if let Some(special) = parse_hipc_special_header(memory, tls) {
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:req] special=0x{:08x} pid={:?} copy={:?} move={:?}",
            special.raw, special.pid, special.copy_handles, special.move_handles,
        );
    }
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:req] data_bytes={}",
        format_hex_bytes(&request_bytes),
    );
}

fn trace_non_domain_response(
    session_kind: ServiceSessionKind,
    memory: &SharedGuestMemory,
    tls: u64,
    request: HipcRequestInfo,
    command_id: u32,
    cmif_result: u32,
    payload: &[u8],
    copy_handle: Option<u32>,
    move_handle: Option<u32>,
    write_ok: bool,
) {
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] cmd=0x{command_id:08x} type={} cmif_result=0x{cmif_result:08x} write_ok={write_ok} copy_handle={copy_handle:?} move_handle={move_handle:?}",
        request.message_type,
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] payload_bytes={}",
        format_hex_bytes(payload),
    );
    eprintln!(
        "[svc::SendSyncRequest:{session_kind:?}:trace:resp] tls_words={}",
        format_u32_words(memory, tls, 24),
    );
    if let Some(special) = parse_hipc_special_header(memory, tls) {
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:resp] special=0x{:08x} pid={:?} copy={:?} move={:?}",
            special.raw, special.pid, special.copy_handles, special.move_handles,
        );
    }
    if let Some(response) = parse_hipc_request(memory, tls) {
        let data_size = response
            .data_end
            .saturating_sub(response.data_start)
            .min(0x80);
        let response_bytes =
            read_guest_buffer(memory, response.data_start, data_size).unwrap_or_default();
        eprintln!(
            "[svc::SendSyncRequest:{session_kind:?}:trace:resp] data_start=0x{:016x} data_end=0x{:016x} data_bytes={}",
            response.data_start,
            response.data_end,
            format_hex_bytes(&response_bytes),
        );
    } else {
        eprintln!("[svc::SendSyncRequest:{session_kind:?}:trace:resp] failed to parse response");
    }
}

fn parse_cmif_context(memory: &SharedGuestMemory, request: HipcRequestInfo) -> Option<u64> {
    if !matches!(
        request.message_type,
        HIPC_TYPE_CMIF_REQUEST_WITH_CONTEXT | HIPC_TYPE_CMIF_CONTROL_WITH_CONTEXT
    ) {
        return None;
    }

    let context_address = request.data_end.checked_sub(8)?;
    if context_address < request.data_start {
        return None;
    }

    let lo = memory.read_u32(context_address)? as u64;
    let hi = memory.read_u32(context_address.wrapping_add(4))? as u64;
    Some(lo | (hi << 32))
}

fn get_info_core_mask() -> u64 {
    static CORE_MASK_OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();

    CORE_MASK_OVERRIDE
        .get_or_init(|| {
            std::env::var("RYURS_CORE_MASK").ok().and_then(|value| {
                if let Some(hex) = value.strip_prefix("0x") {
                    u64::from_str_radix(hex, 16).ok()
                } else {
                    value.parse::<u64>().ok()
                }
            })
        })
        .unwrap_or(0x0f)
}

fn stack_region_size_for_process(address_space_size: u64) -> u64 {
    let quarter = address_space_size / 4;
    let min_reasonable = DEFAULT_STACK_REGION_SIZE.saturating_mul(4);
    let size = quarter.max(min_reasonable).max(DEFAULT_STACK_REGION_SIZE);
    let size = size.min(address_space_size.max(PAGE_SIZE));
    (size / PAGE_SIZE).max(1) * PAGE_SIZE
}

fn write_simple_hipc_response(memory: &SharedGuestMemory, tls: u64) -> bool {
    memory.write_u32(tls, 0) && memory.write_u32(tls.wrapping_add(4), 0)
}

fn write_cmif_response(
    memory: &SharedGuestMemory,
    tls: u64,
    cmif_result: u32,
    payload: &[u8],
    copy_handle: Option<u32>,
    move_handle: Option<u32>,
    prepend_domain_header: bool,
    domain_objects: &[u32],
    context: Option<u64>,
) -> bool {
    let copy_count = if copy_handle.is_some() { 1u32 } else { 0u32 };
    let move_count = if move_handle.is_some() { 1u32 } else { 0u32 };
    let has_special_header = copy_count != 0 || move_count != 0;
    let special_size = if has_special_header {
        4 + 4 * (copy_count + move_count)
    } else {
        0
    };
    let data_words_offset = 8u64 + u64::from(special_size);
    let data_words_address = tls.wrapping_add(data_words_offset);
    let data_start = align_up_16(data_words_address);
    let pad0 = data_start.saturating_sub(data_words_address);
    let domain_header_size = if prepend_domain_header {
        DOMAIN_MESSAGE_HEADER_SIZE
    } else {
        0
    };
    let cmif_header_address = data_start.wrapping_add(domain_header_size);
    let payload_address = cmif_header_address.wrapping_add(CMIF_HEADER_SIZE);
    let payload_padded_size = ((payload.len() as u64) + 3) & !3;
    let domain_objects_address = payload_address.wrapping_add(payload_padded_size);
    let context_address =
        domain_objects_address.wrapping_add((domain_objects.len() as u64).saturating_mul(4));
    let context_size = if context.is_some() { 8u64 } else { 0u64 };
    let raw_message_end = context_address.wrapping_add(context_size);
    // CMIF responses include trailing alignment bytes after the raw payload area.
    let pad1 = 16u64.saturating_sub(pad0).saturating_sub(context_size);
    let message_end = raw_message_end.wrapping_add(pad1);
    let data_bytes = message_end.saturating_sub(data_words_address);
    let num_data_words = ((data_bytes + 3) / 4) as u32;

    let mut word1 = num_data_words & 0x3ff;
    if has_special_header {
        word1 |= 1u32 << 31;
    }

    if !memory.write_u32(tls, 0) || !memory.write_u32(tls.wrapping_add(4), word1) {
        return false;
    }

    if has_special_header {
        let special = ((copy_count & 0xf) << 1) | ((move_count & 0xf) << 5);
        if !memory.write_u32(tls.wrapping_add(8), special) {
            return false;
        }

        let mut handle_address = tls.wrapping_add(12);
        if let Some(handle) = copy_handle {
            if !memory.write_u32(handle_address, handle) {
                return false;
            }
            handle_address = handle_address.wrapping_add(4);
        }

        if let Some(handle) = move_handle {
            if !memory.write_u32(handle_address, handle) {
                return false;
            }
        }
    }

    if prepend_domain_header
        && (!memory.write_u32(data_start, domain_objects.len() as u32)
            || !memory.write_u32(data_start.wrapping_add(4), 0)
            || !memory.write_u32(data_start.wrapping_add(8), 0)
            || !memory.write_u32(data_start.wrapping_add(12), 0))
    {
        return false;
    }

    if !memory.write_u32(cmif_header_address, CMIF_OUT_HEADER_MAGIC)
        || !memory.write_u32(cmif_header_address.wrapping_add(4), 0)
        || !memory.write_u32(cmif_header_address.wrapping_add(8), cmif_result)
        || !memory.write_u32(cmif_header_address.wrapping_add(12), 0)
    {
        return false;
    }

    for (chunk_index, chunk) in payload.chunks(4).enumerate() {
        let mut packed = [0u8; 4];
        packed[..chunk.len()].copy_from_slice(chunk);
        if !memory.write_u32(
            payload_address.wrapping_add((chunk_index as u64) * 4),
            u32::from_le_bytes(packed),
        ) {
            return false;
        }
    }

    for (index, object_id) in domain_objects.iter().enumerate() {
        if !memory.write_u32(
            domain_objects_address.wrapping_add((index as u64) * 4),
            *object_id,
        ) {
            return false;
        }
    }

    if let Some(context) = context
        && !memory.write_u64(context_address, context)
    {
        return false;
    }

    let mut pad_address = raw_message_end;
    while pad_address < message_end {
        if !memory.write_u32(pad_address, 0) {
            return false;
        }
        pad_address = pad_address.wrapping_add(4);
    }

    true
}

fn appletoe_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::AppletProxyService, 0) => Some(DomainObjectKind::ApplicationProxy),
        (DomainObjectKind::ApplicationProxy, 0) => Some(DomainObjectKind::CommonStateGetter),
        (DomainObjectKind::ApplicationProxy, 1) => Some(DomainObjectKind::SelfController),
        (DomainObjectKind::ApplicationProxy, 2) => Some(DomainObjectKind::WindowController),
        (DomainObjectKind::ApplicationProxy, 3) => Some(DomainObjectKind::AudioController),
        (DomainObjectKind::ApplicationProxy, 4) => Some(DomainObjectKind::DisplayController),
        (DomainObjectKind::ApplicationProxy, 10) => {
            Some(DomainObjectKind::ProcessWindingController)
        }
        (DomainObjectKind::ApplicationProxy, 11) => Some(DomainObjectKind::LibraryAppletCreator),
        (DomainObjectKind::ApplicationProxy, 20) => Some(DomainObjectKind::ApplicationFunctions),
        (DomainObjectKind::ApplicationProxy, 1000) => Some(DomainObjectKind::DebugFunctions),
        (DomainObjectKind::LibraryAppletCreator, 0) => Some(DomainObjectKind::Generic),
        (DomainObjectKind::ApplicationFunctions, 1) => Some(DomainObjectKind::AmStorage),
        (DomainObjectKind::AmStorage, 0) => Some(DomainObjectKind::AmStorageAccessor),
        _ => None,
    }
}

fn apm_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::ApmManager, 0) => Some(DomainObjectKind::ApmSession),
        _ => None,
    }
}

fn lm_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::LmLogService, 0) => Some(DomainObjectKind::LmLogger),
        _ => None,
    }
}

fn friend_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::FriendServiceCreator, 0) => Some(DomainObjectKind::FriendService),
        _ => None,
    }
}

fn aoc_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        // Create*Manager commands return service sub-objects.
        (DomainObjectKind::GenericRoot, 100 | 101 | 110) => Some(DomainObjectKind::Generic),
        _ => None,
    }
}

fn pctl_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        // IParentalControlServiceFactory::CreateService{WithoutInitialize}
        (DomainObjectKind::GenericRoot, 0 | 1) => Some(DomainObjectKind::Generic),
        _ => None,
    }
}

fn hid_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::GenericRoot, 0) => Some(DomainObjectKind::Generic),
        _ => None,
    }
}

fn fsp_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        // IFileSystemProxy::SetCurrentProcess(u64) does not return a child object.
        (DomainObjectKind::GenericRoot, 1) => None,
        // IFileSystemProxy::OpenSdCardFileSystem() -> IFileSystem
        (DomainObjectKind::GenericRoot, 18) => Some(DomainObjectKind::Generic),
        // IFileSystemProxy::OpenSaveDataFileSystem(u8, SaveDataAttribute) -> IFileSystem
        (DomainObjectKind::GenericRoot, 51) => Some(DomainObjectKind::Generic),
        // IFileSystemProxy::OpenSaveDataFileSystemBySystemSaveDataId(u8, SaveDataAttribute) -> IFileSystem
        (DomainObjectKind::GenericRoot, 52) => Some(DomainObjectKind::Generic),
        // IFileSystemProxy::OpenDataStorageByCurrentProcess() -> IStorage
        (DomainObjectKind::GenericRoot, 200) => Some(DomainObjectKind::FspStorage),
        // IFileSystemProxy::OpenDataStorageByDataId() -> IStorage
        (DomainObjectKind::GenericRoot, 202) => Some(DomainObjectKind::FspStorage),
        // IFileSystem::OpenFile/OpenDirectory() -> IFile / IDirectory.
        // We currently model both with a generic child domain object.
        (DomainObjectKind::Generic, 8 | 9) => Some(DomainObjectKind::Generic),
        // Keep legacy generic child mapping for now on other root commands.
        (DomainObjectKind::GenericRoot, _) => generic_child_object_kind(command_id),
        _ => None,
    }
}

fn nifm_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        // IStaticService::CreateGeneralService / CreateGeneralServiceOld
        (DomainObjectKind::GenericRoot, 4 | 5) => Some(DomainObjectKind::Generic),
        // IGeneralService::CreateRequest
        (DomainObjectKind::Generic, 4 | 5) => Some(DomainObjectKind::Generic),
        _ => generic_child_object_kind(command_id),
    }
}

fn acc_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        (DomainObjectKind::GenericRoot, 101) => Some(DomainObjectKind::BaasManager),
        (DomainObjectKind::GenericRoot, _) => generic_child_object_kind(command_id),
        _ => None,
    }
}

fn time_child_object_kind(
    parent_kind: DomainObjectKind,
    command_id: u32,
) -> Option<DomainObjectKind> {
    match (parent_kind, command_id) {
        // IStaticService: cmd 0-4 create child clock objects
        (DomainObjectKind::GenericRoot, 0 | 1 | 2 | 3 | 4) => Some(DomainObjectKind::Generic),
        // cmd 20 (GetSharedMemoryNativeHandle) is NOT a child creation
        _ => None,
    }
}

fn generic_child_object_kind(command_id: u32) -> Option<DomainObjectKind> {
    if matches!(command_id, 0 | 1 | 2 | 3 | 4 | 11 | 20 | 1000) {
        Some(DomainObjectKind::Generic)
    } else {
        None
    }
}

pub(crate) fn handle_sm_send_sync_request(
    kernel: &mut KernelState,
    current_thread_handle: u32,
    memory: &SharedGuestMemory,
    tls: u64,
) -> u64 {
    let Some(request) = parse_hipc_request(memory, tls) else {
        return RESULT_INVALID_MEMORY_STATE;
    };
    let request_context = parse_cmif_context(memory, request);

    match request.message_type {
        HIPC_TYPE_CMIF_CLOSE => {
            if write_simple_hipc_response(memory, tls) {
                RESULT_SUCCESS
            } else {
                RESULT_INVALID_MEMORY_STATE
            }
        }
        HIPC_TYPE_CMIF_CONTROL | HIPC_TYPE_CMIF_CONTROL_WITH_CONTEXT => {
            let Some(command_id) = parse_cmif_command_id(memory, request) else {
                return RESULT_INVALID_MEMORY_STATE;
            };

            let ok = match command_id {
                3 => write_cmif_response(
                    memory,
                    tls,
                    RESULT_SUCCESS as u32,
                    &SM_POINTER_BUFFER_SIZE.to_le_bytes(),
                    None,
                    None,
                    false,
                    &[],
                    request_context,
                ),
                _ => write_cmif_response(
                    memory,
                    tls,
                    RESULT_INVALID_COMBINATION as u32,
                    &[],
                    None,
                    None,
                    false,
                    &[],
                    request_context,
                ),
            };

            if ok {
                RESULT_SUCCESS
            } else {
                RESULT_INVALID_MEMORY_STATE
            }
        }
        HIPC_TYPE_CMIF_REQUEST | HIPC_TYPE_CMIF_REQUEST_WITH_CONTEXT => {
            let Some(command_id) = parse_cmif_command_id(memory, request) else {
                return RESULT_INVALID_MEMORY_STATE;
            };

            let ok = match command_id {
                0 => write_cmif_response(
                    memory,
                    tls,
                    RESULT_SUCCESS as u32,
                    &[],
                    None,
                    None,
                    false,
                    &[],
                    request_context,
                ),
                1 => {
                    let name_address = request.data_start.wrapping_add(CMIF_HEADER_SIZE);
                    let name = read_service_name_8(memory, name_address).unwrap_or_default();

                    if name.is_empty() {
                        eprintln!(
                            "[svc::SendSyncRequest:sm] GetService rejected: empty service name"
                        );
                        write_cmif_response(
                            memory,
                            tls,
                            RESULT_NOT_FOUND as u32,
                            &[],
                            None,
                            None,
                            false,
                            &[],
                            request_context,
                        )
                    } else if let Some(service_handle) =
                        kernel.open_named_service_session(current_thread_handle, &name)
                    {
                        eprintln!(
                            "[svc::SendSyncRequest:sm] GetService name={name:?} -> handle={service_handle}"
                        );
                        write_cmif_response(
                            memory,
                            tls,
                            RESULT_SUCCESS as u32,
                            &[],
                            None,
                            Some(service_handle),
                            false,
                            &[],
                            request_context,
                        )
                    } else {
                        eprintln!(
                            "[svc::SendSyncRequest:sm] GetService name={name:?} -> not found"
                        );
                        write_cmif_response(
                            memory,
                            tls,
                            RESULT_NOT_FOUND as u32,
                            &[],
                            None,
                            None,
                            false,
                            &[],
                            request_context,
                        )
                    }
                }
                _ => write_cmif_response(
                    memory,
                    tls,
                    RESULT_INVALID_COMBINATION as u32,
                    &[],
                    None,
                    None,
                    false,
                    &[],
                    request_context,
                ),
            };

            if ok {
                RESULT_SUCCESS
            } else {
                RESULT_INVALID_MEMORY_STATE
            }
        }
        _ => RESULT_INVALID_MEMORY_STATE,
    }
}

pub(crate) fn handle_generic_send_sync_request(
    kernel: &mut KernelState,
    current_thread_handle: u32,
    memory: &SharedGuestMemory,
    tls: u64,
    session_handle: u32,
    session_kind: ServiceSessionKind,
) -> u64 {
    let Some(request) = parse_hipc_request(memory, tls) else {
        return RESULT_INVALID_MEMORY_STATE;
    };
    let request_context = parse_cmif_context(memory, request);

    let domain_header = parse_domain_header(memory, request);
    let is_domain_message = domain_header.is_some();
    let command_id = parse_cmif_command_id(memory, request);
    if let Some((domain_cmd, object_id)) = domain_header {
        let object_kind = kernel.service_session_domain_object_kind(
            current_thread_handle,
            session_handle,
            object_id,
        );
        let object_kind = object_kind
            .map(|kind| format!("{kind:?}"))
            .unwrap_or_else(|| "Unknown".to_string());
        eprintln!(
            "[svc::SendSyncRequest:generic] kind={session_kind:?} type={} cmd_id=0x{:08x} domain_cmd={} object_id={} object_kind={}",
            request.message_type,
            command_id.unwrap_or(u32::MAX),
            domain_cmd,
            object_id,
            object_kind,
        );
    } else {
        eprintln!(
            "[svc::SendSyncRequest:generic] kind={session_kind:?} type={} cmd_id=0x{:08x}",
            request.message_type,
            command_id.unwrap_or(u32::MAX),
        );
    }
    if request.message_type == HIPC_TYPE_CMIF_REQUEST_WITH_CONTEXT && command_id.is_none() {
        let mut words = Vec::with_capacity(24);
        for index in 0..24 {
            words.push(
                memory
                    .read_u32(tls.wrapping_add((index as u64) * 4))
                    .unwrap_or(0xffff_ffff),
            );
        }
        eprintln!(
            "[svc::SendSyncRequest:generic] parse failed data_start=0x{:016x} data_end=0x{:016x}",
            request.data_start, request.data_end
        );
        eprintln!(
            "[svc::SendSyncRequest:generic] raw tls words: {}",
            words
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    let ok = match request.message_type {
        HIPC_TYPE_CMIF_CLOSE => write_simple_hipc_response(memory, tls),
        HIPC_TYPE_CMIF_CONTROL | HIPC_TYPE_CMIF_CONTROL_WITH_CONTEXT => {
            let Some(command_id) = command_id else {
                return RESULT_INVALID_MEMORY_STATE;
            };

            let mut payload = Vec::new();
            let mut cmif_result = RESULT_SUCCESS as u32;
            let mut move_handle = None;
            match command_id {
                3 => payload.extend_from_slice(&SM_POINTER_BUFFER_SIZE.to_le_bytes()),
                0 => {
                    if let Some(object_id) = kernel
                        .convert_service_session_to_domain(current_thread_handle, session_handle)
                    {
                        payload.extend_from_slice(&object_id.to_le_bytes());
                    } else {
                        cmif_result = RESULT_INVALID_HANDLE as u32;
                    }
                }
                2 | 4 => {
                    // CMIF duplicate-session commands. Ryujinx maps both to a cloned
                    // session handle exposed as a move handle in the response.
                    if let Some(dup_handle) =
                        kernel.clone_service_session(current_thread_handle, session_handle)
                    {
                        move_handle = Some(dup_handle);
                    } else {
                        cmif_result = RESULT_INVALID_HANDLE as u32;
                    }
                }
                _ => {
                    cmif_result = unsupported_control_command(session_kind, command_id);
                }
            }

            write_cmif_response(
                memory,
                tls,
                cmif_result,
                &payload,
                None,
                move_handle,
                false,
                &[],
                request_context,
            )
        }
        HIPC_TYPE_CMIF_REQUEST | HIPC_TYPE_CMIF_REQUEST_WITH_CONTEXT => {
            let mut cmif_result = RESULT_SUCCESS as u32;
            let mut payload = Vec::new();
            let mut copy_handle = None;
            let mut move_handle = None;
            let mut domain_objects = Vec::new();
            let mut trace_target = None;
            let mut trace_non_domain_command_id = None;

            if is_domain_message {
                let Some((domain_cmd, object_id)) = domain_header else {
                    return RESULT_INVALID_MEMORY_STATE;
                };

                if domain_cmd == 2 {
                    let _ = kernel.service_session_remove_domain_object(
                        current_thread_handle,
                        session_handle,
                        object_id,
                    );
                } else {
                    let Some(command_id) = command_id else {
                        return RESULT_INVALID_MEMORY_STATE;
                    };
                    if domain_cmd != 1 {
                        cmif_result = RESULT_INVALID_COMBINATION as u32;
                    } else if let Some(object_kind) = kernel.service_session_domain_object_kind(
                        current_thread_handle,
                        session_handle,
                        object_id,
                    ) {
                        if should_trace_domain_command(session_kind, command_id) {
                            trace_domain_request(
                                session_kind,
                                memory,
                                tls,
                                request,
                                command_id,
                                object_id,
                                object_kind,
                            );
                            trace_target = Some((command_id, object_id, object_kind));
                        }

                        let child_kind = match session_kind {
                            ServiceSessionKind::AppletOE => {
                                appletoe_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Apm => {
                                apm_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Lm => lm_child_object_kind(object_kind, command_id),
                            ServiceSessionKind::Friend => {
                                friend_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::AocU => {
                                aoc_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Pctl => {
                                pctl_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Hid => {
                                hid_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::FspSrv => {
                                fsp_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Sm
                            | ServiceSessionKind::EctxAw
                            | ServiceSessionKind::ViU
                            | ServiceSessionKind::Nvdrv
                            | ServiceSessionKind::AudrenU
                            | ServiceSessionKind::AudOut
                            | ServiceSessionKind::AudOutSession => {
                                generic_child_object_kind(command_id)
                            }
                            ServiceSessionKind::Time => {
                                time_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Nifm => {
                                nifm_child_object_kind(object_kind, command_id)
                            }
                            ServiceSessionKind::Acc => {
                                acc_child_object_kind(object_kind, command_id)
                            }
                        };

                        if let Some(new_kind) = child_kind {
                            if let Some(new_object_id) = kernel.service_session_add_domain_object(
                                current_thread_handle,
                                session_handle,
                                new_kind,
                            ) {
                                domain_objects.push(new_object_id);
                            } else {
                                return RESULT_INVALID_MEMORY_STATE;
                            }
                        } else {
                            match session_kind {
                                ServiceSessionKind::AppletOE => match (object_kind, command_id) {
                                    (DomainObjectKind::CommonStateGetter, 0) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        let Some(event_handle) = copy_handle else {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        };

                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            kernel.applet_message_available(),
                                        );
                                    }
                                    (DomainObjectKind::CommonStateGetter, 1) => {
                                        if let Some(message) = kernel.applet_pop_message() {
                                            payload.extend_from_slice(&message.to_le_bytes());
                                            cmif_result = RESULT_SUCCESS as u32;
                                        } else {
                                            cmif_result = RESULT_AM_NO_MESSAGES;
                                        }

                                        if let Some(event_handle) = kernel
                                            .service_session_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                0,
                                            )
                                        {
                                            let _ = kernel.set_event_signaled_for_handle(
                                                current_thread_handle,
                                                event_handle,
                                                kernel.applet_message_available(),
                                            );
                                        }
                                    }
                                    (DomainObjectKind::CommonStateGetter, 5) => payload.push(0),
                                    (DomainObjectKind::CommonStateGetter, 6) => {
                                        payload.extend_from_slice(&0u32.to_le_bytes())
                                    }
                                    (DomainObjectKind::CommonStateGetter, 8) => payload.push(0),
                                    // GetCurrentFocusState: 1=InFocus, 2=OutOfFocus, 3=Background
                                    (DomainObjectKind::CommonStateGetter, 9) => payload.push(1),
                                    (DomainObjectKind::CommonStateGetter, 60) => {
                                        payload.extend_from_slice(&1280u32.to_le_bytes());
                                        payload.extend_from_slice(&720u32.to_le_bytes());
                                    }
                                    (DomainObjectKind::CommonStateGetter, 300) => payload.push(0),
                                    // GetOperationModeSystemInfo / other CommonStateGetter queries
                                    (DomainObjectKind::CommonStateGetter, 62 | 66) => {
                                        payload.extend_from_slice(&0u32.to_le_bytes())
                                    }
                                    (DomainObjectKind::WindowController, 1) => {
                                        let applet_resource_user_id = if let Some(id) = kernel
                                            .service_session_applet_resource_user_id(
                                                current_thread_handle,
                                                session_handle,
                                            ) {
                                            id
                                        } else {
                                            let fallback = kernel
                                                .process_pid_for_handle(
                                                    current_thread_handle,
                                                    SELF_PROCESS_HANDLE,
                                                )
                                                .unwrap_or(1);
                                            let _ = kernel
                                                .service_session_set_applet_resource_user_id(
                                                    current_thread_handle,
                                                    session_handle,
                                                    fallback,
                                                );
                                            fallback
                                        };
                                        payload.extend_from_slice(
                                            &applet_resource_user_id.to_le_bytes(),
                                        );
                                    }
                                    (DomainObjectKind::CommonStateGetter, 13 | 61)
                                    | (DomainObjectKind::ApplicationFunctions, 130) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        if copy_handle.is_none() {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        }
                                    }
                                    // Ryujinx tracks this as an internal latch but it has no
                                    // externally visible effect in current app flow.
                                    (DomainObjectKind::CommonStateGetter, 900) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    (DomainObjectKind::SelfController, 9 | 91) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        let Some(event_handle) = copy_handle else {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        };

                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            true,
                                        );
                                    }
                                    // Matches Ryujinx: accepts three bools and stores no
                                    // externally visible state in this path.
                                    (DomainObjectKind::SelfController, 13) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // SetOperationModeChangedNotification /
                                    // SetPerformanceModeChangedNotification.
                                    (DomainObjectKind::SelfController, 11 | 12) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // Matches Ryujinx: toggles internal suspend policy.
                                    (DomainObjectKind::SelfController, 16) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // SetIdleTimeDetectionExtension(u32) is accepted as success.
                                    // (Note: cmd 19 = SetIdleTimeDetectionExtension per Horizon,
                                    //  cmd 44 = CreateManagedDisplaySeparableLayer)
                                    (DomainObjectKind::SelfController, 19) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // CreateManagedDisplaySeparableLayer() -> (u64 display_layer_id, u64 recording_layer_id)
                                    (DomainObjectKind::SelfController, 44) => {
                                        consume_cmif_payload(memory, request, true);
                                        let display_layer_id = kernel.allocate_display_layer_id();
                                        let recording_layer_id = kernel.allocate_display_layer_id();
                                        payload.extend_from_slice(&display_layer_id.to_le_bytes());
                                        payload
                                            .extend_from_slice(&recording_layer_id.to_le_bytes());
                                    }
                                    // CreateManagedDisplayLayer() -> u64 layer_id
                                    (DomainObjectKind::SelfController, 40) => {
                                        consume_cmif_payload(memory, request, true);
                                        let layer_id = kernel.allocate_display_layer_id();
                                        payload.extend_from_slice(&layer_id.to_le_bytes());
                                    }
                                    // AcquireForegroundRights() is accepted as success.
                                    (DomainObjectKind::WindowController, 10) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // SetTerminateResult(nn::Result) is informational in this
                                    // flow; keep success semantics like Ryujinx.
                                    (DomainObjectKind::ApplicationFunctions, 22) => {
                                        let req_payload =
                                            read_cmif_payload_bytes(memory, request, true);
                                        let term_result =
                                            read_payload_u32(&req_payload, 0).unwrap_or(0xFFFFFFFF);
                                        eprintln!(
                                            "[AppletOE:ApplicationFunctions] SetTerminateResult value=0x{term_result:08x} ({term_result})"
                                        );
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // appletNotifyRunning() expects a bool response. Return true
                                    // to indicate the application is now running.
                                    (DomainObjectKind::ApplicationFunctions, 40) => payload.push(1),
                                    // GetDesiredLanguage() -> nn::settings::LanguageCode (u64)
                                    // Return "en-US" as the default language.
                                    (DomainObjectKind::ApplicationFunctions, 21) => {
                                        // LanguageCode is an 8-byte null-terminated ASCII string.
                                        let mut lang = [0u8; 8];
                                        let s = b"en-US";
                                        lang[..s.len()].copy_from_slice(s);
                                        payload.extend_from_slice(&lang);
                                    }
                                    // PopLaunchParameter(kind) -> IStorage
                                    // The child AmStorage object is created automatically.
                                    // kind=1 (AccountPreselectedUser) is the common case.
                                    (DomainObjectKind::ApplicationFunctions, 1) => {
                                        let plp_payload =
                                            read_cmif_payload_bytes(memory, request, true);
                                        let kind =
                                            read_payload_u32(&plp_payload, 0).unwrap_or(u32::MAX);
                                        eprintln!(
                                            "[PopLaunchParameter] kind={} payload_len={}",
                                            kind,
                                            plp_payload.len()
                                        );
                                    }
                                    // IStorage::Open() -> IStorageAccessor
                                    // Child AmStorageAccessor is created automatically.
                                    (DomainObjectKind::AmStorage, 0) => {}
                                    // IStorageAccessor::GetSize() -> s64
                                    (DomainObjectKind::AmStorageAccessor, 0) => {
                                        // Preselected user data is 0x88 bytes
                                        payload.extend_from_slice(&0x88i64.to_le_bytes())
                                    }
                                    // IStorageAccessor::Read(s64 offset, s64 length) -> buffer
                                    // OR Write that has recv descriptors (game may use either cmd 10/11
                                    // with AutoSelect pointer mode C descriptor for reading storage data).
                                    (DomainObjectKind::AmStorageAccessor, 10)
                                    | (DomainObjectKind::AmStorageAccessor, 11) => {
                                        // Write preselected user launch parameter to the output buffer.
                                        // PreselectedUser format (0x88 bytes):
                                        //   [0x00] u32 magic = 0xC79497CA
                                        //   [0x04] u32 unk = 1
                                        //   [0x08] u128 uid = {1, 0}
                                        //   [0x18..0x88] padding zeros
                                        let descriptor_layout =
                                            parse_hipc_buffer_layout(memory, tls)
                                                .unwrap_or_default();
                                        let out_desc = descriptor_layout
                                            .recv_buffers
                                            .iter()
                                            .copied()
                                            .find(|d| descriptor_has_data(*d))
                                            .or_else(|| {
                                                descriptor_layout
                                                    .recv_statics
                                                    .iter()
                                                    .copied()
                                                    .find(|d| descriptor_has_data(*d))
                                            });
                                        if let Some(desc) = out_desc {
                                            let size = desc.size.min(0x88) as usize;
                                            let mut buf = vec![0u8; size];
                                            if size >= 0x18 {
                                                // Magic
                                                buf[0..4]
                                                    .copy_from_slice(&0xC79497CAu32.to_le_bytes());
                                                // Version
                                                buf[4..8].copy_from_slice(&1u32.to_le_bytes());
                                                // UID low (dummy user id = 1)
                                                buf[8..16].copy_from_slice(&1u64.to_le_bytes());
                                                // UID high
                                                buf[16..24].copy_from_slice(&0u64.to_le_bytes());
                                            }
                                            let _ = write_guest_buffer(memory, desc.address, &buf);
                                        }
                                    }
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::Apm => match (object_kind, command_id) {
                                    (DomainObjectKind::ApmManager, 1)
                                    | (DomainObjectKind::ApmSession, 1) => {
                                        payload.extend_from_slice(&0u32.to_le_bytes())
                                    }
                                    (DomainObjectKind::ApmManager, 6) => payload.push(0),
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::Friend => match (object_kind, command_id) {
                                    (DomainObjectKind::FriendService, 0) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        let Some(event_handle) = copy_handle else {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        };

                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            true,
                                        );
                                    }
                                    (DomainObjectKind::FriendService, 10_120)
                                    | (DomainObjectKind::FriendService, 10_420) => payload.push(1),
                                    (DomainObjectKind::FriendService, 10_121)
                                    | (DomainObjectKind::FriendService, 10_421) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::EctxAw => match (object_kind, command_id) {
                                    // IContextRegistrar::Complete(nn::Result, raw_context)
                                    // returns a context descriptor (0 in Ryujinx stub).
                                    (DomainObjectKind::Generic, 0) => {
                                        payload.extend_from_slice(&0u32.to_le_bytes())
                                    }
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::AocU => match (object_kind, command_id) {
                                    // CountAddOnContent(pid) / ListAddOnContent(start, count, pid)
                                    (DomainObjectKind::GenericRoot, 2 | 3) => {
                                        payload.extend_from_slice(&0u32.to_le_bytes())
                                    }
                                    // GetAddOnContentBaseId(pid)
                                    (DomainObjectKind::GenericRoot, 5) => {
                                        payload.extend_from_slice(&0u64.to_le_bytes())
                                    }
                                    // PrepareAddOnContent(index, pid), mount status checks.
                                    (DomainObjectKind::GenericRoot, 7 | 11 | 12 | 50) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // GetAddOnContentListChangedEvent{WithProcessId}
                                    (DomainObjectKind::GenericRoot, 8 | 10) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        let Some(event_handle) = copy_handle else {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        };

                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            false,
                                        );
                                    }
                                    // GetAddOnContentLostErrorCode() (Ryujinx uses 2164-1000).
                                    (DomainObjectKind::GenericRoot, 9) => payload
                                        .extend_from_slice(&0x0000_03e8_0000_0874u64.to_le_bytes()),
                                    // IPurchaseEventManager::SetDefaultDeliveryTarget
                                    (DomainObjectKind::Generic, 0) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // IPurchaseEventManager::GetPurchasedEventReadableHandle
                                    (DomainObjectKind::Generic, 2) => {
                                        copy_handle = kernel
                                            .service_session_get_or_create_event_handle(
                                                current_thread_handle,
                                                session_handle,
                                                object_id,
                                                command_id,
                                            );
                                        let Some(event_handle) = copy_handle else {
                                            return RESULT_INVALID_MEMORY_STATE;
                                        };

                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            false,
                                        );
                                    }
                                    // IPurchaseEventManager::PopPurchasedProductInfo
                                    (DomainObjectKind::Generic, 3) => payload.resize(0x80, 0),
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::Pctl => match (object_kind, command_id) {
                                    // IParentalControlService::Initialize
                                    (DomainObjectKind::Generic, 1) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // Free communication and stereo/feature checks.
                                    (
                                        DomainObjectKind::Generic,
                                        1001 | 1013 | 1017 | 1018 | 1061 | 1064,
                                    ) => consume_cmif_payload(memory, request, true),
                                    // IsRestrictionEnabled / GetStereoVisionRestriction.
                                    (DomainObjectKind::Generic, 1031 | 1062) => payload.push(0),
                                    // IsStereoVisionPermitted.
                                    (DomainObjectKind::Generic, 1065) => payload.push(1),
                                    // SetStereoVisionRestriction(bool)
                                    (DomainObjectKind::Generic, 1063) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::Lm
                                | ServiceSessionKind::Sm
                                | ServiceSessionKind::ViU
                                | ServiceSessionKind::Nvdrv
                                | ServiceSessionKind::AudrenU
                                | ServiceSessionKind::AudOut
                                | ServiceSessionKind::AudOutSession => {
                                    cmif_result = unsupported_domain_command(
                                        session_kind,
                                        object_kind,
                                        command_id,
                                    );
                                }
                                ServiceSessionKind::Time => {
                                    match (object_kind, command_id) {
                                        // IStaticService: cmd 0-4 create child clocks
                                        // (handled by generic_child_object_kind)
                                        // cmd 20: GetSharedMemoryNativeHandle
                                        (DomainObjectKind::GenericRoot, 20) => {
                                            copy_handle = create_time_shared_memory_handle(
                                                kernel,
                                                current_thread_handle,
                                            );
                                            if copy_handle.is_none() {
                                                cmif_result = RESULT_OUT_OF_MEMORY as u32;
                                            }
                                        }
                                        // ISteadyClock/ISystemClock::GetCurrentTimePoint
                                        // Returns SteadyClockTimePoint {u64 time, u128 source_id}
                                        (DomainObjectKind::Generic, 0) => {
                                            // timestamp = 0 seconds since epoch
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                            // source ID (UUID, use a fixed dummy)
                                            payload.extend_from_slice(&1u64.to_le_bytes());
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                        }
                                        // ISystemClock::GetSystemClockContext
                                        (DomainObjectKind::Generic, 1) => {
                                            // offset = 0
                                            payload.extend_from_slice(&0i64.to_le_bytes());
                                            // source stamp (SteadyClockTimePoint)
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                            payload.extend_from_slice(&1u64.to_le_bytes());
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                        }
                                        _ => {
                                            // Accept unknown Time commands silently
                                            consume_cmif_payload(memory, request, true);
                                        }
                                    }
                                }
                                ServiceSessionKind::Nifm => {
                                    // Nifm domain commands - stub network as offline
                                    match (object_kind, command_id) {
                                        // IStaticService::CreateGeneralService(u32)
                                        // Child creation handled by nifm_child_object_kind
                                        (DomainObjectKind::GenericRoot, 4 | 5) => {
                                            consume_cmif_payload(memory, request, true);
                                        }
                                        // IGeneralService::GetClientId -> u32
                                        (DomainObjectKind::Generic, 1) => {
                                            payload.extend_from_slice(&1u32.to_le_bytes());
                                        }
                                        // IGeneralService::CreateRequest(u32) -> IRequest
                                        // Child creation handled by nifm_child_object_kind
                                        (DomainObjectKind::Generic, 4 | 5) => {
                                            consume_cmif_payload(memory, request, true);
                                        }
                                        // IGeneralService::GetCurrentIpAddress -> u32
                                        (DomainObjectKind::Generic, 12) => {
                                            payload.extend_from_slice(&0u32.to_le_bytes());
                                        }
                                        // IGeneralService::GetInternetConnectionStatus
                                        (DomainObjectKind::Generic, 15) => {
                                            // type=0(none), wifiStrength=0, connectionStatus=0(idle)
                                            payload.push(0);
                                            payload.push(0);
                                            payload.push(0);
                                        }
                                        // IGeneralService::IsAnyForegroundRequestAccepted -> bool
                                        (DomainObjectKind::Generic, 18) => {
                                            payload.push(0);
                                        }
                                        // IRequest::GetRequestState -> u32 (3=errored/not connected)
                                        (DomainObjectKind::Generic, 0) => {
                                            // When used as IRequest::GetRequestState
                                            // 0=not submitted, 1=connecting, 2=connected, 3=errored
                                            payload.extend_from_slice(&3u32.to_le_bytes());
                                        }
                                        _ => {
                                            cmif_result = unsupported_domain_command(
                                                session_kind,
                                                object_kind,
                                                command_id,
                                            );
                                        }
                                    }
                                }
                                ServiceSessionKind::Acc => match (object_kind, command_id) {
                                    // GetUserCount() -> u32
                                    (DomainObjectKind::GenericRoot, 0) => {
                                        payload.extend_from_slice(&1u32.to_le_bytes())
                                    }
                                    // GetUserExistence(nn::account::Uid) -> bool
                                    (DomainObjectKind::GenericRoot, 1) => {
                                        consume_cmif_payload(memory, request, true);
                                        payload.push(1);
                                    }
                                    // ListAllUsers() / ListOpenUsers() -> array of Uid (up to 8)
                                    (DomainObjectKind::GenericRoot, 2 | 3) => {
                                        // Return one dummy user UID (128-bit)
                                        payload.extend_from_slice(&1u64.to_le_bytes());
                                        payload.extend_from_slice(&0u64.to_le_bytes());
                                        // Pad remaining 7 slots with zeros
                                        for _ in 0..7 {
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                            payload.extend_from_slice(&0u64.to_le_bytes());
                                        }
                                    }
                                    // GetLastOpenedUser() -> nn::account::Uid
                                    (DomainObjectKind::GenericRoot, 4) => {
                                        payload.extend_from_slice(&1u64.to_le_bytes());
                                        payload.extend_from_slice(&0u64.to_le_bytes());
                                    }
                                    // InitializeApplicationInfo(pid) / InitializeApplicationInfoV2(pid)
                                    (DomainObjectKind::GenericRoot, 100 | 140) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // IsNetworkServiceAccountAvailable(Uid) -> bool
                                    (DomainObjectKind::GenericRoot, 141) => {
                                        consume_cmif_payload(memory, request, true);
                                        payload.push(0);
                                    }
                                    // TrySelectUserWithoutInteraction(bool) -> nn::account::Uid
                                    (DomainObjectKind::GenericRoot, 150) => {
                                        consume_cmif_payload(memory, request, true);
                                        payload.extend_from_slice(&1u64.to_le_bytes());
                                        payload.extend_from_slice(&0u64.to_le_bytes());
                                    }
                                    // StoreSaveDataThumbnail / ClearSaveDataThumbnail
                                    (DomainObjectKind::GenericRoot, 110 | 111) => {
                                        consume_cmif_payload(memory, request, true)
                                    }
                                    // GetBaasAccountManagerForApplication(Uid)
                                    // Child BaasManager created by acc_child_object_kind.
                                    (DomainObjectKind::GenericRoot, 101) => {
                                        consume_cmif_payload(memory, request, true);
                                    }
                                    // IManagerForApplication::CheckAvailability
                                    (DomainObjectKind::BaasManager, 0) => {}
                                    // IManagerForApplication::GetAccountId -> u64
                                    (DomainObjectKind::BaasManager, 1) => {
                                        payload.extend_from_slice(&0x1234u64.to_le_bytes());
                                    }
                                    _ => {
                                        cmif_result = unsupported_domain_command(
                                            session_kind,
                                            object_kind,
                                            command_id,
                                        );
                                    }
                                },
                                ServiceSessionKind::Hid => match (object_kind, command_id) {
                                    // IAppletResource::GetSharedMemoryHandle()
                                    (DomainObjectKind::Generic, 0) => {
                                        copy_handle = create_hid_shared_memory_handle(
                                            kernel,
                                            current_thread_handle,
                                        );
                                        if let Some(h) = copy_handle {
                                            kernel.set_hid_shared_memory_handle(h);
                                        } else {
                                            cmif_result = RESULT_OUT_OF_MEMORY as u32;
                                        }
                                    }
                                    // Keep remaining HID commands as accepted no-ops for now.
                                    _ => consume_cmif_payload(memory, request, true),
                                },
                                ServiceSessionKind::FspSrv => {
                                    match (object_kind, command_id) {
                                        // IFileSystemProxy::SetCurrentProcess([pid])
                                        // Store the caller pid when present to mirror service
                                        // bootstrap flow used by libnx/Ryujinx.
                                        (DomainObjectKind::GenericRoot, 1) => {
                                            let request_payload =
                                                read_cmif_payload_bytes(memory, request, true);
                                            let pid = parse_hipc_special_header(memory, tls)
                                                .and_then(|special| special.pid)
                                                .or_else(|| read_payload_u64(&request_payload, 0));
                                            if let Some(pid) = pid {
                                                let _ = kernel
                                                    .service_session_set_applet_resource_user_id(
                                                        current_thread_handle,
                                                        session_handle,
                                                        pid,
                                                    );
                                            }
                                        }
                                        // IFileSystemProxy::SetGlobalAccessLogMode(u32)
                                        (DomainObjectKind::GenericRoot, 1004) => {
                                            let request_payload =
                                                read_cmif_payload_bytes(memory, request, true);
                                            let mode =
                                                read_payload_u32(&request_payload, 0).unwrap_or(0);
                                            fsp_global_access_log_mode()
                                                .store(mode, Ordering::Release);
                                        }
                                        // IFileSystemProxy::GetGlobalAccessLogMode() -> u32
                                        (DomainObjectKind::GenericRoot, 1005) => {
                                            let mode = fsp_global_access_log_mode()
                                                .load(Ordering::Acquire);
                                            payload.extend_from_slice(&mode.to_le_bytes());
                                        }
                                        // IFileSystemProxy::OutputAccessLogToSdCard(buffer)
                                        (DomainObjectKind::GenericRoot, 1006) => {
                                            consume_cmif_payload(memory, request, true)
                                        }
                                        // IFileSystemProxy::OpenDataStorageByCurrentProcess()
                                        // Returns an IStorage object for reading RomFS data.
                                        // The child domain object (FspStorage) is created
                                        // automatically; we just accept the command here.
                                        (DomainObjectKind::GenericRoot, 200) => {
                                            consume_cmif_payload(memory, request, true)
                                        }
                                        // IFileSystemProxy::OpenDataStorageByDataId(u8,u64)
                                        (DomainObjectKind::GenericRoot, 202) => {
                                            consume_cmif_payload(memory, request, true)
                                        }
                                        // IFileSystemProxy::OpenSaveDataFileSystem(u8, SaveDataAttribute) -> IFileSystem
                                        // Returns a new IFileSystem domain object for save data.
                                        (DomainObjectKind::GenericRoot, 51) => {
                                            consume_cmif_payload(memory, request, true);
                                        }
                                        // IFileSystemProxy::OpenSaveDataFileSystemBySystemSaveDataId
                                        // Accept as no-ops returning domain objects (handled by child alloc).
                                        (
                                            DomainObjectKind::GenericRoot,
                                            52 | 53 | 58 | 59 | 60 | 61,
                                        ) => {
                                            consume_cmif_payload(memory, request, true);
                                        }
                                        // IFileSystemProxy::GetFreeSpaceSizeForSaveData(u64) -> s64
                                        (DomainObjectKind::GenericRoot, 63) => {
                                            consume_cmif_payload(memory, request, true);
                                            // Return ~8GB of free space as a stub.
                                            payload.extend_from_slice(
                                                &0x0000_0002_0000_0000i64.to_le_bytes(),
                                            );
                                        }
                                        // IStorage::Read(s64 offset, u64 length) -> buffer<bytes>
                                        // Reads from the decrypted RomFS data.
                                        (DomainObjectKind::FspStorage, 0) => {
                                            let request_payload =
                                                read_cmif_payload_bytes(memory, request, true);
                                            let read_offset = read_payload_u64(&request_payload, 0)
                                                .unwrap_or(0)
                                                as usize;
                                            let read_length = read_payload_u64(&request_payload, 8)
                                                .unwrap_or(0)
                                                as usize;

                                            // Find the recv buffer (Type-B) where we write output.
                                            let buf_layout = parse_hipc_buffer_layout(memory, tls)
                                                .unwrap_or_default();
                                            let recv_buf = buf_layout
                                                .recv_buffers
                                                .first()
                                                .copied()
                                                .unwrap_or_default();

                                            eprintln!(
                                                "[FspStorage::Read] tid={} obj={} offset={:#x} len={:#x} recv_addr={:#x} recv_size={:#x}",
                                                current_thread_handle,
                                                object_id,
                                                read_offset,
                                                read_length,
                                                recv_buf.address,
                                                recv_buf.size,
                                            );

                                            if let Some(ref romfs) = kernel.romfs_data {
                                                let romfs_len = romfs.len();
                                                if read_offset < romfs_len {
                                                    let available =
                                                        romfs_len.saturating_sub(read_offset);
                                                    let to_copy = read_length
                                                        .min(available)
                                                        .min(recv_buf.size as usize);
                                                    let slice =
                                                        &romfs[read_offset..read_offset + to_copy];
                                                    write_guest_buffer(
                                                        memory,
                                                        recv_buf.address,
                                                        slice,
                                                    );
                                                } else {
                                                    eprintln!(
                                                        "[FspStorage::Read] offset {read_offset:#x} \
                                                         beyond RomFS size {romfs_len:#x}"
                                                    );
                                                }
                                            } else {
                                                eprintln!(
                                                    "[FspStorage::Read] called but no RomFS data loaded"
                                                );
                                            }
                                        }
                                        // IStorage::Write / Flush / SetSize - accept as no-ops
                                        (DomainObjectKind::FspStorage, 1 | 2 | 3) => {
                                            consume_cmif_payload(memory, request, true)
                                        }
                                        // IStorage::GetSize() -> s64
                                        (DomainObjectKind::FspStorage, 4) => {
                                            let size: i64 = kernel
                                                .romfs_data
                                                .as_ref()
                                                .map(|d| d.len() as i64)
                                                .unwrap_or(0);
                                            eprintln!(
                                                "[FspStorage::GetSize] tid={} obj={} size={:#x}",
                                                current_thread_handle, object_id, size
                                            );
                                            payload.extend_from_slice(&size.to_le_bytes())
                                        }
                                        // IFileSystem metadata and mutating operations.
                                        // These are accepted as no-op stubs for now.
                                        (
                                            DomainObjectKind::Generic,
                                            0 | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 13,
                                        ) => consume_cmif_payload(memory, request, true),
                                        // IFileSystem::GetEntryType(path) -> FsDirEntryType
                                        (DomainObjectKind::Generic, 7) => {
                                            payload.extend_from_slice(&0u32.to_le_bytes())
                                        }
                                        // IFileSystem::OpenFile(path, mode)
                                        (DomainObjectKind::Generic, 8) => {
                                            consume_cmif_payload(memory, request, true);
                                        }
                                        // IFileSystem::{GetFreeSpaceSize,GetTotalSpaceSize}
                                        (DomainObjectKind::Generic, 11 | 12) => payload
                                            .extend_from_slice(
                                                &0x0000_0002_0000_0000u64.to_le_bytes(),
                                            ),
                                        _ => {
                                            cmif_result = unsupported_domain_command(
                                                session_kind,
                                                object_kind,
                                                command_id,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        cmif_result = RESULT_INVALID_HANDLE as u32;
                    }
                }
            }

            if !is_domain_message {
                let Some(command_id) = command_id else {
                    return RESULT_INVALID_MEMORY_STATE;
                };

                if should_trace_non_domain_command(session_kind, command_id) {
                    trace_non_domain_request(session_kind, memory, tls, request, command_id);
                    trace_non_domain_command_id = Some(command_id);
                }

                if session_kind == ServiceSessionKind::Apm {
                    let request_payload = read_cmif_payload_bytes(memory, request, false);

                    match command_id {
                        0 => {
                            if request_payload.len() >= 8 {
                                let mode = read_payload_u32(&request_payload, 0).unwrap_or(0);
                                let configuration =
                                    read_payload_u32(&request_payload, 4).unwrap_or(0);
                                apm_write_performance_configuration(mode, configuration);
                            } else if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        1 => {
                            if request_payload.len() >= 4 {
                                let mode = read_payload_u32(&request_payload, 0).unwrap_or(0);
                                let configuration = apm_read_performance_configuration(mode);
                                payload.extend_from_slice(&configuration.to_le_bytes());
                            } else {
                                let mode = apm_current_performance_mode().load(Ordering::Acquire);
                                payload.extend_from_slice(&mode.to_le_bytes());
                            }
                        }
                        6 => payload.push(0),
                        _ => {
                            cmif_result = unsupported_non_domain_command(session_kind, command_id);
                        }
                    }
                }

                if session_kind == ServiceSessionKind::ViU {
                    let request_payload = read_cmif_payload_bytes(memory, request, false);
                    let vi_role = kernel
                        .service_session_vi_role(current_thread_handle, session_handle)
                        .unwrap_or(ViSessionRole::Root);

                    eprintln!(
                        "[vi] handle={} cmd={} vi_role={:?}",
                        session_handle, command_id, vi_role
                    );

                    match command_id {
                        // vi root cmd 0: GetNativeHandle() capability query.
                        // IHOSBinderDriver cmd 0/3: TransactParcel/TransactParcelAuto.
                        0 | 3 => {
                            if matches!(
                                vi_role,
                                ViSessionRole::BinderRelay | ViSessionRole::BinderIndirect
                            ) {
                                let transaction_id = read_payload_u32(&request_payload, 4).unwrap_or(0);
                                eprintln!(
                                    "[vi:binder] transaction_id={} payload_len={}",
                                    transaction_id, request_payload.len()
                                );

                                // Handle stateful binder transactions that need
                                // kernel state and/or incoming parcel data.
                                let reply_payload = match transaction_id {
                                    // SetPreallocatedBuffer: store the GraphicBuffer
                                    14 => {
                                        if let Some(parcel_data) =
                                            vi_read_binder_parcel(memory, tls)
                                        {
                                            vi_handle_set_preallocated_buffer(
                                                kernel,
                                                &parcel_data,
                                            );
                                        }
                                        vi_binder_reply_payload_for_transaction(14)
                                    }
                                    // RequestBuffer: return stored GraphicBuffer
                                    1 => vi_handle_request_buffer(kernel, memory, tls),
                                    // DequeueBuffer: return the next available slot
                                    3 => vi_handle_dequeue_buffer(kernel, memory, tls),
                                    // Query: return display properties
                                    9 => vi_handle_query(memory, tls),
                                    // QueueBuffer: handle buffer submission
                                    7 => vi_handle_queue_buffer(kernel, memory, tls),
                                    other => vi_binder_reply_payload_for_transaction(other),
                                };

                                if vi_write_parcel_to_out_buffer(memory, tls, &reply_payload)
                                    .is_none()
                                {
                                    cmif_result = RESULT_INVALID_MEMORY_STATE as u32;
                                }

                            } else if command_id == 0 {
                                payload.extend_from_slice(
                                    &u32::from(kernel.has_gpu_backend()).to_le_bytes(),
                                );
                            } else if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                let _ = kernel.service_session_set_vi_role(
                                    current_thread_handle,
                                    handle,
                                    ViSessionRole::ApplicationDisplay,
                                );
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // IHOSBinderDriver::AdjustRefcount(session_id, addval, type).
                        // vi:s/vi:m root cmd 1 can also return a display service handle.
                        1 => {
                            if !matches!(
                                vi_role,
                                ViSessionRole::BinderRelay | ViSessionRole::BinderIndirect
                            ) {
                                if let Some(handle) =
                                    kernel.clone_service_session(current_thread_handle, session_handle)
                                {
                                    let _ = kernel.service_session_set_vi_role(
                                        current_thread_handle,
                                        handle,
                                        ViSessionRole::ApplicationDisplay,
                                    );
                                    move_handle = Some(handle);
                                } else {
                                    cmif_result = RESULT_INVALID_HANDLE as u32;
                                }
                            }
                        }
                        // vi:*::GetDisplayService(policy) -> IApplicationDisplayService
                        // IHOSBinderDriver::GetNativeHandle(session_id, type_id) -> copy handle
                        2 => {
                            if matches!(
                                vi_role,
                                ViSessionRole::BinderRelay | ViSessionRole::BinderIndirect
                            ) {
                                let binder_id = read_payload_u32(&request_payload, 0).unwrap_or(0);
                                let type_id = read_payload_u32(&request_payload, 4).unwrap_or(0);
                                copy_handle = kernel.service_session_get_or_create_event_handle(
                                    current_thread_handle,
                                    session_handle,
                                    binder_id,
                                    type_id,
                                );
                                if let Some(event_handle) = copy_handle {
                                    // Signal the VSync/binder event so the render
                                    // thread unblocks and starts presenting frames.
                                    let ok = kernel.set_event_signaled_for_handle(
                                        current_thread_handle,
                                        event_handle,
                                        true,
                                    );
                                    // Register for periodic VSync signaling.
                                    if let Some(process_handle) = kernel.process_handle_for_thread(current_thread_handle) {
                                        kernel.register_binder_vsync_event(process_handle, event_handle);
                                    }
                                    eprintln!("[vi:GetNativeHandle] binder_id={} type_id={} event_handle={} signal_ok={}", binder_id, type_id, event_handle, ok);
                                } else {
                                    cmif_result = RESULT_INVALID_HANDLE as u32;
                                }
                            } else if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                let _ = kernel.service_session_set_vi_role(
                                    current_thread_handle,
                                    handle,
                                    ViSessionRole::ApplicationDisplay,
                                );
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // IApplicationDisplayService sub-service getters.
                        100 | 101 | 102 | 103 => {
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                let role = match command_id {
                                    100 => ViSessionRole::BinderRelay,
                                    101 => ViSessionRole::SystemDisplay,
                                    102 => ViSessionRole::ManagerDisplay,
                                    103 => ViSessionRole::BinderIndirect,
                                    _ => unreachable!("invalid vi sub-service command id"),
                                };
                                let _ = kernel.service_session_set_vi_role(
                                    current_thread_handle,
                                    handle,
                                    role,
                                );
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // ISystemDisplayService::OpenDisplay(name) -> display_id
                        1010 => {
                            payload.extend_from_slice(&1u64.to_le_bytes());
                        }
                        // IApplicationDisplayService::OpenDefaultDisplay() -> display_id
                        1011 => payload.extend_from_slice(&1u64.to_le_bytes()),
                        // IApplicationDisplayService::CloseDisplay(display_id)
                        1020
                        // IApplicationDisplayService::SetDisplayEnabled(state, display_id)
                        | 1101
                        // IApplicationDisplayService::CloseLayer(layer_id)
                        | 2021
                        // IApplicationDisplayService::DestroyStrayLayer(layer_id)
                        | 2031
                        // IApplicationDisplayService::SetLayerScalingMode(mode, layer_id)
                        | 2101
                        // IApplicationDisplayService::ConvertScalingMode(mode)
                        | 2102
                        // ISystemDisplayService layer-state mutators used by libnx flow.
                        | 2201
                        | 2203
                        | 2204
                        | 2205
                        | 2207
                        | 2209
                        | 2210 => {
                        }
                        // IApplicationDisplayService::GetDisplayResolution(display_id)
                        // returns fixed undocked dimensions like Ryujinx stubs.
                        1102 => {
                            payload.extend_from_slice(&1280u64.to_le_bytes());
                            payload.extend_from_slice(&720u64.to_le_bytes());
                        }
                        // IApplicationDisplayService::OpenLayer(display_name, layer_id, aruid)
                        // -> native window out-buffer + serialized size.
                        2020 => {
                            let binder_id = vi_next_binder_id();
                            let native_window_size =
                                vi_write_native_window_parcel(memory, tls, binder_id);
                            if let Some(size) = native_window_size {
                                payload.extend_from_slice(&size.to_le_bytes());
                            } else {
                                cmif_result = RESULT_INVALID_MEMORY_STATE as u32;
                            }
                        }
                        // IApplicationDisplayService::CreateStrayLayer(flags, display_id)
                        // -> (layer_id, native_window_size) plus out native-window buffer.
                        2030 => {
                            let layer_id = vi_next_stray_layer_id();
                            payload.extend_from_slice(&layer_id.to_le_bytes());

                            let binder_id = vi_next_binder_id();
                            let native_window_size =
                                vi_write_native_window_parcel(memory, tls, binder_id);
                            if let Some(size) = native_window_size {
                                payload.extend_from_slice(&size.to_le_bytes());
                            } else {
                                cmif_result = RESULT_INVALID_MEMORY_STATE as u32;
                            }
                        }
                        // ISystemDisplayService::CreateStrayLayer(layer_flags, display_id)
                        // -> (layer_id, native_window_size) with an out native-window buffer.
                        2012 | 2312 => {
                            let layer_id = vi_next_stray_layer_id();
                            payload.extend_from_slice(&layer_id.to_le_bytes());

                            let binder_id = vi_next_binder_id();
                            let native_window_size =
                                vi_write_native_window_parcel(memory, tls, binder_id);
                            if let Some(size) = native_window_size {
                                payload.extend_from_slice(&size.to_le_bytes());
                            } else {
                                cmif_result = RESULT_INVALID_MEMORY_STATE as u32;
                            }
                        }
                        _ => {
                            cmif_result = unsupported_non_domain_command(session_kind, command_id);
                        }
                    }
                }

                if session_kind == ServiceSessionKind::EctxAw {
                    match command_id {
                        // CreateContextRegistrar(). Domain clients usually consume this via
                        // object creation on the converted root object.
                        0 => consume_cmif_payload(memory, request, false),
                        _ => {
                            cmif_result = unsupported_non_domain_command(session_kind, command_id);
                        }
                    }
                }

                // audren:u (IAudioRendererManager) and IAudioDevice (cloned sessions)
                if session_kind == ServiceSessionKind::AudrenU {
                    match command_id {
                        // OpenAudioRenderer(AudioRendererParameterInternal, ...) -> IAudioRenderer
                        0 => {
                            consume_cmif_payload(memory, request, false);
                            // Return a cloned session handle as the IAudioRenderer session.
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // GetWorkBufferSize(AudioRendererParameterInternal) -> u64
                        1 => {
                            consume_cmif_payload(memory, request, false);
                            // Return a reasonable work buffer size (4 MiB).
                            payload.extend_from_slice(&(4u64 * 1024 * 1024).to_le_bytes());
                        }
                        // GetAudioDeviceService(applet_resource_user_id)
                        // GetAudioDeviceServiceWithRevisionInfo(applet_resource_user_id, revision)
                        2 | 4 => {
                            consume_cmif_payload(memory, request, false);
                            // Return a cloned session handle as the IAudioDevice session.
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // IAudioDevice: ListAudioDevices / ListAudioDevicesAuto
                        6 => {
                            consume_cmif_payload(memory, request, false);
                            // Write device name "AudioTvOutput" to the recv buffer.
                            let descriptor_layout =
                                parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
                            let recv = descriptor_layout
                                .recv_buffers
                                .first()
                                .copied()
                                .or_else(|| descriptor_layout.recv_statics.first().copied())
                                .unwrap_or_default();
                            let device_name = b"AudioTvOutput\0";
                            if recv.address != 0 && recv.size >= device_name.len() as u64 {
                                write_guest_buffer(memory, recv.address, device_name);
                            }
                            // Return count = 1
                            payload.extend_from_slice(&1u32.to_le_bytes());
                        }
                        // IAudioDevice: GetActiveAudioDeviceName / GetActiveAudioDeviceNameAuto /
                        // GetActiveAudioOutputDeviceName (cmd 3/9/12)
                        3 | 9 | 12 => {
                            consume_cmif_payload(memory, request, false);
                            // Write "AudioTvOutput" to the recv buffer or C descriptor.
                            let descriptor_layout =
                                parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
                            let recv = descriptor_layout
                                .recv_buffers
                                .first()
                                .copied()
                                .or_else(|| descriptor_layout.recv_statics.first().copied())
                                .unwrap_or_default();
                            let device_name = b"AudioTvOutput\0";
                            if recv.address != 0 && recv.size >= device_name.len() as u64 {
                                write_guest_buffer(memory, recv.address, device_name);
                            }
                        }
                        // IAudioDevice: QueryAudioDeviceSystemEvent (cmd 4 on IAudioDevice
                        // which conflicts with GetAudioDeviceServiceWithRevisionInfo on the
                        // manager — but since we use cloned sessions with the same kind,
                        // we already handle it above as cmd=4)
                        // IAudioDevice: GetActiveChannelCount (cmd 5) -> u32
                        5 => {
                            consume_cmif_payload(memory, request, false);
                            payload.extend_from_slice(&2u32.to_le_bytes()); // stereo
                        }
                        // IAudioRenderer: RequestUpdate / RequestUpdateAuto (cmd 4/10)
                        // cmd 4 is ambiguous (also GetAudioDeviceService), handled above.
                        10 => {
                            consume_cmif_payload(memory, request, false);
                            // Return empty update response — game will populate output buffer
                        }
                        // IAudioRenderer: Start (cmd 5 on renderer)
                        // cmd 5 is ambiguous with GetActiveChannelCount above.
                        // IAudioRenderer: Stop (cmd 6)
                        // IAudioRenderer: QuerySystemEvent (cmd 7) -> event_handle
                        7 => {
                            consume_cmif_payload(memory, request, false);
                            // Create a signaled event handle for audio renderer notifications
                            if let Some(ev_handle) =
                                kernel.create_event_handle(current_thread_handle)
                            {
                                kernel.set_event_signaled_for_handle(
                                    current_thread_handle,
                                    ev_handle,
                                    true,
                                );
                                copy_handle = Some(ev_handle);
                            }
                        }
                        _ => {
                            consume_cmif_payload(memory, request, false);
                        }
                    }
                }

                // audout:u (IAudioOutManager)
                if session_kind == ServiceSessionKind::AudOut {
                    match command_id {
                        // ListAudioOuts (cmd 0) / ListAudioOutsAuto (cmd 2)
                        0 | 2 => {
                            consume_cmif_payload(memory, request, false);
                            let descriptor_layout =
                                parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
                            let recv = descriptor_layout
                                .recv_buffers
                                .first()
                                .copied()
                                .or_else(|| descriptor_layout.recv_statics.first().copied())
                                .unwrap_or_default();
                            let device_name = b"AudioTvOutput\0";
                            if recv.address != 0 && recv.size >= device_name.len() as u64 {
                                write_guest_buffer(memory, recv.address, device_name);
                            }
                            payload.extend_from_slice(&1u32.to_le_bytes()); // count=1
                        }
                        // OpenAudioOut (cmd 1) / OpenAudioOutAuto (cmd 3) -> IAudioOut + params
                        1 | 3 => {
                            consume_cmif_payload(memory, request, false);
                            payload.extend_from_slice(&48000u32.to_le_bytes()); // sample_rate
                            payload.extend_from_slice(&2u32.to_le_bytes()); // channel_count
                            payload.extend_from_slice(&2u32.to_le_bytes()); // PCM format (PCM16)
                            payload.extend_from_slice(&0u32.to_le_bytes()); // state (stopped)

                            let descriptor_layout =
                                parse_hipc_buffer_layout(memory, tls).unwrap_or_default();
                            let recv = descriptor_layout
                                .recv_buffers
                                .first()
                                .copied()
                                .or_else(|| descriptor_layout.recv_statics.first().copied())
                                .unwrap_or_default();
                            let device_name = b"AudioTvOutput\0";
                            if recv.address != 0 && recv.size >= device_name.len() as u64 {
                                write_guest_buffer(memory, recv.address, device_name);
                            }

                            // Create IAudioOut child session with distinct kind
                            if let Some(handle) = kernel.clone_service_session_as(
                                current_thread_handle,
                                session_handle,
                                Some(ServiceSessionKind::AudOutSession),
                            ) {
                                move_handle = Some(handle);
                            }
                        }
                        _ => {
                            consume_cmif_payload(memory, request, false);
                        }
                    }
                }

                // IAudioOut session commands
                if session_kind == ServiceSessionKind::AudOutSession {
                    let request_payload = read_cmif_payload_bytes(memory, request, false);
                    let descriptor_layout =
                        parse_hipc_buffer_layout(memory, tls).unwrap_or_default();

                    eprintln!("[audio] tid={current_thread_handle} IAudioOut cmd={command_id}");

                    match command_id {
                        // GetAudioOutState (cmd 0) -> u32
                        0 => {
                            payload.extend_from_slice(&1u32.to_le_bytes()); // state=Started
                        }
                        // Start (cmd 1)
                        1 => {}
                        // Stop (cmd 2)
                        2 => {}
                        // AppendAudioOutBuffer (cmd 3) / AppendAudioOutBufferAuto (cmd 7)
                        3 | 7 => {
                            // The raw payload contains the u64 buffer tag.
                            let tag = read_payload_u64(&request_payload, 0).unwrap_or(0);
                            kernel.audio_out_append_buffer(tag);
                            // Don't signal the buffer event yet — the event will be
                            // signaled by tick_audio_buffers when the playback delay
                            // elapses and the buffer moves to the released queue.
                        }
                        // RegisterBufferEvent (cmd 4) -> event handle
                        4 => {
                            if let Some(ev_handle) =
                                kernel.create_event_handle(current_thread_handle)
                            {
                                kernel.set_audio_out_buffer_event(ev_handle);
                                let _ = kernel.set_event_signaled_for_handle(
                                    current_thread_handle,
                                    ev_handle,
                                    true,
                                );
                                copy_handle = Some(ev_handle);
                            }
                        }
                        // GetReleasedAudioOutBuffers (cmd 5) / GetReleasedAudioOutBuffersAuto (cmd 8)
                        5 | 8 => {
                            let released = kernel.audio_out_take_released();
                            let count = released.len() as u32;
                            // Write released tags to the recv buffer.
                            let out_desc = descriptor_layout
                                .recv_buffers
                                .iter()
                                .copied()
                                .find(|d| descriptor_has_data(*d))
                                .or_else(|| {
                                    descriptor_layout
                                        .recv_statics
                                        .iter()
                                        .copied()
                                        .find(|d| descriptor_has_data(*d))
                                });
                            if let Some(desc) = out_desc {
                                let max_tags = (desc.size / 8) as usize;
                                let write_count = released.len().min(max_tags);
                                for (i, tag) in released.iter().take(write_count).enumerate() {
                                    let _ = memory
                                        .write_u64(desc.address.wrapping_add((i * 8) as u64), *tag);
                                }
                            }
                            payload.extend_from_slice(&count.to_le_bytes());
                        }
                        // ContainsAudioOutBuffer (cmd 6) -> bool
                        6 => {
                            payload.extend_from_slice(&0u32.to_le_bytes());
                        }
                        _ => {}
                    }
                }

                if session_kind == ServiceSessionKind::Nvdrv {
                    let request_payload = read_cmif_payload_bytes(memory, request, false);
                    let descriptor_layout =
                        parse_hipc_buffer_layout(memory, tls).unwrap_or_default();

                    match command_id {
                        // Open(path) -> (fd, nv_result)
                        0 => {
                            let path = descriptor_layout
                                .send_statics
                                .iter()
                                .copied()
                                .find(|descriptor| descriptor_has_data(*descriptor))
                                .or_else(|| {
                                    descriptor_layout
                                        .send_buffers
                                        .iter()
                                        .copied()
                                        .find(|descriptor| descriptor_has_data(*descriptor))
                                })
                                .and_then(|descriptor| {
                                    nvdrv_device_path_from_descriptor(memory, descriptor)
                                })
                                .filter(|path| !path.is_empty())
                                .unwrap_or_else(|| {
                                    nvdrv_device_path_from_payload(&request_payload)
                                });
                            if trace_service_traffic_enabled() {
                                eprintln!("[svc::SendSyncRequest:Nvdrv] open path=\"{}\"", path);
                            }
                            eprintln!("[nvdrv:open] path=\"{path}\"");
                            let fd = kernel
                                .nvdrv_open_device(current_thread_handle, session_handle, &path)
                                .unwrap_or(-1);
                            let nv_result = if fd >= 0 {
                                NVDRV_RESULT_SUCCESS
                            } else {
                                NVDRV_RESULT_INVALID_STATE
                            };

                            payload.extend_from_slice(&fd.to_le_bytes());
                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        // Ioctl/Ioctl2/Ioctl3 currently acknowledge the command shape and return
                        // a minimal nv_result while deeper ioctl semantics are still in progress.
                        1 | 11 | 12 => {
                            let fd = read_payload_u32(&request_payload, 0)
                                .map(|raw| raw as i32)
                                .unwrap_or(-1);
                            let ioctl = read_payload_u32(&request_payload, 4).unwrap_or(0);
                            let kind = match command_id {
                                1 => NvdrvIoctlKind::Ioctl,
                                11 => NvdrvIoctlKind::Ioctl2,
                                12 => NvdrvIoctlKind::Ioctl3,
                                _ => unreachable!("invalid nvdrv ioctl command id: {command_id}"),
                            };

                            let non_empty_send_buffers: Vec<HipcBufferDescriptor> =
                                descriptor_layout
                                    .send_buffers
                                    .iter()
                                    .copied()
                                    .filter(|descriptor| descriptor_has_data(*descriptor))
                                    .collect();
                            let non_empty_recv_buffers: Vec<HipcBufferDescriptor> =
                                descriptor_layout
                                    .recv_buffers
                                    .iter()
                                    .copied()
                                    .filter(|descriptor| descriptor_has_data(*descriptor))
                                    .collect();
                            let has_static_args = descriptor_layout
                                .send_statics
                                .iter()
                                .copied()
                                .any(descriptor_has_data);
                            let input_arg_descriptor = descriptor_layout
                                .send_statics
                                .iter()
                                .copied()
                                .find(|descriptor| descriptor_has_data(*descriptor))
                                .or_else(|| non_empty_send_buffers.first().copied());
                            // In pointer mode, the output goes to a C descriptor which
                            // may be larger than the X descriptor used for input.
                            // Use the bigger of the two as the args working buffer.
                            let output_arg_c_descriptor = descriptor_layout
                                .recv_statics
                                .iter()
                                .copied()
                                .find(|d| descriptor_has_data(*d));
                            let effective_arg_descriptor =
                                match (input_arg_descriptor, output_arg_c_descriptor) {
                                    (Some(x), Some(c)) if c.size > x.size => Some(c),
                                    (Some(x), _) => Some(x),
                                    (None, c) => c,
                                };
                            let mut args = effective_arg_descriptor
                                .and_then(|descriptor| read_descriptor_bytes(memory, descriptor))
                                .unwrap_or_else(|| {
                                    request_payload.get(8..).unwrap_or(&[]).to_vec()
                                });
                            // If the C descriptor is larger, copy input data from X into it.
                            if let (Some(x_desc), Some(c_desc)) =
                                (input_arg_descriptor, output_arg_c_descriptor)
                            {
                                if c_desc.size > x_desc.size
                                    && effective_arg_descriptor == Some(c_desc)
                                {
                                    // Re-read the X descriptor input and overlay onto args.
                                    if let Some(input_data) = read_descriptor_bytes(memory, x_desc)
                                    {
                                        let copy_len = input_data.len().min(args.len());
                                        args[..copy_len].copy_from_slice(&input_data[..copy_len]);
                                    }
                                }
                            }

                            let inline_descriptor = match kind {
                                NvdrvIoctlKind::Ioctl => None,
                                NvdrvIoctlKind::Ioctl2 => {
                                    // Ioctl2: extra inline *input* buffer.
                                    // In pointer mode (HipcAutoSelect), B descriptors are
                                    // empty placeholders; the real input buffers are X
                                    // descriptors (send_statics).  The first X descriptor
                                    // holds the ioctl args; the second holds the inline data.
                                    let non_empty_send_statics: Vec<HipcBufferDescriptor> =
                                        descriptor_layout
                                            .send_statics
                                            .iter()
                                            .copied()
                                            .filter(|d| descriptor_has_data(*d))
                                            .collect();
                                    if has_static_args {
                                        non_empty_send_statics
                                            .get(1)
                                            .copied()
                                            .or_else(|| non_empty_send_buffers.first().copied())
                                    } else {
                                        non_empty_send_buffers.get(1).copied()
                                    }
                                }
                                NvdrvIoctlKind::Ioctl3 => {
                                    // In pointer mode (HipcAutoSelect) B descriptors are
                                    // empty placeholders; the real buffers are C descriptors.
                                    // For Ioctl3 the second recv descriptor is the inline
                                    // output buffer.
                                    let non_empty_recv_statics: Vec<HipcBufferDescriptor> =
                                        descriptor_layout
                                            .recv_statics
                                            .iter()
                                            .copied()
                                            .filter(|d| descriptor_has_data(*d))
                                            .collect();
                                    non_empty_recv_buffers
                                        .get(1)
                                        .copied()
                                        .or_else(|| non_empty_recv_statics.get(1).copied())
                                }
                            };
                            let mut inline_buffer = inline_descriptor
                                .and_then(|descriptor| read_descriptor_bytes(memory, descriptor))
                                .unwrap_or_default();

                            let mut nv_result = kernel
                                .nvdrv_ioctl_result(
                                    current_thread_handle,
                                    session_handle,
                                    kind,
                                    fd,
                                    ioctl,
                                    &mut args,
                                    &mut inline_buffer,
                                )
                                .unwrap_or(NVDRV_RESULT_INVALID_STATE);

                            eprintln!(
                                "[nvdrv:ioctl] fd={fd} ioctl=0x{ioctl:08x} kind={kind:?} nv_result={nv_result} args_len={} inline_len={}",
                                args.len(),
                                inline_buffer.len(),
                            );
                            let direction = (ioctl >> 30) & 0x3;
                            if (direction & 0b10) != 0 {
                                // In pointer mode, B descriptors are empty - use C descriptors.
                                let first_recv_static = descriptor_layout
                                    .recv_statics
                                    .iter()
                                    .copied()
                                    .find(|d| descriptor_has_data(*d));
                                let args_output_descriptor = non_empty_recv_buffers
                                    .first()
                                    .copied()
                                    .or(first_recv_static)
                                    .or(input_arg_descriptor);
                                if let Some(descriptor) = args_output_descriptor {
                                    let max = usize::try_from(descriptor.size)
                                        .ok()
                                        .unwrap_or(args.len())
                                        .min(args.len());
                                    if !write_guest_buffer(memory, descriptor.address, &args[..max])
                                    {
                                        nv_result = NVDRV_RESULT_INVALID_STATE;
                                    }
                                }

                                if kind == NvdrvIoctlKind::Ioctl3
                                    && let Some(descriptor) = inline_descriptor
                                {
                                    let max = usize::try_from(descriptor.size)
                                        .ok()
                                        .unwrap_or(inline_buffer.len())
                                        .min(inline_buffer.len());
                                    if !write_guest_buffer(
                                        memory,
                                        descriptor.address,
                                        &inline_buffer[..max],
                                    ) {
                                        nv_result = NVDRV_RESULT_INVALID_STATE;
                                    }
                                }
                            }

                            // After a SubmitGpfifo ioctl, process GPU commands
                            // to execute semaphore writes that the game expects.
                            let ioctl_group = (ioctl >> 8) & 0xFF;
                            let ioctl_cmd = ioctl & 0xFF;
                            eprintln!(
                                "[nvdrv:gpfifo-check] ioctl=0x{:08x} group=0x{:x} cmd=0x{:x} nv_result={} inline_len={}",
                                ioctl,
                                ioctl_group,
                                ioctl_cmd,
                                nv_result,
                                inline_buffer.len()
                            );
                            if ioctl_group == 0x48
                                && (ioctl_cmd == 0x08 || ioctl_cmd == 0x1b)
                                && nv_result == NVDRV_RESULT_SUCCESS
                            {
                                let mem = memory.clone();
                                let mem2 = memory.clone();
                                kernel.nvdrv_process_gpfifo(
                                    &inline_buffer,
                                    &move |addr, len| {
                                        let mut buf = vec![0u8; len];
                                        for i in (0..len).step_by(4) {
                                            let val = mem.read_u32(addr + i as u64)?;
                                            let bytes = val.to_le_bytes();
                                            let end = (i + 4).min(len);
                                            buf[i..end].copy_from_slice(&bytes[..end - i]);
                                        }
                                        Some(buf)
                                    },
                                    &move |addr, data| {
                                        // Write in 4-byte chunks.
                                        for i in (0..data.len()).step_by(4) {
                                            let end = (i + 4).min(data.len());
                                            let mut bytes = [0u8; 4];
                                            bytes[..end - i].copy_from_slice(&data[i..end]);
                                            let val = u32::from_le_bytes(bytes);
                                            if !mem2.write_u32(addr + i as u64, val) {
                                                return false;
                                            }
                                        }
                                        true
                                    },
                                );
                            }

                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        // Close(fd) -> nv_result
                        2 => {
                            let fd = read_payload_u32(&request_payload, 0)
                                .map(|raw| raw as i32)
                                .unwrap_or(-1);
                            let nv_result = if kernel.nvdrv_close_device(
                                current_thread_handle,
                                session_handle,
                                fd,
                            ) {
                                NVDRV_RESULT_SUCCESS
                            } else {
                                NVDRV_RESULT_INVALID_STATE
                            };
                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        // Initialize: expose a tiny capability payload so callers can distinguish
                        // CPU-only fallback from host-GPU availability.
                        3 => {
                            if let Some(init_payload) = kernel
                                .nvdrv_initialize_payload(current_thread_handle, session_handle)
                            {
                                payload.extend_from_slice(&init_payload);
                            } else {
                                payload.extend_from_slice(&0u32.to_le_bytes());
                                payload.extend_from_slice(&0u32.to_le_bytes());
                            }
                        }
                        // QueryEvent: hand out a stable event handle per open fd/event pair.
                        4 => {
                            let fd = read_payload_u32(&request_payload, 0)
                                .map(|raw| raw as i32)
                                .unwrap_or(-1);
                            let event_id = read_payload_u32(&request_payload, 4).unwrap_or(0);
                            eprintln!(
                                "[nvdrv:QueryEvent] fd={} event_id={} session_handle={}",
                                fd, event_id, session_handle
                            );
                            if let Some(event_command) = kernel.nvdrv_query_event_command(
                                current_thread_handle,
                                session_handle,
                                fd,
                                event_id,
                            ) {
                                eprintln!("[nvdrv:QueryEvent] event_command=0x{:x}", event_command);
                                copy_handle = kernel.service_session_get_or_create_event_handle(
                                    current_thread_handle,
                                    session_handle,
                                    fd as u32,
                                    event_command,
                                );
                                eprintln!("[nvdrv:QueryEvent] copy_handle={:?}", copy_handle);
                                if let Some(event_handle) = copy_handle {
                                    // Only auto-signal events that represent completed
                                    // GPU work (syncpoint events).  Error notifier /
                                    // SM exception events must stay unsignaled — the
                                    // game polls them to check for GPU errors.
                                    if kernel.nvdrv_should_auto_signal_event(fd, event_id) {
                                        let _ = kernel.set_event_signaled_for_handle(
                                            current_thread_handle,
                                            event_handle,
                                            true,
                                        );
                                    }
                                    payload.extend_from_slice(&NVDRV_RESULT_SUCCESS.to_le_bytes());
                                } else {
                                    payload.extend_from_slice(
                                        &NVDRV_RESULT_INVALID_STATE.to_le_bytes(),
                                    );
                                }
                            } else {
                                payload
                                    .extend_from_slice(&NVDRV_RESULT_INVALID_STATE.to_le_bytes());
                            }
                        }
                        5 => {
                            let nv_result = kernel
                                .nvdrv_map_shared_memory_result(
                                    current_thread_handle,
                                    session_handle,
                                    &request_payload,
                                )
                                .unwrap_or(NVDRV_RESULT_INVALID_STATE);
                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        6 => {
                            if let Some(status_payload) = kernel.nvdrv_get_status_payload(
                                current_thread_handle,
                                session_handle,
                                &request_payload,
                            ) {
                                payload.extend_from_slice(&status_payload);
                            } else {
                                payload
                                    .extend_from_slice(&NVDRV_RESULT_INVALID_STATE.to_le_bytes());
                            }
                        }
                        8 => {
                            let nv_result = kernel
                                .nvdrv_set_client_pid_result(
                                    current_thread_handle,
                                    session_handle,
                                    &request_payload,
                                )
                                .unwrap_or(NVDRV_RESULT_INVALID_STATE);
                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        13 => {
                            let nv_result = kernel
                                .nvdrv_finish_initialize_result(
                                    current_thread_handle,
                                    session_handle,
                                    &request_payload,
                                )
                                .unwrap_or(NVDRV_RESULT_INVALID_STATE);
                            payload.extend_from_slice(&nv_result.to_le_bytes());
                        }
                        _ => {
                            unimplemented!("nvdrv command {} is not implemented", command_id);
                        }
                    }
                }

                if session_kind == ServiceSessionKind::Time {
                    match command_id {
                        // ITimeService::{Get*Clock,GetTimeZoneService}() -> session
                        0 | 1 | 2 | 3 | 4 => {
                            consume_cmif_payload(memory, request, false);
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // ITimeService::GetSharedMemoryNativeHandle() -> copy handle
                        20 => {
                            copy_handle =
                                create_time_shared_memory_handle(kernel, current_thread_handle);
                            if copy_handle.is_none() {
                                cmif_result = RESULT_OUT_OF_MEMORY as u32;
                            }
                        }
                        _ => consume_cmif_payload(memory, request, false),
                    }
                }

                if session_kind == ServiceSessionKind::Hid {
                    let has_pid = parse_hipc_special_header(memory, tls)
                        .and_then(|special| special.pid)
                        .is_some();

                    match command_id {
                        // IHidServer::CreateAppletResource(applet_resource_user_id)
                        0 if has_pid => {
                            consume_cmif_payload(memory, request, false);
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        // IAppletResource::GetSharedMemoryHandle()
                        0 => {
                            copy_handle =
                                create_hid_shared_memory_handle(kernel, current_thread_handle);
                            if let Some(h) = copy_handle {
                                kernel.set_hid_shared_memory_handle(h);
                            } else {
                                cmif_result = RESULT_OUT_OF_MEMORY as u32;
                            }
                        }
                        // IHidServer::SetSupportedNpadStyleSet(u32 style_set, pid)
                        100 => {
                            let p = read_cmif_payload_bytes(memory, request, false);
                            let style = read_payload_u32(&p, 0).unwrap_or(0);
                            eprintln!("[hid] SetSupportedNpadStyleSet style=0x{:08x}", style);
                            // no response data
                        }
                        // IHidServer::SetSupportedNpadIdType(pid, buffer<NpadIdType[]>)
                        109 => {
                            // Read the buffer descriptor from the HIPC message
                            // The NpadIdType list is sent as a type-5 (C descriptor) or type-9 (X) buffer.
                            // For now just log the raw payload.
                            let p = read_cmif_payload_bytes(memory, request, false);
                            eprintln!(
                                "[hid] SetSupportedNpadIdType payload={:02x?}",
                                &p[..p.len().min(32)]
                            );
                            // Also try to read from type-5 buffer pointer in HIPC header
                            // Header word[0] is type-A count and type-X count
                            let w0 = memory.read_u32(tls).unwrap_or(0);
                            let num_x = (w0 >> 16) & 0xf;
                            let num_a = (w0 >> 20) & 0xf;
                            eprintln!(
                                "[hid]   HIPC w0=0x{:08x} num_x={} num_a={}",
                                w0, num_x, num_a
                            );
                            // If there's a type-A (send) buffer, read its descriptor
                            if num_a > 0 {
                                // type-A descriptors start after the special header (if any) and handle descriptors
                                // For simplicity, scan for it:
                                let special_size = if has_pid { 12 } else { 0 }; // pid(8) + header(4)
                                let desc_offset = tls + 16 + special_size as u64;
                                let buf_addr_lo = memory.read_u32(desc_offset).unwrap_or(0);
                                let buf_addr_hi = memory.read_u32(desc_offset + 4).unwrap_or(0);
                                let buf_info = memory.read_u32(desc_offset + 8).unwrap_or(0);
                                let buf_size = buf_info & 0xFFFF;
                                let addr = (buf_addr_lo as u64)
                                    | ((buf_addr_hi as u64) << 32)
                                    | (((buf_info as u64 >> 28) & 0xF) << 36);
                                eprintln!(
                                    "[hid]   type-A buffer addr=0x{:016x} size={}",
                                    addr, buf_size
                                );
                                if buf_size > 0 && buf_size <= 80 {
                                    let mut ids = Vec::new();
                                    for i in 0..(buf_size / 4) {
                                        if let Some(id) = memory.read_u32(addr + (i as u64) * 4) {
                                            ids.push(id);
                                        }
                                    }
                                    eprintln!("[hid]   NpadIdTypes: {:?}", ids);
                                }
                            }
                        }
                        // IHidServer::CreateActiveVibrationDeviceList() -> IActiveVibrationDeviceList
                        203 => {
                            consume_cmif_payload(memory, request, false);
                            if let Some(handle) =
                                kernel.clone_service_session(current_thread_handle, session_handle)
                            {
                                move_handle = Some(handle);
                            } else {
                                cmif_result = RESULT_INVALID_HANDLE as u32;
                            }
                        }
                        _ => consume_cmif_payload(memory, request, false),
                    }
                }
            }

            let ok = write_cmif_response(
                memory,
                tls,
                cmif_result,
                &payload,
                copy_handle,
                move_handle,
                is_domain_message,
                &domain_objects,
                request_context,
            );

            if let Some((command_id, object_id, object_kind)) = trace_target {
                trace_domain_response(
                    session_kind,
                    memory,
                    tls,
                    request,
                    command_id,
                    object_id,
                    object_kind,
                    cmif_result,
                    &payload,
                    copy_handle,
                    move_handle,
                    &domain_objects,
                    ok,
                );
            }

            if let Some(command_id) = trace_non_domain_command_id {
                trace_non_domain_response(
                    session_kind,
                    memory,
                    tls,
                    request,
                    command_id,
                    cmif_result,
                    &payload,
                    copy_handle,
                    move_handle,
                    ok,
                );
            }

            ok
        }
        _ => false,
    };

    if ok {
        RESULT_SUCCESS
    } else {
        RESULT_INVALID_MEMORY_STATE
    }
}

#[svc(0x06)]
fn svc_query_memory(
    #[x(0)] info_ptr: u64,
    #[x(1)] page_info_ptr: u64,
    #[x(2)] query_address: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;
    let memory = context.memory;
    let info = query_memory_info(memory, query_address);

    if write_query_memory_info(memory, info_ptr, page_info_ptr, query_address) {
        state.write_x(0, RESULT_SUCCESS);
        state.write_x(1, 0);
        eprintln!(
            "[svc::QueryMemoryInfo] query=0x{query_address:016x} region=0x{:016x}..0x{:016x} state=0x{:x} perm=0x{:x}",
            info.address,
            info.address.saturating_add(info.size),
            info.state,
            info.permission,
        );
    } else {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.write_x(1, 0);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x04)]
fn svc_map_memory(
    #[x(0)] dst_address: u64,
    #[x(1)] src_address: u64,
    #[x(2)] size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;
    let memory = context.memory;

    if size == 0
        || (dst_address & (PAGE_SIZE - 1)) != 0
        || (src_address & (PAGE_SIZE - 1)) != 0
        || (size & (PAGE_SIZE - 1)) != 0
    {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let source_info = memory.query_memory(src_address);
    let destination_info = memory.query_memory(dst_address);
    if source_info.state == 0 || destination_info.state != 0 {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if !map_memory_bytes(memory, dst_address, src_address, size) {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let _ = memory.map_high_region(dst_address, size, source_info.state, source_info.permission);

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x05)]
fn svc_unmap_memory(
    #[x(0)] dst_address: u64,
    #[x(1)] src_address: u64,
    #[x(2)] size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;
    let memory = context.memory;

    if size == 0
        || (dst_address & (PAGE_SIZE - 1)) != 0
        || (src_address & (PAGE_SIZE - 1)) != 0
        || (size & (PAGE_SIZE - 1)) != 0
    {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Copy data back from dst to src (reverse of MapMemory).
    if !map_memory_bytes(memory, src_address, dst_address, size) {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Note: We don't actually unmap the dest region since our memory model
    // allows overlapping usage. The region just becomes stale.

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x01)]
fn svc_set_heap_size(
    #[x(0)] _out_heap_address_ptr: u64,
    #[x(1)] raw_size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if trace_svc_enabled() {
        eprintln!("[svc::SetHeapSize] requested_size=0x{raw_size:016x}");
    }

    // Some userspace wrappers pass a 32-bit signed immediate widened into x1.
    // Canonicalize those values back to the intended 32-bit heap size.
    let size = if (raw_size >> 32) == u64::from(u32::MAX) {
        u64::from(raw_size as u32)
    } else {
        raw_size
    };

    // Ryujinx: (size & 0xfffffffc001fffff) != 0 → InvalidSize
    // Size must be 2 MiB aligned and fit in ~34 bits.
    if (size & 0xffff_fffc_001f_ffff) != 0 {
        state.write_x(0, RESULT_INVALID_SIZE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let Some((heap_region_start, heap_region_end, heap_capacity, _current_heap_addr)) =
        kernel.process_heap_region_for_handle(context.current_thread_handle, SELF_PROCESS_HANDLE)
    else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    // Ryujinx: size > HeapRegionEnd - HeapRegionStart || size > _heapCapacity
    let heap_region_size = heap_region_end.saturating_sub(heap_region_start);
    if size > heap_region_size || size > heap_capacity {
        state.write_x(0, RESULT_OUT_OF_MEMORY);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Map the heap region as RW in the high sparse pages.
    // state=0x5 (Heap), permission=0x3 (ReadWrite).
    if size != 0
        && !context
            .memory
            .map_high_region(heap_region_start, size, 0x5, 0x3)
    {
        state.write_x(0, RESULT_OUT_OF_MEMORY);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Update current_heap_addr in the kernel.
    kernel.set_process_current_heap_addr(
        context.current_thread_handle,
        SELF_PROCESS_HANDLE,
        heap_region_start + size,
    );

    if trace_svc_enabled() {
        eprintln!(
            "[svc::SetHeapSize] heap_region_start=0x{heap_region_start:016x} size=0x{size:016x} heap_capacity=0x{heap_capacity:016x} heap_region_end=0x{heap_region_end:016x}"
        );
    }

    // Ryujinx: address = HeapRegionStart (always returns the start of the heap region).
    // The SVC returns the heap address in x1 (not written to memory).
    // The libnx wrapper saves x0 (output pointer) on stack before SVC,
    // then after SVC writes x1 to *saved_pointer.
    state.write_x(0, RESULT_SUCCESS);
    state.write_x(1, heap_region_start);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x02)]
fn svc_set_memory_permission(
    #[x(0)] address: u64,
    #[x(1)] size: u64,
    #[x(2)] _permission: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;

    if (address & (PAGE_SIZE - 1)) != 0 || size == 0 || (size & (PAGE_SIZE - 1)) != 0 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
    } else {
        state.write_x(0, RESULT_SUCCESS);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x03)]
fn svc_set_memory_attribute(
    #[x(0)] address: u64,
    #[x(1)] size: u64,
    #[x(2)] _mask: u32,
    #[x(3)] _value: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;

    if (address & (PAGE_SIZE - 1)) != 0 || size == 0 || (size & (PAGE_SIZE - 1)) != 0 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
    } else {
        state.write_x(0, RESULT_SUCCESS);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x07)]
fn svc_exit_process(context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    state.clear_exit();
    if kernel.exit_process(context.current_thread_handle, state) {
        SyscallDisposition::ThreadExit
    } else {
        SyscallDisposition::Continue
    }
}

#[svc(0x08)]
fn svc_create_thread(
    #[x(1)] entrypoint: u64,
    #[x(2)] argument: u64,
    #[x(3)] stack_top: u64,
    #[x(4)] priority: i32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    let handle = kernel.create_thread(
        context.current_thread_handle,
        entrypoint,
        argument,
        stack_top,
        state,
        priority,
    );
    let thread_ptr = stack_top & !0xfff;
    let _ = context
        .memory
        .write_u64(thread_ptr.wrapping_add(0x1f8), thread_ptr);
    let _ = context
        .memory
        .write_u32(thread_ptr.wrapping_add(0x1b0), handle);

    state.write_x(0, RESULT_SUCCESS);
    state.write_x(1, u64::from(handle));
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x09)]
fn svc_start_thread(#[x(0)] handle: u32, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if kernel.start_thread(context.current_thread_handle, handle) {
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x0a)]
fn svc_exit_thread(context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    state.clear_exit();
    kernel.exit_thread(context.current_thread_handle, state);
    if context.current_thread_handle == MAIN_THREAD_HANDLE {
        SyscallDisposition::ThreadExit
    } else {
        SyscallDisposition::Continue
    }
}

#[svc(0x0b)]
fn svc_sleep_thread(#[x(0)] timeout_ns: i64, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        kernel.sleep_thread(context.current_thread_handle, state, timeout_ns);
    } else if timeout_ns > 0 {
        // Main thread can't truly block.  Defer the sleep to step() where
        // the kernel lock will be released first, giving workers real
        // uncontested time to execute.
        kernel.signal_binder_vsync_events();
        let sleep_us = ((timeout_ns / 1000).max(100).min(2000)) as u64;
        kernel.set_main_thread_pending_sleep(std::time::Duration::from_micros(sleep_us));
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x0c)]
fn svc_get_thread_priority(
    #[x(0)] out_ptr: u64,
    #[x(1)] handle: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if let Some(priority) = kernel.thread_priority_for_handle(context.current_thread_handle, handle)
    {
        if memory.write_u32(out_ptr, priority as u32) {
            state.write_x(0, RESULT_SUCCESS);
            state.write_x(1, priority as u64);
        } else {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.write_x(1, 0);
        }
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.write_x(1, 0);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x0d)]
fn svc_set_thread_priority(
    #[x(0)] handle: u32,
    #[x(1)] priority: i32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if kernel.set_thread_priority(context.current_thread_handle, handle, priority) {
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x0e)]
fn svc_get_thread_core_mask(
    #[x(0)] out_core_id_ptr: u64,
    #[x(1)] out_affinity_mask_ptr: u64,
    #[x(2)] handle: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    let Some((core_id, affinity_mask)) =
        kernel.thread_core_mask_for_handle(context.current_thread_handle, handle)
    else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    if memory.write_u32(out_core_id_ptr, core_id as u32)
        && memory.write_u64(out_affinity_mask_ptr, affinity_mask)
    {
        state.write_x(0, RESULT_SUCCESS);
        state.write_x(1, affinity_mask);
    } else {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.write_x(1, 0);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x0f)]
fn svc_set_thread_core_mask(
    #[x(0)] handle: u32,
    #[x(1)] ideal_core_id: i32,
    #[x(2)] affinity_mask: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    match kernel.set_thread_core_mask(
        context.current_thread_handle,
        handle,
        ideal_core_id,
        affinity_mask,
    ) {
        Ok(()) => state.write_x(0, RESULT_SUCCESS),
        Err(ThreadCoreMaskError::InvalidHandle) => state.write_x(0, RESULT_INVALID_HANDLE),
        Err(ThreadCoreMaskError::InvalidCombination) => {
            state.write_x(0, RESULT_INVALID_COMBINATION)
        }
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x10)]
fn svc_get_current_processor_number(context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &*context.kernel;
    let state = &mut *context.state;

    let core_id = kernel
        .thread_core_mask_for_handle(context.current_thread_handle, SELF_THREAD_HANDLE)
        .map(|(core_id, _)| core_id)
        .unwrap_or(0);

    state.write_x(0, core_id as u64);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x13)]
fn svc_map_shared_memory(
    #[x(0)] handle: u32,
    #[x(1)] address: u64,
    #[x(2)] size: u64,
    #[x(3)] permission: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if size == 0
        || (address & (PAGE_SIZE - 1)) != 0
        || (size & (PAGE_SIZE - 1)) != 0
        || permission == 0
        || (permission & !0x3) != 0
    {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let Some((shared_size, shared_permission)) =
        kernel.shared_memory_for_handle(context.current_thread_handle, handle)
    else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    if size > shared_size || (permission & shared_permission) != permission {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if memory.query_memory(address).state != 0 {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if !memory.map_high_region(address, size, MEMORY_STATE_SHARED, permission) {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Track HID shared memory mapping for Npad state updates.
    kernel.check_hid_shared_memory_mapping(handle, address);

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x14)]
fn svc_unmap_shared_memory(
    #[x(0)] handle: u32,
    #[x(1)] address: u64,
    #[x(2)] size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if size == 0 || (address & (PAGE_SIZE - 1)) != 0 || (size & (PAGE_SIZE - 1)) != 0 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let Some((shared_size, _)) =
        kernel.shared_memory_for_handle(context.current_thread_handle, handle)
    else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    if size > shared_size {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x15)]
fn svc_create_transfer_memory(
    #[x(1)] address: u64,
    #[x(2)] size: u64,
    #[x(3)] permission: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if size == 0 || (address & (PAGE_SIZE - 1)) != 0 || (size & (PAGE_SIZE - 1)) != 0 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if (permission & !0x3) != 0 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let Some(end_address) = address.checked_add(size) else {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    let mut cursor = address;
    while cursor < end_address {
        let info = memory.query_memory(cursor);
        if info.state == 0 {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.clear_exit();
            return SyscallDisposition::Continue;
        }

        let region_end = info.address.saturating_add(info.size).min(end_address);
        if region_end <= cursor {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.clear_exit();
            return SyscallDisposition::Continue;
        }
        cursor = region_end;
    }

    if let Some(handle) = kernel.create_transfer_memory_handle(
        context.current_thread_handle,
        address,
        size,
        permission,
    ) {
        state.write_x(0, RESULT_SUCCESS);
        state.write_x(1, u64::from(handle));
    } else {
        state.write_x(0, RESULT_OUT_OF_MEMORY);
        state.write_x(1, 0);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x16)]
fn svc_close_handle(#[x(0)] handle: u32, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if kernel.close_handle(context.current_thread_handle, handle) {
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x18)]
fn svc_wait_synchronization(
    #[x(0)] out_index_ptr: u64,
    #[x(1)] handles_ptr: u64,
    #[x(2)] num_handles: i32,
    #[x(3)] timeout_ns: i64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if num_handles <= 0 || num_handles > 0x40 {
        state.write_x(0, RESULT_INVALID_COMBINATION);
        state.write_x(1, u64::MAX);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let count = num_handles as usize;
    let mut handles = Vec::with_capacity(count);
    for index in 0..count {
        let address = handles_ptr.wrapping_add((index as u64) * 4);
        let Some(handle) = memory.read_u32(address) else {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.write_x(1, u64::MAX);
            state.clear_exit();
            return SyscallDisposition::Continue;
        };
        handles.push(handle);
    }

    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        let statuses = handles
            .iter()
            .map(|handle| {
                let signaled = kernel
                    .synchronization_handle_is_signaled(context.current_thread_handle, *handle)
                    .map(|flag| if flag { "1" } else { "0" })
                    .unwrap_or("?");
                format!("{handle}:{signaled}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[svc::WaitSynchronization] tid={} timeout_ns={} handles=[{}]",
            context.current_thread_handle, timeout_ns, statuses
        );
    } else if timeout_ns != 0 {
        let statuses = handles
            .iter()
            .map(|handle| {
                let signaled = kernel
                    .synchronization_handle_is_signaled(context.current_thread_handle, *handle)
                    .map(|flag| if flag { "1" } else { "0" })
                    .unwrap_or("?");
                format!("{handle}:{signaled}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[svc::WaitSynchronization] tid=main timeout_ns={} handles=[{}]",
            timeout_ns, statuses
        );
    }

    for (index, handle) in handles.iter().copied().enumerate() {
        let Some(is_signaled) =
            kernel.synchronization_handle_is_signaled(context.current_thread_handle, handle)
        else {
            state.write_x(0, RESULT_INVALID_HANDLE);
            state.write_x(1, u64::MAX);
            state.clear_exit();
            return SyscallDisposition::Continue;
        };

        if is_signaled {
            let selected = index as u32;
            if !memory.write_u32(out_index_ptr, selected) {
                state.write_x(0, RESULT_INVALID_MEMORY_STATE);
                state.write_x(1, u64::MAX);
            } else {
                // Auto-clear the event so subsequent waits don't see it
                // as signaled until it is re-signaled.
                kernel.clear_event_signal_for_handle(context.current_thread_handle, handle);
                state.write_x(0, RESULT_SUCCESS);
                state.write_x(1, u64::from(selected));
                if context.current_thread_handle == MAIN_THREAD_HANDLE {
                    flush_pending_binder_frame(kernel, memory);
                }
            }
            state.clear_exit();
            return SyscallDisposition::Continue;
        }
    }

    if timeout_ns == 0 {
        let _ = memory.write_u32(out_index_ptr, u32::MAX);
        state.write_x(0, RESULT_TIMED_OUT);
        state.write_x(1, u64::MAX);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        let _ = memory.write_u32(out_index_ptr, u32::MAX);
        if timeout_ns > 0 {
            state.write_x(0, RESULT_TIMED_OUT);
            state.write_x(1, u64::MAX);
        } else {
            state.write_x(0, RESULT_SUCCESS);
            state.write_x(1, 0);
        }

        kernel.wait_for_synchronization_handles(
            context.current_thread_handle,
            state,
            &handles,
            out_index_ptr,
            timeout_ns,
        );
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Main thread: signal VSync events first, then check handles normally.
    // If any handle is already signaled, return immediately with success.
    kernel.signal_binder_vsync_events();
    for (index, handle) in handles.iter().copied().enumerate() {
        let Some(is_signaled) =
            kernel.synchronization_handle_is_signaled(context.current_thread_handle, handle)
        else {
            continue;
        };

        if is_signaled {
            let selected = index as u32;
            if !memory.write_u32(out_index_ptr, selected) {
                state.write_x(0, RESULT_INVALID_MEMORY_STATE);
                state.write_x(1, u64::MAX);
            } else {
                kernel.clear_event_signal_for_handle(context.current_thread_handle, handle);
                state.write_x(0, RESULT_SUCCESS);
                state.write_x(1, u64::from(selected));
                flush_pending_binder_frame(kernel, memory);
            }
            state.clear_exit();
            return SyscallDisposition::Continue;
        }
    }

    // No handle is signaled; for the main thread with infinite timeout,
    // auto-signal the first handle and return success.  Since we don't
    // have real GPU processing, GPU-related events would never be signaled
    // otherwise, and the main thread can't truly block.
    if timeout_ns < 0 && !handles.is_empty() {
        let handle = handles[0];
        let _ = kernel.set_event_signaled_for_handle(context.current_thread_handle, handle, true);
        kernel.clear_event_signal_for_handle(context.current_thread_handle, handle);
        let _ = memory.write_u32(out_index_ptr, 0);
        state.write_x(0, RESULT_SUCCESS);
        state.write_x(1, 0);
        flush_pending_binder_frame(kernel, memory);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Non-infinite timeout: return TIMED_OUT so game can continue spinning.
    let _ = memory.write_u32(out_index_ptr, u32::MAX);
    state.write_x(0, RESULT_TIMED_OUT);
    state.write_x(1, u64::MAX);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x1a)]
fn svc_arbitrate_lock(
    #[x(0)] owner_handle: u32,
    #[x(1)] mutex_address: u64,
    #[x(2)] requester_handle: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    // Track main thread lock spins for yield heuristic.
    if context.current_thread_handle == MAIN_THREAD_HANDLE {
        kernel.record_main_thread_lock_spin();
    }

    if (mutex_address & 3) != 0 {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let Some(mutex_value) = context.memory.read_u32(mutex_address) else {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    };

    // Case 1: Lock is free (released between guest CAS fail and SVC entry).
    // Acquire the lock on behalf of the requester.
    if mutex_value == 0 {
        context.memory.write_u32(mutex_address, requester_handle);
        state.write_x(0, RESULT_SUCCESS);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Case 2/3: Lock is held (with or without wait mask).  Ensure the
    // wait-mask bit is set so the owner knows there are waiters, then
    // block until the owner releases the lock to us.
    let current_owner = mutex_value & !HANDLE_WAIT_MASK;

    // Self-lock detection: if the requesting thread already owns the
    // mutex, return immediately to avoid a deadlock.  This can happen
    // when the HANDLE_WAIT_MASK was set by another thread between the
    // owner's CAS and its re-entry check, making the guest think the
    // lock is held by "someone else".
    if current_owner == requester_handle && requester_handle != 0 {
        state.write_x(0, RESULT_SUCCESS);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    if kernel
        .thread_uid_for_handle(context.current_thread_handle, current_owner)
        .is_none()
        && current_owner != 0
    {
        // Owner handle doesn't correspond to a valid thread – but we
        // still need to handle this gracefully.  Return success and let
        // the guest retry.
        state.write_x(0, RESULT_SUCCESS);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    // Set the HANDLE_WAIT_MASK bit so the owner knows there are waiters.
    if mutex_value & HANDLE_WAIT_MASK == 0 {
        context
            .memory
            .write_u32(mutex_address, mutex_value | HANDLE_WAIT_MASK);
    }

    state.write_x(0, RESULT_SUCCESS);
    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        kernel.wait_for_address_with_key(
            context.current_thread_handle,
            state,
            mutex_address,
            -1,
            mutex_address,
            requester_handle,
        );
    } else {
        // Main thread can't truly block; tell the emulator to yield
        // so worker threads (especially the lock owner) can make progress.
        kernel.set_main_thread_yield_mutex(mutex_address);
    }

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x1b)]
fn svc_arbitrate_unlock(#[x(0)] mutex_address: u64, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    eprintln!(
        "[arb_unlock] handle={} mutex=0x{mutex_address:016x}",
        context.current_thread_handle
    );

    // If the main thread is unlocking, clear its yield flag.
    if context.current_thread_handle == MAIN_THREAD_HANDLE {
        kernel.clear_main_thread_yield();
    }

    if (mutex_address & 3) != 0 {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let result = release_mutex_to_next_waiter(
        kernel,
        context.current_thread_handle,
        context.memory,
        mutex_address,
    );
    state.write_x(0, result);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x1c)]
fn svc_wait_process_wide_key_atomic(
    #[x(0)] mutex_address: u64,
    #[x(1)] cv_key: u64,
    #[x(2)] tag: u32,
    #[x(3)] timeout_ns: i64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if (mutex_address & 3) != 0 {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let cv_key = cv_key & !3;
    // Write a sentinel to the condvar address.  On real HOS the kernel
    // does NOT touch the condvar value (the application manages it), but
    // in our cooperative model the main-thread condvar-wait is skipped
    // and must see a changed value so libnx's spin-loop terminates.
    if !context.memory.write_u32(cv_key, 1) {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    let release_result = release_mutex_to_next_waiter(
        kernel,
        context.current_thread_handle,
        context.memory,
        mutex_address,
    );
    if release_result != RESULT_SUCCESS {
        state.write_x(0, release_result);
        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    state.write_x(0, RESULT_SUCCESS);
    if timeout_ns != 0 && context.current_thread_handle != MAIN_THREAD_HANDLE {
        kernel.wait_for_address_with_key(
            context.current_thread_handle,
            state,
            cv_key,
            timeout_ns,
            mutex_address,
            tag,
        );
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x1d)]
fn svc_signal_process_wide_key(
    #[x(0)] cv_key: u64,
    #[x(1)] _count: i32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    let cv_key = cv_key & !3;
    let waiters = kernel.take_waiters_for_address(context.current_thread_handle, cv_key, _count);
    for waiter_thread_handle in waiters {
        let Some((mutex_address, requester_tag)) = kernel.thread_address_key(waiter_thread_handle)
        else {
            let _ = kernel.wake_thread(waiter_thread_handle);
            continue;
        };

        let Some(mutex_value) = context.memory.read_u32(mutex_address) else {
            let _ = kernel.wake_thread(waiter_thread_handle);
            continue;
        };

        if mutex_value == 0 {
            if context.memory.write_u32(mutex_address, requester_tag) {
                let _ = kernel.wake_thread(waiter_thread_handle);
            } else {
                let _ = kernel.wake_thread(waiter_thread_handle);
            }
            continue;
        }

        let _ = context
            .memory
            .write_u32(mutex_address, mutex_value | HANDLE_WAIT_MASK);
        if !kernel.move_blocked_thread_to_address_wait(waiter_thread_handle, mutex_address, -1) {
            let _ = kernel.wake_thread(waiter_thread_handle);
        }
    }

    // Reset the condvar value when no waiters remain, so the
    // main-thread's non-blocking condvar-wait path sees a deterministic
    // value (mirroring the write(1) in svc_wait_process_wide_key_atomic).
    if kernel.waiter_count_for_address(context.current_thread_handle, cv_key) == 0 {
        let _ = context.memory.write_u32(cv_key, 0);
    }

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x1e)]
fn svc_get_system_tick(context: &mut SvcContext<'_>) -> SvcResult {
    context.state.write_x(0, virtual_counter_ticks());

    SyscallDisposition::Continue
}

#[svc(0x1f)]
fn svc_connect_to_named_port(
    #[x(0)] out_handle_ptr: u64,
    #[x(1)] name_ptr: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    let service_name = read_guest_c_string(memory, name_ptr, 12).filter(|name| name.len() <= 11);
    let result = service_name
        .as_deref()
        .and_then(|name| kernel.connect_to_named_port(context.current_thread_handle, name));

    if let Some(handle) = result {
        if memory.write_u32(out_handle_ptr, handle) {
            let stored = memory.read_u32(out_handle_ptr).unwrap_or(0xffff_ffff);
            let stored_wide = memory.read_u64(out_handle_ptr).unwrap_or(u64::MAX);
            eprintln!(
                "[svc::ConnectToNamedPort] name={:?} out=0x{out_handle_ptr:016x} handle={} stored={} stored64=0x{stored_wide:016x}",
                service_name, handle, stored,
            );
            state.write_x(0, RESULT_SUCCESS);
            state.write_x(1, u64::from(handle));
        } else {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.write_x(1, 0);
        }
    } else {
        state.write_x(0, RESULT_NOT_FOUND);
        state.write_x(1, 0);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x21)]
fn svc_send_sync_request(#[x(0)] handle: u32, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if let Some(session_kind) =
        kernel.service_session_for_handle(context.current_thread_handle, handle)
        && let Some(request) = parse_hipc_request(memory, state.tpidrro_el0)
    {
        let command_id = parse_cmif_command_id(memory, request);
        let is_domain = has_domain_message_header(memory, request);
        let obj_info = if is_domain {
            if let Some((_dcmd, oid)) = parse_domain_header(memory, request) {
                let okind = kernel.service_session_domain_object_kind(
                    context.current_thread_handle,
                    handle,
                    oid,
                );
                format!(" obj_id={oid} obj_kind={okind:?}")
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        eprintln!(
            "[svc::SendSyncRequest:dispatch] handle={handle} kind={session_kind:?} type={} domain={} cmd={command_id:?}{obj_info}",
            request.message_type, is_domain,
        );
    }

    let result = ServiceManager::dispatch_send_sync_request(
        kernel,
        context.current_thread_handle,
        handle,
        memory,
        state.tpidrro_el0,
    );

    state.write_x(0, result);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x24)]
fn svc_get_process_id(
    #[x(0)] out_ptr: u64,
    #[x(1)] handle: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if let Some(pid) = kernel.process_pid_for_handle(context.current_thread_handle, handle) {
        if memory.write_u64(out_ptr, pid) {
            state.write_x(0, RESULT_SUCCESS);
            state.write_x(1, pid);
        } else {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.write_x(1, 0);
        }
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.write_x(1, 0);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x25)]
fn svc_get_thread_id(
    #[x(0)] out_ptr: u64,
    #[x(1)] handle: u32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if let Some(thread_uid) = kernel.thread_uid_for_handle(context.current_thread_handle, handle) {
        if memory.write_u64(out_ptr, thread_uid) {
            state.write_x(0, RESULT_SUCCESS);
            state.write_x(1, thread_uid);
        } else {
            state.write_x(0, RESULT_INVALID_MEMORY_STATE);
            state.write_x(1, 0);
        }
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
        state.write_x(1, 0);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum BreakReason {
    Panic = 0,
    Assert = 1,
    User = 2,
    PreLoadDll = 3,
    PostLoadDll = 4,
    PreUnloadDll = 5,
    PostUnloadDll = 6,
    CppException = 7,

    #[allow(unused)]
    NotificationOnlyFlag = 0x80000000,
}

impl TryFrom<u64> for BreakReason {
    type Error = &'static str;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BreakReason::Panic),
            1 => Ok(BreakReason::Assert),
            2 => Ok(BreakReason::User),
            3 => Ok(BreakReason::PreLoadDll),
            4 => Ok(BreakReason::PostLoadDll),
            5 => Ok(BreakReason::PreUnloadDll),
            6 => Ok(BreakReason::PostUnloadDll),
            7 => Ok(BreakReason::CppException),
            _ => Err("unknown break reason"),
        }
    }
}

#[svc(0x26)]
fn svc_break(
    #[x(0)] reason: u64,
    #[x(1)] _address: u64,
    #[x(2)] _size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    state.clear_exit();

    let break_reason = BreakReason::try_from(reason).expect("unknown break reason");
    let detail_word = if _size >= 4 {
        memory.read_u32(_address)
    } else {
        None
    };
    let frame = state.read_x(29);
    let caller_lr = memory.read_u64(frame.wrapping_add(8));
    let parent_frame = memory.read_u64(frame);
    let parent_lr = parent_frame.and_then(|fp| memory.read_u64(fp.wrapping_add(8)));
    let grandparent_frame = parent_frame.and_then(|fp| memory.read_u64(fp));
    let grandparent_lr = grandparent_frame.and_then(|fp| memory.read_u64(fp.wrapping_add(8)));
    eprintln!(
        "[svc::Break] reason={break_reason:?} address=0x{_address:016x} size=0x{_size:x} detail={detail_word:?} lr=0x{:016x} fp=0x{frame:016x} caller_lr={caller_lr:?} parent_fp={parent_frame:?} parent_lr={parent_lr:?} grandparent_fp={grandparent_frame:?} grandparent_lr={grandparent_lr:?}",
        state.read_x(30),
    );

    let ignore_panic_break = std::env::var("RYURS_IGNORE_PANIC_BREAK")
        .map(|value| value == "1")
        .unwrap_or(false);
    if ignore_panic_break && matches!(break_reason, BreakReason::Panic) {
        state.write_x(0, RESULT_SUCCESS);
        return SyscallDisposition::Continue;
    }

    // For worker threads: exit only the crashing thread, not the whole process.
    // Killing all threads on a worker panic prevents the game from making
    // further progress (the main thread becomes a zombie that spins forever).
    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        SyscallDisposition::ThreadExit
    } else if (reason as u64 & (1 << 31)) == 0 {
        if kernel.exit_process(context.current_thread_handle, state) {
            SyscallDisposition::ThreadExit
        } else {
            SyscallDisposition::Continue
        }
    } else {
        SyscallDisposition::Continue
    }
}

#[svc(0x27)]
fn svc_output_debug_string(
    #[x(0)] address: u64,
    #[x(1)] size: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let state = &mut *context.state;
    let memory = context.memory;

    if let Some(bytes) = read_guest_buffer(memory, address, size) {
        let text = String::from_utf8_lossy(&bytes);
        eprintln!("[svc::OutputDebugString] {}", text.trim_end_matches('\0'));
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_MEMORY_STATE);
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x29)]
fn svc_get_info(
    #[x(0)] out_ptr: u64,
    #[x(1)] info_type: u32,
    #[x(2)] handle: u32,
    #[x(3)] sub_id: i64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;
    let current_thread_handle = context.current_thread_handle;

    let value_or_error = match info_type {
        0 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| get_info_core_mask())
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        1 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| u64::MAX)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        2 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, _)| base)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        3 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(_, size)| size)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        4 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, _)| base)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        5 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(_, size)| size)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        6 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(_, size)| reported_total_memory_size(size))
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        7 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, size)| process_used_memory_size(memory, base, size))
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        8 => {
            if handle != 0 {
                Err(RESULT_INVALID_HANDLE)
            } else if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                Ok(0)
            }
        }
        9 => {
            if handle != 0 {
                Err(RESULT_INVALID_HANDLE)
            } else if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                // ResourceLimit handle is optional; return 0 when none is configured.
                Ok(0)
            }
        }
        10 => {
            if handle != 0 {
                Err(RESULT_INVALID_HANDLE)
            } else {
                let current_core = kernel
                    .thread_core_mask_for_handle(current_thread_handle, MAIN_THREAD_HANDLE)
                    .map(|(core_id, _)| i64::from(core_id))
                    .unwrap_or(0);
                if sub_id != -1 && sub_id != current_core {
                    Err(RESULT_INVALID_COMBINATION)
                } else {
                    // Scheduler idle accounting is not modeled yet.
                    Ok(0)
                }
            }
        }
        11 => {
            if handle != 0 {
                Err(RESULT_INVALID_HANDLE)
            } else {
                kernel
                    .random_entropy_word(current_thread_handle, sub_id)
                    .ok_or(RESULT_INVALID_COMBINATION)
            }
        }
        12 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, _)| base)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        13 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| KERNEL_ASLR_REGION_SIZE_64BIT)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        14 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, size)| {
                        let stack_region_size = stack_region_size_for_process(size);
                        base.saturating_add(size).saturating_sub(stack_region_size)
                    })
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        15 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(_, size)| stack_region_size_for_process(size))
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        16 | 17 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 0)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        18 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_pid_for_handle(current_thread_handle, handle)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        20 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 0)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        21 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(_, size)| reported_total_memory_size(size))
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        22 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|(base, size)| process_used_memory_size(memory, base, size))
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        23 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 1)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        24 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 0)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        25 => {
            if !(-1..=3).contains(&sub_id) {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                let current_core = kernel
                    .thread_core_mask_for_handle(current_thread_handle, MAIN_THREAD_HANDLE)
                    .map(|(core_id, _)| i64::from(core_id))
                    .unwrap_or(0);
                kernel
                    .thread_uid_for_handle(current_thread_handle, handle)
                    .ok_or(RESULT_INVALID_HANDLE)
                    .map(|_| {
                        if sub_id != -1 && sub_id != current_core {
                            0
                        } else {
                            // Per-thread tick accounting is not modeled yet.
                            0
                        }
                    })
            }
        }
        26 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 1)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        27 | 28 | 29 => {
            if sub_id != 0 {
                Err(RESULT_INVALID_COMBINATION)
            } else {
                kernel
                    .process_address_space_for_handle(current_thread_handle, handle)
                    .map(|_| 0)
                    .ok_or(RESULT_INVALID_HANDLE)
            }
        }
        _ => {
            eprintln!(
                "unknown info type {} (handle={}, sub_id={})",
                info_type, handle, sub_id
            );
            Err(RESULT_INVALID_COMBINATION)
        }
    };

    match value_or_error {
        Ok(value) => {
            if memory.write_u64(out_ptr, value) {
                state.write_x(0, RESULT_SUCCESS);
                state.write_x(1, value);
            } else {
                state.write_x(0, RESULT_INVALID_MEMORY_STATE);
                state.write_x(1, 0);
            }
        }
        Err(error) => {
            state.write_x(0, error);
            state.write_x(1, 0);
        }
    }

    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x34)]
fn svc_wait_for_address(
    #[x(0)] address: u64,
    #[x(1)] arbitration_type: u32,
    #[x(2)] arbitration_value: u32,
    #[x(3)] timeout_ns: i64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    if context.current_thread_handle != MAIN_THREAD_HANDLE {
        let current = memory.read_u32(address).unwrap_or(0);
        let should_wait = match arbitration_type {
            0 => current < arbitration_value,
            1 => {
                if current < arbitration_value {
                    let _ = memory.write_u32(address, current.wrapping_sub(1));
                    true
                } else {
                    false
                }
            }
            2 => current == arbitration_value,
            _ => false,
        };

        state.write_x(0, RESULT_SUCCESS);
        if should_wait && timeout_ns != 0 {
            kernel.wait_for_address(context.current_thread_handle, state, address, timeout_ns);
        }

        state.clear_exit();
        return SyscallDisposition::Continue;
    }

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x35)]
fn svc_signal_to_address(
    #[x(0)] address: u64,
    #[x(1)] signal_type: u32,
    #[x(2)] signal_value: u32,
    #[x(3)] count: i32,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let memory = context.memory;

    match signal_type {
        0 => {
            let _ = kernel.signal_to_address(context.current_thread_handle, address, count);
        }
        1 => {
            if let Some(current) = memory.read_u32(address) {
                if current == signal_value {
                    let _ = memory.write_u32(address, current.wrapping_add(1));
                    let _ = kernel.signal_to_address(context.current_thread_handle, address, count);
                }
            }
        }
        2 => {
            if let Some(current) = memory.read_u32(address) {
                if current == signal_value {
                    let waiting_count =
                        kernel.waiter_count_for_address(context.current_thread_handle, address);
                    let addend = if waiting_count > 0 {
                        if count <= 0 {
                            -2i32
                        } else if waiting_count < count as usize {
                            -1i32
                        } else {
                            0i32
                        }
                    } else {
                        1i32
                    };
                    let _ = memory.write_u32(address, current.wrapping_add_signed(addend));
                    let _ = kernel.signal_to_address(context.current_thread_handle, address, count);
                }
            }
        }
        _ => {
            unimplemented!("[svc::SignalToAddress] unsupported signal_type={signal_type}")
        }
    }

    state.write_x(0, RESULT_SUCCESS);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x36)]
fn svc_synchronize_preemption_state(context: &mut SvcContext<'_>) -> SvcResult {
    // SynchronizePreemptionState: no-op for our single-process emulator.
    // Just return success without doing anything.
    let state = &mut *context.state;
    state.write_x(0, 0);
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x79)]
fn svc_create_process(
    #[x(0)] entrypoint: u64,
    #[x(1)] argument: u64,
    #[x(2)] stack_top: u64,
    context: &mut SvcContext<'_>,
) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let process_memory = context.memory.fork();
    let (process_handle, main_thread_handle) = kernel.create_process(
        context.current_thread_handle,
        entrypoint,
        argument,
        stack_top,
        state,
        process_memory,
    );

    state.write_x(0, RESULT_SUCCESS);
    state.write_x(1, u64::from(process_handle));
    state.write_x(2, u64::from(main_thread_handle));
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x7a)]
fn svc_start_process(#[x(0)] process_handle: u32, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;

    if kernel.start_process(context.current_thread_handle, process_handle) {
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
    }
    state.clear_exit();
    SyscallDisposition::Continue
}

#[svc(0x7b)]
fn svc_terminate_process(#[x(0)] handle: u32, context: &mut SvcContext<'_>) -> SvcResult {
    let kernel = &mut *context.kernel;
    let state = &mut *context.state;
    let target_process = kernel.resolve_process_handle(context.current_thread_handle, handle);
    let current_process_handle = kernel.process_handle_for_thread(context.current_thread_handle);

    if kernel.terminate_process(context.current_thread_handle, handle) {
        state.write_x(0, RESULT_SUCCESS);
    } else {
        state.write_x(0, RESULT_INVALID_HANDLE);
    }
    state.clear_exit();

    if target_process == current_process_handle && target_process == Some(MAIN_PROCESS_HANDLE) {
        SyscallDisposition::ThreadExit
    } else {
        SyscallDisposition::Continue
    }
}

const fn register_svc_service(
    services: &mut [Option<SvcService>; MAX_SERVICES],
    service: SvcService,
) {
    let index = service.svc_number as usize;
    if index < services.len() {
        services[index] = Some(service);
    } else {
        panic!("Svc number is out of bounds");
    }
}

const fn build_svc_services() -> [Option<SvcService>; MAX_SERVICES] {
    let mut services: [Option<SvcService>; MAX_SERVICES] = [None; MAX_SERVICES];

    register_svc_service(&mut services, svc_set_heap_size_service());
    register_svc_service(&mut services, svc_set_memory_permission_service());
    register_svc_service(&mut services, svc_set_memory_attribute_service());
    register_svc_service(&mut services, svc_map_memory_service());
    register_svc_service(&mut services, svc_unmap_memory_service());
    register_svc_service(&mut services, svc_query_memory_service());
    register_svc_service(&mut services, svc_exit_process_service());
    register_svc_service(&mut services, svc_create_thread_service());
    register_svc_service(&mut services, svc_start_thread_service());
    register_svc_service(&mut services, svc_exit_thread_service());
    register_svc_service(&mut services, svc_sleep_thread_service());
    register_svc_service(&mut services, svc_get_thread_priority_service());
    register_svc_service(&mut services, svc_set_thread_priority_service());
    register_svc_service(&mut services, svc_get_thread_core_mask_service());
    register_svc_service(&mut services, svc_set_thread_core_mask_service());
    register_svc_service(&mut services, svc_get_current_processor_number_service());
    register_svc_service(&mut services, svc_map_shared_memory_service());
    register_svc_service(&mut services, svc_unmap_shared_memory_service());
    register_svc_service(&mut services, svc_create_transfer_memory_service());
    register_svc_service(&mut services, svc_close_handle_service());
    register_svc_service(&mut services, svc_wait_synchronization_service());
    register_svc_service(&mut services, svc_arbitrate_lock_service());
    register_svc_service(&mut services, svc_arbitrate_unlock_service());
    register_svc_service(&mut services, svc_wait_process_wide_key_atomic_service());
    register_svc_service(&mut services, svc_signal_process_wide_key_service());
    register_svc_service(&mut services, svc_get_system_tick_service());
    register_svc_service(&mut services, svc_connect_to_named_port_service());
    register_svc_service(&mut services, svc_send_sync_request_service());
    register_svc_service(&mut services, svc_get_process_id_service());
    register_svc_service(&mut services, svc_get_thread_id_service());
    register_svc_service(&mut services, svc_break_service());
    register_svc_service(&mut services, svc_output_debug_string_service());
    register_svc_service(&mut services, svc_get_info_service());
    register_svc_service(&mut services, svc_wait_for_address_service());
    register_svc_service(&mut services, svc_signal_to_address_service());
    register_svc_service(&mut services, svc_synchronize_preemption_state_service());
    register_svc_service(&mut services, svc_create_process_service());
    register_svc_service(&mut services, svc_start_process_service());
    register_svc_service(&mut services, svc_terminate_process_service());

    services
}

fn svc_services() -> &'static [Option<SvcService>; MAX_SERVICES] {
    static SERVICES: [Option<SvcService>; MAX_SERVICES] = build_svc_services();
    &SERVICES
}

fn trace_service_traffic_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RYURS_TRACE_SERVICE_TRAFFIC")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

fn trace_svc_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RYURS_TRACE_SVC")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

pub fn dispatch_supervisor_call(
    kernel: &mut KernelState,
    current_thread_handle: u32,
    imm16: u16,
    state: &mut CpuState,
    memory: &SharedGuestMemory,
) -> SyscallDisposition {
    let entry_pc = state.pc;
    let arg0 = state.read_x(0);
    let arg1 = state.read_x(1);
    let arg2 = state.read_x(2);
    let arg3 = state.read_x(3);

    let service = svc_services().get(imm16 as usize).and_then(|entry| *entry);

    let disposition = if let Some(service) = service {
        let mut context = SvcContext {
            kernel,
            current_thread_handle,
            state,
            memory,
        };
        (service.handler)(&mut context)
    } else {
        SyscallDisposition::Unhandled
    };

    if trace_svc_enabled() {
        let svc_name = service.map(|svc| svc.name).unwrap_or("Unknown");
        eprintln!(
            "[svc] tid={} pc=0x{:016x} imm=0x{:04x} {} x0=0x{:016x} x1=0x{:016x} x2=0x{:016x} x3=0x{:016x} -> {:?} ret_x0=0x{:016x} ret_x1=0x{:016x}",
            current_thread_handle,
            entry_pc,
            imm16,
            svc_name,
            arg0,
            arg1,
            arg2,
            arg3,
            disposition,
            state.read_x(0),
            state.read_x(1),
        );
    }

    disposition
}

#[cfg(test)]
mod tests {
    use super::{
        CMIF_IN_HEADER_MAGIC, CMIF_OUT_HEADER_MAGIC, DOMAIN_MESSAGE_HEADER_SIZE, HANDLE_WAIT_MASK,
        HIPC_TYPE_CMIF_CLOSE, HIPC_TYPE_CMIF_CONTROL, HIPC_TYPE_CMIF_REQUEST, HipcRequestInfo,
        NVDRV_RESULT_INVALID_STATE, NVDRV_RESULT_SUCCESS, RESULT_AM_NO_MESSAGES,
        RESULT_INVALID_COMBINATION, RESULT_INVALID_HANDLE, RESULT_INVALID_MEMORY_STATE,
        RESULT_NOT_FOUND, RESULT_OUT_OF_MEMORY, RESULT_SUCCESS, RESULT_TIMED_OUT,
        SM_POINTER_BUFFER_SIZE, SyscallDisposition, decode_hipc_buffer_descriptor,
        decode_hipc_static_descriptor, dispatch_supervisor_call, has_domain_message_header,
        parse_domain_header, read_guest_memory_span,
    };
    use crate::hle::KernelState;
    use crate::hle::kernel::{
        MAIN_PROCESS_HANDLE, MAIN_THREAD_HANDLE, SELF_PROCESS_HANDLE, SELF_THREAD_HANDLE,
        ServiceSessionKind,
    };
    use crate::memory::SharedGuestMemory;
    use ryurs_dynarec::{CpuState, ExitReason, MemoryInterface, StepResult};

    const MAIN_PC: u64 = 0x8000_0000;
    const THREAD_PC: u64 = 0x8000_1000;
    const THREAD_SP: u64 = 0x8400_0000;
    const THREAD_ARG: u64 = 0x1234;

    #[test]
    fn read_guest_memory_span_handles_large_high_regions() {
        let memory = SharedGuestMemory::new(MAIN_PC, 0x10_0000);
        let start = 0x84bd_4000;
        let size = 0x20_000usize;
        let mut view = memory.locked_view();

        assert!(memory.map_high_region(start, size as u64, 0xb, 3));
        for offset in 0..size {
            assert!(view.write_u8(start + offset as u64, (offset & 0xff) as u8));
        }

        let bytes = read_guest_memory_span(&memory, start, size).expect("reads high-region span");
        assert_eq!(bytes.len(), size);
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x01);
        assert_eq!(bytes[0x1234], 0x34);
        assert_eq!(bytes[size - 1], 0xff);
    }

    fn write_cmif_request_no_special(
        memory: &SharedGuestMemory,
        tls: u64,
        message_type: u16,
        command_id: u32,
        payload: &[u8],
    ) {
        let data_words_address = tls + 8;
        let data_start = (data_words_address + 15) & !15;
        let payload_address = data_start + 16;
        let end = payload_address + payload.len() as u64;
        let num_data_words = ((end - data_words_address + 3) / 4) as u32;

        assert!(memory.write_u32(tls, message_type as u32));
        assert!(memory.write_u32(tls + 4, num_data_words & 0x3ff));
        assert!(memory.write_u32(data_start, CMIF_IN_HEADER_MAGIC));
        assert!(memory.write_u32(data_start + 4, 0));
        assert!(memory.write_u32(data_start + 8, command_id));
        assert!(memory.write_u32(data_start + 12, 0));

        for (chunk_index, chunk) in payload.chunks(4).enumerate() {
            let mut packed = [0u8; 4];
            packed[..chunk.len()].copy_from_slice(chunk);
            assert!(memory.write_u32(
                payload_address + (chunk_index as u64) * 4,
                u32::from_le_bytes(packed)
            ));
        }
    }

    fn write_cmif_domain_request_no_special(
        memory: &SharedGuestMemory,
        tls: u64,
        message_type: u16,
        object_id: u32,
        command_id: u32,
        payload: &[u8],
    ) {
        let data_words_address = tls + 8;
        let data_start = (data_words_address + 15) & !15;
        let cmif_header_address = data_start + 16;
        let payload_address = cmif_header_address + 16;
        let end = payload_address + payload.len() as u64;
        let num_data_words = ((end - data_words_address + 3) / 4) as u32;

        assert!(memory.write_u32(tls, message_type as u32));
        assert!(memory.write_u32(tls + 4, num_data_words & 0x3ff));

        // Domain invoke command for an existing object id.
        assert!(memory.write_u32(data_start, 0x0010_0001));
        assert!(memory.write_u32(data_start + 4, object_id));
        assert!(memory.write_u32(data_start + 8, 0));
        assert!(memory.write_u32(data_start + 12, 0));

        assert!(memory.write_u32(cmif_header_address, CMIF_IN_HEADER_MAGIC));
        assert!(memory.write_u32(cmif_header_address + 4, 1));
        assert!(memory.write_u32(cmif_header_address + 8, command_id));
        assert!(memory.write_u32(cmif_header_address + 12, 0));

        for (chunk_index, chunk) in payload.chunks(4).enumerate() {
            let mut packed = [0u8; 4];
            packed[..chunk.len()].copy_from_slice(chunk);
            assert!(memory.write_u32(
                payload_address + (chunk_index as u64) * 4,
                u32::from_le_bytes(packed)
            ));
        }
    }

    fn write_cmif_request_with_ptr_buffer(
        memory: &SharedGuestMemory,
        tls: u64,
        message_type: u16,
        command_id: u32,
        payload: &[u8],
        ptr_address: u64,
        ptr_size: u16,
    ) {
        let descriptor_bytes = 8u64;
        let data_words_address = tls + 8 + descriptor_bytes;
        let data_start = (data_words_address + 15) & !15;
        let payload_address = data_start + 16;
        let end = payload_address + payload.len() as u64;
        let num_data_words = ((end - data_words_address + 3) / 4) as u32;

        let mut word0 = message_type as u32;
        word0 |= 1 << 16;

        assert!(memory.write_u32(tls, word0));
        assert!(memory.write_u32(tls + 4, num_data_words & 0x3ff));

        let mut ptr_word0 = 0u32;
        ptr_word0 |= ((ptr_address & 0x0f00_0000_00) >> 20) as u32;
        ptr_word0 |= ((ptr_address & 0x7000_0000_00) >> 30) as u32;
        ptr_word0 |= (ptr_size as u32) << 16;
        let ptr_word1 = ptr_address as u32;

        assert!(memory.write_u32(tls + 8, ptr_word0));
        assert!(memory.write_u32(tls + 12, ptr_word1));

        assert!(memory.write_u32(data_start, CMIF_IN_HEADER_MAGIC));
        assert!(memory.write_u32(data_start + 4, 0));
        assert!(memory.write_u32(data_start + 8, command_id));
        assert!(memory.write_u32(data_start + 12, 0));

        for (chunk_index, chunk) in payload.chunks(4).enumerate() {
            let mut packed = [0u8; 4];
            packed[..chunk.len()].copy_from_slice(chunk);
            assert!(memory.write_u32(
                payload_address + (chunk_index as u64) * 4,
                u32::from_le_bytes(packed)
            ));
        }
    }

    fn write_cmif_request_with_recv_buffer(
        memory: &SharedGuestMemory,
        tls: u64,
        message_type: u16,
        command_id: u32,
        payload: &[u8],
        recv_address: u64,
        recv_size: u64,
    ) {
        let descriptor_bytes = 12u64;
        let data_words_address = tls + 8 + descriptor_bytes;
        let data_start = (data_words_address + 15) & !15;
        let payload_address = data_start + 16;
        let end = payload_address + payload.len() as u64;
        let num_data_words = ((end - data_words_address + 3) / 4) as u32;

        let mut word0 = message_type as u32;
        word0 |= 1 << 24;

        assert!(memory.write_u32(tls, word0));
        assert!(memory.write_u32(tls + 4, num_data_words & 0x3ff));

        let recv_word0 = recv_size as u32;
        let recv_word1 = recv_address as u32;
        let mut recv_word2 = 0u32;
        recv_word2 |= ((recv_address & 0x0f00_0000_00) >> 4) as u32;
        recv_word2 |= ((recv_address & 0x7000_0000_00) >> 34) as u32;
        recv_word2 |= ((recv_size & 0x0f00_0000_00) >> 8) as u32;
        recv_word2 |= 1;

        assert!(memory.write_u32(tls + 8, recv_word0));
        assert!(memory.write_u32(tls + 12, recv_word1));
        assert!(memory.write_u32(tls + 16, recv_word2));

        assert!(memory.write_u32(data_start, CMIF_IN_HEADER_MAGIC));
        assert!(memory.write_u32(data_start + 4, 0));
        assert!(memory.write_u32(data_start + 8, command_id));
        assert!(memory.write_u32(data_start + 12, 0));

        for (chunk_index, chunk) in payload.chunks(4).enumerate() {
            let mut packed = [0u8; 4];
            packed[..chunk.len()].copy_from_slice(chunk);
            assert!(memory.write_u32(
                payload_address + (chunk_index as u64) * 4,
                u32::from_le_bytes(packed)
            ));
        }
    }

    #[test]
    fn create_start_and_exit_worker_thread() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory,),
            SyscallDisposition::Continue
        );

        let handle = state.read_x(1) as u32;
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        state.write_x(0, u64::from(handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x09, &mut state, &memory,),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be runnable");
        let mut worker_state = worker_task.state;
        let worker_thread_handle = worker_task.thread_handle;
        let thread_ptr = THREAD_SP & !0xfff;
        assert_eq!(worker_state.pc, THREAD_PC);
        assert_eq!(worker_state.sp, THREAD_SP & !0xf);
        assert_eq!(worker_state.tpidrro_el0, thread_ptr);
        assert_eq!(worker_state.read_x(0), THREAD_ARG);
        assert_eq!(worker_state.read_x(1), u64::from(handle));

        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_thread_handle,
                0x0a,
                &mut worker_state,
                &memory,
            ),
            SyscallDisposition::Continue
        );
    }

    #[test]
    fn create_thread_initializes_tls_metadata_when_stack_page_is_mapped() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let mapped_stack_top = MAIN_PC + 0x1f000;
        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, mapped_stack_top);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let handle = state.read_x(1) as u32;
        let thread_ptr = mapped_stack_top & !0xfff;
        assert_eq!(memory.read_u64(thread_ptr + 0x1f8), Some(thread_ptr));
        assert_eq!(memory.read_u32(thread_ptr + 0x1b0), Some(handle));
    }

    #[test]
    fn exit_main_thread_without_runnable_peers_returns_thread_exit() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0a, &mut state, &memory,),
            SyscallDisposition::ThreadExit
        );
    }

    #[test]
    fn wait_for_address_and_signal_to_address_roundtrip() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let address = 0x4000_0000;
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory,),
            SyscallDisposition::Continue
        );
        let handle = state.read_x(1) as u32;

        state.write_x(0, u64::from(handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x09, &mut state, &memory,),
            SyscallDisposition::Continue
        );

        let worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be runnable");
        let mut worker_state = worker_task.state;
        let worker_thread_handle = worker_task.thread_handle;

        let _ = memory.write_u32(address, 0);

        worker_state.write_x(0, address);
        worker_state.write_x(1, 2);
        worker_state.write_x(2, 0);
        worker_state.write_x(3, 1_000_000);
        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_thread_handle,
                0x34,
                &mut worker_state,
                &memory,
            ),
            SyscallDisposition::Continue
        );

        assert_eq!(
            kernel.waiter_count_for_address(worker_thread_handle, address),
            1
        );
        assert!(kernel.take_runnable_worker_thread().is_none());

        state.write_x(0, address);
        state.write_x(1, 0);
        state.write_x(2, 0);
        state.write_x(3, 1);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x35, &mut state, &memory,),
            SyscallDisposition::Continue
        );

        let resumed = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be signaled");
        assert_eq!(resumed.thread_handle, worker_thread_handle);
        assert_eq!(resumed.state.read_x(0), RESULT_SUCCESS);
    }

    #[test]
    fn query_memory_populates_memory_info_structure() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let info_ptr = MAIN_PC + 0x100;
        let page_info_ptr = MAIN_PC + 0x200;
        state.write_x(0, info_ptr);
        state.write_x(1, page_info_ptr);
        state.write_x(2, 0);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x06, &mut state, &memory),
            SyscallDisposition::Continue
        );

        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(info_ptr), Some(0));
        assert_eq!(memory.read_u64(page_info_ptr), Some(0));
    }

    #[test]
    fn get_process_id_returns_pid_for_process_and_thread_handles() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x240;

        state.write_x(0, out_ptr);
        state.write_x(1, u64::from(MAIN_PROCESS_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x24, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(state.read_x(1), 1);
        assert_eq!(memory.read_u64(out_ptr), Some(1));

        state.write_x(0, out_ptr);
        state.write_x(1, u64::from(MAIN_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x24, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(state.read_x(1), 1);
        assert_eq!(memory.read_u64(out_ptr), Some(1));
    }

    #[test]
    fn get_and_set_thread_priority_roundtrip() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x340;
        state.write_x(0, out_ptr);
        state.write_x(1, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0c, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert!(memory.read_u32(out_ptr).is_some());

        state.write_x(0, u64::from(SELF_THREAD_HANDLE));
        state.write_x(1, 17);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0d, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        state.write_x(0, out_ptr);
        state.write_x(1, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0c, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(out_ptr), Some(17));

        state.write_x(0, out_ptr);
        state.write_x(1, 0xdead_beef);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0c, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_HANDLE);
    }

    #[test]
    fn get_and_set_thread_core_mask_roundtrip() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_core_id = MAIN_PC + 0x348;
        let out_affinity_mask = MAIN_PC + 0x350;

        state.write_x(0, out_core_id);
        state.write_x(1, out_affinity_mask);
        state.write_x(2, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0e, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let initial_mask = memory
            .read_u64(out_affinity_mask)
            .expect("thread affinity mask should be written");
        assert_ne!(initial_mask, 0);

        let target_core = initial_mask.trailing_zeros() as i32;

        state.write_x(0, u64::from(SELF_THREAD_HANDLE));
        state.write_x(1, target_core as u64);
        state.write_x(2, initial_mask);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        state.write_x(0, out_core_id);
        state.write_x(1, out_affinity_mask);
        state.write_x(2, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0e, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(out_core_id), Some(target_core as u32));
        assert_eq!(memory.read_u64(out_affinity_mask), Some(initial_mask));

        state.write_x(0, u64::from(SELF_THREAD_HANDLE));
        state.write_x(1, 63);
        state.write_x(2, initial_mask);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_COMBINATION);

        state.write_x(0, u64::from(SELF_THREAD_HANDLE));
        state.write_x(1, u64::MAX);
        state.write_x(2, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_COMBINATION);
    }

    #[test]
    fn get_current_processor_number_matches_thread_core_mask() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x10, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let reported_core_id = state.read_x(0) as u32;

        let out_core_id = MAIN_PC + 0x388;
        let out_affinity_mask = MAIN_PC + 0x390;
        state.write_x(0, out_core_id);
        state.write_x(1, out_affinity_mask);
        state.write_x(2, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x0e, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(out_core_id), Some(reported_core_id));
    }

    #[test]
    fn create_transfer_memory_returns_handle_for_mapped_range() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let address = MAIN_PC + 0x4000;
        let size = 0x2000;
        let permission: u32 = 0;

        state.write_x(1, address);
        state.write_x(2, size);
        state.write_x(3, u64::from(permission));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x15, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let handle = state.read_x(1) as u32;
        assert_ne!(handle, 0);
        assert_eq!(
            kernel.transfer_memory_for_handle(MAIN_THREAD_HANDLE, handle),
            Some((address, size, permission))
        );

        state.write_x(0, u64::from(handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x16, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(
            kernel.transfer_memory_for_handle(MAIN_THREAD_HANDLE, handle),
            None
        );
    }

    #[test]
    fn create_transfer_memory_rejects_bad_ranges_and_permissions() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, MAIN_PC + 1);
        state.write_x(2, 0x1000);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x15, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_COMBINATION);

        state.write_x(1, MAIN_PC + 0x40_0000);
        state.write_x(2, 0x1000);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x15, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_MEMORY_STATE);

        state.write_x(1, MAIN_PC + 0x4000);
        state.write_x(2, 0x1000);
        state.write_x(3, 0x10);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x15, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_COMBINATION);
    }

    #[test]
    fn get_thread_id_returns_uid_for_thread_handle() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x360;
        state.write_x(0, out_ptr);
        state.write_x(1, u64::from(SELF_THREAD_HANDLE));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x25, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(1));

        state.write_x(0, out_ptr);
        state.write_x(1, 0xdead_beef);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x25, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_HANDLE);
    }

    #[test]
    fn connect_to_named_port_returns_sm_session_handle() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x3a0;
        let name_ptr = MAIN_PC + 0x3b0;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");

        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let handle = state.read_x(1) as u32;
        assert_ne!(handle, 0);
        assert_eq!(memory.read_u32(out_ptr), Some(handle));
        assert!(
            kernel
                .service_session_for_handle(MAIN_THREAD_HANDLE, handle)
                .is_some(),
            "session handle should resolve in caller process"
        );

        memory
            .load(name_ptr, b"foo\0")
            .expect("service name overwrite should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_NOT_FOUND);

        let tls = MAIN_PC + 0x3c0;
        state.tpidrro_el0 = tls;
        assert!(memory.write_u32(tls, HIPC_TYPE_CMIF_CLOSE as u32));
        assert!(memory.write_u32(tls + 4, 0));

        state.write_x(0, u64::from(handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        state.write_x(0, 0xdead_beef);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_HANDLE);
    }

    #[test]
    fn send_sync_sm_control_query_pointer_buffer_size_returns_cmif_payload() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x420;
        let name_ptr = MAIN_PC + 0x430;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x500;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 3, &[]);

        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        assert_eq!(memory.read_u32(tls + 16), Some(CMIF_OUT_HEADER_MAGIC));
        assert_eq!(memory.read_u32(tls + 24), Some(RESULT_SUCCESS as u32));
        assert_eq!(
            memory
                .read_u32(tls + 32)
                .map(|value| (value & 0xffff) as u16),
            Some(SM_POINTER_BUFFER_SIZE),
            "control cmd 3 should return pointer buffer size"
        );
    }

    #[test]
    fn send_sync_sm_get_service_returns_move_handle() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"vi:u\0\0\0\0");

        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let word1 = memory
            .read_u32(tls + 4)
            .expect("response word1 should exist");
        assert_ne!(word1 & (1 << 31), 0, "response should carry special header");
        let special = memory
            .read_u32(tls + 8)
            .expect("response special header should exist");
        assert_eq!((special >> 5) & 0xf, 1, "response should move one handle");

        let moved_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");
        assert_ne!(moved_handle, 0);
        assert_eq!(memory.read_u32(tls + 16), Some(CMIF_OUT_HEADER_MAGIC));
        assert_eq!(memory.read_u32(tls + 24), Some(RESULT_SUCCESS as u32));
        assert!(
            kernel
                .service_session_for_handle(MAIN_THREAD_HANDLE, moved_handle)
                .is_some(),
            "moved service handle should resolve in caller process"
        );
    }

    #[test]
    fn send_sync_sm_get_service_resolves_vi_m_and_vi_s_aliases() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x4a0;
        let name_ptr = MAIN_PC + 0x4b0;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x5c0;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"vi:m\0\0\0\0");

        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let moved_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, moved_handle),
            Some(ServiceSessionKind::ViU)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"vi:s\0\0\0\0");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let moved_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist for vi:s");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, moved_handle),
            Some(ServiceSessionKind::ViU)
        );
    }

    #[test]
    fn vi_root_get_display_service_returns_move_handle() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x4d0;
        let name_ptr = MAIN_PC + 0x4e0;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x600;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"vi:m\0\0\0\0");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let vi_root_handle = memory
            .read_u32(tls + 12)
            .expect("vi:m response should move one handle");

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 2, &0u32.to_le_bytes());
        state.write_x(0, u64::from(vi_root_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let response_word1 = memory
            .read_u32(tls + 4)
            .expect("response header word1 should exist");
        assert_ne!(
            response_word1 & (1 << 31),
            0,
            "response should carry a special header"
        );
        let special = memory
            .read_u32(tls + 8)
            .expect("response special header should exist");
        assert_eq!((special >> 5) & 0xf, 1, "response should move one handle");

        let moved_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, moved_handle),
            Some(ServiceSessionKind::ViU)
        );

        write_cmif_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            102,
            &0u64.to_le_bytes(),
        );
        state.write_x(0, u64::from(moved_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let response_word1 = memory
            .read_u32(tls + 4)
            .expect("response header word1 should exist");
        assert_ne!(
            response_word1 & (1 << 31),
            0,
            "response should carry a special header"
        );
        let special = memory
            .read_u32(tls + 8)
            .expect("response special header should exist");
        assert_eq!((special >> 5) & 0xf, 1, "response should move one handle");

        let moved_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist for cmd 102");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, moved_handle),
            Some(ServiceSessionKind::ViU)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1010, b"Default\0");
        state.write_x(0, u64::from(moved_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(tls + 32), Some(1));

        let mut create_stray_payload = Vec::new();
        create_stray_payload.extend_from_slice(&0u32.to_le_bytes()); // layer flags
        create_stray_payload.extend_from_slice(&0u32.to_le_bytes()); // padding
        create_stray_payload.extend_from_slice(&1u64.to_le_bytes()); // display id
        let native_window_address = MAIN_PC + 0x1a00;
        write_cmif_request_with_recv_buffer(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            2312,
            &create_stray_payload,
            native_window_address,
            0x100,
        );
        state.write_x(0, u64::from(moved_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let layer_id = memory
            .read_u64(tls + 32)
            .expect("CreateStrayLayer should return a layer id");
        assert_ne!(layer_id, 0);
        assert_eq!(memory.read_u64(tls + 40), Some(60));

        let payload_size = memory
            .read_u32(native_window_address)
            .expect("parcel payload_size");
        let payload_off = memory
            .read_u32(native_window_address + 4)
            .expect("parcel payload_off");
        let objects_size = memory
            .read_u32(native_window_address + 8)
            .expect("parcel objects_size");
        let objects_off = memory
            .read_u32(native_window_address + 12)
            .expect("parcel objects_off");
        let binder_type = memory
            .read_u32(native_window_address + 16)
            .expect("parcel binder type");
        let binder_id = memory
            .read_u32(native_window_address + 24)
            .expect("parcel binder object id");

        assert_eq!(payload_size, 40);
        assert_eq!(payload_off, 16);
        assert_eq!(objects_size, 4);
        assert_eq!(objects_off, 56);
        assert_eq!(binder_type, 2);
        assert_ne!(binder_id, 0);

        write_cmif_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            2021,
            &layer_id.to_le_bytes(),
        );
        state.write_x(0, u64::from(moved_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 32), Some(RESULT_SUCCESS as u32));
    }

    #[test]
    fn vi_relay_binder_commands_return_expected_stubbed_parcels() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x4d0;
        let name_ptr = MAIN_PC + 0x4e0;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("port name write should succeed");

        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let sm_handle = memory
            .read_u32(out_ptr)
            .expect("sm handle should be written");
        let tls = MAIN_PC + 0x680;
        state.tpidrro_el0 = tls;

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"vi:m\0\0\0\0");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let vi_root_handle = memory
            .read_u32(tls + 12)
            .expect("GetService should move vi root handle");

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 2, &0u32.to_le_bytes());
        state.write_x(0, u64::from(vi_root_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let app_display_handle = memory
            .read_u32(tls + 12)
            .expect("GetDisplayService should move app display handle");

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 100, &[]);
        state.write_x(0, u64::from(app_display_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let relay_handle = memory
            .read_u32(tls + 12)
            .expect("GetRelayService should move relay handle");

        let mut adjust_ref_payload = Vec::new();
        adjust_ref_payload.extend_from_slice(&1u32.to_le_bytes());
        adjust_ref_payload.extend_from_slice(&1u32.to_le_bytes());
        adjust_ref_payload.extend_from_slice(&0u32.to_le_bytes());
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, &adjust_ref_payload);
        state.write_x(0, u64::from(relay_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let mut get_native_handle_payload = Vec::new();
        get_native_handle_payload.extend_from_slice(&1u32.to_le_bytes());
        get_native_handle_payload.extend_from_slice(&0x0fu32.to_le_bytes());
        write_cmif_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            2,
            &get_native_handle_payload,
        );
        state.write_x(0, u64::from(relay_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let response_word1 = memory
            .read_u32(tls + 4)
            .expect("response header word1 should exist");
        assert_ne!(
            response_word1 & (1 << 31),
            0,
            "response should carry a special header"
        );
        let special = memory
            .read_u32(tls + 8)
            .expect("response special header should exist");
        assert_eq!((special >> 1) & 0xf, 1, "response should copy one handle");

        let first_native_handle = memory
            .read_u32(tls + 12)
            .expect("response copy handle should exist");
        assert_ne!(first_native_handle, 0);

        write_cmif_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            2,
            &get_native_handle_payload,
        );
        state.write_x(0, u64::from(relay_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 12), Some(first_native_handle));

        let mut transact_payload = Vec::new();
        transact_payload.extend_from_slice(&1u32.to_le_bytes());
        transact_payload.extend_from_slice(&10u32.to_le_bytes());
        transact_payload.extend_from_slice(&0u32.to_le_bytes());

        let parcel_reply_address = MAIN_PC + 0x1d00;
        write_cmif_request_with_recv_buffer(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            3,
            &transact_payload,
            parcel_reply_address,
            0x100,
        );
        state.write_x(0, u64::from(relay_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        assert_eq!(memory.read_u32(parcel_reply_address), Some(20));
        assert_eq!(memory.read_u32(parcel_reply_address + 4), Some(16));
        assert_eq!(memory.read_u32(parcel_reply_address + 8), Some(0));
        assert_eq!(memory.read_u32(parcel_reply_address + 12), Some(36));
        assert_eq!(memory.read_u32(parcel_reply_address + 16), Some(1280));
        assert_eq!(memory.read_u32(parcel_reply_address + 20), Some(720));
        assert_eq!(memory.read_u32(parcel_reply_address + 32), Some(0));
    }

    #[test]
    fn parse_domain_header_accepts_close_without_cmif_magic() {
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let data_start = MAIN_PC + 0x800;
        assert!(memory.write_u32(data_start, 2));
        assert!(memory.write_u32(data_start + 4, 2));

        let request = HipcRequestInfo {
            message_type: HIPC_TYPE_CMIF_REQUEST,
            data_start,
            data_end: data_start + DOMAIN_MESSAGE_HEADER_SIZE,
        };

        assert!(has_domain_message_header(&memory, request));
        assert_eq!(parse_domain_header(&memory, request), Some((2, 2)));
    }

    #[test]
    fn decode_hipc_static_descriptor_matches_ryujinx_layout() {
        let address = 0x0712_3456_78u64;
        let size = 0x2345u64;
        let index = 0x1abu32;

        let mut word0 = 0u32;
        word0 |= ((address & 0x0f00_0000_00) >> 20) as u32;
        word0 |= ((address & 0x7000_0000_00) >> 30) as u32;
        word0 |= (index & 0x03f) << 0;
        word0 |= (index & 0x1c0) << 3;
        word0 |= ((size & 0xffff) as u32) << 16;
        let word1 = address as u32;

        let decoded = decode_hipc_static_descriptor(word0, word1);
        assert_eq!(decoded.address, address);
        assert_eq!(decoded.size, size);
    }

    #[test]
    fn decode_hipc_buffer_descriptor_matches_ryujinx_layout() {
        let address = 0x0612_3456_78u64;
        let size = 0x0a12_3456_78u64;

        let word0 = (size & 0xffff_ffff) as u32;
        let word1 = address as u32;
        let mut word2 = 0u32;
        word2 |= ((address & 0x0f00_0000_00) >> 4) as u32;
        word2 |= ((address & 0x7000_0000_00) >> 34) as u32;
        word2 |= ((size & 0x0f00_0000_00) >> 8) as u32;
        word2 |= 1;

        let decoded = decode_hipc_buffer_descriptor(word0, word1, word2);
        assert_eq!(decoded.address, address);
        assert_eq!(decoded.size, size);
    }

    #[test]
    fn nvdrv_service_tracks_open_fds_and_query_event_handles() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"nvdrv\0\0\0");

        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let nvdrv_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, nvdrv_handle),
            Some(ServiceSessionKind::Nvdrv)
        );

        let open_payload = b"/dev/nvhost-ctrl-gpu\0";
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 0, open_payload);
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 36), Some(NVDRV_RESULT_SUCCESS));

        let fd = memory
            .read_u32(tls + 32)
            .expect("nvdrv open should return an fd") as i32;
        assert!(fd >= 0, "nvdrv open should produce a non-negative fd");

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 3, &[]);
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 32), Some(0));
        assert_eq!(memory.read_u32(tls + 36), Some(0));

        let mut query_payload = Vec::new();
        query_payload.extend_from_slice(&(fd as u32).to_le_bytes());
        query_payload.extend_from_slice(&1u32.to_le_bytes());
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 4, &query_payload);
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 32), Some(NVDRV_RESULT_SUCCESS));

        let special = memory
            .read_u32(tls + 8)
            .expect("special header should exist");
        assert_eq!((special >> 1) & 0xf, 1, "response should copy one handle");
        let first_event = memory
            .read_u32(tls + 12)
            .expect("event handle should exist");
        assert_ne!(first_event, 0);

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 4, &query_payload);
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 12), Some(first_event));

        write_cmif_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            2,
            &(fd as u32).to_le_bytes(),
        );
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 32), Some(NVDRV_RESULT_SUCCESS));

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 4, &query_payload);
        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let response_word1 = memory
            .read_u32(tls + 4)
            .expect("response header word1 should exist");
        assert_eq!(response_word1 & (1 << 31), 0, "no handle should be copied");
        assert_eq!(memory.read_u32(tls + 32), Some(NVDRV_RESULT_INVALID_STATE));
    }

    #[test]
    fn nvdrv_open_accepts_device_path_from_ptr_buffer_descriptor() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"nvdrv\0\0\0");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let nvdrv_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");

        let path = b"/dev/nvhost-ctrl-gpu\0";
        let path_address = MAIN_PC + 0x2000;
        memory
            .load(path_address, path)
            .expect("path write should succeed");
        write_cmif_request_with_ptr_buffer(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            0,
            &[],
            path_address,
            path.len() as u16,
        );

        state.write_x(0, u64::from(nvdrv_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 36), Some(NVDRV_RESULT_SUCCESS));

        let fd = memory
            .read_u32(tls + 32)
            .expect("nvdrv open should return an fd") as i32;
        assert!(fd >= 0, "nvdrv open should produce a non-negative fd");
    }

    #[test]
    fn applet_common_state_getter_events_follow_message_queue() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        let wait_out_index = MAIN_PC + 0x5a0;
        let wait_handles = MAIN_PC + 0x5b0;
        state.tpidrro_el0 = tls;

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"appletOE");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let applet_handle = memory
            .read_u32(tls + 12)
            .expect("applet session handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, applet_handle),
            Some(ServiceSessionKind::AppletOE)
        );

        // Convert applet session into a domain session.
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IAppletProxyService::OpenApplicationProxy
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 0, &[]);
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let application_proxy_object = memory
            .read_u32(tls + 48)
            .expect("application proxy object id should exist");

        // IApplicationProxy::GetApplicationFunctions
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            20,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let application_functions_object = memory
            .read_u32(tls + 48)
            .expect("application functions object id should exist");

        // IApplicationFunctions::GetGpuErrorDetectedSystemEvent should stay unsignaled.
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_functions_object,
            130,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let gpu_event_special = memory
            .read_u32(tls + 8)
            .expect("special header should exist");
        assert_eq!(
            (gpu_event_special >> 1) & 0xf,
            1,
            "response should copy one event handle"
        );
        let gpu_error_event_handle = memory
            .read_u32(tls + 12)
            .expect("gpu error event handle should exist");

        assert!(memory.write_u32(wait_handles, gpu_error_event_handle));
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_TIMED_OUT);
        assert_eq!(memory.read_u32(wait_out_index), Some(u32::MAX));

        // IApplicationProxy::GetCommonStateGetter
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            0,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let common_state_getter_object = memory
            .read_u32(tls + 48)
            .expect("common state getter object id should exist");

        // ICommonStateGetter::GetEventHandle
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            common_state_getter_object,
            0,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let special = memory
            .read_u32(tls + 8)
            .expect("special header should exist");
        assert_eq!(
            (special >> 1) & 0xf,
            1,
            "response should copy one event handle"
        );
        let event_handle = memory
            .read_u32(tls + 12)
            .expect("common-state event handle should exist");

        assert!(memory.write_u32(wait_handles, event_handle));
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(wait_out_index), Some(0));

        // Receive first seeded applet message.
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            common_state_getter_object,
            1,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
        assert_eq!(memory.read_u32(tls + 48), Some(15));

        // One message remains; event stays signaled.
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            common_state_getter_object,
            1,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
        assert_eq!(memory.read_u32(tls + 48), Some(1));

        // Queue is now empty; event should become unsignaled.
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_TIMED_OUT);
        assert_eq!(memory.read_u32(wait_out_index), Some(u32::MAX));

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            common_state_getter_object,
            1,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_AM_NO_MESSAGES));
    }

    #[test]
    fn applet_domain_commands_from_ryujinx_stubs_return_success() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"appletOE");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let applet_handle = memory
            .read_u32(tls + 12)
            .expect("applet session handle should exist");

        // Convert applet session into a domain session.
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IAppletProxyService::OpenApplicationProxy
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 0, &[]);
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let application_proxy_object = memory
            .read_u32(tls + 48)
            .expect("application proxy object id should exist");

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            0,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let common_state_getter_object = memory
            .read_u32(tls + 48)
            .expect("common state getter object id should exist");

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            1,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let self_controller_object = memory
            .read_u32(tls + 48)
            .expect("self controller object id should exist");

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            2,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let window_controller_object = memory
            .read_u32(tls + 48)
            .expect("window controller object id should exist");

        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_proxy_object,
            20,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let application_functions_object = memory
            .read_u32(tls + 48)
            .expect("application functions object id should exist");

        // ICommonStateGetter::SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            common_state_getter_object,
            900,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // ISelfController::SetFocusHandlingMode(b8, b8, b8)
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            self_controller_object,
            13,
            &[1, 0, 1],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // ISelfController::SetOutOfFocusSuspendingEnabled(b8)
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            self_controller_object,
            16,
            &[1],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // ISelfController::SetIdleTimeDetectionExtension()
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            self_controller_object,
            44,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // ISelfController::SetMediaPlaybackStateForApplication(bool)
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            self_controller_object,
            40,
            &[1],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // IWindowController::AcquireForegroundRights
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            window_controller_object,
            10,
            &[],
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // IApplicationFunctions::SetTerminateResult(nn::Result)
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            application_functions_object,
            22,
            &0x0000_f201u32.to_le_bytes(),
        );
        state.write_x(0, u64::from(applet_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
    }

    #[test]
    fn friends_service_supports_domain_ensure_friend_list_available() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"friend:u");

        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let friend_handle = memory
            .read_u32(tls + 12)
            .expect("response move handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, friend_handle),
            Some(ServiceSessionKind::Friend)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(friend_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IServiceCreator::CreateFriendService() on root domain object id 1.
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 0, &[]);
        state.write_x(0, u64::from(friend_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let friend_service_object_id = memory
            .read_u32(tls + 48)
            .expect("domain response object id should exist");
        assert_eq!(friend_service_object_id, 2);

        // IFriendService::GetCompletionEvent() should return a signaled event handle.
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            friend_service_object_id,
            0,
            &[],
        );
        state.write_x(0, u64::from(friend_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let friend_event_special = memory
            .read_u32(tls + 8)
            .expect("special header should exist");
        assert_eq!((friend_event_special >> 1) & 0xf, 1);
        let friend_completion_event = memory
            .read_u32(tls + 12)
            .expect("friend completion event handle should exist");

        let wait_out_index = MAIN_PC + 0x560;
        let wait_handles = MAIN_PC + 0x568;
        assert!(memory.write_u32(wait_handles, friend_completion_event));
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(wait_out_index), Some(0));

        // IFriendService::CheckFriendListAvailability() -> bool(true)
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            friend_service_object_id,
            10_120,
            &[],
        );
        state.write_x(0, u64::from(friend_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 48), Some(1));

        // IFriendService::EnsureFriendListAvailable() should succeed with empty payload.
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            friend_service_object_id,
            10_121,
            &[],
        );
        state.write_x(0, u64::from(friend_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
    }

    #[test]
    fn pctl_and_aoc_domain_bootstrap_commands_return_success() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;

        // Resolve pctl and run the factory -> service initialize sequence on a domain session.
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"pctl");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let pctl_handle = memory
            .read_u32(tls + 12)
            .expect("pctl session handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, pctl_handle),
            Some(ServiceSessionKind::Pctl)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(pctl_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IParentalControlServiceFactory::CreateService
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 0, &[]);
        state.write_x(0, u64::from(pctl_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let pctl_service_object = memory
            .read_u32(tls + 48)
            .expect("pctl service object id should exist");

        // IParentalControlService::Initialize
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            pctl_service_object,
            1,
            &[],
        );
        state.write_x(0, u64::from(pctl_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // Resolve aoc:u and query count/list-change event on a domain session.
        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"aoc:u");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let aoc_handle = memory
            .read_u32(tls + 12)
            .expect("aoc session handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, aoc_handle),
            Some(ServiceSessionKind::AocU)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(aoc_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IAddOnContentManager::CountAddOnContent(pid) -> 0
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 2, &[]);
        state.write_x(0, u64::from(aoc_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 48), Some(0));

        // IAddOnContentManager::GetAddOnContentListChangedEvent
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 8, &[]);
        state.write_x(0, u64::from(aoc_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let event_special = memory
            .read_u32(tls + 8)
            .expect("special header should exist");
        assert_eq!((event_special >> 1) & 0xf, 1);
        let aoc_event = memory
            .read_u32(tls + 12)
            .expect("aoc event handle should exist");

        let wait_out_index = MAIN_PC + 0x560;
        let wait_handles = MAIN_PC + 0x568;
        assert!(memory.write_u32(wait_handles, aoc_event));
        state.write_x(0, wait_out_index);
        state.write_x(1, wait_handles);
        state.write_x(2, 1);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_TIMED_OUT);
        assert_eq!(memory.read_u32(wait_out_index), Some(u32::MAX));
    }

    #[test]
    fn fsp_domain_bootstrap_commands_return_success() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x460;
        let name_ptr = MAIN_PC + 0x470;
        memory
            .load(name_ptr, b"sm:\0")
            .expect("service name write should succeed");
        state.write_x(0, out_ptr);
        state.write_x(1, name_ptr);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1f, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let sm_handle = state.read_x(1) as u32;

        let tls = MAIN_PC + 0x540;
        state.tpidrro_el0 = tls;

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, b"fsp-srv\0");
        state.write_x(0, u64::from(sm_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        let fsp_handle = memory
            .read_u32(tls + 12)
            .expect("fsp session handle should exist");
        assert_eq!(
            kernel.service_session_for_handle(MAIN_THREAD_HANDLE, fsp_handle),
            Some(ServiceSessionKind::FspSrv)
        );

        write_cmif_request_no_special(&memory, tls, HIPC_TYPE_CMIF_CONTROL, 0, &[]);
        state.write_x(0, u64::from(fsp_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // IFileSystemProxy::SetCurrentProcess([pid]).
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 1, &[]);
        state.write_x(0, u64::from(fsp_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
        assert_eq!(memory.read_u32(tls + 16), Some(0));

        // IFileSystemProxy::SetGlobalAccessLogMode(mode).
        let mode = 3u32;
        write_cmif_domain_request_no_special(
            &memory,
            tls,
            HIPC_TYPE_CMIF_REQUEST,
            1,
            1004,
            &mode.to_le_bytes(),
        );
        state.write_x(0, u64::from(fsp_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));

        // IFileSystemProxy::GetGlobalAccessLogMode() -> mode.
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 1005, &[]);
        state.write_x(0, u64::from(fsp_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
        assert_eq!(memory.read_u32(tls + 48), Some(mode));

        // IFileSystemProxy::OpenSdCardFileSystem() -> IFileSystem domain object.
        write_cmif_domain_request_no_special(&memory, tls, HIPC_TYPE_CMIF_REQUEST, 1, 18, &[]);
        state.write_x(0, u64::from(fsp_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x21, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(tls + 40), Some(RESULT_SUCCESS as u32));
        let filesystem_object_id = memory
            .read_u32(tls + 48)
            .expect("fsp domain response object id should exist");
        assert_eq!(filesystem_object_id, 2);
    }

    #[test]
    fn create_and_start_process_makes_worker_task_runnable() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(0, THREAD_PC);
        state.write_x(1, THREAD_ARG);
        state.write_x(2, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x79, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let process_handle = state.read_x(1) as u32;
        let thread_handle = state.read_x(2) as u32;
        assert_eq!(
            kernel.process_main_thread_handle(MAIN_THREAD_HANDLE, process_handle),
            Some(MAIN_THREAD_HANDLE)
        );
        assert_eq!(
            kernel.process_bootstrap_for_handle(MAIN_THREAD_HANDLE, process_handle),
            Some((THREAD_PC, THREAD_ARG, THREAD_SP & !0xf))
        );

        state.write_x(0, u64::from(process_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x7a, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        let task = kernel
            .take_runnable_worker_thread()
            .expect("process main thread should be runnable");
        let target_process = kernel
            .resolve_process_handle(MAIN_THREAD_HANDLE, process_handle)
            .expect("process handle should resolve in creator table");
        assert_eq!(task.process_handle, target_process);
        assert_ne!(task.thread_handle, thread_handle);
        assert_eq!(task.state.pc, THREAD_PC);
        assert_eq!(task.state.sp, THREAD_SP & !0xf);
        assert_eq!(task.state.read_x(0), THREAD_ARG);
        assert_eq!(task.state.read_x(1), u64::from(MAIN_THREAD_HANDLE));
    }

    #[test]
    fn address_arbiter_is_scoped_per_process() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let address = 0x4000_0000;
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(0, THREAD_PC);
        state.write_x(1, THREAD_ARG);
        state.write_x(2, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x79, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let process_handle = state.read_x(1) as u32;
        let _worker_handle = state.read_x(2) as u32;

        state.write_x(0, u64::from(process_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x7a, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker process thread should be runnable");
        let mut worker_state = worker_task.state;
        let worker_thread_handle = worker_task.thread_handle;
        let _ = worker_task.memory.write_u32(address, 0);
        worker_state.write_x(0, address);
        worker_state.write_x(1, 2);
        worker_state.write_x(2, 0);
        worker_state.write_x(3, 1_000_000);
        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_thread_handle,
                0x34,
                &mut worker_state,
                &worker_task.memory,
            ),
            SyscallDisposition::Continue
        );

        assert_eq!(
            kernel.waiter_count_for_address(worker_thread_handle, address),
            1
        );

        state.write_x(0, address);
        state.write_x(1, 0);
        state.write_x(2, 0);
        state.write_x(3, 1);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x35, &mut state, &memory),
            SyscallDisposition::Continue
        );

        assert!(kernel.take_runnable_worker_thread().is_none());
    }

    #[test]
    fn wait_synchronization_selects_signaled_event() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let event_handle = kernel
            .create_event_handle(MAIN_THREAD_HANDLE)
            .expect("event handle should be allocated");
        assert!(kernel.set_event_signaled_for_handle(MAIN_THREAD_HANDLE, event_handle, true,));

        let out_index = MAIN_PC + 0x400;
        let handles = MAIN_PC + 0x500;
        assert!(memory.write_u32(handles, event_handle));

        state.write_x(0, out_index);
        state.write_x(1, handles);
        state.write_x(2, 1);
        state.write_x(3, 0);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(state.read_x(1), 0);
        assert_eq!(memory.read_u32(out_index), Some(0));
    }

    #[test]
    fn wait_synchronization_times_out_for_unsignaled_thread() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let worker_handle = state.read_x(1) as u32;

        let out_index = MAIN_PC + 0x600;
        let handles = MAIN_PC + 0x700;
        assert!(memory.write_u32(handles, worker_handle));

        state.write_x(0, out_index);
        state.write_x(1, handles);
        state.write_x(2, 1);
        state.write_x(3, 0);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x18, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_TIMED_OUT);
        assert_eq!(state.read_x(1), u64::MAX);
        assert_eq!(memory.read_u32(out_index), Some(u32::MAX));
    }

    #[test]
    fn wait_synchronization_worker_wakes_when_event_signals() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let worker_handle = state.read_x(1) as u32;

        state.write_x(0, u64::from(worker_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x09, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let mut worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker should be runnable after start");
        let event_handle = kernel
            .create_event_handle(MAIN_THREAD_HANDLE)
            .expect("event handle should be allocated");

        let out_index = MAIN_PC + 0x840;
        let handles = MAIN_PC + 0x844;
        assert!(worker_task.memory.write_u32(handles, event_handle));

        worker_task.state.write_x(0, out_index);
        worker_task.state.write_x(1, handles);
        worker_task.state.write_x(2, 1);
        worker_task.state.write_x(3, u64::MAX);
        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_handle,
                0x18,
                &mut worker_task.state,
                &worker_task.memory,
            ),
            SyscallDisposition::Continue
        );

        assert!(kernel.take_runnable_worker_thread().is_none());
        assert!(kernel.set_event_signaled_for_handle(MAIN_THREAD_HANDLE, event_handle, true));

        let woken_task = kernel
            .take_runnable_worker_thread()
            .expect("event signal should wake synchronization waiter");
        assert_eq!(woken_task.thread_handle, worker_handle);
        assert_eq!(woken_task.state.read_x(0), RESULT_SUCCESS);
        assert_eq!(woken_task.state.read_x(1), 0);
        assert_eq!(woken_task.memory.read_u32(out_index), Some(0));
    }

    #[test]
    fn output_debug_string_reads_guest_buffer() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let text = b"hello from guest";
        let buffer = MAIN_PC + 0x300;
        memory
            .load(buffer, text)
            .expect("guest buffer write should succeed");

        state.write_x(0, buffer);
        state.write_x(1, text.len() as u64);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x27, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
    }

    #[test]
    fn output_debug_string_rejects_invalid_buffer() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(0, MAIN_PC + 0x20_0000);
        state.write_x(1, 4);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x27, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_MEMORY_STATE);
    }

    #[test]
    fn get_info_random_entropy_writes_output_value() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x380;
        let expected = kernel
            .random_entropy_word(MAIN_THREAD_HANDLE, 2)
            .expect("entropy should exist for current process");

        state.write_x(0, out_ptr);
        state.write_x(1, 11);
        state.write_x(2, 0);
        state.write_x(3, 2);

        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(expected));

        state.write_x(0, out_ptr);
        state.write_x(1, 11);
        state.write_x(2, 0);
        state.write_x(3, 9);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_INVALID_COMBINATION);

        state.write_x(0, out_ptr);
        state.write_x(1, 2);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(MAIN_PC));

        state.write_x(0, out_ptr);
        state.write_x(1, 12);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(MAIN_PC));

        state.write_x(0, out_ptr);
        state.write_x(1, 13);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0x7ff8_0000_00));

        state.write_x(0, out_ptr);
        state.write_x(1, 14);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(MAIN_PC));

        state.write_x(0, out_ptr);
        state.write_x(1, 15);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0x20_0000));

        state.write_x(0, out_ptr);
        state.write_x(1, 22);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0x20_0000));

        state.write_x(0, out_ptr);
        state.write_x(1, 21);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0xc000_0000));
    }

    #[test]
    fn get_info_stubbed_ids_return_consistent_values() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let out_ptr = MAIN_PC + 0x3c0;

        // InfoType::ResourceLimit (9) — no resource limit object in this kernel yet.
        state.write_x(0, out_ptr);
        state.write_x(1, 9);
        state.write_x(2, 0);
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0));

        // InfoType::IsApplication (23).
        state.write_x(0, out_ptr);
        state.write_x(1, 23);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(1));

        // InfoType::FreeThreadCount (24).
        state.write_x(0, out_ptr);
        state.write_x(1, 24);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0));

        // InfoType::ThreadTickCount (25) with sub_id=-1.
        state.write_x(0, out_ptr);
        state.write_x(1, 25);
        state.write_x(2, u64::from(SELF_THREAD_HANDLE));
        state.write_x(3, u64::MAX);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(0));

        // InfoType::IsSvcPermitted (26).
        state.write_x(0, out_ptr);
        state.write_x(1, 26);
        state.write_x(2, u64::from(SELF_PROCESS_HANDLE));
        state.write_x(3, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x29, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u64(out_ptr), Some(1));
    }

    #[test]
    fn set_heap_size_over_capacity_returns_out_of_memory() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);
        // Configure a small heap region: 64 MiB total, with 2 MiB image + 2 MiB stack.
        // capacity = 64 MiB - 4 MiB = 60 MiB = 0x3C0_0000.
        let heap_start = 0x9000_0000u64;
        let heap_end = heap_start + 0x400_0000; // 64 MiB region
        kernel.configure_main_process_heap(heap_start, heap_end, 0x20_0000, 0x20_0000);

        state.write_x(0, MAIN_PC + 0x500);
        // Request 64 MiB — exceeds capacity of 60 MiB.
        state.write_x(1, 0x400_0000);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x01, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_OUT_OF_MEMORY);
    }

    #[test]
    fn set_heap_size_maps_requested_size_without_clamp() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);
        // Configure a heap region with plenty of capacity.
        let heap_start = 0x9000_0000u64;
        let heap_end = heap_start + 0x1_8000_0000; // 6 GiB
        kernel.configure_main_process_heap(heap_start, heap_end, 0x20_0000, 0x20_0000);

        let requested_size = 0x0400_0000u64;
        state.write_x(0, MAIN_PC + 0x500);
        state.write_x(1, requested_size);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x01, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);

        // The heap address is returned in x1 (SVC convention).
        let heap_base = state.read_x(1);
        assert_eq!(heap_base, heap_start);

        let mapped_tail = heap_start + requested_size - 0x1000;
        let mapped_info = memory.query_memory(mapped_tail);
        assert_eq!(mapped_info.state, 0x5);

        let first_unmapped = memory.query_memory(heap_start + requested_size);
        assert_eq!(first_unmapped.state, 0);
    }

    #[test]
    fn arbitrate_lock_blocks_and_unlock_passes_mutex_to_waiter() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let mutex_address = MAIN_PC + 0x1000;
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let requester_handle = state.read_x(1) as u32;

        state.write_x(0, u64::from(requester_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x09, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be runnable");
        let mut worker_state = worker_task.state;
        let worker_thread_handle = worker_task.thread_handle;

        let owner_tag = MAIN_THREAD_HANDLE | HANDLE_WAIT_MASK;
        let _ = memory.write_u32(mutex_address, owner_tag);

        worker_state.write_x(0, u64::from(MAIN_THREAD_HANDLE));
        worker_state.write_x(1, mutex_address);
        worker_state.write_x(2, u64::from(requester_handle));
        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_thread_handle,
                0x1a,
                &mut worker_state,
                &memory,
            ),
            SyscallDisposition::Continue
        );
        assert_eq!(worker_state.read_x(0), RESULT_SUCCESS);
        assert_eq!(
            kernel.waiter_count_for_address(worker_thread_handle, mutex_address),
            1
        );
        assert!(kernel.take_runnable_worker_thread().is_none());

        state.write_x(0, mutex_address);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1b, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(mutex_address), Some(requester_handle));

        let resumed = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be resumed");
        assert_eq!(resumed.thread_handle, worker_thread_handle);
        assert_eq!(resumed.state.read_x(0), RESULT_SUCCESS);
    }

    #[test]
    fn wait_process_wide_key_atomic_and_signal_reacquire_mutex() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        let mutex_address = MAIN_PC + 0x2000;
        let cv_key = MAIN_PC + 0x3000;
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(1, THREAD_PC);
        state.write_x(2, THREAD_ARG);
        state.write_x(3, THREAD_SP);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x08, &mut state, &memory),
            SyscallDisposition::Continue
        );
        let requester_handle = state.read_x(1) as u32;

        state.write_x(0, u64::from(requester_handle));
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x09, &mut state, &memory),
            SyscallDisposition::Continue
        );

        let worker_task = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be runnable");
        let mut worker_state = worker_task.state;
        let worker_thread_handle = worker_task.thread_handle;

        let _ = memory.write_u32(mutex_address, requester_handle);

        worker_state.write_x(0, mutex_address);
        worker_state.write_x(1, cv_key);
        worker_state.write_x(2, u64::from(requester_handle));
        worker_state.write_x(3, 1_000_000);
        assert_eq!(
            dispatch_supervisor_call(
                &mut kernel,
                worker_thread_handle,
                0x1c,
                &mut worker_state,
                &memory,
            ),
            SyscallDisposition::Continue
        );
        assert_eq!(worker_state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(mutex_address), Some(0));
        assert_eq!(memory.read_u32(cv_key), Some(1));
        assert_eq!(
            kernel.waiter_count_for_address(worker_thread_handle, cv_key),
            1
        );
        assert!(kernel.take_runnable_worker_thread().is_none());

        state.write_x(0, cv_key);
        state.write_x(1, 1);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x1d, &mut state, &memory),
            SyscallDisposition::Continue
        );
        assert_eq!(state.read_x(0), RESULT_SUCCESS);
        assert_eq!(memory.read_u32(mutex_address), Some(requester_handle));
        assert_eq!(memory.read_u32(cv_key), Some(0));

        let resumed = kernel
            .take_runnable_worker_thread()
            .expect("worker thread should be resumed");
        assert_eq!(resumed.thread_handle, worker_thread_handle);
        assert_eq!(resumed.state.read_x(0), RESULT_SUCCESS);
    }

    #[test]
    fn break_with_fatal_reason_exits_main_process() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        state.write_x(0, 0);
        assert_eq!(
            dispatch_supervisor_call(&mut kernel, MAIN_THREAD_HANDLE, 0x26, &mut state, &memory),
            SyscallDisposition::ThreadExit
        );
    }

    #[test]
    fn unknown_svc_interrupts_with_unknown_reason() {
        let mut kernel = KernelState::default();
        let mut state = CpuState::default();
        let memory = SharedGuestMemory::new(MAIN_PC, 0x20_0000);
        state.pc = MAIN_PC;
        kernel.reset_threads(&state, &memory);

        let step =
            kernel.handle_supervisor_call(MAIN_PC, MAIN_THREAD_HANDLE, 0x1234, &mut state, &memory);

        match step {
            StepResult::Trap {
                reason: ExitReason::UnknownSupervisorCall(imm),
                ..
            } => assert_eq!(imm, 0x1234),
            other => panic!("expected unknown-svc trap, got {other:?}"),
        }
    }
}
