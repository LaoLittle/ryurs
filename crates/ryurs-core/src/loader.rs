use std::path::Path;

use aes::Aes128;
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecryptMut, KeyInit};
use anyhow::{Context, Result, anyhow, bail};
use ctr::Ctr128BE;
use ctr::cipher::{KeyIvInit, StreamCipher};
use lz4_flex::block::decompress as lz4_decompress;
use object::{Architecture, BinaryFormat, Object, ObjectSegment};
use xts_mode::Xts128;

use crate::keys::KeySet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramKind {
    Raw,
    NsoRelocatable,
    NroRelocatable,
    Elf64Aarch64Relocated,
    NcaExeFsBundle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcaContentType {
    Program,
    Meta,
    Control,
    Manual,
    Data,
    PublicData,
    Unknown(u8),
}

impl NcaContentType {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Program,
            1 => Self::Meta,
            2 => Self::Control,
            3 => Self::Manual,
            4 => Self::Data,
            5 => Self::PublicData,
            value => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcaHeaderInfo {
    pub content_type: NcaContentType,
    pub has_rights_id: bool,
    pub key_generation: u8,
    pub key_area_index: u8,
}

#[derive(Debug, Clone)]
pub struct ProgramSegment {
    pub address: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ProgramImage {
    pub kind: ProgramKind,
    pub entry: u64,
    pub relocation_bias: u64,
    pub segments: Vec<ProgramSegment>,
    pub runtime_hints: ProgramRuntimeHints,
    /// Decrypted RomFS data extracted from the NCA, if present.
    /// Used by `FspStorage::Read` (IFileSystemProxy::OpenDataStorageByCurrentProcess).
    pub romfs_data: Option<Vec<u8>>,
    /// Per-module (text_base, relocation_bias) pairs for ELF relocation.
    /// Populated when loading NCA/NSP bundles with multiple NSOs.
    pub module_bases: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProgramRuntimeHints {
    pub process_flags: Option<u8>,
    pub main_thread_stack_size: Option<u64>,
}

impl ProgramImage {
    pub fn raw(load_address: u64, bytes: &[u8]) -> Self {
        Self {
            kind: ProgramKind::Raw,
            entry: load_address,
            relocation_bias: 0,
            segments: vec![ProgramSegment {
                address: load_address,
                bytes: bytes.to_vec(),
            }],
            runtime_hints: ProgramRuntimeHints::default(),
            romfs_data: None,
            module_bases: Vec::new(),
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.bytes.len())
            .sum()
    }
}

pub fn read_program(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path)
        .with_context(|| format!("failed to read program image from {}", path.display()))
}

pub fn parse_program_image(bytes: &[u8], fallback_load_address: u64) -> Result<ProgramImage> {
    if let Some(image) = parse_nso_image(bytes, fallback_load_address)? {
        return Ok(image);
    }

    if let Some(image) = parse_nro_image(bytes, fallback_load_address)? {
        return Ok(image);
    }

    if looks_like_nca_container(bytes) {
        bail!(
            "NCA container detected (magic NCA3 at 0x200). Use parse_nca_program_image (or Emulator::load_program_nca) with keys to extract ExeFS programs before loading"
        );
    }

    if looks_like_nsp_container(bytes) {
        bail!(
            "NSP container detected (PFS0). Use parse_nsp_program_image (or Emulator::load_program_nsp) with keys to extract Program NCAs before loading"
        );
    }

    let has_elf_magic = bytes.starts_with(b"\x7fELF");
    let file = match object::File::parse(bytes) {
        Ok(file) => file,
        Err(err) => {
            if has_elf_magic {
                return Err(anyhow!("failed to parse ELF image: {err}"));
            }

            return Ok(ProgramImage::raw(fallback_load_address, bytes));
        }
    };

    if file.format() != BinaryFormat::Elf {
        return Ok(ProgramImage::raw(fallback_load_address, bytes));
    }

    if file.architecture() != Architecture::Aarch64 {
        bail!(
            "unsupported ELF architecture {:?}; expected AArch64",
            file.architecture()
        );
    }

    let mut raw_segments = Vec::new();
    for segment in file.segments() {
        let data = segment
            .data()
            .context("failed to access ELF segment bytes")?;
        if data.is_empty() {
            continue;
        }

        raw_segments.push((segment.address(), data.to_vec()));
    }

    if raw_segments.is_empty() {
        bail!("ELF image has no loadable segment data");
    }

    let min_address = raw_segments
        .iter()
        .map(|(address, _)| *address)
        .min()
        .expect("segments are non-empty");

    let relocation_bias = fallback_load_address
        .checked_sub(min_address)
        .ok_or_else(|| anyhow!("ELF minimum virtual address is above fallback load address"))?;

    let entry = file
        .entry()
        .checked_add(relocation_bias)
        .ok_or_else(|| anyhow!("relocated entry address overflowed"))?;

    let mut segments = Vec::with_capacity(raw_segments.len());
    for (address, data) in raw_segments {
        let relocated = address
            .checked_add(relocation_bias)
            .ok_or_else(|| anyhow!("relocated segment address overflowed"))?;

        segments.push(ProgramSegment {
            address: relocated,
            bytes: data,
        });
    }

    Ok(ProgramImage {
        kind: ProgramKind::Elf64Aarch64Relocated,
        entry,
        relocation_bias,
        segments,
        runtime_hints: ProgramRuntimeHints::default(),
        romfs_data: None,
        module_bases: vec![(entry, relocation_bias)],
    })
}

pub fn parse_nsp_program_image(
    bytes: &[u8],
    fallback_load_address: u64,
    key_set: &KeySet,
) -> Result<ProgramImage> {
    #[derive(Debug)]
    struct NspNcaCandidate<'a> {
        name: String,
        bytes: &'a [u8],
        content_type: Option<NcaContentType>,
    }

    let entries = parse_pfs0_entries(bytes).context("failed to parse NSP PFS0 container")?;
    let mut merged_keys = key_set.clone();
    import_title_keys_from_nsp_entries(&mut merged_keys, &entries);

    let mut candidates = Vec::new();
    for entry in entries {
        if !entry.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }

        let content_type = inspect_nca_header(entry.bytes, &merged_keys)
            .map(|header| header.content_type)
            .ok();
        candidates.push(NspNcaCandidate {
            name: entry.name.to_string(),
            bytes: entry.bytes,
            content_type,
        });
    }

    candidates.sort_by(|a, b| {
        nca_content_type_rank(a.content_type)
            .cmp(&nca_content_type_rank(b.content_type))
            .then_with(|| a.name.cmp(&b.name))
    });

    if candidates.is_empty() {
        bail!("NSP container does not contain any .nca entries");
    }

    let mut failures = Vec::new();
    for candidate in candidates {
        match parse_nca_program_image(candidate.bytes, fallback_load_address, &merged_keys) {
            Ok(image) => return Ok(image),
            Err(err) => {
                let kind = candidate
                    .content_type
                    .map(|value| format!("{:?}", value))
                    .unwrap_or_else(|| "unknown".to_string());
                failures.push(format!("{} ({kind}) -> {err}", candidate.name));
            }
        }
    }

    let mut message = String::from("no executable Program NCA found in NSP container");
    for reason in failures.iter().take(3) {
        message.push_str("\n- ");
        message.push_str(reason);
    }
    if failures.len() > 3 {
        message.push_str("\n- ...");
    }

    bail!(message)
}

pub fn parse_nca_program_image(
    bytes: &[u8],
    fallback_load_address: u64,
    key_set: &KeySet,
) -> Result<ProgramImage> {
    let extraction = extract_exefs_programs_from_nca(bytes, key_set)?;

    let mut entry = None;
    let mut next_load_address = fallback_load_address;
    let mut segments = Vec::new();
    let mut module_bases: Vec<(u64, u64)> = Vec::new();

    for program in extraction.programs {
        let image = parse_program_image(&program.bytes, next_load_address).with_context(|| {
            format!(
                "failed to parse ExeFS program '{}' extracted from NCA",
                program.name
            )
        })?;

        if entry.is_none() {
            entry = Some(image.entry);
        }

        // Collect per-module bases for relocation processing.
        if !image.module_bases.is_empty() {
            module_bases.extend_from_slice(&image.module_bases);
        } else if image.relocation_bias != 0 {
            module_bases.push((image.entry, image.relocation_bias));
        }

        next_load_address = advance_load_base_from_image(&image, next_load_address)?;
        segments.extend(image.segments);
    }

    let entry = entry.ok_or_else(|| anyhow!("ExeFS did not yield any loadable program"))?;

    // Also try to extract the RomFS from the same NCA for FspStorage support.
    let romfs_data = match extract_romfs_from_nca(bytes, key_set) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("[loader] warning: failed to extract RomFS from NCA: {err}");
            None
        }
    };

    Ok(ProgramImage {
        kind: ProgramKind::NcaExeFsBundle,
        entry,
        relocation_bias: 0,
        segments,
        runtime_hints: extraction.runtime_hints,
        romfs_data,
        module_bases,
    })
}

pub fn list_nca_exefs_program_names(bytes: &[u8], key_set: &KeySet) -> Result<Vec<String>> {
    let extraction = extract_exefs_programs_from_nca(bytes, key_set)?;
    Ok(extraction
        .programs
        .into_iter()
        .map(|program| program.name)
        .collect())
}

fn extract_exefs_programs_from_nca(bytes: &[u8], key_set: &KeySet) -> Result<ExeFsExtraction> {
    let header = decrypt_nca_header(bytes, key_set)?;
    if &header[0x200..0x204] != b"NCA3" {
        bail!("unsupported NCA version; only NCA3 is currently supported");
    }

    let key_generation_old = header[0x206];
    let key_area_index = header[0x207];
    let key_generation_new = header[0x220];
    let rights_id = &header[0x230..0x240];
    let has_rights_id = rights_id.iter().any(|byte| *byte != 0);

    let mut crypto_type = key_generation_old.max(key_generation_new);
    if crypto_type > 0 {
        crypto_type -= 1;
    }

    let section_key = if has_rights_id {
        derive_nca_title_key(key_set, crypto_type, rights_id)?
    } else {
        derive_nca_standard_section_key(key_set, crypto_type, key_area_index, &header)?
    };

    let sections = parse_nca_sections(&header, bytes.len())?;
    for section in sections {
        if section.partition_type != 1 || section.fs_type != 2 {
            continue;
        }

        let pfs0_offset = read_u64_le(&header, section.fs_header_offset + 0x40)? as usize;
        let pfs0_size = read_u64_le(&header, section.fs_header_offset + 0x48)? as usize;
        if pfs0_size == 0 {
            continue;
        }

        let pfs0 = read_nca_section_bytes(bytes, &section, pfs0_offset, pfs0_size, &section_key)?;
        let programs = extract_exefs_programs(&pfs0)?;
        if programs.is_empty() {
            continue;
        }

        let runtime_hints = extract_exefs_runtime_hints(&pfs0)?;

        return Ok(ExeFsExtraction {
            programs,
            runtime_hints,
        });
    }

    bail!("NCA parsed successfully, but no ExeFS program set was found")
}

/// Extracts the decrypted RomFS data from an NCA.
///
/// Finds the RomFS section (partition_type=0, fs_type=3 / HierarchicalIntegrity),
/// parses the IVFC header to locate the data level, and decrypts it.
fn extract_romfs_from_nca(bytes: &[u8], key_set: &KeySet) -> Result<Option<Vec<u8>>> {
    let header = decrypt_nca_header(bytes, key_set)?;
    if &header[0x200..0x204] != b"NCA3" {
        bail!("unsupported NCA version for RomFS extraction; only NCA3 is supported");
    }

    let key_generation_old = header[0x206];
    let key_area_index = header[0x207];
    let key_generation_new = header[0x220];
    let rights_id = &header[0x230..0x240];
    let has_rights_id = rights_id.iter().any(|byte| *byte != 0);

    let mut crypto_type = key_generation_old.max(key_generation_new);
    if crypto_type > 0 {
        crypto_type -= 1;
    }

    let section_key = if has_rights_id {
        derive_nca_title_key(key_set, crypto_type, rights_id)?
    } else {
        derive_nca_standard_section_key(key_set, crypto_type, key_area_index, &header)?
    };

    let sections = parse_nca_sections(&header, bytes.len())?;
    for section in sections {
        // RomFS: partition_type=0 (RomFS), fs_type=3 (HierarchicalIntegrity / IVFC)
        if section.partition_type != 0 || section.fs_type != 3 {
            continue;
        }

        let ivfc_base = section.fs_header_offset + 0x08;
        let magic = header
            .get(ivfc_base..ivfc_base + 4)
            .ok_or_else(|| anyhow!("IVFC header is out of range in fs header"))?;
        if magic != b"IVFC" {
            eprintln!(
                "[loader] warning: RomFS section IVFC magic mismatch: expected IVFC, got {:?}",
                magic
            );
            continue;
        }

        let num_levels = read_u32_le(&header, ivfc_base + 0x0C)? as usize;
        if num_levels < 2 {
            eprintln!("[loader] warning: RomFS IVFC has too few levels: {num_levels}");
            continue;
        }

        // The level_headers array has (num_levels - 1) entries starting at ivfc_base + 0x10.
        // Each entry is 0x18 bytes: offset(u64), size(u64), block_size_log2(u32), reserved(u32).
        // The data level is the LAST entry (index num_levels - 2).
        let data_level_index = num_levels - 2;
        let data_level_hdr = ivfc_base + 0x10 + data_level_index * 0x18;

        let data_offset = read_u64_le(&header, data_level_hdr)? as usize;
        let data_size = read_u64_le(&header, data_level_hdr + 0x08)? as usize;

        if data_size == 0 {
            eprintln!("[loader] warning: RomFS IVFC data level has zero size");
            continue;
        }

        eprintln!(
            "[loader] RomFS: IVFC data level at section-relative offset {data_offset:#x}, size {data_size:#x} ({data_size} bytes)"
        );

        let romfs_data =
            read_nca_section_bytes(bytes, &section, data_offset, data_size, &section_key)?;
        return Ok(Some(romfs_data));
    }

    Ok(None)
}

fn advance_load_base_from_image(image: &ProgramImage, current_base: u64) -> Result<u64> {
    let mut end = current_base;
    for segment in &image.segments {
        let segment_end = segment
            .address
            .checked_add(segment.bytes.len() as u64)
            .ok_or_else(|| anyhow!("program segment end address overflowed"))?;
        end = end.max(segment_end);
    }

    align_up_u64(end, 0x1000)
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("alignment must be a non-zero power of two");
    }

    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| anyhow!("alignment overflowed"))
}

fn nca_content_type_rank(content_type: Option<NcaContentType>) -> u8 {
    match content_type {
        Some(NcaContentType::Program) => 0,
        Some(_) => 1,
        None => 2,
    }
}

fn import_title_keys_from_nsp_entries(key_set: &mut KeySet, entries: &[Pfs0Entry<'_>]) {
    for entry in entries {
        if !entry.name.to_ascii_lowercase().ends_with(".tik") {
            continue;
        }

        if let Some((rights_id_hex, encrypted_title_key)) = parse_ticket_title_key(entry.bytes) {
            key_set
                .title
                .entry(rights_id_hex)
                .or_insert_with(|| encrypted_title_key.to_vec());
        }
    }
}

fn parse_ticket_title_key(ticket_bytes: &[u8]) -> Option<(String, [u8; 16])> {
    const TITLE_KEY_OFFSET: usize = 0x180;
    const RIGHTS_ID_OFFSET: usize = 0x2a0;

    let title_key_bytes = ticket_bytes.get(TITLE_KEY_OFFSET..TITLE_KEY_OFFSET + 16)?;
    let rights_id_bytes = ticket_bytes.get(RIGHTS_ID_OFFSET..RIGHTS_ID_OFFSET + 16)?;

    if rights_id_bytes.iter().all(|byte| *byte == 0) {
        return None;
    }

    let mut title_key = [0u8; 16];
    title_key.copy_from_slice(title_key_bytes);

    Some((bytes_to_lower_hex(rights_id_bytes), title_key))
}

struct ExeFsProgram {
    name: String,
    bytes: Vec<u8>,
}

struct ExeFsExtraction {
    programs: Vec<ExeFsProgram>,
    runtime_hints: ProgramRuntimeHints,
}

fn extract_exefs_runtime_hints(pfs0_bytes: &[u8]) -> Result<ProgramRuntimeHints> {
    let entries = parse_pfs0_entries(pfs0_bytes)?;

    let Some(npdm) = entries.iter().find(|entry| entry.name == "main.npdm") else {
        return Ok(ProgramRuntimeHints::default());
    };

    Ok(parse_npdm_runtime_hints(npdm.bytes).unwrap_or_default())
}

fn parse_npdm_runtime_hints(npdm_bytes: &[u8]) -> Option<ProgramRuntimeHints> {
    if npdm_bytes.len() < 0x20 || !npdm_bytes.starts_with(b"META") {
        return None;
    }

    let process_flags = *npdm_bytes.get(0x0c)?;
    let stack_size = read_u32_le(npdm_bytes, 0x1c).ok()? as u64;

    Some(ProgramRuntimeHints {
        process_flags: Some(process_flags),
        main_thread_stack_size: (stack_size > 0).then_some(stack_size),
    })
}

fn extract_exefs_programs(pfs0_bytes: &[u8]) -> Result<Vec<ExeFsProgram>> {
    const EXEFS_LOAD_ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];

    let entries = parse_pfs0_entries(pfs0_bytes)?;
    let mut programs = Vec::new();

    for expected_name in EXEFS_LOAD_ORDER {
        if let Some(entry) = entries.iter().find(|entry| entry.name == *expected_name) {
            programs.push(ExeFsProgram {
                name: (*expected_name).to_string(),
                bytes: entry.bytes.to_vec(),
            });
        }
    }

    Ok(programs)
}

struct Pfs0Entry<'a> {
    name: &'a str,
    bytes: &'a [u8],
}

pub fn inspect_nca_header(bytes: &[u8], key_set: &KeySet) -> Result<NcaHeaderInfo> {
    let header = decrypt_nca_header(bytes, key_set)?;

    if &header[0x200..0x204] != b"NCA3" {
        bail!("unsupported NCA version; only NCA3 is currently supported");
    }

    let key_generation_old = header[0x206];
    let key_generation_new = header[0x220];
    let mut key_generation = key_generation_old.max(key_generation_new);
    if key_generation > 0 {
        key_generation -= 1;
    }

    let rights_id = &header[0x230..0x240];
    let has_rights_id = rights_id.iter().any(|byte| *byte != 0);

    Ok(NcaHeaderInfo {
        content_type: NcaContentType::from_raw(header[0x205]),
        has_rights_id,
        key_generation,
        key_area_index: header[0x207],
    })
}

#[derive(Debug, Clone, Copy)]
struct NsoSectionDescriptor {
    file_offset: usize,
    memory_offset: usize,
    memory_size: usize,
    file_size: usize,
}

fn parse_nso_image(bytes: &[u8], load_address: u64) -> Result<Option<ProgramImage>> {
    const NSO_HEADER_SIZE: usize = 0x100;
    const NSO_MAGIC: &[u8; 4] = b"NSO0";

    if bytes.len() < NSO_HEADER_SIZE {
        return Ok(None);
    }

    if &bytes[..4] != NSO_MAGIC {
        return Ok(None);
    }

    let flags = read_u32_le(bytes, 0x0c)?;
    let text = read_nso_section_descriptor(bytes, 0x10, 0x60)?;
    let ro = read_nso_section_descriptor(bytes, 0x20, 0x64)?;
    let data = read_nso_section_descriptor(bytes, 0x30, 0x68)?;
    let bss_size = read_u32_le(bytes, 0x3c)? as usize;

    let text_bytes = extract_nso_section(bytes, text, (flags & (1 << 0)) != 0, "text")?;
    let ro_bytes = extract_nso_section(bytes, ro, (flags & (1 << 1)) != 0, "ro")?;
    let data_bytes = extract_nso_section(bytes, data, (flags & (1 << 2)) != 0, "data")?;

    let min_offset = text
        .memory_offset
        .min(ro.memory_offset)
        .min(data.memory_offset) as u64;
    let relocation_bias = load_address
        .checked_sub(min_offset)
        .ok_or_else(|| anyhow!("NSO minimum virtual offset is above fallback load address"))?;

    let entry = relocation_bias
        .checked_add(text.memory_offset as u64)
        .ok_or_else(|| anyhow!("NSO relocated entry address overflowed"))?;

    let mut segments = Vec::new();
    append_nso_segment(
        &mut segments,
        relocation_bias,
        text.memory_offset,
        text_bytes,
    )?;
    append_nso_segment(&mut segments, relocation_bias, ro.memory_offset, ro_bytes)?;
    append_nso_segment(
        &mut segments,
        relocation_bias,
        data.memory_offset,
        data_bytes,
    )?;

    if bss_size > 0 {
        let bss_offset = data
            .memory_offset
            .checked_add(data.memory_size)
            .ok_or_else(|| anyhow!("NSO data+bss offset overflowed"))?;
        let bss_address = relocation_bias
            .checked_add(bss_offset as u64)
            .ok_or_else(|| anyhow!("NSO BSS segment address overflowed"))?;
        segments.push(ProgramSegment {
            address: bss_address,
            bytes: vec![0; bss_size],
        });
    }

    Ok(Some(ProgramImage {
        kind: ProgramKind::NsoRelocatable,
        entry,
        relocation_bias,
        segments,
        runtime_hints: ProgramRuntimeHints::default(),
        romfs_data: None,
        module_bases: vec![(entry, relocation_bias)],
    }))
}

fn append_nso_segment(
    segments: &mut Vec<ProgramSegment>,
    relocation_bias: u64,
    memory_offset: usize,
    bytes: Vec<u8>,
) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }

    let address = relocation_bias
        .checked_add(memory_offset as u64)
        .ok_or_else(|| anyhow!("NSO segment address overflowed"))?;
    segments.push(ProgramSegment { address, bytes });
    Ok(())
}

fn read_nso_section_descriptor(
    bytes: &[u8],
    header_offset: usize,
    file_size_offset: usize,
) -> Result<NsoSectionDescriptor> {
    Ok(NsoSectionDescriptor {
        file_offset: read_u32_le(bytes, header_offset)? as usize,
        memory_offset: read_u32_le(bytes, header_offset + 0x4)? as usize,
        memory_size: read_u32_le(bytes, header_offset + 0x8)? as usize,
        file_size: read_u32_le(bytes, file_size_offset)? as usize,
    })
}

fn extract_nso_section(
    bytes: &[u8],
    section: NsoSectionDescriptor,
    compressed: bool,
    label: &str,
) -> Result<Vec<u8>> {
    if section.memory_size == 0 {
        return Ok(Vec::new());
    }

    let section_file_size = if section.file_size == 0 {
        section.memory_size
    } else {
        section.file_size
    };

    let end = section
        .file_offset
        .checked_add(section_file_size)
        .ok_or_else(|| anyhow!("NSO {label} file range overflowed"))?;
    if end > bytes.len() {
        bail!(
            "NSO {label} file range ({:#x}..{:#x}) exceeds input size {:#x}",
            section.file_offset,
            end,
            bytes.len()
        );
    }

    let source = &bytes[section.file_offset..end];
    if compressed {
        let decompressed = lz4_decompress(source, section.memory_size)
            .with_context(|| format!("failed to decompress NSO {label} section"))?;
        if decompressed.len() != section.memory_size {
            bail!(
                "NSO {label} decompressed size mismatch: expected {}, got {}",
                section.memory_size,
                decompressed.len()
            );
        }

        return Ok(decompressed);
    }

    if source.len() < section.memory_size {
        bail!(
            "NSO {label} section data is too small: expected at least {}, got {}",
            section.memory_size,
            source.len()
        );
    }

    Ok(source[..section.memory_size].to_vec())
}

#[derive(Debug, Clone, Copy)]
struct NcaSectionInfo {
    start_offset: usize,
    end_offset: usize,
    fs_header_offset: usize,
    partition_type: u8,
    fs_type: u8,
    crypt_type: u8,
    section_ctr: [u8; 8],
}

fn decrypt_nca_header(bytes: &[u8], key_set: &KeySet) -> Result<Vec<u8>> {
    const NCA_HEADER_SIZE: usize = 0xc00;

    if bytes.len() < NCA_HEADER_SIZE {
        bail!(
            "input is too small for NCA header: need at least 0x{NCA_HEADER_SIZE:x} bytes, got 0x{:x}",
            bytes.len()
        );
    }

    if matches!(&bytes[0x200..0x204], b"NCA3" | b"NCA2") {
        return Ok(bytes[..NCA_HEADER_SIZE].to_vec());
    }

    let header_key = require_key_with_len(&key_set.prod, "header_key", 32)?;
    let mut decrypted = bytes[..NCA_HEADER_SIZE].to_vec();

    let key_1 = Aes128::new(GenericArray::from_slice(&header_key[..16]));
    let key_2 = Aes128::new(GenericArray::from_slice(&header_key[16..]));
    let xts = Xts128::<Aes128>::new(key_1, key_2);

    xts.decrypt_area(&mut decrypted[..0x400], 0x200, 0, nintendo_tweak);

    match &decrypted[0x200..0x204] {
        b"NCA3" => {
            xts.decrypt_area(&mut decrypted[0x400..], 0x200, 2, nintendo_tweak);
            Ok(decrypted)
        }
        b"NCA2" => bail!("NCA2 headers are not supported yet"),
        _ => {
            bail!("failed to decrypt NCA header with provided header_key (magic is not NCA3)")
        }
    }
}

fn derive_nca_standard_section_key(
    key_set: &KeySet,
    crypto_type: u8,
    key_area_index: u8,
    header: &[u8],
) -> Result<[u8; 16]> {
    let key_area_class = match key_area_index {
        0 => "application",
        1 => "ocean",
        2 => "system",
        _ => {
            bail!(
                "unsupported NCA key area key index {key_area_index}; expected 0(application), 1(ocean), or 2(system)"
            )
        }
    };

    let key_name = format!("key_area_key_{key_area_class}_{crypto_type:02x}");
    let key_area_key = require_key_with_len(&key_set.prod, &key_name, 16)?;

    let encrypted_key_area = &header[0x300..0x340];
    let mut decrypted_key_area = [0u8; 0x40];
    aes128_ecb_decrypt(key_area_key, encrypted_key_area, &mut decrypted_key_area)?;

    let mut section_key = [0u8; 16];
    section_key.copy_from_slice(&decrypted_key_area[0x20..0x30]);
    Ok(section_key)
}

fn derive_nca_title_key(key_set: &KeySet, crypto_type: u8, rights_id: &[u8]) -> Result<[u8; 16]> {
    if rights_id.len() != 16 {
        bail!(
            "invalid rights ID length {}; expected 16 bytes",
            rights_id.len()
        );
    }

    let titlekek_name = format!("titlekek_{crypto_type:02x}");
    let titlekek = require_key_with_len(&key_set.prod, &titlekek_name, 16)?;
    let rights_id_hex = bytes_to_lower_hex(rights_id);

    let encrypted_title_key = lookup_key_value(&key_set.title, &rights_id_hex)
        .ok_or_else(|| anyhow!("missing encrypted title key for rights ID {rights_id_hex}"))?;

    if encrypted_title_key.len() != 16 {
        bail!(
            "title key for rights ID {rights_id_hex} has invalid length {}; expected 16 bytes",
            encrypted_title_key.len()
        );
    }

    let mut decrypted_title_key = [0u8; 16];
    aes128_ecb_decrypt(titlekek, encrypted_title_key, &mut decrypted_title_key)?;
    Ok(decrypted_title_key)
}

fn parse_nca_sections(header: &[u8], input_size: usize) -> Result<Vec<NcaSectionInfo>> {
    let mut sections = Vec::new();

    for index in 0..4usize {
        let entry_offset = 0x240 + index * 0x10;
        let media_start = read_u32_le(header, entry_offset)? as usize;
        let media_end = read_u32_le(header, entry_offset + 0x4)? as usize;

        if media_start == 0 && media_end == 0 {
            continue;
        }

        if media_end <= media_start {
            bail!(
                "NCA section {index} has invalid media range: start={} end={}",
                media_start,
                media_end
            );
        }

        let start_offset = media_start
            .checked_mul(0x200)
            .ok_or_else(|| anyhow!("NCA section {index} start offset overflowed"))?;
        let end_offset = media_end
            .checked_mul(0x200)
            .ok_or_else(|| anyhow!("NCA section {index} end offset overflowed"))?;

        if end_offset > input_size {
            bail!(
                "NCA section {index} range ({start_offset:#x}..{end_offset:#x}) exceeds input size {input_size:#x}"
            );
        }

        let fs_header_offset = 0x400 + index * 0x200;
        let partition_type = *header
            .get(fs_header_offset + 0x2)
            .ok_or_else(|| anyhow!("NCA section {index} fs header is out of range"))?;
        let fs_type = *header
            .get(fs_header_offset + 0x3)
            .ok_or_else(|| anyhow!("NCA section {index} fs header is out of range"))?;
        let crypt_type = *header
            .get(fs_header_offset + 0x4)
            .ok_or_else(|| anyhow!("NCA section {index} fs header is out of range"))?;

        let ctr_slice = header
            .get(fs_header_offset + 0x140..fs_header_offset + 0x148)
            .ok_or_else(|| anyhow!("NCA section {index} ctr metadata is out of range"))?;
        let mut section_ctr = [0u8; 8];
        section_ctr.copy_from_slice(ctr_slice);

        sections.push(NcaSectionInfo {
            start_offset,
            end_offset,
            fs_header_offset,
            partition_type,
            fs_type,
            crypt_type,
            section_ctr,
        });
    }

    Ok(sections)
}

fn read_nca_section_bytes(
    bytes: &[u8],
    section: &NcaSectionInfo,
    section_relative_offset: usize,
    size: usize,
    section_key: &[u8; 16],
) -> Result<Vec<u8>> {
    let section_size = section.end_offset - section.start_offset;
    let end = section_relative_offset
        .checked_add(size)
        .ok_or_else(|| anyhow!("NCA section read range overflowed"))?;
    if end > section_size {
        bail!(
            "NCA section read ({section_relative_offset:#x}..{end:#x}) exceeds section size {section_size:#x}"
        );
    }

    let absolute_start = section
        .start_offset
        .checked_add(section_relative_offset)
        .ok_or_else(|| anyhow!("NCA absolute section start overflowed"))?;
    let absolute_end = absolute_start
        .checked_add(size)
        .ok_or_else(|| anyhow!("NCA absolute section end overflowed"))?;

    let encrypted = bytes
        .get(absolute_start..absolute_end)
        .ok_or_else(|| anyhow!("NCA section read is out of input range"))?;

    match section.crypt_type {
        1 => Ok(encrypted.to_vec()),
        3 => decrypt_nca_ctr_range(bytes, encrypted.len(), absolute_start, section, section_key),
        other => bail!(
            "unsupported NCA section encryption type {other} (currently supports none=1 and aes-ctr=3)"
        ),
    }
}

fn decrypt_nca_ctr_range(
    full_bytes: &[u8],
    size: usize,
    absolute_start: usize,
    section: &NcaSectionInfo,
    section_key: &[u8; 16],
) -> Result<Vec<u8>> {
    let aligned_start = absolute_start & !0x0f;
    let absolute_end = absolute_start
        .checked_add(size)
        .ok_or_else(|| anyhow!("NCA CTR range end overflowed"))?;
    let aligned_end = (absolute_end + 0x0f) & !0x0f;

    let mut encrypted = full_bytes
        .get(aligned_start..aligned_end)
        .ok_or_else(|| anyhow!("NCA CTR aligned range is out of input bounds"))?
        .to_vec();

    let iv = nca_section_ctr_iv(section.section_ctr, aligned_start as u64);
    let mut cipher = Ctr128BE::<Aes128>::new(
        GenericArray::from_slice(section_key),
        GenericArray::from_slice(&iv),
    );
    cipher.apply_keystream(&mut encrypted);

    let start_skip = absolute_start - aligned_start;
    let end = start_skip
        .checked_add(size)
        .ok_or_else(|| anyhow!("NCA CTR result range overflowed"))?;
    Ok(encrypted[start_skip..end].to_vec())
}

fn nca_section_ctr_iv(section_ctr: [u8; 8], absolute_offset: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];

    for index in 0..8 {
        iv[index] = section_ctr[7 - index];
    }

    let mut block_offset = absolute_offset >> 4;
    for index in 0..8 {
        iv[15 - index] = (block_offset & 0xff) as u8;
        block_offset >>= 8;
    }

    iv
}

#[cfg(test)]
fn extract_main_from_pfs0(pfs0_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    let entries = parse_pfs0_entries(pfs0_bytes)?;
    for entry in entries {
        if entry.name == "main" {
            return Ok(Some(entry.bytes.to_vec()));
        }
    }

    Ok(None)
}

fn parse_pfs0_entries<'a>(pfs0_bytes: &'a [u8]) -> Result<Vec<Pfs0Entry<'a>>> {
    if pfs0_bytes.len() < 0x10 {
        bail!("PFS0 payload is too small to contain a header");
    }

    if &pfs0_bytes[0..4] != b"PFS0" {
        bail!("PFS0 payload has invalid magic");
    }

    let entry_count = read_u32_le(pfs0_bytes, 0x4)? as usize;
    let string_table_size = read_u32_le(pfs0_bytes, 0x8)? as usize;

    let entries_size = entry_count
        .checked_mul(0x18)
        .ok_or_else(|| anyhow!("PFS0 entries size overflowed"))?;
    let header_size = 0x10usize
        .checked_add(entries_size)
        .and_then(|size| size.checked_add(string_table_size))
        .ok_or_else(|| anyhow!("PFS0 header size overflowed"))?;
    if header_size > pfs0_bytes.len() {
        bail!(
            "PFS0 header size {header_size:#x} exceeds payload size {:#x}",
            pfs0_bytes.len()
        );
    }

    let strings = &pfs0_bytes[0x10 + entries_size..header_size];
    let mut entries = Vec::with_capacity(entry_count);
    for index in 0..entry_count {
        let entry_offset = 0x10 + index * 0x18;
        let file_offset = read_u64_le(pfs0_bytes, entry_offset)? as usize;
        let file_size = read_u64_le(pfs0_bytes, entry_offset + 0x8)? as usize;
        let string_offset = read_u32_le(pfs0_bytes, entry_offset + 0x10)? as usize;

        let name = pfs0_file_name(strings, string_offset)
            .with_context(|| format!("failed to decode PFS0 file name for entry {index}"))?;

        let file_start = header_size
            .checked_add(file_offset)
            .ok_or_else(|| anyhow!("PFS0 file start overflowed for entry {index}"))?;
        let file_end = file_start
            .checked_add(file_size)
            .ok_or_else(|| anyhow!("PFS0 file end overflowed for entry {index}"))?;

        let file = pfs0_bytes
            .get(file_start..file_end)
            .ok_or_else(|| anyhow!("PFS0 file range is out of bounds for entry {index}"))?;

        entries.push(Pfs0Entry { name, bytes: file });
    }

    Ok(entries)
}

fn pfs0_file_name(strings: &[u8], offset: usize) -> Result<&str> {
    if offset >= strings.len() {
        bail!("PFS0 string table offset {offset:#x} is out of range");
    }

    let tail = &strings[offset..];
    let end = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| anyhow!("PFS0 file name is not NUL-terminated"))?;
    let name_bytes = &tail[..end];
    let name = std::str::from_utf8(name_bytes).context("PFS0 file name is not UTF-8")?;
    Ok(name)
}

fn require_key_with_len<'a>(
    map: &'a std::collections::BTreeMap<String, Vec<u8>>,
    name: &str,
    expected_len: usize,
) -> Result<&'a [u8]> {
    let key = lookup_key_value(map, name).ok_or_else(|| anyhow!("missing key '{name}'"))?;
    if key.len() != expected_len {
        bail!(
            "key '{name}' has invalid length {}; expected {}",
            key.len(),
            expected_len
        );
    }

    Ok(key)
}

fn lookup_key_value<'a>(
    map: &'a std::collections::BTreeMap<String, Vec<u8>>,
    name: &str,
) -> Option<&'a [u8]> {
    map.get(name)
        .or_else(|| map.get(&name.to_ascii_lowercase()))
        .or_else(|| map.get(&name.to_ascii_uppercase()))
        .map(|value| value.as_slice())
}

fn bytes_to_lower_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn aes128_ecb_decrypt(key: &[u8], input: &[u8], output: &mut [u8]) -> Result<()> {
    if input.len() != output.len() {
        bail!(
            "AES-ECB input/output length mismatch: {} vs {}",
            input.len(),
            output.len()
        );
    }
    if input.len() % 16 != 0 {
        bail!(
            "AES-ECB input size {} is not a multiple of 16 bytes",
            input.len()
        );
    }

    let mut cipher = Aes128::new(GenericArray::from_slice(key));
    for (source, destination) in input.chunks(16).zip(output.chunks_mut(16)) {
        let mut block = GenericArray::clone_from_slice(source);
        cipher.decrypt_block_mut(&mut block);
        destination.copy_from_slice(&block);
    }

    Ok(())
}

fn nintendo_tweak(sector_index: u128) -> [u8; 16] {
    sector_index.to_be_bytes()
}

fn looks_like_nca_container(bytes: &[u8]) -> bool {
    const NCA_MAGIC_OFFSET: usize = 0x200;
    const NCA_MAGIC: &[u8; 4] = b"NCA3";

    bytes
        .get(NCA_MAGIC_OFFSET..NCA_MAGIC_OFFSET + NCA_MAGIC.len())
        .map(|slice| slice == NCA_MAGIC)
        .unwrap_or(false)
}

fn looks_like_nsp_container(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == b"PFS0"
}

fn parse_nro_image(bytes: &[u8], load_address: u64) -> Result<Option<ProgramImage>> {
    const NRO_START_SIZE: usize = 0x10;
    const NRO_HEADER_SIZE: usize = 0x80;

    if bytes.len() < NRO_START_SIZE + NRO_HEADER_SIZE {
        return Ok(None);
    }

    let magic_offset = NRO_START_SIZE;
    if &bytes[magic_offset..magic_offset + 4] != b"NRO0" {
        return Ok(None);
    }

    let size = read_u32_le(bytes, magic_offset + 0x08)? as usize;
    if size == 0 {
        bail!("NRO image reports zero executable size");
    }
    if size > bytes.len() {
        bail!(
            "NRO executable size ({size}) exceeds input size ({})",
            bytes.len()
        );
    }

    let text_offset = read_u32_le(bytes, magic_offset + 0x10)? as usize;
    let text_size = read_u32_le(bytes, magic_offset + 0x14)? as usize;
    let ro_offset = read_u32_le(bytes, magic_offset + 0x18)? as usize;
    let ro_size = read_u32_le(bytes, magic_offset + 0x1c)? as usize;
    let data_offset = read_u32_le(bytes, magic_offset + 0x20)? as usize;
    let data_size = read_u32_le(bytes, magic_offset + 0x24)? as usize;
    let bss_size = read_u32_le(bytes, magic_offset + 0x28)? as usize;

    validate_nro_segment(size, text_offset, text_size, "text")?;
    validate_nro_segment(size, ro_offset, ro_size, "ro")?;
    validate_nro_segment(size, data_offset, data_size, "data")?;

    let bss_offset = data_offset
        .checked_add(data_size)
        .ok_or_else(|| anyhow!("NRO data segment range overflowed"))?;
    let image_size = bss_offset
        .checked_add(bss_size)
        .ok_or_else(|| anyhow!("NRO image size overflowed when extending BSS"))?
        .max(size);

    let mut image_bytes = vec![0u8; image_size];
    image_bytes[..size].copy_from_slice(&bytes[..size]);

    let entry = load_address
        .checked_add(text_offset as u64)
        .ok_or_else(|| anyhow!("NRO relocated entry address overflowed"))?;

    Ok(Some(ProgramImage {
        kind: ProgramKind::NroRelocatable,
        entry,
        relocation_bias: 0,
        segments: vec![ProgramSegment {
            address: load_address,
            bytes: image_bytes,
        }],
        runtime_hints: ProgramRuntimeHints::default(),
        romfs_data: None,
        module_bases: vec![(entry, load_address)],
    }))
}

fn validate_nro_segment(file_size: usize, offset: usize, size: usize, label: &str) -> Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| anyhow!("NRO {label} segment range overflowed"))?;

    if end > file_size {
        bail!("NRO {label} segment ({offset:#x}..{end:#x}) exceeds executable size {file_size:#x}");
    }

    Ok(())
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("header read out of range at offset {offset:#x}"))?;

    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| anyhow!("header read out of range at offset {offset:#x}"))?;

    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

pub fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use lz4_flex::block::compress as lz4_compress;

    use crate::keys::KeySet;

    use super::{
        NcaContentType, ProgramKind, derive_nca_title_key, extract_exefs_programs,
        extract_main_from_pfs0, inspect_nca_header, parse_program_image,
    };

    fn write_u32(buffer: &mut [u8], offset: usize, value: u32) {
        buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn build_pfs0(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut string_table = Vec::new();
        let mut string_offsets = Vec::new();
        for (name, _) in entries {
            string_offsets.push(string_table.len() as u32);
            string_table.extend_from_slice(name.as_bytes());
            string_table.push(0);
        }

        let table_size = entries.len() * 0x18;
        let header_size = 0x10 + table_size + string_table.len();
        let data_size: usize = entries.iter().map(|(_, bytes)| bytes.len()).sum();
        let mut output = vec![0u8; header_size + data_size];

        output[0..4].copy_from_slice(b"PFS0");
        write_u32(&mut output, 0x4, entries.len() as u32);
        write_u32(&mut output, 0x8, string_table.len() as u32);

        let mut running_data_offset = 0u64;
        for (index, (_, bytes)) in entries.iter().enumerate() {
            let entry_offset = 0x10 + index * 0x18;
            output[entry_offset..entry_offset + 8]
                .copy_from_slice(&running_data_offset.to_le_bytes());
            output[entry_offset + 8..entry_offset + 16]
                .copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            write_u32(&mut output, entry_offset + 0x10, string_offsets[index]);

            running_data_offset += bytes.len() as u64;
        }

        output[0x10 + table_size..header_size].copy_from_slice(&string_table);

        let mut write_offset = header_size;
        for (_, bytes) in entries {
            output[write_offset..write_offset + bytes.len()].copy_from_slice(bytes);
            write_offset += bytes.len();
        }

        output
    }

    #[test]
    fn treats_non_object_input_as_raw_binary() {
        let image = parse_program_image(&[0x01, 0x02, 0x03, 0x04], 0x8000_0000)
            .expect("raw input should parse as raw program");

        assert_eq!(image.kind, ProgramKind::Raw);
        assert_eq!(image.entry, 0x8000_0000);
        assert_eq!(image.segments.len(), 1);
        assert_eq!(image.segments[0].address, 0x8000_0000);
        assert_eq!(image.segments[0].bytes, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn parses_uncompressed_nso_sections_and_bss() {
        let text = vec![0x11, 0x22, 0x33, 0x44];
        let ro = vec![0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let data = vec![0x10, 0x20, 0x30, 0x40];

        let text_offset = 0x100usize;
        let ro_offset = text_offset + text.len();
        let data_offset = ro_offset + ro.len();
        let mut bytes = vec![0u8; data_offset + data.len()];

        bytes[0..4].copy_from_slice(b"NSO0");
        write_u32(&mut bytes, 0x10, text_offset as u32);
        write_u32(&mut bytes, 0x14, 0x0000_0000);
        write_u32(&mut bytes, 0x18, text.len() as u32);

        write_u32(&mut bytes, 0x20, ro_offset as u32);
        write_u32(&mut bytes, 0x24, 0x0000_1000);
        write_u32(&mut bytes, 0x28, ro.len() as u32);

        write_u32(&mut bytes, 0x30, data_offset as u32);
        write_u32(&mut bytes, 0x34, 0x0000_2000);
        write_u32(&mut bytes, 0x38, data.len() as u32);
        write_u32(&mut bytes, 0x3c, 0x20);

        bytes[text_offset..text_offset + text.len()].copy_from_slice(&text);
        bytes[ro_offset..ro_offset + ro.len()].copy_from_slice(&ro);
        bytes[data_offset..data_offset + data.len()].copy_from_slice(&data);

        let image = parse_program_image(&bytes, 0x8000_0000).expect("nso parse should succeed");

        assert_eq!(image.kind, ProgramKind::NsoRelocatable);
        assert_eq!(image.entry, 0x8000_0000);
        assert_eq!(image.segments.len(), 4);
        assert_eq!(image.segments[0].address, 0x8000_0000);
        assert_eq!(image.segments[0].bytes, text);
        assert_eq!(image.segments[1].address, 0x8000_1000);
        assert_eq!(image.segments[1].bytes, ro);
        assert_eq!(image.segments[2].address, 0x8000_2000);
        assert_eq!(image.segments[2].bytes, data);
        assert_eq!(image.segments[3].address, 0x8000_2004);
        assert_eq!(image.segments[3].bytes.len(), 0x20);
        assert!(image.segments[3].bytes.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn parses_lz4_compressed_nso_sections() {
        let text = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
        let ro = vec![0x88, 0x77, 0x66, 0x55];
        let data = vec![0x01, 0x03, 0x05, 0x07, 0x09];

        let text_c = lz4_compress(&text);
        let ro_c = lz4_compress(&ro);
        let data_c = lz4_compress(&data);

        let text_offset = 0x100usize;
        let ro_offset = text_offset + text_c.len();
        let data_offset = ro_offset + ro_c.len();

        let mut bytes = vec![0u8; data_offset + data_c.len()];
        bytes[0..4].copy_from_slice(b"NSO0");
        write_u32(&mut bytes, 0x0c, 0b111);

        write_u32(&mut bytes, 0x10, text_offset as u32);
        write_u32(&mut bytes, 0x14, 0x0000_0000);
        write_u32(&mut bytes, 0x18, text.len() as u32);

        write_u32(&mut bytes, 0x20, ro_offset as u32);
        write_u32(&mut bytes, 0x24, 0x0000_0800);
        write_u32(&mut bytes, 0x28, ro.len() as u32);

        write_u32(&mut bytes, 0x30, data_offset as u32);
        write_u32(&mut bytes, 0x34, 0x0000_1000);
        write_u32(&mut bytes, 0x38, data.len() as u32);

        write_u32(&mut bytes, 0x60, text_c.len() as u32);
        write_u32(&mut bytes, 0x64, ro_c.len() as u32);
        write_u32(&mut bytes, 0x68, data_c.len() as u32);

        bytes[text_offset..text_offset + text_c.len()].copy_from_slice(&text_c);
        bytes[ro_offset..ro_offset + ro_c.len()].copy_from_slice(&ro_c);
        bytes[data_offset..data_offset + data_c.len()].copy_from_slice(&data_c);

        let image =
            parse_program_image(&bytes, 0x8000_0000).expect("compressed nso parse should succeed");

        assert_eq!(image.kind, ProgramKind::NsoRelocatable);
        assert_eq!(image.segments.len(), 3);
        assert_eq!(image.segments[0].bytes, text);
        assert_eq!(image.segments[1].bytes, ro);
        assert_eq!(image.segments[2].bytes, data);
    }

    #[test]
    fn parses_nro_and_extends_bss_region() {
        let mut bytes = vec![0u8; 0x200];

        for (i, byte) in bytes[..0x100].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(3).wrapping_add(1);
        }

        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        bytes[0x18..0x1c].copy_from_slice(&0x100u32.to_le_bytes());

        bytes[0x20..0x24].copy_from_slice(&0x40u32.to_le_bytes());
        bytes[0x24..0x28].copy_from_slice(&0x20u32.to_le_bytes());

        bytes[0x28..0x2c].copy_from_slice(&0x60u32.to_le_bytes());
        bytes[0x2c..0x30].copy_from_slice(&0x20u32.to_le_bytes());

        bytes[0x30..0x34].copy_from_slice(&0xf0u32.to_le_bytes());
        bytes[0x34..0x38].copy_from_slice(&0x08u32.to_le_bytes());
        bytes[0x38..0x3c].copy_from_slice(&0x20u32.to_le_bytes());

        let image = parse_program_image(&bytes, 0x8000_0000).expect("nro parse should succeed");

        assert_eq!(image.kind, ProgramKind::NroRelocatable);
        assert_eq!(image.entry, 0x8000_0040);
        assert_eq!(image.segments.len(), 1);
        assert_eq!(image.segments[0].address, 0x8000_0000);
        assert_eq!(image.segments[0].bytes.len(), 0x118);
        assert_eq!(image.segments[0].bytes[0x20], bytes[0x20]);
        assert_eq!(image.segments[0].bytes[0xff], bytes[0xff]);
        assert!(
            image.segments[0].bytes[0x100..]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn rejects_nro_with_invalid_segment_range() {
        let mut bytes = vec![0u8; 0x140];
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        bytes[0x18..0x1c].copy_from_slice(&0x100u32.to_le_bytes());

        bytes[0x20..0x24].copy_from_slice(&0x40u32.to_le_bytes());
        bytes[0x24..0x28].copy_from_slice(&0x20u32.to_le_bytes());
        bytes[0x28..0x2c].copy_from_slice(&0x60u32.to_le_bytes());
        bytes[0x2c..0x30].copy_from_slice(&0x20u32.to_le_bytes());

        bytes[0x30..0x34].copy_from_slice(&0xf8u32.to_le_bytes());
        bytes[0x34..0x38].copy_from_slice(&0x20u32.to_le_bytes());

        let err = parse_program_image(&bytes, 0x8000_0000)
            .expect_err("invalid segment bounds should fail");
        let text = err.to_string();
        assert!(text.contains("NRO data segment"));
    }

    #[test]
    fn rejects_nca_container_until_support_is_implemented() {
        let mut bytes = vec![0u8; 0x500];
        bytes[0x200..0x204].copy_from_slice(b"NCA3");

        let err = parse_program_image(&bytes, 0x8000_0000)
            .expect_err("nca input should fail with explicit unsupported message");
        assert!(err.to_string().contains("NCA container detected"));
    }

    #[test]
    fn rejects_nsp_container_until_support_is_implemented() {
        let bytes = build_pfs0(&[("main", &[0xaa, 0xbb, 0xcc, 0xdd])]);

        let err = parse_program_image(&bytes, 0x8000_0000)
            .expect_err("nsp input should fail with explicit unsupported message");
        assert!(err.to_string().contains("NSP container detected"));
    }

    #[test]
    fn extracts_main_file_from_pfs0_payload() {
        let bytes = build_pfs0(&[
            ("main", &[0xde, 0xad, 0xbe, 0xef]),
            ("rtld", &[0x11, 0x22, 0x33]),
        ]);

        let main = extract_main_from_pfs0(&bytes)
            .expect("pfs0 parse should succeed")
            .expect("main file should exist");
        assert_eq!(main, vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn extracts_exefs_programs_in_expected_order() {
        let bytes = build_pfs0(&[
            ("main", &[0xaa]),
            ("sdk", &[0xbb]),
            ("rtld", &[0xcc]),
            ("readme", &[0xdd]),
        ]);

        let programs = extract_exefs_programs(&bytes).expect("exefs extraction should succeed");

        assert_eq!(programs.len(), 3);
        assert_eq!(programs[0].name, "rtld");
        assert_eq!(programs[0].bytes, vec![0xcc]);
        assert_eq!(programs[1].name, "main");
        assert_eq!(programs[1].bytes, vec![0xaa]);
        assert_eq!(programs[2].name, "sdk");
        assert_eq!(programs[2].bytes, vec![0xbb]);
    }

    #[test]
    fn decrypts_title_key_from_rights_id_entry() {
        let mut prod = BTreeMap::new();
        prod.insert(
            "titlekek_00".to_string(),
            vec![
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f,
            ],
        );

        let rights_id = [0x44u8; 16];
        let rights_id_hex = rights_id
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();

        let mut title = BTreeMap::new();
        title.insert(
            rights_id_hex,
            vec![
                0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
                0xc5, 0x5a,
            ],
        );

        let key_set = KeySet { prod, title };
        let title_key = derive_nca_title_key(&key_set, 0, &rights_id)
            .expect("title key decryption should succeed");

        assert_eq!(
            title_key,
            [
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ]
        );
    }

    #[test]
    fn inspects_plaintext_nca_header_content_type() {
        let mut bytes = vec![0u8; 0xc00];
        bytes[0x200..0x204].copy_from_slice(b"NCA3");
        bytes[0x205] = 0;
        bytes[0x220] = 0x15;

        let info = inspect_nca_header(&bytes, &KeySet::default())
            .expect("plaintext NCA3 header should be inspectable");
        assert_eq!(info.content_type, NcaContentType::Program);
        assert_eq!(info.key_generation, 0x14);
        assert!(!info.has_rights_id);
    }
}
