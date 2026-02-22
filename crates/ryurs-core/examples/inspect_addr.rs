use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ryurs_core::{KeySet, parse_nsp_program_image, read_program};

fn parse_hex_u64(input: &str) -> Result<u64> {
    let trimmed = input.trim_start_matches("0x");
    u64::from_str_radix(trimmed, 16)
        .with_context(|| format!("failed to parse hex address: {input}"))
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let program = PathBuf::from(args.next().context("missing NSP path")?);
    let prod_keys = PathBuf::from(args.next().context("missing prod.keys path")?);
    let title_keys = PathBuf::from(args.next().context("missing title.keys path")?);
    let address = parse_hex_u64(&args.next().context("missing target address")?)?;
    let radius = args
        .next()
        .map(|value| parse_hex_u64(&value))
        .transpose()?
        .unwrap_or(0x60);

    let key_set = KeySet::read_from_paths(Some(&prod_keys), Some(&title_keys))?;
    let bytes = read_program(&program)?;
    let image = parse_nsp_program_image(&bytes, 0x8000_0000, &key_set)?;

    for segment in image.segments {
        let start = segment.address;
        let end = start.saturating_add(segment.bytes.len() as u64);
        if !(start..end).contains(&address) {
            continue;
        }

        println!(
            "segment: [0x{start:016x}..0x{end:016x}) len=0x{:x}",
            segment.bytes.len()
        );
        let window_start = address.saturating_sub(radius).max(start);
        let window_end = address.saturating_add(radius).min(end.saturating_sub(4));

        let mut pc = window_start & !3;
        while pc <= window_end {
            let offset = (pc - start) as usize;
            if offset + 4 > segment.bytes.len() {
                break;
            }
            let word = u32::from_le_bytes([
                segment.bytes[offset],
                segment.bytes[offset + 1],
                segment.bytes[offset + 2],
                segment.bytes[offset + 3],
            ]);
            let marker = if pc == address { "<-- target" } else { "" };
            println!("0x{pc:016x}: 0x{word:08x} {marker}");
            pc = pc.saturating_add(4);
        }

        return Ok(());
    }

    anyhow::bail!("address 0x{address:016x} not found in any segment")
}
