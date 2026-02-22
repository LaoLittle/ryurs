use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ryurs_core::{Emulator, EmulatorConfig, KeySet, read_program};

fn parse_u64(input: &str) -> Result<u64> {
    if let Some(hex) = input.strip_prefix("0x") {
        return Ok(u64::from_str_radix(hex, 16)?);
    }
    Ok(input.parse::<u64>()?)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let program_path = PathBuf::from(args.next().context("missing program path")?);
    let prod_keys = PathBuf::from(args.next().context("missing prod.keys path")?);
    let title_keys = PathBuf::from(args.next().context("missing title.keys path")?);
    let start = parse_u64(&args.next().context("missing start address")?)?;
    let end = parse_u64(&args.next().context("missing end address")?)?;
    let output = PathBuf::from(args.next().context("missing output path")?);

    if end <= start {
        anyhow::bail!("end must be greater than start");
    }

    let key_set = KeySet::read_from_paths(Some(&prod_keys), Some(&title_keys))?;
    let program_bytes = read_program(&program_path)?;

    let mut config = EmulatorConfig::default();
    config.gpu_enabled = false;

    let mut emulator = Emulator::new(config)?;
    let _ = emulator.load_program_nsp(0x8000_0000, &program_bytes, &key_set)?;

    let mut out = Vec::with_capacity((end - start) as usize);
    let mut address = start;
    while address < end {
        let value = emulator.read_memory_u32(address).unwrap_or(0);
        out.extend_from_slice(&value.to_le_bytes());
        address = address.saturating_add(4);
    }

    let mut file = File::create(&output)?;
    file.write_all(&out)?;

    println!("wrote {} bytes to {}", out.len(), output.display());
    Ok(())
}
