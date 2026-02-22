use std::path::PathBuf;

use anyhow::{Context, Result};
use ryurs_core::{Emulator, EmulatorConfig, KeySet, read_program};
use ryurs_dynarec::decode_block;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let program_path = PathBuf::from(args.next().context("missing program path")?);
    let prod_keys = PathBuf::from(args.next().context("missing prod.keys path")?);
    let title_keys = PathBuf::from(args.next().context("missing title.keys path")?);

    let key_set = KeySet::read_from_paths(Some(&prod_keys), Some(&title_keys))?;
    let program_bytes = read_program(&program_path)?;

    let mut config = EmulatorConfig::default();
    config.gpu_enabled = false;

    let mut emulator = Emulator::new(config)?;
    let _ = emulator.load_program_nsp(0x8000_0000, &program_bytes, &key_set)?;

    let mut pcs = vec![
        0x8000_0000u64,
        0x8000_0008,
        0x8000_0010,
        0x8000_0018,
        0x8000_0020,
        0x8000_0030,
        0x8000_0040,
        0x8000_0050,
        0x8000_0060,
        0x8000_0070,
        0x8000_07f4,
        0x8000_07f8,
        0x8000_0800,
        0x8000_0804,
        0x8000_0808,
        0x8000_0810,
        0x8000_0814,
        0x8000_0818,
        0x8000_00d8,
        0x8000_00e0,
        0x8000_00e8,
        0x8000_00f0,
        0x8000_0100,
        0x8000_07f0,
        0x8000_0944,
        0x8000_0948,
        0x8000_094c,
        0x8000_0950,
        0x8000_0db4,
        0x8000_0994,
        0x8000_1600,
        0x8000_160c,
        0x8000_1610,
        0x8000_1614,
        0x8000_1620,
        0x8000_162c,
        0x8000_1630,
        0x8000_163c,
        0x8000_164c,
        0x8000_1658,
        0x8000_1660,
        0x8000_1664,
        0x8000_1668,
        0x8000_1674,
        0x8000_1680,
        0x8000_1684,
        0x8000_168c,
        0x8000_16a0,
        0x8000_16a4,
        0x8000_143c,
        0x8000_0954,
        0x8000_0dbc,
        0x8000_0dc0,
        0x8000_0dcc,
        0x8000_0e4c,
        0x8000_0e58,
        0x8000_0e5c,
        0x8000_0e60,
        0x8000_0e64,
        0x8000_0e68,
        0x8000_0e6c,
        0x8000_0e70,
        0x8000_0e74,
        0x8000_1444,
        0x8000_1448,
        0x8000_144c,
        0x8000_1450,
        0x8000_1454,
        0x8000_1458,
        0x8000_145c,
        0x8000_1460,
        0x8000_1464,
        0x8000_1468,
        0x8000_0f04,
        0x8000_0f24,
        0x8000_0fd4,
        0x8000_0fd8,
        0x8000_0fe4,
        0x8000_0fe8,
        0x8000_0fec,
        0x8000_0ff0,
        0x8000_0ff4,
        0x8000_0ff8,
        0x82db_c3b0u64,
        0x82db_c3c0,
        0x82db_c3d0,
        0x82db_c3e4u64,
        0x82db_c3f0,
        0x82db_c400,
        0x82db_c418,
        0x82db_c440,
        0x82d0_9de0,
        0x82d0_9dfc,
        0x82d0_9e04,
        0x82d0_9e20,
        0x82d0_9e14,
        0x82d0_9e28,
        0x82d0_9e34,
        0x82d0_ae20,
        0x82d0_ae60,
        0x82d0_ae68,
        0x82d0_ae6c,
        0x82d0_ae70,
        0x82d0_ae74,
        0x82d0_ae78,
        0x82d0_ae7c,
        0x82d0_ae80,
        0x82d0_ae84,
        0x82d0_aea4,
        0x82d0_aeac,
        0x82d0_aeb4,
        0x82fa_0b90,
        0x82fa_0ba0,
        0x82fa_1030,
        0x82fa_8750,
        0x82fa_8770,
        0x82fa_8780,
        0x82d7_cea0,
        0x82d7_cec0,
        0x82d7_cf80,
        0x82d7_cf60,
        0x82d7_cf74,
        0x82d7_d3b4,
        0x82d7_ea94,
        0x82d7_eac4,
        0x82d7_eae0,
        0x82d7_eaf0,
        0x82d7_eb04,
        0x82d7_eb20,
        0x82d7_eb60,
        0x82d7_eb74,
        0x82d7_ec80,
        0x82d7_ec84,
        0x82d7_ec9c,
        0x82d7_eca4,
        0x82d7_ecc8,
        0x82d7_ed00,
        0x82d7_ed20,
        0x82d7_ed90,
        0x82d7_ec94,
        0x82fb_6990,
        0x82fb_6970,
        0x82fb_7170,
        0x82fb_71c0,
        0x82fa_0640,
        0x82fa_06d0,
        0x82f6_fd78,
        0x82d8_6810,
        0x82d8_6834,
        0x82d8_2d20,
        0x82d8_2d40,
        0x82d8_2d58,
        0x82d8_2d60,
        0x82d8_2d88,
        0x82d8_2d80,
        0x82d8_2da0,
        0x82d8_2da4,
        0x82d8_2dc0,
        0x82d8_2de0,
        0x82d8_2de8,
        0x82d8_2df0,
        0x82d8_2e00,
        0x82d8_2e08,
        0x82d8_2e18,
        0x82d8_2e40,
        0x82b7_9f70,
        0x82b7_a000,
        0x82b7_a024,
        0x82b7_a034,
        0x82b7_a038,
        0x82b7_a170,
        0x82b7_a194,
        0x82b7_a198,
        0x82b7_b870,
        0x82b7_b874,
        0x82b7_b878,
        0x82b7_b87c,
        0x82b7_b880,
        0x82b7_b884,
        0x82b7_b888,
        0x82b7_b890,
        0x82b7_b8b0,
        0x82dd_4bd0,
        0x82dd_4bd4,
        0x82dd_4b84,
        0x82dd_4b98,
        0x82dd_4bd8,
        0x82dd_4bf4,
    ];

    for pc in (0x82d7_ea94u64..=0x82d7_ee20).step_by(4) {
        pcs.push(pc);
    }

    pcs.sort_unstable();
    pcs.dedup();

    for pc in pcs {
        println!("block @ 0x{pc:016x}");
        match decode_block(pc, 24, |fetch_pc| emulator.read_memory_u32(fetch_pc)) {
            Ok(block) => {
                for (index, instruction) in block.instructions.iter().enumerate() {
                    println!("  [{index:02}] {instruction:?}");
                }
                println!("  fallthrough=0x{:016x}", block.fallthrough_pc);
            }
            Err(err) => {
                println!("  decode error: {err}");
            }
        }
    }

    Ok(())
}
