use std::collections::VecDeque;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ryurs_core::{Emulator, EmulatorConfig, ExitReason, KeySet, RunReport, read_program};
use ryurs_dynarec::StepResult;

fn parse_u64(input: &str) -> Result<u64> {
    if let Some(hex) = input.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16)
            .with_context(|| format!("failed to parse hex value: {input}"));
    }

    input
        .parse::<u64>()
        .with_context(|| format!("failed to parse integer value: {input}"))
}

fn read_ascii_c_string(emulator: &Emulator, address: u64, max_len: usize) -> Option<String> {
    let mut bytes = Vec::with_capacity(max_len);
    for offset in 0..max_len {
        let byte_address = address.wrapping_add(offset as u64);
        let word_address = byte_address & !3;
        let shift = ((byte_address & 3) * 8) as u32;
        let word = emulator.read_memory_u32(word_address).unwrap_or(0);
        let byte = ((word >> shift) & 0xff) as u8;
        if byte == 0 {
            break;
        }
        if !(0x20..=0x7e).contains(&byte) {
            return None;
        }
        bytes.push(byte);
    }

    if bytes.is_empty() {
        None
    } else {
        String::from_utf8(bytes).ok()
    }
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let program_path = PathBuf::from(args.next().context("missing program path")?);
    let prod_keys = PathBuf::from(args.next().context("missing prod.keys path")?);
    let title_keys = PathBuf::from(args.next().context("missing title.keys path")?);
    let budget = parse_u64(&args.next().context("missing block budget")?)? as usize;
    let trace_tail = args
        .next()
        .map(|value| parse_u64(&value))
        .transpose()?
        .unwrap_or(0) as usize;
    let memory_size_mib = args
        .next()
        .map(|value| parse_u64(&value))
        .transpose()?
        .unwrap_or(64) as usize;
    let initial_x0 = args.next().map(|value| parse_u64(&value)).transpose()?;
    let run_time_limit = env::var("RYURS_TIME_LIMIT_MS")
        .ok()
        .map(|value| parse_u64(value.trim()))
        .transpose()?
        .map(Duration::from_millis);
    let watch_pcs = match env::var("RYURS_WATCH_PCS") {
        Ok(raw) if !raw.trim().is_empty() => raw
            .split(',')
            .map(|value| parse_u64(value.trim()))
            .collect::<Result<Vec<_>>>()?,
        _ => vec![0x82d7_ea94, 0x82d7_ec80, 0x82dd_4bd4],
    };
    let stop_on_watch = env::var("RYURS_STOP_ON_WATCH")
        .map(|value| value != "0")
        .unwrap_or(false);
    let mut recent_trace: VecDeque<(usize, u64, u32, u64, u64, u64, u64, u64, u64, u64)> =
        VecDeque::with_capacity(24);

    let key_set = KeySet::read_from_paths(Some(&prod_keys), Some(&title_keys))?;
    let program_bytes = read_program(&program_path)?;

    let mut config = EmulatorConfig::default();
    config.gpu_enabled = false;
    config.memory_size = memory_size_mib.saturating_mul(1024 * 1024);

    let mut emulator = Emulator::new(config)?;
    let _ = emulator.load_program_nsp(0x8000_0000, &program_bytes, &key_set)?;
    if let Some(x0) = initial_x0 {
        emulator.state.write_x(0, x0);
    }

    let mut report = RunReport {
        executed_blocks: 0,
        stop_reason: ExitReason::Running,
        end_pc: emulator.state.pc,
        last_block_start_pc: None,
        last_block_start_instruction: None,
        fault_instruction: None,
        fault_prev_instruction: None,
        fault_prev2_instruction: None,
        fault_prev3_instruction: None,
        fault_address: None,
        decode_error_pc: None,
        decode_error_opcode: None,
    };
    let run_start = Instant::now();
    let mut time_limit_hit = false;

    for index in 0..budget {
        if let Some(limit) = run_time_limit
            && run_start.elapsed() >= limit
        {
            report.end_pc = emulator.state.pc;
            time_limit_hit = true;
            break;
        }

        let pc = emulator.state.pc;
        let insn = emulator.read_memory_u32(pc).unwrap_or(0);
        let x0 = emulator.state.read_x(0);
        let x1 = emulator.state.read_x(1);
        let x2 = emulator.state.read_x(2);
        let x8 = emulator.state.read_x(8);
        let x19 = emulator.state.read_x(19);
        let x20 = emulator.state.read_x(20);
        let x21 = emulator.state.read_x(21);
        let x22 = emulator.state.read_x(22);
        let x30 = emulator.state.read_x(30);

        if watch_pcs.contains(&pc) {
            println!(
                "watch pc=0x{pc:016x} step={index} insn=0x{insn:08x} x0=0x{x0:016x} x1=0x{x1:016x} x2=0x{x2:016x} x8=0x{x8:016x} x19=0x{x19:016x} x20=0x{x20:016x} x21=0x{x21:016x} x22=0x{x22:016x} x30=0x{x30:016x} sp=0x{:016x}",
                emulator.state.sp,
            );

            for (
                recent_step,
                recent_pc,
                recent_insn,
                recent_x0,
                recent_x1,
                recent_x2,
                recent_x8,
                recent_x19,
                recent_x20,
                recent_x21,
            ) in recent_trace.iter()
            {
                println!(
                    "watch_recent step={recent_step} pc=0x{recent_pc:016x} insn=0x{recent_insn:08x} x0=0x{recent_x0:016x} x1=0x{recent_x1:016x} x2=0x{recent_x2:016x} x8=0x{recent_x8:016x} x19=0x{recent_x19:016x} x20=0x{recent_x20:016x} x21=0x{recent_x21:016x}",
                );
            }

            if stop_on_watch {
                report.end_pc = pc;
                break;
            }
        }

        if trace_tail != 0 && budget.saturating_sub(index) <= trace_tail {
            let d8 = emulator.read_memory_u64(0x8373_80d8).unwrap_or(0);
            let e0 = emulator.read_memory_u64(0x8373_80e0).unwrap_or(0);
            let h188 = emulator.read_memory_u64(0x8373_8188).unwrap_or(0);
            println!(
                "trace[{index}] pc=0x{pc:016x} insn=0x{insn:08x} x0=0x{x0:016x} x1=0x{x1:016x} x2=0x{x2:016x} x8=0x{x8:016x} x19=0x{x19:016x} x20=0x{x20:016x} x21=0x{x21:016x} x22=0x{x22:016x} x23=0x{:016x} x24=0x{:016x} x30=0x{x30:016x} sp=0x{:016x} mem[d8]=0x{d8:016x} mem[e0]=0x{e0:016x} mem[188]=0x{h188:016x}",
                emulator.state.read_x(23),
                emulator.state.read_x(24),
                emulator.state.sp,
            );
        }

        recent_trace.push_back((index, pc, insn, x0, x1, x2, x8, x19, x20, x21));
        if recent_trace.len() > 24 {
            recent_trace.pop_front();
        }

        match emulator.step()? {
            StepResult::Advanced { start_pc, end_pc } => {
                report.executed_blocks = report.executed_blocks.saturating_add(1);
                report.end_pc = end_pc;
                report.last_block_start_pc = Some(start_pc);
                report.last_block_start_instruction = emulator.read_memory_u32(start_pc);
            }
            StepResult::Trap { pc, reason } => {
                println!("trace trap at step {index}: pc=0x{pc:016x} reason={reason:?}");
                report.end_pc = pc;
                report.stop_reason = reason;
                report.fault_instruction = emulator.read_memory_u32(pc);
                break;
            }
        }
    }

    println!("executed_blocks={}", report.executed_blocks);
    println!("end_pc=0x{:016x}", report.end_pc);
    println!("stop_reason={:?}", report.stop_reason);
    println!("time_limit_hit={time_limit_hit}");
    println!("x0=0x{:016x}", emulator.state.read_x(0));
    println!("x1=0x{:016x}", emulator.state.read_x(1));
    println!("x2=0x{:016x}", emulator.state.read_x(2));
    println!("x3=0x{:016x}", emulator.state.read_x(3));
    println!("x8=0x{:016x}", emulator.state.read_x(8));
    println!("x29=0x{:016x}", emulator.state.read_x(29));
    println!("x30=0x{:016x}", emulator.state.read_x(30));
    println!("sp=0x{:016x}", emulator.state.sp);
    println!("x19=0x{:016x}", emulator.state.read_x(19));
    println!("x20=0x{:016x}", emulator.state.read_x(20));
    println!("x21=0x{:016x}", emulator.state.read_x(21));
    println!("x22=0x{:016x}", emulator.state.read_x(22));
    println!("x23=0x{:016x}", emulator.state.read_x(23));
    println!("x24=0x{:016x}", emulator.state.read_x(24));
    for offset in (0..=0x40).step_by(8) {
        let address = emulator.state.sp.wrapping_add(offset);
        let value = emulator.read_memory_u64(address).unwrap_or(0);
        println!("sp_mem[0x{address:016x}]=0x{value:016x}");
    }

    let mut frame = emulator.state.read_x(29);
    for index in 0..4 {
        if frame == 0 {
            break;
        }
        let next = emulator.read_memory_u64(frame).unwrap_or(0);
        let ret = emulator.read_memory_u64(frame.wrapping_add(8)).unwrap_or(0);
        println!("fp[{index}] frame=0x{frame:016x} next=0x{next:016x} ret=0x{ret:016x}");
        frame = next;
    }

    if let Some(text) = read_ascii_c_string(&emulator, emulator.state.read_x(21), 0x100) {
        println!("x21_str={text:?}");
    }
    let x21 = emulator.state.read_x(21);
    for offset in (0..=0x80).step_by(8) {
        let address = x21.wrapping_add(offset);
        let value = emulator.read_memory_u64(address).unwrap_or(0);
        println!("x21_mem[0x{address:016x}]=0x{value:016x}");
    }

    let break_ptr = emulator.state.read_x(1);
    let break_size = emulator.state.read_x(2);
    if break_ptr != 0 && break_size != 0 && break_size <= 0x40 {
        let mut offset = 0u64;
        while offset < break_size {
            let address = break_ptr.wrapping_add(offset);
            let word = emulator.read_memory_u32(address).unwrap_or(0);
            println!("break_mem[0x{address:016x}]=0x{word:08x}");
            offset = offset.saturating_add(4);
        }
    }

    let probes = [
        0x82e5_3ba0u64,
        0x82e5_3ba8,
        0x82e5_3bac,
        0x82e5_3bb0,
        0x82e5_3bb4,
        0x82e5_3bb8,
        0x82e5_3bbc,
        0x82e5_3bc0,
        0x82e5_3bc4,
        0x82e5_3bc8,
        0x82e5_3bcc,
        0x8373_80d8u64,
        0x8373_80e0,
        0x8373_80f0,
        0x8373_80f8,
        0x8373_8110,
        0x8373_8120,
        0x8373_8128,
        0x8373_8130,
        0x8373_8188,
        0x8373_8190,
        0x8373_8198,
        0x8373_81a0,
        0x8373_81a8,
        0x8373_81b0,
        0x8373_81b8,
        0x8373_81c0,
        0x8373_81c8,
        0x8373_81d0,
        0x8373_81d8,
        0x8373_81e0,
        0x8373_8520,
        0x8369_a8d8,
        0x8369_a8e0,
        0x8369_abb8,
        0x8369_ab28,
        0x8369_a6b8,
        0x8369_a6c8,
        0x8369_a6d0,
        0x8369_af20,
        0x8369_afd8,
        0x8369_a638,
        0x8369_a678,
        0x8369_e6b8,
        0x8369_e6c8,
        0x8369_e6d0,
        0x836a_57c8,
        0x836a_57d0,
        0x836a_57d8,
        0x836a_5bc8,
        0x836a_5b60,
        0x836a_5b68,
        0x836a_5b70,
        0x836a_5bd0,
        0x836a_5bd8,
        0x836a_5be0,
        0x836a_5be8,
        0x836a_5bf0,
        0x836a_5bf8,
        0x836a_57c0,
        0x836a_57c8,
        0x836a_57d0,
        0x836a_57d8,
        0x836a_58e0,
        0x836a_5930,
        0x836a_5938,
        0x8373_4058,
        0x8373_4060,
        0x8373_4068,
        0x8373_4070,
        0x8373_4078,
        0x8373_4080,
        0x8373_4088,
        0x0056_b640,
    ];
    for probe in probes {
        let value = emulator.read_memory_u64(probe).unwrap_or(0);
        println!("mem[0x{probe:016x}]=0x{value:016x}");
    }

    let first = emulator.read_memory_u64(0x8373_80d8).unwrap_or(0);
    if first != 0 {
        let second = emulator.read_memory_u64(first).unwrap_or(0);
        let method = emulator
            .read_memory_u64(second.wrapping_add(0x20))
            .unwrap_or(0);
        println!("chain0=0x{first:016x}");
        println!("chain1=0x{second:016x}");
        println!("chain_method=0x{method:016x}");
    }

    for offset in (0..=0x60).step_by(8) {
        let address = 0x8373_8120u64 + offset;
        let value = emulator.read_memory_u64(address).unwrap_or(0);
        println!("heap[0x{address:016x}]=0x{value:016x}");
    }

    Ok(())
}
