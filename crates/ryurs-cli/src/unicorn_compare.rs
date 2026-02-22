use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use unicorn_engine::unicorn_const::{Arch, HookType, MemType, Mode, Prot};
use unicorn_engine::{RegisterARM64, Unicorn};

use ryurs_core::{
    CpuState, Emulator, EmulatorConfig, ExitReason, KeySet, ProgramImage, StepResult, read_program,
};

const THREAD_EXIT_TRAMPOLINE_ADDR: u64 = 0x1000_0000;
const STARTUP_FRAME_OFFSET: u64 = 0x80;
const SVC_EXIT_THREAD: u32 = 0xd400_0141;
const BRK_PANIC: u32 = 0xd420_0000;
const PAGE_SIZE: u64 = 0x1000;

/// Map a RegisterARM64 enum for x0..x30.
fn xreg(index: u8) -> RegisterARM64 {
    match index {
        0 => RegisterARM64::X0,
        1 => RegisterARM64::X1,
        2 => RegisterARM64::X2,
        3 => RegisterARM64::X3,
        4 => RegisterARM64::X4,
        5 => RegisterARM64::X5,
        6 => RegisterARM64::X6,
        7 => RegisterARM64::X7,
        8 => RegisterARM64::X8,
        9 => RegisterARM64::X9,
        10 => RegisterARM64::X10,
        11 => RegisterARM64::X11,
        12 => RegisterARM64::X12,
        13 => RegisterARM64::X13,
        14 => RegisterARM64::X14,
        15 => RegisterARM64::X15,
        16 => RegisterARM64::X16,
        17 => RegisterARM64::X17,
        18 => RegisterARM64::X18,
        19 => RegisterARM64::X19,
        20 => RegisterARM64::X20,
        21 => RegisterARM64::X21,
        22 => RegisterARM64::X22,
        23 => RegisterARM64::X23,
        24 => RegisterARM64::X24,
        25 => RegisterARM64::X25,
        26 => RegisterARM64::X26,
        27 => RegisterARM64::X27,
        28 => RegisterARM64::X28,
        29 => RegisterARM64::X29,
        30 => RegisterARM64::X30,
        _ => panic!("invalid register index {index}"),
    }
}

fn vreg(index: u8) -> RegisterARM64 {
    match index {
        0 => RegisterARM64::Q0,
        1 => RegisterARM64::Q1,
        2 => RegisterARM64::Q2,
        3 => RegisterARM64::Q3,
        4 => RegisterARM64::Q4,
        5 => RegisterARM64::Q5,
        6 => RegisterARM64::Q6,
        7 => RegisterARM64::Q7,
        8 => RegisterARM64::Q8,
        9 => RegisterARM64::Q9,
        10 => RegisterARM64::Q10,
        11 => RegisterARM64::Q11,
        12 => RegisterARM64::Q12,
        13 => RegisterARM64::Q13,
        14 => RegisterARM64::Q14,
        15 => RegisterARM64::Q15,
        16 => RegisterARM64::Q16,
        17 => RegisterARM64::Q17,
        18 => RegisterARM64::Q18,
        19 => RegisterARM64::Q19,
        20 => RegisterARM64::Q20,
        21 => RegisterARM64::Q21,
        22 => RegisterARM64::Q22,
        23 => RegisterARM64::Q23,
        24 => RegisterARM64::Q24,
        25 => RegisterARM64::Q25,
        26 => RegisterARM64::Q26,
        27 => RegisterARM64::Q27,
        28 => RegisterARM64::Q28,
        29 => RegisterARM64::Q29,
        30 => RegisterARM64::Q30,
        31 => RegisterARM64::Q31,
        _ => panic!("invalid vreg index {index}"),
    }
}

/// State captured from Unicorn after each block.
#[derive(Debug, Clone)]
struct UnicornBlockState {
    pc: u64,
    sp: u64,
    nzcv: u32,
    x: [u64; 31],
}

/// Count how many instructions make up the basic block starting at `start_pc`.
/// Scans AArch64 instructions until a branch/SVC/terminator is found (up to max_insns).
fn count_block_instructions(uc: &Unicorn<()>, start_pc: u64, max_insns: usize) -> usize {
    for i in 0..max_insns {
        let pc = start_pc + (i as u64) * 4;
        let mut buf = [0u8; 4];
        if uc.mem_read(pc, &mut buf).is_err() {
            return i.max(1);
        }
        let w = u32::from_le_bytes(buf);

        // Check for block-terminating instructions:
        // RET: 0xd65f03c0
        if w == 0xd65f03c0 {
            return i + 1;
        }
        // RET variants: 0xd65f0000 mask 0xfffffc1f
        if (w & 0xfffffc1f) == 0xd65f0000 {
            return i + 1;
        }
        // BR xn: 0xd61f0000 mask 0xfffffc1f
        if (w & 0xfffffc1f) == 0xd61f0000 {
            return i + 1;
        }
        // BLR xn: 0xd63f0000 mask 0xfffffc1f
        if (w & 0xfffffc1f) == 0xd63f0000 {
            return i + 1;
        }
        // B imm26: 0x14000000 mask 0xfc000000
        if (w & 0xfc000000) == 0x14000000 {
            return i + 1;
        }
        // BL imm26: 0x94000000 mask 0xfc000000
        if (w & 0xfc000000) == 0x94000000 {
            return i + 1;
        }
        // B.cond: 0x54000000 mask 0xff000010
        if (w & 0xff000010) == 0x54000000 {
            return i + 1;
        }
        // CBZ/CBNZ: 0x34000000/0x35000000 mask 0x7e000000
        if (w & 0x7e000000) == 0x34000000 {
            return i + 1;
        }
        // TBZ/TBNZ: 0x36000000/0x37000000 mask 0x7e000000
        if (w & 0x7e000000) == 0x36000000 {
            return i + 1;
        }
        // SVC: 0xd4000001 mask 0xffe0001f
        if (w & 0xffe0001f) == 0xd4000001 {
            return i + 1;
        }
        // HVC: 0xd4000002 mask 0xffe0001f
        if (w & 0xffe0001f) == 0xd4000002 {
            return i + 1;
        }
        // BRK: 0xd4200000 mask 0xffe0001f
        if (w & 0xffe0001f) == 0xd4200000 {
            return i + 1;
        }
    }
    max_insns
}

/// Check if the block starting at `start_pc` contains an exclusive store
/// (STXR/STLXR/STXP/STLXP) that lacks a preceding exclusive load (LDXR/LDAXR)
/// in the same block.  When we step Unicorn block-by-block, `emu_start()` resets
/// the internal exclusive monitor, so an "orphan" exclusive store will always
/// fail (status = 1) in Unicorn even though ryurs correctly carries the monitor
/// across interpreter blocks.  We detect this situation and skip comparison.
fn block_has_orphan_exclusive_store(uc: &Unicorn<()>, start_pc: u64, insn_count: usize) -> bool {
    let mut has_exclusive_load = false;
    for i in 0..insn_count {
        let pc = start_pc + (i as u64) * 4;
        let mut buf = [0u8; 4];
        if uc.mem_read(pc, &mut buf).is_err() {
            return false;
        }
        let w = u32::from_le_bytes(buf);

        // ARM64 Load/Store exclusive encoding family:
        //   bits [29:24] = 001000, bit [23] (o2) = 0
        //   bit [22] (L): 0 = store, 1 = load
        //
        // Exclusive load  (LDXR/LDAXR/LDXP/LDAXP): (w & 0x3FC00000) == 0x08400000
        // Exclusive store (STXR/STLXR/STXP/STLXP): (w & 0x3FC00000) == 0x08000000

        if (w & 0x3FC00000) == 0x08400000 {
            has_exclusive_load = true;
        }

        if (w & 0x3FC00000) == 0x08000000 {
            if !has_exclusive_load {
                return true; // orphan exclusive store — no matching LDXR in this block
            }
        }
    }
    false
}

/// Detect blocks that read host-driven timer registers.
/// Unicorn's values will naturally differ from ryurs's virtual counter,
/// so comparing post-block register state is not meaningful.
fn block_reads_nondeterministic_timer(uc: &Unicorn<()>, start_pc: u64, insn_count: usize) -> bool {
    for i in 0..insn_count {
        let pc = start_pc + (i as u64) * 4;
        let mut buf = [0u8; 4];
        if uc.mem_read(pc, &mut buf).is_err() {
            return false;
        }
        let w = u32::from_le_bytes(buf);
        let masked = w & 0xffff_ffe0;

        if masked == 0xd53b_e000 || masked == 0xd53b_e020 || masked == 0xd53b_e040 {
            return true;
        }
    }

    false
}

fn read_unicorn_state(uc: &Unicorn<()>) -> UnicornBlockState {
    let pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);
    let sp = uc.reg_read(RegisterARM64::SP).unwrap_or(0);
    let nzcv_raw = uc.reg_read(RegisterARM64::NZCV).unwrap_or(0) as u32;
    let nzcv = nzcv_raw >> 28; // NZCV flags are in bits [31:28]

    let mut x = [0u64; 31];
    for i in 0..31u8 {
        x[i as usize] = uc.reg_read(xreg(i)).unwrap_or(0);
    }

    UnicornBlockState { pc, sp, nzcv, x }
}

fn write_ryurs_state_to_unicorn(uc: &mut Unicorn<()>, state: &CpuState) {
    for i in 0..31u8 {
        let _ = uc.reg_write(xreg(i), state.read_x(i));
    }
    let _ = uc.reg_write(RegisterARM64::SP, state.sp);
    let _ = uc.reg_write(RegisterARM64::PC, state.pc);

    // Write NZCV register: CpuState stores flags in bits [31:28]
    let nzcv_val = (state.nzcv & 0xF000_0000) as u64;
    let _ = uc.reg_write(RegisterARM64::NZCV, nzcv_val);

    // Write TPIDR_EL0 and TPIDRRO_EL0
    let _ = uc.reg_write(RegisterARM64::TPIDR_EL0, state.tpidr_el0);
    let _ = uc.reg_write(RegisterARM64::TPIDRRO_EL0, state.tpidrro_el0);

    // Write vector registers
    for i in 0..32u8 {
        let (lo, hi) = state.read_v128(i);
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&lo.to_le_bytes());
        bytes[8..].copy_from_slice(&hi.to_le_bytes());
        // Use reg_write_long for 128-bit Q registers
        let _ = uc.reg_write_long(vreg(i), &bytes);
    }
}

fn compare_states(
    block_num: usize,
    ryurs_state: &CpuState,
    uc_state: &UnicornBlockState,
) -> Option<String> {
    // Compare PC
    if ryurs_state.pc != uc_state.pc {
        return Some(format!(
            "block {block_num}: PC diverged: ryurs=0x{:016x} unicorn=0x{:016x}",
            ryurs_state.pc, uc_state.pc
        ));
    }

    // Compare SP
    if ryurs_state.sp != uc_state.sp {
        return Some(format!(
            "block {block_num} (pc=0x{:016x}): SP diverged: ryurs=0x{:016x} unicorn=0x{:016x}",
            ryurs_state.pc, ryurs_state.sp, uc_state.sp
        ));
    }

    // Compare NZCV (ryurs stores flags in bits [31:28] of nzcv field)
    let ryurs_nzcv = (ryurs_state.nzcv >> 28) & 0xF;
    let uc_nzcv = uc_state.nzcv & 0xF;
    if ryurs_nzcv != uc_nzcv {
        return Some(format!(
            "block {block_num} (pc=0x{:016x}): NZCV diverged: ryurs=0x{:x} unicorn=0x{:x}",
            ryurs_state.pc, ryurs_nzcv, uc_nzcv
        ));
    }

    // Compare x0..x30
    for i in 0..31u8 {
        let ryurs_val = ryurs_state.read_x(i);
        let uc_val = uc_state.x[i as usize];
        if ryurs_val != uc_val {
            return Some(format!(
                "block {block_num} (pc=0x{:016x}): x{i} diverged: ryurs=0x{:016x} unicorn=0x{:016x}",
                ryurs_state.pc, ryurs_val, uc_val
            ));
        }
    }

    None
}

/// Align value up to a page boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    (value.wrapping_add(alignment - 1)) & !(alignment - 1)
}

/// Setup Unicorn with the same memory layout as the ryurs emulator.
fn setup_unicorn(
    image: &ProgramImage,
    reset_vector: u64,
    memory_size: usize,
) -> Result<Unicorn<()>> {
    let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN)
        .map_err(|e| anyhow::anyhow!("failed to create Unicorn engine: {:?}", e))?;

    let high_memory_end = reset_vector.saturating_add(memory_size as u64);

    // Map the high memory region (code + data + stack) as one large RWX region.
    // Unicorn needs page-aligned regions.
    let high_base = reset_vector & !0xFFF;
    // Map a large contiguous region covering all potential guest addresses.
    // The Switch uses a 39-bit virtual address space. We map 0 to 8GB to cover
    // code, heap, shared memory, etc.
    let total_map_size: u64 = 8 * 1024 * 1024 * 1024; // 8 GB
    uc.mem_map(0, total_map_size, Prot::ALL)
        .map_err(|e| anyhow::anyhow!("failed to map address space: {:?}", e))?;

    // Write program segments
    for segment in &image.segments {
        uc.mem_write(segment.address, &segment.bytes).map_err(|e| {
            anyhow::anyhow!(
                "failed to write segment at 0x{:016x} ({} bytes): {:?}",
                segment.address,
                segment.bytes.len(),
                e
            )
        })?;
    }

    // Write trampoline: SVC #0xA (exit thread) + BRK #0
    uc.mem_write(THREAD_EXIT_TRAMPOLINE_ADDR, &SVC_EXIT_THREAD.to_le_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write trampoline: {:?}", e))?;
    uc.mem_write(THREAD_EXIT_TRAMPOLINE_ADDR + 4, &BRK_PANIC.to_le_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write trampoline brk: {:?}", e))?;

    // Setup initial registers (same as ryurs apply_runtime_hints)
    let stack_size = image
        .runtime_hints
        .main_thread_stack_size
        .unwrap_or(0x10_0000);
    let stack_size = align_up(stack_size.max(PAGE_SIZE), PAGE_SIZE);
    let stack_end = high_memory_end & !0xFFF;
    let sp = (stack_end.saturating_sub(0x10)) & !0xf;
    let thread_ptr = stack_end
        .saturating_sub(stack_size)
        .saturating_sub(PAGE_SIZE);

    // Write startup return frame
    let frame_x29 = 0u64;
    let frame_x30 = THREAD_EXIT_TRAMPOLINE_ADDR;
    uc.mem_write(
        sp.wrapping_add(STARTUP_FRAME_OFFSET),
        &frame_x29.to_le_bytes(),
    )
    .ok();
    uc.mem_write(
        sp.wrapping_add(STARTUP_FRAME_OFFSET + 8),
        &frame_x30.to_le_bytes(),
    )
    .ok();

    // Write thread pointer data
    uc.mem_write(thread_ptr.wrapping_add(0x1f8), &thread_ptr.to_le_bytes())
        .ok();

    // Set registers
    let _ = uc.reg_write(RegisterARM64::SP, sp);
    let _ = uc.reg_write(RegisterARM64::PC, image.entry);
    let _ = uc.reg_write(RegisterARM64::X0, 0u64);
    let _ = uc.reg_write(RegisterARM64::X1, 1u64); // MAIN_THREAD_HANDLE = 1
    let _ = uc.reg_write(RegisterARM64::X18, 1u64);
    let _ = uc.reg_write(RegisterARM64::X29, 0u64);
    let _ = uc.reg_write(RegisterARM64::X30, THREAD_EXIT_TRAMPOLINE_ADDR);
    let _ = uc.reg_write(RegisterARM64::TPIDRRO_EL0, thread_ptr);
    let _ = uc.reg_write(RegisterARM64::TPIDR_EL0, 0u64);

    Ok(uc)
}

/// Run block-by-block comparison between ryurs JIT and Unicorn.
pub fn run_comparison(
    program_path: &Path,
    key_set: &KeySet,
    load_address: u64,
    max_blocks: usize,
) -> Result<()> {
    // 1. Load the program image
    let program_bytes = read_program(program_path)
        .with_context(|| format!("failed to read {}", program_path.display()))?;

    let ext = program_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let mut config = EmulatorConfig::default();
    // Start in JIT mode for fast execution through matching SVCs;
    // we'll switch to interpreter mode later for precise per-block comparison.
    let reset_vector = config.reset_vector;
    let memory_size = config.memory_size;
    let mut emulator = Emulator::new(config)?;

    let image = if ext.eq_ignore_ascii_case("nsp") {
        emulator.load_program_nsp(load_address, &program_bytes, key_set)?
    } else if ext.eq_ignore_ascii_case("nca") {
        emulator.load_program_nca(load_address, &program_bytes, key_set)?
    } else {
        emulator.load_program_auto(load_address, &program_bytes)?
    };

    println!(
        "[unicorn-compare] loaded program: entry=0x{:016x} segments={} bytes={}",
        image.entry,
        image.segments.len(),
        image.total_bytes()
    );

    // 2. Setup Unicorn with same memory
    let mut uc = setup_unicorn(&image, reset_vector, memory_size)?;

    // Sync Unicorn state to match ryurs post-load state
    write_ryurs_state_to_unicorn(&mut uc, &emulator.state);

    // Also need to sync memory: write the stack contents that ryurs wrote
    // (startup frame, thread pointer). This is already done in setup_unicorn.
    // But if the emulator wrote anything else during apply_runtime_hints, we may
    // need to read-back from ryurs memory. For now the setup_unicorn mirrors it.

    println!("[unicorn-compare] Unicorn initialized. Starting block-by-block comparison...");
    println!(
        "[unicorn-compare] initial PC=0x{:016x} SP=0x{:016x}",
        emulator.state.pc, emulator.state.sp
    );

    // 3. Run both, syncing at SVC boundaries.
    // Use step_raw() to see raw SVC traps before kernel handles them.
    // At each SVC: compare state, then let ryurs handle the SVC, sync to Unicorn.
    let mut svc_count = 0usize;
    let mut block_count = 0usize;
    let compare_blocks_after_svc = std::env::var("RYURS_UNICORN_COMPARE_AFTER_SVC")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(161);

    // Add persistent INTR hook to catch SVCs in Unicorn
    let svc_hit = Rc::new(RefCell::new(false));
    let svc_hit_clone = Rc::clone(&svc_hit);

    let _intr_hook = uc
        .add_intr_hook(move |uc: &mut Unicorn<'_, ()>, intno| {
            if intno == 2 {
                *svc_hit_clone.borrow_mut() = true;
                uc.emu_stop().ok();
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to add INTR hook: {:?}", e))?;

    // Add hook to catch unmapped memory accesses (for debugging)
    let _mem_hook = uc
        .add_mem_hook(
            HookType::MEM_WRITE_UNMAPPED | HookType::MEM_READ_UNMAPPED | HookType::MEM_FETCH_UNMAPPED,
            0,
            u64::MAX,
            |uc: &mut Unicorn<'_, ()>, mem_type: MemType, address: u64, size: usize, _value: i64| {
                let pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);
                eprintln!(
                    "[unicorn-mem-fault] {mem_type:?} at addr=0x{address:016x} size={size} PC=0x{pc:016x}"
                );
                false // don't allow the access
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to add mem hook: {:?}", e))?;

    loop {
        if block_count >= max_blocks {
            println!("[unicorn-compare] reached block limit ({max_blocks}), no divergence found!");
            break;
        }

        let block_start_pc = emulator.state.pc;

        // For per-block comparison after SVC 161, sync Unicorn to ryurs's
        // CURRENT (pre-step) state BEFORE stepping ryurs — this ensures Unicorn
        // sees the same memory as ryurs did when it executed the block.
        if svc_count >= compare_blocks_after_svc {
            write_ryurs_state_to_unicorn(&mut uc, &emulator.state);
            sync_memory_from_ryurs(&mut uc, &emulator, reset_vector, memory_size);
        }

        let pre_step_state = emulator.state.clone();

        // -- Step ryurs (raw, no SVC handling) --
        let raw_result = emulator.step_raw()?;

        block_count += 1;

        match raw_result {
            ryurs_core::StepResult::Trap {
                reason: ExitReason::SupervisorCall(imm),
                pc,
            } => {
                // ryurs hit an SVC. Run Unicorn to the same point.
                let uc_pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);
                *svc_hit.borrow_mut() = false;

                let uc_result = uc.emu_start(
                    uc_pc,
                    u64::MAX, // run until SVC hook stops us
                    0,
                    0,
                );

                let uc_svc_was_hit = *svc_hit.borrow();

                if !uc_svc_was_hit {
                    println!(
                        "[unicorn-compare] SVC {svc_count} (block {block_count}): ryurs hit SVC #{imm} at 0x{pc:016x} but Unicorn didn't"
                    );
                    println!(
                        "  unicorn PC=0x{:016x} uc_result={:?}",
                        uc.reg_read(RegisterARM64::PC).unwrap_or(0),
                        uc_result
                    );
                    break;
                }

                // Compare state at the SVC entry point
                let uc_state = read_unicorn_state(&uc);
                if let Some(diff) = compare_states(svc_count, &emulator.state, &uc_state) {
                    println!(
                        "[unicorn-compare] DIVERGENCE at SVC {svc_count} (block {block_count}, start=0x{block_start_pc:016x}, SVC #{imm}): {diff}"
                    );
                    print_full_state_diff(&emulator.state, &uc_state, svc_count);
                    break;
                }

                // Now let ryurs handle the SVC
                let resolved = emulator.resolve_svc(raw_result);

                // If the SVC resulted in a thread exit, stop
                if matches!(resolved, ryurs_core::StepResult::Trap { .. }) {
                    println!(
                        "[unicorn-compare] SVC {svc_count} resulted in trap: {:?}",
                        resolved
                    );
                    break;
                }

                // Sync Unicorn from ryurs state (kernel may have changed registers/memory)
                write_ryurs_state_to_unicorn(&mut uc, &emulator.state);
                sync_memory_from_ryurs(&mut uc, &emulator, reset_vector, memory_size);

                svc_count += 1;

                if svc_count % 100 == 0 || svc_count >= 150 {
                    println!(
                        "[unicorn-compare] SVC {svc_count} done, {block_count} blocks, PC=0x{:016x}",
                        emulator.state.pc
                    );
                }

                // After the last known matching SVC, switch to interpreter mode
                // for precise per-block comparison (no JIT chaining).
                if svc_count == compare_blocks_after_svc {
                    println!(
                        "[unicorn-compare] Switching to interpreter mode after SVC {compare_blocks_after_svc}"
                    );
                    emulator.set_force_interpreter(true);
                }
            }
            ryurs_core::StepResult::Trap { pc, reason } => {
                // Non-SVC trap (crash, decode error, etc.)
                println!(
                    "[unicorn-compare] ryurs trapped at block {block_count}: {reason:?} at PC=0x{pc:016x}"
                );

                // Run Unicorn to see if it also traps or reaches further
                let uc_pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);
                *svc_hit.borrow_mut() = false;

                // Run Unicorn with a generous instruction limit
                let _uc_result = uc.emu_start(
                    uc_pc,
                    u64::MAX,
                    0,
                    10_000_000, // up to 10M instructions
                );

                let uc_end_pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);
                println!(
                    "  Unicorn ended at PC=0x{uc_end_pc:016x} (svc_hit={})",
                    *svc_hit.borrow()
                );

                // Show state diff
                let uc_state = read_unicorn_state(&uc);
                print_full_state_diff(&emulator.state, &uc_state, block_count);

                break;
            }
            ryurs_core::StepResult::Advanced { start_pc, end_pc } => {
                // After the last SVC, do per-block comparison.
                // Unicorn is already synced to pre-step state (done above before step_raw).
                if svc_count >= compare_blocks_after_svc {
                    // Count instructions in the block, then step Unicorn for exactly that many.
                    let insn_count = count_block_instructions(&uc, start_pc, 48);

                    // Detect blocks with "orphan" exclusive stores (STXR/STLXR without
                    // a preceding LDXR in the same block). Unicorn's emu_start() resets
                    // the exclusive monitor, so these will always diverge as a comparison
                    // artifact — skip comparison and trust ryurs's result.
                    if block_has_orphan_exclusive_store(&uc, start_pc, insn_count) {
                        // Just resync Unicorn to ryurs's post-step state and continue
                        if block_count % 5000 == 0 {
                            println!(
                                "[unicorn-compare] block {block_count}: skipping orphan exclusive store at 0x{start_pc:016x}"
                            );
                        }
                        continue;
                    }

                    if block_reads_nondeterministic_timer(&uc, start_pc, insn_count) {
                        if block_count % 5000 == 0 {
                            println!(
                                "[unicorn-compare] block {block_count}: skipping timer read at 0x{start_pc:016x}"
                            );
                        }
                        write_ryurs_state_to_unicorn(&mut uc, &emulator.state);
                        sync_memory_from_ryurs(&mut uc, &emulator, reset_vector, memory_size);
                        continue;
                    }

                    *svc_hit.borrow_mut() = false;
                    let _uc_result = uc.emu_start(start_pc, 0, 0, insn_count);
                    let uc_end_pc = uc.reg_read(RegisterARM64::PC).unwrap_or(0);

                    if *svc_hit.borrow() {
                        // Unicorn hit an SVC but ryurs returned Advanced (no SVC).
                        // This means ryurs's interpreter evaluated a branch differently,
                        // skipping the SVC path. THIS IS THE BUG.
                        println!("[unicorn-compare] INTERPRETER BUG at block {block_count}!");
                        println!(
                            "  ryurs: start=0x{start_pc:016x} end=0x{end_pc:016x} (Advanced, no SVC)"
                        );
                        println!(
                            "  Unicorn: start=0x{start_pc:016x} hit SVC at PC=0x{uc_end_pc:016x}"
                        );
                        print_block_instructions(&uc, start_pc, 32);
                        let uc_state = read_unicorn_state(&uc);
                        print_full_state_diff(&emulator.state, &uc_state, block_count);
                        break;
                    }

                    // Unicorn didn't hit SVC — compare end states
                    let uc_state = read_unicorn_state(&uc);
                    if let Some(diff) = compare_states(block_count, &emulator.state, &uc_state) {
                        println!(
                            "[unicorn-compare] DIVERGENCE at block {block_count} (start=0x{start_pc:016x}): {diff}"
                        );
                        println!(
                            "  ryurs state.pc=0x{:016x}, end_pc=0x{end_pc:016x}",
                            emulator.state.pc
                        );
                        println!(
                            "  uc state.pc=0x{:016x}, uc_end_pc=0x{uc_end_pc:016x}",
                            uc_state.pc
                        );
                        println!("  insn_count={insn_count}");
                        // Show pre-step state for diverging registers
                        println!("  Pre-step state (synced to Unicorn before stepping):");
                        for i in 0..31u8 {
                            let rv = emulator.state.read_x(i);
                            let uv = uc_state.x[i as usize];
                            if rv != uv {
                                println!("    pre x{:<2}: 0x{:016x}", i, pre_step_state.read_x(i));
                            }
                        }
                        print_full_state_diff(&emulator.state, &uc_state, block_count);
                        print_block_instructions(&uc, start_pc, 32);
                        break;
                    }

                    if block_count % 5000 == 0 {
                        println!(
                            "[unicorn-compare] block {block_count}: PC=0x{:016x} (no divergence so far)",
                            emulator.state.pc
                        );
                    }
                }
            }
        }
    }

    println!("[unicorn-compare] done. {svc_count} SVCs compared, {block_count} blocks executed.");
    Ok(())
}

fn sync_memory_from_ryurs(
    uc: &mut Unicorn<()>,
    emulator: &Emulator,
    _reset_vector: u64,
    _memory_size: usize,
) {
    // Bulk-copy the entire guest memory region into Unicorn.
    let base = emulator.guest_base_address();
    let slice = unsafe { emulator.guest_memory_slice() };
    let _ = uc.mem_write(base, slice);

    // Also sync sparse pages (high regions and low regions)
    for (page_addr, data) in emulator.sparse_pages() {
        let _ = uc.mem_write(page_addr, &data);
    }

    // Flush Unicorn's translation block cache to ensure re-translation
    // after memory writes (prevents stale JIT blocks from being used).
    let _ = uc.ctl_flush_tb();
}

fn print_full_state_diff(ryurs: &CpuState, uc: &UnicornBlockState, block_num: usize) {
    println!("\n  Full state diff at block {block_num}:");
    if ryurs.pc != uc.pc {
        println!(
            "    PC:   ryurs=0x{:016x}  unicorn=0x{:016x}  ***",
            ryurs.pc, uc.pc
        );
    }
    if ryurs.sp != uc.sp {
        println!(
            "    SP:   ryurs=0x{:016x}  unicorn=0x{:016x}  ***",
            ryurs.sp, uc.sp
        );
    }
    let r_nzcv = (ryurs.nzcv >> 28) & 0xF;
    let u_nzcv = uc.nzcv & 0xF;
    if r_nzcv != u_nzcv {
        println!(
            "    NZCV: ryurs=0x{:x}            unicorn=0x{:x}            ***",
            r_nzcv, u_nzcv
        );
    }
    for i in 0..31u8 {
        let rv = ryurs.read_x(i);
        let uv = uc.x[i as usize];
        if rv != uv {
            println!(
                "    x{:<2}:  ryurs=0x{:016x}  unicorn=0x{:016x}  ***",
                i, rv, uv
            );
        }
    }
}

fn print_block_instructions(uc: &Unicorn<()>, start_pc: u64, max_insns: usize) {
    println!("  Instructions at block start 0x{start_pc:016x}:");
    for i in 0..max_insns {
        let pc = start_pc + (i as u64) * 4;
        let mut buf = [0u8; 4];
        if uc.mem_read(pc, &mut buf).is_err() {
            break;
        }
        let word = u32::from_le_bytes(buf);
        if word == 0 {
            break;
        }
        println!("    0x{pc:016x}: 0x{word:08x}");
    }
}
