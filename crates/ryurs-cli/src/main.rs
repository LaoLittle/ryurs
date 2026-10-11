use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{Context as AnyhowContext, Result};
use clap::Parser;
use gpui::{
    App, Application, Bounds, Context, Render, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use tracing::info;

use ryurs_core::{
    DisplayFrame, Emulator, EmulatorConfig, ExitReason, KeyInventory, KeySet, NcaContentType,
    ProgramImage, ProgramKind, inspect_nca_header, list_nca_exefs_program_names, read_program,
};

// mod unicorn_compare;

#[derive(Debug, Clone, Parser)]
#[command(name = "ryurs")]
#[command(about = "Rust rewrite scaffold for Ryujinx")]
struct Cli {
    #[arg(long)]
    program: Option<PathBuf>,

    #[arg(long, help = "Path to prod.keys")]
    prod_keys: Option<PathBuf>,

    #[arg(long, help = "Path to title.keys")]
    title_keys: Option<PathBuf>,

    #[arg(long, default_value_t = 0x8000_0000)]
    load_address: u64,

    #[arg(long, default_value_t = 256)]
    blocks: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Run until trap (or limit) in headless mode"
    )]
    until_trap: bool,

    #[arg(long, default_value_t = 1_000_000)]
    until_trap_limit: usize,

    #[arg(long, default_value_t = false, help = "Run terminal mode without GPUI")]
    headless: bool,

    #[arg(long, default_value_t = false, help = "Disable the wgpu backend")]
    no_gpu: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Open a window and display rendered frames live"
    )]
    display: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Compare JIT output against Unicorn block-by-block"
    )]
    unicorn_compare: bool,
}

#[derive(Debug, Clone)]
struct UiStartupOptions {
    program_path: Option<PathBuf>,
    prod_keys_path: Option<PathBuf>,
    title_keys_path: Option<PathBuf>,
    key_set: KeySet,
    key_inventory: KeyInventory,
    load_address: u64,
    blocks: usize,
    until_trap_limit: usize,
    run_until_trap_in_headless: bool,
    gpu_enabled: bool,
    display: bool,
}

#[derive(Debug, Clone)]
struct UiRunReport {
    source: String,
    executed_blocks: usize,
    end_pc: u64,
    last_block_start_pc: Option<u64>,
    last_block_start_instruction: Option<u32>,
    sp: u64,
    x29: u64,
    x30: u64,
    x0: u64,
    x1: u64,
    x8: u64,
    x9: u64,
    fault_instruction: Option<u32>,
    fault_prev_instruction: Option<u32>,
    fault_prev2_instruction: Option<u32>,
    fault_prev3_instruction: Option<u32>,
    fault_address: Option<u64>,
    decode_error_pc: Option<u64>,
    decode_error_opcode: Option<u32>,
    stop_reason_raw: ExitReason,
    stop_reason: String,
}

#[derive(Debug, Clone)]
enum RunTarget {
    Demo,
    Program(PathBuf),
}

struct RyursUi {
    startup: UiStartupOptions,
    emulator: Option<Emulator>,
    loaded_source: String,
    status: String,
    last_report: Option<UiRunReport>,
    history: Vec<UiRunReport>,
}

impl RyursUi {
    fn new(startup: UiStartupOptions) -> Self {
        let mut ui = Self {
            startup,
            emulator: None,
            loaded_source: "(none)".to_string(),
            status: "Initializing emulator session...".to_string(),
            last_report: None,
            history: Vec::new(),
        };

        ui.bootstrap();
        ui
    }

    fn bootstrap(&mut self) {
        let target = self.default_target();
        match self.load_target(target) {
            Ok(source) => {
                self.status = format!("Loaded {source}. Ready for stepping.");
            }
            Err(err) => {
                self.status = format!("Startup failed: {err:#}");
                self.emulator = None;
            }
        }
    }

    fn default_target(&self) -> RunTarget {
        match self.startup.program_path.clone() {
            Some(path) => RunTarget::Program(path),
            None => RunTarget::Demo,
        }
    }

    fn load_target(&mut self, target: RunTarget) -> Result<String> {
        let mut emulator = build_emulator(&self.startup)?;
        let source = load_into_emulator(&mut emulator, &self.startup, target)?;

        self.loaded_source = source.clone();
        self.last_report = None;
        self.history.clear();
        self.emulator = Some(emulator);
        Ok(source)
    }

    fn record_report(&mut self, report: UiRunReport) {
        self.last_report = Some(report.clone());
        self.history.push(report);
        if self.history.len() > 12 {
            let remove_count = self.history.len() - 12;
            self.history.drain(0..remove_count);
        }
    }

    fn run_steps(&mut self, blocks: usize) -> Result<UiRunReport> {
        let emulator = self
            .emulator
            .as_mut()
            .with_context(|| "emulator session is not initialized")?;

        let report = emulator.run_steps(blocks)?;

        Ok(UiRunReport {
            source: self.loaded_source.clone(),
            executed_blocks: report.executed_blocks,
            end_pc: report.end_pc,
            last_block_start_pc: report.last_block_start_pc,
            last_block_start_instruction: report.last_block_start_instruction,
            sp: emulator.state.sp,
            x29: emulator.state.read_x(29),
            x30: emulator.state.read_x(30),
            x0: emulator.state.read_x(0),
            x1: emulator.state.read_x(1),
            x8: emulator.state.read_x(8),
            x9: emulator.state.read_x(9),
            fault_instruction: report.fault_instruction,
            fault_prev_instruction: report.fault_prev_instruction,
            fault_prev2_instruction: report.fault_prev2_instruction,
            fault_prev3_instruction: report.fault_prev3_instruction,
            fault_address: report.fault_address,
            decode_error_pc: report.decode_error_pc,
            decode_error_opcode: report.decode_error_opcode,
            stop_reason_raw: report.stop_reason,
            stop_reason: reason_label(report.stop_reason),
        })
    }

    fn reload_demo(&mut self, cx: &mut Context<Self>) {
        match self.load_target(RunTarget::Demo) {
            Ok(source) => {
                self.status = format!("Loaded {source}. Ready.");
            }
            Err(err) => {
                self.status = format!("Failed to load demo: {err:#}");
            }
        }

        cx.notify();
    }

    fn reload_program(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.startup.program_path.clone() else {
            self.status = "No --program path was provided on startup.".to_string();
            cx.notify();
            return;
        };

        match self.load_target(RunTarget::Program(path)) {
            Ok(source) => {
                self.status = format!("Loaded {source}. Ready.");
            }
            Err(err) => {
                self.status = format!("Failed to load program: {err:#}");
            }
        }

        cx.notify();
    }

    fn step_once(&mut self, cx: &mut Context<Self>) {
        match self.run_steps(1) {
            Ok(report) => {
                self.status = format!("Stepped 1 block from {}.", report.source);
                self.record_report(report);
            }
            Err(err) => {
                self.status = format!("Step failed: {err:#}");
            }
        }

        cx.notify();
    }

    fn run_batch(&mut self, cx: &mut Context<Self>) {
        match self.run_steps(self.startup.blocks) {
            Ok(report) => {
                self.status = format!(
                    "Ran {} blocks from {}.",
                    report.executed_blocks, report.source
                );
                self.record_report(report);
            }
            Err(err) => {
                self.status = format!("Batch run failed: {err:#}");
            }
        }

        cx.notify();
    }

    fn run_until_trap(&mut self, cx: &mut Context<Self>) {
        match self.run_steps(self.startup.until_trap_limit) {
            Ok(report) => {
                if matches!(report.stop_reason_raw, ExitReason::Running) {
                    self.status = format!(
                        "Reached until-trap limit ({} blocks) with no trap.",
                        self.startup.until_trap_limit
                    );
                } else {
                    self.status = format!(
                        "Stopped after {} blocks with {}.",
                        report.executed_blocks, report.stop_reason
                    );
                }
                self.record_report(report);
            }
            Err(err) => {
                self.status = format!("Run-until-trap failed: {err:#}");
            }
        }

        cx.notify();
    }
}

impl Render for RyursUi {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let program_label = self
            .startup
            .program_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(none)".to_string());
        let prod_keys_label = self
            .startup
            .prod_keys_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(not found)".to_string());
        let title_keys_label = self
            .startup
            .title_keys_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(not found)".to_string());

        let snapshot = self.emulator.as_ref().map(|emulator| emulator.snapshot());
        let pc = snapshot.map_or(0, |state| state.pc);
        let sp = snapshot.map_or(0, |state| state.sp);
        let x0 = snapshot.map_or(0, |state| state.x0);
        let x1 = snapshot.map_or(0, |state| state.x1);
        let x2 = snapshot.map_or(0, |state| state.x2);
        let x3 = snapshot.map_or(0, |state| state.x3);
        let nzcv = snapshot.map_or(0, |state| state.nzcv);
        let kernel_ticks = snapshot.map_or(0, |state| state.kernel_ticks);
        let current_el = snapshot
            .map(|state| format!("{:?}", state.current_el))
            .unwrap_or_else(|| "n/a".to_string());

        let mut report_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_md()
            .bg(rgb(0x1a2230))
            .text_color(rgb(0xdce6f3))
            .child("No run report yet.");

        if let Some(report) = &self.last_report {
            report_panel = div()
                .flex()
                .flex_col()
                .gap_1()
                .p_3()
                .rounded_md()
                .bg(rgb(0x1a2230))
                .text_color(rgb(0xdce6f3))
                .child(format!("source: {}", report.source))
                .child(format!("executed blocks: {}", report.executed_blocks))
                .child(format!("end pc: 0x{:016x}", report.end_pc))
                .child(format!("x0: 0x{:016x}", report.x0))
                .child(format!("x1: 0x{:016x}", report.x1))
                .child(format!("x8: 0x{:016x}", report.x8))
                .child(format!("x9: 0x{:016x}", report.x9))
                .child(format!("stop reason: {}", report.stop_reason));

            if let Some(word) = report.fault_instruction {
                report_panel = report_panel.child(format!("fault insn: 0x{word:08x}"));
            }
            if let Some(word) = report.fault_prev_instruction {
                report_panel = report_panel.child(format!("fault prev: 0x{word:08x}"));
            }
            if let Some(word) = report.fault_prev2_instruction {
                report_panel = report_panel.child(format!("fault prev2: 0x{word:08x}"));
            }
            if let Some(word) = report.fault_prev3_instruction {
                report_panel = report_panel.child(format!("fault prev3: 0x{word:08x}"));
            }
            if let Some(address) = report.fault_address {
                report_panel = report_panel.child(format!("fault addr: 0x{address:016x}"));
            }
            if let Some(pc) = report.decode_error_pc {
                report_panel = report_panel.child(format!("decode error pc: 0x{pc:016x}"));
            }
            if let Some(word) = report.decode_error_opcode {
                report_panel = report_panel.child(format!("decode error opcode: 0x{word:08x}"));
            }
        }

        let mut history_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_md()
            .bg(rgb(0x15202d))
            .text_color(rgb(0xc8d7e8))
            .child("No history yet.");

        if !self.history.is_empty() {
            history_panel = self.history.iter().rev().take(6).fold(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .p_3()
                    .rounded_md()
                    .bg(rgb(0x15202d))
                    .text_color(rgb(0xc8d7e8))
                    .child("Recent runs:"),
                |panel, item| {
                    panel.child(format!(
                        "- {} blocks, pc=0x{:016x}, {}",
                        item.executed_blocks, item.end_pc, item.stop_reason
                    ))
                },
            );
        }

        div()
            .flex()
            .flex_col()
            .gap_3()
            .size_full()
            .p_4()
            .bg(rgb(0x101823))
            .text_color(rgb(0xe8edf5))
            .child(div().text_xl().child("ryurs GPUI Frontend"))
            .child(div().text_sm().opacity(0.9).child(format!(
                "load address: 0x{:016x}",
                self.startup.load_address
            )))
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("batch blocks: {}", self.startup.blocks)),
            )
            .child(div().text_sm().opacity(0.9).child(format!(
                "until-trap limit: {}",
                self.startup.until_trap_limit
            )))
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("gpu enabled: {}", self.startup.gpu_enabled)),
            )
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("configured program path: {}", program_label)),
            )
            .child(div().text_sm().opacity(0.9).child(format!(
                "keys: prod={} title={} total={}",
                self.startup.key_inventory.prod_count,
                self.startup.key_inventory.title_count,
                self.startup.key_inventory.total_count()
            )))
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("prod.keys path: {}", prod_keys_label)),
            )
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("title.keys path: {}", title_keys_label)),
            )
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("loaded source: {}", self.loaded_source)),
            )
            .child(div().text_sm().opacity(0.9).child(format!(
                "live pc: 0x{:016x}  sp: 0x{:016x}  el: {}  nzcv: 0x{:08x}",
                pc, sp, current_el, nzcv
            )))
            .child(div().text_sm().opacity(0.9).child(format!(
                "x0: 0x{:016x}  x1: 0x{:016x}  x2: 0x{:016x}  x3: 0x{:016x}",
                x0, x1, x2, x3
            )))
            .child(
                div()
                    .text_sm()
                    .opacity(0.9)
                    .child(format!("kernel ticks: {}", kernel_ticks)),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .id("load-demo")
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x2b65d9))
                            .hover(|style| style.bg(rgb(0x3b74e6)))
                            .cursor_pointer()
                            .text_color(rgb(0xffffff))
                            .child("Load Demo")
                            .on_click(cx.listener(|this, _, _, cx| this.reload_demo(cx))),
                    )
                    .child(
                        div()
                            .id("load-program")
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x2b65d9))
                            .hover(|style| style.bg(rgb(0x3b74e6)))
                            .cursor_pointer()
                            .text_color(rgb(0xffffff))
                            .child("Load Program")
                            .on_click(cx.listener(|this, _, _, cx| this.reload_program(cx))),
                    )
                    .child(
                        div()
                            .id("step-once")
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x2b8b57))
                            .hover(|style| style.bg(rgb(0x35a16a)))
                            .cursor_pointer()
                            .text_color(rgb(0xffffff))
                            .child("Step 1")
                            .on_click(cx.listener(|this, _, _, cx| this.step_once(cx))),
                    )
                    .child(
                        div()
                            .id("run-batch")
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x2b8b57))
                            .hover(|style| style.bg(rgb(0x35a16a)))
                            .cursor_pointer()
                            .text_color(rgb(0xffffff))
                            .child(format!("Run {}", self.startup.blocks))
                            .on_click(cx.listener(|this, _, _, cx| this.run_batch(cx))),
                    )
                    .child(
                        div()
                            .id("run-until-trap")
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0xa45a21))
                            .hover(|style| style.bg(rgb(0xb7692d)))
                            .cursor_pointer()
                            .text_color(rgb(0xffffff))
                            .child("Run Until Trap")
                            .on_click(cx.listener(|this, _, _, cx| this.run_until_trap(cx))),
                    ),
            )
            .child(
                div()
                    .p_3()
                    .rounded_md()
                    .bg(rgb(0x192230))
                    .text_color(rgb(0xdce6f3))
                    .child(format!("status: {}", self.status)),
            )
            .child(history_panel)
            .child(report_panel)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let current_dir = std::env::current_dir().context("failed to determine current directory")?;
    let prod_keys_path = resolve_keys_path(cli.prod_keys.clone(), &current_dir, "prod.keys");
    let title_keys_path = resolve_keys_path(cli.title_keys.clone(), &current_dir, "title.keys");

    let key_set = KeySet::read_from_paths(prod_keys_path.as_deref(), title_keys_path.as_deref())?;
    let key_inventory = key_set.inventory();

    info!(
        prod_keys = key_inventory.prod_count,
        title_keys = key_inventory.title_count,
        "loaded key inventory"
    );

    let startup = UiStartupOptions {
        program_path: cli.program.clone(),
        prod_keys_path,
        title_keys_path,
        key_set,
        key_inventory,
        load_address: cli.load_address,
        blocks: cli.blocks,
        until_trap_limit: cli.until_trap_limit,
        run_until_trap_in_headless: cli.until_trap,
        gpu_enabled: !cli.no_gpu,
        display: cli.display,
    };

    if cli.headless || cli.until_trap {
        run_headless(&startup)?;
        return Ok(());
    }

    if cli.display {
        run_windowed(&startup)?;
        return Ok(());
    }

    if cli.unicorn_compare {
        let program_path = cli
            .program
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--program is required for --unicorn-compare"))?;
        /* unicorn_compare::run_comparison(
            program_path,
            &startup.key_set,
            cli.load_address,
            startup.until_trap_limit,
        )?;
        */
        return Ok(());
    }

    launch_ui(startup);
    Ok(())
}

fn launch_ui(startup: UiStartupOptions) {
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1040.0), px(720.0)), cx);
        let options = startup.clone();

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |_, cx| {
                let options = options.clone();
                cx.new(move |_| RyursUi::new(options))
            },
        )
        .expect("failed to open GPUI window");

        cx.activate(true);
    });
}

fn resolve_keys_path(
    explicit: Option<PathBuf>,
    current_dir: &Path,
    file_name: &str,
) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path);
    }

    let mut roots = vec![current_dir.to_path_buf()];
    if let Some(parent) = current_dir.parent() {
        roots.push(parent.to_path_buf());
    }

    for root in roots {
        if let Some(found) = find_file_named(&root, file_name, 3) {
            return Some(found);
        }
    }

    None
}

fn find_file_named(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
    let mut queue = VecDeque::new();
    queue.push_back((root.to_path_buf(), 0usize));

    while let Some((directory, depth)) = queue.pop_front() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry_result in entries {
            let entry = match entry_result {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };

            let name = entry.file_name();
            let path = entry.path();
            if file_type.is_file() && name.to_string_lossy() == file_name {
                return Some(path);
            }

            if !file_type.is_dir() || depth >= max_depth {
                continue;
            }

            let name = name.to_string_lossy();
            if name == ".git" || name == "target" {
                continue;
            }

            queue.push_back((path, depth + 1));
        }
    }

    None
}

fn run_windowed(startup: &UiStartupOptions) -> Result<()> {
    use std::sync::Arc;
    use std::sync::mpsc;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowId};

    use ryurs_gpu_wgpu::display::DisplayRenderer;

    let target = match startup.program_path.clone() {
        Some(path) => RunTarget::Program(path),
        None => RunTarget::Demo,
    };

    // Channel for frame data: emulator → display.
    // Buffer up to 2 frames to allow the emulator to run ahead slightly.
    let (frame_tx, frame_rx) = mpsc::sync_channel::<DisplayFrame>(2);

    // Clone startup options for the emulator thread.
    let startup_clone = startup.clone();
    let budget = startup.until_trap_limit;

    struct App {
        window: Option<Arc<Window>>,
        renderer: Option<DisplayRenderer>,
        frame_rx: mpsc::Receiver<DisplayFrame>,
        startup: UiStartupOptions,
        target: RunTarget,
        frame_tx: Option<mpsc::SyncSender<DisplayFrame>>,
        budget: usize,
        emu_thread: Option<thread::JoinHandle<Result<()>>>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.renderer.is_some() {
                return; // Already initialized.
            }

            let attrs = Window::default_attributes()
                .with_title(format!("ryurs - {:?}", self.target))
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
                .with_resizable(true);

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("failed to create window"),
            );

            let display = DisplayRenderer::new(Arc::clone(&window))
                .expect("failed to create display renderer");
            self.renderer = Some(display);
            self.window = Some(Arc::clone(&window));

            // Start emulator on a background thread.
            let frame_tx = self.frame_tx.take().expect("frame_tx already taken");
            let startup = self.startup.clone();
            let target = self.target.clone();
            let budget = self.budget;

            self.emu_thread = Some(thread::spawn(move || {
                let mut emulator = build_emulator(&startup)?;
                emulator.set_frame_sender(frame_tx);
                load_into_emulator(&mut emulator, &startup, target)?;
                let _report = emulator.run_steps(budget)?;
                Ok(())
            }));

            window.request_redraw();
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(new_size) => {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(new_size.width, new_size.height);
                    }
                }
                WindowEvent::RedrawRequested => {
                    // Drain channel and use the latest frame.
                    let mut latest_frame = None;
                    while let Ok(frame) = self.frame_rx.try_recv() {
                        latest_frame = Some(frame);
                    }
                    if let Some(data) = latest_frame {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.update_framebuffer(data.width, data.height, &data.pixels);
                        }
                    }

                    if let Some(renderer) = &self.renderer {
                        if let Err(e) = renderer.render() {
                            eprintln!("[display] render error: {e}");
                        }
                    }

                    // Check if the emulator thread has finished.
                    if let Some(handle) = &self.emu_thread {
                        if handle.is_finished() {
                            if let Some(handle) = self.emu_thread.take() {
                                if let Err(e) = handle.join().expect("emulator thread panicked") {
                                    eprintln!("[display] emulator error: {e}");
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            // Request continuous redraws to poll for new frames from the emulator.
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App {
        window: None,
        renderer: None,
        frame_rx,
        startup: startup_clone,
        target,
        frame_tx: Some(frame_tx),
        budget,
        emu_thread: None,
    };

    event_loop.run_app(&mut app).context("event loop error")?;

    Ok(())
}

fn run_headless(startup: &UiStartupOptions) -> Result<()> {
    let target = match startup.program_path.clone() {
        Some(path) => RunTarget::Program(path),
        None => RunTarget::Demo,
    };

    let report = thread::scope(|scope| {
        let handle = scope.spawn(|| {
            if startup.run_until_trap_in_headless {
                run_emulation_with_budget(startup, target, startup.until_trap_limit)
            } else {
                run_emulation(startup, target)
            }
        });

        handle.join().expect("emulation thread panicked")
    })?;

    println!("source: {}", report.source);
    println!(
        "keys: prod={} title={} total={}",
        startup.key_inventory.prod_count,
        startup.key_inventory.title_count,
        startup.key_inventory.total_count()
    );
    println!("executed blocks: {}", report.executed_blocks);
    if let Some(pc) = report.last_block_start_pc {
        println!("last block start pc: 0x{pc:016x}");
    }
    if let Some(word) = report.last_block_start_instruction {
        println!("last block start insn: 0x{word:08x}");
    }
    println!("end pc: 0x{:016x}", report.end_pc);
    println!("sp: 0x{:016x}", report.sp);
    println!("x29: 0x{:016x}", report.x29);
    println!("x30: 0x{:016x}", report.x30);
    println!("x0: 0x{:016x}", report.x0);
    println!("x1: 0x{:016x}", report.x1);
    println!("x8: 0x{:016x}", report.x8);
    println!("x9: 0x{:016x}", report.x9);
    if let Some(word) = report.fault_instruction {
        println!("fault insn: 0x{word:08x}");
    }
    if let Some(word) = report.fault_prev_instruction {
        println!("fault prev: 0x{word:08x}");
    }
    if let Some(word) = report.fault_prev2_instruction {
        println!("fault prev2: 0x{word:08x}");
    }
    if let Some(word) = report.fault_prev3_instruction {
        println!("fault prev3: 0x{word:08x}");
    }
    if let Some(address) = report.fault_address {
        println!("fault addr: 0x{address:016x}");
    }
    if let Some(pc) = report.decode_error_pc {
        println!("decode error pc: 0x{pc:016x}");
    }
    if let Some(word) = report.decode_error_opcode {
        println!("decode error opcode: 0x{word:08x}");
    }
    println!("stop reason: {}", report.stop_reason);
    Ok(())
}

fn run_emulation(startup: &UiStartupOptions, target: RunTarget) -> Result<UiRunReport> {
    run_emulation_with_budget(startup, target, startup.blocks)
}

fn run_emulation_with_budget(
    startup: &UiStartupOptions,
    target: RunTarget,
    budget_blocks: usize,
) -> Result<UiRunReport> {
    let mut emulator = build_emulator(startup)?;
    let source = load_into_emulator(&mut emulator, startup, target)?;
    let report = emulator.run_steps(budget_blocks)?;

    Ok(UiRunReport {
        source,
        executed_blocks: report.executed_blocks,
        end_pc: report.end_pc,
        last_block_start_pc: report.last_block_start_pc,
        last_block_start_instruction: report.last_block_start_instruction,
        sp: emulator.state.sp,
        x29: emulator.state.read_x(29),
        x30: emulator.state.read_x(30),
        x0: emulator.state.read_x(0),
        x1: emulator.state.read_x(1),
        x8: emulator.state.read_x(8),
        x9: emulator.state.read_x(9),
        fault_instruction: report.fault_instruction,
        fault_prev_instruction: report.fault_prev_instruction,
        fault_prev2_instruction: report.fault_prev2_instruction,
        fault_prev3_instruction: report.fault_prev3_instruction,
        fault_address: report.fault_address,
        decode_error_pc: report.decode_error_pc,
        decode_error_opcode: report.decode_error_opcode,
        stop_reason_raw: report.stop_reason,
        stop_reason: reason_label(report.stop_reason),
    })
}

fn build_emulator(startup: &UiStartupOptions) -> Result<Emulator> {
    let mut config = EmulatorConfig::default();
    config.gpu_enabled = startup.gpu_enabled;
    Emulator::new(config)
}

fn load_into_emulator(
    emulator: &mut Emulator,
    startup: &UiStartupOptions,
    target: RunTarget,
) -> Result<String> {
    match target {
        RunTarget::Demo => {
            emulator.load_program_words(startup.load_address, &demo_program())?;
            info!("loaded built-in demo program");
            Ok("demo".to_string())
        }
        RunTarget::Program(path) => {
            let (display_source, image) = if path.is_dir() {
                load_from_nca_directory(emulator, startup, &path)?
            } else {
                let program_bytes = read_program(&path).with_context(|| {
                    format!("failed to load external program from {}", path.display())
                })?;

                let image = if is_nca_path(&path) {
                    let image = emulator
                        .load_program_nca(startup.load_address, &program_bytes, &startup.key_set)
                        .with_context(|| {
                            format!("failed to extract executable from NCA {}", path.display())
                        })?;

                    let program_names =
                        list_nca_exefs_program_names(&program_bytes, &startup.key_set).ok();
                    let display = format_nca_source_label(&path, program_names.as_deref());
                    (display, image)
                } else if is_nsp_path(&path) {
                    (
                        path.display().to_string(),
                        emulator
                            .load_program_nsp(
                                startup.load_address,
                                &program_bytes,
                                &startup.key_set,
                            )
                            .with_context(|| {
                                format!("failed to extract executable from NSP {}", path.display())
                            })?,
                    )
                } else {
                    (
                        path.display().to_string(),
                        emulator.load_program_auto(startup.load_address, &program_bytes)?,
                    )
                };

                image
            };

            let image_desc = describe_program_image(&image);
            let entry_hex = format!("0x{:016x}", image.entry);

            info!(
                path = %path.display(),
                kind = ?image.kind,
                entry = %entry_hex,
                segments = image.segments.len(),
                "loaded external program image"
            );

            Ok(format!("{} ({})", display_source, image_desc))
        }
    }
}

fn load_from_nca_directory(
    emulator: &mut Emulator,
    startup: &UiStartupOptions,
    path: &Path,
) -> Result<(String, ProgramImage)> {
    #[derive(Debug)]
    struct NcaCandidate {
        path: PathBuf,
        bytes: Vec<u8>,
        content_type: Option<NcaContentType>,
    }

    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to list directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("failed reading entry in {}", path.display()))?;
        let candidate = entry.path();
        if is_nca_path(&candidate) {
            let bytes = read_program(&candidate)
                .with_context(|| format!("failed to read NCA candidate {}", candidate.display()))?;
            let content_type = inspect_nca_header(&bytes, &startup.key_set)
                .map(|header| header.content_type)
                .ok();

            candidates.push(NcaCandidate {
                path: candidate,
                bytes,
                content_type,
            });
        }
    }

    candidates.sort_by(|a, b| {
        nca_candidate_rank(a.content_type)
            .cmp(&nca_candidate_rank(b.content_type))
            .then_with(|| a.path.cmp(&b.path))
    });

    if candidates.is_empty() {
        anyhow::bail!(
            "directory {} does not contain any .nca files",
            path.display()
        );
    }

    let mut failures = Vec::new();
    for candidate in candidates {
        match emulator.load_program_nca(startup.load_address, &candidate.bytes, &startup.key_set) {
            Ok(image) => {
                let program_names =
                    list_nca_exefs_program_names(&candidate.bytes, &startup.key_set).ok();
                let label = format_nca_source_label(&candidate.path, program_names.as_deref());
                return Ok((label, image));
            }
            Err(err) => {
                let kind = candidate
                    .content_type
                    .map(nca_content_type_label)
                    .unwrap_or("unknown");
                failures.push(format!(
                    "{} ({}) -> {}",
                    candidate.path.display(),
                    kind,
                    err
                ));
            }
        }
    }

    let mut message = format!(
        "no executable ExeFS program set was found in {} after checking NCA files",
        path.display()
    );
    for reason in failures.iter().take(3) {
        message.push_str("\n- ");
        message.push_str(reason);
    }
    if failures.len() > 3 {
        message.push_str("\n- ...");
    }

    anyhow::bail!(message)
}

fn format_nca_source_label(path: &Path, program_names: Option<&[String]>) -> String {
    match program_names {
        Some(names) if !names.is_empty() => {
            format!("{} [exefs: {}]", path.display(), names.join(","))
        }
        _ => path.display().to_string(),
    }
}

fn nca_candidate_rank(content_type: Option<NcaContentType>) -> u8 {
    match content_type {
        Some(NcaContentType::Program) => 0,
        Some(_) => 1,
        None => 2,
    }
}

fn nca_content_type_label(content_type: NcaContentType) -> &'static str {
    match content_type {
        NcaContentType::Program => "program",
        NcaContentType::Meta => "meta",
        NcaContentType::Control => "control",
        NcaContentType::Manual => "manual",
        NcaContentType::Data => "data",
        NcaContentType::PublicData => "public-data",
        NcaContentType::Unknown(_) => "unknown",
    }
}

fn is_nca_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("nca"))
        .unwrap_or(false)
}

fn is_nsp_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("nsp"))
        .unwrap_or(false)
}

fn describe_program_image(image: &ProgramImage) -> String {
    let hints = format_runtime_hints_suffix(image);

    match image.kind {
        ProgramKind::Raw => format!(
            "raw, entry=0x{:016x}, bytes={}{}",
            image.entry,
            image.total_bytes(),
            hints,
        ),
        ProgramKind::NsoRelocatable => format!(
            "nso relocatable, entry=0x{:016x}, segments={}, bytes={}{}",
            image.entry,
            image.segments.len(),
            image.total_bytes(),
            hints,
        ),
        ProgramKind::NroRelocatable => format!(
            "nro relocatable, entry=0x{:016x}, bytes={}{}",
            image.entry,
            image.total_bytes(),
            hints,
        ),
        ProgramKind::Elf64Aarch64Relocated => format!(
            "elf64-aarch64 relocated, entry=0x{:016x}, bias=0x{:016x}, segments={}, bytes={}{}",
            image.entry,
            image.relocation_bias,
            image.segments.len(),
            image.total_bytes(),
            hints,
        ),
        ProgramKind::NcaExeFsBundle => format!(
            "nca exefs bundle, entry=0x{:016x}, segments={}, bytes={}{}",
            image.entry,
            image.segments.len(),
            image.total_bytes(),
            hints,
        ),
    }
}

fn format_runtime_hints_suffix(image: &ProgramImage) -> String {
    match (
        image.runtime_hints.process_flags,
        image.runtime_hints.main_thread_stack_size,
    ) {
        (Some(flags), Some(stack_size)) => {
            format!(", process_flags=0x{flags:02x}, stack_size=0x{stack_size:x}")
        }
        (Some(flags), None) => format!(", process_flags=0x{flags:02x}"),
        (None, Some(stack_size)) => format!(", stack_size=0x{stack_size:x}"),
        (None, None) => String::new(),
    }
}

fn reason_label(reason: ExitReason) -> String {
    match reason {
        ExitReason::Running => "running".to_string(),
        ExitReason::DispatcherInterrupt => "dispatcher-interrupt".to_string(),
        ExitReason::DecodeFault => "decode-fault".to_string(),
        ExitReason::MemoryFault => "memory-fault".to_string(),
        ExitReason::PrivilegeFault => "privilege-fault".to_string(),
        ExitReason::ThreadExit => "thread-exit".to_string(),
        ExitReason::Breakpoint(imm) => format!("breakpoint(imm={imm})"),
        ExitReason::SupervisorCall(imm) => format!("svc(imm={imm})"),
        ExitReason::UnknownSupervisorCall(imm) => format!("unknown-svc(imm={imm})"),
        ExitReason::HypervisorCall(imm) => format!("hvc(imm={imm})"),
        ExitReason::SecureMonitorCall(imm) => format!("smc(imm={imm})"),
        ExitReason::ExceptionReturn => "eret".to_string(),
        ExitReason::GranularityFault => "granularity-fault".to_string(),
    }
}

fn demo_program() -> Vec<u32> {
    vec![
        0xd280_0060, // movz x0, #3
        0xf100_0400, // subs x0, x0, #1
        0xb5ff_ffe0, // cbnz x0, -4
        0xd280_0541, // movz x1, #42
        0xd420_0020, // brk #1
    ]
}
