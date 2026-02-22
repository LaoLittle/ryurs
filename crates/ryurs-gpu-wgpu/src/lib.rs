use anyhow::{Context, Result};

pub mod display;
pub mod nvdrv;

pub struct WgpuBackend {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    pub fn submit_idle(&self) {
        self.queue.submit(std::iter::empty());
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a compatible GPU adapter")?;

        let descriptor = wgpu::DeviceDescriptor {
            label: Some("ryurs-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };

        let (device, queue) = adapter
            .request_device(&descriptor)
            .await
            .context("failed to create wgpu device")?;

        let adapter_info = adapter.get_info();

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            adapter_info,
        })
    }
}
