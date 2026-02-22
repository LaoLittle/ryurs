GPU support using [wgpu](https://github.com/gfx-rs/wgpu).

TODO:

- [ ] Interpreting GPU inst using Compute Shader (for immediate rendering to avoid stutter).
- [ ] Translating GPU inst into WGSL in background (for pipeline cache we may use [Device::create_pipeline_cache](https://docs.rs/wgpu/latest/wgpu/struct.Device.html#method.create_pipeline_cache)).