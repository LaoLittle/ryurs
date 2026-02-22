use serde::{Deserialize, Serialize};

use ryurs_dynarec::RESET_VECTOR;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmulatorConfig {
    pub reset_vector: u64,
    pub memory_size: usize,
    pub block_max_instructions: usize,
    pub gpu_enabled: bool,
    #[serde(default)]
    pub force_interpreter: bool,
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        Self {
            reset_vector: RESET_VECTOR,
            memory_size: 64 * 1024 * 1024,
            block_max_instructions: 48,
            gpu_enabled: true,
            force_interpreter: false,
        }
    }
}
