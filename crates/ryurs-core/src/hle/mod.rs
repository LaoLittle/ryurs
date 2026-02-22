mod kernel;
mod service_manager;
mod syscall;

pub use kernel::KernelState;
pub use kernel::MAIN_THREAD_HANDLE;
pub use service_manager::ServiceManager;
