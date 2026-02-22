use super::kernel::{KernelState, ServiceSessionKind};
use crate::memory::SharedGuestMemory;

const RESULT_INVALID_HANDLE: u64 = 0x1;

pub trait Service {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64;
}

#[derive(Debug, Clone, Copy)]
struct SmService;

impl Service for SmService {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        _session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        super::syscall::handle_sm_send_sync_request(kernel, current_thread_handle, memory, tls)
    }
}

#[derive(Debug, Clone, Copy)]
struct AppletOeService;

impl Service for AppletOeService {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        super::syscall::handle_generic_send_sync_request(
            kernel,
            current_thread_handle,
            memory,
            tls,
            session_handle,
            ServiceSessionKind::AppletOE,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ApmService;

impl Service for ApmService {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        super::syscall::handle_generic_send_sync_request(
            kernel,
            current_thread_handle,
            memory,
            tls,
            session_handle,
            ServiceSessionKind::Apm,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct LmService;

impl Service for LmService {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        super::syscall::handle_generic_send_sync_request(
            kernel,
            current_thread_handle,
            memory,
            tls,
            session_handle,
            ServiceSessionKind::Lm,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct StubService {
    session_kind: ServiceSessionKind,
}

impl Service for StubService {
    fn handle_send_sync_request(
        &mut self,
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        super::syscall::handle_generic_send_sync_request(
            kernel,
            current_thread_handle,
            memory,
            tls,
            session_handle,
            self.session_kind,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ServiceManager {
    services: Vec<String>,
}

impl ServiceManager {
    pub fn with_defaults() -> Self {
        Self {
            services: vec![
                "sm:".to_string(),
                "appletOE".to_string(),
                "apm".to_string(),
                "apm:am".to_string(),
                "lm".to_string(),
                "friend:a".to_string(),
                "friend:u".to_string(),
                "friend:v".to_string(),
                "friend:m".to_string(),
                "friend:s".to_string(),
                "ectx:aw".to_string(),
                "aoc:u".to_string(),
                "pctl".to_string(),
                "pctl:s".to_string(),
                "pctl:a".to_string(),
                "pctl:r".to_string(),
                "vi:u".to_string(),
                "vi:m".to_string(),
                "vi:s".to_string(),
                "nvdrv".to_string(),
                "hid".to_string(),
                "time:u".to_string(),
                "audren:u".to_string(),
                "audout:u".to_string(),
                "fsp-srv".to_string(),
            ],
        }
    }

    pub fn names(&self) -> &[String] {
        &self.services
    }

    pub fn resolve_named_port_kind(name: &str) -> Option<ServiceSessionKind> {
        match name {
            "sm:" => Some(ServiceSessionKind::Sm),
            _ => None,
        }
    }

    pub fn resolve_sm_service_kind(name: &str) -> Option<ServiceSessionKind> {
        match name {
            "appletOE" => Some(ServiceSessionKind::AppletOE),
            "apm" | "apm:am" => Some(ServiceSessionKind::Apm),
            "lm" => Some(ServiceSessionKind::Lm),
            "friend:a" | "friend:u" | "friend:v" | "friend:m" | "friend:s" => {
                Some(ServiceSessionKind::Friend)
            }
            "ectx:aw" => Some(ServiceSessionKind::EctxAw),
            "aoc:u" => Some(ServiceSessionKind::AocU),
            "pctl" | "pctl:s" | "pctl:a" | "pctl:r" => Some(ServiceSessionKind::Pctl),
            "vi:u" | "vi:m" | "vi:s" => Some(ServiceSessionKind::ViU),
            "nvdrv" => Some(ServiceSessionKind::Nvdrv),
            "hid" => Some(ServiceSessionKind::Hid),
            "time:u" | "time:a" | "time:s" | "time:r" | "time:su" => Some(ServiceSessionKind::Time),
            "audren:u" => Some(ServiceSessionKind::AudrenU),
            "audout:u" => Some(ServiceSessionKind::AudOut),
            "fsp-srv" => Some(ServiceSessionKind::FspSrv),
            "acc:u0" | "acc:u1" => Some(ServiceSessionKind::Acc),
            "nifm:u" | "nifm:s" | "nifm:a" => Some(ServiceSessionKind::Nifm),
            _ => None,
        }
    }

    pub fn dispatch_send_sync_request(
        kernel: &mut KernelState,
        current_thread_handle: u32,
        session_handle: u32,
        memory: &SharedGuestMemory,
        tls: u64,
    ) -> u64 {
        let Some(session_kind) =
            kernel.service_session_for_handle(current_thread_handle, session_handle)
        else {
            return RESULT_INVALID_HANDLE;
        };

        match session_kind {
            ServiceSessionKind::Sm => {
                let mut service = SmService;
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::AppletOE => {
                let mut service = AppletOeService;
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::Time => {
                let mut service = StubService {
                    session_kind: ServiceSessionKind::Time,
                };
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::Apm => {
                let mut service = ApmService;
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::Lm => {
                let mut service = LmService;
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::Friend => {
                let mut service = StubService { session_kind };
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
            ServiceSessionKind::EctxAw
            | ServiceSessionKind::AocU
            | ServiceSessionKind::Pctl
            | ServiceSessionKind::ViU
            | ServiceSessionKind::Nvdrv
            | ServiceSessionKind::Hid
            | ServiceSessionKind::AudrenU
            | ServiceSessionKind::AudOut
            | ServiceSessionKind::AudOutSession
            | ServiceSessionKind::FspSrv
            | ServiceSessionKind::Acc
            | ServiceSessionKind::Nifm => {
                let mut service = StubService { session_kind };
                service.handle_send_sync_request(
                    kernel,
                    current_thread_handle,
                    session_handle,
                    memory,
                    tls,
                )
            }
        }
    }
}
