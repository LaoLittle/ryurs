use crate::decoder::{BranchCondition, SystemRegister};
use std::time::{SystemTime, UNIX_EPOCH};

pub const RESET_VECTOR: u64 = 0x8000_0000;
pub const CNTFRQ_EL0_HZ: u64 = 19_200_000;
pub const CTR_EL0_VALUE: u64 = 0x8444_c004;
pub const DCZID_EL0_VALUE: u64 = 0x0000_0004;

pub fn virtual_counter_ticks() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    nanos
        .saturating_mul(CNTFRQ_EL0_HZ as u128)
        .saturating_div(1_000_000_000) as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ExceptionLevel {
    El0 = 0,
    El1 = 1,
    El2 = 2,
    El3 = 3,
}

impl ExceptionLevel {
    pub fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::El0),
            1 => Some(Self::El1),
            2 => Some(Self::El2),
            3 => Some(Self::El3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionKind {
    Svc,
    Hvc,
    Smc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    Running,
    ThreadExit,
    DispatcherInterrupt,
    Breakpoint(u16),
    DecodeFault,
    MemoryFault,
    GranularityFault,
    PrivilegeFault,
    SupervisorCall(u16),
    UnknownSupervisorCall(u16),
    HypervisorCall(u16),
    SecureMonitorCall(u16),
    ExceptionReturn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysRegError {
    PrivilegeFault,
    ReadOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionError {
    PrivilegeFault,
    InvalidReturnLevel,
}

pub const CPUSTATE_FASTMEM_SIZE_OFFSET: i32 = std::mem::offset_of!(CpuState, fastmem_size) as i32;
pub const CPUSTATE_FASTMEM_HOST_PTR_OFFSET: i32 =
    std::mem::offset_of!(CpuState, fastmem_host_ptr) as i32;
pub const CPUSTATE_PTE_PTR_OFFSET: i32 = std::mem::offset_of!(CpuState, pte_ptr) as i32;
pub const CPUSTATE_PTE_PAGE_COUNT_OFFSET: i32 =
    std::mem::offset_of!(CpuState, pte_page_count) as i32;
pub const CPUSTATE_PTE_GUEST_BASE_OFFSET: i32 =
    std::mem::offset_of!(CpuState, pte_guest_base) as i32;
pub const CPUSTATE_EXCLUSIVE_VALID_OFFSET: i32 =
    std::mem::offset_of!(CpuState, exclusive_valid) as i32;
pub const CPUSTATE_EXCLUSIVE_SIZE_OFFSET: i32 =
    std::mem::offset_of!(CpuState, exclusive_size) as i32;
pub const CPUSTATE_EXCLUSIVE_ADDRESS_OFFSET: i32 =
    std::mem::offset_of!(CpuState, exclusive_address) as i32;
pub const CPUSTATE_VREG_BASE_OFFSET: i32 = std::mem::offset_of!(CpuState, v) as i32;
pub const CPUSTATE_VREG_STRIDE: i32 = std::mem::size_of::<[u64; 2]>() as i32;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CpuState {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub nzcv: u32,
    pub exit_code: u32,
    pub trap_imm: u16,
    _padding: u16,

    pub current_el: u8,
    pub daif: u8,
    pub spsel: u8,
    _reserved0: u8,

    pub exclusive_valid: u8,
    pub exclusive_size: u8,
    _reserved1: u16,
    _reserved2: u32,
    pub exclusive_address: u64,

    pub fastmem_guest_base: u64,
    pub fastmem_size: u64,
    pub fastmem_host_ptr: u64,

    pub pte_guest_base: u64,
    pub pte_page_count: u64,
    pub pte_ptr: u64,

    pub sp_el0: u64,
    pub sp_el1: u64,
    pub sp_el2: u64,
    pub sp_el3: u64,

    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub tpidr_el1: u64,

    pub spsr_el1: u64,
    pub elr_el1: u64,
    pub spsr_el2: u64,
    pub elr_el2: u64,
    pub spsr_el3: u64,
    pub elr_el3: u64,

    pub vbar_el1: u64,
    pub vbar_el2: u64,
    pub vbar_el3: u64,

    v: [[u64; 2]; 32],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub context: CpuState,
    pub return_address: usize,
    pub late_link_source_pc: u64,
    pub late_link_next_pc: u64,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            context: CpuState::default(),
            return_address: 0,
            late_link_source_pc: u64::MAX,
            late_link_next_pc: 0,
        }
    }
}

impl Default for CpuState {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: RESET_VECTOR,
            nzcv: 0,
            exit_code: 0,
            trap_imm: 0,
            _padding: 0,

            current_el: ExceptionLevel::El1 as u8,
            daif: 0,
            spsel: 1,
            _reserved0: 0,

            exclusive_valid: 0,
            exclusive_size: 0,
            _reserved1: 0,
            _reserved2: 0,
            exclusive_address: 0,

            fastmem_guest_base: 0,
            fastmem_size: 0,
            fastmem_host_ptr: 0,

            pte_guest_base: 0,
            pte_page_count: 0,
            pte_ptr: 0,

            sp_el0: 0,
            sp_el1: 0,
            sp_el2: 0,
            sp_el3: 0,

            tpidr_el0: 0,
            tpidrro_el0: 0,
            tpidr_el1: 0,

            spsr_el1: 0,
            elr_el1: 0,
            spsr_el2: 0,
            elr_el2: 0,
            spsr_el3: 0,
            elr_el3: 0,

            vbar_el1: RESET_VECTOR,
            vbar_el2: RESET_VECTOR,
            vbar_el3: RESET_VECTOR,

            v: [[0; 2]; 32],
        }
    }
}

impl CpuState {
    pub fn read_x(&self, reg: u8) -> u64 {
        if reg >= 31 {
            0
        } else {
            self.x[reg as usize]
        }
    }

    pub fn write_x(&mut self, reg: u8, value: u64) {
        if reg < 31 {
            self.x[reg as usize] = value;
        }
    }

    pub fn read_v128(&self, reg: u8) -> (u64, u64) {
        if reg >= 32 {
            (0, 0)
        } else {
            let lanes = self.v[reg as usize];
            (lanes[0], lanes[1])
        }
    }

    pub fn write_v128(&mut self, reg: u8, lo: u64, hi: u64) {
        if reg < 32 {
            self.v[reg as usize] = [lo, hi];
        }
    }

    pub fn current_el(&self) -> ExceptionLevel {
        ExceptionLevel::from_raw(self.current_el).unwrap_or(ExceptionLevel::El1)
    }

    pub fn set_current_el(&mut self, el: ExceptionLevel) {
        self.current_el = el as u8;
    }

    pub fn clear_exit(&mut self) {
        self.exit_code = 0;
        self.trap_imm = 0;
    }

    pub fn set_exclusive_reservation(&mut self, address: u64, size: u8) {
        self.exclusive_valid = 1;
        self.exclusive_size = size;
        self.exclusive_address = address;
    }

    pub fn clear_exclusive_reservation(&mut self) {
        self.exclusive_valid = 0;
        self.exclusive_size = 0;
        self.exclusive_address = 0;
    }

    pub fn check_exclusive_reservation(&self, address: u64, size: u8) -> bool {
        self.exclusive_valid != 0
            && self.exclusive_size == size
            && self.exclusive_address == address
    }

    pub fn set_fastmem_region(&mut self, guest_base: u64, host_ptr: *mut u8, size: u64) {
        self.fastmem_guest_base = guest_base;
        self.fastmem_size = size;
        self.fastmem_host_ptr = host_ptr as usize as u64;
    }

    pub fn clear_fastmem_region(&mut self) {
        self.fastmem_guest_base = 0;
        self.fastmem_size = 0;
        self.fastmem_host_ptr = 0;
    }

    pub fn set_pte_region(&mut self, guest_base: u64, page_count: u64, pte_ptr: *const u64) {
        self.pte_guest_base = guest_base;
        self.pte_page_count = page_count;
        self.pte_ptr = pte_ptr as usize as u64;
    }

    pub fn clear_pte_region(&mut self) {
        self.pte_guest_base = 0;
        self.pte_page_count = 0;
        self.pte_ptr = 0;
    }

    pub fn set_breakpoint(&mut self, imm: u16) {
        self.exit_code = 1;
        self.trap_imm = imm;
    }

    pub fn set_decode_fault(&mut self) {
        self.exit_code = 2;
    }

    pub fn set_memory_fault(&mut self) {
        self.exit_code = 3;
    }

    pub fn set_granularity_fault(&mut self) {
        self.exit_code = 10;
    }

    pub fn set_dispatcher_interrupt(&mut self) {
        self.exit_code = 9;
    }

    pub fn set_privilege_fault(&mut self) {
        self.exit_code = 4;
    }

    pub fn set_svc(&mut self, imm: u16) {
        self.exit_code = 5;
        self.trap_imm = imm;
    }

    pub fn set_hvc(&mut self, imm: u16) {
        self.exit_code = 6;
        self.trap_imm = imm;
    }

    pub fn set_smc(&mut self, imm: u16) {
        self.exit_code = 7;
        self.trap_imm = imm;
    }

    pub fn set_eret(&mut self) {
        self.exit_code = 8;
    }

    pub fn exit_reason(&self) -> ExitReason {
        match self.exit_code {
            0 => ExitReason::Running,
            1 => ExitReason::Breakpoint(self.trap_imm),
            2 => ExitReason::DecodeFault,
            3 => ExitReason::MemoryFault,
            4 => ExitReason::PrivilegeFault,
            5 => ExitReason::SupervisorCall(self.trap_imm),
            6 => ExitReason::HypervisorCall(self.trap_imm),
            7 => ExitReason::SecureMonitorCall(self.trap_imm),
            8 => ExitReason::ExceptionReturn,
            9 => ExitReason::DispatcherInterrupt,
            10 => ExitReason::GranularityFault,
            _ => ExitReason::DecodeFault,
        }
    }

    pub fn read_sys_reg(&self, reg: SystemRegister) -> Result<u64, SysRegError> {
        self.check_sys_reg_access(reg, false)?;

        Ok(match reg {
            SystemRegister::CurrentEl => (u64::from(self.current_el)) << 2,
            SystemRegister::Nzcv => u64::from(self.nzcv),
            SystemRegister::Daif => u64::from(self.daif),
            SystemRegister::Spsel => u64::from(self.spsel & 1),
            SystemRegister::CntfrqEl0 => CNTFRQ_EL0_HZ,
            SystemRegister::CtrEl0 => CTR_EL0_VALUE,
            SystemRegister::DczidEl0 => DCZID_EL0_VALUE,
            SystemRegister::CntpctEl0 => virtual_counter_ticks(),
            SystemRegister::CntvctEl0 => virtual_counter_ticks(),
            SystemRegister::TpidrEl0 => self.tpidr_el0,
            SystemRegister::TpidrroEl0 => self.tpidrro_el0,
            SystemRegister::TpidrEl1 => self.tpidr_el1,
            SystemRegister::SpEl0 => self.sp_el0,
            SystemRegister::SpEl1 => self.sp_el1,
            SystemRegister::SpEl2 => self.sp_el2,
            SystemRegister::SpEl3 => self.sp_el3,
            SystemRegister::SpsrEl1 => self.spsr_el1,
            SystemRegister::ElrEl1 => self.elr_el1,
            SystemRegister::SpsrEl2 => self.spsr_el2,
            SystemRegister::ElrEl2 => self.elr_el2,
            SystemRegister::SpsrEl3 => self.spsr_el3,
            SystemRegister::ElrEl3 => self.elr_el3,
            SystemRegister::VbarEl1 => self.vbar_el1,
            SystemRegister::VbarEl2 => self.vbar_el2,
            SystemRegister::VbarEl3 => self.vbar_el3,
        })
    }

    pub fn write_sys_reg(&mut self, reg: SystemRegister, value: u64) -> Result<(), SysRegError> {
        self.check_sys_reg_access(reg, true)?;

        match reg {
            SystemRegister::CurrentEl => return Err(SysRegError::ReadOnly),
            SystemRegister::Nzcv => self.nzcv = (value as u32) & 0xf000_0000,
            SystemRegister::Daif => self.daif = (value as u8) & 0x0f,
            SystemRegister::Spsel => self.spsel = (value as u8) & 1,
            SystemRegister::CntfrqEl0
            | SystemRegister::CtrEl0
            | SystemRegister::DczidEl0
            | SystemRegister::CntpctEl0
            | SystemRegister::CntvctEl0 => return Err(SysRegError::ReadOnly),
            SystemRegister::TpidrEl0 => self.tpidr_el0 = value,
            SystemRegister::TpidrroEl0 => self.tpidrro_el0 = value,
            SystemRegister::TpidrEl1 => self.tpidr_el1 = value,
            SystemRegister::SpEl0 => self.sp_el0 = value,
            SystemRegister::SpEl1 => self.sp_el1 = value,
            SystemRegister::SpEl2 => self.sp_el2 = value,
            SystemRegister::SpEl3 => self.sp_el3 = value,
            SystemRegister::SpsrEl1 => self.spsr_el1 = value,
            SystemRegister::ElrEl1 => self.elr_el1 = value,
            SystemRegister::SpsrEl2 => self.spsr_el2 = value,
            SystemRegister::ElrEl2 => self.elr_el2 = value,
            SystemRegister::SpsrEl3 => self.spsr_el3 = value,
            SystemRegister::ElrEl3 => self.elr_el3 = value,
            SystemRegister::VbarEl1 => self.vbar_el1 = value,
            SystemRegister::VbarEl2 => self.vbar_el2 = value,
            SystemRegister::VbarEl3 => self.vbar_el3 = value,
        }

        Ok(())
    }

    pub fn condition_holds(&self, condition: BranchCondition) -> bool {
        let n = ((self.nzcv >> 31) & 1) != 0;
        let z = ((self.nzcv >> 30) & 1) != 0;
        let c = ((self.nzcv >> 29) & 1) != 0;
        let v = ((self.nzcv >> 28) & 1) != 0;

        match condition {
            BranchCondition::Eq => z,
            BranchCondition::Ne => !z,
            BranchCondition::Cs => c,
            BranchCondition::Cc => !c,
            BranchCondition::Mi => n,
            BranchCondition::Pl => !n,
            BranchCondition::Vs => v,
            BranchCondition::Vc => !v,
            BranchCondition::Hi => c && !z,
            BranchCondition::Ls => !(c && !z),
            BranchCondition::Ge => n == v,
            BranchCondition::Lt => n != v,
            BranchCondition::Gt => !z && (n == v),
            BranchCondition::Le => z || (n != v),
            BranchCondition::Al => true,
            BranchCondition::Nv => false,
        }
    }

    pub fn update_nzcv_add(&mut self, lhs: u64, rhs: u64, result: u64) {
        let carry = (lhs as u128) + (rhs as u128) > (u64::MAX as u128);
        let overflow = (((lhs ^ result) & (rhs ^ result)) >> 63) != 0;
        self.set_nzcv_from_bools((result >> 63) != 0, result == 0, carry, overflow);
    }

    pub fn update_nzcv_add32(&mut self, lhs: u32, rhs: u32, result: u32) {
        let carry = (lhs as u64) + (rhs as u64) > (u32::MAX as u64);
        let overflow = (((lhs ^ result) & (rhs ^ result)) >> 31) != 0;
        self.set_nzcv_from_bools((result >> 31) != 0, result == 0, carry, overflow);
    }

    pub fn update_nzcv_sub(&mut self, lhs: u64, rhs: u64, result: u64) {
        let carry = lhs >= rhs;
        let overflow = (((lhs ^ rhs) & (lhs ^ result)) >> 63) != 0;
        self.set_nzcv_from_bools((result >> 63) != 0, result == 0, carry, overflow);
    }

    pub fn update_nzcv_sub32(&mut self, lhs: u32, rhs: u32, result: u32) {
        let carry = lhs >= rhs;
        let overflow = (((lhs ^ rhs) & (lhs ^ result)) >> 31) != 0;
        self.set_nzcv_from_bools((result >> 31) != 0, result == 0, carry, overflow);
    }

    pub fn take_exception(&mut self, kind: ExceptionKind) -> Result<(), ExceptionError> {
        let from_el = self.current_el();
        let target_el = match kind {
            ExceptionKind::Svc => {
                if from_el < ExceptionLevel::El1 {
                    ExceptionLevel::El1
                } else {
                    from_el
                }
            }
            ExceptionKind::Hvc => {
                if from_el < ExceptionLevel::El2 {
                    ExceptionLevel::El2
                } else {
                    from_el
                }
            }
            ExceptionKind::Smc => ExceptionLevel::El3,
        };

        if from_el > target_el {
            return Err(ExceptionError::PrivilegeFault);
        }

        let pstate = self.encode_pstate();
        let return_pc = self.pc;

        match target_el {
            ExceptionLevel::El1 => {
                self.spsr_el1 = pstate;
                self.elr_el1 = return_pc;
            }
            ExceptionLevel::El2 => {
                self.spsr_el2 = pstate;
                self.elr_el2 = return_pc;
            }
            ExceptionLevel::El3 => {
                self.spsr_el3 = pstate;
                self.elr_el3 = return_pc;
            }
            ExceptionLevel::El0 => return Err(ExceptionError::PrivilegeFault),
        }

        self.current_el = target_el as u8;
        self.spsel = 1;
        let vector_offset = if from_el < target_el { 0x400 } else { 0x200 };
        self.pc = self.vector_base(target_el).wrapping_add(vector_offset);

        Ok(())
    }

    pub fn exception_return(&mut self) -> Result<(), ExceptionError> {
        let current = self.current_el();
        if current == ExceptionLevel::El0 {
            return Err(ExceptionError::PrivilegeFault);
        }

        let (saved_spsr, saved_elr) = match current {
            ExceptionLevel::El1 => (self.spsr_el1, self.elr_el1),
            ExceptionLevel::El2 => (self.spsr_el2, self.elr_el2),
            ExceptionLevel::El3 => (self.spsr_el3, self.elr_el3),
            ExceptionLevel::El0 => return Err(ExceptionError::PrivilegeFault),
        };

        let target_el_raw = ((saved_spsr >> 2) & 0x3) as u8;
        let target_el =
            ExceptionLevel::from_raw(target_el_raw).ok_or(ExceptionError::InvalidReturnLevel)?;

        if target_el > current {
            return Err(ExceptionError::InvalidReturnLevel);
        }

        self.current_el = target_el as u8;
        self.spsel = (saved_spsr & 1) as u8;
        self.daif = ((saved_spsr >> 6) & 0x0f) as u8;
        self.nzcv = ((saved_spsr >> 28) as u32) << 28;
        self.pc = saved_elr;

        Ok(())
    }

    fn check_sys_reg_access(&self, reg: SystemRegister, write: bool) -> Result<(), SysRegError> {
        let el = self.current_el() as u8;
        let required_el = reg.required_el(write) as u8;
        if el < required_el {
            return Err(SysRegError::PrivilegeFault);
        }
        Ok(())
    }

    fn vector_base(&self, el: ExceptionLevel) -> u64 {
        match el {
            ExceptionLevel::El1 => self.vbar_el1,
            ExceptionLevel::El2 => self.vbar_el2,
            ExceptionLevel::El3 => self.vbar_el3,
            ExceptionLevel::El0 => self.vbar_el1,
        }
    }

    fn encode_pstate(&self) -> u64 {
        let el = u64::from(self.current_el);
        let spsel = u64::from(self.spsel & 1);
        let daif = u64::from(self.daif & 0x0f);
        let nzcv = u64::from((self.nzcv >> 28) & 0x0f);

        spsel | (el << 2) | (daif << 6) | (nzcv << 28)
    }

    fn set_nzcv_from_bools(&mut self, n: bool, z: bool, c: bool, v: bool) {
        self.nzcv =
            ((n as u32) << 31) | ((z as u32) << 30) | ((c as u32) << 29) | ((v as u32) << 28);
    }
}
