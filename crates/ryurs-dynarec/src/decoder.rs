use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("missing instruction at 0x{pc:016x}")]
    MissingInstruction { pc: u64 },
    #[error("unsupported opcode 0x{opcode:08x} at 0x{pc:016x}")]
    UnsupportedOpcode { pc: u64, opcode: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCondition {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

impl BranchCondition {
    fn decode(raw: u8) -> Self {
        match raw & 0x0f {
            0x0 => Self::Eq,
            0x1 => Self::Ne,
            0x2 => Self::Cs,
            0x3 => Self::Cc,
            0x4 => Self::Mi,
            0x5 => Self::Pl,
            0x6 => Self::Vs,
            0x7 => Self::Vc,
            0x8 => Self::Hi,
            0x9 => Self::Ls,
            0xa => Self::Ge,
            0xb => Self::Lt,
            0xc => Self::Gt,
            0xd => Self::Le,
            0xe => Self::Al,
            _ => Self::Nv,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemRegister {
    CurrentEl,
    Nzcv,
    Daif,
    Spsel,
    CntfrqEl0,
    CtrEl0,
    DczidEl0,
    CntpctEl0,
    CntvctEl0,
    TpidrEl0,
    TpidrroEl0,
    TpidrEl1,
    SpEl0,
    SpEl1,
    SpEl2,
    SpEl3,
    SpsrEl1,
    ElrEl1,
    SpsrEl2,
    ElrEl2,
    SpsrEl3,
    ElrEl3,
    VbarEl1,
    VbarEl2,
    VbarEl3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendType {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairAddressMode {
    SignedOffset,
    PreIndex,
    PostIndex,
}

impl ShiftType {
    fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Lsl),
            1 => Some(Self::Lsr),
            2 => Some(Self::Asr),
            3 => Some(Self::Ror),
            _ => None,
        }
    }
}

impl ExtendType {
    fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0b000 => Some(Self::Uxtb),
            0b001 => Some(Self::Uxth),
            0b010 => Some(Self::Uxtw),
            0b011 => Some(Self::Uxtx),
            0b100 => Some(Self::Sxtb),
            0b101 => Some(Self::Sxth),
            0b110 => Some(Self::Sxtw),
            0b111 => Some(Self::Sxtx),
            _ => None,
        }
    }
}

impl SystemRegister {
    pub fn from_encoding(raw: u16) -> Option<Self> {
        match raw {
            0x4212 => Some(Self::CurrentEl),
            0x5a10 => Some(Self::Nzcv),
            0x5a18 => Some(Self::Daif),
            0x4210 => Some(Self::Spsel),
            0x5f00 => Some(Self::CntfrqEl0),
            0x5801 => Some(Self::CtrEl0),
            0x5807 => Some(Self::DczidEl0),
            0x5f01 => Some(Self::CntpctEl0),
            0x5f02 => Some(Self::CntvctEl0),
            0x5e82 => Some(Self::TpidrEl0),
            0x5e83 => Some(Self::TpidrroEl0),
            0x4684 => Some(Self::TpidrEl1),
            0x4208 => Some(Self::SpEl0),
            0x4209 => Some(Self::SpEl1),
            0x6209 => Some(Self::SpEl2),
            0x7209 => Some(Self::SpEl3),
            0x4200 => Some(Self::SpsrEl1),
            0x4201 => Some(Self::ElrEl1),
            0x6200 => Some(Self::SpsrEl2),
            0x6201 => Some(Self::ElrEl2),
            0x7200 => Some(Self::SpsrEl3),
            0x7201 => Some(Self::ElrEl3),
            0x4600 => Some(Self::VbarEl1),
            0x6600 => Some(Self::VbarEl2),
            0x7600 => Some(Self::VbarEl3),
            _ => None,
        }
    }

    pub fn encoding(self) -> u16 {
        match self {
            Self::CurrentEl => 0x4212,
            Self::Nzcv => 0x5a10,
            Self::Daif => 0x5a18,
            Self::Spsel => 0x4210,
            Self::CntfrqEl0 => 0x5f00,
            Self::CtrEl0 => 0x5801,
            Self::DczidEl0 => 0x5807,
            Self::CntpctEl0 => 0x5f01,
            Self::CntvctEl0 => 0x5f02,
            Self::TpidrEl0 => 0x5e82,
            Self::TpidrroEl0 => 0x5e83,
            Self::TpidrEl1 => 0x4684,
            Self::SpEl0 => 0x4208,
            Self::SpEl1 => 0x4209,
            Self::SpEl2 => 0x6209,
            Self::SpEl3 => 0x7209,
            Self::SpsrEl1 => 0x4200,
            Self::ElrEl1 => 0x4201,
            Self::SpsrEl2 => 0x6200,
            Self::ElrEl2 => 0x6201,
            Self::SpsrEl3 => 0x7200,
            Self::ElrEl3 => 0x7201,
            Self::VbarEl1 => 0x4600,
            Self::VbarEl2 => 0x6600,
            Self::VbarEl3 => 0x7600,
        }
    }

    pub fn required_el(&self, write: bool) -> u8 {
        match self {
            Self::CurrentEl
            | Self::Nzcv
            | Self::CntfrqEl0
            | Self::CtrEl0
            | Self::DczidEl0
            | Self::CntpctEl0
            | Self::CntvctEl0
            | Self::TpidrEl0 => 0,
            Self::TpidrroEl0 if !write => 0,
            Self::Daif
            | Self::Spsel
            | Self::TpidrroEl0
            | Self::TpidrEl1
            | Self::SpEl0
            | Self::SpEl1
            | Self::SpsrEl1
            | Self::ElrEl1
            | Self::VbarEl1 => 1,
            Self::SpEl2 | Self::SpsrEl2 | Self::ElrEl2 | Self::VbarEl2 => 2,
            Self::SpEl3 | Self::SpsrEl3 | Self::ElrEl3 | Self::VbarEl3 => 3,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecodedInstruction {
    Nop,
    MovZ {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    MovZ32 {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    MovN {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    MovN32 {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    MovK {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    MovK32 {
        rd: u8,
        imm16: u16,
        shift: u8,
    },
    AndImm {
        rd: u8,
        rn: u8,
        imm: u64,
    },
    AndImm32 {
        rd: u8,
        rn: u8,
        imm: u32,
    },
    OrrImm {
        rd: u8,
        rn: u8,
        imm: u64,
    },
    OrrImm32 {
        rd: u8,
        rn: u8,
        imm: u32,
    },
    EorImm {
        rd: u8,
        rn: u8,
        imm: u64,
    },
    EorImm32 {
        rd: u8,
        rn: u8,
        imm: u32,
    },
    AndsImm {
        rd: u8,
        rn: u8,
        imm: u64,
    },
    AndsImm32 {
        rd: u8,
        rn: u8,
        imm: u32,
    },
    LslImm {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    LslImm32 {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    LsrImm {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    LsrImm32 {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    AsrImm {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    AsrImm32 {
        rd: u8,
        rn: u8,
        amount: u8,
    },
    Sbfm {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    Sbfm32 {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    Ubfm {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    Ubfm32 {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    Bfm {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    Bfm32 {
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    AddImm {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    AddImm32 {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    SubImm {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    SubImm32 {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    AddsImm {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    AddsImm32 {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    SubsImm {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    SubsImm32 {
        rd: u8,
        rn: u8,
        imm12: u16,
        shift12: bool,
    },
    AddReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AddReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    SubReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    SubReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AddsReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AddsReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    SubsReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    SubsReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AddExt {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    AddExt32 {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    SubExt {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    SubExt32 {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    AddsExt {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    AddsExt32 {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    SubsExt {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    SubsExt32 {
        rd: u8,
        rn: u8,
        rm: u8,
        extend: ExtendType,
        amount: u8,
    },
    AndReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AndReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    BicReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    BicReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    OrrReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    OrrReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    OrnReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    OrnReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    EorReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    EorReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    EonReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    EonReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AndsReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    AndsReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    BicsReg {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    BicsReg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u8,
    },
    Csel {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csel32 {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csinc {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csinc32 {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csinv {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csinv32 {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csneg {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    Csneg32 {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    CcmpReg {
        rn: u8,
        rm: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmpReg32 {
        rn: u8,
        rm: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmpImm {
        rn: u8,
        imm5: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmpImm32 {
        rn: u8,
        imm5: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmnReg {
        rn: u8,
        rm: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmnReg32 {
        rn: u8,
        rm: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmnImm {
        rn: u8,
        imm5: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    CcmnImm32 {
        rn: u8,
        imm5: u8,
        nzcv: u8,
        condition: BranchCondition,
    },
    Lslv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Lslv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Lsrv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Lsrv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Asrv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Asrv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Rorv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Rorv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Extr {
        rd: u8,
        rn: u8,
        rm: u8,
        lsb: u8,
    },
    Extr32 {
        rd: u8,
        rn: u8,
        rm: u8,
        lsb: u8,
    },
    Rbit {
        rd: u8,
        rn: u8,
    },
    Rbit32 {
        rd: u8,
        rn: u8,
    },
    Clz {
        rd: u8,
        rn: u8,
    },
    Clz32 {
        rd: u8,
        rn: u8,
    },
    Rev {
        rd: u8,
        rn: u8,
    },
    Rev32 {
        rd: u8,
        rn: u8,
    },
    Mul {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Mul32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Madd {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Madd32 {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Msub {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Msub32 {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Smaddl {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Smsubl {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Umaddl {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Umsubl {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    Smulh {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Umulh {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Udiv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Udiv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Sdiv {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Sdiv32 {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    LdxrU32 {
        rt: u8,
        rn: u8,
    },
    LdxrU64 {
        rt: u8,
        rn: u8,
    },
    LdaxrU32 {
        rt: u8,
        rn: u8,
    },
    LdaxrU64 {
        rt: u8,
        rn: u8,
    },
    LdxpU32 {
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    LdxpU64 {
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    LdaxpU32 {
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    LdaxpU64 {
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    LdarU32 {
        rt: u8,
        rn: u8,
    },
    LdarbU32 {
        rt: u8,
        rn: u8,
    },
    LdarhU32 {
        rt: u8,
        rn: u8,
    },
    LdarU64 {
        rt: u8,
        rn: u8,
    },
    StxrU32 {
        rs: u8,
        rt: u8,
        rn: u8,
    },
    StxrU64 {
        rs: u8,
        rt: u8,
        rn: u8,
    },
    StxpU32 {
        rs: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    StxpU64 {
        rs: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    StlxrU32 {
        rs: u8,
        rt: u8,
        rn: u8,
    },
    StlxrU64 {
        rs: u8,
        rt: u8,
        rn: u8,
    },
    StlxpU32 {
        rs: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    StlxpU64 {
        rs: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
    },
    StlrU32 {
        rt: u8,
        rn: u8,
    },
    StlrbU32 {
        rt: u8,
        rn: u8,
    },
    StlrhU32 {
        rt: u8,
        rn: u8,
    },
    StlrU64 {
        rt: u8,
        rn: u8,
    },
    LdrImmU64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrImmU64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrImmU32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrImmU32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrRegU64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrRegU64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrRegU32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrRegU32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrRegU16 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrRegU16 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrRegU8 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrRegU8 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrsbReg64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrsbReg32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrshReg64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrshReg32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrswReg64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrImmU16 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrImmU16 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrImmU8 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrImmU8 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrsbImm64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrsbImm32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrshImm64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrshImm32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrswImm64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdurI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturI16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturI8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdursbI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdursbI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurshI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurshI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurswI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    // Sign-extending post/pre-index load variants
    LdrshPostI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrshPostI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrshPreI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrshPreI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrsbPostI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrsbPostI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrsbPreI32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrsbPreI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrswPostI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrswPreI64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdpI64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    StpI64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    LdpI32 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    Ldpsw {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    StpI32 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    LdpFp32 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    StpFp32 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    LdpFp64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    StpFp64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    LdpV128 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    StpV128 {
        rt: u8,
        rt2: u8,
        rn: u8,
        imm7: i16,
        mode: PairAddressMode,
    },
    LdrFpImm8 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrFpImm16 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrFpImm32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrFpImm64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrFpImm128 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    LdrFpReg128 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrFpReg32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdrFpReg64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrFpReg32 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrFpReg64 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    StrFpImm8 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrFpImm16 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrFpImm32 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrFpImm64 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrFpImm128 {
        rt: u8,
        rn: u8,
        imm12: u16,
    },
    StrFpReg128 {
        rt: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: bool,
    },
    LdurFp8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurFp16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdurFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturFp8 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturFp16 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    SturFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreFp128 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreFp64 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPostFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    LdrPreFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPostFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    StrPreFp32 {
        rt: u8,
        rn: u8,
        imm9: i16,
    },
    /// AdvSIMD load multiple structures (LD1/LD2/LD3/LD4)
    /// nregs: number of registers (1-4)
    /// interleave: true for LD2/3/4 deinterleave, false for LD1
    /// post_index: None=no writeback, Some(31)=immediate, Some(rm)=register
    SimdLdMulti {
        q: bool,
        nregs: u8,
        interleave: bool,
        size: u8,
        rn: u8,
        vt: u8,
        post_index: Option<u8>,
    },
    /// AdvSIMD store multiple structures (ST1/ST2/ST3/ST4)
    SimdStMulti {
        q: bool,
        nregs: u8,
        interleave: bool,
        size: u8,
        rn: u8,
        vt: u8,
        post_index: Option<u8>,
    },
    /// AdvSIMD load single structure and replicate (LD1R/LD2R/LD3R/LD4R)
    /// Loads nregs consecutive elements and replicates each to all lanes of a register
    SimdLdReplicate {
        q: bool,
        vt: u8,
        rn: u8,
        size: u8,               // element size: 0=B, 1=H, 2=S, 3=D
        nregs: u8,              // 1, 2, 3, 4
        post_index: Option<u8>, // None=no writeback, Some(31)=immediate, Some(rm)=register
    },
    /// AdvSIMD load single structure to one lane (LD1/LD2/LD3/LD4 single element)
    SimdLdSingle {
        q: bool,
        vt: u8,
        rn: u8,
        size: u8,  // element size: 0=B, 1=H, 2=S, 3=D
        index: u8, // lane index
        nregs: u8, // 1, 2, 3, 4
        post_index: Option<u8>,
    },
    /// AdvSIMD store single structure from one lane (ST1/ST2/ST3/ST4 single element)
    SimdStSingle {
        q: bool,
        vt: u8,
        rn: u8,
        size: u8,
        index: u8,
        nregs: u8,
        post_index: Option<u8>,
    },
    /// AdvSIMD FP three-same (vector)
    VecFadd {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFsub {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmul {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFdiv {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmax {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmin {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmaxnm {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFminnm {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmulx {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFabd {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFaddp {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmaxp {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFminp {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFrecps {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFrsqrts {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFcmeq {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFcmge {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFcmgt {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFacge {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFacgt {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmla {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    VecFmls {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        sz: u8,
    },
    /// AdvSIMD FP two-register miscellaneous (vector unary) — flattened variants
    VecFrecpe {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrsqrte {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFabs {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFneg {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFsqrt {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtns {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtnu {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtms {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtmu {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtas {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtau {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtps {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtpu {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtzs {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcvtzu {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcmeq0 {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcmge0 {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcmgt0 {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcmle0 {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFcmlt0 {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecScvtf {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecUcvtf {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrint32x {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrintn {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrintm {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrintp {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrintz {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrinta {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrintx {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    VecFrinti {
        vd: u8,
        vn: u8,
        q: bool,
        sz: u8,
    },
    /// AdvSIMD FP vector × indexed element — flattened variants
    VecFmulElem {
        vd: u8,
        vn: u8,
        vm: u8,
        index: u8,
        q: bool,
        sz: u8,
    },
    VecFmlaElem {
        vd: u8,
        vn: u8,
        vm: u8,
        index: u8,
        q: bool,
        sz: u8,
    },
    VecFmlsElem {
        vd: u8,
        vn: u8,
        vm: u8,
        index: u8,
        q: bool,
        sz: u8,
    },
    VecFmulxElem {
        vd: u8,
        vn: u8,
        vm: u8,
        index: u8,
        q: bool,
        sz: u8,
    },
    /// AdvSIMD three-different (widening) — SADDL / SADDL2
    SimdSaddl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — UADDL / UADDL2
    SimdUaddl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (wide) — SADDW / SADDW2
    SimdSaddw {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (wide) — UADDW / UADDW2
    SimdUaddw {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — SSUBL / SSUBL2
    SimdSsubl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — USUBL / USUBL2
    SimdUsubl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (wide) — SSUBW / SSUBW2
    SimdSsubw {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (wide) — USUBW / USUBW2
    SimdUsubw {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — SABDL / SABDL2
    SimdSabdl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — UABDL / UABDL2
    SimdUabdl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — SMLAL / SMLAL2
    SimdSmlal {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — UMLAL / UMLAL2
    SimdUmlal {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — SMLSL / SMLSL2
    SimdSmlsl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — UMLSL / UMLSL2
    SimdUmlsl {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — SMULL / SMULL2
    SimdSmull {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (widening) — UMULL / UMULL2
    SimdUmull {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (narrowing) — ADDHN / ADDHN2
    SimdAddhn {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (narrowing) — RADDHN / RADDHN2
    SimdRaddhn {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (narrowing) — SUBHN / SUBHN2
    SimdSubhn {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD three-different (narrowing) — RSUBHN / RSUBHN2
    SimdRsubhn {
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// AdvSIMD shift by immediate (widening) — SSHLL / SSHLL2 (alias SXTL / SXTL2)
    SimdSshll {
        q: bool,
        size: u8,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// AdvSIMD shift by immediate (widening) — USHLL / USHLL2 (alias UXTL / UXTL2)
    SimdUshll {
        q: bool,
        size: u8,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// AdvSIMD shift by immediate — SSHR
    SimdSshr {
        q: bool,
        size: u8,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// AdvSIMD shift by immediate — USHR
    SimdUshr {
        q: bool,
        size: u8,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// AdvSIMD shift by immediate — SHL
    SimdShl {
        q: bool,
        size: u8,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    B {
        target: u64,
    },
    Bl {
        target: u64,
    },
    BCond {
        target: u64,
        condition: BranchCondition,
    },
    Cbz {
        rt: u8,
        target: u64,
    },
    Cbnz {
        rt: u8,
        target: u64,
    },
    Cbz32 {
        rt: u8,
        target: u64,
    },
    Cbnz32 {
        rt: u8,
        target: u64,
    },
    Tbz {
        rt: u8,
        bit: u8,
        target: u64,
    },
    Tbnz {
        rt: u8,
        bit: u8,
        target: u64,
    },
    Adr {
        rd: u8,
        target: u64,
    },
    Adrp {
        rd: u8,
        target: u64,
    },
    Br {
        rn: u8,
    },
    Blr {
        rn: u8,
    },
    Ret {
        rn: u8,
    },
    Brk {
        imm16: u16,
    },
    Svc {
        imm16: u16,
    },
    Hvc {
        imm16: u16,
    },
    Smc {
        imm16: u16,
    },
    ScvtfS32 {
        rd: u8,
        rn: u8,
    },
    UcvtfS32 {
        rd: u8,
        rn: u8,
    },
    UcvtfD32 {
        rd: u8,
        rn: u8,
    },
    ScvtfD32 {
        rd: u8,
        rn: u8,
    },
    UcvtfSReg32 {
        rd: u8,
        rn: u8,
    },
    UcvtfDReg64 {
        rd: u8,
        rn: u8,
    },
    ScvtfSReg32 {
        rd: u8,
        rn: u8,
    },
    ScvtfDReg64 {
        rd: u8,
        rn: u8,
    },
    /// SCVTF Sd, Xn — signed 64-bit integer to single-precision float
    ScvtfS64 {
        rd: u8,
        rn: u8,
    },
    /// SCVTF Dd, Xn — signed 64-bit integer to double-precision float
    ScvtfD64 {
        rd: u8,
        rn: u8,
    },
    /// UCVTF Sd, Xn — unsigned 64-bit integer to single-precision float
    UcvtfS64 {
        rd: u8,
        rn: u8,
    },
    /// UCVTF Dd, Xn — unsigned 64-bit integer to double-precision float
    UcvtfD64 {
        rd: u8,
        rn: u8,
    },
    FaddS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FsubS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmulS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmaxS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FminS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmaxnmS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FminnmS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FdivS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FaddD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FsubD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmulD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmaxD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FminD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmaxnmD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FminnmD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FdivD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// FNMUL (scalar) - floating-point negate multiply: -(Rn * Rm)
    FnmulS {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FnmulD {
        rd: u8,
        rn: u8,
        rm: u8,
    },
    FmovImmS {
        rd: u8,
        imm8: u8,
    },
    FmovImmD {
        rd: u8,
        imm8: u8,
    },
    FmovSFromW {
        rd: u8,
        rn: u8,
    },
    FmovWFromS {
        rd: u8,
        rn: u8,
    },
    FmovDFromX {
        rd: u8,
        rn: u8,
    },
    FmovXFromD {
        rd: u8,
        rn: u8,
    },
    FcvtDFromS {
        rd: u8,
        rn: u8,
    },
    FcvtSFromD {
        rd: u8,
        rn: u8,
    },
    FcvtzuXFromD {
        rd: u8,
        rn: u8,
    },
    /// FCVTZS (scalar integer) - FP to signed int, round toward zero
    FcvtzsWFromS {
        rd: u8,
        rn: u8,
    },
    FcvtzsXFromS {
        rd: u8,
        rn: u8,
    },
    FcvtzsWFromD {
        rd: u8,
        rn: u8,
    },
    FcvtzsXFromD {
        rd: u8,
        rn: u8,
    },
    /// FCVTZU (scalar integer) - FP to unsigned int, round toward zero
    FcvtzuWFromS {
        rd: u8,
        rn: u8,
    },
    FcvtzuXFromS {
        rd: u8,
        rn: u8,
    },
    FcvtzuWFromD {
        rd: u8,
        rn: u8,
    },
    /// FP to integer with explicit rounding mode (FCVTNS/NU/PS/PU/MS/MU/AS/AU).
    /// Covers: sf 0 0 11110 ftype 1 rmode opcode 000000 Rn Rd
    /// where rmode/opcode select the rounding mode and signedness.
    FcvtRoundFromFloat {
        rd: u8,
        rn: u8,
        sf: bool,          // true = 64-bit dest (Xd), false = 32-bit (Wd)
        ftype: u8,         // 0 = single source, 1 = double source
        rmode: u8,         // 0=nearest, 1=+inf, 2=-inf
        is_unsigned: bool, // false=signed, true=unsigned
    },
    /// FCVTZS (fixed-point) - FP to signed int with fractional bits
    FcvtzsFixedWFromS {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzsFixedXFromS {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzsFixedWFromD {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzsFixedXFromD {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    /// FCVTZU (fixed-point) - FP to unsigned int with fractional bits
    FcvtzuFixedWFromS {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzuFixedXFromS {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzuFixedWFromD {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtzuFixedXFromD {
        rd: u8,
        rn: u8,
        fbits: u8,
    },
    FcvtmuXFromD {
        rd: u8,
        rn: u8,
    },
    MovXFromVD {
        rd: u8,
        vn: u8,
        index: u8,
    },
    MovVD1FromX {
        vd: u8,
        rn: u8,
    },
    /// INS (general) / MOV alias — insert low bits of a GPR into a vector lane.
    InsGeneralVector {
        vd: u8,
        rn: u8,
        dst_index: u8,
        size: u8, // 0=B, 1=H, 2=S, 3=D
    },
    /// INS (element) — copy an element from one SIMD register to another.
    InsElement {
        vd: u8,
        vn: u8,
        dst_index: u8,
        src_index: u8,
        size: u8, // 0=B, 1=H, 2=S, 3=D
    },
    DupV16BFromW {
        vd: u8,
        rn: u8,
    },
    DupV4SFromW {
        vd: u8,
        rn: u8,
    },
    MoviV2DZero {
        vd: u8,
    },
    MoviV4SZero {
        vd: u8,
    },
    /// General MOVI (modified immediate) — sets vector to precomputed immediate.
    MoviVector {
        vd: u8,
        imm_lo: u64,
        imm_hi: u64,
    },
    /// MVNI (modified immediate NOT) — sets vector to precomputed inverted immediate.
    MvniVector {
        vd: u8,
        imm_lo: u64,
        imm_hi: u64,
    },
    /// ORR immediate (modified immediate OR) into vector.
    OrrImmVector {
        vd: u8,
        imm_lo: u64,
        imm_hi: u64,
    },
    /// BIC immediate (modified immediate AND-NOT) into vector.
    BicImmVector {
        vd: u8,
        imm_lo: u64,
        imm_hi: u64,
    },
    AddVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    AddpVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    UminpVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    UmaxpVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    ShrnVector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
        shift: u8,
    },
    AndVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
    },
    OrrVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
    },
    CmeqVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    CmhsVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    /// DUP (general) — broadcast GPR to all vector lanes.
    DupGeneralVector {
        vd: u8,
        rn: u8,
        q: bool,
        size: u8,
    },
    /// DUP (element) — broadcast vector element to all lanes.
    DupElementVector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
        index: u8,
    },
    /// CMHI - Unsigned greater-than comparison.
    CmhiVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    /// USHL - Unsigned shift left (variable, per-element).
    UshlVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
        size: u8,
    },
    BifVector {
        vd: u8,
        vn: u8,
        vm: u8,
        q: bool,
    },
    CmeqVectorZero {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
    },
    /// NEG - Negate vector elements.
    NegVector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
    },
    /// REV64 - Reverse elements within 64-bit doublewords.
    Rev64Vector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
    },
    /// CNT - Population count per byte in vector.
    CntVector {
        vd: u8,
        vn: u8,
        q: bool,
    },
    /// NOT (MVN) - Bitwise NOT of vector.
    NotVector {
        vd: u8,
        vn: u8,
        q: bool,
    },
    /// RBIT - Reverse bit order per byte in vector.
    RbitVector {
        vd: u8,
        vn: u8,
        q: bool,
    },
    /// UADDLV - Unsigned Add Long Across Vector.
    UaddlvVector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
    },
    /// SADDLV - Signed Add Long Across Vector.
    SaddlvVector {
        vd: u8,
        vn: u8,
        q: bool,
        size: u8,
    },
    FcmpS {
        rn: u8,
        rm: u8,
    },
    FcmpD {
        rn: u8,
        rm: u8,
    },
    FcselS {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    FcselD {
        rd: u8,
        rn: u8,
        rm: u8,
        condition: BranchCondition,
    },
    FrintmS {
        rd: u8,
        rn: u8,
    },
    FrintmD {
        rd: u8,
        rn: u8,
    },
    /// FNEG (scalar) - floating point negate
    FnegS {
        rd: u8,
        rn: u8,
    },
    FnegD {
        rd: u8,
        rn: u8,
    },
    /// FABS (scalar) - floating point absolute value
    FabsS {
        rd: u8,
        rn: u8,
    },
    FabsD {
        rd: u8,
        rn: u8,
    },
    /// FSQRT (scalar)
    FsqrtS {
        rd: u8,
        rn: u8,
    },
    FsqrtD {
        rd: u8,
        rn: u8,
    },
    DcZva {
        rt: u8,
    },
    DcCivac {
        rt: u8,
    },
    Yield,
    Wfe,
    Wfi,
    Sev,
    Sevl,
    Bti {
        btype: u8,
    },
    Dmb {
        option: u8,
    },
    Dsb {
        option: u8,
    },
    Isb {
        option: u8,
    },
    Clrex {
        option: u8,
    },
    Xpaclri,
    Eret,
    Mrs {
        rt: u8,
        reg: SystemRegister,
    },
    Msr {
        reg: SystemRegister,
        rt: u8,
    },
}

impl DecodedInstruction {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            DecodedInstruction::B { .. }
                | DecodedInstruction::Bl { .. }
                | DecodedInstruction::BCond { .. }
                | DecodedInstruction::Cbz { .. }
                | DecodedInstruction::Cbnz { .. }
                | DecodedInstruction::Cbz32 { .. }
                | DecodedInstruction::Cbnz32 { .. }
                | DecodedInstruction::Tbz { .. }
                | DecodedInstruction::Tbnz { .. }
                | DecodedInstruction::Blr { .. }
                | DecodedInstruction::Br { .. }
                | DecodedInstruction::Ret { .. }
                | DecodedInstruction::Brk { .. }
                | DecodedInstruction::Svc { .. }
                | DecodedInstruction::Hvc { .. }
                | DecodedInstruction::Smc { .. }
                | DecodedInstruction::Eret
        )
    }

    pub fn requires_interpreter(&self) -> bool {
        matches!(
            self,
            DecodedInstruction::LdxpU32 { .. }
                | DecodedInstruction::LdxpU64 { .. }
                | DecodedInstruction::LdaxpU32 { .. }
                | DecodedInstruction::LdaxpU64 { .. }
                | DecodedInstruction::StxpU32 { .. }
                | DecodedInstruction::StxpU64 { .. }
                | DecodedInstruction::StlxpU32 { .. }
                | DecodedInstruction::StlxpU64 { .. }
                | DecodedInstruction::SimdSshll { .. }
                | DecodedInstruction::SimdUshll { .. }
                | DecodedInstruction::SimdSshr { .. }
                | DecodedInstruction::SimdUshr { .. }
                | DecodedInstruction::SimdShl { .. }
                | DecodedInstruction::FcvtRoundFromFloat { .. }
                | DecodedInstruction::LdrshPostI32 { .. }
                | DecodedInstruction::LdrshPostI64 { .. }
                | DecodedInstruction::LdrshPreI32 { .. }
                | DecodedInstruction::LdrshPreI64 { .. }
                | DecodedInstruction::LdrsbPostI32 { .. }
                | DecodedInstruction::LdrsbPostI64 { .. }
                | DecodedInstruction::LdrsbPreI32 { .. }
                | DecodedInstruction::LdrsbPreI64 { .. }
                | DecodedInstruction::LdrswPostI64 { .. }
                | DecodedInstruction::LdrswPreI64 { .. }
        )
    }
}

#[derive(Debug, Clone)]
pub struct DecodedBlock {
    pub start_pc: u64,
    pub instructions: smallvec::SmallVec<[DecodedInstruction; 64]>,
    pub fallthrough_pc: u64,
}

pub fn decode_block<F>(
    start_pc: u64,
    max_instructions: usize,
    mut fetch: F,
) -> Result<DecodedBlock, DecodeError>
where
    F: FnMut(u64) -> Option<u32>,
{
    let mut instructions = smallvec::SmallVec::new();
    let mut pc = start_pc;

    for _ in 0..max_instructions.max(1) {
        let opcode = fetch(pc).ok_or(DecodeError::MissingInstruction { pc })?;
        let instruction = decode_instruction(pc, opcode)?;
        let terminal = instruction.is_terminal();
        instructions.push(instruction);
        pc = pc.wrapping_add(4);

        if terminal {
            break;
        }
    }

    Ok(DecodedBlock {
        start_pc,
        instructions,
        fallthrough_pc: pc,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeClass {
    SimdFp,
    SimdStructureLoadStore,
    BranchSystem,
    DataProcessingImmediate,
    DataProcessingRegister,
    LoadStore,
}

#[inline]
fn is_simd_structure_load_store(opcode: u32) -> bool {
    (opcode & 0xbf00_0000 == 0x0c00_0000)
        || (opcode & 0xbf40_c000 == 0x0d40_c000)
        || (opcode & 0xbf00_0000 == 0x0d00_0000)
}

#[inline]
fn classify_decode(opcode: u32) -> Option<DecodeClass> {
    let op1 = ((opcode >> 25) & 0x0f) as u8;

    match op1 {
        0b0111 | 0b1111 => Some(DecodeClass::SimdFp),
        0b0110 => {
            if is_simd_structure_load_store(opcode) {
                Some(DecodeClass::SimdStructureLoadStore)
            } else {
                Some(DecodeClass::LoadStore)
            }
        }
        0b1010 | 0b1011 => Some(DecodeClass::BranchSystem),
        0b1000 | 0b1001 => Some(DecodeClass::DataProcessingImmediate),
        0b0101 | 0b1101 => Some(DecodeClass::DataProcessingRegister),
        0b0100 | 0b1100 | 0b1110 => Some(DecodeClass::LoadStore),
        _ => None,
    }
}

fn decode_instruction(pc: u64, opcode: u32) -> Result<DecodedInstruction, DecodeError> {
    let rd = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;

    // Fast hierarchical dispatch: route by the top opcode buckets so non-SIMD
    // instructions avoid walking the large AdvSIMD/FP decode chain.
    if opcode & 0x7f80_0000 == 0x1380_0000 {
        let sf = ((opcode >> 31) & 1) != 0;
        let n = ((opcode >> 22) & 1) as u8;
        let rm = ((opcode >> 16) & 0x1f) as u8;
        let lsb = ((opcode >> 10) & 0x3f) as u8;

        if (!sf && n != 0) || (sf && n != 1) {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        if !sf && lsb >= 32 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(if sf {
            DecodedInstruction::Extr { rd, rn, rm, lsb }
        } else {
            DecodedInstruction::Extr32 { rd, rn, rm, lsb }
        });
    }

    match classify_decode(opcode) {
        Some(DecodeClass::SimdFp) | Some(DecodeClass::SimdStructureLoadStore) => {
            decode_simd_fp_or_structure(pc, opcode)
        }
        Some(DecodeClass::BranchSystem) => decode_branch_system(pc, opcode),
        Some(DecodeClass::DataProcessingImmediate) => decode_data_processing_immediate(pc, opcode),
        Some(DecodeClass::DataProcessingRegister) => decode_data_processing_register(pc, opcode),
        Some(DecodeClass::LoadStore) => decode_load_store(pc, opcode),
        None => Err(DecodeError::UnsupportedOpcode { pc, opcode }),
    }
}

fn decode_simd_fp_or_structure(pc: u64, opcode: u32) -> Result<DecodedInstruction, DecodeError> {
    let rd = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;
    match opcode & 0xffff_fc00 {
        0x1e22_0000 => return Ok(DecodedInstruction::ScvtfS32 { rd, rn }),
        0x1e23_0000 => return Ok(DecodedInstruction::UcvtfS32 { rd, rn }),
        0x1e63_0000 => return Ok(DecodedInstruction::UcvtfD32 { rd, rn }),
        0x1e62_0000 => return Ok(DecodedInstruction::ScvtfD32 { rd, rn }),
        // 64-bit integer → float conversions (sf=1)
        0x9e22_0000 => return Ok(DecodedInstruction::ScvtfS64 { rd, rn }),
        0x9e23_0000 => return Ok(DecodedInstruction::UcvtfS64 { rd, rn }),
        0x9e62_0000 => return Ok(DecodedInstruction::ScvtfD64 { rd, rn }),
        0x9e63_0000 => return Ok(DecodedInstruction::UcvtfD64 { rd, rn }),
        0x7e21_d800 => return Ok(DecodedInstruction::UcvtfSReg32 { rd, rn }),
        0x5e21_d800 => return Ok(DecodedInstruction::ScvtfSReg32 { rd, rn }),
        0x7e61_d800 => return Ok(DecodedInstruction::UcvtfDReg64 { rd, rn }),
        0x5e61_d800 => return Ok(DecodedInstruction::ScvtfDReg64 { rd, rn }),
        0x1e27_0000 => return Ok(DecodedInstruction::FmovSFromW { rd, rn }),
        0x1e26_0000 => return Ok(DecodedInstruction::FmovWFromS { rd, rn }),
        0x9e67_0000 => return Ok(DecodedInstruction::FmovDFromX { rd, rn }),
        0x9e66_0000 => return Ok(DecodedInstruction::FmovXFromD { rd, rn }),
        0x1e22_c000 => return Ok(DecodedInstruction::FcvtDFromS { rd, rn }),
        0x1e62_4000 => return Ok(DecodedInstruction::FcvtSFromD { rd, rn }),
        0x9e79_0000 => return Ok(DecodedInstruction::FcvtzuXFromD { rd, rn }),
        // FCVTZS (scalar integer): sf 0 0 11110 ftype 1 11 000 000000 Rn Rd
        0x1e38_0000 => return Ok(DecodedInstruction::FcvtzsWFromS { rd, rn }),
        0x9e38_0000 => return Ok(DecodedInstruction::FcvtzsXFromS { rd, rn }),
        0x1e78_0000 => return Ok(DecodedInstruction::FcvtzsWFromD { rd, rn }),
        0x9e78_0000 => return Ok(DecodedInstruction::FcvtzsXFromD { rd, rn }),
        // FCVTZU (scalar integer): sf 0 0 11110 ftype 1 11 001 000000 Rn Rd
        0x1e39_0000 => return Ok(DecodedInstruction::FcvtzuWFromS { rd, rn }),
        0x9e39_0000 => return Ok(DecodedInstruction::FcvtzuXFromS { rd, rn }),
        0x1e79_0000 => return Ok(DecodedInstruction::FcvtzuWFromD { rd, rn }),
        0x9e71_0000 => return Ok(DecodedInstruction::FcvtmuXFromD { rd, rn }),
        0x1e25_4000 => return Ok(DecodedInstruction::FrintmS { rd, rn }),
        0x1e65_4000 => return Ok(DecodedInstruction::FrintmD { rd, rn }),
        // FNEG (scalar): 0 00 11110 ptype 1 000010 10000 Rn Rd
        0x1e21_4000 => return Ok(DecodedInstruction::FnegS { rd, rn }),
        0x1e61_4000 => return Ok(DecodedInstruction::FnegD { rd, rn }),
        // FABS (scalar): 0 00 11110 ptype 1 000001 10000 Rn Rd
        0x1e20_c000 => return Ok(DecodedInstruction::FabsS { rd, rn }),
        0x1e60_c000 => return Ok(DecodedInstruction::FabsD { rd, rn }),
        // FSQRT (scalar): 0 00 11110 ptype 1 000011 10000 Rn Rd
        0x1e21_c000 => return Ok(DecodedInstruction::FsqrtS { rd, rn }),
        0x1e61_c000 => return Ok(DecodedInstruction::FsqrtD { rd, rn }),
        0x4e18_1c00 => return Ok(DecodedInstruction::MovVD1FromX { vd: rd, rn }),
        0x4e01_0c00 => return Ok(DecodedInstruction::DupV16BFromW { vd: rd, rn }),
        0x4e04_0c00 => return Ok(DecodedInstruction::DupV4SFromW { vd: rd, rn }),
        _ => {}
    }

    // FP to integer with rounding mode: sf 0 0 11110 ftype 1 rmode opcode 000000 Rn Rd
    // Covers FCVTNS/NU/PS/PU/MS/MU/AS/AU (rmode=0..2, opcode=0 or 1, or rmode=0 opcode=4/5)
    // Mask: bits 30:29=00, 28:24=11110, 21=1, 15:10=000000
    // Excludes rmode=11 (FCVTZS/FCVTZU already handled above) and rmode=00,opcode=010/011 (SCVTF/UCVTF)
    if opcode & 0x7E20_FC00 == 0x1E20_0000 {
        let sf = ((opcode >> 31) & 1) != 0;
        let ftype = ((opcode >> 22) & 3) as u8;
        let rmode = ((opcode >> 19) & 3) as u8;
        let fp_op = ((opcode >> 16) & 7) as u8;
        // Only handle FP→int conversions (opcode=000 signed, 001 unsigned for rmode 0-2,
        // and opcode=100 FCVTAS, 101 FCVTAU for rmode=0)
        if ftype <= 1
            && rmode < 3
            && (fp_op == 0 || fp_op == 1 || (rmode == 0 && (fp_op == 4 || fp_op == 5)))
        {
            let is_unsigned = (fp_op & 1) != 0;
            let actual_rmode = if fp_op >= 4 {
                // FCVTAS/FCVTAU: rmode=0, opcode=100/101 → treat as rmode=4 (ties away)
                4u8
            } else {
                rmode
            };
            return Ok(DecodedInstruction::FcvtRoundFromFloat {
                rd,
                rn,
                sf,
                ftype,
                rmode: actual_rmode,
                is_unsigned,
            });
        }
    }

    // FCVTZS (fixed-point): sf 0 0 11110 ftype 0 11 000 scale(6) Rn Rd
    if opcode & 0x7f3f_0000 == 0x1e18_0000 {
        let sf = ((opcode >> 31) & 1) != 0;
        let ftype = ((opcode >> 22) & 3) as u8;
        let scale = ((opcode >> 10) & 0x3f) as u8;
        let fbits = (64 - scale) as u8;
        match (sf, ftype) {
            (false, 0) => return Ok(DecodedInstruction::FcvtzsFixedWFromS { rd, rn, fbits }),
            (true, 0) => return Ok(DecodedInstruction::FcvtzsFixedXFromS { rd, rn, fbits }),
            (false, 1) => return Ok(DecodedInstruction::FcvtzsFixedWFromD { rd, rn, fbits }),
            (true, 1) => return Ok(DecodedInstruction::FcvtzsFixedXFromD { rd, rn, fbits }),
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        }
    }

    // FCVTZU (fixed-point): sf 0 0 11110 ftype 0 11 001 scale(6) Rn Rd
    // Mask: top 8 bits = sf_00_11110, bits 21:16 = 0_11_001, scale varies
    if opcode & 0x7f3f_0000 == 0x1e19_0000 {
        let sf = ((opcode >> 31) & 1) != 0;
        let ftype = ((opcode >> 22) & 3) as u8;
        let scale = ((opcode >> 10) & 0x3f) as u8;
        let fbits = (64 - scale) as u8;
        match (sf, ftype) {
            (false, 0) => return Ok(DecodedInstruction::FcvtzuFixedWFromS { rd, rn, fbits }),
            (true, 0) => return Ok(DecodedInstruction::FcvtzuFixedXFromS { rd, rn, fbits }),
            (false, 1) => return Ok(DecodedInstruction::FcvtzuFixedWFromD { rd, rn, fbits }),
            (true, 1) => return Ok(DecodedInstruction::FcvtzuFixedXFromD { rd, rn, fbits }),
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        }
    }

    if opcode & 0xffe0_fc00 == 0x4e00_3c00 {
        let imm5 = ((opcode >> 16) & 0x1f) as u8;
        if imm5 == 0x08 || imm5 == 0x18 {
            let index = (imm5 >> 4) & 1;
            return Ok(DecodedInstruction::MovXFromVD { rd, vn: rn, index });
        }
    }

    if opcode & 0xffe0_fc00 == 0x4e00_1c00 {
        let imm5 = ((opcode >> 16) & 0x1f) as u8;
        let (size, dst_index) = if imm5 & 1 != 0 {
            (0u8, (imm5 >> 1) & 0xf)
        } else if imm5 & 2 != 0 {
            (1u8, (imm5 >> 2) & 0x7)
        } else if imm5 & 4 != 0 {
            (2u8, (imm5 >> 3) & 0x3)
        } else if imm5 & 8 != 0 {
            (3u8, (imm5 >> 4) & 0x1)
        } else {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        };

        return Ok(DecodedInstruction::InsGeneralVector {
            vd: rd,
            rn,
            dst_index,
            size,
        });
    }

    let rm = ((opcode >> 16) & 0x1f) as u8;

    if opcode & 0xbf20_fc00 == 0x0e20_8400 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::AddVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x0e20_bc00 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::AddpVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x2e20_ac00 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::UminpVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x2e20_a400 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::UmaxpVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0x3f80_fc00 == 0x0f00_8400 {
        let q = ((opcode >> 30) & 1) != 0;
        let immh = ((opcode >> 19) & 0xf) as u8;
        let immb = ((opcode >> 16) & 0x7) as u8;

        if immh == 0 || (immh & 0b1000) != 0 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        let size = if (immh & 0b0100) != 0 {
            3
        } else if (immh & 0b0010) != 0 {
            2
        } else {
            1
        };

        let source_bits = 8u32 << u32::from(size);
        let imm = (u32::from(immh) << 3) | u32::from(immb);
        if imm >= source_bits {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        let shift = source_bits - imm;
        if shift == 0 || shift > (source_bits / 2) {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::ShrnVector {
            vd: rd,
            vn: rn,
            q,
            size,
            shift: shift as u8,
        });
    }

    if opcode & 0xbfa0_fc00 == 0x0ea0_1c00 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::OrrVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
        });
    }

    if opcode & 0xbfa0_fc00 == 0x0e20_1c00 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::AndVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
        });
    }

    if opcode & 0xbf20_fc00 == 0x0e20_9800 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::CmeqVectorZero {
            vd: rd,
            vn: rn,
            q,
            size,
        });
    }

    // NEG Vd.<T>, Vn.<T> — negate vector elements
    // Encoding: 0 Q 1 01110 size 10000 01011 10 Rn Rd
    if opcode & 0xbf3f_fc00 == 0x2e20_b800 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        return Ok(DecodedInstruction::NegVector {
            vd: rd,
            vn: rn,
            q,
            size,
        });
    }

    // INS (element) — copy element between vector registers
    // Encoding: 0 1 1 01110 000 imm5 0 imm4 1 Rn Rd
    if opcode & 0xffe0_8400 == 0x6e00_0400 {
        let imm5 = ((opcode >> 16) & 0x1f) as u8;
        let imm4 = ((opcode >> 11) & 0xf) as u8;
        let (size, dst_index, src_index) = if imm5 & 1 != 0 {
            (0u8, (imm5 >> 1) & 0xf, imm4 & 0xf)
        } else if imm5 & 2 != 0 {
            (1u8, (imm5 >> 2) & 0x7, (imm4 >> 1) & 0x7)
        } else if imm5 & 4 != 0 {
            (2u8, (imm5 >> 3) & 0x3, (imm4 >> 2) & 0x3)
        } else if imm5 & 8 != 0 {
            (3u8, (imm5 >> 4) & 0x1, (imm4 >> 3) & 0x1)
        } else {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        };
        return Ok(DecodedInstruction::InsElement {
            vd: rd,
            vn: rn,
            dst_index,
            src_index,
            size,
        });
    }

    // DUP (general) Vd.<T>, Rn — broadcast GPR to all vector lanes
    // Encoding: 0 Q 0 01110 000 imm5 0 00011 Rn Rd
    // imm5 encodes element size: bit0→B, bit1→H, bit2→S, bit3→D
    if opcode & 0xbfe0_fc00 == 0x0e00_0c00 {
        let q = ((opcode >> 30) & 1) != 0;
        let imm5 = ((opcode >> 16) & 0x1f) as u8;
        let size = if imm5 & 1 != 0 {
            0u8
        } else if imm5 & 2 != 0 {
            1u8
        } else if imm5 & 4 != 0 {
            2u8
        } else if imm5 & 8 != 0 {
            3u8
        } else {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        };
        return Ok(DecodedInstruction::DupGeneralVector {
            vd: rd,
            rn,
            q,
            size,
        });
    }

    // DUP (element) Vd.<T>, Vn.<Ts>[index] — broadcast vector element to all lanes
    // Encoding: 0 Q 0 01110 000 imm5 0 00001 Rn Rd
    if opcode & 0xbfe0_fc00 == 0x0e00_0400 {
        let q = ((opcode >> 30) & 1) != 0;
        let imm5 = ((opcode >> 16) & 0x1f) as u8;
        let (size, index) = if imm5 & 1 != 0 {
            (0u8, (imm5 >> 1) & 0xf)
        } else if imm5 & 2 != 0 {
            (1u8, (imm5 >> 2) & 0x7)
        } else if imm5 & 4 != 0 {
            (2u8, (imm5 >> 3) & 0x3)
        } else if imm5 & 8 != 0 {
            (3u8, (imm5 >> 4) & 0x1)
        } else {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        };
        return Ok(DecodedInstruction::DupElementVector {
            vd: rd,
            vn: rn,
            q,
            size,
            index,
        });
    }

    // REV64 Vd.<T>, Vn.<T> — reverse elements within 64-bit doublewords
    // Encoding: 0 Q 0 01110 size 10000 00000 10 Rn Rd
    if opcode & 0xbf3f_fc00 == 0x0e20_0800 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        return Ok(DecodedInstruction::Rev64Vector {
            vd: rd,
            vn: rn,
            q,
            size,
        });
    }

    // CNT Vd.<T>, Vn.<T> — pop-count per byte
    // Encoding: 0 Q 0 01110 00 10000 00101 10 Rn Rd
    if opcode & 0xbfff_fc00 == 0x0e20_5800 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::CntVector { vd: rd, vn: rn, q });
    }

    // NOT (MVN) Vd.<T>, Vn.<T> — bitwise NOT
    // Encoding: 0 Q 1 01110 00 10000 00101 10 Rn Rd  (U=1, size=00, opcode=00101)
    if opcode & 0xbfff_fc00 == 0x2e20_5800 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::NotVector { vd: rd, vn: rn, q });
    }

    // RBIT Vd.<T>, Vn.<T> — reverse bits per byte
    // Encoding: 0 Q 1 01110 01 10000 00101 10 Rn Rd  (U=1, size=01, opcode=00101)
    if opcode & 0xbfff_fc00 == 0x2e60_5800 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::RbitVector { vd: rd, vn: rn, q });
    }

    // UADDLV — Unsigned Add Long Across Vector
    // Encoding: 0 Q 1 01110 size 11000 00011 10 Rn Rd
    if opcode & 0xbf3f_fc00 == 0x2e30_3800 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if size == 3 || (!q && size == 2) {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        return Ok(DecodedInstruction::UaddlvVector {
            vd: rd,
            vn: rn,
            q,
            size,
        });
    }

    // SADDLV — Signed Add Long Across Vector
    // Encoding: 0 Q 0 01110 size 11000 00011 10 Rn Rd
    if opcode & 0xbf3f_fc00 == 0x0e30_3800 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if size == 3 || (!q && size == 2) {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        return Ok(DecodedInstruction::SaddlvVector {
            vd: rd,
            vn: rn,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x2e20_8c00 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::CmeqVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x2e20_3c00 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::CmhsVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    // CMHI (unsigned >) three-same: U=1, opcode=00110
    if opcode & 0xbf20_fc00 == 0x2e20_3400 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        return Ok(DecodedInstruction::CmhiVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    // USHL Vd.<T>, Vn.<T>, Vm.<T> — unsigned shift left (variable per-element)
    // Encoding: 0 Q 1 01110 size 1 Rm 01000 1 Rn Rd
    if opcode & 0xbf20_fc00 == 0x2e20_4400 {
        let q = ((opcode >> 30) & 1) != 0;
        let size = ((opcode >> 22) & 0x3) as u8;
        if !q && size == 3 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }
        return Ok(DecodedInstruction::UshlVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
            size,
        });
    }

    if opcode & 0xbf20_fc00 == 0x2e20_1c00 {
        let q = ((opcode >> 30) & 1) != 0;
        return Ok(DecodedInstruction::BifVector {
            vd: rd,
            vn: rn,
            vm: rm,
            q,
        });
    }

    // AdvSIMD FP two-register miscellaneous: 0 Q U 01110 size(2) 10000 opcode(5) 10 Rn Rd
    // bits21:17 = 10000 (fixed), bits11:10 = 10
    if opcode & 0x9f3e_0c00 == 0x0e20_0800 {
        let q = ((opcode >> 30) & 1) != 0;
        let u = ((opcode >> 29) & 1) as u8;
        let size = ((opcode >> 22) & 3) as u8;
        let opcode5 = ((opcode >> 12) & 0x1f) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let rd = (opcode & 0x1f) as u8;

        // FP ops have size<1>=1 (bit23=1), size<0>=sz
        let s1 = (size >> 1) & 1; // bit23
        let sz = size & 1; // bit22

        // Reject Q=0, sz=1 (no 64-bit lanes in 64-bit register)
        if !q && sz == 1 && s1 == 1 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        macro_rules! vu {
            ($V:ident) => {
                DecodedInstruction::$V {
                    vd: rd,
                    vn: rn,
                    q,
                    sz,
                }
            };
        }
        let fp_inst = match (u, s1, opcode5) {
            // U=0, s1=0: integer-to-FP and rounding
            (0, 0, 0b11000) => Some(vu!(VecFrintn)),
            (0, 0, 0b11001) => Some(vu!(VecFrintm)),
            (0, 0, 0b11010) => Some(vu!(VecFcvtns)),
            (0, 0, 0b11011) => Some(vu!(VecFcvtms)),
            (0, 0, 0b11100) => Some(vu!(VecFcvtas)),
            (0, 0, 0b11101) => Some(vu!(VecScvtf)),
            // U=0, s1=1: FP misc
            (0, 1, 0b01100) => Some(vu!(VecFcmgt0)),
            (0, 1, 0b01101) => Some(vu!(VecFcmeq0)),
            (0, 1, 0b01110) => Some(vu!(VecFcmlt0)),
            (0, 1, 0b01111) => Some(vu!(VecFabs)),
            (0, 1, 0b11000) => Some(vu!(VecFrintp)),
            (0, 1, 0b11001) => Some(vu!(VecFrintz)),
            (0, 1, 0b11010) => Some(vu!(VecFcvtps)),
            (0, 1, 0b11011) => Some(vu!(VecFcvtzs)),
            (0, 1, 0b11101) => Some(vu!(VecFrecpe)),
            // U=1, s1=0: rounding and unsigned convert
            (1, 0, 0b11000) => Some(vu!(VecFrinta)),
            (1, 0, 0b11001) => Some(vu!(VecFrintx)),
            (1, 0, 0b11010) => Some(vu!(VecFcvtnu)),
            (1, 0, 0b11011) => Some(vu!(VecFcvtmu)),
            (1, 0, 0b11100) => Some(vu!(VecFcvtau)),
            (1, 0, 0b11101) => Some(vu!(VecUcvtf)),
            // U=1, s1=1: FP misc
            (1, 1, 0b01100) => Some(vu!(VecFcmge0)),
            (1, 1, 0b01101) => Some(vu!(VecFcmle0)),
            (1, 1, 0b01111) => Some(vu!(VecFneg)),
            (1, 1, 0b11000) => Some(vu!(VecFrinti)),
            (1, 1, 0b11010) => Some(vu!(VecFcvtpu)),
            (1, 1, 0b11011) => Some(vu!(VecFcvtzu)),
            (1, 1, 0b11101) => Some(vu!(VecFrsqrte)),
            (1, 1, 0b11111) => Some(vu!(VecFsqrt)),
            _ => None,
        };

        if let Some(inst) = fp_inst {
            return Ok(inst);
        }
        // Fall through for non-FP two-reg misc or unimplemented ones
    }

    // AdvSIMD FP three-same (vector): 0 Q U 01110 s1:sz 1 Rm opcode(5) 1 Rn Rd
    // opcode bits 15:14 = 11, bit31=0, bits28:24=01110, bit21=1, bit10=1
    if opcode & 0x9f20_c400 == 0x0e20_c400 {
        let q = ((opcode >> 30) & 1) != 0;
        let u = ((opcode >> 29) & 1) as u8;
        let s1 = ((opcode >> 23) & 1) as u8;
        let sz = ((opcode >> 22) & 1) as u8;
        let rm = ((opcode >> 16) & 0x1f) as u8;
        let opcode3 = ((opcode >> 11) & 0x7) as u8; // bits 13:11
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let rd = (opcode & 0x1f) as u8;

        // Reject Q=0, sz=1 (scalar D lane only for Q=1)
        if !q && sz == 1 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        macro_rules! v {
            ($V:ident) => {
                DecodedInstruction::$V {
                    vd: rd,
                    vn: rn,
                    vm: rm,
                    q,
                    sz,
                }
            };
        }
        let inst = match (u, s1, opcode3) {
            (0, 0, 0b000) => v!(VecFmaxnm),
            (0, 0, 0b001) => v!(VecFmla),
            (0, 0, 0b010) => v!(VecFadd),
            (0, 0, 0b011) => v!(VecFmulx),
            (0, 0, 0b100) => v!(VecFcmeq),
            (0, 0, 0b110) => v!(VecFmax),
            (0, 0, 0b111) => v!(VecFrecps),
            (0, 1, 0b000) => v!(VecFminnm),
            (0, 1, 0b001) => v!(VecFmls),
            (0, 1, 0b010) => v!(VecFsub),
            (0, 1, 0b110) => v!(VecFmin),
            (0, 1, 0b111) => v!(VecFrsqrts),
            (1, 0, 0b010) => v!(VecFaddp),
            (1, 0, 0b011) => v!(VecFmul),
            (1, 0, 0b100) => v!(VecFcmge),
            (1, 0, 0b101) => v!(VecFacge),
            (1, 0, 0b110) => v!(VecFmaxp),
            (1, 0, 0b111) => v!(VecFdiv),
            (1, 1, 0b010) => v!(VecFabd),
            (1, 1, 0b100) => v!(VecFcmgt),
            (1, 1, 0b101) => v!(VecFacgt),
            (1, 1, 0b110) => v!(VecFminp),
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        };

        return Ok(inst);
    }

    // AdvSIMD vector × indexed element (FP): 0 Q op 01111 size L M Rm opcode(4) H 0 Rn Rd
    // bit31=0, bits27:24=1111, bit10=0
    if opcode & 0x9f00_0400 == 0x0f00_0000 {
        let q = ((opcode >> 30) & 1) != 0;
        let u = ((opcode >> 29) & 1) as u8;
        let size = ((opcode >> 22) & 3) as u8;
        let l_bit = ((opcode >> 21) & 1) as u8;
        let m_bit = ((opcode >> 20) & 1) as u8;
        let rm4 = ((opcode >> 16) & 0xf) as u8;
        let opcode4 = ((opcode >> 12) & 0xf) as u8;
        let h_bit = ((opcode >> 11) & 1) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let rd = (opcode & 0x1f) as u8;

        // FP ops require size=1x (bit23=1)
        if size >= 2 {
            let sz = size & 1; // 0→f32, 1→f64
            let (vm, index) = if sz == 0 {
                // f32: Vm = M:Rm, index = H:L
                ((m_bit << 4) | rm4, (h_bit << 1) | l_bit)
            } else {
                // f64: Vm = M:Rm, index = H
                ((m_bit << 4) | rm4, h_bit)
            };

            // Reject Q=0, sz=1 for vector variant (no .1D arrangement)
            if !q && sz == 1 {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            macro_rules! ve {
                ($V:ident) => {
                    DecodedInstruction::$V {
                        vd: rd,
                        vn: rn,
                        vm,
                        index,
                        q,
                        sz,
                    }
                };
            }
            let fp_elem_inst = match (u, opcode4) {
                (0, 0b0001) => Some(ve!(VecFmlaElem)),
                (0, 0b0101) => Some(ve!(VecFmlsElem)),
                (0, 0b1001) => Some(ve!(VecFmulElem)),
                (1, 0b1001) => Some(ve!(VecFmulxElem)),
                _ => None,
            };

            if let Some(inst) = fp_elem_inst {
                return Ok(inst);
            }
        }
        // Fall through for integer by-element or unimplemented
    }

    match opcode & 0xffe0_fc00 {
        0x1e20_2800 => return Ok(DecodedInstruction::FaddS { rd, rn, rm }),
        0x1e20_3800 => return Ok(DecodedInstruction::FsubS { rd, rn, rm }),
        0x1e20_0800 => return Ok(DecodedInstruction::FmulS { rd, rn, rm }),
        0x1e20_4800 => return Ok(DecodedInstruction::FmaxS { rd, rn, rm }),
        0x1e20_5800 => return Ok(DecodedInstruction::FminS { rd, rn, rm }),
        0x1e20_6800 => return Ok(DecodedInstruction::FmaxnmS { rd, rn, rm }),
        0x1e20_7800 => return Ok(DecodedInstruction::FminnmS { rd, rn, rm }),
        0x1e20_1800 => return Ok(DecodedInstruction::FdivS { rd, rn, rm }),
        0x1e60_2800 => return Ok(DecodedInstruction::FaddD { rd, rn, rm }),
        0x1e60_3800 => return Ok(DecodedInstruction::FsubD { rd, rn, rm }),
        0x1e60_0800 => return Ok(DecodedInstruction::FmulD { rd, rn, rm }),
        0x1e60_4800 => return Ok(DecodedInstruction::FmaxD { rd, rn, rm }),
        0x1e60_5800 => return Ok(DecodedInstruction::FminD { rd, rn, rm }),
        0x1e60_6800 => return Ok(DecodedInstruction::FmaxnmD { rd, rn, rm }),
        0x1e60_7800 => return Ok(DecodedInstruction::FminnmD { rd, rn, rm }),
        0x1e60_1800 => return Ok(DecodedInstruction::FdivD { rd, rn, rm }),
        0x1e20_8800 => return Ok(DecodedInstruction::FnmulS { rd, rn, rm }),
        0x1e60_8800 => return Ok(DecodedInstruction::FnmulD { rd, rn, rm }),
        _ => {}
    }

    if opcode & 0xffe0_1fe0 == 0x1e20_1000 {
        let imm8 = ((opcode >> 13) & 0xff) as u8;
        return Ok(DecodedInstruction::FmovImmS { rd, imm8 });
    }

    if opcode & 0xffe0_1fe0 == 0x1e60_1000 {
        let imm8 = ((opcode >> 13) & 0xff) as u8;
        return Ok(DecodedInstruction::FmovImmD { rd, imm8 });
    }

    // AdvSIMD modified immediate: MOVI / MVNI / ORR imm / BIC imm / FMOV imm
    // Encoding: 0 Q op 0111100000 abc cmode o2 1 defgh Rd
    if opcode & 0x9ff8_0400 == 0x0f00_0400 {
        let q = ((opcode >> 30) & 1) != 0;
        let op_bit = ((opcode >> 29) & 1) as u8;
        let abc = ((opcode >> 16) & 0x7) as u8;
        let cmode = ((opcode >> 12) & 0xf) as u8;
        let o2 = ((opcode >> 11) & 1) as u8;
        let defgh = ((opcode >> 5) & 0x1f) as u8;
        let imm8 = ((abc as u16) << 5 | defgh as u16) as u8;
        let vd = rd;

        // Keep the legacy zero variants for backward compat
        if imm8 == 0 && op_bit == 1 && cmode == 0xe && o2 == 0 {
            return Ok(DecodedInstruction::MoviV2DZero { vd });
        }
        if imm8 == 0 && op_bit == 0 && cmode == 0x0 && o2 == 0 && q {
            return Ok(DecodedInstruction::MoviV4SZero { vd });
        }

        // Compute the replicated immediate pattern for one 64-bit half
        let (imm64, is_orr, is_bic) = decode_advsimd_modified_imm(op_bit, cmode, o2, imm8)?;
        let imm_lo = imm64;
        let imm_hi = if q { imm64 } else { 0 };

        if is_orr {
            return Ok(DecodedInstruction::OrrImmVector { vd, imm_lo, imm_hi });
        }
        if is_bic {
            return Ok(DecodedInstruction::BicImmVector { vd, imm_lo, imm_hi });
        }

        // MVNI variants
        if op_bit == 1 && cmode != 0xe && cmode != 0xf {
            return Ok(DecodedInstruction::MvniVector { vd, imm_lo, imm_hi });
        }

        return Ok(DecodedInstruction::MoviVector { vd, imm_lo, imm_hi });
    }

    // AdvSIMD shift by immediate: 0 Q U 011110 immh immb opcode 1 Rn Rd
    // immh != 0 distinguishes from modified-immediate above.
    if opcode & 0x9f00_0400 == 0x0f00_0400 {
        let immh = ((opcode >> 19) & 0xf) as u8;
        if immh != 0 {
            let q = ((opcode >> 30) & 1) != 0;
            let u = ((opcode >> 29) & 1) != 0;
            let immb = ((opcode >> 16) & 0x7) as u8;
            let simd_op = ((opcode >> 11) & 0x1f) as u8;
            let rn = ((opcode >> 5) & 0x1f) as u8;
            let rd = (opcode & 0x1f) as u8;

            // Determine source element size from immh.
            let (size, esize): (u8, u8) = if immh & 0b1000 != 0 {
                (3, 64)
            } else if immh & 0b0100 != 0 {
                (2, 32)
            } else if immh & 0b0010 != 0 {
                (1, 16)
            } else {
                (0, 8)
            };

            let immh_immb = ((immh as u16) << 3) | (immb as u16);

            match (u, simd_op) {
                // SSHR: shift = 2*esize - immh_immb
                (false, 0b00000) => {
                    let shift = ((esize as u16) * 2 - immh_immb) as u8;
                    return Ok(DecodedInstruction::SimdSshr {
                        q,
                        size,
                        shift,
                        rd,
                        rn,
                    });
                }
                // USHR: shift = 2*esize - immh_immb
                (true, 0b00000) => {
                    let shift = ((esize as u16) * 2 - immh_immb) as u8;
                    return Ok(DecodedInstruction::SimdUshr {
                        q,
                        size,
                        shift,
                        rd,
                        rn,
                    });
                }
                // SHL: shift = immh_immb - esize
                (false, 0b01010) => {
                    let shift = (immh_immb - esize as u16) as u8;
                    return Ok(DecodedInstruction::SimdShl {
                        q,
                        size,
                        shift,
                        rd,
                        rn,
                    });
                }
                // SSHLL / SSHLL2: shift = immh_immb - esize
                (false, 0b10100) => {
                    let shift = (immh_immb - esize as u16) as u8;
                    return Ok(DecodedInstruction::SimdSshll {
                        q,
                        size,
                        shift,
                        rd,
                        rn,
                    });
                }
                // USHLL / USHLL2: shift = immh_immb - esize
                (true, 0b10100) => {
                    let shift = (immh_immb - esize as u16) as u8;
                    return Ok(DecodedInstruction::SimdUshll {
                        q,
                        size,
                        shift,
                        rd,
                        rn,
                    });
                }
                _ => {} // fall through to unsupported
            }
        }
    }

    if opcode & 0xffe0_fc07 == 0x1e20_2000 {
        return Ok(DecodedInstruction::FcmpS { rn, rm });
    }

    if opcode & 0xffe0_fc07 == 0x1e60_2000 {
        return Ok(DecodedInstruction::FcmpD { rn, rm });
    }

    // FCSEL (single/double)
    if opcode & 0xffe0_0c00 == 0x1e20_0c00 {
        let condition = BranchCondition::decode(((opcode >> 12) & 0xf) as u8);
        return Ok(DecodedInstruction::FcselS {
            rd,
            rn,
            rm,
            condition,
        });
    }
    if opcode & 0xffe0_0c00 == 0x1e60_0c00 {
        let condition = BranchCondition::decode(((opcode >> 12) & 0xf) as u8);
        return Ok(DecodedInstruction::FcselD {
            rd,
            rn,
            rm,
            condition,
        });
    }

    // AdvSIMD load/store multiple structures
    // No post-index:  0 Q 001100 0 L 000000 opcode size Rn Rt
    // Post-index:     0 Q 001100 1 L Rm     opcode size Rn Rt
    if opcode & 0xbf00_0000 == 0x0c00_0000 {
        let q = ((opcode >> 30) & 1) != 0;
        let is_post_index = ((opcode >> 23) & 1) != 0;
        let load = ((opcode >> 22) & 1) != 0;
        let rm_field = ((opcode >> 16) & 0x1f) as u8;
        let simd_opcode = ((opcode >> 12) & 0xf) as u8;
        let size = ((opcode >> 10) & 3) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let vt = (opcode & 0x1f) as u8;

        // Non-post-index requires bits [21:16] = 000000
        if !is_post_index && rm_field != 0 {
            // invalid encoding, fall through
        } else {
            let (nregs, interleave) = match simd_opcode {
                0b0000 => (4u8, true), // LD/ST4
                0b0010 => (4, false),  // LD/ST1 x4
                0b0100 => (3, true),   // LD/ST3
                0b0110 => (3, false),  // LD/ST1 x3
                0b0111 => (1, false),  // LD/ST1 x1
                0b1000 => (2, true),   // LD/ST2
                0b1010 => (2, false),  // LD/ST1 x2
                _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
            };
            let post_index = if is_post_index { Some(rm_field) } else { None };
            if load {
                return Ok(DecodedInstruction::SimdLdMulti {
                    q,
                    nregs,
                    interleave,
                    size,
                    rn,
                    vt,
                    post_index,
                });
            } else {
                return Ok(DecodedInstruction::SimdStMulti {
                    q,
                    nregs,
                    interleave,
                    size,
                    rn,
                    vt,
                    post_index,
                });
            }
        }
    }

    // AdvSIMD load single structure and replicate (LD1R/LD2R/LD3R/LD4R)
    // Non-post: 0 Q 001101 0 1 R 00000 11x S size Rn Rt
    // Post:     0 Q 001101 1 1 R mmmmm 11x S size Rn Rt
    // Match: bit31=0, bits29:24=001101, bit22=1(load), bits15:14=11
    if opcode & 0xbf40_c000 == 0x0d40_c000 {
        let q = ((opcode >> 30) & 1) != 0;
        let is_post_index = ((opcode >> 23) & 1) != 0;
        let r_bit = ((opcode >> 21) & 1) as u8;
        let rm_field = ((opcode >> 16) & 0x1f) as u8;
        let opcode_lsb = ((opcode >> 13) & 1) as u8;
        let s_bit = ((opcode >> 12) & 1) as u8;
        let size = ((opcode >> 10) & 3) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let vt = (opcode & 0x1f) as u8;

        // LD1R: R=0, opc_lsb=0; LD2R: R=1, opc_lsb=0
        // LD3R: R=0, opc_lsb=1; LD4R: R=1, opc_lsb=1
        let nregs = match (r_bit, opcode_lsb) {
            (0, 0) => 1u8,
            (1, 0) => 2,
            (0, 1) => 3,
            (1, 1) => 4,
            _ => unreachable!(),
        };

        // S must be 0 for replicate
        if s_bit != 0 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        // Non-post-index requires rm_field = 00000
        if !is_post_index && rm_field != 0 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        let post_index = if is_post_index { Some(rm_field) } else { None };
        return Ok(DecodedInstruction::SimdLdReplicate {
            q,
            vt,
            rn,
            size,
            nregs,
            post_index,
        });
    }

    // AdvSIMD load/store single structure to one lane
    // Non-post: 0 Q 001101 0 L R 00000 opcode S size Rn Rt
    // Post:     0 Q 001101 1 L R mmmmm opcode S size Rn Rt
    // bits[31]=0, bits[29:24]=001101, bits[15:14]!=11 (replicate handled above)
    if opcode & 0xbf00_0000 == 0x0d00_0000 {
        let q = ((opcode >> 30) & 1) != 0;
        let is_post_index = ((opcode >> 23) & 1) != 0;
        let load = ((opcode >> 22) & 1) != 0;
        let r_bit = ((opcode >> 21) & 1) as u8;
        let rm_field = ((opcode >> 16) & 0x1f) as u8;
        let opcode3 = ((opcode >> 13) & 0x7) as u8;
        let s_bit = ((opcode >> 12) & 1) as u8;
        let size_field = ((opcode >> 10) & 3) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let vt = (opcode & 0x1f) as u8;

        // Non-post-index requires rm_field = 00000
        if !is_post_index && rm_field != 0 {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        }

        // Decode element size and index from opcode3/S/size_field
        // R determines nregs: R=0 → 1 or 3 regs, R=1 → 2 or 4 regs
        let (elem_size, index, nregs) = match opcode3 {
            // opcode=000: B[index], nregs=1(R=0) or 2(R=1)
            0b000 => {
                let idx = (q as u8) << 3 | (s_bit << 2) | size_field;
                (0u8, idx, if r_bit == 0 { 1u8 } else { 2 })
            }
            // opcode=001: B[index], nregs=3(R=0) or 4(R=1)
            0b001 => {
                let idx = (q as u8) << 3 | (s_bit << 2) | size_field;
                (0u8, idx, if r_bit == 0 { 3u8 } else { 4 })
            }
            // opcode=010: H[index], nregs=1(R=0) or 2(R=1)
            0b010 => {
                if size_field & 1 != 0 {
                    return Err(DecodeError::UnsupportedOpcode { pc, opcode });
                }
                let idx = (q as u8) << 2 | (s_bit << 1) | (size_field >> 1);
                (1u8, idx, if r_bit == 0 { 1u8 } else { 2 })
            }
            // opcode=011: H[index], nregs=3(R=0) or 4(R=1)
            0b011 => {
                if size_field & 1 != 0 {
                    return Err(DecodeError::UnsupportedOpcode { pc, opcode });
                }
                let idx = (q as u8) << 2 | (s_bit << 1) | (size_field >> 1);
                (1u8, idx, if r_bit == 0 { 3u8 } else { 4 })
            }
            // opcode=100: S or D[index], nregs=1(R=0) or 2(R=1)
            0b100 => {
                if s_bit == 0 && (size_field & 1) == 0 {
                    // S[index]
                    let idx = (q as u8) << 1 | (size_field >> 1);
                    (2u8, idx, if r_bit == 0 { 1u8 } else { 2 })
                } else if s_bit == 1 && size_field == 0 {
                    // D[index]
                    let idx = q as u8;
                    (3u8, idx, if r_bit == 0 { 1u8 } else { 2 })
                } else {
                    return Err(DecodeError::UnsupportedOpcode { pc, opcode });
                }
            }
            // opcode=101: S or D[index], nregs=3(R=0) or 4(R=1)
            0b101 => {
                if s_bit == 0 && (size_field & 1) == 0 {
                    let idx = (q as u8) << 1 | (size_field >> 1);
                    (2u8, idx, if r_bit == 0 { 3u8 } else { 4 })
                } else if s_bit == 1 && size_field == 0 {
                    let idx = q as u8;
                    (3u8, idx, if r_bit == 0 { 3u8 } else { 4 })
                } else {
                    return Err(DecodeError::UnsupportedOpcode { pc, opcode });
                }
            }
            // opcode=110,111 already handled by LdReplicate above
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        };

        let post_index = if is_post_index { Some(rm_field) } else { None };
        if load {
            return Ok(DecodedInstruction::SimdLdSingle {
                q,
                vt,
                rn,
                size: elem_size,
                index,
                nregs,
                post_index,
            });
        } else {
            return Ok(DecodedInstruction::SimdStSingle {
                q,
                vt,
                rn,
                size: elem_size,
                index,
                nregs,
                post_index,
            });
        }
    }

    // AdvSIMD three-different: 0 Q U 01110 sz 0 Rm opcode(4) 00 Rn Rd
    // bit 31=0, bits 28:24=01110, bit 21=0, bits 11:10=00
    if opcode & 0x9f200c00 == 0x0e000000 {
        let q = ((opcode >> 30) & 1) != 0;
        let u = ((opcode >> 29) & 1) != 0;
        let size = ((opcode >> 22) & 3) as u8;
        let rm = ((opcode >> 16) & 0x1f) as u8;
        let simd_opcode = ((opcode >> 12) & 0xf) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let rd = (opcode & 0x1f) as u8;
        return Ok(match (u, simd_opcode) {
            (false, 0) => DecodedInstruction::SimdSaddl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 0) => DecodedInstruction::SimdUaddl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 1) => DecodedInstruction::SimdSaddw {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 1) => DecodedInstruction::SimdUaddw {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 2) => DecodedInstruction::SimdSsubl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 2) => DecodedInstruction::SimdUsubl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 3) => DecodedInstruction::SimdSsubw {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 3) => DecodedInstruction::SimdUsubw {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 4) => DecodedInstruction::SimdAddhn {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 4) => DecodedInstruction::SimdRaddhn {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 6) => DecodedInstruction::SimdSubhn {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 6) => DecodedInstruction::SimdRsubhn {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 7) => DecodedInstruction::SimdSabdl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 7) => DecodedInstruction::SimdUabdl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 8) => DecodedInstruction::SimdSmlal {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 8) => DecodedInstruction::SimdUmlal {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 10) => DecodedInstruction::SimdSmlsl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 10) => DecodedInstruction::SimdUmlsl {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (false, 12) => DecodedInstruction::SimdSmull {
                q,
                size,
                rd,
                rn,
                rm,
            },
            (true, 12) => DecodedInstruction::SimdUmull {
                q,
                size,
                rd,
                rn,
                rm,
            },
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        });
    }

    Err(DecodeError::UnsupportedOpcode { pc, opcode })
}
fn decode_branch_system(pc: u64, opcode: u32) -> Result<DecodedInstruction, DecodeError> {
    let rt = (opcode & 0x1f) as u8;

    if opcode & 0xffff_f01f == 0xd503_201f {
        let imm7 = ((opcode >> 5) & 0x7f) as u8;
        return Ok(match imm7 {
            0x00 => DecodedInstruction::Nop,
            0x01 => DecodedInstruction::Yield,
            0x02 => DecodedInstruction::Wfe,
            0x03 => DecodedInstruction::Wfi,
            0x04 => DecodedInstruction::Sev,
            0x05 => DecodedInstruction::Sevl,
            0x07 => DecodedInstruction::Xpaclri,
            0x20 | 0x22 | 0x24 | 0x26 => DecodedInstruction::Bti { btype: imm7 & 0x0f },
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        });
    }

    match opcode & 0xffff_fc1f {
        0xd61f_0000 => {
            let rn = ((opcode >> 5) & 0x1f) as u8;
            return Ok(DecodedInstruction::Br { rn });
        }
        0xd63f_0000 => {
            let rn = ((opcode >> 5) & 0x1f) as u8;
            return Ok(DecodedInstruction::Blr { rn });
        }
        0xd65f_0000 => {
            let rn = ((opcode >> 5) & 0x1f) as u8;
            return Ok(DecodedInstruction::Ret { rn });
        }
        _ => {}
    }

    let imm16 = ((opcode >> 5) & 0xffff) as u16;
    match opcode & 0xffe0_001f {
        0xd420_0000 => return Ok(DecodedInstruction::Brk { imm16 }),
        0xd400_0001 => return Ok(DecodedInstruction::Svc { imm16 }),
        0xd400_0002 => return Ok(DecodedInstruction::Hvc { imm16 }),
        0xd400_0003 => return Ok(DecodedInstruction::Smc { imm16 }),
        _ => {}
    }

    match opcode & 0xffff_ffe0 {
        0xd50b_7e20 => return Ok(DecodedInstruction::DcCivac { rt }),
        0xd50b_7420 => return Ok(DecodedInstruction::DcZva { rt }),
        _ => {}
    }

    let option = ((opcode >> 8) & 0xf) as u8;
    match opcode & 0xffff_f0ff {
        0xd503_30bf => return Ok(DecodedInstruction::Dmb { option }),
        0xd503_309f => return Ok(DecodedInstruction::Dsb { option }),
        0xd503_30df => return Ok(DecodedInstruction::Isb { option }),
        0xd503_305f => return Ok(DecodedInstruction::Clrex { option }),
        _ => {}
    }

    if opcode == 0xd69f_03e0 {
        return Ok(DecodedInstruction::Eret);
    }

    match opcode & 0xfff0_0000 {
        0xd530_0000 => {
            let raw = ((opcode >> 5) & 0x7fff) as u16;
            if let Some(reg) = SystemRegister::from_encoding(raw) {
                return Ok(DecodedInstruction::Mrs { rt, reg });
            }
        }
        0xd510_0000 => {
            let raw = ((opcode >> 5) & 0x7fff) as u16;
            if let Some(reg) = SystemRegister::from_encoding(raw) {
                return Ok(DecodedInstruction::Msr { reg, rt });
            }
        }
        _ => {}
    }

    match opcode & 0xfc00_0000 {
        0x1400_0000 => {
            let imm26 = opcode & 0x03ff_ffff;
            let signed = sign_extend_26(imm26) << 2;
            let target = ((pc as i64).wrapping_add(signed)) as u64;
            return Ok(DecodedInstruction::B { target });
        }
        0x9400_0000 => {
            let imm26 = opcode & 0x03ff_ffff;
            let signed = sign_extend_26(imm26) << 2;
            let target = ((pc as i64).wrapping_add(signed)) as u64;
            return Ok(DecodedInstruction::Bl { target });
        }
        _ => {}
    }

    if opcode & 0xff00_0010 == 0x5400_0000 {
        let imm19 = (opcode >> 5) & 0x7ffff;
        let signed = sign_extend_19(imm19) << 2;
        let target = ((pc as i64).wrapping_add(signed)) as u64;
        let condition = BranchCondition::decode((opcode & 0xf) as u8);
        return Ok(DecodedInstruction::BCond { target, condition });
    }

    let cb_class = opcode & 0xff00_0000;
    if matches!(
        cb_class,
        0xb400_0000 | 0xb500_0000 | 0x3400_0000 | 0x3500_0000
    ) {
        let imm19 = (opcode >> 5) & 0x7ffff;
        let signed = sign_extend_19(imm19) << 2;
        let target = ((pc as i64).wrapping_add(signed)) as u64;
        return Ok(match cb_class {
            0xb400_0000 => DecodedInstruction::Cbz { rt, target },
            0xb500_0000 => DecodedInstruction::Cbnz { rt, target },
            0x3400_0000 => DecodedInstruction::Cbz32 { rt, target },
            0x3500_0000 => DecodedInstruction::Cbnz32 { rt, target },
            _ => unreachable!(),
        });
    }

    let tb_class = opcode & 0x7f00_0000;
    if matches!(tb_class, 0x3600_0000 | 0x3700_0000) {
        let b40 = ((opcode >> 19) & 0x1f) as u8;
        let b5 = ((opcode >> 31) & 1) as u8;
        let bit = (b5 << 5) | b40;
        let imm14 = (opcode >> 5) & 0x3fff;
        let signed = sign_extend_14(imm14) << 2;
        let target = ((pc as i64).wrapping_add(signed)) as u64;
        return Ok(match tb_class {
            0x3600_0000 => DecodedInstruction::Tbz { rt, bit, target },
            0x3700_0000 => DecodedInstruction::Tbnz { rt, bit, target },
            _ => unreachable!(),
        });
    }

    Err(DecodeError::UnsupportedOpcode { pc, opcode })
}

fn decode_data_processing_immediate(
    pc: u64,
    opcode: u32,
) -> Result<DecodedInstruction, DecodeError> {
    let rd = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;
    let sf = ((opcode >> 31) & 1) != 0;

    match opcode & 0x9f00_0000 {
        0x1000_0000 => {
            let immlo = (opcode >> 29) & 0x3;
            let immhi = (opcode >> 5) & 0x7ffff;
            let imm = sign_extend_21((immhi << 2) | immlo);
            let target = ((pc as i64).wrapping_add(imm)) as u64;
            return Ok(DecodedInstruction::Adr { rd, target });
        }
        0x9000_0000 => {
            let immlo = (opcode >> 29) & 0x3;
            let immhi = (opcode >> 5) & 0x7ffff;
            let imm = sign_extend_21((immhi << 2) | immlo);
            let page_base = (pc & !0xfff) as i64;
            let target = page_base.wrapping_add(imm << 12) as u64;
            return Ok(DecodedInstruction::Adrp { rd, target });
        }
        _ => {}
    }

    if opcode & 0x1f80_0000 == 0x1200_0000 {
        let opc = ((opcode >> 29) & 0x3) as u8;
        let n = ((opcode >> 22) & 1) as u8;
        let immr = ((opcode >> 16) & 0x3f) as u8;
        let imms = ((opcode >> 10) & 0x3f) as u8;

        let Some(imm) = decode_logical_immediate(sf, n, immr, imms) else {
            return Err(DecodeError::UnsupportedOpcode { pc, opcode });
        };

        return Ok(match (sf, opc) {
            (true, 0) => DecodedInstruction::AndImm { rd, rn, imm },
            (false, 0) => DecodedInstruction::AndImm32 {
                rd,
                rn,
                imm: imm as u32,
            },
            (true, 1) => DecodedInstruction::OrrImm { rd, rn, imm },
            (false, 1) => DecodedInstruction::OrrImm32 {
                rd,
                rn,
                imm: imm as u32,
            },
            (true, 2) => DecodedInstruction::EorImm { rd, rn, imm },
            (false, 2) => DecodedInstruction::EorImm32 {
                rd,
                rn,
                imm: imm as u32,
            },
            (true, 3) => DecodedInstruction::AndsImm { rd, rn, imm },
            (false, 3) => DecodedInstruction::AndsImm32 {
                rd,
                rn,
                imm: imm as u32,
            },
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        });
    }

    match opcode & 0x7f80_0000 {
        0x1300_0000 | 0x3300_0000 | 0x5300_0000 => {
            let opc = ((opcode >> 29) & 0x3) as u8;
            let n = ((opcode >> 22) & 1) as u8;
            let immr = ((opcode >> 16) & 0x3f) as u8;
            let imms = ((opcode >> 10) & 0x3f) as u8;

            let data_size = if sf { 64u8 } else { 32u8 };
            if (!sf && n != 0) || (sf && n != 1) {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }
            if !sf && (immr >= 32 || imms >= 32) {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            let max = data_size - 1;
            let lsl_alias = imms < max && immr == ((imms + 1) % data_size);

            return Ok(match (opc, sf) {
                (2, true) if imms == max => DecodedInstruction::LsrImm {
                    rd,
                    rn,
                    amount: immr,
                },
                (2, false) if imms == max => DecodedInstruction::LsrImm32 {
                    rd,
                    rn,
                    amount: immr,
                },
                (2, true) if lsl_alias => DecodedInstruction::LslImm {
                    rd,
                    rn,
                    amount: max - imms,
                },
                (2, false) if lsl_alias => DecodedInstruction::LslImm32 {
                    rd,
                    rn,
                    amount: max - imms,
                },
                (2, true) => DecodedInstruction::Ubfm { rd, rn, immr, imms },
                (2, false) => DecodedInstruction::Ubfm32 { rd, rn, immr, imms },
                (1, true) => DecodedInstruction::Bfm { rd, rn, immr, imms },
                (1, false) => DecodedInstruction::Bfm32 { rd, rn, immr, imms },
                (0, true) if imms == max => DecodedInstruction::AsrImm {
                    rd,
                    rn,
                    amount: immr,
                },
                (0, false) if imms == max => DecodedInstruction::AsrImm32 {
                    rd,
                    rn,
                    amount: immr,
                },
                (0, true) => DecodedInstruction::Sbfm { rd, rn, immr, imms },
                (0, false) => DecodedInstruction::Sbfm32 { rd, rn, immr, imms },
                _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
            });
        }
        _ => {}
    }

    if opcode & 0x1f00_0000 == 0x1100_0000 {
        let imm12 = ((opcode >> 10) & 0x0fff) as u16;
        let shift12 = ((opcode >> 22) & 1) != 0;
        let key = (((opcode >> 31) & 1) << 2) | (((opcode >> 30) & 1) << 1) | ((opcode >> 29) & 1);

        return Ok(match key {
            0b100 => DecodedInstruction::AddImm {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b000 => DecodedInstruction::AddImm32 {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b110 => DecodedInstruction::SubImm {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b010 => DecodedInstruction::SubImm32 {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b101 => DecodedInstruction::AddsImm {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b001 => DecodedInstruction::AddsImm32 {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b111 => DecodedInstruction::SubsImm {
                rd,
                rn,
                imm12,
                shift12,
            },
            0b011 => DecodedInstruction::SubsImm32 {
                rd,
                rn,
                imm12,
                shift12,
            },
            _ => unreachable!(),
        });
    }

    match opcode & 0xff80_0000 {
        0x5280_0000 | 0x1280_0000 | 0x7280_0000 => {
            let imm16 = ((opcode >> 5) & 0xffff) as u16;
            let hw = ((opcode >> 21) & 0x3) as u8;
            if hw >= 2 {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            let shift = hw * 16;
            return Ok(match opcode & 0xff80_0000 {
                0x5280_0000 => DecodedInstruction::MovZ32 { rd, imm16, shift },
                0x1280_0000 => DecodedInstruction::MovN32 { rd, imm16, shift },
                0x7280_0000 => DecodedInstruction::MovK32 { rd, imm16, shift },
                _ => unreachable!(),
            });
        }
        0xd280_0000 | 0x9280_0000 | 0xf280_0000 => {
            let imm16 = ((opcode >> 5) & 0xffff) as u16;
            let shift = ((opcode >> 21) & 0x3) as u8 * 16;

            return Ok(match opcode & 0xff80_0000 {
                0xd280_0000 => DecodedInstruction::MovZ { rd, imm16, shift },
                0x9280_0000 => DecodedInstruction::MovN { rd, imm16, shift },
                0xf280_0000 => DecodedInstruction::MovK { rd, imm16, shift },
                _ => unreachable!(),
            });
        }
        _ => {}
    }

    Err(DecodeError::UnsupportedOpcode { pc, opcode })
}

fn decode_data_processing_register(
    pc: u64,
    opcode: u32,
) -> Result<DecodedInstruction, DecodeError> {
    let rd = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;
    let rm = ((opcode >> 16) & 0x1f) as u8;
    let sf = ((opcode >> 31) & 1) != 0;
    let class = opcode & 0xff20_0000;

    match class {
        0x8b00_0000 | 0x0b00_0000 | 0xcb00_0000 | 0x4b00_0000 | 0xab00_0000 | 0x2b00_0000
        | 0xeb00_0000 | 0x6b00_0000 => {
            let amount = ((opcode >> 10) & 0x3f) as u8;
            if !sf && amount >= 32 {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            let Some(shift) = ShiftType::from_u8(((opcode >> 22) & 0x3) as u8) else {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            };

            return Ok(match class {
                0x8b00_0000 => DecodedInstruction::AddReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x0b00_0000 => DecodedInstruction::AddReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xcb00_0000 => DecodedInstruction::SubReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x4b00_0000 => DecodedInstruction::SubReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xab00_0000 => DecodedInstruction::AddsReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x2b00_0000 => DecodedInstruction::AddsReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xeb00_0000 => DecodedInstruction::SubsReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x6b00_0000 => DecodedInstruction::SubsReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                _ => unreachable!(),
            });
        }
        0x8b20_0000 | 0x0b20_0000 | 0xcb20_0000 | 0x4b20_0000 | 0xab20_0000 | 0x2b20_0000
        | 0xeb20_0000 | 0x6b20_0000 => {
            let amount = ((opcode >> 10) & 0x7) as u8;
            if amount > 4 {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            let Some(extend) = ExtendType::from_u8(((opcode >> 13) & 0x7) as u8) else {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            };

            return Ok(match class {
                0x8b20_0000 => DecodedInstruction::AddExt {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0x0b20_0000 => DecodedInstruction::AddExt32 {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0xcb20_0000 => DecodedInstruction::SubExt {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0x4b20_0000 => DecodedInstruction::SubExt32 {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0xab20_0000 => DecodedInstruction::AddsExt {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0x2b20_0000 => DecodedInstruction::AddsExt32 {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0xeb20_0000 => DecodedInstruction::SubsExt {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                0x6b20_0000 => DecodedInstruction::SubsExt32 {
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                },
                _ => unreachable!(),
            });
        }
        0x8a00_0000 | 0x8a20_0000 | 0x0a00_0000 | 0x0a20_0000 | 0xaa00_0000 | 0xaa20_0000
        | 0x2a00_0000 | 0x2a20_0000 | 0xca00_0000 | 0xca20_0000 | 0x4a00_0000 | 0x4a20_0000
        | 0xea00_0000 | 0xea20_0000 | 0x6a00_0000 | 0x6a20_0000 => {
            let amount = ((opcode >> 10) & 0x3f) as u8;
            if !sf && amount >= 32 {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            let Some(shift) = ShiftType::from_u8(((opcode >> 22) & 0x3) as u8) else {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            };

            return Ok(match class {
                0x8a00_0000 => DecodedInstruction::AndReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x8a20_0000 => DecodedInstruction::BicReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x0a00_0000 => DecodedInstruction::AndReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x0a20_0000 => DecodedInstruction::BicReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xaa00_0000 => DecodedInstruction::OrrReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xaa20_0000 => DecodedInstruction::OrnReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x2a00_0000 => DecodedInstruction::OrrReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x2a20_0000 => DecodedInstruction::OrnReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xca00_0000 => DecodedInstruction::EorReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xca20_0000 => DecodedInstruction::EonReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x4a00_0000 => DecodedInstruction::EorReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x4a20_0000 => DecodedInstruction::EonReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xea00_0000 => DecodedInstruction::AndsReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0xea20_0000 => DecodedInstruction::BicsReg {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x6a00_0000 => DecodedInstruction::AndsReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                0x6a20_0000 => DecodedInstruction::BicsReg32 {
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                _ => unreachable!(),
            });
        }
        _ => {}
    }

    let condition = BranchCondition::decode(((opcode >> 12) & 0xf) as u8);
    match opcode & 0xffe0_0c00 {
        0x9a80_0000 => {
            return Ok(DecodedInstruction::Csel {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0x1a80_0000 => {
            return Ok(DecodedInstruction::Csel32 {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0x9a80_0400 => {
            return Ok(DecodedInstruction::Csinc {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0x1a80_0400 => {
            return Ok(DecodedInstruction::Csinc32 {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0xda80_0000 => {
            return Ok(DecodedInstruction::Csinv {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0x5a80_0000 => {
            return Ok(DecodedInstruction::Csinv32 {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0xda80_0400 => {
            return Ok(DecodedInstruction::Csneg {
                rd,
                rn,
                rm,
                condition,
            });
        }
        0x5a80_0400 => {
            return Ok(DecodedInstruction::Csneg32 {
                rd,
                rn,
                rm,
                condition,
            });
        }
        _ => {}
    }

    let nzcv = (opcode & 0xf) as u8;
    match opcode & 0xffe0_0c10 {
        0xfa40_0000 => {
            return Ok(DecodedInstruction::CcmpReg {
                rn,
                rm,
                nzcv,
                condition,
            });
        }
        0x7a40_0000 => {
            return Ok(DecodedInstruction::CcmpReg32 {
                rn,
                rm,
                nzcv,
                condition,
            });
        }
        0xfa40_0800 => {
            let imm5 = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::CcmpImm {
                rn,
                imm5,
                nzcv,
                condition,
            });
        }
        0x7a40_0800 => {
            let imm5 = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::CcmpImm32 {
                rn,
                imm5,
                nzcv,
                condition,
            });
        }
        0xba40_0000 => {
            return Ok(DecodedInstruction::CcmnReg {
                rn,
                rm,
                nzcv,
                condition,
            });
        }
        0x3a40_0000 => {
            return Ok(DecodedInstruction::CcmnReg32 {
                rn,
                rm,
                nzcv,
                condition,
            });
        }
        0xba40_0800 => {
            let imm5 = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::CcmnImm {
                rn,
                imm5,
                nzcv,
                condition,
            });
        }
        0x3a40_0800 => {
            let imm5 = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::CcmnImm32 {
                rn,
                imm5,
                nzcv,
                condition,
            });
        }
        _ => {}
    }

    match opcode & 0xffe0_fc00 {
        0x9ac0_2000 => return Ok(DecodedInstruction::Lslv { rd, rn, rm }),
        0x1ac0_2000 => return Ok(DecodedInstruction::Lslv32 { rd, rn, rm }),
        0x9ac0_2400 => return Ok(DecodedInstruction::Lsrv { rd, rn, rm }),
        0x1ac0_2400 => return Ok(DecodedInstruction::Lsrv32 { rd, rn, rm }),
        0x9ac0_2800 => return Ok(DecodedInstruction::Asrv { rd, rn, rm }),
        0x1ac0_2800 => return Ok(DecodedInstruction::Asrv32 { rd, rn, rm }),
        0x9ac0_2c00 => return Ok(DecodedInstruction::Rorv { rd, rn, rm }),
        0x1ac0_2c00 => return Ok(DecodedInstruction::Rorv32 { rd, rn, rm }),
        0x9b40_7c00 => return Ok(DecodedInstruction::Smulh { rd, rn, rm }),
        0x9bc0_7c00 => return Ok(DecodedInstruction::Umulh { rd, rn, rm }),
        0x9ac0_0800 => return Ok(DecodedInstruction::Udiv { rd, rn, rm }),
        0x1ac0_0800 => return Ok(DecodedInstruction::Udiv32 { rd, rn, rm }),
        0x9ac0_0c00 => return Ok(DecodedInstruction::Sdiv { rd, rn, rm }),
        0x1ac0_0c00 => return Ok(DecodedInstruction::Sdiv32 { rd, rn, rm }),
        _ => {}
    }

    match opcode & 0xffff_fc00 {
        0xdac0_0000 => return Ok(DecodedInstruction::Rbit { rd, rn }),
        0x5ac0_0000 => return Ok(DecodedInstruction::Rbit32 { rd, rn }),
        0xdac0_0c00 => return Ok(DecodedInstruction::Rev { rd, rn }),
        0x5ac0_0800 => return Ok(DecodedInstruction::Rev32 { rd, rn }),
        0xdac0_1000 => return Ok(DecodedInstruction::Clz { rd, rn }),
        0x5ac0_1000 => return Ok(DecodedInstruction::Clz32 { rd, rn }),
        _ => {}
    }

    let ra = ((opcode >> 10) & 0x1f) as u8;
    match opcode & 0xffe0_8000 {
        0x9b00_0000 => {
            if ra == 31 {
                return Ok(DecodedInstruction::Mul { rd, rn, rm });
            }
            return Ok(DecodedInstruction::Madd { rd, rn, rm, ra });
        }
        0x1b00_0000 => {
            if ra == 31 {
                return Ok(DecodedInstruction::Mul32 { rd, rn, rm });
            }
            return Ok(DecodedInstruction::Madd32 { rd, rn, rm, ra });
        }
        0x9b00_8000 => return Ok(DecodedInstruction::Msub { rd, rn, rm, ra }),
        0x1b00_8000 => return Ok(DecodedInstruction::Msub32 { rd, rn, rm, ra }),
        0x9b20_0000 => return Ok(DecodedInstruction::Smaddl { rd, rn, rm, ra }),
        0x9b20_8000 => return Ok(DecodedInstruction::Smsubl { rd, rn, rm, ra }),
        0x9ba0_0000 => return Ok(DecodedInstruction::Umaddl { rd, rn, rm, ra }),
        0x9ba0_8000 => return Ok(DecodedInstruction::Umsubl { rd, rn, rm, ra }),
        _ => {}
    }

    Err(DecodeError::UnsupportedOpcode { pc, opcode })
}

fn decode_load_store(pc: u64, opcode: u32) -> Result<DecodedInstruction, DecodeError> {
    let rt = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;

    match opcode & 0xffe0_fc00 {
        0x8840_7c00 => return Ok(DecodedInstruction::LdxrU32 { rt, rn }),
        0xc840_7c00 => return Ok(DecodedInstruction::LdxrU64 { rt, rn }),
        0x8840_fc00 => return Ok(DecodedInstruction::LdaxrU32 { rt, rn }),
        0xc840_fc00 => return Ok(DecodedInstruction::LdaxrU64 { rt, rn }),
        0x88c0_fc00 => return Ok(DecodedInstruction::LdarU32 { rt, rn }),
        0x08c0_fc00 => return Ok(DecodedInstruction::LdarbU32 { rt, rn }),
        0x48c0_fc00 => return Ok(DecodedInstruction::LdarhU32 { rt, rn }),
        0xc8c0_fc00 => return Ok(DecodedInstruction::LdarU64 { rt, rn }),
        0x8800_7c00 => {
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StxrU32 { rs, rt, rn });
        }
        0xc800_7c00 => {
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StxrU64 { rs, rt, rn });
        }
        0x8800_fc00 => {
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StlxrU32 { rs, rt, rn });
        }
        0xc800_fc00 => {
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StlxrU64 { rs, rt, rn });
        }
        0x8880_fc00 => return Ok(DecodedInstruction::StlrU32 { rt, rn }),
        0x0880_fc00 => return Ok(DecodedInstruction::StlrbU32 { rt, rn }),
        0x4880_fc00 => return Ok(DecodedInstruction::StlrhU32 { rt, rn }),
        0xc880_fc00 => return Ok(DecodedInstruction::StlrU64 { rt, rn }),
        _ => {}
    }

    match opcode & 0xffe0_8000 {
        0x8860_0000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            return Ok(DecodedInstruction::LdxpU32 { rt, rt2, rn });
        }
        0xc860_0000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            return Ok(DecodedInstruction::LdxpU64 { rt, rt2, rn });
        }
        0x8860_8000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            return Ok(DecodedInstruction::LdaxpU32 { rt, rt2, rn });
        }
        0xc860_8000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            return Ok(DecodedInstruction::LdaxpU64 { rt, rt2, rn });
        }
        0x8820_0000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StxpU32 { rs, rt, rt2, rn });
        }
        0xc820_0000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StxpU64 { rs, rt, rt2, rn });
        }
        0x8820_8000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StlxpU32 { rs, rt, rt2, rn });
        }
        0xc820_8000 => {
            let rt2 = ((opcode >> 10) & 0x1f) as u8;
            let rs = ((opcode >> 16) & 0x1f) as u8;
            return Ok(DecodedInstruction::StlxpU64 { rs, rt, rt2, rn });
        }
        _ => {}
    }

    if let Some(instruction) = decode_pair_instruction(opcode) {
        return Ok(instruction);
    }

    let imm9 = sign_extend_9((opcode >> 12) & 0x1ff) as i16;
    match opcode & 0xffe0_0c00 {
        0xf840_0400 => return Ok(DecodedInstruction::LdrPostI64 { rt, rn, imm9 }),
        0xf840_0c00 => return Ok(DecodedInstruction::LdrPreI64 { rt, rn, imm9 }),
        0xf800_0400 => return Ok(DecodedInstruction::StrPostI64 { rt, rn, imm9 }),
        0xf800_0c00 => return Ok(DecodedInstruction::StrPreI64 { rt, rn, imm9 }),
        0xb840_0400 => return Ok(DecodedInstruction::LdrPostI32 { rt, rn, imm9 }),
        0xb840_0c00 => return Ok(DecodedInstruction::LdrPreI32 { rt, rn, imm9 }),
        0xb800_0400 => return Ok(DecodedInstruction::StrPostI32 { rt, rn, imm9 }),
        0xb800_0c00 => return Ok(DecodedInstruction::StrPreI32 { rt, rn, imm9 }),
        0x3840_0400 => return Ok(DecodedInstruction::LdrPostI8 { rt, rn, imm9 }),
        0x3840_0c00 => return Ok(DecodedInstruction::LdrPreI8 { rt, rn, imm9 }),
        0x3800_0400 => return Ok(DecodedInstruction::StrPostI8 { rt, rn, imm9 }),
        0x3800_0c00 => return Ok(DecodedInstruction::StrPreI8 { rt, rn, imm9 }),
        0x7840_0400 => return Ok(DecodedInstruction::LdrPostI16 { rt, rn, imm9 }),
        0x7840_0c00 => return Ok(DecodedInstruction::LdrPreI16 { rt, rn, imm9 }),
        0x7800_0400 => return Ok(DecodedInstruction::StrPostI16 { rt, rn, imm9 }),
        0x7800_0c00 => return Ok(DecodedInstruction::StrPreI16 { rt, rn, imm9 }),
        0xf840_0000 => return Ok(DecodedInstruction::LdurI64 { rt, rn, imm9 }),
        0xf800_0000 => return Ok(DecodedInstruction::SturI64 { rt, rn, imm9 }),
        0xb840_0000 => return Ok(DecodedInstruction::LdurI32 { rt, rn, imm9 }),
        0xb800_0000 => return Ok(DecodedInstruction::SturI32 { rt, rn, imm9 }),
        0x7840_0000 => return Ok(DecodedInstruction::LdurI16 { rt, rn, imm9 }),
        0x7800_0000 => return Ok(DecodedInstruction::SturI16 { rt, rn, imm9 }),
        0x3840_0000 => return Ok(DecodedInstruction::LdurI8 { rt, rn, imm9 }),
        0x3800_0000 => return Ok(DecodedInstruction::SturI8 { rt, rn, imm9 }),
        0x3880_0000 => return Ok(DecodedInstruction::LdursbI64 { rt, rn, imm9 }),
        0x38c0_0000 => return Ok(DecodedInstruction::LdursbI32 { rt, rn, imm9 }),
        0x7880_0000 => return Ok(DecodedInstruction::LdurshI64 { rt, rn, imm9 }),
        0x78c0_0000 => return Ok(DecodedInstruction::LdurshI32 { rt, rn, imm9 }),
        0xb880_0000 => return Ok(DecodedInstruction::LdurswI64 { rt, rn, imm9 }),
        // Sign-extending post/pre-index loads
        0x78c0_0400 => return Ok(DecodedInstruction::LdrshPostI32 { rt, rn, imm9 }),
        0x7880_0400 => return Ok(DecodedInstruction::LdrshPostI64 { rt, rn, imm9 }),
        0x78c0_0c00 => return Ok(DecodedInstruction::LdrshPreI32 { rt, rn, imm9 }),
        0x7880_0c00 => return Ok(DecodedInstruction::LdrshPreI64 { rt, rn, imm9 }),
        0x38c0_0400 => return Ok(DecodedInstruction::LdrsbPostI32 { rt, rn, imm9 }),
        0x3880_0400 => return Ok(DecodedInstruction::LdrsbPostI64 { rt, rn, imm9 }),
        0x38c0_0c00 => return Ok(DecodedInstruction::LdrsbPreI32 { rt, rn, imm9 }),
        0x3880_0c00 => return Ok(DecodedInstruction::LdrsbPreI64 { rt, rn, imm9 }),
        0xb880_0400 => return Ok(DecodedInstruction::LdrswPostI64 { rt, rn, imm9 }),
        0xb880_0c00 => return Ok(DecodedInstruction::LdrswPreI64 { rt, rn, imm9 }),
        _ => {}
    }

    let reg_offset_class = opcode & 0xffe0_0c00;
    match reg_offset_class {
        0xf860_0800 | 0xf820_0800 | 0xb860_0800 | 0xb820_0800 | 0x7860_0800 | 0x7820_0800
        | 0x3860_0800 | 0x3820_0800 | 0x38a0_0800 | 0x38e0_0800 | 0x78a0_0800 | 0x78e0_0800
        | 0xb8a0_0800 | 0x3ca0_0800 | 0x3ce0_0800 | 0xbc60_0800 | 0xbc20_0800 | 0xfc60_0800
        | 0xfc20_0800 => {
            let rm = ((opcode >> 16) & 0x1f) as u8;
            let option = ((opcode >> 13) & 0x7) as u8;
            let shift = ((opcode >> 12) & 1) != 0;

            if !matches!(option, 0b010 | 0b011 | 0b110 | 0b111) {
                return Err(DecodeError::UnsupportedOpcode { pc, opcode });
            }

            return Ok(match reg_offset_class {
                0xf860_0800 => DecodedInstruction::LdrRegU64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xf820_0800 => DecodedInstruction::StrRegU64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xb860_0800 => DecodedInstruction::LdrRegU32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xb820_0800 => DecodedInstruction::StrRegU32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x7860_0800 => DecodedInstruction::LdrRegU16 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x7820_0800 => DecodedInstruction::StrRegU16 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x3860_0800 => DecodedInstruction::LdrRegU8 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x3820_0800 => DecodedInstruction::StrRegU8 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x38a0_0800 => DecodedInstruction::LdrsbReg64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x38e0_0800 => DecodedInstruction::LdrsbReg32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x78a0_0800 => DecodedInstruction::LdrshReg64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x78e0_0800 => DecodedInstruction::LdrshReg32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xb8a0_0800 => DecodedInstruction::LdrswReg64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x3ca0_0800 => DecodedInstruction::StrFpReg128 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0x3ce0_0800 => DecodedInstruction::LdrFpReg128 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xbc60_0800 => DecodedInstruction::LdrFpReg32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xbc20_0800 => DecodedInstruction::StrFpReg32 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xfc60_0800 => DecodedInstruction::LdrFpReg64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                0xfc20_0800 => DecodedInstruction::StrFpReg64 {
                    rt,
                    rn,
                    rm,
                    option,
                    shift,
                },
                _ => unreachable!(),
            });
        }
        _ => {}
    }

    let fp_unsigned_class = opcode & 0x3f40_0000;
    match fp_unsigned_class {
        0x3d40_0000 | 0x3d00_0000 => {
            let imm12 = ((opcode >> 10) & 0x0fff) as u16;
            let size = ((opcode >> 30) & 0x3) as u8;
            let opc = ((opcode >> 22) & 0x3) as u8;

            return Ok(match (fp_unsigned_class, opc, size) {
                (0x3d40_0000, 1, 0) => DecodedInstruction::LdrFpImm8 { rt, rn, imm12 },
                (0x3d40_0000, 1, 1) => DecodedInstruction::LdrFpImm16 { rt, rn, imm12 },
                (0x3d40_0000, 1, 2) => DecodedInstruction::LdrFpImm32 { rt, rn, imm12 },
                (0x3d40_0000, 1, 3) => DecodedInstruction::LdrFpImm64 { rt, rn, imm12 },
                (0x3d40_0000, 3, 0) => DecodedInstruction::LdrFpImm128 { rt, rn, imm12 },
                (0x3d00_0000, 0, 0) => DecodedInstruction::StrFpImm8 { rt, rn, imm12 },
                (0x3d00_0000, 0, 1) => DecodedInstruction::StrFpImm16 { rt, rn, imm12 },
                (0x3d00_0000, 0, 2) => DecodedInstruction::StrFpImm32 { rt, rn, imm12 },
                (0x3d00_0000, 0, 3) => DecodedInstruction::StrFpImm64 { rt, rn, imm12 },
                (0x3d00_0000, 2, 0) => DecodedInstruction::StrFpImm128 { rt, rn, imm12 },
                _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
            });
        }
        _ => {}
    }

    let fp_post_pre_class = opcode & 0x3f20_0c00;
    match fp_post_pre_class {
        0x3c00_0400 | 0x3c00_0c00 => {
            let imm9 = sign_extend_9((opcode >> 12) & 0x1ff) as i16;
            let size = ((opcode >> 30) & 0x3) as u8;
            let opc = ((opcode >> 22) & 0x3) as u8;

            return Ok(match (fp_post_pre_class, opc, size) {
                // 128-bit (Qt) post/pre
                (0x3c00_0400, 2, 0) => DecodedInstruction::StrPostFp128 { rt, rn, imm9 },
                (0x3c00_0400, 3, 0) => DecodedInstruction::LdrPostFp128 { rt, rn, imm9 },
                (0x3c00_0c00, 2, 0) => DecodedInstruction::StrPreFp128 { rt, rn, imm9 },
                (0x3c00_0c00, 3, 0) => DecodedInstruction::LdrPreFp128 { rt, rn, imm9 },
                // 64-bit (Dt) post/pre
                (0x3c00_0400, 0, 3) => DecodedInstruction::StrPostFp64 { rt, rn, imm9 },
                (0x3c00_0400, 1, 3) => DecodedInstruction::LdrPostFp64 { rt, rn, imm9 },
                (0x3c00_0c00, 0, 3) => DecodedInstruction::StrPreFp64 { rt, rn, imm9 },
                (0x3c00_0c00, 1, 3) => DecodedInstruction::LdrPreFp64 { rt, rn, imm9 },
                // 32-bit (St) post/pre
                (0x3c00_0400, 0, 2) => DecodedInstruction::StrPostFp32 { rt, rn, imm9 },
                (0x3c00_0400, 1, 2) => DecodedInstruction::LdrPostFp32 { rt, rn, imm9 },
                (0x3c00_0c00, 0, 2) => DecodedInstruction::StrPreFp32 { rt, rn, imm9 },
                (0x3c00_0c00, 1, 2) => DecodedInstruction::LdrPreFp32 { rt, rn, imm9 },
                _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
            });
        }
        _ => {}
    }

    if opcode & 0x3f20_0c00 == 0x3c00_0000 {
        let rt = (opcode & 0x1f) as u8;
        let rn = ((opcode >> 5) & 0x1f) as u8;
        let imm9 = sign_extend_9((opcode >> 12) & 0x1ff) as i16;
        let size = ((opcode >> 30) & 0x3) as u8;
        let opc = ((opcode >> 22) & 0x3) as u8;

        return Ok(match (opc, size) {
            (0, 0) => DecodedInstruction::SturFp8 { rt, rn, imm9 },
            (0, 1) => DecodedInstruction::SturFp16 { rt, rn, imm9 },
            (0, 2) => DecodedInstruction::SturFp32 { rt, rn, imm9 },
            (0, 3) => DecodedInstruction::SturFp64 { rt, rn, imm9 },
            (2, 0) => DecodedInstruction::SturFp128 { rt, rn, imm9 },
            (1, 0) => DecodedInstruction::LdurFp8 { rt, rn, imm9 },
            (1, 1) => DecodedInstruction::LdurFp16 { rt, rn, imm9 },
            (1, 2) => DecodedInstruction::LdurFp32 { rt, rn, imm9 },
            (1, 3) => DecodedInstruction::LdurFp64 { rt, rn, imm9 },
            (3, 0) => DecodedInstruction::LdurFp128 { rt, rn, imm9 },
            _ => return Err(DecodeError::UnsupportedOpcode { pc, opcode }),
        });
    }

    let imm12 = ((opcode >> 10) & 0x0fff) as u16;
    match opcode & 0xffc0_0000 {
        0xf900_0000 => return Ok(DecodedInstruction::StrImmU64 { rt, rn, imm12 }),
        0xf940_0000 => return Ok(DecodedInstruction::LdrImmU64 { rt, rn, imm12 }),
        0xb900_0000 => return Ok(DecodedInstruction::StrImmU32 { rt, rn, imm12 }),
        0xb940_0000 => return Ok(DecodedInstruction::LdrImmU32 { rt, rn, imm12 }),
        0x7900_0000 => return Ok(DecodedInstruction::StrImmU16 { rt, rn, imm12 }),
        0x7940_0000 => return Ok(DecodedInstruction::LdrImmU16 { rt, rn, imm12 }),
        0x3900_0000 => return Ok(DecodedInstruction::StrImmU8 { rt, rn, imm12 }),
        0x3940_0000 => return Ok(DecodedInstruction::LdrImmU8 { rt, rn, imm12 }),
        0x3980_0000 => return Ok(DecodedInstruction::LdrsbImm64 { rt, rn, imm12 }),
        0x39c0_0000 => return Ok(DecodedInstruction::LdrsbImm32 { rt, rn, imm12 }),
        0x7980_0000 => return Ok(DecodedInstruction::LdrshImm64 { rt, rn, imm12 }),
        0x79c0_0000 => return Ok(DecodedInstruction::LdrshImm32 { rt, rn, imm12 }),
        0xb980_0000 => return Ok(DecodedInstruction::LdrswImm64 { rt, rn, imm12 }),
        _ => {}
    }

    Err(DecodeError::UnsupportedOpcode { pc, opcode })
}

fn sign_extend_26(value: u32) -> i64 {
    let signed = (value & 0x03ff_ffff) as i64;
    (signed << 38) >> 38
}

fn sign_extend_19(value: u32) -> i64 {
    let signed = (value & 0x7ffff) as i64;
    (signed << 45) >> 45
}

fn sign_extend_14(value: u32) -> i64 {
    let signed = (value & 0x3fff) as i64;
    (signed << 50) >> 50
}

fn sign_extend_7(value: u32) -> i64 {
    let signed = (value & 0x7f) as i64;
    (signed << 57) >> 57
}

fn sign_extend_9(value: u32) -> i64 {
    let signed = (value & 0x1ff) as i64;
    (signed << 55) >> 55
}

fn sign_extend_21(value: u32) -> i64 {
    let signed = (value & 0x1f_ffff) as i64;
    (signed << 43) >> 43
}

fn decode_logical_immediate(sf: bool, n: u8, immr: u8, imms: u8) -> Option<u64> {
    if !sf && n != 0 {
        return None;
    }

    let n_imms = (u32::from(n) << 6) | (u32::from(!imms) & 0x3f);
    if n_imms == 0 {
        return None;
    }

    let len = 31 - n_imms.leading_zeros();
    let element_size = 1u32 << len;
    if element_size < 2 {
        return None;
    }

    let levels = element_size - 1;
    let s = u32::from(imms) & levels;
    let r = u32::from(immr) & levels;
    if s == levels {
        return None;
    }

    let mut pattern = if s == 63 {
        u64::MAX
    } else {
        (1u64 << (s + 1)) - 1
    };

    pattern = rotate_right_in_size(pattern, r, element_size);

    let data_size = if sf { 64 } else { 32 };
    Some(replicate_in_size(pattern, element_size, data_size))
}

fn rotate_right_in_size(value: u64, rot: u32, size: u32) -> u64 {
    let mask = if size == 64 {
        u64::MAX
    } else {
        (1u64 << size) - 1
    };
    let rot = rot % size;
    let value = value & mask;

    if rot == 0 {
        return value;
    }

    ((value >> rot) | (value << (size - rot))) & mask
}

fn replicate_in_size(pattern: u64, element_size: u32, data_size: u32) -> u64 {
    let mut value = 0u64;
    let mut bit = 0u32;

    while bit < data_size {
        value |= pattern << bit;
        bit += element_size;
    }

    if data_size == 32 {
        value & 0xffff_ffff
    } else {
        value
    }
}

/// Decode AdvSIMD modified immediate into a 64-bit replicated pattern.
/// Returns (imm64, is_orr, is_bic).
fn decode_advsimd_modified_imm(
    op: u8,
    cmode: u8,
    o2: u8,
    imm8: u8,
) -> Result<(u64, bool, bool), DecodeError> {
    let imm = imm8 as u64;
    let cmode_hi = cmode >> 1;

    match (op, cmode_hi, cmode & 1, o2) {
        // MOVI 32-bit shifted: cmode = 0xx0
        (0, 0..=3, 0, 0) => {
            let shift = (cmode_hi as u64) * 8;
            let elem = imm << shift;
            Ok((elem | (elem << 32), false, false))
        }
        // ORR 32-bit shifted: cmode = 0xx1
        (0, 0..=3, 1, 0) => {
            let shift = (cmode_hi as u64) * 8;
            let elem = imm << shift;
            Ok((elem | (elem << 32), true, false))
        }
        // MOVI 16-bit shifted: cmode = 10x0
        (0, 4..=5, 0, 0) => {
            let shift = ((cmode_hi & 1) as u64) * 8;
            let h = imm << shift;
            let elem = h | (h << 16) | (h << 32) | (h << 48);
            Ok((elem, false, false))
        }
        // ORR 16-bit shifted: cmode = 10x1
        (0, 4..=5, 1, 0) => {
            let shift = ((cmode_hi & 1) as u64) * 8;
            let h = imm << shift;
            let elem = h | (h << 16) | (h << 32) | (h << 48);
            Ok((elem, true, false))
        }
        // MOVI 32-bit shifting ones (MSL): cmode = 110x
        (0, 6, _, 0) => {
            let shift = if cmode & 1 == 0 { 8u64 } else { 16u64 };
            let ones = (1u64 << shift) - 1;
            let elem = (imm << shift) | ones;
            Ok((elem | (elem << 32), false, false))
        }
        // MOVI 8-bit: cmode = 1110, op=0
        (0, 7, 0, 0) => {
            let mut elem = 0u64;
            for i in 0..8 {
                elem |= imm << (i * 8);
            }
            Ok((elem, false, false))
        }
        // FMOV f32: cmode = 1111, op=0
        (0, 7, 1, 0) => {
            // imm8 -> fp32
            let sign = ((imm >> 7) & 1) as u32;
            let exp_hi = if (imm >> 6) & 1 != 0 { 0b0 } else { 0b1 };
            let exp_lo = if (imm >> 6) & 1 != 0 {
                0b11111u32
            } else {
                0b00000u32
            };
            let frac = ((imm & 0x3f) as u32) << 19;
            let fp32 = (sign << 31) | (exp_hi << 30) | (exp_lo << 25) | frac;
            let elem = (fp32 as u64) | ((fp32 as u64) << 32);
            Ok((elem, false, false))
        }
        // MVNI 32-bit shifted: cmode = 0xx0, op=1
        (1, 0..=3, 0, 0) => {
            let shift = (cmode_hi as u64) * 8;
            let elem = !(imm << shift) & 0xffff_ffff;
            Ok((elem | (elem << 32), false, false))
        }
        // BIC 32-bit shifted: cmode = 0xx1, op=1
        (1, 0..=3, 1, 0) => {
            let shift = (cmode_hi as u64) * 8;
            let elem = imm << shift;
            Ok((elem | (elem << 32), false, true))
        }
        // MVNI 16-bit shifted: cmode = 10x0, op=1
        (1, 4..=5, 0, 0) => {
            let shift = ((cmode_hi & 1) as u64) * 8;
            let h = !(imm << shift) & 0xffff;
            let elem = h | (h << 16) | (h << 32) | (h << 48);
            Ok((elem, false, false))
        }
        // BIC 16-bit shifted: cmode = 10x1, op=1
        (1, 4..=5, 1, 0) => {
            let shift = ((cmode_hi & 1) as u64) * 8;
            let h = imm << shift;
            let elem = h | (h << 16) | (h << 32) | (h << 48);
            Ok((elem, false, true))
        }
        // MVNI 32-bit shifting ones: cmode = 110x, op=1
        (1, 6, _, 0) => {
            let shift = if cmode & 1 == 0 { 8u64 } else { 16u64 };
            let ones = (1u64 << shift) - 1;
            let elem = !((imm << shift) | ones) & 0xffff_ffff;
            Ok((elem | (elem << 32), false, false))
        }
        // MOVI 64-bit: cmode = 1110, op=1, o2=0
        (1, 7, 0, 0) => {
            // Each bit of imm8 expands to a byte of 0xFF or 0x00
            let mut val = 0u64;
            for i in 0..8 {
                if (imm >> i) & 1 != 0 {
                    val |= 0xFF << (i * 8);
                }
            }
            Ok((val, false, false))
        }
        // FMOV f64: cmode = 1111, op=1, o2=0
        (1, 7, 1, 0) => {
            let sign = ((imm >> 7) & 1) as u64;
            let exp_hi = if (imm >> 6) & 1 != 0 { 0b0u64 } else { 0b1u64 };
            let exp_lo = if (imm >> 6) & 1 != 0 {
                0b11111111u64
            } else {
                0b00000000u64
            };
            let frac = (imm & 0x3f) << 46;
            let fp64 = (sign << 63) | (exp_hi << 62) | (exp_lo << 54) | frac;
            Ok((fp64, false, false))
        }
        _ => Err(DecodeError::UnsupportedOpcode { pc: 0, opcode: 0 }),
    }
}

fn decode_pair_instruction(opcode: u32) -> Option<DecodedInstruction> {
    let rt = (opcode & 0x1f) as u8;
    let rn = ((opcode >> 5) & 0x1f) as u8;
    let rt2 = ((opcode >> 10) & 0x1f) as u8;
    let imm7 = sign_extend_7((opcode >> 15) & 0x7f) as i16;

    match opcode & 0xffc0_0000 {
        0xa900_0000 => Some(DecodedInstruction::StpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0xa940_0000 => Some(DecodedInstruction::LdpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0xa980_0000 => Some(DecodedInstruction::StpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0xa9c0_0000 => Some(DecodedInstruction::LdpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0xa880_0000 => Some(DecodedInstruction::StpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0xa8c0_0000 => Some(DecodedInstruction::LdpI64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x2900_0000 => Some(DecodedInstruction::StpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x2940_0000 => Some(DecodedInstruction::LdpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x6940_0000 => Some(DecodedInstruction::Ldpsw {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x2980_0000 => Some(DecodedInstruction::StpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x29c0_0000 => Some(DecodedInstruction::LdpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x69c0_0000 => Some(DecodedInstruction::Ldpsw {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x2880_0000 => Some(DecodedInstruction::StpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x28c0_0000 => Some(DecodedInstruction::LdpI32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x68c0_0000 => Some(DecodedInstruction::Ldpsw {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x2d00_0000 => Some(DecodedInstruction::StpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x2d40_0000 => Some(DecodedInstruction::LdpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x2d80_0000 => Some(DecodedInstruction::StpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x2dc0_0000 => Some(DecodedInstruction::LdpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x2c80_0000 => Some(DecodedInstruction::StpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x2cc0_0000 => Some(DecodedInstruction::LdpFp32 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x6d00_0000 => Some(DecodedInstruction::StpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x6d40_0000 => Some(DecodedInstruction::LdpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0x6d80_0000 => Some(DecodedInstruction::StpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x6dc0_0000 => Some(DecodedInstruction::LdpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0x6c80_0000 => Some(DecodedInstruction::StpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0x6cc0_0000 => Some(DecodedInstruction::LdpFp64 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0xad00_0000 => Some(DecodedInstruction::StpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0xad40_0000 => Some(DecodedInstruction::LdpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::SignedOffset,
        }),
        0xad80_0000 => Some(DecodedInstruction::StpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0xadc0_0000 => Some(DecodedInstruction::LdpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PreIndex,
        }),
        0xac80_0000 => Some(DecodedInstruction::StpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        0xacc0_0000 => Some(DecodedInstruction::LdpV128 {
            rt,
            rt2,
            rn,
            imm7,
            mode: PairAddressMode::PostIndex,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BranchCondition, DecodedInstruction, ExtendType, PairAddressMode, ShiftType,
        SystemRegister, decode_block,
    };

    fn encode_mrs(rt: u8, reg: SystemRegister) -> u32 {
        0xd530_0000 | (u32::from(reg.encoding()) << 5) | u32::from(rt)
    }

    fn encode_msr(reg: SystemRegister, rt: u8) -> u32 {
        0xd510_0000 | (u32::from(reg.encoding()) << 5) | u32::from(rt)
    }

    fn encode_ccmp_reg(sf: bool, rn: u8, rm: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xfa40_0000 } else { 0x7a40_0000 };
        base | (u32::from(rm & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmp_imm(sf: bool, rn: u8, imm5: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xfa40_0800 } else { 0x7a40_0800 };
        base | (u32::from(imm5 & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmn_reg(sf: bool, rn: u8, rm: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xba40_0000 } else { 0x3a40_0000 };
        base | (u32::from(rm & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn encode_ccmn_imm(sf: bool, rn: u8, imm5: u8, nzcv: u8, condition: u8) -> u32 {
        let base = if sf { 0xba40_0800 } else { 0x3a40_0800 };
        base | (u32::from(imm5 & 0x1f) << 16)
            | (u32::from(condition & 0x0f) << 12)
            | (u32::from(rn & 0x1f) << 5)
            | u32::from(nzcv & 0x0f)
    }

    fn decode_single(base: u64, opcode: u32) -> DecodedInstruction {
        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            [opcode].get(idx).copied()
        })
        .expect("single-instruction decode should succeed");

        block.instructions[0].clone()
    }

    #[test]
    fn decodes_conditional_and_compare_branches() {
        let base = 0x8000_0000;
        let words = [
            0xf100_0400, // subs x0, x0, #1
            0xb5ff_ffe0, // cbnz x0, -4
            0x5400_0060, // b.eq +12
            0xd420_0020, // brk #1
        ];

        let block = decode_block(base, 16, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::SubsImm { .. }
        ));

        match block.instructions[1] {
            DecodedInstruction::Cbnz { rt, target } => {
                assert_eq!(rt, 0);
                assert_eq!(target, base);
            }
            _ => panic!("expected cbnz"),
        }

        let cond_block = decode_block(base + 8, 4, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("conditional decode should succeed");

        match cond_block.instructions[0] {
            DecodedInstruction::BCond { condition, .. } => {
                assert_eq!(condition, BranchCondition::Eq);
            }
            _ => panic!("expected b.cond"),
        }
    }

    #[test]
    fn decodes_wide_and_narrow_compare_branches() {
        let base = 0x8000_0090;
        let words = [
            0xb400_0061, // cbz x1, +12
            0x3400_0062, // cbz w2, +12
            0xb500_0063, // cbnz x3, +12
            0x3500_0064, // cbnz w4, +12
        ];

        let cbz_x = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cbz x decode should succeed");
        assert!(matches!(
            cbz_x.instructions[0],
            DecodedInstruction::Cbz { rt: 1, target } if target == base + 12
        ));

        let cbz_w = decode_block(base + 4, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cbz w decode should succeed");
        assert!(matches!(
            cbz_w.instructions[0],
            DecodedInstruction::Cbz32 { rt: 2, target } if target == base + 16
        ));

        let cbnz_x = decode_block(base + 8, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cbnz x decode should succeed");
        assert!(matches!(
            cbnz_x.instructions[0],
            DecodedInstruction::Cbnz { rt: 3, target } if target == base + 20
        ));

        let cbnz_w = decode_block(base + 12, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cbnz w decode should succeed");
        assert!(matches!(
            cbnz_w.instructions[0],
            DecodedInstruction::Cbnz32 { rt: 4, target } if target == base + 24
        ));
    }

    #[test]
    fn decodes_bl_and_br() {
        let base = 0x8000_0100;
        let words = [
            0x9400_0002, // bl +8
            0xd61f_03c0, // br x30
        ];

        let block = decode_block(base, 4, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Bl { target } => assert_eq!(target, base + 8),
            _ => panic!("expected bl"),
        }

        let branch_block = decode_block(base + 4, 4, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("branch decode should succeed");

        match branch_block.instructions[0] {
            DecodedInstruction::Br { rn } => assert_eq!(rn, 30),
            _ => panic!("expected br"),
        }
    }

    #[test]
    fn decodes_system_and_privileged_instructions() {
        let base = 0x8000_0200;
        let words = [
            encode_mrs(0, SystemRegister::CurrentEl),
            encode_msr(SystemRegister::VbarEl1, 0),
            0xd503_3bbf, // dmb sy
            0xd503_3f9f, // dsb sy
            0xd503_3fdf, // isb
            0xd503_305f, // clrex
            0xd400_0021, // svc #1
            0xd69f_03e0, // eret
        ];

        let sys_block = decode_block(base, 8, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("system decode should succeed");

        match sys_block.instructions[0] {
            DecodedInstruction::Mrs { rt, reg } => {
                assert_eq!(rt, 0);
                assert_eq!(reg, SystemRegister::CurrentEl);
            }
            _ => panic!("expected mrs"),
        }

        match sys_block.instructions[1] {
            DecodedInstruction::Msr { reg, rt } => {
                assert_eq!(rt, 0);
                assert_eq!(reg, SystemRegister::VbarEl1);
            }
            _ => panic!("expected msr"),
        }

        assert!(matches!(
            sys_block.instructions[2],
            DecodedInstruction::Dmb { option: 0b1011 }
        ));
        assert!(matches!(
            sys_block.instructions[3],
            DecodedInstruction::Dsb { option: 0b1111 }
        ));
        assert!(matches!(
            sys_block.instructions[4],
            DecodedInstruction::Isb { option: 0b1111 }
        ));
        assert!(matches!(
            sys_block.instructions[5],
            DecodedInstruction::Clrex { option: 0 }
        ));

        match sys_block.instructions[6] {
            DecodedInstruction::Svc { imm16 } => assert_eq!(imm16, 1),
            _ => panic!("expected svc"),
        }

        let eret_block = decode_block(base + 28, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("eret decode should succeed");

        assert!(matches!(
            eret_block.instructions[0],
            DecodedInstruction::Eret
        ));
    }

    #[test]
    fn decodes_dmb_ishld_variant() {
        let base = 0x8000_0240;
        let words = [0xd503_39bf]; // dmb ishld

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("dmb variant decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Dmb { option: 0b1001 }
        ));
    }

    #[test]
    fn decodes_dc_civac_variant() {
        let base = 0x8000_0244;
        let words = [0xd50b_7e2a]; // dc civac, x10

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("dc civac decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::DcCivac { rt: 10 }
        ));
    }

    #[test]
    fn decodes_scvtf_s32_variant() {
        let base = 0x8000_0280;
        let words = [0x1e22_0300]; // scvtf s0, w24

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("scvtf decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::ScvtfS32 { rd, rn } => {
                assert_eq!(rd, 0);
                assert_eq!(rn, 24);
            }
            _ => panic!("expected scvtf s32"),
        }
    }

    #[test]
    fn decodes_fdiv_s_variant() {
        let base = 0x8000_02a0;
        let words = [0x1e21_1800]; // fdiv s0, s0, s1

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fdiv decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::FdivS { rd, rn, rm } => {
                assert_eq!(rd, 0);
                assert_eq!(rn, 0);
                assert_eq!(rm, 1);
            }
            _ => panic!("expected fdiv s"),
        }
    }

    #[test]
    fn decodes_fadd_fsub_fmul_s_variants() {
        let base = 0x8000_02b8;
        let words = [
            0x1e23_2841, // fadd s1, s2, s3
            0x1e26_38a4, // fsub s4, s5, s6
            0x1e29_0907, // fmul s7, s8, s9
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fadd/fsub/fmul decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FaddS {
                rd: 1,
                rn: 2,
                rm: 3
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FsubS {
                rd: 4,
                rn: 5,
                rm: 6
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::FmulS {
                rd: 7,
                rn: 8,
                rm: 9
            }
        ));
    }

    #[test]
    fn decodes_fmax_fmin_s_variants() {
        let base = 0x8000_02bc;
        let words = [
            0x1e23_4841, // fmax s1, s2, s3
            0x1e26_58a4, // fmin s4, s5, s6
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fmax/fmin decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FmaxS {
                rd: 1,
                rn: 2,
                rm: 3
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FminS {
                rd: 4,
                rn: 5,
                rm: 6
            }
        ));
    }

    #[test]
    fn decodes_fadd_fsub_fmul_fmax_fmin_fdiv_d_variants() {
        let base = 0x8000_02c0;
        let words = [
            0x1e62_2820, // fadd d0, d1, d2
            0x1e62_3820, // fsub d0, d1, d2
            0x1e62_0820, // fmul d0, d1, d2
            0x1e62_4820, // fmax d0, d1, d2
            0x1e62_5820, // fmin d0, d1, d2
            0x1e62_1820, // fdiv d0, d1, d2
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("f64 arithmetic decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FaddD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FsubD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::FmulD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::FmaxD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::FminD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::FdivD {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));
    }

    #[test]
    fn decodes_ucvtf_scalar_reg_variant() {
        let base = 0x8000_02b0;
        let words = [0x7e21_d841]; // ucvtf s1, s2

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("ucvtf scalar reg decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::UcvtfSReg32 { rd, rn } => {
                assert_eq!(rd, 1);
                assert_eq!(rn, 2);
            }
            _ => panic!("expected ucvtf scalar reg"),
        }
    }

    #[test]
    fn decodes_ucvtf_scvtf_scalar_d_reg_variants() {
        let base = 0x8000_02b8;
        let words = [
            0x7e61_db80, // ucvtf d0, d28
            0x5e61_db80, // scvtf d0, d28
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("ucvtf/scvtf scalar d reg decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::UcvtfDReg64 { rd: 0, rn: 28 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::ScvtfDReg64 { rd: 0, rn: 28 }
        ));
    }

    #[test]
    fn decodes_ucvtf_scvtf_d_from_w_variants() {
        let base = 0x8000_02bc;
        let words = [
            0x1e63_0000, // ucvtf d0, w0
            0x1e62_001d, // scvtf d29, w0
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("ucvtf/scvtf d,w decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::UcvtfD32 { rd: 0, rn: 0 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::ScvtfD32 { rd: 29, rn: 0 }
        ));
    }

    #[test]
    fn decodes_fmov_immediate_s_variant() {
        let base = 0x8000_02c0;
        let words = [0x1e2e_1001]; // fmov s1, #1.0

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fmov immediate decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::FmovImmS { rd, imm8 } => {
                assert_eq!(rd, 1);
                assert_eq!(imm8, 0x70);
            }
            _ => panic!("expected fmov immediate s"),
        }
    }

    #[test]
    fn decodes_fmov_immediate_d_variant() {
        let base = 0x8000_02c8;
        let words = [0x1e64_d01f]; // fmov d31, #11.0

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fmov immediate decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::FmovImmD { rd, imm8 } => {
                assert_eq!(rd, 31);
                assert_eq!(imm8, 0x26);
            }
            _ => panic!("expected fmov immediate d"),
        }
    }

    #[test]
    fn decodes_fcmp_s_variant() {
        let base = 0x8000_02e0;
        let words = [0x1e21_2000]; // fcmp s0, s1

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fcmp decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::FcmpS { rn, rm } => {
                assert_eq!(rn, 0);
                assert_eq!(rm, 1);
            }
            _ => panic!("expected fcmp s"),
        }
    }

    #[test]
    fn decodes_fcmp_and_fcmpe_d_variants() {
        let base = 0x8000_02e4;
        let words = [
            0x1e61_2000, // fcmp d0, d1
            0x1e7f_2010, // fcmpe d0, d31
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fcmp/fcmpe d decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FcmpD { rn: 0, rm: 1 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FcmpD { rn: 0, rm: 31 }
        ));
    }

    #[test]
    fn decodes_frintm_s_and_d_variants() {
        let base = 0x8000_02e8;
        let words = [
            0x1e25_4000, // frintm s0, s0
            0x1e65_43ff, // frintm d31, d31
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("frintm decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FrintmS { rd: 0, rn: 0 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FrintmD { rd: 31, rn: 31 }
        ));
    }

    #[test]
    fn decodes_fmov_between_w_and_s_variants() {
        let base = 0x8000_02f0;
        let words = [
            0x1e27_0104, // fmov s4, w8
            0x1e26_0149, // fmov w9, s10
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fmov w/s decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FmovSFromW { rd: 4, rn: 8 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FmovWFromS { rd: 9, rn: 10 }
        ));
    }

    #[test]
    fn decodes_fmov_between_x_and_d_variants() {
        let base = 0x8000_02f8;
        let words = [
            0x9e67_01a0, // fmov d0, x13
            0x9e66_0149, // fmov x9, d10
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fmov x/d decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FmovDFromX { rd: 0, rn: 13 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FmovXFromD { rd: 9, rn: 10 }
        ));
    }

    #[test]
    fn decodes_fcvt_between_s_and_d_variants() {
        let base = 0x8000_02fc;
        let words = [
            0x1e22_c000, // fcvt d0, s0
            0x1e62_4000, // fcvt s0, d0
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fcvt decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FcvtDFromS { rd: 0, rn: 0 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::FcvtSFromD { rd: 0, rn: 0 }
        ));
    }

    #[test]
    fn decodes_fcvtzu_x_from_d_variant() {
        let base = 0x8000_0300;
        let words = [0x9e79_03a2]; // fcvtzu x2, d29

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fcvtzu x,d decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FcvtzuXFromD { rd: 2, rn: 29 }
        ));
    }

    #[test]
    fn decodes_fcvtmu_x_from_d_variant() {
        let base = 0x8000_0304;
        let words = [0x9e71_0381]; // fcvtmu x1, d28

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fcvtmu x,d decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::FcvtmuXFromD { rd: 1, rn: 28 }
        ));
    }

    #[test]
    fn decodes_mov_vector_high_lane_from_x_variant() {
        let base = 0x8000_0308;
        let words = [0x4e18_1da0]; // mov v0.d[1], x13

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("mov v.d[1], x decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::MovVD1FromX { vd: 0, rn: 13 }
        ));
    }

    #[test]
    fn decodes_mov_x_from_vector_d_lane_variants() {
        let base = 0x8000_030c;
        let words = [
            0x4e08_3c01, // mov x1, v0.d[0]
            0x4e18_3d49, // mov x9, v10.d[1]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("mov x, v.d[index] decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::MovXFromVD {
                rd: 1,
                vn: 0,
                index: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::MovXFromVD {
                rd: 9,
                vn: 10,
                index: 1
            }
        ));
    }

    #[test]
    fn decodes_movi_v2d_zero_variant() {
        let base = 0x8000_0310;
        let words = [0x6f00_e400]; // movi v0.2d, #0

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("movi v.2d zero decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::MoviV2DZero { vd: 0 }
        ));
    }

    #[test]
    fn decodes_movi_v4s_zero_variant() {
        let base = 0x8000_0314;
        let words = [0x4f00_041f]; // movi v31.4s, #0

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("movi v.4s zero decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::MoviV4SZero { vd: 31 }
        ));
    }

    #[test]
    fn decodes_vector_add_variants() {
        let base = 0x8000_030c;
        let words = [
            0x4ebe_87fe, // add v30.4s, v31.4s, v30.4s
            0x0e7e_87fe, // add v30.4h, v31.4h, v30.4h
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("vector add decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::AddVector {
                vd: 30,
                vn: 31,
                vm: 30,
                q: true,
                size: 2
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::AddVector {
                vd: 30,
                vn: 31,
                vm: 30,
                q: false,
                size: 1
            }
        ));
    }

    #[test]
    fn decodes_vector_addp_variants() {
        let base = 0x8000_0314;
        let words = [
            0x4e32_be31, // addp v17.16b, v17.16b, v18.16b
            0x0e62_be31, // addp v17.4h, v17.4h, v2.4h
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("vector addp decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::AddpVector {
                vd: 17,
                vn: 17,
                vm: 18,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::AddpVector {
                vd: 17,
                vn: 17,
                vm: 2,
                q: false,
                size: 1
            }
        ));
    }

    #[test]
    fn decodes_vector_uminp_variants() {
        let base = 0x8000_031c;
        let words = [
            0x6e22_ac20, // uminp v0.16b, v1.16b, v2.16b
            0x2e62_ac20, // uminp v0.4h, v1.4h, v2.4h
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("vector uminp decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::UminpVector {
                vd: 0,
                vn: 1,
                vm: 2,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::UminpVector {
                vd: 0,
                vn: 1,
                vm: 2,
                q: false,
                size: 1
            }
        ));
    }

    #[test]
    fn decodes_vector_umaxp_variants() {
        let base = 0x8000_0320;
        let words = [
            0x6e22_a420, // umaxp v0.16b, v1.16b, v2.16b
            0x2e62_a420, // umaxp v0.4h, v1.4h, v2.4h
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("vector umaxp decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::UmaxpVector {
                vd: 0,
                vn: 1,
                vm: 2,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::UmaxpVector {
                vd: 0,
                vn: 1,
                vm: 2,
                q: false,
                size: 1
            }
        ));
    }

    #[test]
    fn decodes_shrn_vector_variants() {
        let base = 0x8000_0328;
        let words = [
            0x0f0c_8400, // shrn v0.8b, v0.8h, #4
            0x4f2c_8400, // shrn2 v0.4s, v0.2d, #20
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("shrn vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::ShrnVector {
                vd: 0,
                vn: 0,
                q: false,
                size: 1,
                shift: 4
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::ShrnVector {
                vd: 0,
                vn: 0,
                q: true,
                size: 3,
                shift: 20
            }
        ));
    }

    #[test]
    fn decodes_cmeq_vector_zero_variants() {
        let base = 0x8000_0310;
        let words = [
            0x4e20_9823, // cmeq v3.16b, v1.16b, #0
            0x0e60_9823, // cmeq v3.4h, v1.4h, #0
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cmeq vector zero decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::CmeqVectorZero {
                vd: 3,
                vn: 1,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::CmeqVectorZero {
                vd: 3,
                vn: 1,
                q: false,
                size: 1
            }
        ));
    }

    #[test]
    fn decodes_cmeq_vector_variants() {
        let base = 0x8000_0318;
        let words = [
            0x6e20_8c25, // cmeq v5.16b, v1.16b, v0.16b
            0x2ea0_8c25, // cmeq v5.2s, v1.2s, v0.2s
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cmeq vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::CmeqVector {
                vd: 5,
                vn: 1,
                vm: 0,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::CmeqVector {
                vd: 5,
                vn: 1,
                vm: 0,
                q: false,
                size: 2
            }
        ));
    }

    #[test]
    fn decodes_cmhs_vector_variants() {
        let base = 0x8000_031c;
        let words = [
            0x6e21_3ca3, // cmhs v3.16b, v5.16b, v1.16b
            0x2e21_3ca3, // cmhs v3.8b, v5.8b, v1.8b
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("cmhs vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::CmhsVector {
                vd: 3,
                vn: 5,
                vm: 1,
                q: true,
                size: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::CmhsVector {
                vd: 3,
                vn: 5,
                vm: 1,
                q: false,
                size: 0
            }
        ));
    }

    #[test]
    fn decodes_bif_vector_variants() {
        let base = 0x8000_0320;
        let words = [
            0x6ee7_1ca3, // bif v3.16b, v5.16b, v7.16b
            0x2ee7_1ca3, // bif v3.8b, v5.8b, v7.8b
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("bif vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::BifVector {
                vd: 3,
                vn: 5,
                vm: 7,
                q: true
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::BifVector {
                vd: 3,
                vn: 5,
                vm: 7,
                q: false
            }
        ));
    }

    #[test]
    fn decodes_and_vector_variants() {
        let base = 0x8000_0324;
        let words = [
            0x4e30_1c71, // and v17.16b, v3.16b, v16.16b
            0x0e30_1c71, // and v17.8b, v3.8b, v16.8b
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("and vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::AndVector {
                vd: 17,
                vn: 3,
                vm: 16,
                q: true
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::AndVector {
                vd: 17,
                vn: 3,
                vm: 16,
                q: false
            }
        ));
    }

    #[test]
    fn decodes_orr_vector_variants() {
        let base = 0x8000_032c;
        let words = [
            0x4ea4_1c71, // orr v17.16b, v3.16b, v4.16b
            0x0ea4_1c71, // orr v17.8b, v3.8b, v4.8b
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("orr vector decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::OrrVector {
                vd: 17,
                vn: 3,
                vm: 4,
                q: true
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::OrrVector {
                vd: 17,
                vn: 3,
                vm: 4,
                q: false
            }
        ));
    }

    #[test]
    fn decodes_dup_v16b_from_w_variant() {
        let base = 0x8000_030c;
        let words = [0x4e01_0c20]; // dup v0.16b, w1

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("dup v16b decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::DupV16BFromW { vd: 0, rn: 1 }
        ));
    }

    #[test]
    fn decodes_dup_v4s_from_w_variant() {
        let base = 0x8000_0310;
        let words = [0x4e04_0c90]; // dup v16.4s, w4

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("dup v4s decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::DupV4SFromW { vd: 16, rn: 4 }
        ));
    }

    #[test]
    fn decodes_ld1_two_v16b_post_index_variant() {
        let base = 0x8000_0314;
        let words = [0x4cdf_a041]; // ld1 {v1.16b, v2.16b}, [x2], #32

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("ld1 {vt.16b, vt2.16b}, [xn], #32 decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::SimdLdMulti {
                q: true,
                nregs: 2,
                interleave: false,
                size: 0,
                rn: 2,
                vt: 1,
                post_index: Some(31)
            }
        ));
    }

    #[test]
    fn decodes_fp_simd_unsigned_immediate_memory_instructions() {
        let base = 0x8000_0310;
        let words = [
            0x3d40_0501, // ldr b1, [x8, #1]
            0x7d40_0502, // ldr h2, [x8, #2]
            0xbd40_0503, // ldr s3, [x8, #4]
            0xfd40_0504, // ldr d4, [x8, #8]
            0x3dc0_0505, // ldr q5, [x8, #16]
            0x3d00_0506, // str b6, [x8, #1]
            0x7d00_0507, // str h7, [x8, #2]
            0xbd00_0508, // str s8, [x8, #4]
            0xfd00_0509, // str d9, [x8, #8]
            0x3d80_050a, // str q10, [x8, #16]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fp simd load/store decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdrFpImm8 {
                rt: 1,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdrFpImm16 {
                rt: 2,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdrFpImm32 {
                rt: 3,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdrFpImm64 {
                rt: 4,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::LdrFpImm128 {
                rt: 5,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::StrFpImm8 {
                rt: 6,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::StrFpImm16 {
                rt: 7,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::StrFpImm32 {
                rt: 8,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::StrFpImm64 {
                rt: 9,
                rn: 8,
                imm12: 1
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::StrFpImm128 {
                rt: 10,
                rn: 8,
                imm12: 1
            }
        ));
    }

    #[test]
    fn decodes_fp_simd_register_offset_q_instructions() {
        let base = 0x8000_0320;
        let words = [
            0x3cf2_691e, // ldr q30, [x8, x18]
            0x3cb2_691e, // str q30, [x8, x18]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fp simd register-offset decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdrFpReg128 {
                rt: 30,
                rn: 8,
                rm: 18,
                option: 0b011,
                shift: false
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::StrFpReg128 {
                rt: 30,
                rn: 8,
                rm: 18,
                option: 0b011,
                shift: false
            }
        ));
    }

    #[test]
    fn decodes_fp_simd_post_pre_index_q_instructions() {
        let base = 0x8000_0330;
        let words = [
            0x3c81_071e, // str q30, [x24], #16
            0x3cc1_071e, // ldr q30, [x24], #16
            0x3c81_0f1e, // str q30, [x24, #16]!
            0x3cc1_0f1e, // ldr q30, [x24, #16]!
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fp simd post/pre q decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::StrPostFp128 {
                rt: 30,
                rn: 24,
                imm9: 16
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdrPostFp128 {
                rt: 30,
                rn: 24,
                imm9: 16
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::StrPreFp128 {
                rt: 30,
                rn: 24,
                imm9: 16
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdrPreFp128 {
                rt: 30,
                rn: 24,
                imm9: 16
            }
        ));
    }

    #[test]
    fn decodes_fp_simd_unscaled_immediate_memory_instructions() {
        let base = 0x8000_0338;
        let words = [
            0x3c40_1101, // ldur b1, [x8, #1]
            0x7c40_2102, // ldur h2, [x8, #2]
            0xbc40_4103, // ldur s3, [x8, #4]
            0xfc40_8104, // ldur d4, [x8, #8]
            0x3cc1_0105, // ldur q5, [x8, #16]
            0x3c1f_f106, // stur b6, [x8, #-1]
            0x7c1f_e107, // stur h7, [x8, #-2]
            0xbc1f_c108, // stur s8, [x8, #-4]
            0xfc1f_8109, // stur d9, [x8, #-8]
            0x3c84_810a, // stur q10, [x8, #72]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("fp simd unscaled decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdurFp8 {
                rt: 1,
                rn: 8,
                imm9: 1
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdurFp16 {
                rt: 2,
                rn: 8,
                imm9: 2
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdurFp32 {
                rt: 3,
                rn: 8,
                imm9: 4
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdurFp64 {
                rt: 4,
                rn: 8,
                imm9: 8
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::LdurFp128 {
                rt: 5,
                rn: 8,
                imm9: 16
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::SturFp8 {
                rt: 6,
                rn: 8,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::SturFp16 {
                rt: 7,
                rn: 8,
                imm9: -2
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::SturFp32 {
                rt: 8,
                rn: 8,
                imm9: -4
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::SturFp64 {
                rt: 9,
                rn: 8,
                imm9: -8
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::SturFp128 {
                rt: 10,
                rn: 8,
                imm9: 72
            }
        ));
    }

    #[test]
    fn decodes_tpidrro_el0_mrs() {
        let base = 0x8000_02c0;
        let words = [encode_mrs(8, SystemRegister::TpidrroEl0)];

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("system decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Mrs { rt, reg } => {
                assert_eq!(rt, 8);
                assert_eq!(reg, SystemRegister::TpidrroEl0);
            }
            _ => panic!("expected mrs tpidrro_el0"),
        }
    }

    #[test]
    fn decodes_cntpct_el0_mrs() {
        let base = 0x8000_02d0;
        let words = [encode_mrs(0, SystemRegister::CntpctEl0)];

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("system decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Mrs { rt, reg } => {
                assert_eq!(rt, 0);
                assert_eq!(reg, SystemRegister::CntpctEl0);
            }
            _ => panic!("expected mrs cntpct_el0"),
        }
    }

    #[test]
    fn decodes_ctr_el0_mrs() {
        let base = 0x8000_02d8;
        let words = [encode_mrs(9, SystemRegister::CtrEl0)];

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("system decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Mrs { rt, reg } => {
                assert_eq!(rt, 9);
                assert_eq!(reg, SystemRegister::CtrEl0);
            }
            _ => panic!("expected mrs ctr_el0"),
        }
    }

    #[test]
    fn decodes_dczid_el0_mrs() {
        let base = 0x8000_02dc;
        let words = [encode_mrs(5, SystemRegister::DczidEl0)];

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("system decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Mrs { rt, reg } => {
                assert_eq!(rt, 5);
                assert_eq!(reg, SystemRegister::DczidEl0);
            }
            _ => panic!("expected mrs dczid_el0"),
        }
    }

    #[test]
    fn decodes_xpaclri_hint() {
        let base = 0x8000_02e0;
        let words = [0xd503_20ff];

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("xpaclri decode should succeed");

        assert!(matches!(block.instructions[0], DecodedInstruction::Xpaclri));
    }

    #[test]
    fn decodes_standard_hint_family() {
        let base = 0x8000_02e4;
        let words = [
            0xd503_201f, // nop
            0xd503_203f, // yield
            0xd503_205f, // wfe
            0xd503_207f, // wfi
            0xd503_209f, // sev
            0xd503_20bf, // sevl
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("hint family decode should succeed");

        assert!(matches!(block.instructions[0], DecodedInstruction::Nop));
        assert!(matches!(block.instructions[1], DecodedInstruction::Yield));
        assert!(matches!(block.instructions[2], DecodedInstruction::Wfe));
        assert!(matches!(block.instructions[3], DecodedInstruction::Wfi));
        assert!(matches!(block.instructions[4], DecodedInstruction::Sev));
        assert!(matches!(block.instructions[5], DecodedInstruction::Sevl));
    }

    #[test]
    fn decodes_bti_with_btype() {
        let base = 0x8000_02fc;
        let words = [0xd503_245f, 0xd503_249f, 0xd503_24df];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("bti decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Bti { btype: 2 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Bti { btype: 4 }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::Bti { btype: 6 }
        ));
    }

    #[test]
    fn decodes_dc_zva() {
        let base = 0x8000_02f0;
        let words = [0xd50b_7423]; // dc zva, x3

        let block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("dc zva decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::DcZva { rt: 3 }
        ));
    }

    #[test]
    fn decodes_register_alu_and_memory_instructions() {
        let base = 0x8000_0300;
        let words = [
            0x8b01_0002, // add x2, x0, x1
            0xea01_005f, // ands xzr, x2, x1
            0xf900_0820, // str x0, [x1, #16]
            0xf940_0823, // ldr x3, [x1, #16]
        ];

        let block = decode_block(base, 4, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::AddReg {
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                assert_eq!(rd, 2);
                assert_eq!(rn, 0);
                assert_eq!(rm, 1);
                assert_eq!(shift, ShiftType::Lsl);
                assert_eq!(amount, 0);
            }
            _ => panic!("expected add reg"),
        }

        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::AndsReg { rd: 31, .. }
        ));

        match block.instructions[2] {
            DecodedInstruction::StrImmU64 { rt, rn, imm12 } => {
                assert_eq!(rt, 0);
                assert_eq!(rn, 1);
                assert_eq!(imm12, 2);
            }
            _ => panic!("expected str imm"),
        }

        match block.instructions[3] {
            DecodedInstruction::LdrImmU64 { rt, rn, imm12 } => {
                assert_eq!(rt, 3);
                assert_eq!(rn, 1);
                assert_eq!(imm12, 2);
            }
            _ => panic!("expected ldr imm"),
        }
    }

    #[test]
    fn decodes_u32_alu_immediate_and_register_instructions() {
        let base = 0x8000_0380;
        let words = [
            0x1100_1420, // add w0, w1, #5
            0x5140_0462, // sub w2, w3, #1, lsl #12
            0x3100_1ca4, // adds w4, w5, #7
            0x7100_20e6, // subs w6, w7, #8
            0x0b0a_0528, // add w8, w9, w10, lsl #1
            0x4b4d_098b, // sub w11, w12, w13, lsr #2
            0x0b2b_0129, // add w9, w9, w11, uxtb
            0x0a10_0dee, // and w14, w15, w16, lsl #3
            0x2a53_1251, // orr w17, w18, w19, lsr #4
            0x4a96_16b4, // eor w20, w21, w22, asr #5
            0x6a19_1b17, // ands w23, w24, w25, lsl #6
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::AddImm32 {
                rd: 0,
                rn: 1,
                imm12: 5,
                shift12: false
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::SubImm32 {
                rd: 2,
                rn: 3,
                imm12: 1,
                shift12: true
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::AddsImm32 {
                rd: 4,
                rn: 5,
                imm12: 7,
                shift12: false
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::SubsImm32 {
                rd: 6,
                rn: 7,
                imm12: 8,
                shift12: false
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::AddReg32 {
                rd: 8,
                rn: 9,
                rm: 10,
                shift: ShiftType::Lsl,
                amount: 1
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::SubReg32 {
                rd: 11,
                rn: 12,
                rm: 13,
                shift: ShiftType::Lsr,
                amount: 2
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::AddExt32 {
                rd: 9,
                rn: 9,
                rm: 11,
                extend: ExtendType::Uxtb,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::AndReg32 {
                rd: 14,
                rn: 15,
                rm: 16,
                shift: ShiftType::Lsl,
                amount: 3
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::OrrReg32 {
                rd: 17,
                rn: 18,
                rm: 19,
                shift: ShiftType::Lsr,
                amount: 4
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::EorReg32 {
                rd: 20,
                rn: 21,
                rm: 22,
                shift: ShiftType::Asr,
                amount: 5
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::AndsReg32 {
                rd: 23,
                rn: 24,
                rm: 25,
                shift: ShiftType::Lsl,
                amount: 6
            }
        ));
    }

    #[test]
    fn decodes_logical_not_register_variants() {
        let base = 0x8000_03c0;
        let words = [
            0x8a22_0023, // bic x3, x1, x2
            0xaa22_0024, // orn x4, x1, x2
            0xca22_0025, // eon x5, x1, x2
            0xea22_0026, // bics x6, x1, x2
            0x0a28_00e9, // bic w9, w7, w8
            0x2a28_00ea, // orn w10, w7, w8
            0x4a28_00eb, // eon w11, w7, w8
            0x6a28_00ec, // bics w12, w7, w8
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::BicReg {
                rd: 3,
                rn: 1,
                rm: 2,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::OrnReg {
                rd: 4,
                rn: 1,
                rm: 2,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::EonReg {
                rd: 5,
                rn: 1,
                rm: 2,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::BicsReg {
                rd: 6,
                rn: 1,
                rm: 2,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::BicReg32 {
                rd: 9,
                rn: 7,
                rm: 8,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::OrnReg32 {
                rd: 10,
                rn: 7,
                rm: 8,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::EonReg32 {
                rd: 11,
                rn: 7,
                rm: 8,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::BicsReg32 {
                rd: 12,
                rn: 7,
                rm: 8,
                shift: ShiftType::Lsl,
                amount: 0,
            }
        ));
    }

    #[test]
    fn decodes_u32_move_wide_instructions() {
        let base = 0x8000_03f0;
        let words = [
            0x5280_0001, // movz w1, #0
            0x12a2_4685, // movn w5, #0x1234, lsl #16
            0x72a2_4685, // movk w5, #0x1234, lsl #16
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::MovZ32 {
                rd: 1,
                imm16: 0,
                shift: 0
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::MovN32 {
                rd: 5,
                imm16: 0x1234,
                shift: 16
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::MovK32 {
                rd: 5,
                imm16: 0x1234,
                shift: 16
            }
        ));
    }

    #[test]
    fn decodes_logical_immediate_instructions() {
        let base = 0x8000_03fc;
        let words = [
            0x1200_1c29, // and w9, w1, #0xff
            0x3200_1c4a, // orr w10, w2, #0xff
            0x5200_1c6b, // eor w11, w3, #0xff
            0x7200_1c8c, // ands w12, w4, #0xff
            0x9240_1ca1, // and x1, x5, #0xff
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::AndImm32 {
                rd: 9,
                rn: 1,
                imm: 0xff
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::OrrImm32 {
                rd: 10,
                rn: 2,
                imm: 0xff
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::EorImm32 {
                rd: 11,
                rn: 3,
                imm: 0xff
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::AndsImm32 {
                rd: 12,
                rn: 4,
                imm: 0xff
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::AndImm {
                rd: 1,
                rn: 5,
                imm: 0xff
            }
        ));
    }

    #[test]
    fn decodes_bitfield_shift_aliases() {
        let base = 0x8000_0410;
        let words = [
            0xd343_fd29, // lsr x9, x9, #3
            0xd37c_ed08, // lsl x8, x8, #4
            0x9345_fd4a, // asr x10, x10, #5
            0x5303_7dad, // lsr w13, w13, #3
            0x531c_6d8c, // lsl w12, w12, #4
            0x1307_7d6b, // asr w11, w11, #7
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LsrImm {
                rd: 9,
                rn: 9,
                amount: 3
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LslImm {
                rd: 8,
                rn: 8,
                amount: 4
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::AsrImm {
                rd: 10,
                rn: 10,
                amount: 5
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LsrImm32 {
                rd: 13,
                rn: 13,
                amount: 3
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::LslImm32 {
                rd: 12,
                rn: 12,
                amount: 4
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::AsrImm32 {
                rd: 11,
                rn: 11,
                amount: 7
            }
        ));
    }

    #[test]
    fn decodes_generic_ubfm_instruction() {
        let base = 0x8000_0428;
        let words = [0xd360_a50b];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Ubfm {
                rd: 11,
                rn: 8,
                immr: 32,
                imms: 41
            }
        ));
    }

    #[test]
    fn decodes_generic_sbfm_instruction() {
        let base = 0x8000_042a;
        let words = [0x9340_7c00];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Sbfm {
                rd: 0,
                rn: 0,
                immr: 0,
                imms: 31
            }
        ));
    }

    #[test]
    fn decodes_generic_bfm_instruction() {
        let base = 0x8000_042c;
        let words = [0xb360_7d4b];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Bfm {
                rd: 11,
                rn: 10,
                immr: 32,
                imms: 31
            }
        ));
    }

    #[test]
    fn decodes_adr_adrp_and_blr() {
        let base = 0x8000_0400;
        let words = [
            0x1000_0040, // adr x0, #8
            0x9000_0001, // adrp x1, #0
            0xd63f_0040, // blr x2
        ];

        let block = decode_block(base, 8, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        match block.instructions[0] {
            DecodedInstruction::Adr { rd, target } => {
                assert_eq!(rd, 0);
                assert_eq!(target, base + 8);
            }
            _ => panic!("expected adr"),
        }

        match block.instructions[1] {
            DecodedInstruction::Adrp { rd, target } => {
                assert_eq!(rd, 1);
                assert_eq!(target, base & !0xfff);
            }
            _ => panic!("expected adrp"),
        }

        match block.instructions[2] {
            DecodedInstruction::Blr { rn } => assert_eq!(rn, 2),
            _ => panic!("expected blr"),
        }
    }

    #[test]
    fn decodes_u32_and_unscaled_memory_instructions() {
        let base = 0x8000_0450;
        let words = [
            0xb900_0820, // str w0, [x1, #8]
            0xb940_0822, // ldr w2, [x1, #8]
            0xf81f_8020, // stur x0, [x1, #-8]
            0xf85f_8023, // ldur x3, [x1, #-8]
            0xb81f_c024, // stur w4, [x1, #-4]
            0xb85f_c025, // ldur w5, [x1, #-4]
            0xf840_8423, // ldr x3, [x1], #8
            0xf85f_8c24, // ldr x4, [x1, #-8]!
            0xf801_0425, // str x5, [x1], #16
            0xf81f_0c26, // str x6, [x1, #-16]!
            0xb840_4427, // ldr w7, [x1], #4
            0xb85f_cc28, // ldr w8, [x1, #-4]!
            0xb800_c429, // str w9, [x1], #12
            0xb81f_4c2a, // str w10, [x1, #-12]!
            0x3840_142b, // ldrb w11, [x1], #1
            0x385f_fc2c, // ldrb w12, [x1, #-1]!
            0x3800_242d, // strb w13, [x1], #2
            0x381f_ec2e, // strb w14, [x1, #-2]!
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::StrImmU32 {
                rt: 0,
                rn: 1,
                imm12: 2
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdrImmU32 {
                rt: 2,
                rn: 1,
                imm12: 2
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::SturI64 {
                rt: 0,
                rn: 1,
                imm9: -8
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdurI64 {
                rt: 3,
                rn: 1,
                imm9: -8
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::SturI32 {
                rt: 4,
                rn: 1,
                imm9: -4
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::LdurI32 {
                rt: 5,
                rn: 1,
                imm9: -4
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::LdrPostI64 {
                rt: 3,
                rn: 1,
                imm9: 8
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::LdrPreI64 {
                rt: 4,
                rn: 1,
                imm9: -8
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::StrPostI64 {
                rt: 5,
                rn: 1,
                imm9: 16
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::StrPreI64 {
                rt: 6,
                rn: 1,
                imm9: -16
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::LdrPostI32 {
                rt: 7,
                rn: 1,
                imm9: 4
            }
        ));
        assert!(matches!(
            block.instructions[11],
            DecodedInstruction::LdrPreI32 {
                rt: 8,
                rn: 1,
                imm9: -4
            }
        ));
        assert!(matches!(
            block.instructions[12],
            DecodedInstruction::StrPostI32 {
                rt: 9,
                rn: 1,
                imm9: 12
            }
        ));
        assert!(matches!(
            block.instructions[13],
            DecodedInstruction::StrPreI32 {
                rt: 10,
                rn: 1,
                imm9: -12
            }
        ));
        assert!(matches!(
            block.instructions[14],
            DecodedInstruction::LdrPostI8 {
                rt: 11,
                rn: 1,
                imm9: 1
            }
        ));
        assert!(matches!(
            block.instructions[15],
            DecodedInstruction::LdrPreI8 {
                rt: 12,
                rn: 1,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[16],
            DecodedInstruction::StrPostI8 {
                rt: 13,
                rn: 1,
                imm9: 2
            }
        ));
        assert!(matches!(
            block.instructions[17],
            DecodedInstruction::StrPreI8 {
                rt: 14,
                rn: 1,
                imm9: -2
            }
        ));
    }

    #[test]
    fn decodes_register_offset_memory_instructions() {
        let base = 0x8000_0490;
        let words = [
            0xf86e_7970, // ldr x16, [x11, x14, lsl #3]
            0xf82e_7971, // str x17, [x11, x14, lsl #3]
            0xb86e_7972, // ldr w18, [x11, x14, lsl #2]
            0xb82e_7973, // str w19, [x11, x14, lsl #2]
            0xf86e_4974, // ldr x20, [x11, w14, uxtw]
            0x3868_6b2a, // ldrb w10, [x25, x8]
            0x3828_6b2b, // strb w11, [x25, x8]
            0xb8a8_7b2c, // ldrsw x12, [x25, x8, lsl #2]
            0x78a8_7b2d, // ldrsh x13, [x25, x8, lsl #1]
            0x38e8_6b2e, // ldrsb w14, [x25, x8]
            0x7868_7b2f, // ldrh w15, [x25, x8, lsl #1]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdrRegU64 {
                rt: 16,
                rn: 11,
                rm: 14,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::StrRegU64 {
                rt: 17,
                rn: 11,
                rm: 14,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdrRegU32 {
                rt: 18,
                rn: 11,
                rm: 14,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::StrRegU32 {
                rt: 19,
                rn: 11,
                rm: 14,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::LdrRegU64 {
                rt: 20,
                rn: 11,
                rm: 14,
                option: 0b010,
                shift: false,
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::LdrRegU8 {
                rt: 10,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: false,
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::StrRegU8 {
                rt: 11,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: false,
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::LdrswReg64 {
                rt: 12,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::LdrshReg64 {
                rt: 13,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: true,
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::LdrsbReg32 {
                rt: 14,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: false,
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::LdrRegU16 {
                rt: 15,
                rn: 25,
                rm: 8,
                option: 0b011,
                shift: true,
            }
        ));
    }

    #[test]
    fn decodes_byte_halfword_and_signed_memory_instructions() {
        let base = 0x8000_04b0;
        let words = [
            0x3900_2020, // strb w0, [x1, #8]
            0x3940_2022, // ldrb w2, [x1, #8]
            0x7900_0c23, // strh w3, [x1, #6]
            0x7940_0c24, // ldrh w4, [x1, #6]
            0x381f_f025, // sturb w5, [x1, #-1]
            0x385f_f026, // ldurb w6, [x1, #-1]
            0x781f_e027, // sturh w7, [x1, #-2]
            0x785f_e028, // ldurh w8, [x1, #-2]
            0x3980_2029, // ldrsb x9, [x1, #8]
            0x39c0_202a, // ldrsb w10, [x1, #8]
            0x7980_0c2b, // ldrsh x11, [x1, #6]
            0x79c0_0c2c, // ldrsh w12, [x1, #6]
            0xb980_0c2d, // ldrsw x13, [x1, #12]
            0x389f_f02e, // ldursb x14, [x1, #-1]
            0x38df_f02f, // ldursb w15, [x1, #-1]
            0x789f_e030, // ldursh x16, [x1, #-2]
            0x78df_e031, // ldursh w17, [x1, #-2]
            0xb89f_c032, // ldursw x18, [x1, #-4]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::StrImmU8 {
                rt: 0,
                rn: 1,
                imm12: 8
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdrImmU8 {
                rt: 2,
                rn: 1,
                imm12: 8
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::StrImmU16 {
                rt: 3,
                rn: 1,
                imm12: 3
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdrImmU16 {
                rt: 4,
                rn: 1,
                imm12: 3
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::SturI8 {
                rt: 5,
                rn: 1,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::LdurI8 {
                rt: 6,
                rn: 1,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::SturI16 {
                rt: 7,
                rn: 1,
                imm9: -2
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::LdurI16 {
                rt: 8,
                rn: 1,
                imm9: -2
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::LdrsbImm64 {
                rt: 9,
                rn: 1,
                imm12: 8
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::LdrsbImm32 {
                rt: 10,
                rn: 1,
                imm12: 8
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::LdrshImm64 {
                rt: 11,
                rn: 1,
                imm12: 3
            }
        ));
        assert!(matches!(
            block.instructions[11],
            DecodedInstruction::LdrshImm32 {
                rt: 12,
                rn: 1,
                imm12: 3
            }
        ));
        assert!(matches!(
            block.instructions[12],
            DecodedInstruction::LdrswImm64 {
                rt: 13,
                rn: 1,
                imm12: 3
            }
        ));
        assert!(matches!(
            block.instructions[13],
            DecodedInstruction::LdursbI64 {
                rt: 14,
                rn: 1,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[14],
            DecodedInstruction::LdursbI32 {
                rt: 15,
                rn: 1,
                imm9: -1
            }
        ));
        assert!(matches!(
            block.instructions[15],
            DecodedInstruction::LdurshI64 {
                rt: 16,
                rn: 1,
                imm9: -2
            }
        ));
        assert!(matches!(
            block.instructions[16],
            DecodedInstruction::LdurshI32 {
                rt: 17,
                rn: 1,
                imm9: -2
            }
        ));
        assert!(matches!(
            block.instructions[17],
            DecodedInstruction::LdurswI64 {
                rt: 18,
                rn: 1,
                imm9: -4
            }
        ));
    }

    #[test]
    fn decodes_multiply_and_divide_instructions() {
        let base = 0x8000_0500;
        let words = [
            0x9b027c20,  // mul x0, x1, x2
            0x9b020023,  // madd x3, x1, x2, x0
            0x9b028024,  // msub x4, x1, x2, x0
            0x9ac10865,  // udiv x5, x3, x1
            0x9ac10c66,  // sdiv x6, x3, x1
            0x9b42_7c27, // smulh x7, x1, x2
            0x9bc2_7c28, // umulh x8, x1, x2
            0x9bb6_2515, // umaddl x21, w8, w22, x9
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Mul {
                rd: 0,
                rn: 1,
                rm: 2
            }
        ));

        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Madd {
                rd: 3,
                rn: 1,
                rm: 2,
                ra: 0
            }
        ));

        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::Msub {
                rd: 4,
                rn: 1,
                rm: 2,
                ra: 0
            }
        ));

        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::Udiv {
                rd: 5,
                rn: 3,
                rm: 1
            }
        ));

        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::Sdiv {
                rd: 6,
                rn: 3,
                rm: 1
            }
        ));

        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::Smulh {
                rd: 7,
                rn: 1,
                rm: 2
            }
        ));

        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::Umulh {
                rd: 8,
                rn: 1,
                rm: 2
            }
        ));

        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::Umaddl {
                rd: 21,
                rn: 8,
                rm: 22,
                ra: 9
            }
        ));
    }

    #[test]
    fn decodes_conditional_select_and_variable_shift_instructions() {
        let base = 0x8000_0580;
        let words = [
            0x9a82_0020, // csel x0, x1, x2, eq
            0x9a85_1483, // csinc x3, x4, x5, ne
            0xda88_a0e6, // csinv x6, x7, x8, ge
            0xda8b_b549, // csneg x9, x10, x11, lt
            0x9ad0_21ee, // lslv x14, x15, x16
            0x9ad3_2651, // lsrv x17, x18, x19
            0x9ad6_2ab4, // asrv x20, x21, x22
            0x9ad9_2f17, // rorv x23, x24, x25
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Csel {
                rd: 0,
                rn: 1,
                rm: 2,
                condition: BranchCondition::Eq
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Csinc {
                rd: 3,
                rn: 4,
                rm: 5,
                condition: BranchCondition::Ne
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::Csinv {
                rd: 6,
                rn: 7,
                rm: 8,
                condition: BranchCondition::Ge
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::Csneg {
                rd: 9,
                rn: 10,
                rm: 11,
                condition: BranchCondition::Lt
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::Lslv {
                rd: 14,
                rn: 15,
                rm: 16
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::Lsrv {
                rd: 17,
                rn: 18,
                rm: 19
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::Asrv {
                rd: 20,
                rn: 21,
                rm: 22
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::Rorv {
                rd: 23,
                rn: 24,
                rm: 25
            }
        ));
    }

    #[test]
    fn decodes_u32_conditional_shift_and_mul_div_instructions() {
        let base = 0x8000_05c0;
        let words = [
            0x1a82_0020, // csel w0, w1, w2, eq
            0x1a85_1483, // csinc w3, w4, w5, ne
            0x5a88_a0e6, // csinv w6, w7, w8, ge
            0x5a8b_b549, // csneg w9, w10, w11, lt
            0x1ace_21ac, // lslv w12, w13, w14
            0x1ad1_260f, // lsrv w15, w16, w17
            0x1ad4_2a72, // asrv w18, w19, w20
            0x1ad7_2ed5, // rorv w21, w22, w23
            0x1b1a_7f38, // mul w24, w25, w26
            0x1b1d_039b, // madd w27, w28, w29, w0
            0x1b03_9041, // msub w1, w2, w3, w4
            0x1ac7_08c5, // udiv w5, w6, w7
            0x1aca_0d28, // sdiv w8, w9, w10
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Csel32 {
                rd: 0,
                rn: 1,
                rm: 2,
                condition: BranchCondition::Eq
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Csinc32 {
                rd: 3,
                rn: 4,
                rm: 5,
                condition: BranchCondition::Ne
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::Csinv32 {
                rd: 6,
                rn: 7,
                rm: 8,
                condition: BranchCondition::Ge
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::Csneg32 {
                rd: 9,
                rn: 10,
                rm: 11,
                condition: BranchCondition::Lt
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::Lslv32 {
                rd: 12,
                rn: 13,
                rm: 14
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::Lsrv32 {
                rd: 15,
                rn: 16,
                rm: 17
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::Asrv32 {
                rd: 18,
                rn: 19,
                rm: 20
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::Rorv32 {
                rd: 21,
                rn: 22,
                rm: 23
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::Mul32 {
                rd: 24,
                rn: 25,
                rm: 26
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::Madd32 {
                rd: 27,
                rn: 28,
                rm: 29,
                ra: 0
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::Msub32 {
                rd: 1,
                rn: 2,
                rm: 3,
                ra: 4
            }
        ));
        assert!(matches!(
            block.instructions[11],
            DecodedInstruction::Udiv32 {
                rd: 5,
                rn: 6,
                rm: 7
            }
        ));
        assert!(matches!(
            block.instructions[12],
            DecodedInstruction::Sdiv32 {
                rd: 8,
                rn: 9,
                rm: 10
            }
        ));
    }

    #[test]
    fn decodes_extr_and_ror_immediate_aliases() {
        let base = 0x8000_05f8;
        let words = [
            0x93c1_2002, // extr x2, x0, x1, #8
            0x1385_3083, // extr w3, w4, w5, #12
            0x93c8_3908, // ror x8, x8, #14 (alias of extr x8, x8, x8, #14)
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Extr {
                rd: 2,
                rn: 0,
                rm: 1,
                lsb: 8
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Extr32 {
                rd: 3,
                rn: 4,
                rm: 5,
                lsb: 12
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::Extr {
                rd: 8,
                rn: 8,
                rm: 8,
                lsb: 14
            }
        ));
    }

    #[test]
    fn decodes_rbit_instructions() {
        let base = 0x8000_0610;
        let words = [
            0xdac0_01ae, // rbit x14, x13
            0x5ac0_012a, // rbit w10, w9
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Rbit { rd: 14, rn: 13 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Rbit32 { rd: 10, rn: 9 }
        ));
    }

    #[test]
    fn decodes_clz_instructions() {
        let base = 0x8000_0620;
        let words = [
            0xdac0_11ce, // clz x14, x14
            0x5ac0_114a, // clz w10, w10
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Clz { rd: 14, rn: 14 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Clz32 { rd: 10, rn: 10 }
        ));
    }

    #[test]
    fn decodes_rev_instructions() {
        let base = 0x8000_0628;
        let words = [
            0xdac0_0c84, // rev x4, x4
            0x5ac0_08a2, // rev w2, w5
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::Rev { rd: 4, rn: 4 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::Rev32 { rd: 2, rn: 5 }
        ));
    }

    #[test]
    fn decodes_pair_and_exclusive_instructions() {
        let base = 0x8000_0600;
        let words = [
            0xa901_0820, // stp x0, x2, [x1, #16]
            0xa941_1023, // ldp x3, x4, [x1, #16]
            0xa9bf_1825, // stp x5, x6, [x1, #-16]!
            0xa8c1_2027, // ldp x7, x8, [x1], #16
            0x2901_2829, // stp w9, w10, [x1, #8]
            0x2941_302b, // ldp w11, w12, [x1, #8]
            0x6941_382d, // ldpsw x13, x14, [x1, #8]
            0xc85f_7c29, // ldxr x9, [x1]
            0xc80a_7c22, // stxr w10, x2, [x1]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::StpI64 {
                rt: 0,
                rt2: 2,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdpI64 {
                rt: 3,
                rt2: 4,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::StpI64 {
                rt: 5,
                rt2: 6,
                rn: 1,
                imm7: -2,
                mode: PairAddressMode::PreIndex
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdpI64 {
                rt: 7,
                rt2: 8,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::PostIndex
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::StpI32 {
                rt: 9,
                rt2: 10,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::LdpI32 {
                rt: 11,
                rt2: 12,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::Ldpsw {
                rt: 13,
                rt2: 14,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::LdxrU64 { rt: 9, rn: 1 }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::StxrU64 {
                rs: 10,
                rt: 2,
                rn: 1
            }
        ));
    }

    #[test]
    fn decodes_simd_pair_instructions() {
        let base = 0x8000_06a0;
        let words = [
            0x2d00_0420, // stp s0, s1, [x1]
            0x2d40_0c22, // ldp s2, s3, [x1]
            0x2dbf_1424, // stp s4, s5, [x1, #-8]!
            0x2cc1_1c26, // ldp s6, s7, [x1], #8
            0x6d00_2428, // stp d8, d9, [x1]
            0x6d40_2c2a, // ldp d10, d11, [x1]
            0x6dbf_342c, // stp d12, d13, [x1, #-16]!
            0x6cc1_3c2e, // ldp d14, d15, [x1], #16
            0xad00_0420, // stp q0, q1, [x1]
            0xad40_0c22, // ldp q2, q3, [x1]
            0xadbf_1424, // stp q4, q5, [x1, #-32]!
            0xacc1_1c26, // ldp q6, q7, [x1], #32
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::StpFp32 {
                rt: 0,
                rt2: 1,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdpFp32 {
                rt: 2,
                rt2: 3,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::StpFp32 {
                rt: 4,
                rt2: 5,
                rn: 1,
                imm7: -2,
                mode: PairAddressMode::PreIndex
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdpFp32 {
                rt: 6,
                rt2: 7,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::PostIndex
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::StpFp64 {
                rt: 8,
                rt2: 9,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::LdpFp64 {
                rt: 10,
                rt2: 11,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::StpFp64 {
                rt: 12,
                rt2: 13,
                rn: 1,
                imm7: -2,
                mode: PairAddressMode::PreIndex
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::LdpFp64 {
                rt: 14,
                rt2: 15,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::PostIndex
            }
        ));
        assert!(matches!(
            block.instructions[8],
            DecodedInstruction::StpV128 {
                rt: 0,
                rt2: 1,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[9],
            DecodedInstruction::LdpV128 {
                rt: 2,
                rt2: 3,
                rn: 1,
                imm7: 0,
                mode: PairAddressMode::SignedOffset
            }
        ));
        assert!(matches!(
            block.instructions[10],
            DecodedInstruction::StpV128 {
                rt: 4,
                rt2: 5,
                rn: 1,
                imm7: -2,
                mode: PairAddressMode::PreIndex
            }
        ));
        assert!(matches!(
            block.instructions[11],
            DecodedInstruction::LdpV128 {
                rt: 6,
                rt2: 7,
                rn: 1,
                imm7: 2,
                mode: PairAddressMode::PostIndex
            }
        ));
    }

    #[test]
    fn decodes_tbz_tbnz_and_exclusive_variants() {
        let base = 0x8000_0700;
        let words = [
            0x3628_0043, // tbz w3, #5, +8
            0x3738_0024, // tbnz w4, #7, +4
            0x885f_7c41, // ldxr w1, [x2]
            0x8803_7c44, // stxr w3, w4, [x2]
            0xc85f_fcc5, // ldaxr x5, [x6]
            0x885f_fd07, // ldaxr w7, [x8]
            0xc809_fd6a, // stlxr w9, x10, [x11]
            0x880c_fdcd, // stlxr w12, w13, [x14]
        ];

        let tbz_block = decode_block(base, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("tbz decode should succeed");
        assert!(matches!(
            tbz_block.instructions[0],
            DecodedInstruction::Tbz {
                rt: 3,
                bit: 5,
                target
            } if target == base + 8
        ));

        let tbnz_block = decode_block(base + 4, 1, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("tbnz decode should succeed");
        assert!(matches!(
            tbnz_block.instructions[0],
            DecodedInstruction::Tbnz {
                rt: 4,
                bit: 7,
                target
            } if target == base + 8
        ));

        let block = decode_block(base + 8, words.len() - 2, |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("exclusive variants decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdxrU32 { rt: 1, rn: 2 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::StxrU32 {
                rs: 3,
                rt: 4,
                rn: 2
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdaxrU64 { rt: 5, rn: 6 }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdaxrU32 { rt: 7, rn: 8 }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::StlxrU64 {
                rs: 9,
                rt: 10,
                rn: 11
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::StlxrU32 {
                rs: 12,
                rt: 13,
                rn: 14
            }
        ));
    }

    #[test]
    fn decodes_ldar_stlr_variants() {
        let base = 0x8000_0800;
        let words = [
            0xc8df_fc41, // ldar x1, [x2]
            0x88df_fc83, // ldar w3, [x4]
            0x08df_fd28, // ldarb w8, [x9]
            0x48df_fd6a, // ldarh w10, [x11]
            0xc89f_fcc5, // stlr x5, [x6]
            0x889f_fd07, // stlr w7, [x8]
            0x089f_fdac, // stlrb w12, [x13]
            0x489f_fdee, // stlrh w14, [x15]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("ldar/stlr decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdarU64 { rt: 1, rn: 2 }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::LdarU32 { rt: 3, rn: 4 }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdarbU32 { rt: 8, rn: 9 }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::LdarhU32 { rt: 10, rn: 11 }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::StlrU64 { rt: 5, rn: 6 }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::StlrU32 { rt: 7, rn: 8 }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::StlrbU32 { rt: 12, rn: 13 }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::StlrhU32 { rt: 14, rn: 15 }
        ));
    }

    #[test]
    fn decodes_conditional_compare_variants() {
        let base = 0x8000_08c0;
        let words = [
            encode_ccmp_reg(true, 1, 2, 0xf, 0x0),
            encode_ccmp_reg(false, 3, 4, 0xe, 0x1),
            encode_ccmp_imm(true, 5, 6, 0xd, 0xa),
            encode_ccmp_imm(false, 7, 8, 0xc, 0xb),
            encode_ccmn_reg(true, 9, 10, 0x3, 0xc),
            encode_ccmn_reg(false, 11, 12, 0x4, 0xd),
            encode_ccmn_imm(true, 13, 14, 0x5, 0xe),
            encode_ccmn_imm(false, 15, 16, 0x6, 0xf),
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("conditional compare decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::CcmpReg {
                rn: 1,
                rm: 2,
                nzcv: 0xf,
                condition: BranchCondition::Eq
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::CcmpReg32 {
                rn: 3,
                rm: 4,
                nzcv: 0xe,
                condition: BranchCondition::Ne
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::CcmpImm {
                rn: 5,
                imm5: 6,
                nzcv: 0xd,
                condition: BranchCondition::Ge
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::CcmpImm32 {
                rn: 7,
                imm5: 8,
                nzcv: 0xc,
                condition: BranchCondition::Lt
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::CcmnReg {
                rn: 9,
                rm: 10,
                nzcv: 0x3,
                condition: BranchCondition::Gt
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::CcmnReg32 {
                rn: 11,
                rm: 12,
                nzcv: 0x4,
                condition: BranchCondition::Le
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::CcmnImm {
                rn: 13,
                imm5: 14,
                nzcv: 0x5,
                condition: BranchCondition::Al
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::CcmnImm32 {
                rn: 15,
                imm5: 16,
                nzcv: 0x6,
                condition: BranchCondition::Nv
            }
        ));
    }

    #[test]
    fn decodes_pair_exclusive_variants() {
        let base = 0x8000_0900;
        let words = [
            0xc87f_0861, // ldxp x1, x2, [x3]
            0xc824_18e5, // stxp w4, x5, x6, [x7]
            0xc87f_a548, // ldaxp x8, x9, [x10]
            0xc82b_b5cc, // stlxp w11, x12, x13, [x14]
            0x887f_0861, // ldxp w1, w2, [x3]
            0x8824_18e5, // stxp w4, w5, w6, [x7]
            0x887f_a548, // ldaxp w8, w9, [x10]
            0x882b_b5cc, // stlxp w11, w12, w13, [x14]
        ];

        let block = decode_block(base, words.len(), |pc| {
            let idx = ((pc - base) / 4) as usize;
            words.get(idx).copied()
        })
        .expect("pair exclusive decode should succeed");

        assert!(matches!(
            block.instructions[0],
            DecodedInstruction::LdxpU64 {
                rt: 1,
                rt2: 2,
                rn: 3
            }
        ));
        assert!(matches!(
            block.instructions[1],
            DecodedInstruction::StxpU64 {
                rs: 4,
                rt: 5,
                rt2: 6,
                rn: 7
            }
        ));
        assert!(matches!(
            block.instructions[2],
            DecodedInstruction::LdaxpU64 {
                rt: 8,
                rt2: 9,
                rn: 10
            }
        ));
        assert!(matches!(
            block.instructions[3],
            DecodedInstruction::StlxpU64 {
                rs: 11,
                rt: 12,
                rt2: 13,
                rn: 14
            }
        ));
        assert!(matches!(
            block.instructions[4],
            DecodedInstruction::LdxpU32 {
                rt: 1,
                rt2: 2,
                rn: 3
            }
        ));
        assert!(matches!(
            block.instructions[5],
            DecodedInstruction::StxpU32 {
                rs: 4,
                rt: 5,
                rt2: 6,
                rn: 7
            }
        ));
        assert!(matches!(
            block.instructions[6],
            DecodedInstruction::LdaxpU32 {
                rt: 8,
                rt2: 9,
                rn: 10
            }
        ));
        assert!(matches!(
            block.instructions[7],
            DecodedInstruction::StlxpU32 {
                rs: 11,
                rt: 12,
                rt2: 13,
                rn: 14
            }
        ));
    }

    #[test]
    fn decodes_additional_vector_insert_and_dup_variants() {
        assert!(matches!(
            decode_single(0x8000_0910, 0x6e0c_0445),
            DecodedInstruction::InsElement {
                vd: 5,
                vn: 2,
                dst_index: 1,
                src_index: 0,
                size: 2,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0914, 0x0e02_0ce4),
            DecodedInstruction::DupGeneralVector {
                vd: 4,
                rn: 7,
                q: false,
                size: 1,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0918, 0x4e18_0c41),
            DecodedInstruction::DupGeneralVector {
                vd: 1,
                rn: 2,
                q: true,
                size: 3,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_091c, 0x4e18_0443),
            DecodedInstruction::DupElementVector {
                vd: 3,
                vn: 2,
                q: true,
                size: 3,
                index: 1,
            }
        ));
    }

    #[test]
    fn decodes_additional_vector_misc_variants() {
        assert!(matches!(
            decode_single(0x8000_0920, 0x6e23_34a7),
            DecodedInstruction::CmhiVector {
                vd: 7,
                vn: 5,
                vm: 3,
                q: true,
                size: 0,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0924, 0x2e20_4422),
            DecodedInstruction::UshlVector {
                vd: 2,
                vn: 1,
                vm: 0,
                q: false,
                size: 0,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0928, 0x6e20_b824),
            DecodedInstruction::NegVector {
                vd: 4,
                vn: 1,
                q: true,
                size: 0,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_092c, 0x0e60_0844),
            DecodedInstruction::Rev64Vector {
                vd: 4,
                vn: 2,
                q: false,
                size: 1,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0930, 0x4e20_5844),
            DecodedInstruction::CntVector {
                vd: 4,
                vn: 2,
                q: true,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0934, 0x6e20_5865),
            DecodedInstruction::NotVector {
                vd: 5,
                vn: 3,
                q: true,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0938, 0x2e60_5866),
            DecodedInstruction::RbitVector {
                vd: 6,
                vn: 3,
                q: false,
            }
        ));
    }

    #[test]
    fn decodes_additional_vector_reduction_variants() {
        assert!(matches!(
            decode_single(0x8000_0940, 0x0e30_3867),
            DecodedInstruction::SaddlvVector {
                vd: 7,
                vn: 3,
                q: false,
                size: 0,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0944, 0x6e70_3888),
            DecodedInstruction::UaddlvVector {
                vd: 8,
                vn: 4,
                q: true,
                size: 1,
            }
        ));
    }

    #[test]
    fn decodes_ins_general_vector_from_gpr_variants() {
        assert!(matches!(
            decode_single(0x8000_0948, 0x4e0f_1c24),
            DecodedInstruction::InsGeneralVector {
                vd: 4,
                rn: 1,
                dst_index: 7,
                size: 0,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_094c, 0x4e0e_1c44),
            DecodedInstruction::InsGeneralVector {
                vd: 4,
                rn: 2,
                dst_index: 3,
                size: 1,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0950, 0x4e0c_1c1f),
            DecodedInstruction::InsGeneralVector {
                vd: 31,
                rn: 0,
                dst_index: 1,
                size: 2,
            }
        ));

        assert!(matches!(
            decode_single(0x8000_0954, 0x4e08_1c41),
            DecodedInstruction::InsGeneralVector {
                vd: 1,
                rn: 2,
                dst_index: 0,
                size: 3,
            }
        ));
    }
}
