use crate::value::Value;
use std::collections::HashMap;

pub type Reg = u8;
pub type ConstIdx = u16;
pub type Offset = i16;

/// Packed 32-bit instruction format (like Lua)
///
/// Two formats:
/// - ABC:  [opcode:8][A:8][B:8][C:8] - for 3-operand instructions
/// - ABx:  [opcode:8][A:8][Bx:16]    - for instructions with 16-bit constant/offset
///
/// The sBx (signed Bx) variant interprets Bx as i16 for jump offsets.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Op(pub u32);

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opcode = self.opcode();
        match opcode {
            Self::LOAD_CONST => write!(f, "LoadConst({}, {})", self.a(), self.bx()),
            Self::LOAD_NIL => write!(f, "LoadNil({})", self.a()),
            Self::LOAD_TRUE => write!(f, "LoadTrue({})", self.a()),
            Self::LOAD_FALSE => write!(f, "LoadFalse({})", self.a()),
            Self::MOVE => write!(f, "Move({}, {})", self.a(), self.b()),
            Self::GET_GLOBAL => write!(f, "GetGlobal({}, {})", self.a(), self.bx()),
            Self::SET_GLOBAL => write!(f, "SetGlobal({}, {})", self.bx(), self.a()),
            Self::CLOSURE => write!(f, "Closure({}, {})", self.a(), self.bx()),
            Self::JUMP => write!(f, "Jump({})", self.sbx()),
            Self::JUMP_IF_FALSE => write!(f, "JumpIfFalse({}, {})", self.a(), self.sbx()),
            Self::JUMP_IF_TRUE => write!(f, "JumpIfTrue({}, {})", self.a(), self.sbx()),
            Self::CALL => write!(f, "Call({}, {}, {})", self.a(), self.b(), self.c()),
            Self::CALL_GLOBAL => write!(f, "CallGlobal({}, {}, {})", self.a(), self.b(), self.c()),
            Self::TAIL_CALL => write!(f, "TailCall({}, {})", self.a(), self.b()),
            Self::TAIL_CALL_GLOBAL => write!(f, "TailCallGlobal({}, {}, {})", self.a(), self.b(), self.c()),
            Self::RETURN => write!(f, "Return({})", self.a()),
            Self::ADD => write!(f, "Add({}, {}, {})", self.a(), self.b(), self.c()),
            Self::SUB => write!(f, "Sub({}, {}, {})", self.a(), self.b(), self.c()),
            Self::MUL => write!(f, "Mul({}, {}, {})", self.a(), self.b(), self.c()),
            Self::DIV => write!(f, "Div({}, {}, {})", self.a(), self.b(), self.c()),
            Self::MOD => write!(f, "Mod({}, {}, {})", self.a(), self.b(), self.c()),
            Self::NEG => write!(f, "Neg({}, {})", self.a(), self.b()),
            Self::ADD_IMM => write!(f, "AddImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::SUB_IMM => write!(f, "SubImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::LT => write!(f, "Lt({}, {}, {})", self.a(), self.b(), self.c()),
            Self::LE => write!(f, "Le({}, {}, {})", self.a(), self.b(), self.c()),
            Self::GT => write!(f, "Gt({}, {}, {})", self.a(), self.b(), self.c()),
            Self::GE => write!(f, "Ge({}, {}, {})", self.a(), self.b(), self.c()),
            Self::EQ => write!(f, "Eq({}, {}, {})", self.a(), self.b(), self.c()),
            Self::NE => write!(f, "Ne({}, {}, {})", self.a(), self.b(), self.c()),
            Self::NOT => write!(f, "Not({}, {})", self.a(), self.b()),
            Self::NEW_LIST => write!(f, "NewList({}, {})", self.a(), self.b()),
            Self::GET_LIST => write!(f, "GetList({}, {}, {})", self.a(), self.b(), self.c()),
            Self::SET_LIST => write!(f, "SetList({}, {}, {})", self.a(), self.b(), self.c()),
            Self::CAR => write!(f, "Car({}, {})", self.a(), self.b()),
            Self::CDR => write!(f, "Cdr({}, {})", self.a(), self.b()),
            Self::LT_IMM => write!(f, "LtImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::LE_IMM => write!(f, "LeImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::GT_IMM => write!(f, "GtImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::GE_IMM => write!(f, "GeImm({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::JUMP_IF_LT => write!(f, "JumpIfLt({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::JUMP_IF_LE => write!(f, "JumpIfLe({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::JUMP_IF_GT => write!(f, "JumpIfGt({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::JUMP_IF_GE => write!(f, "JumpIfGe({}, {}, {})", self.a(), self.b(), self.c() as i8),
            Self::JUMP_IF_LT_IMM => write!(f, "JumpIfLtImm({}, {}, {})", self.a(), self.b() as i8, self.c() as i8),
            Self::JUMP_IF_LE_IMM => write!(f, "JumpIfLeImm({}, {}, {})", self.a(), self.b() as i8, self.c() as i8),
            Self::JUMP_IF_GT_IMM => write!(f, "JumpIfGtImm({}, {}, {})", self.a(), self.b() as i8, self.c() as i8),
            Self::JUMP_IF_GE_IMM => write!(f, "JumpIfGeImm({}, {}, {})", self.a(), self.b() as i8, self.c() as i8),
            Self::JUMP_IF_NIL => write!(f, "JumpIfNil({}, {})", self.a(), self.sbx()),
            Self::JUMP_IF_NOT_NIL => write!(f, "JumpIfNotNil({}, {})", self.a(), self.sbx()),
            Self::CONS => write!(f, "Cons({}, {}, {})", self.a(), self.b(), self.c()),
            Self::MOVE_LAST => write!(f, "MoveLast({}, {})", self.a(), self.b()),
            Self::CAR_LAST => write!(f, "CarLast({}, {})", self.a(), self.b()),
            Self::CDR_LAST => write!(f, "CdrLast({}, {})", self.a(), self.b()),
            Self::CONS_MOVE => write!(f, "ConsMove({}, {}, {})", self.a(), self.b(), self.c()),
            _ => write!(f, "Unknown(0x{:08x})", self.0),
        }
    }
}

impl Op {
    // Opcode constants (8-bit, so up to 256 opcodes)
    pub const LOAD_CONST: u8 = 0;      // ABx: dest, const_idx
    pub const LOAD_NIL: u8 = 1;        // A: dest
    pub const LOAD_TRUE: u8 = 2;       // A: dest
    pub const LOAD_FALSE: u8 = 3;      // A: dest
    pub const MOVE: u8 = 4;            // AB: dest, src
    pub const GET_GLOBAL: u8 = 5;      // ABx: dest, name_idx
    pub const SET_GLOBAL: u8 = 6;      // ABx: src (in A), name_idx (in Bx)
    pub const CLOSURE: u8 = 7;         // ABx: dest, proto_idx
    pub const JUMP: u8 = 8;            // sBx: signed offset (A unused)
    pub const JUMP_IF_FALSE: u8 = 9;   // A: reg, sBx: signed offset
    pub const JUMP_IF_TRUE: u8 = 10;   // A: reg, sBx: signed offset
    pub const CALL: u8 = 11;           // ABC: dest, func, nargs
    pub const CALL_GLOBAL: u8 = 12;    // ABC: dest, name_idx(8-bit), nargs
    pub const TAIL_CALL: u8 = 13;      // AB: func, nargs
    pub const TAIL_CALL_GLOBAL: u8 = 14; // ABC: name_idx(8-bit), arg_start, nargs
    pub const RETURN: u8 = 15;         // A: src
    pub const ADD: u8 = 16;            // ABC: dest, left, right
    pub const SUB: u8 = 17;
    pub const MUL: u8 = 18;
    pub const DIV: u8 = 19;
    pub const MOD: u8 = 20;
    pub const NEG: u8 = 21;            // AB: dest, src
    pub const ADD_IMM: u8 = 22;        // ABC: dest, src, imm (C is i8)
    pub const SUB_IMM: u8 = 23;
    pub const LT: u8 = 24;             // ABC: dest, left, right
    pub const LE: u8 = 25;
    pub const GT: u8 = 26;
    pub const GE: u8 = 27;
    pub const EQ: u8 = 28;
    pub const NE: u8 = 29;
    pub const NOT: u8 = 30;            // AB: dest, src
    pub const NEW_LIST: u8 = 31;       // AB: dest, nargs
    pub const GET_LIST: u8 = 32;       // ABC: dest, list, index
    pub const SET_LIST: u8 = 33;       // ABC: list, index, value
    pub const CAR: u8 = 34;            // AB: dest, src - get head of cons/list
    pub const CDR: u8 = 35;            // AB: dest, src - get tail of cons/list
    pub const LT_IMM: u8 = 36;         // ABC: dest, src, imm (C is i8) - src < imm
    pub const LE_IMM: u8 = 37;         // ABC: dest, src, imm (C is i8) - src <= imm
    pub const GT_IMM: u8 = 38;         // ABC: dest, src, imm (C is i8) - src > imm
    pub const GE_IMM: u8 = 39;         // ABC: dest, src, imm (C is i8) - src >= imm
    // Combined compare-and-jump opcodes (saves a register + dispatch overhead)
    pub const JUMP_IF_LT: u8 = 40;     // ABC: left, right, offset (i8) - jump if left < right
    pub const JUMP_IF_LE: u8 = 41;     // ABC: left, right, offset (i8) - jump if left <= right
    pub const JUMP_IF_GT: u8 = 42;     // ABC: left, right, offset (i8) - jump if left > right
    pub const JUMP_IF_GE: u8 = 43;     // ABC: left, right, offset (i8) - jump if left >= right
    pub const JUMP_IF_LT_IMM: u8 = 44; // ABC: src, imm (i8), offset (i8) - jump if src < imm
    pub const JUMP_IF_LE_IMM: u8 = 45; // ABC: src, imm (i8), offset (i8) - jump if src <= imm
    pub const JUMP_IF_GT_IMM: u8 = 46; // ABC: src, imm (i8), offset (i8) - jump if src > imm
    pub const JUMP_IF_GE_IMM: u8 = 47; // ABC: src, imm (i8), offset (i8) - jump if src >= imm
    // Specialized nil check opcodes (common in list processing)
    pub const JUMP_IF_NIL: u8 = 48;    // A: src, sBx: offset - jump if src is nil
    pub const JUMP_IF_NOT_NIL: u8 = 49; // A: src, sBx: offset - jump if src is NOT nil
    // Specialized cons opcode (very common in list construction)
    pub const CONS: u8 = 50;           // ABC: dest, car, cdr - create cons cell

    // Move-semantics variants (last-use optimization)
    // These opcodes move from source instead of cloning (source becomes nil)
    pub const MOVE_LAST: u8 = 51;      // AB: dest, src - move (don't clone) from src
    pub const CAR_LAST: u8 = 52;       // AB: dest, src - move car (don't clone list)
    pub const CDR_LAST: u8 = 53;       // AB: dest, src - move cdr (don't clone list)
    // For CONS, we use high bit of car/cdr registers to indicate move
    // If B & 0x80, move from car register (B & 0x7F); if C & 0x80, move from cdr register (C & 0x7F)
    pub const CONS_MOVE: u8 = 54;      // ABC: dest, car|0x80?, cdr|0x80? - cons with optional moves

    // ========== Constructors ==========

    /// Create ABC format instruction: [opcode:8][A:8][B:8][C:8]
    #[inline(always)]
    pub const fn abc(opcode: u8, a: u8, b: u8, c: u8) -> Self {
        Op((opcode as u32) | ((a as u32) << 8) | ((b as u32) << 16) | ((c as u32) << 24))
    }

    /// Create ABx format instruction: [opcode:8][A:8][Bx:16]
    #[inline(always)]
    pub const fn abx(opcode: u8, a: u8, bx: u16) -> Self {
        Op((opcode as u32) | ((a as u32) << 8) | ((bx as u32) << 16))
    }

    /// Create ABx format with signed Bx (for jumps with register)
    #[inline(always)]
    pub const fn asbx(opcode: u8, a: u8, sbx: i16) -> Self {
        Op((opcode as u32) | ((a as u32) << 8) | ((sbx as u16 as u32) << 16))
    }

    /// Create sBx format instruction (A=0, Bx is signed): for Jump
    #[inline(always)]
    pub const fn make_sbx(opcode: u8, offset: i16) -> Self {
        Op((opcode as u32) | ((offset as u16 as u32) << 16))
    }

    // ========== Decoders ==========

    #[inline(always)]
    pub const fn opcode(self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    pub const fn a(self) -> u8 {
        (self.0 >> 8) as u8
    }

    #[inline(always)]
    pub const fn b(self) -> u8 {
        (self.0 >> 16) as u8
    }

    #[inline(always)]
    pub const fn c(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline(always)]
    pub const fn bx(self) -> u16 {
        (self.0 >> 16) as u16
    }

    #[inline(always)]
    pub const fn sbx(self) -> i16 {
        (self.0 >> 16) as i16
    }

    // ========== Named constructors for each instruction ==========

    #[inline(always)]
    pub const fn load_const(dest: Reg, idx: ConstIdx) -> Self {
        Self::abx(Self::LOAD_CONST, dest, idx)
    }

    #[inline(always)]
    pub const fn load_nil(dest: Reg) -> Self {
        Self::abc(Self::LOAD_NIL, dest, 0, 0)
    }

    #[inline(always)]
    pub const fn load_true(dest: Reg) -> Self {
        Self::abc(Self::LOAD_TRUE, dest, 0, 0)
    }

    #[inline(always)]
    pub const fn load_false(dest: Reg) -> Self {
        Self::abc(Self::LOAD_FALSE, dest, 0, 0)
    }

    #[inline(always)]
    pub const fn mov(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::MOVE, dest, src, 0)
    }

    #[inline(always)]
    pub const fn get_global(dest: Reg, name_idx: ConstIdx) -> Self {
        Self::abx(Self::GET_GLOBAL, dest, name_idx)
    }

    #[inline(always)]
    pub const fn set_global(name_idx: ConstIdx, src: Reg) -> Self {
        Self::abx(Self::SET_GLOBAL, src, name_idx)
    }

    #[inline(always)]
    pub const fn closure(dest: Reg, proto_idx: ConstIdx) -> Self {
        Self::abx(Self::CLOSURE, dest, proto_idx)
    }

    #[inline(always)]
    pub const fn jump(offset: Offset) -> Self {
        Self::make_sbx(Self::JUMP, offset)
    }

    #[inline(always)]
    pub const fn jump_if_false(reg: Reg, offset: Offset) -> Self {
        Self::asbx(Self::JUMP_IF_FALSE, reg, offset)
    }

    #[inline(always)]
    pub const fn jump_if_true(reg: Reg, offset: Offset) -> Self {
        Self::asbx(Self::JUMP_IF_TRUE, reg, offset)
    }

    #[inline(always)]
    pub const fn call(dest: Reg, func: Reg, nargs: u8) -> Self {
        Self::abc(Self::CALL, dest, func, nargs)
    }

    /// CallGlobal with 8-bit name_idx (max 256 unique global function names)
    #[inline(always)]
    pub const fn call_global(dest: Reg, name_idx: u8, nargs: u8) -> Self {
        Self::abc(Self::CALL_GLOBAL, dest, name_idx, nargs)
    }

    #[inline(always)]
    pub const fn tail_call(func: Reg, nargs: u8) -> Self {
        Self::abc(Self::TAIL_CALL, func, nargs, 0)
    }

    /// TailCallGlobal with 8-bit name_idx
    #[inline(always)]
    pub const fn tail_call_global(name_idx: u8, arg_start: Reg, nargs: u8) -> Self {
        Self::abc(Self::TAIL_CALL_GLOBAL, name_idx, arg_start, nargs)
    }

    #[inline(always)]
    pub const fn ret(src: Reg) -> Self {
        Self::abc(Self::RETURN, src, 0, 0)
    }

    #[inline(always)]
    pub const fn add(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::ADD, dest, left, right)
    }

    #[inline(always)]
    pub const fn sub(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::SUB, dest, left, right)
    }

    #[inline(always)]
    pub const fn mul(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::MUL, dest, left, right)
    }

    #[inline(always)]
    pub const fn div(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::DIV, dest, left, right)
    }

    #[inline(always)]
    pub const fn modulo(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::MOD, dest, left, right)
    }

    #[inline(always)]
    pub const fn neg(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::NEG, dest, src, 0)
    }

    #[inline(always)]
    pub const fn add_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::ADD_IMM, dest, src, imm as u8)
    }

    #[inline(always)]
    pub const fn sub_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::SUB_IMM, dest, src, imm as u8)
    }

    #[inline(always)]
    pub const fn lt(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::LT, dest, left, right)
    }

    #[inline(always)]
    pub const fn le(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::LE, dest, left, right)
    }

    #[inline(always)]
    pub const fn gt(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::GT, dest, left, right)
    }

    #[inline(always)]
    pub const fn ge(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::GE, dest, left, right)
    }

    #[inline(always)]
    pub const fn eq(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::EQ, dest, left, right)
    }

    #[inline(always)]
    pub const fn ne(dest: Reg, left: Reg, right: Reg) -> Self {
        Self::abc(Self::NE, dest, left, right)
    }

    #[inline(always)]
    pub const fn not(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::NOT, dest, src, 0)
    }

    #[inline(always)]
    pub const fn new_list(dest: Reg, nargs: u8) -> Self {
        Self::abc(Self::NEW_LIST, dest, nargs, 0)
    }

    #[inline(always)]
    pub const fn get_list(dest: Reg, list: Reg, index: Reg) -> Self {
        Self::abc(Self::GET_LIST, dest, list, index)
    }

    #[inline(always)]
    pub const fn set_list(list: Reg, index: Reg, value: Reg) -> Self {
        Self::abc(Self::SET_LIST, list, index, value)
    }

    #[inline(always)]
    pub const fn car(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::CAR, dest, src, 0)
    }

    #[inline(always)]
    pub const fn cdr(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::CDR, dest, src, 0)
    }

    #[inline(always)]
    pub const fn lt_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::LT_IMM, dest, src, imm as u8)
    }

    #[inline(always)]
    pub const fn le_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::LE_IMM, dest, src, imm as u8)
    }

    #[inline(always)]
    pub const fn gt_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::GT_IMM, dest, src, imm as u8)
    }

    #[inline(always)]
    pub const fn ge_imm(dest: Reg, src: Reg, imm: i8) -> Self {
        Self::abc(Self::GE_IMM, dest, src, imm as u8)
    }

    // Combined compare-and-jump (register vs register)
    #[inline(always)]
    pub const fn jump_if_lt(left: Reg, right: Reg, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_LT, left, right, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_le(left: Reg, right: Reg, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_LE, left, right, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_gt(left: Reg, right: Reg, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_GT, left, right, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_ge(left: Reg, right: Reg, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_GE, left, right, offset as u8)
    }

    // Combined compare-and-jump (register vs immediate)
    #[inline(always)]
    pub const fn jump_if_lt_imm(src: Reg, imm: i8, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_LT_IMM, src, imm as u8, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_le_imm(src: Reg, imm: i8, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_LE_IMM, src, imm as u8, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_gt_imm(src: Reg, imm: i8, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_GT_IMM, src, imm as u8, offset as u8)
    }

    #[inline(always)]
    pub const fn jump_if_ge_imm(src: Reg, imm: i8, offset: i8) -> Self {
        Self::abc(Self::JUMP_IF_GE_IMM, src, imm as u8, offset as u8)
    }

    // Specialized nil check (uses ABx format like JumpIfFalse for 16-bit offset)
    #[inline(always)]
    pub const fn jump_if_nil(src: Reg, offset: Offset) -> Self {
        Self::asbx(Self::JUMP_IF_NIL, src, offset)
    }

    #[inline(always)]
    pub const fn jump_if_not_nil(src: Reg, offset: Offset) -> Self {
        Self::asbx(Self::JUMP_IF_NOT_NIL, src, offset)
    }

    // Specialized cons (very common in list construction)
    #[inline(always)]
    pub const fn cons(dest: Reg, car: Reg, cdr: Reg) -> Self {
        Self::abc(Self::CONS, dest, car, cdr)
    }

    // ========== Move-semantics variants (last-use optimization) ==========

    /// Move from src to dest (source becomes nil after move)
    #[inline(always)]
    pub const fn move_last(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::MOVE_LAST, dest, src, 0)
    }

    /// Car with move semantics (source list is consumed)
    #[inline(always)]
    pub const fn car_last(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::CAR_LAST, dest, src, 0)
    }

    /// Cdr with move semantics (source list is consumed)
    #[inline(always)]
    pub const fn cdr_last(dest: Reg, src: Reg) -> Self {
        Self::abc(Self::CDR_LAST, dest, src, 0)
    }

    /// Cons with optional move semantics
    /// move_car: if true, move from car register instead of clone
    /// move_cdr: if true, move from cdr register instead of clone
    #[inline(always)]
    pub const fn cons_move(dest: Reg, car: Reg, cdr: Reg, move_car: bool, move_cdr: bool) -> Self {
        let car_with_flag = if move_car { car | 0x80 } else { car };
        let cdr_with_flag = if move_cdr { cdr | 0x80 } else { cdr };
        Self::abc(Self::CONS_MOVE, dest, car_with_flag, cdr_with_flag)
    }

    // ========== Jump patching helpers ==========

    /// Check if this is a jump instruction (for patching)
    #[inline(always)]
    pub const fn is_jump(self) -> bool {
        let op = self.opcode();
        op == Self::JUMP || op == Self::JUMP_IF_FALSE || op == Self::JUMP_IF_TRUE
            || op == Self::JUMP_IF_LT || op == Self::JUMP_IF_LE
            || op == Self::JUMP_IF_GT || op == Self::JUMP_IF_GE
            || op == Self::JUMP_IF_LT_IMM || op == Self::JUMP_IF_LE_IMM
            || op == Self::JUMP_IF_GT_IMM || op == Self::JUMP_IF_GE_IMM
            || op == Self::JUMP_IF_NIL || op == Self::JUMP_IF_NOT_NIL
    }

    /// Patch the offset of a jump instruction
    #[inline(always)]
    pub fn patch_offset(&mut self, new_offset: i16) {
        let opcode = self.opcode();
        let a = self.a();
        let b = self.b();
        // Combined compare-and-jump use ABC format with C as 8-bit offset
        if opcode >= Self::JUMP_IF_LT && opcode <= Self::JUMP_IF_GE_IMM {
            // For these opcodes: A=left/src, B=right/imm, C=offset (i8)
            *self = Self::abc(opcode, a, b, new_offset as i8 as u8);
        } else if opcode == Self::JUMP {
            *self = Self::make_sbx(opcode, new_offset);
        } else {
            *self = Self::asbx(opcode, a, new_offset);
        }
    }
}

/// A compiled function prototype
#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<Op>,
    pub constants: Vec<Value>,
    pub num_params: u8,
    pub num_registers: u8,
    pub protos: Vec<Chunk>, // nested function prototypes
    // Constant deduplication indexes for O(1) lookup of common types
    #[allow(dead_code)]
    int_const_idx: HashMap<i64, ConstIdx>,
    #[allow(dead_code)]
    symbol_const_idx: HashMap<String, ConstIdx>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            num_params: 0,
            num_registers: 0,
            protos: Vec::new(),
            int_const_idx: HashMap::new(),
            symbol_const_idx: HashMap::new(),
        }
    }

    /// Perform liveness analysis and upgrade opcodes to move variants where profitable.
    /// This is a backward dataflow analysis that tracks which registers are "live"
    /// (will be used later). When a register is used for the last time, we can
    /// use move semantics instead of clone.
    pub fn optimize_moves(&mut self) {
        if self.code.is_empty() {
            return;
        }

        // First, recursively optimize nested function prototypes
        for proto in &mut self.protos {
            proto.optimize_moves();
        }

        // Step 1: Find all jump targets (instructions that can be jumped to)
        // At jump targets, we need to be conservative because registers might
        // be live from a different path
        let mut jump_targets: Vec<bool> = vec![false; self.code.len()];
        for (i, op) in self.code.iter().enumerate() {
            let opcode = op.opcode();
            // Check if this is a jump instruction and mark the target
            if opcode == Op::JUMP || opcode == Op::JUMP_IF_FALSE || opcode == Op::JUMP_IF_TRUE
                || opcode == Op::JUMP_IF_NIL || opcode == Op::JUMP_IF_NOT_NIL
            {
                let offset = op.sbx() as isize;
                let target = (i as isize + 1 + offset) as usize;
                if target < self.code.len() {
                    jump_targets[target] = true;
                }
            } else if opcode >= Op::JUMP_IF_LT && opcode <= Op::JUMP_IF_GE_IMM {
                // These use i8 offset in C byte
                let offset = op.c() as i8 as isize;
                let target = (i as isize + 1 + offset) as usize;
                if target < self.code.len() {
                    jump_targets[target] = true;
                }
            }
        }

        // Step 2: Compute "ever_live" - registers that are ever used in the function
        // This is needed to be conservative at join points
        let mut ever_live: u128 = 0;
        for op in self.code.iter() {
            let opcode = op.opcode();
            // Add all registers that are read by any instruction
            match opcode {
                Op::MOVE | Op::CAR | Op::CDR | Op::NEG | Op::NOT => {
                    ever_live |= 1u128 << op.b();
                }
                Op::ADD | Op::SUB | Op::MUL | Op::DIV | Op::MOD |
                Op::LT | Op::LE | Op::GT | Op::GE | Op::EQ | Op::NE |
                Op::CONS | Op::GET_LIST => {
                    ever_live |= 1u128 << op.b();
                    ever_live |= 1u128 << op.c();
                }
                Op::ADD_IMM | Op::SUB_IMM |
                Op::LT_IMM | Op::LE_IMM | Op::GT_IMM | Op::GE_IMM => {
                    ever_live |= 1u128 << op.b();
                }
                Op::SET_GLOBAL | Op::RETURN => {
                    ever_live |= 1u128 << op.a();
                }
                Op::JUMP_IF_FALSE | Op::JUMP_IF_TRUE |
                Op::JUMP_IF_NIL | Op::JUMP_IF_NOT_NIL => {
                    ever_live |= 1u128 << op.a();
                }
                Op::JUMP_IF_LT | Op::JUMP_IF_LE | Op::JUMP_IF_GT | Op::JUMP_IF_GE => {
                    ever_live |= 1u128 << op.a();
                    ever_live |= 1u128 << op.b();
                }
                Op::JUMP_IF_LT_IMM | Op::JUMP_IF_LE_IMM |
                Op::JUMP_IF_GT_IMM | Op::JUMP_IF_GE_IMM => {
                    ever_live |= 1u128 << op.a();
                }
                _ => {}
            }
        }

        // Track which registers are live at each point (bitset for efficiency)
        // We use u128 to support up to 128 registers (more than enough)
        let mut live: u128 = 0;

        // Walk backward through the code
        for i in (0..self.code.len()).rev() {
            // At jump targets, be conservative: all ever-used registers might be live
            if jump_targets[i] {
                live |= ever_live;
            }

            let op = self.code[i];
            let opcode = op.opcode();

            match opcode {
                // ===== AB-format instructions that can use move semantics =====

                Op::MOVE => {
                    let dest = op.a();
                    let src = op.b();
                    // If src is not live after this instruction, use move semantics
                    let src_live_after = (live & (1u128 << src)) != 0;
                    // Update liveness: dest is now dead (overwritten), src is now live
                    live &= !(1u128 << dest);
                    live |= 1u128 << src;
                    // Upgrade to MOVE_LAST if src was not live (this is its last use)
                    if !src_live_after && src != dest {
                        self.code[i] = Op::move_last(dest, src);
                    }
                }

                Op::CAR => {
                    let dest = op.a();
                    let src = op.b();
                    let src_live_after = (live & (1u128 << src)) != 0;
                    live &= !(1u128 << dest);
                    live |= 1u128 << src;
                    if !src_live_after && src != dest {
                        self.code[i] = Op::car_last(dest, src);
                    }
                }

                Op::CDR => {
                    let dest = op.a();
                    let src = op.b();
                    let src_live_after = (live & (1u128 << src)) != 0;
                    live &= !(1u128 << dest);
                    live |= 1u128 << src;
                    if !src_live_after && src != dest {
                        self.code[i] = Op::cdr_last(dest, src);
                    }
                }

                Op::CONS => {
                    let dest = op.a();
                    let car = op.b();
                    let cdr = op.c();
                    let car_live_after = (live & (1u128 << car)) != 0;
                    let cdr_live_after = (live & (1u128 << cdr)) != 0;
                    live &= !(1u128 << dest);
                    live |= 1u128 << car;
                    live |= 1u128 << cdr;
                    // Upgrade to CONS_MOVE if either car or cdr is last use
                    let move_car = !car_live_after && car != dest;
                    let move_cdr = !cdr_live_after && cdr != dest;
                    if move_car || move_cdr {
                        self.code[i] = Op::cons_move(dest, car, cdr, move_car, move_cdr);
                    }
                }

                // ===== Other AB-format instructions (read src, write dest) =====

                Op::NEG | Op::NOT => {
                    let dest = op.a();
                    let src = op.b();
                    live &= !(1u128 << dest);
                    live |= 1u128 << src;
                }

                // ===== ABC-format arithmetic/comparison (read B, C; write A) =====

                Op::ADD | Op::SUB | Op::MUL | Op::DIV | Op::MOD |
                Op::LT | Op::LE | Op::GT | Op::GE | Op::EQ | Op::NE => {
                    let dest = op.a();
                    let b = op.b();
                    let c = op.c();
                    live &= !(1u128 << dest);
                    live |= 1u128 << b;
                    live |= 1u128 << c;
                }

                // ===== Immediate variants (read B only) =====

                Op::ADD_IMM | Op::SUB_IMM |
                Op::LT_IMM | Op::LE_IMM | Op::GT_IMM | Op::GE_IMM => {
                    let dest = op.a();
                    let src = op.b();
                    live &= !(1u128 << dest);
                    live |= 1u128 << src;
                }

                // ===== Load instructions (write only) =====

                Op::LOAD_CONST | Op::LOAD_NIL | Op::LOAD_TRUE | Op::LOAD_FALSE |
                Op::GET_GLOBAL | Op::CLOSURE => {
                    let dest = op.a();
                    live &= !(1u128 << dest);
                }

                // ===== Store instructions (read only) =====

                Op::SET_GLOBAL | Op::RETURN => {
                    let src = op.a();
                    live |= 1u128 << src;
                }

                // ===== Jump instructions (conditionally read) =====

                Op::JUMP => {
                    // Unconditional jump reads nothing
                    // Note: For proper analysis we'd need to handle control flow
                    // This simplified version treats code as linear (conservative for loops)
                }

                Op::JUMP_IF_FALSE | Op::JUMP_IF_TRUE |
                Op::JUMP_IF_NIL | Op::JUMP_IF_NOT_NIL => {
                    let reg = op.a();
                    live |= 1u128 << reg;
                }

                Op::JUMP_IF_LT | Op::JUMP_IF_LE | Op::JUMP_IF_GT | Op::JUMP_IF_GE => {
                    let left = op.a();
                    let right = op.b();
                    live |= 1u128 << left;
                    live |= 1u128 << right;
                }

                Op::JUMP_IF_LT_IMM | Op::JUMP_IF_LE_IMM |
                Op::JUMP_IF_GT_IMM | Op::JUMP_IF_GE_IMM => {
                    let src = op.a();
                    live |= 1u128 << src;
                }

                // ===== Call instructions (complex, treat all args as live) =====

                Op::CALL => {
                    let dest = op.a();
                    let func = op.b();
                    let nargs = op.c();
                    live &= !(1u128 << dest);
                    live |= 1u128 << func;
                    // Args are in consecutive registers after func
                    for j in 0..nargs {
                        live |= 1u128 << (func + 1 + j);
                    }
                }

                Op::CALL_GLOBAL => {
                    let dest = op.a();
                    let nargs = op.c();
                    live &= !(1u128 << dest);
                    // Args are in consecutive registers after dest
                    for j in 0..nargs {
                        live |= 1u128 << (dest + 1 + j);
                    }
                }

                Op::TAIL_CALL => {
                    let func = op.a();
                    let nargs = op.b();
                    live |= 1u128 << func;
                    for j in 0..nargs {
                        live |= 1u128 << (func + 1 + j);
                    }
                }

                Op::TAIL_CALL_GLOBAL => {
                    let first_arg = op.b();
                    let nargs = op.c();
                    for j in 0..nargs {
                        live |= 1u128 << (first_arg + j);
                    }
                }

                // ===== List operations =====

                Op::NEW_LIST => {
                    let dest = op.a();
                    let nargs = op.b();
                    live &= !(1u128 << dest);
                    for j in 0..nargs {
                        live |= 1u128 << (dest + 1 + j);
                    }
                }

                Op::GET_LIST => {
                    let dest = op.a();
                    let list = op.b();
                    let index = op.c();
                    live &= !(1u128 << dest);
                    live |= 1u128 << list;
                    live |= 1u128 << index;
                }

                Op::SET_LIST => {
                    let list = op.a();
                    let index = op.b();
                    let value = op.c();
                    live |= 1u128 << list;
                    live |= 1u128 << index;
                    live |= 1u128 << value;
                }

                // Already upgraded move variants - shouldn't appear in initial code
                Op::MOVE_LAST | Op::CAR_LAST | Op::CDR_LAST | Op::CONS_MOVE => {}

                _ => {}
            }
        }
    }

    pub fn add_constant(&mut self, value: Value) -> ConstIdx {
        // Fast path: check specialized indexes for common types (O(1))
        if let Some(n) = value.as_int() {
            if let Some(&idx) = self.int_const_idx.get(&n) {
                return idx;
            }
        }
        if let Some(s) = value.as_symbol() {
            if let Some(&idx) = self.symbol_const_idx.get(s) {
                return idx;
            }
        }

        // Slow path: linear scan for other types (strings, floats, etc.)
        for (i, c) in self.constants.iter().enumerate() {
            if *c == value {
                return i as ConstIdx;
            }
        }

        // Not found - add new constant and update indexes
        let idx = self.constants.len() as ConstIdx;

        // Update specialized indexes
        if let Some(n) = value.as_int() {
            self.int_const_idx.insert(n, idx);
        }
        if let Some(s) = value.as_symbol() {
            self.symbol_const_idx.insert(s.to_string(), idx);
        }

        self.constants.push(value);
        idx
    }

    pub fn emit(&mut self, op: Op) -> usize {
        let pos = self.code.len();
        self.code.push(op);
        pos
    }

    pub fn current_pos(&self) -> usize {
        self.code.len()
    }

    pub fn patch_jump(&mut self, pos: usize, target: usize) {
        let offset = (target as isize - pos as isize - 1) as Offset;
        self.code[pos].patch_offset(offset);
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_size() {
        // Verify we achieved 4-byte instructions
        assert_eq!(std::mem::size_of::<Op>(), 4);
    }

    #[test]
    fn test_abc_encoding() {
        let op = Op::abc(Op::ADD, 1, 2, 3);
        assert_eq!(op.opcode(), Op::ADD);
        assert_eq!(op.a(), 1);
        assert_eq!(op.b(), 2);
        assert_eq!(op.c(), 3);
    }

    #[test]
    fn test_abx_encoding() {
        let op = Op::abx(Op::LOAD_CONST, 5, 1000);
        assert_eq!(op.opcode(), Op::LOAD_CONST);
        assert_eq!(op.a(), 5);
        assert_eq!(op.bx(), 1000);
    }

    #[test]
    fn test_sbx_encoding() {
        // Test positive offset
        let op = Op::make_sbx(Op::JUMP, 100);
        assert_eq!(op.opcode(), Op::JUMP);
        assert_eq!(op.sbx(), 100);

        // Test negative offset
        let op = Op::make_sbx(Op::JUMP, -50);
        assert_eq!(op.opcode(), Op::JUMP);
        assert_eq!(op.sbx(), -50);
    }

    #[test]
    fn test_chunk_constants() {
        let mut chunk = Chunk::new();
        let idx1 = chunk.add_constant(Value::Int(42));
        let idx2 = chunk.add_constant(Value::Int(100));
        let idx3 = chunk.add_constant(Value::Int(42)); // duplicate
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 0); // reuses existing
    }

    #[test]
    fn test_chunk_emit() {
        let mut chunk = Chunk::new();
        chunk.emit(Op::load_nil(0));
        chunk.emit(Op::ret(0));
        assert_eq!(chunk.code.len(), 2);
    }

    #[test]
    fn test_jump_patching() {
        let mut chunk = Chunk::new();
        chunk.emit(Op::load_true(0));
        let jump_pos = chunk.emit(Op::jump_if_false(0, 0)); // placeholder
        chunk.emit(Op::load_const(1, 0));
        let target = chunk.current_pos();
        chunk.emit(Op::ret(1));

        chunk.patch_jump(jump_pos, target);

        // Offset should be: target - jump_pos - 1 = 3 - 1 - 1 = 1
        assert_eq!(chunk.code[jump_pos].sbx(), 1);
    }
}
