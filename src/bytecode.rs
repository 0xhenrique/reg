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

    // ========== Jump patching helpers ==========

    /// Check if this is a jump instruction (for patching)
    #[inline(always)]
    pub const fn is_jump(self) -> bool {
        let op = self.opcode();
        op == Self::JUMP || op == Self::JUMP_IF_FALSE || op == Self::JUMP_IF_TRUE
    }

    /// Patch the offset of a jump instruction
    #[inline(always)]
    pub fn patch_offset(&mut self, new_offset: i16) {
        let opcode = self.opcode();
        let a = self.a();
        // Reconstruct with new offset
        if opcode == Self::JUMP {
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
