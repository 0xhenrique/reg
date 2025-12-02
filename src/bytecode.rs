use crate::value::Value;

pub type Reg = u8;
pub type ConstIdx = u16;
pub type Offset = i16;

#[derive(Debug, Clone)]
pub enum Op {
    // Load constants/literals
    LoadConst(Reg, ConstIdx),
    LoadNil(Reg),
    LoadTrue(Reg),
    LoadFalse(Reg),

    // Register operations
    Move(Reg, Reg), // dest, src

    // Global variables (for def)
    GetGlobal(Reg, ConstIdx), // dest, name_idx (name is in constants)
    SetGlobal(ConstIdx, Reg), // name_idx, src

    // Closures
    Closure(Reg, ConstIdx), // dest, proto_idx

    // Control flow
    Jump(Offset),
    JumpIfFalse(Reg, Offset),
    JumpIfTrue(Reg, Offset),

    // Function calls
    Call(Reg, Reg, u8),     // dest, func, nargs (args in func+1, func+2, ...)
    TailCall(Reg, u8),      // func, nargs
    Return(Reg),

    // Arithmetic (dest, left, right)
    Add(Reg, Reg, Reg),
    Sub(Reg, Reg, Reg),
    Mul(Reg, Reg, Reg),
    Div(Reg, Reg, Reg),
    Mod(Reg, Reg, Reg),
    Neg(Reg, Reg), // dest, src (unary minus)

    // Comparison
    Lt(Reg, Reg, Reg),
    Le(Reg, Reg, Reg),
    Gt(Reg, Reg, Reg),
    Ge(Reg, Reg, Reg),
    Eq(Reg, Reg, Reg),
    Ne(Reg, Reg, Reg),

    // Logic
    Not(Reg, Reg), // dest, src

    // List operations
    NewList(Reg, u8),        // dest, nargs (elements in dest+1, dest+2, ...)
    GetList(Reg, Reg, Reg),  // dest, list, index
    SetList(Reg, Reg, Reg),  // list, index, value
}

/// A compiled function prototype
#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<Op>,
    pub constants: Vec<Value>,
    pub num_params: u8,
    pub num_registers: u8,
    pub protos: Vec<Chunk>, // nested function prototypes
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            num_params: 0,
            num_registers: 0,
            protos: Vec::new(),
        }
    }

    pub fn add_constant(&mut self, value: Value) -> ConstIdx {
        // Check if constant already exists
        for (i, c) in self.constants.iter().enumerate() {
            if *c == value {
                return i as ConstIdx;
            }
        }
        let idx = self.constants.len();
        self.constants.push(value);
        idx as ConstIdx
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
        match &mut self.code[pos] {
            Op::Jump(ref mut o) => *o = offset,
            Op::JumpIfFalse(_, ref mut o) => *o = offset,
            Op::JumpIfTrue(_, ref mut o) => *o = offset,
            _ => panic!("Not a jump instruction at {}", pos),
        }
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
        chunk.emit(Op::LoadNil(0));
        chunk.emit(Op::Return(0));
        assert_eq!(chunk.code.len(), 2);
    }
}
