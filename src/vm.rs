use crate::bytecode::{Chunk, Op};
use crate::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const MAX_REGISTERS: usize = 256;
const MAX_FRAMES: usize = 256;

#[derive(Clone)]
struct CallFrame {
    chunk: Rc<Chunk>,
    ip: usize,
    base: usize,
}

pub struct VM {
    registers: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Rc<RefCell<HashMap<String, Value>>>,
}

impl VM {
    pub fn new() -> Self {
        VM {
            registers: vec![Value::Nil; MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        VM {
            registers: vec![Value::Nil; MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals,
        }
    }

    pub fn define_global(&mut self, name: &str, value: Value) {
        self.globals.borrow_mut().insert(name.to_string(), value);
    }

    pub fn run(&mut self, chunk: Chunk) -> Result<Value, String> {
        self.frames.push(CallFrame {
            chunk: Rc::new(chunk),
            ip: 0,
            base: 0,
        });

        self.execute()
    }

    fn execute(&mut self) -> Result<Value, String> {
        loop {
            let frame = self.frames.last().ok_or("No active frame")?;
            let op = frame.chunk.code.get(frame.ip).cloned();
            let base = frame.base;
            let chunk = frame.chunk.clone();

            // Advance IP
            if let Some(frame) = self.frames.last_mut() {
                frame.ip += 1;
            }

            let op = match op {
                Some(op) => op,
                None => return Err("Unexpected end of bytecode".to_string()),
            };

            match op {
                Op::LoadConst(dest, idx) => {
                    let value = chunk.constants[idx as usize].clone();
                    self.registers[base + dest as usize] = value;
                }

                Op::LoadNil(dest) => {
                    self.registers[base + dest as usize] = Value::Nil;
                }

                Op::LoadTrue(dest) => {
                    self.registers[base + dest as usize] = Value::Bool(true);
                }

                Op::LoadFalse(dest) => {
                    self.registers[base + dest as usize] = Value::Bool(false);
                }

                Op::Move(dest, src) => {
                    let value = self.registers[base + src as usize].clone();
                    self.registers[base + dest as usize] = value;
                }

                Op::GetGlobal(dest, name_idx) => {
                    let name = match &chunk.constants[name_idx as usize] {
                        Value::Symbol(s) => s.to_string(),
                        _ => return Err("GetGlobal: expected symbol".to_string()),
                    };
                    let value = self.globals.borrow().get(&name).cloned()
                        .ok_or_else(|| format!("Undefined variable: {}", name))?;
                    self.registers[base + dest as usize] = value;
                }

                Op::SetGlobal(name_idx, src) => {
                    let name = match &chunk.constants[name_idx as usize] {
                        Value::Symbol(s) => s.to_string(),
                        _ => return Err("SetGlobal: expected symbol".to_string()),
                    };
                    let value = self.registers[base + src as usize].clone();
                    self.globals.borrow_mut().insert(name, value);
                }

                Op::Closure(dest, proto_idx) => {
                    let proto = chunk.protos[proto_idx as usize].clone();
                    let func = Value::CompiledFunction(Rc::new(proto));
                    self.registers[base + dest as usize] = func;
                }

                Op::Jump(offset) => {
                    if let Some(frame) = self.frames.last_mut() {
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Op::JumpIfFalse(reg, offset) => {
                    if !self.registers[base + reg as usize].is_truthy() {
                        if let Some(frame) = self.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                    }
                }

                Op::JumpIfTrue(reg, offset) => {
                    if self.registers[base + reg as usize].is_truthy() {
                        if let Some(frame) = self.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                    }
                }

                Op::Call(dest, func_reg, nargs) => {
                    let func = self.registers[base + func_reg as usize].clone();
                    let args: Vec<Value> = (0..nargs)
                        .map(|i| self.registers[base + func_reg as usize + 1 + i as usize].clone())
                        .collect();

                    let result = self.call_function(&func, &args)?;
                    self.registers[base + dest as usize] = result;
                }

                Op::TailCall(func_reg, nargs) => {
                    let func = self.registers[base + func_reg as usize].clone();
                    let args: Vec<Value> = (0..nargs)
                        .map(|i| self.registers[base + func_reg as usize + 1 + i as usize].clone())
                        .collect();

                    match &func {
                        Value::CompiledFunction(cf) => {
                            if cf.num_params as usize != args.len() {
                                return Err(format!(
                                    "Expected {} arguments, got {}",
                                    cf.num_params, args.len()
                                ));
                            }
                            // Reuse current frame for tail call
                            if let Some(frame) = self.frames.last_mut() {
                                frame.chunk = cf.clone();
                                frame.ip = 0;
                            }
                            // Copy args to base registers
                            for (i, arg) in args.iter().enumerate() {
                                self.registers[base + i] = arg.clone();
                            }
                        }
                        Value::NativeFunction(nf) => {
                            let result = (nf.func)(&args)?;
                            self.frames.pop();
                            if self.frames.is_empty() {
                                return Ok(result);
                            }
                            let caller_base = self.frames.last().unwrap().base;
                            self.registers[caller_base] = result;
                        }
                        _ => return Err(format!("Not a function: {}", func)),
                    }
                }

                Op::Return(reg) => {
                    let result = self.registers[base + reg as usize].clone();
                    self.frames.pop();

                    if self.frames.is_empty() {
                        return Ok(result);
                    }
                    let caller_base = self.frames.last().unwrap().base;
                    self.registers[caller_base] = result;
                }

                Op::Add(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = binary_arith(va, vb, |x, y| x + y, |x, y| x + y, "+")?;
                    self.registers[base + dest as usize] = result;
                }

                Op::Sub(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = binary_arith(va, vb, |x, y| x - y, |x, y| x - y, "-")?;
                    self.registers[base + dest as usize] = result;
                }

                Op::Mul(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = binary_arith(va, vb, |x, y| x * y, |x, y| x * y, "*")?;
                    self.registers[base + dest as usize] = result;
                }

                Op::Div(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = match (va, vb) {
                        (Value::Int(x), Value::Int(y)) => {
                            if *y == 0 { return Err("Division by zero".to_string()); }
                            Value::Float(*x as f64 / *y as f64)
                        }
                        (Value::Float(x), Value::Float(y)) => {
                            if *y == 0.0 { return Err("Division by zero".to_string()); }
                            Value::Float(x / y)
                        }
                        (Value::Int(x), Value::Float(y)) => {
                            if *y == 0.0 { return Err("Division by zero".to_string()); }
                            Value::Float(*x as f64 / y)
                        }
                        (Value::Float(x), Value::Int(y)) => {
                            if *y == 0 { return Err("Division by zero".to_string()); }
                            Value::Float(x / *y as f64)
                        }
                        _ => return Err("/ expects numbers".to_string()),
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Mod(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = match (va, vb) {
                        (Value::Int(x), Value::Int(y)) => {
                            if *y == 0 { return Err("Division by zero".to_string()); }
                            Value::Int(x % y)
                        }
                        _ => return Err("mod expects integers".to_string()),
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Neg(dest, src) => {
                    let v = &self.registers[base + src as usize];
                    let result = match v {
                        Value::Int(x) => Value::Int(-x),
                        Value::Float(x) => Value::Float(-x),
                        _ => return Err("- expects a number".to_string()),
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Lt(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::Bool(compare_values(va, vb)? == std::cmp::Ordering::Less);
                    self.registers[base + dest as usize] = result;
                }

                Op::Le(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::Bool(compare_values(va, vb)? != std::cmp::Ordering::Greater);
                    self.registers[base + dest as usize] = result;
                }

                Op::Gt(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::Bool(compare_values(va, vb)? == std::cmp::Ordering::Greater);
                    self.registers[base + dest as usize] = result;
                }

                Op::Ge(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::Bool(compare_values(va, vb)? != std::cmp::Ordering::Less);
                    self.registers[base + dest as usize] = result;
                }

                Op::Eq(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    self.registers[base + dest as usize] = Value::Bool(va == vb);
                }

                Op::Ne(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    self.registers[base + dest as usize] = Value::Bool(va != vb);
                }

                Op::Not(dest, src) => {
                    let v = &self.registers[base + src as usize];
                    self.registers[base + dest as usize] = Value::Bool(!v.is_truthy());
                }

                Op::NewList(dest, nargs) => {
                    let items: Vec<Value> = (0..nargs)
                        .map(|i| self.registers[base + dest as usize + 1 + i as usize].clone())
                        .collect();
                    self.registers[base + dest as usize] = Value::list(items);
                }

                Op::GetList(dest, list, index) => {
                    let list_val = &self.registers[base + list as usize];
                    let idx = &self.registers[base + index as usize];
                    let result = match (list_val, idx) {
                        (Value::List(items), Value::Int(i)) => {
                            items.get(*i as usize).cloned().unwrap_or(Value::Nil)
                        }
                        _ => return Err("GetList expects list and int".to_string()),
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::SetList(_, _, _) => {
                    return Err("SetList not implemented (immutable lists)".to_string());
                }
            }
        }
    }

    fn call_function(&mut self, func: &Value, args: &[Value]) -> Result<Value, String> {
        match func {
            Value::CompiledFunction(cf) => {
                if cf.num_params as usize != args.len() {
                    return Err(format!(
                        "Expected {} arguments, got {}",
                        cf.num_params, args.len()
                    ));
                }

                let new_base = self.frames.last()
                    .map(|f| f.base + f.chunk.num_registers as usize)
                    .unwrap_or(0);

                if new_base + cf.num_registers as usize > MAX_REGISTERS {
                    return Err("Stack overflow".to_string());
                }

                for (i, arg) in args.iter().enumerate() {
                    self.registers[new_base + i] = arg.clone();
                }

                self.frames.push(CallFrame {
                    chunk: cf.clone(),
                    ip: 0,
                    base: new_base,
                });

                self.execute()
            }

            Value::NativeFunction(nf) => (nf.func)(args),

            Value::Function(_) => {
                Err("Cannot call interpreted function from VM".to_string())
            }

            _ => Err(format!("Not a function: {}", func)),
        }
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

fn binary_arith<FI, FF>(
    a: &Value,
    b: &Value,
    int_op: FI,
    float_op: FF,
    name: &str,
) -> Result<Value, String>
where
    FI: Fn(i64, i64) -> i64,
    FF: Fn(f64, f64) -> f64,
{
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(int_op(*x, *y))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(float_op(*x, *y))),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(float_op(*x as f64, *y))),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(float_op(*x, *y as f64))),
        _ => Err(format!("{} expects numbers", name)),
    }
}

fn compare_values(a: &Value, b: &Value) -> Result<std::cmp::Ordering, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => x
            .partial_cmp(y)
            .ok_or_else(|| "Cannot compare NaN".to_string()),
        (Value::Int(x), Value::Float(y)) => (*x as f64)
            .partial_cmp(y)
            .ok_or_else(|| "Cannot compare NaN".to_string()),
        (Value::Float(x), Value::Int(y)) => x
            .partial_cmp(&(*y as f64))
            .ok_or_else(|| "Cannot compare NaN".to_string()),
        _ => Err(format!(
            "Cannot compare {} and {}",
            a.type_name(),
            b.type_name()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Op;

    #[test]
    fn test_vm_simple() {
        let mut chunk = Chunk::new();
        let idx = chunk.add_constant(Value::Int(42));
        chunk.emit(Op::LoadConst(0, idx));
        chunk.emit(Op::Return(0));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_vm_arithmetic() {
        let mut chunk = Chunk::new();
        let idx1 = chunk.add_constant(Value::Int(10));
        let idx2 = chunk.add_constant(Value::Int(3));
        chunk.emit(Op::LoadConst(0, idx1));
        chunk.emit(Op::LoadConst(1, idx2));
        chunk.emit(Op::Add(2, 0, 1));
        chunk.emit(Op::Return(2));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::Int(13));
    }

    #[test]
    fn test_vm_comparison() {
        let mut chunk = Chunk::new();
        let idx1 = chunk.add_constant(Value::Int(5));
        let idx2 = chunk.add_constant(Value::Int(10));
        chunk.emit(Op::LoadConst(0, idx1));
        chunk.emit(Op::LoadConst(1, idx2));
        chunk.emit(Op::Lt(2, 0, 1));
        chunk.emit(Op::Return(2));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_vm_jump() {
        let mut chunk = Chunk::new();
        chunk.emit(Op::LoadTrue(0));
        let jump_pos = chunk.emit(Op::JumpIfFalse(0, 0));
        let idx = chunk.add_constant(Value::Int(1));
        chunk.emit(Op::LoadConst(1, idx));
        let jump_over = chunk.emit(Op::Jump(0));
        let else_pos = chunk.current_pos();
        let idx2 = chunk.add_constant(Value::Int(2));
        chunk.emit(Op::LoadConst(1, idx2));
        let end = chunk.current_pos();
        chunk.patch_jump(jump_pos, else_pos);
        chunk.patch_jump(jump_over, end);
        chunk.emit(Op::Return(1));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_vm_globals() {
        let mut chunk = Chunk::new();
        let name_idx = chunk.add_constant(Value::symbol("x"));
        let val_idx = chunk.add_constant(Value::Int(42));
        chunk.emit(Op::LoadConst(0, val_idx));
        chunk.emit(Op::SetGlobal(name_idx, 0));
        chunk.emit(Op::GetGlobal(1, name_idx));
        chunk.emit(Op::Return(1));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::Int(42));
    }
}
