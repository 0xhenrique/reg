use crate::bytecode::{Chunk, Op};
use crate::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const MAX_REGISTERS: usize = 8192;  // Increased for deep recursion (fib needs ~30 depth)
const MAX_FRAMES: usize = 1024;

#[derive(Clone)]
struct CallFrame {
    chunk: Rc<Chunk>,
    ip: usize,
    base: usize,
    return_reg: usize, // where to store return value in caller's frame
}

pub struct VM {
    registers: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Rc<RefCell<HashMap<String, Value>>>,
    // Cache for global lookups: name -> Value
    // Keyed by name to allow targeted invalidation when a specific global changes
    global_cache: HashMap<String, Value>,
}

impl VM {
    pub fn new() -> Self {
        VM {
            registers: vec![Value::nil(); MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals: Rc::new(RefCell::new(HashMap::new())),
            global_cache: HashMap::new(),
        }
    }

    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        VM {
            registers: vec![Value::nil(); MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals,
            global_cache: HashMap::new(),
        }
    }

    pub fn define_global(&mut self, name: &str, value: Value) {
        // Avoid String allocation if key already exists
        {
            let mut globals = self.globals.borrow_mut();
            if let Some(existing) = globals.get_mut(name) {
                *existing = value.clone();
            } else {
                globals.insert(name.to_string(), value.clone());
            }
        }
        // Update cache (same optimization)
        if let Some(existing) = self.global_cache.get_mut(name) {
            *existing = value;
        } else {
            self.global_cache.insert(name.to_string(), value);
        }
    }

    pub fn run(&mut self, chunk: Chunk) -> Result<Value, String> {
        self.frames.push(CallFrame {
            chunk: Rc::new(chunk),
            ip: 0,
            base: 0,
            return_reg: 0, // Not used for top-level frame
        });

        self.execute()
    }

    fn execute(&mut self) -> Result<Value, String> {
        loop {
            // Get op and base (Op is Copy, so no allocation needed)
            let (op, base) = {
                let frame = self.frames.last().ok_or("No active frame")?;
                (frame.chunk.code.get(frame.ip).copied(), frame.base)
            };

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
                    // Access chunk only when needed
                    let value = self.frames.last().unwrap().chunk.constants[idx as usize].clone();
                    self.registers[base + dest as usize] = value;
                }

                Op::LoadNil(dest) => {
                    self.registers[base + dest as usize] = Value::nil();
                }

                Op::LoadTrue(dest) => {
                    self.registers[base + dest as usize] = Value::bool(true);
                }

                Op::LoadFalse(dest) => {
                    self.registers[base + dest as usize] = Value::bool(false);
                }

                Op::Move(dest, src) => {
                    let value = self.registers[base + src as usize].clone();
                    self.registers[base + dest as usize] = value;
                }

                Op::GetGlobal(dest, name_idx) => {
                    // Get name from constant pool
                    let chunk = &self.frames.last().unwrap().chunk;
                    let name = chunk.constants[name_idx as usize].as_symbol()
                        .ok_or("GetGlobal: expected symbol")?;

                    let value = if let Some(cached) = self.global_cache.get(name) {
                        // Cache hit - avoid globals HashMap lookup
                        cached.clone()
                    } else {
                        // Cache miss - look up in globals
                        let v = self.globals.borrow().get(name).cloned()
                            .ok_or_else(|| format!("Undefined variable: {}", name))?;
                        // Cache for future lookups
                        self.global_cache.insert(name.to_string(), v.clone());
                        v
                    };
                    self.registers[base + dest as usize] = value;
                }

                Op::SetGlobal(name_idx, src) => {
                    let chunk = &self.frames.last().unwrap().chunk;
                    let name = chunk.constants[name_idx as usize].as_symbol()
                        .ok_or("SetGlobal: expected symbol")?;
                    let value = self.registers[base + src as usize].clone();
                    // Avoid String allocation if key already exists
                    {
                        let mut globals = self.globals.borrow_mut();
                        if let Some(existing) = globals.get_mut(name) {
                            *existing = value.clone();
                        } else {
                            globals.insert(name.to_string(), value.clone());
                        }
                    }
                    // Update cache (same optimization)
                    if let Some(existing) = self.global_cache.get_mut(name) {
                        *existing = value;
                    } else {
                        self.global_cache.insert(name.to_string(), value);
                    }
                }

                Op::Closure(dest, proto_idx) => {
                    let chunk = &self.frames.last().unwrap().chunk;
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
                    let func_val = &self.registers[base + func_reg as usize];
                    if let Some(cf) = func_val.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let num_registers = self.frames.last().unwrap().chunk.num_registers;
                        let new_base = base + num_registers as usize;

                        if new_base + cf.num_registers as usize > MAX_REGISTERS {
                            return Err("Stack overflow".to_string());
                        }

                        // Clone the Rc<Chunk> directly (not the entire Value)
                        let cf_chunk = cf.clone();

                        // Copy args directly to new frame's registers (no Vec allocation!)
                        let arg_start = base + func_reg as usize + 1;
                        for i in 0..nargs as usize {
                            self.registers[new_base + i] = self.registers[arg_start + i].clone();
                        }

                        // Push new frame - will continue in the loop
                        self.frames.push(CallFrame {
                            chunk: cf_chunk,
                            ip: 0,
                            base: new_base,
                            return_reg: base + dest as usize,
                        });
                    } else if let Some(nf) = func_val.as_native_function() {
                        // Clone the function pointer before the mutable borrow
                        let func_ptr = nf.func;
                        // Pass a slice directly to native function (no Vec allocation!)
                        let arg_start = base + func_reg as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        self.registers[base + dest as usize] = result;
                    } else if func_val.as_function().is_some() {
                        return Err("Cannot call interpreted function from VM".to_string());
                    } else {
                        return Err(format!("Not a function: {}", func_val));
                    }
                }

                Op::TailCall(func_reg, nargs) => {
                    let func_val = &self.registers[base + func_reg as usize];
                    if let Some(cf) = func_val.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }
                        // Clone the Rc<Chunk> directly (not the entire Value)
                        let cf_chunk = cf.clone();
                        // Reuse current frame for tail call
                        if let Some(frame) = self.frames.last_mut() {
                            frame.chunk = cf_chunk;
                            frame.ip = 0;
                        }
                        // Copy args directly to base registers (no Vec allocation!)
                        // Forward iteration is safe: source (base + func_reg + 1 + i)
                        // is always >= destination (base + i) since func_reg >= 0
                        let arg_start = base + func_reg as usize + 1;
                        for i in 0..nargs as usize {
                            self.registers[base + i] = self.registers[arg_start + i].clone();
                        }
                    } else if let Some(nf) = func_val.as_native_function() {
                        // Clone the function pointer before the mutable borrow
                        let func_ptr = nf.func;
                        let return_reg = self.frames.last().unwrap().return_reg;
                        // Pass a slice directly to native function (no Vec allocation!)
                        let arg_start = base + func_reg as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        self.frames.pop();
                        if self.frames.is_empty() {
                            return Ok(result);
                        }
                        // Store result in caller's designated register
                        self.registers[return_reg] = result;
                    } else if func_val.as_function().is_some() {
                        return Err("Cannot call interpreted function from VM".to_string());
                    } else {
                        return Err(format!("Not a function: {}", func_val));
                    }
                }

                Op::CallGlobal(dest, name_idx, nargs) => {
                    // Optimized: look up global and call directly without intermediate register
                    let chunk = &self.frames.last().unwrap().chunk;
                    let name = chunk.constants[name_idx as usize].as_symbol()
                        .ok_or("CallGlobal: expected symbol")?;

                    // Get function from cache or globals
                    let func_value = if let Some(cached) = self.global_cache.get(name) {
                        cached
                    } else {
                        let v = self.globals.borrow().get(name).cloned()
                            .ok_or_else(|| format!("Undefined function: {}", name))?;
                        self.global_cache.insert(name.to_string(), v);
                        self.global_cache.get(name).unwrap()
                    };

                    if let Some(cf) = func_value.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let num_registers = self.frames.last().unwrap().chunk.num_registers;
                        let new_base = base + num_registers as usize;

                        if new_base + cf.num_registers as usize > MAX_REGISTERS {
                            return Err("Stack overflow".to_string());
                        }

                        let cf_chunk = cf.clone();

                        // Copy args from dest+1, dest+2, ... to new frame
                        let arg_start = base + dest as usize + 1;
                        for i in 0..nargs as usize {
                            self.registers[new_base + i] = self.registers[arg_start + i].clone();
                        }

                        self.frames.push(CallFrame {
                            chunk: cf_chunk,
                            ip: 0,
                            base: new_base,
                            return_reg: base + dest as usize,
                        });
                    } else if let Some(nf) = func_value.as_native_function() {
                        let func_ptr = nf.func;
                        let arg_start = base + dest as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        self.registers[base + dest as usize] = result;
                    } else {
                        return Err(format!("Not a function: {}", func_value));
                    }
                }

                Op::TailCallGlobal(name_idx, arg_start, nargs) => {
                    // Optimized: look up global and tail-call directly
                    let chunk = &self.frames.last().unwrap().chunk;
                    let name = chunk.constants[name_idx as usize].as_symbol()
                        .ok_or("TailCallGlobal: expected symbol")?;

                    // Get function from cache or globals
                    let func_value = if let Some(cached) = self.global_cache.get(name) {
                        cached
                    } else {
                        let v = self.globals.borrow().get(name).cloned()
                            .ok_or_else(|| format!("Undefined function: {}", name))?;
                        self.global_cache.insert(name.to_string(), v);
                        self.global_cache.get(name).unwrap()
                    };

                    if let Some(cf) = func_value.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }
                        let cf_chunk = cf.clone();
                        if let Some(frame) = self.frames.last_mut() {
                            frame.chunk = cf_chunk;
                            frame.ip = 0;
                        }
                        // Copy args from temp registers (at arg_start+1, arg_start+2, ...) to base+0, base+1, ...
                        let src_start = base + arg_start as usize + 1;
                        for i in 0..nargs as usize {
                            self.registers[base + i] = self.registers[src_start + i].clone();
                        }
                    } else if let Some(nf) = func_value.as_native_function() {
                        let func_ptr = nf.func;
                        let return_reg = self.frames.last().unwrap().return_reg;
                        let src_start = base + arg_start as usize + 1;
                        let src_end = src_start + nargs as usize;
                        let result = func_ptr(&self.registers[src_start..src_end])?;
                        self.frames.pop();
                        if self.frames.is_empty() {
                            return Ok(result);
                        }
                        self.registers[return_reg] = result;
                    } else {
                        return Err(format!("Not a function: {}", func_value));
                    }
                }

                Op::Return(reg) => {
                    let result = self.registers[base + reg as usize].clone();
                    let return_reg = self.frames.last().unwrap().return_reg;
                    self.frames.pop();

                    if self.frames.is_empty() {
                        return Ok(result);
                    }
                    // Store result in caller's designated register
                    self.registers[return_reg] = result;
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
                    let result = if let (Some(x), Some(y)) = (va.as_int(), vb.as_int()) {
                        if y == 0 { return Err("Division by zero".to_string()); }
                        Value::float(x as f64 / y as f64)
                    } else if let (Some(x), Some(y)) = (va.as_float(), vb.as_float()) {
                        if y == 0.0 { return Err("Division by zero".to_string()); }
                        Value::float(x / y)
                    } else if let (Some(x), Some(y)) = (va.as_int(), vb.as_float()) {
                        if y == 0.0 { return Err("Division by zero".to_string()); }
                        Value::float(x as f64 / y)
                    } else if let (Some(x), Some(y)) = (va.as_float(), vb.as_int()) {
                        if y == 0 { return Err("Division by zero".to_string()); }
                        Value::float(x / y as f64)
                    } else {
                        return Err("/ expects numbers".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Mod(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = if let (Some(x), Some(y)) = (va.as_int(), vb.as_int()) {
                        if y == 0 { return Err("Division by zero".to_string()); }
                        Value::int(x % y)
                    } else {
                        return Err("mod expects integers".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Neg(dest, src) => {
                    let v = &self.registers[base + src as usize];
                    let result = if let Some(x) = v.as_int() {
                        Value::int(-x)
                    } else if let Some(x) = v.as_float() {
                        Value::float(-x)
                    } else {
                        return Err("- expects a number".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                // Specialized immediate arithmetic - avoids LoadConst instruction
                Op::AddImm(dest, src, imm) => {
                    let v = &self.registers[base + src as usize];
                    let result = if let Some(x) = v.as_int() {
                        Value::int(x + imm as i64)
                    } else if let Some(x) = v.as_float() {
                        Value::float(x + imm as f64)
                    } else {
                        return Err("+ expects numbers".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::SubImm(dest, src, imm) => {
                    let v = &self.registers[base + src as usize];
                    let result = if let Some(x) = v.as_int() {
                        Value::int(x - imm as i64)
                    } else if let Some(x) = v.as_float() {
                        Value::float(x - imm as f64)
                    } else {
                        return Err("- expects numbers".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::Lt(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::bool(compare_values(va, vb)? == std::cmp::Ordering::Less);
                    self.registers[base + dest as usize] = result;
                }

                Op::Le(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::bool(compare_values(va, vb)? != std::cmp::Ordering::Greater);
                    self.registers[base + dest as usize] = result;
                }

                Op::Gt(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::bool(compare_values(va, vb)? == std::cmp::Ordering::Greater);
                    self.registers[base + dest as usize] = result;
                }

                Op::Ge(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    let result = Value::bool(compare_values(va, vb)? != std::cmp::Ordering::Less);
                    self.registers[base + dest as usize] = result;
                }

                Op::Eq(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    self.registers[base + dest as usize] = Value::bool(va == vb);
                }

                Op::Ne(dest, a, b) => {
                    let va = &self.registers[base + a as usize];
                    let vb = &self.registers[base + b as usize];
                    self.registers[base + dest as usize] = Value::bool(va != vb);
                }

                Op::Not(dest, src) => {
                    let v = &self.registers[base + src as usize];
                    self.registers[base + dest as usize] = Value::bool(!v.is_truthy());
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
                    let result = if let (Some(items), Some(i)) = (list_val.as_list(), idx.as_int()) {
                        items.get(i as usize).cloned().unwrap_or(Value::nil())
                    } else {
                        return Err("GetList expects list and int".to_string());
                    };
                    self.registers[base + dest as usize] = result;
                }

                Op::SetList(_, _, _) => {
                    return Err("SetList not implemented (immutable lists)".to_string());
                }
            }
        }
    }

}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a VM with standard built-in functions
pub fn standard_vm() -> VM {
    use crate::value::NativeFunction;

    let mut vm = VM::new();

    let native = |name: &str, f: fn(&[Value]) -> Result<Value, String>| {
        Value::NativeFunction(Rc::new(NativeFunction {
            name: name.to_string(),
            func: f,
        }))
    };

    // Arithmetic
    vm.define_global("+", native("+", |args| {
        let mut sum = 0i64;
        let mut is_float = false;
        let mut fsum = 0.0f64;
        for arg in args {
            if let Some(n) = arg.as_int() {
                if is_float { fsum += n as f64 } else { sum += n }
            } else if let Some(n) = arg.as_float() {
                if !is_float { is_float = true; fsum = sum as f64; }
                fsum += n
            } else {
                return Err(format!("+ expects numbers, got {}", arg.type_name()));
            }
        }
        Ok(if is_float { Value::float(fsum) } else { Value::int(sum) })
    }));

    vm.define_global("-", native("-", |args| {
        if args.is_empty() { return Err("- expects at least 1 argument".to_string()); }
        if args.len() == 1 {
            if let Some(n) = args[0].as_int() {
                return Ok(Value::int(-n));
            } else if let Some(n) = args[0].as_float() {
                return Ok(Value::float(-n));
            } else {
                return Err("- expects numbers".to_string());
            }
        }
        let mut is_float = args[0].is_float();
        let mut result = if let Some(n) = args[0].as_int() {
            n as f64
        } else if let Some(n) = args[0].as_float() {
            n
        } else {
            return Err("- expects numbers".to_string());
        };
        for arg in &args[1..] {
            if let Some(n) = arg.as_int() {
                result -= n as f64;
            } else if let Some(n) = arg.as_float() {
                is_float = true;
                result -= n;
            } else {
                return Err("- expects numbers".to_string());
            }
        }
        Ok(if is_float { Value::float(result) } else { Value::int(result as i64) })
    }));

    vm.define_global("*", native("*", |args| {
        let mut prod = 1i64;
        let mut is_float = false;
        let mut fprod = 1.0f64;
        for arg in args {
            if let Some(n) = arg.as_int() {
                if is_float { fprod *= n as f64 } else { prod *= n }
            } else if let Some(n) = arg.as_float() {
                if !is_float { is_float = true; fprod = prod as f64; }
                fprod *= n
            } else {
                return Err("* expects numbers".to_string());
            }
        }
        Ok(if is_float { Value::float(fprod) } else { Value::int(prod) })
    }));

    vm.define_global("/", native("/", |args| {
        if args.len() != 2 { return Err("/ expects 2 arguments".to_string()); }
        let a = if let Some(n) = args[0].as_int() { n as f64 } else if let Some(n) = args[0].as_float() { n } else { return Err("/ expects numbers".to_string()) };
        let b = if let Some(n) = args[1].as_int() { n as f64 } else if let Some(n) = args[1].as_float() { n } else { return Err("/ expects numbers".to_string()) };
        if b == 0.0 { return Err("Division by zero".to_string()); }
        Ok(Value::float(a / b))
    }));

    vm.define_global("mod", native("mod", |args| {
        if args.len() != 2 { return Err("mod expects 2 arguments".to_string()); }
        if let (Some(a), Some(b)) = (args[0].as_int(), args[1].as_int()) {
            if b == 0 { return Err("Division by zero".to_string()); }
            Ok(Value::int(a % b))
        } else {
            Err("mod expects integers".to_string())
        }
    }));

    // Comparison
    vm.define_global("<", native("<", |args| {
        if args.len() != 2 { return Err("< expects 2 arguments".to_string()); }
        Ok(Value::bool(compare_values(&args[0], &args[1])? == std::cmp::Ordering::Less))
    }));

    vm.define_global("<=", native("<=", |args| {
        if args.len() != 2 { return Err("<= expects 2 arguments".to_string()); }
        Ok(Value::bool(compare_values(&args[0], &args[1])? != std::cmp::Ordering::Greater))
    }));

    vm.define_global(">", native(">", |args| {
        if args.len() != 2 { return Err("> expects 2 arguments".to_string()); }
        Ok(Value::bool(compare_values(&args[0], &args[1])? == std::cmp::Ordering::Greater))
    }));

    vm.define_global(">=", native(">=", |args| {
        if args.len() != 2 { return Err(">= expects 2 arguments".to_string()); }
        Ok(Value::bool(compare_values(&args[0], &args[1])? != std::cmp::Ordering::Less))
    }));

    vm.define_global("=", native("=", |args| {
        if args.len() != 2 { return Err("= expects 2 arguments".to_string()); }
        Ok(Value::bool(args[0] == args[1]))
    }));

    vm.define_global("!=", native("!=", |args| {
        if args.len() != 2 { return Err("!= expects 2 arguments".to_string()); }
        Ok(Value::bool(args[0] != args[1]))
    }));

    vm.define_global("not", native("not", |args| {
        if args.len() != 1 { return Err("not expects 1 argument".to_string()); }
        Ok(Value::bool(!args[0].is_truthy()))
    }));

    // List operations
    vm.define_global("list", native("list", |args| Ok(Value::list(args.to_vec()))));

    // cons creates a cons cell - O(1) operation!
    vm.define_global("cons", native("cons", |args| {
        if args.len() != 2 { return Err("cons expects 2 arguments".to_string()); }
        // Accept nil, list, or cons cell as tail
        let tail = &args[1];
        if !tail.is_nil() && tail.as_list().is_none() && tail.as_cons().is_none() {
            return Err("cons expects nil, list, or cons cell as second argument".to_string());
        }
        Ok(Value::cons(args[0].clone(), tail.clone()))
    }));

    // car returns the head of a list or cons cell - O(1) operation
    vm.define_global("car", native("car", |args| {
        if args.len() != 1 { return Err("car expects 1 argument".to_string()); }
        // Handle cons cells
        if let Some(cons) = args[0].as_cons() {
            return Ok(cons.car.clone());
        }
        // Handle array lists
        if let Some(list) = args[0].as_list() {
            return list.first().cloned().ok_or_else(|| "car on empty list".to_string());
        }
        Err("car expects a list or cons cell".to_string())
    }));

    // cdr returns the tail of a list or cons cell - O(1) for cons cells!
    vm.define_global("cdr", native("cdr", |args| {
        if args.len() != 1 { return Err("cdr expects 1 argument".to_string()); }
        // Handle cons cells - O(1)!
        if let Some(cons) = args[0].as_cons() {
            return Ok(cons.cdr.clone());
        }
        // Handle array lists - O(n) but rarely used now
        if let Some(list) = args[0].as_list() {
            if list.is_empty() { return Err("cdr on empty list".to_string()); }
            return Ok(Value::list(list[1..].to_vec()));
        }
        Err("cdr expects a list or cons cell".to_string())
    }));

    vm.define_global("length", native("length", |args| {
        if args.len() != 1 { return Err("length expects 1 argument".to_string()); }
        // Handle array lists
        if let Some(items) = args[0].as_list() {
            return Ok(Value::int(items.len() as i64));
        }
        // Handle cons cells - traverse and count
        if args[0].as_cons().is_some() {
            let mut count = 0i64;
            let mut current = &args[0];
            while let Some(cons) = current.as_cons() {
                count += 1;
                current = &cons.cdr;
            }
            // If we ended on an array list, add its length
            if let Some(items) = current.as_list() {
                count += items.len() as i64;
            }
            return Ok(Value::int(count));
        }
        // Handle strings
        if let Some(s) = args[0].as_string() {
            return Ok(Value::int(s.len() as i64));
        }
        Err("length expects list, cons cell, or string".to_string())
    }));

    // I/O
    vm.define_global("print", native("print", |args| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 { print!(" "); }
            if let Some(s) = arg.as_string() {
                print!("{}", s);
            } else {
                print!("{}", arg);
            }
        }
        Ok(Value::nil())
    }));

    vm.define_global("println", native("println", |args| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 { print!(" "); }
            if let Some(s) = arg.as_string() {
                print!("{}", s);
            } else {
                print!("{}", arg);
            }
        }
        println!();
        Ok(Value::nil())
    }));

    // Type predicates
    vm.define_global("nil?", native("nil?", |args| {
        if args.len() != 1 { return Err("nil? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].is_nil()))
    }));

    vm.define_global("int?", native("int?", |args| {
        if args.len() != 1 { return Err("int? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].is_int()))
    }));

    vm.define_global("float?", native("float?", |args| {
        if args.len() != 1 { return Err("float? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].is_float()))
    }));

    vm.define_global("string?", native("string?", |args| {
        if args.len() != 1 { return Err("string? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].as_string().is_some()))
    }));

    vm.define_global("list?", native("list?", |args| {
        if args.len() != 1 { return Err("list? expects 1 argument".to_string()); }
        // Both array lists and cons cells are "lists"
        Ok(Value::bool(args[0].as_list().is_some() || args[0].as_cons().is_some()))
    }));

    vm.define_global("empty?", native("empty?", |args| {
        if args.len() != 1 { return Err("empty? expects 1 argument".to_string()); }
        // nil is the empty list in Lisp
        if args[0].is_nil() {
            return Ok(Value::bool(true));
        }
        // Empty array list
        if let Some(list) = args[0].as_list() {
            return Ok(Value::bool(list.is_empty()));
        }
        // Cons cells are never empty (they have at least car)
        if args[0].as_cons().is_some() {
            return Ok(Value::bool(false));
        }
        // Empty string
        if let Some(s) = args[0].as_string() {
            return Ok(Value::bool(s.is_empty()));
        }
        Err("empty? expects nil, list, cons cell, or string".to_string())
    }));

    vm.define_global("fn?", native("fn?", |args| {
        if args.len() != 1 { return Err("fn? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].as_function().is_some() || args[0].as_native_function().is_some() || args[0].as_compiled_function().is_some()))
    }));

    vm.define_global("symbol?", native("symbol?", |args| {
        if args.len() != 1 { return Err("symbol? expects 1 argument".to_string()); }
        Ok(Value::bool(args[0].as_symbol().is_some()))
    }));

    // Symbol operations (useful for macros)
    vm.define_global("gensym", native("gensym", |_args| {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(Value::symbol(&format!("G__{}", id)))
    }));

    // Timing functions (for benchmarking)
    vm.define_global("clock", native("clock", |_args| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs_f64();
        Ok(Value::float(secs))
    }));

    vm
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
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return Ok(Value::int(int_op(x, y)));
    }
    if let (Some(x), Some(y)) = (a.as_float(), b.as_float()) {
        return Ok(Value::float(float_op(x, y)));
    }
    if let (Some(x), Some(y)) = (a.as_int(), b.as_float()) {
        return Ok(Value::float(float_op(x as f64, y)));
    }
    if let (Some(x), Some(y)) = (a.as_float(), b.as_int()) {
        return Ok(Value::float(float_op(x, y as f64)));
    }
    Err(format!("{} expects numbers", name))
}

fn compare_values(a: &Value, b: &Value) -> Result<std::cmp::Ordering, String> {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return Ok(x.cmp(&y));
    }
    if let (Some(x), Some(y)) = (a.as_float(), b.as_float()) {
        return x.partial_cmp(&y).ok_or_else(|| "Cannot compare NaN".to_string());
    }
    if let (Some(x), Some(y)) = (a.as_int(), b.as_float()) {
        return (x as f64).partial_cmp(&y).ok_or_else(|| "Cannot compare NaN".to_string());
    }
    if let (Some(x), Some(y)) = (a.as_float(), b.as_int()) {
        return x.partial_cmp(&(y as f64)).ok_or_else(|| "Cannot compare NaN".to_string());
    }
    Err(format!(
        "Cannot compare {} and {}",
        a.type_name(),
        b.type_name()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Op;

    #[test]
    fn test_vm_simple() {
        let mut chunk = Chunk::new();
        let idx = chunk.add_constant(Value::int(42));
        chunk.emit(Op::LoadConst(0, idx));
        chunk.emit(Op::Return(0));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::int(42));
    }

    #[test]
    fn test_vm_arithmetic() {
        let mut chunk = Chunk::new();
        let idx1 = chunk.add_constant(Value::int(10));
        let idx2 = chunk.add_constant(Value::int(3));
        chunk.emit(Op::LoadConst(0, idx1));
        chunk.emit(Op::LoadConst(1, idx2));
        chunk.emit(Op::Add(2, 0, 1));
        chunk.emit(Op::Return(2));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::int(13));
    }

    #[test]
    fn test_vm_comparison() {
        let mut chunk = Chunk::new();
        let idx1 = chunk.add_constant(Value::int(5));
        let idx2 = chunk.add_constant(Value::int(10));
        chunk.emit(Op::LoadConst(0, idx1));
        chunk.emit(Op::LoadConst(1, idx2));
        chunk.emit(Op::Lt(2, 0, 1));
        chunk.emit(Op::Return(2));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::bool(true));
    }

    #[test]
    fn test_vm_jump() {
        let mut chunk = Chunk::new();
        chunk.emit(Op::LoadTrue(0));
        let jump_pos = chunk.emit(Op::JumpIfFalse(0, 0));
        let idx = chunk.add_constant(Value::int(1));
        chunk.emit(Op::LoadConst(1, idx));
        let jump_over = chunk.emit(Op::Jump(0));
        let else_pos = chunk.current_pos();
        let idx2 = chunk.add_constant(Value::int(2));
        chunk.emit(Op::LoadConst(1, idx2));
        let end = chunk.current_pos();
        chunk.patch_jump(jump_pos, else_pos);
        chunk.patch_jump(jump_over, end);
        chunk.emit(Op::Return(1));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::int(1));
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

    // Integration tests - compile and run through VM
    fn vm_eval(input: &str) -> Result<Value, String> {
        use crate::compiler::Compiler;
        use crate::parser::parse;

        let expr = parse(input)?;
        let chunk = Compiler::compile(&expr)?;
        let mut vm = standard_vm();
        vm.run(chunk)
    }

    #[test]
    fn test_integration_literals() {
        assert_eq!(vm_eval("42").unwrap(), Value::Int(42));
        assert_eq!(vm_eval("3.14").unwrap(), Value::Float(3.14));
        assert_eq!(vm_eval("true").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("nil").unwrap(), Value::Nil);
    }

    #[test]
    fn test_integration_arithmetic() {
        assert_eq!(vm_eval("(+ 1 2)").unwrap(), Value::Int(3));
        assert_eq!(vm_eval("(+ 1 2 3)").unwrap(), Value::Int(6));
        assert_eq!(vm_eval("(- 5 3)").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(* 4 5)").unwrap(), Value::Int(20));
    }

    #[test]
    fn test_integration_comparison() {
        assert_eq!(vm_eval("(< 1 2)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(<= 2 2)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(> 3 2)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(= 1 1)").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_integration_if() {
        assert_eq!(vm_eval("(if true 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if false 1 2)").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(if nil 1 2)").unwrap(), Value::Int(2));
    }

    #[test]
    fn test_integration_def_and_fn() {
        assert_eq!(vm_eval("(do (def x 10) x)").unwrap(), Value::Int(10));
        assert_eq!(vm_eval("(do (def square (fn (n) (* n n))) (square 5))").unwrap(), Value::Int(25));
    }

    #[test]
    fn test_integration_let() {
        assert_eq!(vm_eval("(let (x 10) x)").unwrap(), Value::Int(10));
        assert_eq!(vm_eval("(let (x 10 y 20) (+ x y))").unwrap(), Value::Int(30));
    }

    #[test]
    fn test_integration_milestone() {
        // Phase 1 milestone
        let result = vm_eval("(do (def x 10) (def square (fn (n) (* n n))) (square x))").unwrap();
        assert_eq!(result, Value::Int(100));
    }

    #[test]
    fn test_integration_recursion() {
        let result = vm_eval("(do (def factorial (fn (n) (if (<= n 1) 1 (* n (factorial (- n 1)))))) (factorial 5))").unwrap();
        assert_eq!(result, Value::Int(120));
    }

    #[test]
    fn test_integration_tail_call() {
        // Phase 2 milestone - should not stack overflow
        let result = vm_eval("(do (def sum (fn (n acc) (if (<= n 0) acc (sum (- n 1) (+ acc n))))) (sum 10000 0))").unwrap();
        assert_eq!(result, Value::Int(50005000));
    }
}
