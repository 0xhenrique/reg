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
    // Cache for global lookups: (chunk_ptr, const_idx) -> Value
    // This avoids repeated HashMap lookups for recursive function calls
    global_cache: HashMap<(usize, u16), Value>,
}

impl VM {
    pub fn new() -> Self {
        VM {
            registers: vec![Value::Nil; MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals: Rc::new(RefCell::new(HashMap::new())),
            global_cache: HashMap::new(),
        }
    }

    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        VM {
            registers: vec![Value::Nil; MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals,
            global_cache: HashMap::new(),
        }
    }

    pub fn define_global(&mut self, name: &str, value: Value) {
        self.globals.borrow_mut().insert(name.to_string(), value);
        // Invalidate cache when globals change
        self.global_cache.clear();
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
                    // Use chunk pointer as part of cache key for uniqueness
                    let chunk_ptr = Rc::as_ptr(&chunk) as usize;
                    let cache_key = (chunk_ptr, name_idx);

                    let value = if let Some(cached) = self.global_cache.get(&cache_key) {
                        // Cache hit - avoid HashMap lookup entirely
                        cached.clone()
                    } else {
                        // Cache miss - do the lookup (using &str to avoid String allocation)
                        let name: &str = match &chunk.constants[name_idx as usize] {
                            Value::Symbol(s) => s,
                            _ => return Err("GetGlobal: expected symbol".to_string()),
                        };
                        let v = self.globals.borrow().get(name).cloned()
                            .ok_or_else(|| format!("Undefined variable: {}", name))?;
                        // Cache for future lookups
                        self.global_cache.insert(cache_key, v.clone());
                        v
                    };
                    self.registers[base + dest as usize] = value;
                }

                Op::SetGlobal(name_idx, src) => {
                    let name: &str = match &chunk.constants[name_idx as usize] {
                        Value::Symbol(s) => s,
                        _ => return Err("SetGlobal: expected symbol".to_string()),
                    };
                    let value = self.registers[base + src as usize].clone();
                    self.globals.borrow_mut().insert(name.to_string(), value);
                    // Invalidate cache - global was modified
                    self.global_cache.clear();
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

                    match &func {
                        Value::CompiledFunction(cf) => {
                            if cf.num_params != nargs {
                                return Err(format!(
                                    "Expected {} arguments, got {}",
                                    cf.num_params, nargs
                                ));
                            }

                            let new_base = base + chunk.num_registers as usize;

                            if new_base + cf.num_registers as usize > MAX_REGISTERS {
                                return Err("Stack overflow".to_string());
                            }

                            // Copy args directly to new frame's registers (no Vec allocation!)
                            let arg_start = base + func_reg as usize + 1;
                            for i in 0..nargs as usize {
                                self.registers[new_base + i] = self.registers[arg_start + i].clone();
                            }

                            // Push new frame - will continue in the loop
                            self.frames.push(CallFrame {
                                chunk: cf.clone(),
                                ip: 0,
                                base: new_base,
                                return_reg: base + dest as usize,
                            });
                        }
                        Value::NativeFunction(nf) => {
                            // Pass a slice directly to native function (no Vec allocation!)
                            let arg_start = base + func_reg as usize + 1;
                            let arg_end = arg_start + nargs as usize;
                            let result = (nf.func)(&self.registers[arg_start..arg_end])?;
                            self.registers[base + dest as usize] = result;
                        }
                        Value::Function(_) => {
                            return Err("Cannot call interpreted function from VM".to_string());
                        }
                        _ => return Err(format!("Not a function: {}", func)),
                    }
                }

                Op::TailCall(func_reg, nargs) => {
                    let func = self.registers[base + func_reg as usize].clone();

                    match &func {
                        Value::CompiledFunction(cf) => {
                            if cf.num_params != nargs {
                                return Err(format!(
                                    "Expected {} arguments, got {}",
                                    cf.num_params, nargs
                                ));
                            }
                            // Reuse current frame for tail call
                            if let Some(frame) = self.frames.last_mut() {
                                frame.chunk = cf.clone();
                                frame.ip = 0;
                            }
                            // Copy args directly to base registers (no Vec allocation!)
                            // Forward iteration is safe: source (base + func_reg + 1 + i)
                            // is always >= destination (base + i) since func_reg >= 0
                            let arg_start = base + func_reg as usize + 1;
                            for i in 0..nargs as usize {
                                self.registers[base + i] = self.registers[arg_start + i].clone();
                            }
                        }
                        Value::NativeFunction(nf) => {
                            // Pass a slice directly to native function (no Vec allocation!)
                            let arg_start = base + func_reg as usize + 1;
                            let arg_end = arg_start + nargs as usize;
                            let result = (nf.func)(&self.registers[arg_start..arg_end])?;
                            let return_reg = self.frames.last().unwrap().return_reg;
                            self.frames.pop();
                            if self.frames.is_empty() {
                                return Ok(result);
                            }
                            // Store result in caller's designated register
                            self.registers[return_reg] = result;
                        }
                        Value::Function(_) => {
                            return Err("Cannot call interpreted function from VM".to_string());
                        }
                        _ => return Err(format!("Not a function: {}", func)),
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
            match arg {
                Value::Int(n) => if is_float { fsum += *n as f64 } else { sum += n },
                Value::Float(n) => { if !is_float { is_float = true; fsum = sum as f64; } fsum += n },
                _ => return Err(format!("+ expects numbers, got {}", arg.type_name())),
            }
        }
        Ok(if is_float { Value::Float(fsum) } else { Value::Int(sum) })
    }));

    vm.define_global("-", native("-", |args| {
        if args.is_empty() { return Err("- expects at least 1 argument".to_string()); }
        if args.len() == 1 {
            return match &args[0] {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(n) => Ok(Value::Float(-n)),
                _ => Err("- expects numbers".to_string()),
            };
        }
        let mut result = match &args[0] {
            Value::Int(n) => *n as f64,
            Value::Float(n) => *n,
            _ => return Err("- expects numbers".to_string()),
        };
        let mut is_float = matches!(&args[0], Value::Float(_));
        for arg in &args[1..] {
            match arg {
                Value::Int(n) => result -= *n as f64,
                Value::Float(n) => { is_float = true; result -= n },
                _ => return Err("- expects numbers".to_string()),
            }
        }
        Ok(if is_float { Value::Float(result) } else { Value::Int(result as i64) })
    }));

    vm.define_global("*", native("*", |args| {
        let mut prod = 1i64;
        let mut is_float = false;
        let mut fprod = 1.0f64;
        for arg in args {
            match arg {
                Value::Int(n) => if is_float { fprod *= *n as f64 } else { prod *= n },
                Value::Float(n) => { if !is_float { is_float = true; fprod = prod as f64; } fprod *= n },
                _ => return Err("* expects numbers".to_string()),
            }
        }
        Ok(if is_float { Value::Float(fprod) } else { Value::Int(prod) })
    }));

    vm.define_global("/", native("/", |args| {
        if args.len() != 2 { return Err("/ expects 2 arguments".to_string()); }
        let a = match &args[0] { Value::Int(n) => *n as f64, Value::Float(n) => *n, _ => return Err("/ expects numbers".to_string()) };
        let b = match &args[1] { Value::Int(n) => *n as f64, Value::Float(n) => *n, _ => return Err("/ expects numbers".to_string()) };
        if b == 0.0 { return Err("Division by zero".to_string()); }
        Ok(Value::Float(a / b))
    }));

    vm.define_global("mod", native("mod", |args| {
        if args.len() != 2 { return Err("mod expects 2 arguments".to_string()); }
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 { return Err("Division by zero".to_string()); }
                Ok(Value::Int(a % b))
            }
            _ => Err("mod expects integers".to_string()),
        }
    }));

    // Comparison
    vm.define_global("<", native("<", |args| {
        if args.len() != 2 { return Err("< expects 2 arguments".to_string()); }
        Ok(Value::Bool(compare_values(&args[0], &args[1])? == std::cmp::Ordering::Less))
    }));

    vm.define_global("<=", native("<=", |args| {
        if args.len() != 2 { return Err("<= expects 2 arguments".to_string()); }
        Ok(Value::Bool(compare_values(&args[0], &args[1])? != std::cmp::Ordering::Greater))
    }));

    vm.define_global(">", native(">", |args| {
        if args.len() != 2 { return Err("> expects 2 arguments".to_string()); }
        Ok(Value::Bool(compare_values(&args[0], &args[1])? == std::cmp::Ordering::Greater))
    }));

    vm.define_global(">=", native(">=", |args| {
        if args.len() != 2 { return Err(">= expects 2 arguments".to_string()); }
        Ok(Value::Bool(compare_values(&args[0], &args[1])? != std::cmp::Ordering::Less))
    }));

    vm.define_global("=", native("=", |args| {
        if args.len() != 2 { return Err("= expects 2 arguments".to_string()); }
        Ok(Value::Bool(args[0] == args[1]))
    }));

    vm.define_global("!=", native("!=", |args| {
        if args.len() != 2 { return Err("!= expects 2 arguments".to_string()); }
        Ok(Value::Bool(args[0] != args[1]))
    }));

    vm.define_global("not", native("not", |args| {
        if args.len() != 1 { return Err("not expects 1 argument".to_string()); }
        Ok(Value::Bool(!args[0].is_truthy()))
    }));

    // List operations
    vm.define_global("list", native("list", |args| Ok(Value::list(args.to_vec()))));

    vm.define_global("cons", native("cons", |args| {
        if args.len() != 2 { return Err("cons expects 2 arguments".to_string()); }
        let tail = args[1].as_list().ok_or("cons expects list as second argument")?;
        let mut new_list = vec![args[0].clone()];
        new_list.extend(tail.iter().cloned());
        Ok(Value::list(new_list))
    }));

    vm.define_global("car", native("car", |args| {
        if args.len() != 1 { return Err("car expects 1 argument".to_string()); }
        let list = args[0].as_list().ok_or("car expects a list")?;
        list.first().cloned().ok_or_else(|| "car on empty list".to_string())
    }));

    vm.define_global("cdr", native("cdr", |args| {
        if args.len() != 1 { return Err("cdr expects 1 argument".to_string()); }
        let list = args[0].as_list().ok_or("cdr expects a list")?;
        if list.is_empty() { return Err("cdr on empty list".to_string()); }
        Ok(Value::list(list[1..].to_vec()))
    }));

    vm.define_global("length", native("length", |args| {
        if args.len() != 1 { return Err("length expects 1 argument".to_string()); }
        match &args[0] {
            Value::List(items) => Ok(Value::Int(items.len() as i64)),
            Value::String(s) => Ok(Value::Int(s.len() as i64)),
            _ => Err("length expects list or string".to_string()),
        }
    }));

    // I/O
    vm.define_global("print", native("print", |args| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 { print!(" "); }
            match arg {
                Value::String(s) => print!("{}", s),
                other => print!("{}", other),
            }
        }
        Ok(Value::Nil)
    }));

    vm.define_global("println", native("println", |args| {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 { print!(" "); }
            match arg {
                Value::String(s) => print!("{}", s),
                other => print!("{}", other),
            }
        }
        println!();
        Ok(Value::Nil)
    }));

    // Type predicates
    vm.define_global("nil?", native("nil?", |args| {
        if args.len() != 1 { return Err("nil? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::Nil)))
    }));

    vm.define_global("int?", native("int?", |args| {
        if args.len() != 1 { return Err("int? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::Int(_))))
    }));

    vm.define_global("float?", native("float?", |args| {
        if args.len() != 1 { return Err("float? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::Float(_))))
    }));

    vm.define_global("string?", native("string?", |args| {
        if args.len() != 1 { return Err("string? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::String(_))))
    }));

    vm.define_global("list?", native("list?", |args| {
        if args.len() != 1 { return Err("list? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::List(_))))
    }));

    vm.define_global("fn?", native("fn?", |args| {
        if args.len() != 1 { return Err("fn? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::Function(_) | Value::NativeFunction(_) | Value::CompiledFunction(_))))
    }));

    vm.define_global("symbol?", native("symbol?", |args| {
        if args.len() != 1 { return Err("symbol? expects 1 argument".to_string()); }
        Ok(Value::Bool(matches!(args[0], Value::Symbol(_))))
    }));

    // Symbol operations (useful for macros)
    vm.define_global("gensym", native("gensym", |_args| {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(Value::symbol(&format!("G__{}", id)))
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
