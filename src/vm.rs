use crate::bytecode::{Chunk, Op};
use crate::jit::{is_jit_compatible, JitCompiler};
use crate::value::{intern_symbol, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

// Register file size - each function call uses ~(locals + 16) registers
// For 10000 deep recursion with ~20 registers per call = 200000 registers
const MAX_REGISTERS: usize = 262144;  // 256K registers (~2MB) for deep recursion
const MAX_FRAMES: usize = 16384;      // 16K frames for deep recursion

/// A wrapper for interned symbol Rc<str> that hashes/compares by pointer
/// Since symbols are interned, equal symbols share the same Rc, so pointer
/// comparison is correct and O(1) instead of O(n) string comparison
#[derive(Clone)]
struct SymbolKey(Rc<str>);

impl Hash for SymbolKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by pointer address only - O(1)
        // Cast fat pointer (*const str) to thin pointer (*const u8) first
        (Rc::as_ptr(&self.0) as *const u8 as usize).hash(state);
    }
}

impl PartialEq for SymbolKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Compare by pointer - O(1) since symbols are interned
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SymbolKey {}

#[derive(Clone)]
struct CallFrame {
    chunk: Rc<Chunk>,
    ip: usize,
    base: usize,
    return_reg: usize, // where to store return value in caller's frame
}

/// JIT compilation threshold (number of calls before compiling)
const JIT_THRESHOLD: u32 = 100;

pub struct VM {
    registers: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Rc<RefCell<HashMap<String, Value>>>,
    // Cache for global lookups: SymbolKey -> Value
    // Uses pointer-based hashing for O(1) lookup instead of string hashing
    global_cache: HashMap<SymbolKey, Value>,
    // Inline caches for function calls - store typed function pointers directly
    // This eliminates the as_compiled_function()/as_native_function() type checks
    compiled_fn_cache: HashMap<SymbolKey, Rc<Chunk>>,
    // For native functions, we only need the function pointer
    native_fn_cache: HashMap<SymbolKey, fn(&[Value]) -> Result<Value, String>>,
    // JIT compiler (lazily initialized when jit_enabled is true)
    jit: Option<JitCompiler>,
    // Call counts for JIT compilation triggering (chunk pointer -> count)
    call_counts: HashMap<usize, u32>,
    // JIT enabled flag (opt-in via --jit flag)
    jit_enabled: bool,
}

impl VM {
    pub fn new() -> Self {
        VM {
            registers: vec![Value::nil(); MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals: Rc::new(RefCell::new(HashMap::new())),
            global_cache: HashMap::new(),
            compiled_fn_cache: HashMap::new(),
            native_fn_cache: HashMap::new(),
            jit: None,
            call_counts: HashMap::new(),
            jit_enabled: false,
        }
    }

    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        VM {
            registers: vec![Value::nil(); MAX_REGISTERS],
            frames: Vec::with_capacity(MAX_FRAMES),
            globals,
            global_cache: HashMap::new(),
            compiled_fn_cache: HashMap::new(),
            native_fn_cache: HashMap::new(),
            jit: None,
            call_counts: HashMap::new(),
            jit_enabled: false,
        }
    }

    /// Enable JIT compilation (opt-in via --jit flag)
    pub fn enable_jit(&mut self) {
        self.jit_enabled = true;
    }

    /// Get a global variable by name (returns None if not found)
    pub fn get_global(&self, name: &str) -> Option<Value> {
        self.globals.borrow().get(name).cloned()
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
        // Update cache using interned symbol key
        let key = SymbolKey(intern_symbol(name));

        // Populate inline caches for functions - this allows the VM skip
        // the type check in CALL_GLOBAL/TAIL_CALL_GLOBAL
        if let Some(cf) = value.as_compiled_function() {
            self.compiled_fn_cache.insert(key.clone(), cf.clone());
            self.native_fn_cache.remove(&key);
        } else if let Some(nf) = value.as_native_function() {
            self.native_fn_cache.insert(key.clone(), nf.func);
            self.compiled_fn_cache.remove(&key);
        } else {
            // Not a function so clear both inline caches
            self.compiled_fn_cache.remove(&key);
            self.native_fn_cache.remove(&key);
        }

        self.global_cache.insert(key, value);
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

    /// Dispatch loop with packed 4-byte instructions
    /// Uses opcode-based dispatch with match on u8 opcode constants
    #[inline(never)] // Prevent inlining to keep hot loop tight in cache
    fn execute(&mut self) -> Result<Value, String> {
        // Cache frequently accessed values outside the loop
        // These get reloaded on frame changes (call/return)
        let mut code_ptr: *const Op;
        let mut constants_ptr: *const Value;
        let mut ip: usize;
        let mut base: usize;

        // Initialize from current frame
        {
            let frame = unsafe { self.frames.last().unwrap_unchecked() };
            code_ptr = frame.chunk.code.as_ptr();
            constants_ptr = frame.chunk.constants.as_ptr();
            ip = frame.ip;
            base = frame.base;
        }

        // This loop is necessary so LLVM can apply DFA jump threading and generate a jump table
        // The explicit labels and continues give LLVM better hints for jump tables
        // More: https://github.com/llvm/llvm-project/commit/02077da7e7a8ff76c0576bb33adb462c337013f5
        loop {
            // Fetch instruction - pure pointer arithmetic, no bounds check
            let instr = unsafe { *code_ptr.add(ip) };
            ip += 1;

            // Dispatch using match on opcode (LLVM should give a dense jump table here)
            match instr.opcode() {
                Op::LOAD_CONST => {
                    let dest = instr.a();
                    let idx = instr.bx();
                    let value = unsafe { (*constants_ptr.add(idx as usize)).clone() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = value };
                }

                Op::LOAD_NIL => {
                    let dest = instr.a();
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::nil() };
                }

                Op::LOAD_TRUE => {
                    let dest = instr.a();
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(true) };
                }

                Op::LOAD_FALSE => {
                    let dest = instr.a();
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(false) };
                }

                Op::MOVE => {
                    let dest = instr.a();
                    let src = instr.b();
                    let value = unsafe { self.registers.get_unchecked(base + src as usize).clone() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = value };
                }

                Op::GET_GLOBAL => {
                    let dest = instr.a();
                    let name_idx = instr.bx();
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    let symbol_rc = unsafe { frame.chunk.constants.get_unchecked(name_idx as usize) }
                        .as_symbol_rc()
                        .ok_or("GetGlobal: expected symbol")?;
                    let key = SymbolKey(symbol_rc);

                    let value = if let Some(cached) = self.global_cache.get(&key) {
                        cached.clone()
                    } else {
                        let v = self.globals.borrow().get(&*key.0).cloned()
                            .ok_or_else(|| format!("Undefined variable: {}", &*key.0))?;
                        self.global_cache.insert(key, v.clone());
                        v
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = value };
                }

                Op::SET_GLOBAL => {
                    let src = instr.a();
                    let name_idx = instr.bx();
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    let symbol_rc = unsafe { frame.chunk.constants.get_unchecked(name_idx as usize) }
                        .as_symbol_rc()
                        .ok_or("SetGlobal: expected symbol")?;
                    let value = unsafe { self.registers.get_unchecked(base + src as usize).clone() };
                    {
                        let mut globals = self.globals.borrow_mut();
                        if let Some(existing) = globals.get_mut(&*symbol_rc) {
                            *existing = value.clone();
                        } else {
                            globals.insert(symbol_rc.to_string(), value.clone());
                        }
                    }
                    let key = SymbolKey(symbol_rc);

                    // Update inline caches on global redefinition
                    if let Some(cf) = value.as_compiled_function() {
                        self.compiled_fn_cache.insert(key.clone(), cf.clone());
                        self.native_fn_cache.remove(&key);
                    } else if let Some(nf) = value.as_native_function() {
                        self.native_fn_cache.insert(key.clone(), nf.func);
                        self.compiled_fn_cache.remove(&key);
                    } else {
                        self.compiled_fn_cache.remove(&key);
                        self.native_fn_cache.remove(&key);
                    }

                    self.global_cache.insert(key, value);
                }

                Op::CLOSURE => {
                    let dest = instr.a();
                    let proto_idx = instr.bx();
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    let proto = unsafe { frame.chunk.protos.get_unchecked(proto_idx as usize).clone() };
                    let func = Value::CompiledFunction(Rc::new(proto));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = func };
                }

                Op::JUMP => {
                    let offset = instr.sbx();
                    ip = (ip as isize + offset as isize) as usize;
                }

                Op::JUMP_IF_FALSE => {
                    let reg = instr.a();
                    let offset = instr.sbx();
                    if !unsafe { self.registers.get_unchecked(base + reg as usize) }.is_truthy() {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_TRUE => {
                    let reg = instr.a();
                    let offset = instr.sbx();
                    if unsafe { self.registers.get_unchecked(base + reg as usize) }.is_truthy() {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::CALL => {
                    let dest = instr.a();
                    let func_reg = instr.b();
                    let nargs = instr.c();
                    let func_val = unsafe { self.registers.get_unchecked(base + func_reg as usize) };
                    if let Some(cf) = func_val.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let frame = unsafe { self.frames.last().unwrap_unchecked() };
                        let num_registers = frame.chunk.num_registers;
                        let new_base = base + num_registers as usize;

                        if new_base + cf.num_registers as usize > MAX_REGISTERS {
                            return Err("Stack overflow".to_string());
                        }

                        // Save current IP to frame before pushing new frame
                        unsafe { self.frames.last_mut().unwrap_unchecked() }.ip = ip;

                        let cf_chunk = cf.clone();

                        let arg_start = base + func_reg as usize + 1;
                        for i in 0..nargs as usize {
                            unsafe {
                                *self.registers.get_unchecked_mut(new_base + i) =
                                    self.registers.get_unchecked(arg_start + i).clone();
                            }
                        }

                        self.frames.push(CallFrame {
                            chunk: cf_chunk,
                            ip: 0,
                            base: new_base,
                            return_reg: base + dest as usize,
                        });

                        // Update cached frame values
                        let frame = unsafe { self.frames.last().unwrap_unchecked() };
                        code_ptr = frame.chunk.code.as_ptr();
                        constants_ptr = frame.chunk.constants.as_ptr();
                        ip = 0;
                        base = new_base;
                    } else if let Some(nf) = func_val.as_native_function() {
                        let func_ptr = nf.func;
                        let arg_start = base + func_reg as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        // Native function calls need slice
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                    } else if func_val.as_function().is_some() {
                        return Err("Cannot call interpreted function from VM".to_string());
                    } else {
                        return Err(format!("Not a function: {}", func_val));
                    }
                }

                Op::TAIL_CALL => {
                    let func_reg = instr.a();
                    let nargs = instr.b();
                    let func_val = unsafe { self.registers.get_unchecked(base + func_reg as usize) };
                    if let Some(cf) = func_val.as_compiled_function() {
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let frame = unsafe { self.frames.last_mut().unwrap_unchecked() };
                        if !Rc::ptr_eq(&frame.chunk, cf) {
                            frame.chunk = cf.clone();
                            // Update cached pointers for new function
                            code_ptr = frame.chunk.code.as_ptr();
                            constants_ptr = frame.chunk.constants.as_ptr();
                        }

                        let arg_start = base + func_reg as usize + 1;
                        for i in 0..nargs as usize {
                            unsafe {
                                // Use take instead of clone
                                // The source registers won't be accessed again after this tail call
                                *self.registers.get_unchecked_mut(base + i) =
                                    self.registers.get_unchecked_mut(arg_start + i).take();
                            }
                        }
                        ip = 0;
                    } else if let Some(nf) = func_val.as_native_function() {
                        let func_ptr = nf.func;
                        let return_reg = unsafe { self.frames.last().unwrap_unchecked() }.return_reg;
                        let arg_start = base + func_reg as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        // Native function calls need slice
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        self.frames.pop();
                        if self.frames.is_empty() {
                            return Ok(result);
                        }
                        unsafe { *self.registers.get_unchecked_mut(return_reg) = result };

                        // Update cached frame values
                        let frame = unsafe { self.frames.last().unwrap_unchecked() };
                        code_ptr = frame.chunk.code.as_ptr();
                        constants_ptr = frame.chunk.constants.as_ptr();
                        ip = frame.ip;
                        base = frame.base;
                    } else if func_val.as_function().is_some() {
                        return Err("Cannot call interpreted function from VM".to_string());
                    } else {
                        return Err(format!("Not a function: {}", func_val));
                    }
                }

                Op::CALL_GLOBAL => {
                    let dest = instr.a();
                    let name_idx = instr.b(); // 8-bit constant index
                    let nargs = instr.c();
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    let symbol_rc = unsafe { frame.chunk.constants.get_unchecked(name_idx as usize) }
                        .as_symbol_rc()
                        .ok_or("CallGlobal: expected symbol")?;
                    let key = SymbolKey(symbol_rc);

                    // INLINE CACHE: Check typed caches first (no type check needed)
                    if let Some(cf) = self.compiled_fn_cache.get(&key) {
                        // Fast path: compiled function from inline cache
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let num_registers = frame.chunk.num_registers;
                        let new_base = base + num_registers as usize;
                        let cf_num_registers = cf.num_registers as usize;

                        if new_base + cf_num_registers > MAX_REGISTERS {
                            return Err("Stack overflow".to_string());
                        }

                        // Clone chunk and get pointer before releasing borrow
                        let cf_chunk = cf.clone();
                        let chunk_ptr = Rc::as_ptr(&cf_chunk) as usize;

                        // Copy arguments to new frame's registers
                        let arg_start = base + dest as usize + 1;
                        for i in 0..nargs as usize {
                            unsafe {
                                *self.registers.get_unchecked_mut(new_base + i) =
                                    self.registers.get_unchecked(arg_start + i).clone();
                            }
                        }

                        // Try JIT execution first (fast path) - only if JIT is enabled
                        if self.jit_enabled {
                            if let Some(jit) = &self.jit {
                                if let Some(jit_code) = jit.get_compiled(chunk_ptr) {
                                    let result = unsafe {
                                        jit_code.execute(&mut self.registers[new_base..new_base + cf_num_registers])
                                    };
                                    if let Ok(value) = result {
                                        // JIT succeeded - store result and continue
                                        unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = value };
                                        continue;
                                    }
                                }
                            }

                            // JIT not available or failed - fall back to interpreter
                            // Update call count for potential future JIT compilation
                            {
                                let count = self.call_counts.entry(chunk_ptr).or_insert(0);
                                *count += 1;
                                if *count == JIT_THRESHOLD && is_jit_compatible(&cf_chunk) {
                                    // Trigger JIT compilation (lazily, will be used on next call)
                                    if self.jit.is_none() {
                                        self.jit = JitCompiler::new().ok();
                                    }
                                    if let Some(jit) = &mut self.jit {
                                        let name = format!("jit_func_{:x}", chunk_ptr);
                                        let _ = jit.compile_function(&cf_chunk, &name);
                                    }
                                }
                            }
                        }

                        // Save current IP
                        unsafe { self.frames.last_mut().unwrap_unchecked() }.ip = ip;

                        self.frames.push(CallFrame {
                            chunk: cf_chunk,
                            ip: 0,
                            base: new_base,
                            return_reg: base + dest as usize,
                        });

                        // Update cached frame values
                        let frame = unsafe { self.frames.last().unwrap_unchecked() };
                        code_ptr = frame.chunk.code.as_ptr();
                        constants_ptr = frame.chunk.constants.as_ptr();
                        ip = 0;
                        base = new_base;
                    } else if let Some(&func_ptr) = self.native_fn_cache.get(&key) {
                        // Fast path: native function from inline cache
                        let arg_start = base + dest as usize + 1;
                        let arg_end = arg_start + nargs as usize;
                        let result = func_ptr(&self.registers[arg_start..arg_end])?;
                        unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                    } else {
                        // Slow path: fall back to global cache with type check
                        use std::collections::hash_map::Entry;
                        let func_value = match self.global_cache.entry(key.clone()) {
                            Entry::Occupied(e) => e.into_mut(),
                            Entry::Vacant(e) => {
                                let name = &*e.key().0;
                                let v = self.globals.borrow().get(name).cloned()
                                    .ok_or_else(|| format!("Undefined function: {}", name))?;
                                e.insert(v)
                            }
                        };

                        if let Some(cf) = func_value.as_compiled_function() {
                            // Populate inline cache for next time
                            self.compiled_fn_cache.insert(key, cf.clone());

                            if cf.num_params != nargs {
                                return Err(format!(
                                    "Expected {} arguments, got {}",
                                    cf.num_params, nargs
                                ));
                            }

                            let num_registers = frame.chunk.num_registers;
                            let new_base = base + num_registers as usize;

                            if new_base + cf.num_registers as usize > MAX_REGISTERS {
                                return Err("Stack overflow".to_string());
                            }

                            unsafe { self.frames.last_mut().unwrap_unchecked() }.ip = ip;

                            let cf_chunk = cf.clone();

                            let arg_start = base + dest as usize + 1;
                            for i in 0..nargs as usize {
                                unsafe {
                                    *self.registers.get_unchecked_mut(new_base + i) =
                                        self.registers.get_unchecked(arg_start + i).clone();
                                }
                            }

                            self.frames.push(CallFrame {
                                chunk: cf_chunk,
                                ip: 0,
                                base: new_base,
                                return_reg: base + dest as usize,
                            });

                            let frame = unsafe { self.frames.last().unwrap_unchecked() };
                            code_ptr = frame.chunk.code.as_ptr();
                            constants_ptr = frame.chunk.constants.as_ptr();
                            ip = 0;
                            base = new_base;
                        } else if let Some(nf) = func_value.as_native_function() {
                            // Populate inline cache for next time
                            self.native_fn_cache.insert(key, nf.func);

                            let func_ptr = nf.func;
                            let arg_start = base + dest as usize + 1;
                            let arg_end = arg_start + nargs as usize;
                            let result = func_ptr(&self.registers[arg_start..arg_end])?;
                            unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                        } else {
                            return Err(format!("Not a function: {}", func_value));
                        }
                    }
                }

                Op::TAIL_CALL_GLOBAL => {
                    let name_idx = instr.a(); // 8-bit constant index
                    let first_arg = instr.b();
                    let nargs = instr.c();
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    let symbol_rc = unsafe { frame.chunk.constants.get_unchecked(name_idx as usize) }
                        .as_symbol_rc()
                        .ok_or("TailCallGlobal: expected symbol")?;
                    let key = SymbolKey(symbol_rc);

                    // INLINE CACHE: Check typed caches first (no type check needed)
                    if let Some(cf) = self.compiled_fn_cache.get(&key) {
                        // Fast path: compiled function from inline cache
                        if cf.num_params != nargs {
                            return Err(format!(
                                "Expected {} arguments, got {}",
                                cf.num_params, nargs
                            ));
                        }

                        let frame = unsafe { self.frames.last_mut().unwrap_unchecked() };
                        if !Rc::ptr_eq(&frame.chunk, cf) {
                            frame.chunk = cf.clone();
                            code_ptr = frame.chunk.code.as_ptr();
                            constants_ptr = frame.chunk.constants.as_ptr();
                        }

                        let src_start = base + first_arg as usize;
                        for i in 0..nargs as usize {
                            unsafe {
                                // Use take instead of clone
                                // The source registers won't be accessed again after this tail call
                                *self.registers.get_unchecked_mut(base + i) =
                                    self.registers.get_unchecked_mut(src_start + i).take();
                            }
                        }
                        ip = 0;
                    } else if let Some(&func_ptr) = self.native_fn_cache.get(&key) {
                        // Fast path: native function from inline cache
                        let return_reg = unsafe { self.frames.last().unwrap_unchecked() }.return_reg;
                        let src_start = base + first_arg as usize;
                        let src_end = src_start + nargs as usize;
                        let result = func_ptr(&self.registers[src_start..src_end])?;
                        self.frames.pop();
                        if self.frames.is_empty() {
                            return Ok(result);
                        }
                        unsafe { *self.registers.get_unchecked_mut(return_reg) = result };

                        // Update cached frame values
                        let frame = unsafe { self.frames.last().unwrap_unchecked() };
                        code_ptr = frame.chunk.code.as_ptr();
                        constants_ptr = frame.chunk.constants.as_ptr();
                        ip = frame.ip;
                        base = frame.base;
                    } else {
                        // Slow path: fall back to global cache with type check
                        use std::collections::hash_map::Entry;
                        let func_value = match self.global_cache.entry(key.clone()) {
                            Entry::Occupied(e) => e.into_mut(),
                            Entry::Vacant(e) => {
                                let name = &*e.key().0;
                                let v = self.globals.borrow().get(name).cloned()
                                    .ok_or_else(|| format!("Undefined function: {}", name))?;
                                e.insert(v)
                            }
                        };

                        if let Some(cf) = func_value.as_compiled_function() {
                            // Populate inline cache for next time
                            self.compiled_fn_cache.insert(key, cf.clone());

                            if cf.num_params != nargs {
                                return Err(format!(
                                    "Expected {} arguments, got {}",
                                    cf.num_params, nargs
                                ));
                            }

                            let frame = unsafe { self.frames.last_mut().unwrap_unchecked() };
                            if !Rc::ptr_eq(&frame.chunk, cf) {
                                frame.chunk = cf.clone();
                                code_ptr = frame.chunk.code.as_ptr();
                                constants_ptr = frame.chunk.constants.as_ptr();
                            }

                            let src_start = base + first_arg as usize;
                            for i in 0..nargs as usize {
                                unsafe {
                                    // Use take instead of clone
                                    // The source registers won't be accessed again after this tail call
                                    *self.registers.get_unchecked_mut(base + i) =
                                        self.registers.get_unchecked_mut(src_start + i).take();
                                }
                            }
                            ip = 0;
                        } else if let Some(nf) = func_value.as_native_function() {
                            // Populate inline cache for next time
                            self.native_fn_cache.insert(key, nf.func);

                            let func_ptr = nf.func;
                            let return_reg = unsafe { self.frames.last().unwrap_unchecked() }.return_reg;
                            let src_start = base + first_arg as usize;
                            let src_end = src_start + nargs as usize;
                            let result = func_ptr(&self.registers[src_start..src_end])?;
                            self.frames.pop();
                            if self.frames.is_empty() {
                                return Ok(result);
                            }
                            unsafe { *self.registers.get_unchecked_mut(return_reg) = result };

                            // Update cached frame values
                            let frame = unsafe { self.frames.last().unwrap_unchecked() };
                            code_ptr = frame.chunk.code.as_ptr();
                            constants_ptr = frame.chunk.constants.as_ptr();
                            ip = frame.ip;
                            base = frame.base;
                        } else {
                            return Err(format!("Not a function: {}", func_value));
                        }
                    }
                }

                Op::RETURN => {
                    let src = instr.a();
                    let result = unsafe { self.registers.get_unchecked(base + src as usize).clone() };
                    let return_reg = unsafe { self.frames.last().unwrap_unchecked() }.return_reg;
                    self.frames.pop();

                    if self.frames.is_empty() {
                        return Ok(result);
                    }
                    unsafe { *self.registers.get_unchecked_mut(return_reg) = result };

                    // Update cached frame values
                    let frame = unsafe { self.frames.last().unwrap_unchecked() };
                    code_ptr = frame.chunk.code.as_ptr();
                    constants_ptr = frame.chunk.constants.as_ptr();
                    ip = frame.ip;
                    base = frame.base;
                }

                Op::ADD => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = binary_arith(va, vb, |x, y| x + y, |x, y| x + y, "+")?;
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::SUB => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = binary_arith(va, vb, |x, y| x - y, |x, y| x - y, "-")?;
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::MUL => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = binary_arith(va, vb, |x, y| x * y, |x, y| x * y, "*")?;
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::DIV => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
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
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::MOD => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = if let (Some(x), Some(y)) = (va.as_int(), vb.as_int()) {
                        if y == 0 { return Err("Division by zero".to_string()); }
                        Value::int(x % y)
                    } else {
                        return Err("mod expects integers".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::NEG => {
                    let dest = instr.a();
                    let src = instr.b();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::int(-x)
                    } else if let Some(x) = v.as_float() {
                        Value::float(-x)
                    } else {
                        return Err("- expects a number".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::ADD_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::int(x + imm as i64)
                    } else if let Some(x) = v.as_float() {
                        Value::float(x + imm as f64)
                    } else {
                        return Err("+ expects numbers".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::SUB_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::int(x - imm as i64)
                    } else if let Some(x) = v.as_float() {
                        Value::float(x - imm as f64)
                    } else {
                        return Err("- expects numbers".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::LT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = Value::bool(compare_values(va, vb)? == std::cmp::Ordering::Less);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::LE => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = Value::bool(compare_values(va, vb)? != std::cmp::Ordering::Greater);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::GT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = Value::bool(compare_values(va, vb)? == std::cmp::Ordering::Greater);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::GE => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let result = Value::bool(compare_values(va, vb)? != std::cmp::Ordering::Less);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::EQ => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(va == vb) };
                }

                Op::NE => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(va != vb) };
                }

                Op::NOT => {
                    let dest = instr.a();
                    let src = instr.b();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(!v.is_truthy()) };
                }

                Op::NEW_LIST => {
                    let dest = instr.a();
                    let nargs = instr.b();
                    let items: Vec<Value> = (0..nargs)
                        .map(|i| unsafe { self.registers.get_unchecked(base + dest as usize + 1 + i as usize) }.clone())
                        .collect();
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::list(items) };
                }

                Op::GET_LIST => {
                    let dest = instr.a();
                    let list = instr.b();
                    let index = instr.c();
                    let list_val = unsafe { self.registers.get_unchecked(base + list as usize) };
                    let idx = unsafe { self.registers.get_unchecked(base + index as usize) };
                    let result = if let (Some(items), Some(i)) = (list_val.as_list(), idx.as_int()) {
                        items.get(i as usize).cloned().unwrap_or(Value::nil())
                    } else {
                        return Err("GetList expects list and int".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::SET_LIST => {
                    return Err("SetList not implemented (immutable lists)".to_string());
                }

                Op::CAR => {
                    let dest = instr.a();
                    let src = instr.b();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    // Fast path: cons cell (most common in list processing)
                    let result = if let Some(cons) = v.as_cons() {
                        cons.car.clone()
                    } else if let Some(list) = v.as_list() {
                        // Array list
                        list.first().cloned().ok_or_else(|| "car on empty list".to_string())?
                    } else {
                        return Err("car expects a list or cons cell".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::CDR => {
                    let dest = instr.a();
                    let src = instr.b();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    // Fast path: cons cell (most common in list processing)
                    let result = if let Some(cons) = v.as_cons() {
                        cons.cdr.clone()
                    } else if let Some(list) = v.as_list() {
                        // Array list - convert to cons chain for O(1) following CDRs
                        if list.is_empty() {
                            return Err("cdr on empty list".to_string());
                        }
                        Value::slice_to_cons(&list[1..])
                    } else {
                        return Err("cdr expects a list or cons cell".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::LT_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::bool(x < imm)
                    } else if let Some(x) = v.as_float() {
                        Value::bool(x < imm as f64)
                    } else {
                        return Err("< expects a number".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::LE_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::bool(x <= imm)
                    } else if let Some(x) = v.as_float() {
                        Value::bool(x <= imm as f64)
                    } else {
                        return Err("<= expects a number".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::GT_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::bool(x > imm)
                    } else if let Some(x) = v.as_float() {
                        Value::bool(x > imm as f64)
                    } else {
                        return Err("> expects a number".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::GE_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let result = if let Some(x) = v.as_int() {
                        Value::bool(x >= imm)
                    } else if let Some(x) = v.as_float() {
                        Value::bool(x >= imm as f64)
                    } else {
                        return Err(">= expects a number".to_string());
                    };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                // Combined compare-and-jump (register vs register)
                Op::JUMP_IF_LT => {
                    let left = instr.a();
                    let right = instr.b();
                    let offset = instr.c() as i8;
                    let va = unsafe { self.registers.get_unchecked(base + left as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + right as usize) };
                    if compare_values(va, vb)? == std::cmp::Ordering::Less {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_LE => {
                    let left = instr.a();
                    let right = instr.b();
                    let offset = instr.c() as i8;
                    let va = unsafe { self.registers.get_unchecked(base + left as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + right as usize) };
                    if compare_values(va, vb)? != std::cmp::Ordering::Greater {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GT => {
                    let left = instr.a();
                    let right = instr.b();
                    let offset = instr.c() as i8;
                    let va = unsafe { self.registers.get_unchecked(base + left as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + right as usize) };
                    if compare_values(va, vb)? == std::cmp::Ordering::Greater {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GE => {
                    let left = instr.a();
                    let right = instr.b();
                    let offset = instr.c() as i8;
                    let va = unsafe { self.registers.get_unchecked(base + left as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + right as usize) };
                    if compare_values(va, vb)? != std::cmp::Ordering::Less {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                // Combined compare-and-jump (register vs immediate)
                Op::JUMP_IF_LT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let should_jump = if let Some(x) = v.as_int() {
                        x < imm
                    } else if let Some(x) = v.as_float() {
                        x < imm as f64
                    } else {
                        return Err("< expects a number".to_string());
                    };
                    if should_jump {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_LE_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let should_jump = if let Some(x) = v.as_int() {
                        x <= imm
                    } else if let Some(x) = v.as_float() {
                        x <= imm as f64
                    } else {
                        return Err("<= expects a number".to_string());
                    };
                    if should_jump {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let should_jump = if let Some(x) = v.as_int() {
                        x > imm
                    } else if let Some(x) = v.as_float() {
                        x > imm as f64
                    } else {
                        return Err("> expects a number".to_string());
                    };
                    if should_jump {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GE_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let should_jump = if let Some(x) = v.as_int() {
                        x >= imm
                    } else if let Some(x) = v.as_float() {
                        x >= imm as f64
                    } else {
                        return Err(">= expects a number".to_string());
                    };
                    if should_jump {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                // Specialized nil check opcodes
                Op::JUMP_IF_NIL => {
                    let src = instr.a();
                    let offset = instr.sbx();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    if v.is_nil() {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_NOT_NIL => {
                    let src = instr.a();
                    let offset = instr.sbx();
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    if !v.is_nil() {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                // Specialized cons opcode
                Op::CONS => {
                    let dest = instr.a();
                    let car_reg = instr.b();
                    let cdr_reg = instr.c();
                    let car = unsafe { self.registers.get_unchecked(base + car_reg as usize).clone() };
                    let cdr = unsafe { self.registers.get_unchecked(base + cdr_reg as usize).clone() };
                    let cons = Value::cons(car, cdr);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = cons };
                }

                // ========== Move-semantics opcodes (liveness-optimized) ==========

                Op::MOVE_LAST => {
                    let dest = instr.a();
                    let src = instr.b();
                    // Take the value from src (leaving nil), avoiding Rc clone
                    let value = unsafe { self.registers.get_unchecked_mut(base + src as usize).take() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = value };
                }

                Op::CAR_LAST => {
                    let dest = instr.a();
                    let src = instr.b();
                    // Take the cons cell and extract car
                    let cell = unsafe { self.registers.get_unchecked_mut(base + src as usize).take() };
                    let result = cell.take_car()
                        .ok_or_else(|| "car expects a list or cons cell".to_string())?;
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::CDR_LAST => {
                    let dest = instr.a();
                    let src = instr.b();
                    // Take the cons cell and extract cdr
                    let cell = unsafe { self.registers.get_unchecked_mut(base + src as usize).take() };
                    let result = cell.take_cdr()
                        .ok_or_else(|| "cdr expects a list or cons cell".to_string())?;
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::CONS_MOVE => {
                    let dest = instr.a();
                    let car_with_flag = instr.b();
                    let cdr_with_flag = instr.c();

                    // Extract move flags from high bit
                    let move_car = (car_with_flag & 0x80) != 0;
                    let move_cdr = (cdr_with_flag & 0x80) != 0;
                    let car_reg = car_with_flag & 0x7F;
                    let cdr_reg = cdr_with_flag & 0x7F;

                    // Get car value (move or clone based on flag)
                    let car = if move_car {
                        unsafe { self.registers.get_unchecked_mut(base + car_reg as usize).take() }
                    } else {
                        unsafe { self.registers.get_unchecked(base + car_reg as usize).clone() }
                    };

                    // Get cdr value (move or clone based on flag)
                    let cdr = if move_cdr {
                        unsafe { self.registers.get_unchecked_mut(base + cdr_reg as usize).take() }
                    } else {
                        unsafe { self.registers.get_unchecked(base + cdr_reg as usize).clone() }
                    };

                    let cons = Value::cons(car, cdr);
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = cons };
                }

                // ========== Unboxed integer opcodes (skip type checking) ==========
                // These are emitted when the compiler can prove operands are integers.
                // They use as_int_unchecked which skips the is_int branch.

                Op::ADD_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    // SAFETY: Compiler guarantees both operands are integers
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    let result = Value::int(x.wrapping_add(y));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::SUB_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    let result = Value::int(x.wrapping_sub(y));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::MUL_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    let result = Value::int(x.wrapping_mul(y));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::ADD_INT_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    let result = Value::int(x.wrapping_add(imm));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::SUB_INT_IMM => {
                    let dest = instr.a();
                    let src = instr.b();
                    let imm = instr.c() as i8 as i64;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    let result = Value::int(x.wrapping_sub(imm));
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = result };
                }

                Op::LT_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(x < y) };
                }

                Op::LE_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(x <= y) };
                }

                Op::GT_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(x > y) };
                }

                Op::GE_INT => {
                    let dest = instr.a();
                    let a = instr.b();
                    let b = instr.c();
                    let va = unsafe { self.registers.get_unchecked(base + a as usize) };
                    let vb = unsafe { self.registers.get_unchecked(base + b as usize) };
                    let x = unsafe { va.as_int_unchecked() };
                    let y = unsafe { vb.as_int_unchecked() };
                    unsafe { *self.registers.get_unchecked_mut(base + dest as usize) = Value::bool(x >= y) };
                }

                // Combined compare-and-jump for integers (most common loop pattern)
                Op::JUMP_IF_LE_INT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    if x <= imm {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GT_INT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    if x > imm {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_LT_INT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    if x < imm {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                Op::JUMP_IF_GE_INT_IMM => {
                    let src = instr.a();
                    let imm = instr.b() as i8 as i64;
                    let offset = instr.c() as i8;
                    let v = unsafe { self.registers.get_unchecked(base + src as usize) };
                    let x = unsafe { v.as_int_unchecked() };
                    if x >= imm {
                        ip = (ip as isize + offset as isize) as usize;
                    }
                }

                _ => {
                    return Err(format!("Unknown opcode: {}", instr.opcode()));
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

/// Create a VM
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
        // Handle array lists - convert to cons chain for O(1) subsequent CDRs
        if let Some(list) = args[0].as_list() {
            if list.is_empty() { return Err("cdr on empty list".to_string()); }
            return Ok(Value::slice_to_cons(&list[1..]));
        }
        Err("cdr expects a list or cons cell".to_string())
    }));

    vm.define_global("length", native("length", |args| {
        if args.len() != 1 { return Err("length expects 1 argument".to_string()); }
        // Handle array lists
        if let Some(items) = args[0].as_list() {
            return Ok(Value::int(items.len() as i64));
        }
        // Handle cons cells (traverse and count)
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

    // Timing functions
    vm.define_global("clock", native("clock", |_args| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs_f64();
        Ok(Value::float(secs))
    }));

    // Additional list functions
    vm.define_global("nth", native("nth", |args| {
        if args.len() != 2 { return Err("nth expects 2 arguments".to_string()); }

        let index = args[1].as_int().ok_or_else(|| "nth expects integer index".to_string())?;
        if index < 0 { return Err("nth expects non-negative index".to_string()); }
        let idx = index as usize;

        // Handle array lists
        if let Some(items) = args[0].as_list() {
            return items.get(idx).cloned().ok_or_else(|| format!("nth: index {} out of bounds", index));
        }

        // Handle cons cells (traverse the chain)
        if args[0].as_cons().is_some() {
            let mut current = &args[0];
            let mut i = 0usize;
            while let Some(cons) = current.as_cons() {
                if i == idx {
                    return Ok(cons.car.clone());
                }
                i += 1;
                current = &cons.cdr;
            }
            // If we ended on an array list, check there
            if let Some(items) = current.as_list() {
                if idx >= i {
                    let remaining_idx = idx - i;
                    return items.get(remaining_idx).cloned().ok_or_else(|| format!("nth: index {} out of bounds", index));
                }
            }
            return Err(format!("nth: index {} out of bounds", index));
        }

        Err("nth expects a list or cons cell".to_string())
    }));

    vm.define_global("append", native("append", |args| {
        let mut result = Vec::new();
        for arg in args {
            if arg.is_nil() {
                continue;
            }
            if let Some(items) = arg.as_list() {
                result.extend_from_slice(items);
            } else if arg.as_cons().is_some() {
                // Convert cons chain to vec
                let mut current = arg;
                while let Some(cons) = current.as_cons() {
                    result.push(cons.car.clone());
                    current = &cons.cdr;
                }
                // If ended on array list, append it
                if let Some(items) = current.as_list() {
                    result.extend_from_slice(items);
                }
            } else {
                return Err("append expects lists or cons cells".to_string());
            }
        }
        Ok(Value::list(result))
    }));

    vm.define_global("reverse", native("reverse", |args| {
        if args.len() != 1 { return Err("reverse expects 1 argument".to_string()); }

        let mut result = Vec::new();

        if let Some(items) = args[0].as_list() {
            result.extend_from_slice(items);
        } else if args[0].as_cons().is_some() {
            // Convert cons chain to vec
            let mut current = &args[0];
            while let Some(cons) = current.as_cons() {
                result.push(cons.car.clone());
                current = &cons.cdr;
            }
            // If ended on array list, append it
            if let Some(items) = current.as_list() {
                result.extend_from_slice(items);
            }
        } else if args[0].is_nil() {
            return Ok(Value::nil());
        } else {
            return Err("reverse expects a list or cons cell".to_string());
        }

        result.reverse();
        Ok(Value::list(result))
    }));

    // String functions
    vm.define_global("string-length", native("string-length", |args| {
        if args.len() != 1 { return Err("string-length expects 1 argument".to_string()); }
        if let Some(s) = args[0].as_string() {
            Ok(Value::int(s.len() as i64))
        } else {
            Err("string-length expects a string".to_string())
        }
    }));

    vm.define_global("string-append", native("string-append", |args| {
        let mut result = String::new();
        for arg in args {
            if let Some(s) = arg.as_string() {
                result.push_str(s);
            } else {
                return Err("string-append expects strings".to_string());
            }
        }
        Ok(Value::string(&result))
    }));

    vm.define_global("substring", native("substring", |args| {
        if args.len() != 3 { return Err("substring expects 3 arguments (string, start, end)".to_string()); }

        let s = args[0].as_string().ok_or_else(|| "substring expects a string as first argument".to_string())?;
        let start = args[1].as_int().ok_or_else(|| "substring expects integer start".to_string())?;
        let end = args[2].as_int().ok_or_else(|| "substring expects integer end".to_string())?;

        if start < 0 || end < 0 {
            return Err("substring expects non-negative indices".to_string());
        }

        let start = start as usize;
        let end = end as usize;

        if start > s.len() || end > s.len() || start > end {
            return Err(format!("substring: invalid range [{}, {}] for string of length {}", start, end, s.len()));
        }

        Ok(Value::string(&s[start..end]))
    }));

    vm.define_global("string->list", native("string->list", |args| {
        if args.len() != 1 { return Err("string->list expects 1 argument".to_string()); }

        let s = args[0].as_string().ok_or_else(|| "string->list expects a string".to_string())?;
        let chars: Vec<Value> = s.chars().map(|c| {
            let ch_str = c.to_string();
            Value::string(&ch_str)
        }).collect();
        Ok(Value::list(chars))
    }));

    vm.define_global("list->string", native("list->string", |args| {
        if args.len() != 1 { return Err("list->string expects 1 argument".to_string()); }

        let mut result = String::new();

        // Handle array lists
        if let Some(items) = args[0].as_list() {
            for item in items.iter() {
                if let Some(s) = item.as_string() {
                    result.push_str(s);
                } else {
                    return Err("list->string expects a list of strings".to_string());
                }
            }
            return Ok(Value::string(&result));
        }

        // Handle cons cells
        if args[0].as_cons().is_some() {
            let mut current = &args[0];
            while let Some(cons) = current.as_cons() {
                if let Some(s) = cons.car.as_string() {
                    result.push_str(s);
                } else {
                    return Err("list->string expects a list of strings".to_string());
                }
                current = &cons.cdr;
            }
            if let Some(items) = current.as_list() {
                for item in items.iter() {
                    if let Some(s) = item.as_string() {
                        result.push_str(s);
                    } else {
                        return Err("list->string expects a list of strings".to_string());
                    }
                }
            }
            return Ok(Value::string(&result));
        }

        if args[0].is_nil() {
            return Ok(Value::string(""));
        }

        Err("list->string expects a list or cons cell".to_string())
    }));

    vm.define_global("format", native("format", |args| {
        if args.is_empty() { return Err("format expects at least 1 argument".to_string()); }

        let fmt = args[0].as_string().ok_or_else(|| "format expects a string as first argument".to_string())?;
        let mut result = String::new();
        let mut arg_idx = 1;
        let mut chars = fmt.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '~' {
                if let Some(&next) = chars.peek() {
                    chars.next(); // consume the format specifier
                    match next {
                        'a' | 'A' => {
                            // Display value
                            if arg_idx < args.len() {
                                if let Some(s) = args[arg_idx].as_string() {
                                    result.push_str(s);
                                } else {
                                    result.push_str(&format!("{}", args[arg_idx]));
                                }
                                arg_idx += 1;
                            } else {
                                return Err("format: not enough arguments".to_string());
                            }
                        }
                        '~' => result.push('~'),
                        _ => {
                            result.push('~');
                            result.push(next);
                        }
                    }
                } else {
                    result.push('~');
                }
            } else {
                result.push(ch);
            }
        }

        Ok(Value::string(&result))
    }));

    // I/O functions
    vm.define_global("read-line", native("read-line", |_args| {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line)
            .map_err(|e| format!("read-line error: {}", e))?;
        // Remove trailing newline if present
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        Ok(Value::string(&line))
    }));

    vm.define_global("read-file", native("read-file", |args| {
        if args.len() != 1 { return Err("read-file expects 1 argument".to_string()); }

        let path = args[0].as_string().ok_or_else(|| "read-file expects a string path".to_string())?;
        std::fs::read_to_string(path)
            .map(|s| Value::string(&s))
            .map_err(|e| format!("read-file error: {}", e))
    }));

    vm.define_global("write-file", native("write-file", |args| {
        if args.len() != 2 { return Err("write-file expects 2 arguments (path, content)".to_string()); }

        let path = args[0].as_string().ok_or_else(|| "write-file expects a string path".to_string())?;
        let content = args[1].as_string().ok_or_else(|| "write-file expects a string content".to_string())?;

        std::fs::write(path, content)
            .map(|_| Value::nil())
            .map_err(|e| format!("write-file error: {}", e))
    }));

    vm.define_global("spawn", native("spawn", |args| {
        use std::sync::{Arc, Mutex};
        use std::thread;

        if args.len() != 1 {
            return Err("spawn expects 1 argument (function)".to_string());
        }

        // Get the function to spawn
        let func = &args[0];

        // Only compiled functions can be spawned (not closures with captured environments)
        let chunk = if let Some(chunk) = func.as_compiled_function() {
            chunk.clone()
        } else {
            return Err("spawn expects a compiled function (not a closure)".to_string());
        };

        // Convert Rc<Chunk> to Arc<Chunk> for thread safety
        let arc_chunk = Arc::new((*chunk).clone());

        // Spawn the thread
        let handle = thread::spawn(move || {
            // Create a new VM for this thread
            let mut thread_vm = standard_vm();

            // Execute the function (it should be zero-argument)
            match thread_vm.run((*arc_chunk).clone()) {
                Ok(result) => {
                    // Convert result to SharedValue
                    result.make_shared()
                }
                Err(e) => Err(format!("Thread execution error: {}", e)),
            }
        });

        // Wrap the JoinHandle in Arc<Mutex<Option<>>> so it can be joined once
        let thread_handle = Arc::new(Mutex::new(Some(handle)));

        // Create a HeapObject::ThreadHandle and wrap it in a Value
        let heap = Rc::new(crate::value::HeapObject::ThreadHandle(thread_handle));
        Ok(Value::from_heap(heap))
    }));

    vm.define_global("join", native("join", |args| {
        use crate::value::HeapObject;

        if args.len() != 1 {
            return Err("join expects 1 argument (thread-handle)".to_string());
        }

        // Get the thread handle
        let handle_obj = args[0].as_heap()
            .ok_or_else(|| "join expects a thread-handle".to_string())?;

        if let HeapObject::ThreadHandle(handle_mutex) = handle_obj {
            // Take the handle from the mutex (can only join once)
            let handle_opt = handle_mutex.lock()
                .map_err(|_| "Failed to lock thread handle".to_string())?
                .take();

            let handle = handle_opt
                .ok_or_else(|| "Thread already joined".to_string())?;

            // Wait for the thread to complete
            let result = handle.join()
                .map_err(|_| "Thread panicked".to_string())?;

            // Convert the result back from Result<SharedValue, String> to Value
            match result {
                Ok(shared_value) => Ok(Value::from_shared(&shared_value)),
                Err(e) => Err(format!("Thread error: {}", e)),
            }
        } else {
            Err("join expects a thread-handle".to_string())
        }
    }));

    vm.define_global("channel", native("channel", |args| {
        use std::sync::mpsc;
        use std::sync::{Arc, Mutex};
        use crate::value::HeapObject;

        if !args.is_empty() {
            return Err("channel expects 0 arguments".to_string());
        }

        // Create an unbounded channel
        let (sender, receiver) = mpsc::channel();

        // Wrap sender and receiver in Arc<Mutex<>> for thread-safety
        let sender_arc = Arc::new(Mutex::new(sender));
        let receiver_arc = Arc::new(Mutex::new(receiver));

        // Create HeapObject values
        let sender_heap = Rc::new(HeapObject::ChannelSender(sender_arc));
        let receiver_heap = Rc::new(HeapObject::ChannelReceiver(receiver_arc));

        // Return a list [sender receiver]
        Ok(Value::list(vec![
            Value::from_heap(sender_heap),
            Value::from_heap(receiver_heap),
        ]))
    }));

    vm.define_global("send!", native("send!", |args| {
        use crate::value::HeapObject;

        if args.len() != 2 {
            return Err("send! expects 2 arguments (sender value)".to_string());
        }

        // Get the sender
        let sender_obj = args[0].as_heap()
            .ok_or_else(|| "send! expects a channel-sender as first argument".to_string())?;

        if let HeapObject::ChannelSender(sender_mutex) = sender_obj {
            // Convert the value to SharedValue for thread-safe sending
            let shared_value = args[1].make_shared()
                .map_err(|e| format!("send! failed to convert value: {}", e))?;

            // Send the value
            let sender = sender_mutex.lock()
                .map_err(|_| "Failed to lock channel sender".to_string())?;

            sender.send(shared_value)
                .map_err(|_| "send! failed: receiver has been dropped".to_string())?;

            // Return nil
            Ok(Value::nil())
        } else {
            Err("send! expects a channel-sender as first argument".to_string())
        }
    }));

    vm.define_global("recv", native("recv", |args| {
        use crate::value::HeapObject;

        if args.len() != 1 {
            return Err("recv expects 1 argument (receiver)".to_string());
        }

        // Get the receiver
        let receiver_obj = args[0].as_heap()
            .ok_or_else(|| "recv expects a channel-receiver".to_string())?;

        if let HeapObject::ChannelReceiver(receiver_mutex) = receiver_obj {
            // Receive a value (blocking)
            let receiver = receiver_mutex.lock()
                .map_err(|_| "Failed to lock channel receiver".to_string())?;

            let shared_value = receiver.recv()
                .map_err(|_| "recv failed: sender has been dropped".to_string())?;

            // Convert back to Rc-based Value
            Ok(Value::from_shared(&shared_value))
        } else {
            Err("recv expects a channel-receiver".to_string())
        }
    }));

    vm
}

#[inline(always)]
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
    // Fast path: integer arithmetic
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

#[inline(always)]
fn compare_values(a: &Value, b: &Value) -> Result<std::cmp::Ordering, String> {
    // Fast path: integer comparison
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
        chunk.emit(Op::load_const(0, idx));
        chunk.emit(Op::ret(0));
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
        chunk.emit(Op::load_const(0, idx1));
        chunk.emit(Op::load_const(1, idx2));
        chunk.emit(Op::add(2, 0, 1));
        chunk.emit(Op::ret(2));
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
        chunk.emit(Op::load_const(0, idx1));
        chunk.emit(Op::load_const(1, idx2));
        chunk.emit(Op::lt(2, 0, 1));
        chunk.emit(Op::ret(2));
        chunk.num_registers = 16;

        let mut vm = VM::new();
        let result = vm.run(chunk).unwrap();
        assert_eq!(result, Value::bool(true));
    }

    #[test]
    fn test_vm_jump() {
        let mut chunk = Chunk::new();
        chunk.emit(Op::load_true(0));
        let jump_pos = chunk.emit(Op::jump_if_false(0, 0));
        let idx = chunk.add_constant(Value::int(1));
        chunk.emit(Op::load_const(1, idx));
        let jump_over = chunk.emit(Op::jump(0));
        let else_pos = chunk.current_pos();
        let idx2 = chunk.add_constant(Value::int(2));
        chunk.emit(Op::load_const(1, idx2));
        let end = chunk.current_pos();
        chunk.patch_jump(jump_pos, else_pos);
        chunk.patch_jump(jump_over, end);
        chunk.emit(Op::ret(1));
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
        chunk.emit(Op::load_const(0, val_idx));
        chunk.emit(Op::set_global(name_idx, 0));
        chunk.emit(Op::get_global(1, name_idx));
        chunk.emit(Op::ret(1));
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
        // Should not stack overflow
        let result = vm_eval("(do (def sum (fn (n acc) (if (<= n 0) acc (sum (- n 1) (+ acc n))))) (sum 10000 0))").unwrap();
        assert_eq!(result, Value::Int(50005000));
    }

    #[test]
    fn test_mutual_recursion_tail_call() {
        // Mutual recursion with tail calls - should not stack overflow
        let result = vm_eval("(do
            (def is-even (fn (n) (if (= n 0) true (is-odd (- n 1)))))
            (def is-odd (fn (n) (if (= n 0) false (is-even (- n 1)))))
            (is-even 100))").unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_specialized_car_cdr() {
        // Test car/cdr opcodes with cons cells
        assert_eq!(vm_eval("(car (cons 1 (cons 2 nil)))").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(car (cdr (cons 1 (cons 2 nil))))").unwrap(), Value::Int(2));

        // Test with nested list operations
        let result = vm_eval("(do (def reverse-acc (fn (lst acc) (if (empty? lst) acc (reverse-acc (cdr lst) (cons (car lst) acc))))) (reverse-acc (cons 1 (cons 2 (cons 3 nil))) nil))").unwrap();
        // reversed (1 2 3) -> (3 2 1)
        let cons = result.as_cons().expect("expected cons cell");
        assert_eq!(cons.car.as_int(), Some(3));
    }

    #[test]
    fn test_comparison_immediate() {
        // Test comparison with immediate values
        assert_eq!(vm_eval("(< 5 10)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(< 10 5)").unwrap(), Value::Bool(false));
        assert_eq!(vm_eval("(<= 5 5)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(<= 6 5)").unwrap(), Value::Bool(false));
        assert_eq!(vm_eval("(> 10 5)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(> 5 10)").unwrap(), Value::Bool(false));
        assert_eq!(vm_eval("(>= 5 5)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(>= 4 5)").unwrap(), Value::Bool(false));

        // Test with negative immediates
        assert_eq!(vm_eval("(< -5 0)").unwrap(), Value::Bool(true));
        assert_eq!(vm_eval("(<= 0 0)").unwrap(), Value::Bool(true));

        // Test in a loop (sum function uses (<= n 0))
        let result = vm_eval("(do (def count-down (fn (n) (if (<= n 0) 0 (+ 1 (count-down (- n 1)))))) (count-down 10))").unwrap();
        assert_eq!(result, Value::Int(10));
    }

    #[test]
    fn test_combined_compare_jump() {
        // Test combined compare-and-jump opcodes
        // These are generated for (if (< a b) ...) patterns

        // Immediate variants
        assert_eq!(vm_eval("(if (< 5 10) 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if (< 15 10) 1 2)").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(if (<= 10 10) 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if (<= 11 10) 1 2)").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(if (> 15 10) 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if (> 5 10) 1 2)").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(if (>= 10 10) 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if (>= 9 10) 1 2)").unwrap(), Value::Int(2));

        // Variable vs immediate (uses JumpIfXxxImm)
        assert_eq!(vm_eval("(let (x 5) (if (< x 10) 1 2))").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(let (x 15) (if (< x 10) 1 2))").unwrap(), Value::Int(2));
        assert_eq!(vm_eval("(let (n 0) (if (<= n 0) 1 2))").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(let (n 1) (if (<= n 0) 1 2))").unwrap(), Value::Int(2));

        // Variable vs variable (uses JumpIfXxx)
        assert_eq!(vm_eval("(let (a 5 b 10) (if (< a b) 1 2))").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(let (a 15 b 10) (if (< a b) 1 2))").unwrap(), Value::Int(2));

        // Recursive function using combined compare-jump
        let result = vm_eval("(do (def sum (fn (n acc) (if (<= n 0) acc (sum (- n 1) (+ acc n))))) (sum 100 0))").unwrap();
        assert_eq!(result, Value::Int(5050));
    }

    #[test]
    fn test_nil_check_jump() {
        // Test nil check opcodes
        // These are generated for (if (nil? x) ...) patterns

        // Direct nil check
        assert_eq!(vm_eval("(if (nil? nil) 1 2)").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(if (nil? 42) 1 2)").unwrap(), Value::Int(2));

        // Variable nil check
        assert_eq!(vm_eval("(let (x nil) (if (nil? x) 1 2))").unwrap(), Value::Int(1));
        assert_eq!(vm_eval("(let (x 42) (if (nil? x) 1 2))").unwrap(), Value::Int(2));

        // Recursive list processing using nil check (with cons-based lists)
        let result = vm_eval("(do
            (def list-length (fn (lst)
                (if (nil? lst)
                    0
                    (+ 1 (list-length (cdr lst))))))
            (list-length (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil)))))))").unwrap();
        assert_eq!(result, Value::Int(5));

        // List sum using nil? (with cons-based lists)
        let result = vm_eval("(do
            (def list-sum (fn (lst acc)
                (if (nil? lst)
                    acc
                    (list-sum (cdr lst) (+ acc (car lst))))))
            (list-sum (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil))))) 0))").unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_move_semantics_basic() {
        // Test that move semantics work correctly for simple cases
        // (fn (x) x) - x is used exactly once, should use MoveLast
        let result = vm_eval("(do (def id (fn (x) x)) (id 42))").unwrap();
        assert_eq!(result, Value::Int(42));

        // (fn (x) (+ x x)) - x is used twice, first should be Move, second MoveLast
        let result = vm_eval("(do (def double (fn (x) (+ x x))) (double 21))").unwrap();
        assert_eq!(result, Value::Int(42));

        // Test with cons cells - move semantics for car/cdr
        let result = vm_eval("(do (def get-car (fn (lst) (car lst))) (get-car (cons 42 nil)))").unwrap();
        assert_eq!(result, Value::Int(42));

        let result = vm_eval("(do (def get-cdr (fn (lst) (cdr lst))) (get-cdr (cons 1 (cons 2 nil))))").unwrap();
        let cons = result.as_cons().expect("expected cons");
        assert_eq!(cons.car.as_int(), Some(2));
    }

    #[test]
    fn test_move_semantics_cons() {
        // Test CONS_MOVE optimization
        // When building lists, last-use values should be moved into cons cells
        let result = vm_eval("(do
            (def build-list (fn (a b c)
                (cons a (cons b (cons c nil)))))
            (car (build-list 1 2 3)))").unwrap();
        assert_eq!(result, Value::Int(1));

        // Verify the full list structure
        let result = vm_eval("(do
            (def build-list (fn (a b c)
                (cons a (cons b (cons c nil)))))
            (build-list 1 2 3))").unwrap();
        let cons1 = result.as_cons().expect("expected cons");
        assert_eq!(cons1.car.as_int(), Some(1));
        let cons2 = cons1.cdr.as_cons().expect("expected cons");
        assert_eq!(cons2.car.as_int(), Some(2));
        let cons3 = cons2.cdr.as_cons().expect("expected cons");
        assert_eq!(cons3.car.as_int(), Some(3));
    }

    #[test]
    fn test_move_semantics_recursion() {
        // Test move semantics in recursive functions
        // List reversal - heavily uses car/cdr/cons
        let result = vm_eval("(do
            (def reverse-acc (fn (lst acc)
                (if (nil? lst)
                    acc
                    (reverse-acc (cdr lst) (cons (car lst) acc)))))
            (def reverse (fn (lst) (reverse-acc lst nil)))
            (car (reverse (cons 1 (cons 2 (cons 3 nil))))))").unwrap();
        assert_eq!(result, Value::Int(3));

        // List sum using car/cdr
        let result = vm_eval("(do
            (def sum (fn (lst)
                (if (nil? lst)
                    0
                    (+ (car lst) (sum (cdr lst))))))
            (sum (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil)))))))").unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_specialized_cons() {
        // Test cons opcode
        // Basic cons
        let result = vm_eval("(cons 1 nil)").unwrap();
        let cons = result.as_cons().expect("expected cons cell");
        assert_eq!(cons.car.as_int(), Some(1));
        assert!(cons.cdr.is_nil());

        // Nested cons
        let result = vm_eval("(cons 1 (cons 2 (cons 3 nil)))").unwrap();
        let cons = result.as_cons().expect("expected cons cell");
        assert_eq!(cons.car.as_int(), Some(1));

        // List reversal using cons (should use the opcode a lot)
        let result = vm_eval("(do
            (def reverse-acc (fn (lst acc)
                (if (nil? lst)
                    acc
                    (reverse-acc (cdr lst) (cons (car lst) acc)))))
            (reverse-acc (cons 1 (cons 2 (cons 3 nil))) nil))").unwrap();
        let cons = result.as_cons().expect("expected cons cell");
        assert_eq!(cons.car.as_int(), Some(3));

        // Map function using cons
        let result = vm_eval("(do
            (def map-double (fn (lst)
                (if (nil? lst)
                    nil
                    (cons (* 2 (car lst)) (map-double (cdr lst))))))
            (car (map-double (cons 5 (cons 10 nil)))))").unwrap();
        assert_eq!(result, Value::Int(10)); // first element 5 * 2 = 10
    }

    #[test]
    fn test_integer_specialized_opcodes() {
        // Test that integer-specialized opcodes produces correct results
        // These are used in loop-optimized functions where types are known

        // Test sum function - should use ADD_INT_IMM, SUB_INT_IMM, ADD_INT
        let result = vm_eval("(do
            (def sum (fn (n acc)
                (if (<= n 0) acc (sum (- n 1) (+ acc n)))))
            (sum 100 0))").unwrap();
        assert_eq!(result, Value::Int(5050));

        // Test with larger numbers to verify integer arithmetic
        let result = vm_eval("(do
            (def sum (fn (n acc)
                (if (<= n 0) acc (sum (- n 1) (+ acc n)))))
            (sum 1000 0))").unwrap();
        assert_eq!(result, Value::Int(500500));

        // Test multiplication in loops - should use MUL_INT
        let result = vm_eval("(do
            (def factorial (fn (n acc)
                (if (<= n 1) acc (factorial (- n 1) (* acc n)))))
            (factorial 10 1))").unwrap();
        assert_eq!(result, Value::Int(3628800));

        // Test integer comparisons with both operands as integers
        let result = vm_eval("(do
            (def count-down (fn (n acc)
                (if (< n 1) acc (count-down (- n 1) (+ acc 1)))))
            (count-down 50 0))").unwrap();
        assert_eq!(result, Value::Int(50));
    }
}
