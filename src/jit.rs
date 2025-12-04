//! JIT Compiler using Cranelift
//!
//! Phase 1: Basic JIT Infrastructure
//! - Profiling counters to identify hot functions
//! - Cranelift integration for code generation
//! - Simple function compilation (no optimization)

use crate::bytecode::{Chunk, Op};
use crate::value::Value;
use std::collections::HashMap;
use std::rc::Rc;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

/// Threshold for triggering JIT compilation
const JIT_THRESHOLD: u32 = 100;

/// Type profile for a value position
#[derive(Debug, Clone, Default)]
pub struct TypeProfile {
    pub int_count: u32,
    pub float_count: u32,
    pub other_count: u32,
}

impl TypeProfile {
    /// Check if this position is mostly integers (>90%)
    pub fn is_mostly_int(&self) -> bool {
        let total = self.int_count + self.float_count + self.other_count;
        if total == 0 {
            return false;
        }
        self.int_count > (total * 9 / 10)
    }
}

/// Profiling data for a function
#[derive(Debug, Clone)]
pub struct FunctionProfile {
    /// Number of times this function has been called
    pub call_count: u32,
    /// Type profiles for parameters
    pub param_types: Vec<TypeProfile>,
    /// Whether this function has been JIT compiled
    pub is_compiled: bool,
}

impl FunctionProfile {
    pub fn new(num_params: usize) -> Self {
        FunctionProfile {
            call_count: 0,
            param_types: vec![TypeProfile::default(); num_params],
            is_compiled: false,
        }
    }

    /// Record a call with given argument types
    pub fn record_call(&mut self, args: &[Value]) {
        self.call_count += 1;
        for (i, arg) in args.iter().enumerate() {
            if i < self.param_types.len() {
                if arg.is_int() {
                    self.param_types[i].int_count += 1;
                } else if arg.is_float() {
                    self.param_types[i].float_count += 1;
                } else {
                    self.param_types[i].other_count += 1;
                }
            }
        }
    }

    /// Check if this function should be JIT compiled
    pub fn should_compile(&self) -> bool {
        !self.is_compiled && self.call_count >= JIT_THRESHOLD
    }
}

/// A wrapper for an interned symbol Rc<str> that hashes/compares by pointer
#[derive(Clone)]
pub struct SymbolKey(pub Rc<str>);

impl std::hash::Hash for SymbolKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as *const u8 as usize).hash(state);
    }
}

impl PartialEq for SymbolKey {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SymbolKey {}

/// JIT-compiled native code
pub struct JitCode {
    /// The function pointer to the compiled code
    func_ptr: *const u8,
    /// Function ID in the JIT module (for cleanup)
    #[allow(dead_code)]
    func_id: FuncId,
}

impl JitCode {
    /// Execute the JIT-compiled code with the given registers
    /// Returns the result value
    ///
    /// # Safety
    /// The caller must ensure:
    /// - registers slice has at least num_registers elements
    /// - The function signature matches what was compiled
    pub unsafe fn execute(&self, registers: &mut [Value]) -> Result<Value, String> {
        // The JIT function signature is:
        // fn(registers: *mut Value) -> u64
        // Returns the NaN-boxed result value
        let func: unsafe extern "C" fn(*mut Value) -> u64 =
            std::mem::transmute(self.func_ptr);

        let result_bits = func(registers.as_mut_ptr());
        Ok(Value::from_bits(result_bits))
    }
}

/// The JIT compiler
pub struct JitCompiler {
    /// Cranelift JIT module
    module: JITModule,
    /// Builder context for function definitions
    builder_context: FunctionBuilderContext,
    /// Cranelift context for function compilation
    ctx: codegen::Context,
    /// Compiled functions indexed by function chunk pointer
    compiled_functions: HashMap<usize, JitCode>,
    /// Profiling data for functions
    profiles: HashMap<SymbolKey, FunctionProfile>,
}

impl JitCompiler {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        // Set up for the host machine
        flag_builder.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;
        flag_builder.set("is_pic", "false").map_err(|e| e.to_string())?;
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(JitCompiler {
            module,
            builder_context: FunctionBuilderContext::new(),
            ctx: codegen::Context::new(),
            compiled_functions: HashMap::new(),
            profiles: HashMap::new(),
        })
    }

    /// Get or create a profile for a function
    pub fn get_or_create_profile(&mut self, key: SymbolKey, num_params: usize) -> &mut FunctionProfile {
        self.profiles.entry(key).or_insert_with(|| FunctionProfile::new(num_params))
    }

    /// Record a function call for profiling
    pub fn record_call(&mut self, key: SymbolKey, num_params: usize, args: &[Value]) -> bool {
        let profile = self.get_or_create_profile(key, num_params);
        profile.record_call(args);
        profile.should_compile()
    }

    /// Check if a function is already compiled
    pub fn is_compiled(&self, chunk_ptr: usize) -> bool {
        self.compiled_functions.contains_key(&chunk_ptr)
    }

    /// Get a compiled function by chunk pointer
    pub fn get_compiled(&self, chunk_ptr: usize) -> Option<&JitCode> {
        self.compiled_functions.get(&chunk_ptr)
    }

    /// Compile a function to native code
    /// This is Phase 1: simple translation without optimizations
    pub fn compile_function(&mut self, chunk: &Chunk, name: &str) -> Result<usize, String> {
        let chunk_ptr = chunk as *const Chunk as usize;

        if self.compiled_functions.contains_key(&chunk_ptr) {
            return Ok(chunk_ptr);
        }

        // Create function signature
        // fn(registers: *mut Value) -> u64
        let pointer_type = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(pointer_type)); // registers ptr
        sig.returns.push(AbiParam::new(types::I64)); // result as NaN-boxed u64

        // Declare the function
        let func_id = self.module
            .declare_function(name, Linkage::Local, &sig)
            .map_err(|e| e.to_string())?;

        // Clear the context for a new function
        self.ctx.clear();
        self.ctx.func.signature = sig;

        // Build the function - need to avoid borrowing self
        let ptr_type = pointer_type;
        self.build_function(chunk, ptr_type)?;

        // Finalize the function
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| e.to_string())?;

        self.module.clear_context(&mut self.ctx);

        // Finalize all functions and get the code
        self.module.finalize_definitions().map_err(|e| e.to_string())?;

        let func_ptr = self.module.get_finalized_function(func_id);

        self.compiled_functions.insert(chunk_ptr, JitCode {
            func_ptr,
            func_id,
        });

        Ok(chunk_ptr)
    }

    /// Build the function IR
    fn build_function(&mut self, chunk: &Chunk, pointer_type: Type) -> Result<(), String> {
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get the registers pointer parameter
        let registers_ptr = builder.block_params(entry_block)[0];

        // Compile the bytecode
        compile_bytecode(&mut builder, chunk, registers_ptr, pointer_type)?;

        builder.finalize();
        Ok(())
    }

    /// Get profiling statistics
    pub fn get_stats(&self) -> JitStats {
        let mut total_calls = 0u64;
        let mut compiled_count = 0usize;

        for profile in self.profiles.values() {
            total_calls += profile.call_count as u64;
            if profile.is_compiled {
                compiled_count += 1;
            }
        }

        JitStats {
            total_functions_profiled: self.profiles.len(),
            total_functions_compiled: compiled_count,
            total_call_count: total_calls,
            jit_threshold: JIT_THRESHOLD,
        }
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create JIT compiler")
    }
}

/// Compile bytecode to Cranelift IR (standalone function to avoid borrowing issues)
fn compile_bytecode(
    builder: &mut FunctionBuilder,
    chunk: &Chunk,
    registers_ptr: cranelift::prelude::Value,
    _pointer_type: Type,
) -> Result<(), String> {
    // Create blocks for each instruction that can be a jump target
    let mut blocks: HashMap<usize, Block> = HashMap::new();
    // Track blocks that are backward jump targets (loop headers)
    let mut loop_headers: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Track which blocks have been terminated (to avoid is_filled)
    let mut terminated = false;

    // First pass: find all jump targets
    for (i, op) in chunk.code.iter().enumerate() {
        let opcode = op.opcode();
        if opcode == Op::JUMP || opcode == Op::JUMP_IF_FALSE || opcode == Op::JUMP_IF_TRUE
            || opcode == Op::JUMP_IF_NIL || opcode == Op::JUMP_IF_NOT_NIL
            || (opcode >= Op::JUMP_IF_LT && opcode <= Op::JUMP_IF_GE_IMM)
            || opcode == Op::JUMP_IF_LE_INT_IMM || opcode == Op::JUMP_IF_GT_INT_IMM
            || opcode == Op::JUMP_IF_LT_INT_IMM || opcode == Op::JUMP_IF_GE_INT_IMM
        {
            let offset = if opcode == Op::JUMP || opcode == Op::JUMP_IF_FALSE
                || opcode == Op::JUMP_IF_TRUE || opcode == Op::JUMP_IF_NIL
                || opcode == Op::JUMP_IF_NOT_NIL
            {
                op.sbx() as isize
            } else {
                op.c() as i8 as isize
            };
            let target = (i as isize + 1 + offset) as usize;
            if !blocks.contains_key(&target) {
                blocks.insert(target, builder.create_block());
            }
            // Check if this is a backward jump (loop)
            if (offset as isize) < 0 {
                loop_headers.insert(target);
            }
            // Also need a block for the fallthrough
            let fallthrough = i + 1;
            if !blocks.contains_key(&fallthrough) {
                blocks.insert(fallthrough, builder.create_block());
            }
        }
    }

    // Use Cranelift's Variable system for constants - this handles SSA properly across blocks
    let var_tag_nil = Variable::new(0);
    let var_tag_int = Variable::new(1);
    let var_tag_true = Variable::new(2);
    let var_tag_false = Variable::new(3);
    let var_payload_mask = Variable::new(4);
    let var_registers_ptr = Variable::new(5);

    // Declare variables
    builder.declare_var(var_tag_nil, types::I64);
    builder.declare_var(var_tag_int, types::I64);
    builder.declare_var(var_tag_true, types::I64);
    builder.declare_var(var_tag_false, types::I64);
    builder.declare_var(var_payload_mask, types::I64);
    builder.declare_var(var_registers_ptr, types::I64);

    // Define constants in entry block
    let tag_nil_val = builder.ins().iconst(types::I64, 0x7FFC_0000_0000_0000u64 as i64);
    let tag_int_val = builder.ins().iconst(types::I64, 0x7FFD_0000_0000_0000u64 as i64);
    let tag_true_val = builder.ins().iconst(types::I64, 0x7FFC_0000_0000_0002u64 as i64);
    let tag_false_val = builder.ins().iconst(types::I64, 0x7FFC_0000_0000_0001u64 as i64);
    let payload_mask_val = builder.ins().iconst(types::I64, 0x0000_FFFF_FFFF_FFFFu64 as i64);

    builder.def_var(var_tag_nil, tag_nil_val);
    builder.def_var(var_tag_int, tag_int_val);
    builder.def_var(var_tag_true, tag_true_val);
    builder.def_var(var_tag_false, tag_false_val);
    builder.def_var(var_payload_mask, payload_mask_val);
    builder.def_var(var_registers_ptr, registers_ptr);

    // Compile each instruction
    for (ip, op) in chunk.code.iter().enumerate() {
        // If this instruction is a jump target, switch to its block
        if let Some(&block) = blocks.get(&ip) {
            // If current block is not terminated, jump to the new block
            if !terminated {
                builder.ins().jump(block, &[]);
            }
            builder.switch_to_block(block);
            // Only seal blocks that are not loop headers
            // Loop headers need to be sealed after all predecessors are known
            if !loop_headers.contains(&ip) {
                builder.seal_block(block);
            }
            terminated = false;
        }

        // Get current values of our variables (SSA handles phi nodes automatically)
        let registers_ptr = builder.use_var(var_registers_ptr);
        let tag_nil = builder.use_var(var_tag_nil);
        let tag_int = builder.use_var(var_tag_int);
        let tag_true = builder.use_var(var_tag_true);
        let tag_false = builder.use_var(var_tag_false);
        let payload_mask = builder.use_var(var_payload_mask);

        match op.opcode() {
            Op::LOAD_NIL => {
                let dest = op.a();
                store_reg(builder, dest, tag_nil, registers_ptr);
            }

            Op::LOAD_TRUE => {
                let dest = op.a();
                store_reg(builder, dest, tag_true, registers_ptr);
            }

            Op::LOAD_FALSE => {
                let dest = op.a();
                store_reg(builder, dest, tag_false, registers_ptr);
            }

            Op::LOAD_CONST => {
                let dest = op.a();
                let idx = op.bx() as usize;
                if idx < chunk.constants.len() {
                    let const_val = &chunk.constants[idx];
                    let bits = const_val.to_bits();
                    let value = builder.ins().iconst(types::I64, bits as i64);
                    store_reg(builder, dest, value, registers_ptr);
                }
            }

            Op::MOVE => {
                let dest = op.a();
                let src = op.b();
                let value = load_reg(builder, src, registers_ptr);
                store_reg(builder, dest, value, registers_ptr);
            }

            Op::MOVE_LAST => {
                // For now, same as MOVE (move semantics optimization is runtime)
                let dest = op.a();
                let src = op.b();
                let value = load_reg(builder, src, registers_ptr);
                store_reg(builder, dest, value, registers_ptr);
            }

            Op::ADD_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                // Extract integer payload (lower 48 bits)
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                // Add
                let sum = builder.ins().iadd(ia, ib);
                // Re-box as integer
                let sum_masked = builder.ins().band(sum, payload_mask);
                let result = builder.ins().bor(sum_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::SUB_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let diff = builder.ins().isub(ia, ib);
                let diff_masked = builder.ins().band(diff, payload_mask);
                let result = builder.ins().bor(diff_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::MUL_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let prod = builder.ins().imul(ia, ib);
                let prod_masked = builder.ins().band(prod, payload_mask);
                let result = builder.ins().bor(prod_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::ADD_INT_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let sum = builder.ins().iadd(i, imm_val);
                let sum_masked = builder.ins().band(sum, payload_mask);
                let result = builder.ins().bor(sum_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::SUB_INT_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let diff = builder.ins().isub(i, imm_val);
                let diff_masked = builder.ins().band(diff, payload_mask);
                let result = builder.ins().bor(diff_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            // Generic arithmetic - same as integer-specialized (assumes int operands)
            Op::ADD => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let sum = builder.ins().iadd(ia, ib);
                let sum_masked = builder.ins().band(sum, payload_mask);
                let result = builder.ins().bor(sum_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::SUB => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let diff = builder.ins().isub(ia, ib);
                let diff_masked = builder.ins().band(diff, payload_mask);
                let result = builder.ins().bor(diff_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::MUL => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let prod = builder.ins().imul(ia, ib);
                let prod_masked = builder.ins().band(prod, payload_mask);
                let result = builder.ins().bor(prod_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::ADD_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let sum = builder.ins().iadd(i, imm_val);
                let sum_masked = builder.ins().band(sum, payload_mask);
                let result = builder.ins().bor(sum_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::SUB_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let diff = builder.ins().isub(i, imm_val);
                let diff_masked = builder.ins().band(diff, payload_mask);
                let result = builder.ins().bor(diff_masked, tag_int);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::LT_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                // Sign-extend from 48 bits for comparison
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                // Use signed comparison (note: need proper sign extension for negative numbers)
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::LE_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GT_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GE_INT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            // Generic comparisons - same as integer-specialized (assumes int operands)
            Op::LT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::LE => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GT => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GE => {
                let dest = op.a();
                let a = op.b();
                let b = op.c();
                let va = load_reg(builder, a, registers_ptr);
                let vb = load_reg(builder, b, registers_ptr);
                let ia = builder.ins().band(va, payload_mask);
                let ib = builder.ins().band(vb, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, ia, ib);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::LT_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, imm_val);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::LE_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, i, imm_val);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GT_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, i, imm_val);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::GE_IMM => {
                let dest = op.a();
                let src = op.b();
                let imm = op.c() as i8 as i64;
                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, imm_val);
                let result = builder.ins().select(cmp, tag_true, tag_false);
                store_reg(builder, dest, result, registers_ptr);
            }

            Op::JUMP => {
                let offset = op.sbx() as isize;
                let target = (ip as isize + 1 + offset) as usize;
                if let Some(&target_block) = blocks.get(&target) {
                    builder.ins().jump(target_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_LE_INT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GT_INT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_LT_INT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GE_INT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            // Generic compare-jump opcodes (register vs register)
            Op::JUMP_IF_LT => {
                let left = op.a();
                let right = op.b();
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let vl = load_reg(builder, left, registers_ptr);
                let vr = load_reg(builder, right, registers_ptr);
                let il = builder.ins().band(vl, payload_mask);
                let ir = builder.ins().band(vr, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, il, ir);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_LE => {
                let left = op.a();
                let right = op.b();
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let vl = load_reg(builder, left, registers_ptr);
                let vr = load_reg(builder, right, registers_ptr);
                let il = builder.ins().band(vl, payload_mask);
                let ir = builder.ins().band(vr, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, il, ir);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GT => {
                let left = op.a();
                let right = op.b();
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let vl = load_reg(builder, left, registers_ptr);
                let vr = load_reg(builder, right, registers_ptr);
                let il = builder.ins().band(vl, payload_mask);
                let ir = builder.ins().band(vr, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, il, ir);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GE => {
                let left = op.a();
                let right = op.b();
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let vl = load_reg(builder, left, registers_ptr);
                let vr = load_reg(builder, right, registers_ptr);
                let il = builder.ins().band(vl, payload_mask);
                let ir = builder.ins().band(vr, payload_mask);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, il, ir);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            // Generic compare-jump with immediate
            Op::JUMP_IF_LT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_LE_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GT_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::JUMP_IF_GE_IMM => {
                let src = op.a();
                let imm = op.b() as i8 as i64;
                let offset = op.c() as i8 as isize;
                let target = (ip as isize + 1 + offset) as usize;
                let fallthrough = ip + 1;

                let v = load_reg(builder, src, registers_ptr);
                let i = builder.ins().band(v, payload_mask);
                let imm_val = builder.ins().iconst(types::I64, imm);
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, imm_val);

                if let (Some(&target_block), Some(&fallthrough_block)) =
                    (blocks.get(&target), blocks.get(&fallthrough))
                {
                    builder.ins().brif(cmp, target_block, &[], fallthrough_block, &[]);
                    terminated = true;
                }
            }

            Op::RETURN => {
                let src = op.a();
                let result = load_reg(builder, src, registers_ptr);
                builder.ins().return_(&[result]);
                terminated = true;
            }

            // For opcodes we don't yet support, we need to bail out
            // In a real JIT, this would trigger deoptimization
            _ => {
                // For unsupported opcodes, return nil and let interpreter handle it
                // This is a simple "give up" strategy for Phase 1
                return Err(format!("Unsupported opcode: {:?}", op));
            }
        }
    }

    // If we reach the end without a return, return nil
    if !terminated {
        let tag_nil = builder.use_var(var_tag_nil);
        builder.ins().return_(&[tag_nil]);
    }

    // Seal all remaining unsealed blocks (loop headers)
    for &ip in &loop_headers {
        if let Some(&block) = blocks.get(&ip) {
            builder.seal_block(block);
        }
    }

    Ok(())
}

/// Helper to load a register value (8 bytes = u64)
fn load_reg(builder: &mut FunctionBuilder, reg: u8, registers_ptr: cranelift::prelude::Value) -> cranelift::prelude::Value {
    let offset = reg as i32 * 8; // sizeof(Value) = 8
    builder.ins().load(types::I64, MemFlags::trusted(), registers_ptr, offset)
}

/// Helper to store a register value
fn store_reg(builder: &mut FunctionBuilder, reg: u8, value: cranelift::prelude::Value, registers_ptr: cranelift::prelude::Value) {
    let offset = reg as i32 * 8;
    builder.ins().store(MemFlags::trusted(), value, registers_ptr, offset);
}

/// Check if a chunk can be JIT compiled (all opcodes are supported)
pub fn is_jit_compatible(chunk: &Chunk) -> bool {
    for op in &chunk.code {
        match op.opcode() {
            // Constants and moves
            Op::LOAD_NIL | Op::LOAD_TRUE | Op::LOAD_FALSE | Op::LOAD_CONST |
            Op::MOVE | Op::MOVE_LAST |
            // Integer-specialized arithmetic
            Op::ADD_INT | Op::SUB_INT | Op::MUL_INT | Op::ADD_INT_IMM | Op::SUB_INT_IMM |
            // Generic arithmetic (assumes integer operands at runtime)
            Op::ADD | Op::SUB | Op::MUL | Op::ADD_IMM | Op::SUB_IMM |
            // Integer-specialized comparisons
            Op::LT_INT | Op::LE_INT | Op::GT_INT | Op::GE_INT |
            // Generic comparisons (assumes integer operands at runtime)
            Op::LT | Op::LE | Op::GT | Op::GE |
            Op::LT_IMM | Op::LE_IMM | Op::GT_IMM | Op::GE_IMM |
            // Control flow
            Op::JUMP |
            Op::JUMP_IF_LE_INT_IMM | Op::JUMP_IF_GT_INT_IMM |
            Op::JUMP_IF_LT_INT_IMM | Op::JUMP_IF_GE_INT_IMM |
            // Generic compare-jump (assumes integer operands at runtime)
            Op::JUMP_IF_LT | Op::JUMP_IF_LE | Op::JUMP_IF_GT | Op::JUMP_IF_GE |
            Op::JUMP_IF_LT_IMM | Op::JUMP_IF_LE_IMM | Op::JUMP_IF_GT_IMM | Op::JUMP_IF_GE_IMM |
            Op::RETURN => continue,
            _ => return false,
        }
    }
    true
}

/// JIT compilation statistics
#[derive(Debug)]
pub struct JitStats {
    pub total_functions_profiled: usize,
    pub total_functions_compiled: usize,
    pub total_call_count: u64,
    pub jit_threshold: u32,
}

impl std::fmt::Display for JitStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "JIT Stats: {} profiled, {} compiled, {} total calls (threshold: {})",
            self.total_functions_profiled,
            self.total_functions_compiled,
            self.total_call_count,
            self.jit_threshold
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let jit = JitCompiler::new();
        assert!(jit.is_ok(), "JIT compiler should be created successfully");
    }

    #[test]
    fn test_function_profile() {
        let mut profile = FunctionProfile::new(2);
        assert_eq!(profile.call_count, 0);
        assert!(!profile.should_compile());

        // Simulate calls with integer arguments
        for _ in 0..100 {
            profile.record_call(&[Value::int(42), Value::int(10)]);
        }

        assert_eq!(profile.call_count, 100);
        assert!(profile.should_compile());
        assert!(profile.param_types[0].is_mostly_int());
        assert!(profile.param_types[1].is_mostly_int());
    }

    #[test]
    fn test_type_profile() {
        let mut profile = TypeProfile::default();
        assert!(!profile.is_mostly_int());

        // Add mostly integers
        profile.int_count = 95;
        profile.float_count = 3;
        profile.other_count = 2;

        assert!(profile.is_mostly_int());
    }

    #[test]
    fn test_simple_jit_compilation() {
        let mut jit = JitCompiler::new().expect("Failed to create JIT");

        // Create a simple bytecode function: return constant 42
        let mut chunk = Chunk::new();
        chunk.constants.push(Value::int(42));
        chunk.code.push(Op::load_const(0, 0)); // Load 42 into r0
        chunk.code.push(Op::ret(0)); // Return r0
        chunk.num_registers = 1;
        chunk.num_params = 0;

        let result = jit.compile_function(&chunk, "test_const");
        assert!(result.is_ok(), "Should compile simple constant return");
    }

    #[test]
    fn test_jit_execute_constant() {
        let mut jit = JitCompiler::new().expect("Failed to create JIT");

        // Create bytecode: return constant 42
        let mut chunk = Chunk::new();
        chunk.constants.push(Value::int(42));
        chunk.code.push(Op::load_const(0, 0));
        chunk.code.push(Op::ret(0));
        chunk.num_registers = 1;
        chunk.num_params = 0;

        let chunk_ptr = jit.compile_function(&chunk, "test_exec_const").expect("Compile failed");
        let jit_code = jit.get_compiled(chunk_ptr).expect("Should have compiled code");

        // Execute the JIT-compiled function
        let mut registers = vec![Value::nil(); 16];
        let result = unsafe { jit_code.execute(&mut registers) }.expect("Execute failed");

        assert_eq!(result.as_int(), Some(42), "JIT should return 42");
    }

    #[test]
    fn test_jit_execute_add() {
        let mut jit = JitCompiler::new().expect("Failed to create JIT");

        // Create bytecode: return 10 + 5 = 15
        let mut chunk = Chunk::new();
        chunk.constants.push(Value::int(10));
        chunk.constants.push(Value::int(5));
        chunk.code.push(Op::load_const(0, 0)); // r0 = 10
        chunk.code.push(Op::load_const(1, 1)); // r1 = 5
        chunk.code.push(Op::add_int(2, 0, 1)); // r2 = r0 + r1
        chunk.code.push(Op::ret(2));
        chunk.num_registers = 3;
        chunk.num_params = 0;

        let chunk_ptr = jit.compile_function(&chunk, "test_add").expect("Compile failed");
        let jit_code = jit.get_compiled(chunk_ptr).expect("Should have compiled code");

        let mut registers = vec![Value::nil(); 16];
        let result = unsafe { jit_code.execute(&mut registers) }.expect("Execute failed");

        assert_eq!(result.as_int(), Some(15), "JIT should return 15");
    }

    #[test]
    fn test_jit_execute_loop() {
        let mut jit = JitCompiler::new().expect("Failed to create JIT");

        // Create bytecode for: sum = 0; n = 10; while n > 0: sum += n; n -= 1; return sum
        // Sum of 1..10 = 55
        let mut chunk = Chunk::new();
        chunk.constants.push(Value::int(0));  // 0: initial sum
        chunk.constants.push(Value::int(10)); // 1: initial n

        // r0 = sum, r1 = n
        chunk.code.push(Op::load_const(0, 0)); // 0: r0 = 0 (sum)
        chunk.code.push(Op::load_const(1, 1)); // 1: r1 = 10 (n)

        // Loop header at instruction 2
        // if n <= 0, jump to return (offset +4 -> instruction 7)
        chunk.code.push(Op::jump_if_le_int_imm(1, 0, 4)); // 2: if r1 <= 0, goto 7

        // Loop body
        chunk.code.push(Op::add_int(0, 0, 1));      // 3: r0 = r0 + r1 (sum += n)
        chunk.code.push(Op::sub_int_imm(1, 1, 1));  // 4: r1 = r1 - 1 (n -= 1)
        chunk.code.push(Op::jump(-4));              // 5: goto 2 (loop header)

        // Unreachable but needed for block structure
        chunk.code.push(Op::ret(0));                // 6: return r0 (shouldn't reach here)
        chunk.code.push(Op::ret(0));                // 7: return r0

        chunk.num_registers = 2;
        chunk.num_params = 0;

        let chunk_ptr = jit.compile_function(&chunk, "test_loop").expect("Compile failed");
        let jit_code = jit.get_compiled(chunk_ptr).expect("Should have compiled code");

        let mut registers = vec![Value::nil(); 16];
        let result = unsafe { jit_code.execute(&mut registers) }.expect("Execute failed");

        assert_eq!(result.as_int(), Some(55), "JIT should return sum 1+2+...+10 = 55");
    }

    #[test]
    fn test_jit_countdown_loop() {
        let mut jit = JitCompiler::new().expect("Failed to create JIT");

        // Simple countdown: n = 5; while n > 0: n -= 1; return n (should be 0)
        let mut chunk = Chunk::new();
        chunk.constants.push(Value::int(5));

        chunk.code.push(Op::load_const(0, 0));       // 0: r0 = 5
        chunk.code.push(Op::jump_if_le_int_imm(0, 0, 2)); // 1: if r0 <= 0, goto 4
        chunk.code.push(Op::sub_int_imm(0, 0, 1));   // 2: r0 -= 1
        chunk.code.push(Op::jump(-3));               // 3: goto 1
        chunk.code.push(Op::ret(0));                 // 4: return r0

        chunk.num_registers = 1;
        chunk.num_params = 0;

        let chunk_ptr = jit.compile_function(&chunk, "countdown").expect("Compile failed");
        let jit_code = jit.get_compiled(chunk_ptr).expect("Should have compiled code");

        let mut registers = vec![Value::nil(); 16];
        let result = unsafe { jit_code.execute(&mut registers) }.expect("Execute failed");

        assert_eq!(result.as_int(), Some(0), "JIT countdown should end at 0");
    }
}
