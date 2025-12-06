use crate::bytecode::{Chunk, ConstIdx, Op, Reg};
use crate::value::Value;
use std::collections::HashMap;

/// Static type information for type-specialized code generation
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StaticType {
    Unknown,
    Int,
    Float,
    Bool,
}

/// A function definition (for compile-time evaluation and inlining)
#[derive(Clone)]
struct FunctionDef {
    params: Vec<String>,
    body: Value,
}

/// Registry of pure functions for compile-time evaluation
#[derive(Clone, Default)]
struct PureFunctions {
    funcs: HashMap<String, FunctionDef>,
}

impl PureFunctions {
    fn new() -> Self {
        PureFunctions {
            funcs: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, params: Vec<String>, body: Value) {
        self.funcs.insert(name.to_string(), FunctionDef { params, body });
    }

    fn get(&self, name: &str) -> Option<&FunctionDef> {
        self.funcs.get(name)
    }
}

/// Registry of ALL function definitions for inlining (not matter the purity)
#[derive(Clone, Default)]
struct InlineCandidates {
    funcs: HashMap<String, FunctionDef>,
}

impl InlineCandidates {
    fn new() -> Self {
        InlineCandidates {
            funcs: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, params: Vec<String>, body: Value) {
        self.funcs.insert(name.to_string(), FunctionDef { params, body });
    }

    fn get(&self, name: &str) -> Option<&FunctionDef> {
        self.funcs.get(name)
    }
}

pub struct Compiler {
    chunk: Chunk,
    locals: Vec<String>,
    /// Static type information for each register (for type-specialized opcodes)
    reg_types: Vec<StaticType>,
    scope_depth: usize,
    pure_fns: PureFunctions,
    inline_candidates: InlineCandidates,
    /// Name of the current function being compiled (for self-recursion detection)
    self_name: Option<String>,
    /// Number of parameters of the current function (for loop optimization)
    self_arity: usize,
    /// Stack of functions currently being inlined (for mutual recursion detection)
    inline_stack: Vec<String>,
    /// Parameter types (for type specialization in recursive functions)
    param_types: Vec<StaticType>,
}

/// Check if an expression is pure
fn is_pure_expr_with_fns(expr: &Value, pure_fns: &PureFunctions) -> bool {
    // Literals are pure
    if expr.is_nil() || expr.is_bool() || expr.is_int() || expr.is_float() {
        return true;
    }
    if expr.as_string().is_some() {
        return true;
    }
    // Symbols are pure (just variable references)
    if expr.as_symbol().is_some() {
        return true;
    }
    // Lists need to be checked
    if let Some(items) = expr.as_list() {
        if items.is_empty() {
            return true;
        }
        let first = &items[0];
        if let Some(sym) = first.as_symbol() {
            match sym {
                // Pure built-in operations
                "+" | "-" | "*" | "/" | "mod" | "<" | "<=" | ">" | ">=" | "=" | "!=" | "not" => {
                    return items[1..].iter().all(|e| is_pure_expr_with_fns(e, pure_fns));
                }
                // Conditional is pure if branches are pure
                "if" => {
                    return items[1..].iter().all(|e| is_pure_expr_with_fns(e, pure_fns));
                }
                // Let is pure if bindings and body are pure
                "let" => {
                    if items.len() >= 3 {
                        if let Some(bindings) = items[1].as_list() {
                            return bindings.iter().all(|e| is_pure_expr_with_fns(e, pure_fns)) &&
                                items[2..].iter().all(|e| is_pure_expr_with_fns(e, pure_fns));
                        }
                    }
                    return false;
                }
                // Quote is pure
                "quote" => return true,
                // Check if it's a known pure function
                _ => {
                    if pure_fns.get(sym).is_some() {
                        // It's a call to a known pure function, check args are pure
                        return items[1..].iter().all(|e| is_pure_expr_with_fns(e, pure_fns));
                    }
                    return false;
                }
            }
        }
        return false;
    }
    false
}


/// Try to evaluate a constant expression recursively, including pure function calls
fn try_const_eval_with_fns(expr: &Value, pure_fns: &PureFunctions) -> Option<Value> {
    // Literals are constants
    if expr.is_nil() || expr.is_bool() || expr.is_int() || expr.is_float() {
        return Some(expr.clone());
    }
    if expr.as_string().is_some() {
        return Some(expr.clone());
    }

    // Try to fold function calls
    if let Some(items) = expr.as_list() {
        if items.is_empty() {
            return None;
        }
        let op = items[0].as_symbol()?;
        let args = &items[1..];

        // First try built-in operations
        let const_args: Option<Vec<Value>> = args.iter()
            .map(|a| try_const_eval_with_fns(a, pure_fns))
            .collect();
        let const_args = const_args?;
        let const_refs: Vec<&Value> = const_args.iter().collect();

        if let Some(result) = fold_op(op, &const_refs) {
            return Some(result);
        }

        // Try pure user-defined functions
        if let Some(pure_fn) = pure_fns.get(op) {
            if pure_fn.params.len() == const_args.len() {
                // Substitute parameters with constant values
                let substituted = substitute(&pure_fn.body, &pure_fn.params, &const_args);
                // Recursively evaluate the substituted body
                return try_const_eval_with_fns(&substituted, pure_fns);
            }
        }

        return None;
    }
    None
}

/// Substitute parameters with values in an expression
fn substitute(expr: &Value, params: &[String], args: &[Value]) -> Value {
    if let Some(name) = expr.as_symbol() {
        for (i, param) in params.iter().enumerate() {
            if param == name {
                return args[i].clone();
            }
        }
        return expr.clone();
    }
    if let Some(items) = expr.as_list() {
        let new_items: Vec<Value> = items
            .iter()
            .map(|item| substitute(item, params, args))
            .collect();
        return Value::list(new_items);
    }
    expr.clone()
}

/// Check if expression contains a call to the given function name (recursion check)
fn contains_call_to(expr: &Value, name: &str) -> bool {
    if let Some(items) = expr.as_list() {
        if !items.is_empty() {
            // Check if this is a call to the function
            if let Some(first) = items[0].as_symbol() {
                if first == name {
                    return true;
                }
            }
            // Recursively check all sub-expressions
            for item in items.iter() {
                if contains_call_to(item, name) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if expression contains a call to any function in the given list (mutual recursion check)
fn contains_call_to_any(expr: &Value, names: &[String]) -> bool {
    if names.is_empty() {
        return false;
    }
    if let Some(items) = expr.as_list() {
        if !items.is_empty() {
            // Check if this is a call to any of the functions
            if let Some(first) = items[0].as_symbol() {
                if names.iter().any(|n| n == first) {
                    return true;
                }
            }
            // Recursively check all sub-expressions
            for item in items.iter() {
                if contains_call_to_any(item, names) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a function body is small enough to inline
/// We inline if the body is a single expression that is:
/// >a simple call to another function (thin wrapper)
/// >a simple arithmetic/comparison expression
/// >a small if expression
fn is_small_body(body: &Value) -> bool {
    if let Some(items) = body.as_list() {
        // Body is a list (an expression)
        if items.is_empty() {
            return true;
        }
        if let Some(first) = items[0].as_symbol() {
            match first {
                // Simple operations are always small
                "+" | "-" | "*" | "/" | "mod" | "<" | "<=" | ">" | ">=" | "=" | "!=" | "not" => {
                    return items.len() <= 4; // op + up to 3 args
                }
                // A call to another function (thin wrapper pattern)
                _ => {
                    // Consider it small if it's just a function call with up to 4 args
                    return items.len() <= 5;
                }
            }
        }
    }
    // Non-list values (symbols, literals) are too small
    true
}

/// Try to fold an operation with constant arguments
fn fold_op(op: &str, args: &[&Value]) -> Option<Value> {
    match op {
        "+" => fold_add(args),
        "-" => fold_sub(args),
        "*" => fold_mul(args),
        "/" => fold_div(args),
        "mod" => fold_mod(args),
        "<" => fold_cmp(args, |a, b| a < b),
        "<=" => fold_cmp(args, |a, b| a <= b),
        ">" => fold_cmp(args, |a, b| a > b),
        ">=" => fold_cmp(args, |a, b| a >= b),
        "=" => fold_eq(args),
        "!=" => fold_ne(args),
        "not" => fold_not(args),
        _ => None,
    }
}

fn fold_add(args: &[&Value]) -> Option<Value> {
    let mut sum_int: i64 = 0;
    let mut sum_float: f64 = 0.0;
    let mut is_float = false;

    for arg in args {
        if let Some(n) = arg.as_int() {
            if is_float {
                sum_float += n as f64;
            } else {
                sum_int += n;
            }
        } else if let Some(n) = arg.as_float() {
            if !is_float {
                is_float = true;
                sum_float = sum_int as f64;
            }
            sum_float += n;
        } else {
            return None;
        }
    }

    Some(if is_float {
        Value::float(sum_float)
    } else {
        Value::int(sum_int)
    })
}

fn fold_sub(args: &[&Value]) -> Option<Value> {
    if args.is_empty() {
        return None;
    }
    if args.len() == 1 {
        if let Some(n) = args[0].as_int() {
            return Some(Value::int(-n));
        } else if let Some(n) = args[0].as_float() {
            return Some(Value::float(-n));
        } else {
            return None;
        }
    }

    let mut is_float = args[0].is_float();
    let mut result = if let Some(n) = args[0].as_int() {
        n as f64
    } else if let Some(n) = args[0].as_float() {
        n
    } else {
        return None;
    };

    for arg in &args[1..] {
        if let Some(n) = arg.as_int() {
            result -= n as f64;
        } else if let Some(n) = arg.as_float() {
            is_float = true;
            result -= n;
        } else {
            return None;
        }
    }

    Some(if is_float {
        Value::float(result)
    } else {
        Value::int(result as i64)
    })
}

fn fold_mul(args: &[&Value]) -> Option<Value> {
    let mut prod_int: i64 = 1;
    let mut prod_float: f64 = 1.0;
    let mut is_float = false;

    for arg in args {
        if let Some(n) = arg.as_int() {
            if is_float {
                prod_float *= n as f64;
            } else {
                prod_int *= n;
            }
        } else if let Some(n) = arg.as_float() {
            if !is_float {
                is_float = true;
                prod_float = prod_int as f64;
            }
            prod_float *= n;
        } else {
            return None;
        }
    }

    Some(if is_float {
        Value::float(prod_float)
    } else {
        Value::int(prod_int)
    })
}

fn fold_div(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    let a = if let Some(n) = args[0].as_int() {
        n as f64
    } else if let Some(n) = args[0].as_float() {
        n
    } else {
        return None;
    };
    let b = if let Some(n) = args[1].as_int() {
        n as f64
    } else if let Some(n) = args[1].as_float() {
        n
    } else {
        return None;
    };
    if b == 0.0 {
        return None; // don't fold division by zero
    }
    Some(Value::float(a / b))
}

fn fold_mod(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    if let (Some(a), Some(b)) = (args[0].as_int(), args[1].as_int()) {
        if b == 0 {
            return None;
        }
        return Some(Value::int(a % b));
    }
    None
}

fn to_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_int() {
        Some(n as f64)
    } else if let Some(n) = v.as_float() {
        Some(n)
    } else {
        None
    }
}

fn fold_cmp<F: Fn(f64, f64) -> bool>(args: &[&Value], cmp: F) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    let a = to_f64(args[0])?;
    let b = to_f64(args[1])?;
    Some(Value::bool(cmp(a, b)))
}

fn fold_eq(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    Some(Value::bool(args[0] == args[1]))
}

fn fold_ne(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    Some(Value::bool(args[0] != args[1]))
}

fn fold_not(args: &[&Value]) -> Option<Value> {
    if args.len() != 1 {
        return None;
    }
    Some(Value::bool(!args[0].is_truthy()))
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            chunk: Chunk::new(),
            locals: Vec::new(),
            reg_types: Vec::new(),
            scope_depth: 0,
            pure_fns: PureFunctions::new(),
            inline_candidates: InlineCandidates::new(),
            self_name: None,
            self_arity: 0,
            inline_stack: Vec::new(),
            param_types: Vec::new(),
        }
    }

    pub fn compile(expr: &Value) -> Result<Chunk, String> {
        let mut compiler = Compiler::new();
        let dest = compiler.alloc_reg();
        compiler.compile_expr(expr, dest, true)?;
        compiler.emit(Op::ret(dest));
        compiler.chunk.num_registers = compiler.locals.len().max(1) as u8 + 16; // extra for temps
        // Optimize move semantics using liveness analysis
        compiler.chunk.optimize_moves();
        Ok(compiler.chunk)
    }

    /// Compile multiple expressions, allowing pure function definitions to be used
    /// in next expressions
    pub fn compile_all(exprs: &[Value]) -> Result<Chunk, String> {
        let mut compiler = Compiler::new();
        let dest = compiler.alloc_reg();

        if exprs.is_empty() {
            compiler.emit(Op::load_nil(dest));
        } else {
            // Compile all but last expression (not in tail position)
            for expr in &exprs[..exprs.len() - 1] {
                compiler.compile_expr(expr, dest, false)?;
            }
            // Last expression in tail position
            compiler.compile_expr(&exprs[exprs.len() - 1], dest, true)?;
        }

        compiler.emit(Op::ret(dest));
        compiler.chunk.num_registers = compiler.locals.len().max(1) as u8 + 16;
        // Optimize move semantics using liveness analysis
        compiler.chunk.optimize_moves();
        Ok(compiler.chunk)
    }

    pub fn compile_function(params: &[String], body: &Value) -> Result<Chunk, String> {
        Self::compile_function_named(params, body, None)
    }

    /// Compile a function with an optional name for self-recursion loop optimization
    /// When a name is provided, tail-recursive calls to that name are compiled as loops
    pub fn compile_function_named(params: &[String], body: &Value, name: Option<&str>) -> Result<Chunk, String> {
        let mut compiler = Compiler::new();
        compiler.chunk.num_params = params.len() as u8;

        // Enable loop optimization for self-recursive functions
        if let Some(n) = name {
            compiler.self_name = Some(n.to_string());
            compiler.self_arity = params.len();
        }

        // Parameters occupy the first registers
        // Note: We don't assume parameter types because:
        // >self-recursive functions may not be tail-recursive
        // >even tail-recursive functions might be called with different types
        // Integer-specialized opcodes will still fire for integer literals and
        // results of integer operations, which covers most hot paths
        for param in params {
            compiler.locals.push(param.clone());
            compiler.reg_types.push(StaticType::Unknown);
            compiler.param_types.push(StaticType::Unknown);
        }

        let dest = compiler.alloc_reg();
        compiler.compile_expr(body, dest, true)?;
        compiler.emit(Op::ret(dest));
        compiler.chunk.num_registers = compiler.locals.len().max(1) as u8 + 16;
        // Note: Don't call optimize_moves here, it's called recursively from
        // parent chunk's optimize_moves() to ensure each proto is only optimized once
        Ok(compiler.chunk)
    }

    fn alloc_reg(&mut self) -> Reg {
        let reg = self.locals.len() as Reg;
        self.locals.push(String::new()); // placeholder
        self.reg_types.push(StaticType::Unknown);
        reg
    }

    fn free_reg(&mut self) {
        if !self.locals.is_empty() && self.locals.last().map(|s| s.is_empty()).unwrap_or(false) {
            self.locals.pop();
            self.reg_types.pop();
        }
    }

    /// Set the known static type for a register
    fn set_reg_type(&mut self, reg: Reg, typ: StaticType) {
        if (reg as usize) < self.reg_types.len() {
            self.reg_types[reg as usize] = typ;
        }
    }

    /// Get the known static type for a register
    fn get_reg_type(&self, reg: Reg) -> StaticType {
        self.reg_types.get(reg as usize).copied().unwrap_or(StaticType::Unknown)
    }

    /// Infer the static type of an expression without compiling it
    fn infer_type(&self, expr: &Value) -> StaticType {
        // Integer literals are Int
        if expr.is_int() {
            return StaticType::Int;
        }
        // Float literals are Float
        if expr.is_float() {
            return StaticType::Float;
        }
        // Bool literals are Bool
        if expr.is_bool() {
            return StaticType::Bool;
        }
        // Variables: check if it's a parameter with known type
        if let Some(name) = expr.as_symbol() {
            if let Some(reg) = self.resolve_local(name) {
                return self.get_reg_type(reg);
            }
        }
        // Arithmetic operations on ints produce ints (+ - * mod)
        if let Some(items) = expr.as_list() {
            if !items.is_empty() {
                if let Some(op) = items[0].as_symbol() {
                    match op {
                        "+" | "-" | "*" | "mod" => {
                            // If all operands are Int, result is Int
                            let all_int = items[1..].iter().all(|e| self.infer_type(e) == StaticType::Int);
                            if all_int {
                                return StaticType::Int;
                            }
                        }
                        "<" | "<=" | ">" | ">=" | "=" | "!=" | "not" => {
                            return StaticType::Bool;
                        }
                        _ => {}
                    }
                }
            }
        }
        StaticType::Unknown
    }

    fn emit(&mut self, op: Op) -> usize {
        self.chunk.emit(op)
    }

    fn add_constant(&mut self, value: Value) -> ConstIdx {
        self.chunk.add_constant(value)
    }

    fn resolve_local(&self, name: &str) -> Option<Reg> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local == name {
                return Some(i as Reg);
            }
        }
        None
    }

    fn compile_expr(&mut self, expr: &Value, dest: Reg, tail_pos: bool) -> Result<(), String> {
        if expr.is_nil() {
            self.emit(Op::load_nil(dest));
        } else if let Some(b) = expr.as_bool() {
            if b {
                self.emit(Op::load_true(dest));
            } else {
                self.emit(Op::load_false(dest));
            }
            self.set_reg_type(dest, StaticType::Bool);
        } else if expr.is_int() {
            let idx = self.add_constant(expr.clone());
            self.emit(Op::load_const(dest, idx));
            self.set_reg_type(dest, StaticType::Int);
        } else if expr.is_float() {
            let idx = self.add_constant(expr.clone());
            self.emit(Op::load_const(dest, idx));
            self.set_reg_type(dest, StaticType::Float);
        } else if expr.as_string().is_some() {
            let idx = self.add_constant(expr.clone());
            self.emit(Op::load_const(dest, idx));
        } else if let Some(name) = expr.as_symbol() {
            if let Some(reg) = self.resolve_local(name) {
                if reg != dest {
                    self.emit(Op::mov(dest, reg));
                }
                // Propagate the type from source register
                self.set_reg_type(dest, self.get_reg_type(reg));
            } else {
                let idx = self.add_constant(Value::symbol(name));
                self.emit(Op::get_global(dest, idx));
            }
        } else if let Some(items) = expr.as_list() {
            if items.is_empty() {
                let idx = self.add_constant(Value::list(vec![]));
                self.emit(Op::load_const(dest, idx));
                return Ok(());
            }

            let first = &items[0];
            if let Some(sym) = first.as_symbol() {
                match sym {
                    "quote" => return self.compile_quote(&items[1..], dest),
                    "if" => return self.compile_if(&items[1..], dest, tail_pos),
                    "cond" => return self.compile_cond(&items[1..], dest, tail_pos),
                    "and" => return self.compile_and(&items[1..], dest),
                    "or" => return self.compile_or(&items[1..], dest),
                    "def" => return self.compile_def(&items[1..], dest),
                    "let" => return self.compile_let(&items[1..], dest, tail_pos),
                    "fn" | "lambda" => return self.compile_fn(&items[1..], dest),
                    "do" => return self.compile_do(&items[1..], dest, tail_pos),
                    _ => {}
                }
            }

            // Func call
            self.compile_call(items, dest, tail_pos)?;
        } else if expr.as_function().is_some() || expr.as_native_function().is_some() || expr.as_compiled_function().is_some() {
            return Err("Cannot compile function value directly".to_string());
        }
        Ok(())
    }

    fn compile_quote(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        if args.len() != 1 {
            return Err("quote expects exactly 1 argument".to_string());
        }
        let idx = self.add_constant(args[0].clone());
        self.emit(Op::load_const(dest, idx));
        Ok(())
    }

    fn compile_if(&mut self, args: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if args.len() < 2 || args.len() > 3 {
            return Err("if expects 2 or 3 arguments".to_string());
        }

        // Try to use combined compare-and-jump opcodes for comparison conditions
        if let Some(jump_to_else) = self.try_compile_compare_jump(&args[0], dest)? {
            // Successfully emitted a combined compare-and-jump opcode
            return self.compile_if_branches(args, dest, tail_pos, jump_to_else);
        }

        // Try to use specialized nil check opcodes
        if let Some(jump_to_else) = self.try_compile_nil_check_jump(&args[0], dest)? {
            return self.compile_if_branches(args, dest, tail_pos, jump_to_else);
        }

        // Fallback: compile condition into dest and use JumpIfFalse
        self.compile_expr(&args[0], dest, false)?;

        // Jump to else if false
        let jump_to_else = self.emit(Op::jump_if_false(dest, 0));

        self.compile_if_branches(args, dest, tail_pos, jump_to_else)
    }

    /// Compile the then/else branches of an if expression
    fn compile_if_branches(&mut self, args: &[Value], dest: Reg, tail_pos: bool, jump_to_else: usize) -> Result<(), String> {
        // Compile then branch
        self.compile_expr(&args[1], dest, tail_pos)?;

        if args.len() == 3 {
            // Jump over else
            let jump_over_else = self.emit(Op::jump(0));

            // Patch jump to else
            let else_start = self.chunk.current_pos();
            self.chunk.patch_jump(jump_to_else, else_start);

            // Compile else branch
            self.compile_expr(&args[2], dest, tail_pos)?;

            // Patch jump over else
            let end = self.chunk.current_pos();
            self.chunk.patch_jump(jump_over_else, end);
        } else {
            // No else: load nil
            let jump_over_nil = self.emit(Op::jump(0));

            let else_start = self.chunk.current_pos();
            self.chunk.patch_jump(jump_to_else, else_start);
            self.emit(Op::load_nil(dest));

            let end = self.chunk.current_pos();
            self.chunk.patch_jump(jump_over_nil, end);
        }

        Ok(())
    }

    /// Compile cond expression: (cond (test1 result1) (test2 result2) ... (else default))
    /// Each clause is a list where the first element is the condition and the rest is the body.
    /// The special condition 'else' or 'true' matches unconditionally.
    fn compile_cond(&mut self, clauses: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if clauses.is_empty() {
            // Empty cond returns nil
            self.emit(Op::load_nil(dest));
            return Ok(());
        }

        // Track jumps to end (from successful branches)
        let mut jumps_to_end: Vec<usize> = Vec::new();

        for (i, clause) in clauses.iter().enumerate() {
            let is_last = i == clauses.len() - 1;

            // Each clause should be a list: (condition body...)
            let clause_items = clause.as_list()
                .ok_or_else(|| "cond clause must be a list".to_string())?;

            if clause_items.is_empty() {
                return Err("cond clause cannot be empty".to_string());
            }

            let condition = &clause_items[0];
            let body = &clause_items[1..];

            // Check for else/true clause (unconditional)
            let is_else = condition.as_symbol()
                .map(|s| s == "else" || s == "true")
                .unwrap_or(false);

            if is_else {
                // Else clause - just compile the body
                if body.is_empty() {
                    self.emit(Op::load_nil(dest));
                } else if body.len() == 1 {
                    self.compile_expr(&body[0], dest, tail_pos)?;
                } else {
                    // Multiple expressions: wrap in implicit do
                    let mut do_list = vec![Value::symbol("do")];
                    do_list.extend(body.iter().cloned());
                    self.compile_expr(&Value::list(do_list), dest, tail_pos)?;
                }
                // No need to jump - this is the final case
                break;
            }

            // Compile condition
            self.compile_expr(condition, dest, false)?;

            if is_last {
                // Last clause without else: if false, result is nil
                let jump_to_nil = self.emit(Op::jump_if_false(dest, 0));

                // Compile body
                if body.is_empty() {
                    // If body is empty, the result is the condition value (already in dest)
                    // But we need to handle the nil case
                } else if body.len() == 1 {
                    self.compile_expr(&body[0], dest, tail_pos)?;
                } else {
                    let mut do_list = vec![Value::symbol("do")];
                    do_list.extend(body.iter().cloned());
                    self.compile_expr(&Value::list(do_list), dest, tail_pos)?;
                }

                // Jump over nil
                let jump_over_nil = self.emit(Op::jump(0));

                // Nil case
                let nil_pos = self.chunk.current_pos();
                self.chunk.patch_jump(jump_to_nil, nil_pos);
                self.emit(Op::load_nil(dest));

                // Patch jump over nil
                let end_pos = self.chunk.current_pos();
                self.chunk.patch_jump(jump_over_nil, end_pos);
            } else {
                // Not the last clause: if false, jump to next clause
                let jump_to_next = self.emit(Op::jump_if_false(dest, 0));

                // Compile body
                if body.is_empty() {
                    // Empty body - condition value is the result (already in dest)
                } else if body.len() == 1 {
                    self.compile_expr(&body[0], dest, tail_pos)?;
                } else {
                    let mut do_list = vec![Value::symbol("do")];
                    do_list.extend(body.iter().cloned());
                    self.compile_expr(&Value::list(do_list), dest, tail_pos)?;
                }

                // Jump to end (skip remaining clauses)
                jumps_to_end.push(self.emit(Op::jump(0)));

                // Patch jump to next clause
                let next_clause_pos = self.chunk.current_pos();
                self.chunk.patch_jump(jump_to_next, next_clause_pos);
            }
        }

        // Patch all jumps to end
        let end_pos = self.chunk.current_pos();
        for jump_pos in jumps_to_end {
            self.chunk.patch_jump(jump_pos, end_pos);
        }

        Ok(())
    }

    fn compile_and(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        // (and) => true
        if args.is_empty() {
            self.emit(Op::load_true(dest));
            return Ok(());
        }

        // (and x) => x
        if args.len() == 1 {
            return self.compile_expr(&args[0], dest, false);
        }

        // (and x y z ...) => short-circuit evaluation
        // Evaluate each expression, if any is false, jump to end with false
        let mut jump_to_end = Vec::new();

        for (i, arg) in args.iter().enumerate() {
            self.compile_expr(arg, dest, false)?;

            // For all but the last, check if false and jump to end
            if i < args.len() - 1 {
                let jump = self.emit(Op::jump_if_false(dest, 0));
                jump_to_end.push(jump);
            }
        }

        // Patch all jumps to end (dest already has the last value or the false value)
        let end = self.chunk.current_pos();
        for jump in jump_to_end {
            self.chunk.patch_jump(jump, end);
        }

        Ok(())
    }

    fn compile_or(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        // (or) => false
        if args.is_empty() {
            self.emit(Op::load_false(dest));
            return Ok(());
        }

        // (or x) => x
        if args.len() == 1 {
            return self.compile_expr(&args[0], dest, false);
        }

        // (or x y z ...) => short-circuit evaluation
        // Evaluate each expression, if any is true, jump to end with that value
        let mut jump_to_end = Vec::new();

        for (i, arg) in args.iter().enumerate() {
            self.compile_expr(arg, dest, false)?;

            // For all but the last, check if true and jump to end
            if i < args.len() - 1 {
                let jump = self.emit(Op::jump_if_true(dest, 0));
                jump_to_end.push(jump);
            }
        }

        // Patch all jumps to end (dest already has the last value or the true value)
        let end = self.chunk.current_pos();
        for jump in jump_to_end {
            self.chunk.patch_jump(jump, end);
        }

        Ok(())
    }

    /// Try to compile a comparison condition as a combined compare-and-jump opcode
    /// Returns Some(jump_pos) if successful, None if condition is not a simple comparison
    ///
    /// For `(if (< a b) then else)`, we need to jump to else when condition is FALSE.
    /// So we emit the OPPOSITE comparison:
    /// - `<`  → JumpIfGe (jump if a >= b)
    /// - `<=` → JumpIfGt (jump if a > b)
    /// - `>`  → JumpIfLe (jump if a <= b)
    /// - `>=` → JumpIfLt (jump if a < b)
    ///
    /// When the left operand is known to be an integer, we use integer-specialized
    /// opcodes that skip type checking (faster in tight loops).    
    fn try_compile_compare_jump(&mut self, cond: &Value, dest: Reg) -> Result<Option<usize>, String> {
        let items = match cond.as_list() {
            Some(items) if items.len() == 3 => items,
            _ => return Ok(None),
        };

        let op = match items[0].as_symbol() {
            Some(s) if s == "<" || s == "<=" || s == ">" || s == ">=" => s,
            _ => return Ok(None),
        };

        let left = &items[1];
        let right = &items[2];

        // Infer type of left operand to determine if we can use integer-specialized opcodes
        let left_type = self.infer_type(left);
        let use_int_opcodes = left_type == StaticType::Int;

        // Check for immediate optimization: (< x 0), (<= n 10), etc.
        if let Some(imm) = right.as_int() {
            if imm >= i8::MIN as i64 && imm <= i8::MAX as i64 {
                // Compile left operand into dest
                self.compile_expr(left, dest, false)?;

                // Emit combined compare-jump with OPPOSITE comparison
                // placeholder offset 0, will be patched later
                let jump_pos = if use_int_opcodes {
                    match op {
                        "<"  => self.emit(Op::jump_if_ge_int_imm(dest, imm as i8, 0)),
                        "<=" => self.emit(Op::jump_if_gt_int_imm(dest, imm as i8, 0)),
                        ">"  => self.emit(Op::jump_if_le_int_imm(dest, imm as i8, 0)),
                        ">=" => self.emit(Op::jump_if_lt_int_imm(dest, imm as i8, 0)),
                        _ => unreachable!(),
                    }
                } else {
                    match op {
                        "<"  => self.emit(Op::jump_if_ge_imm(dest, imm as i8, 0)),
                        "<=" => self.emit(Op::jump_if_gt_imm(dest, imm as i8, 0)),
                        ">"  => self.emit(Op::jump_if_le_imm(dest, imm as i8, 0)),
                        ">=" => self.emit(Op::jump_if_lt_imm(dest, imm as i8, 0)),
                        _ => unreachable!(),
                    }
                };
                return Ok(Some(jump_pos));
            }
        }

        // Register-register comparison
        // Compile left into dest
        self.compile_expr(left, dest, false)?;
        // Compile right into temp register
        let right_reg = self.alloc_reg();
        self.compile_expr(right, right_reg, false)?;

        // Emit combined compare-jump with OPPOSITE comparison
        let jump_pos = match op {
            "<"  => self.emit(Op::jump_if_ge(dest, right_reg, 0)),
            "<=" => self.emit(Op::jump_if_gt(dest, right_reg, 0)),
            ">"  => self.emit(Op::jump_if_le(dest, right_reg, 0)),
            ">=" => self.emit(Op::jump_if_lt(dest, right_reg, 0)),
            _ => unreachable!(),
        };

        self.free_reg(); // free right_reg
        Ok(Some(jump_pos))
    }

    /// Try to compile a nil check condition as a specialized jump opcode
    /// Returns Some(jump_pos) if successful, None if condition is not a nil check
    ///
    /// Handles pattern:
    /// - `(nil? x)` → JumpIfNotNil (jump to else if x is NOT nil)
    ///
    /// Note: We don't optimize `(empty? x)` because it handles array-based lists
    /// differently (empty array [] is not nil but is empty)
    fn try_compile_nil_check_jump(&mut self, cond: &Value, dest: Reg) -> Result<Option<usize>, String> {
        let items = match cond.as_list() {
            Some(items) if items.len() == 2 => items,
            _ => return Ok(None),
        };

        // Only optimize nil? - not empty? which has different semantics for array lists
        if items[0].as_symbol() != Some("nil?") {
            return Ok(None);
        }

        let arg = &items[1];

        // Compile argument into dest
        self.compile_expr(arg, dest, false)?;

        // For `(if (nil? x) then else)`, we jump to else when condition is FALSE
        // Condition is TRUE when x IS nil
        // So we jump to else when x is NOT nil
        let jump_pos = self.emit(Op::jump_if_not_nil(dest, 0));

        Ok(Some(jump_pos))
    }

    fn compile_def(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        if args.len() != 2 {
            return Err("def expects exactly 2 arguments".to_string());
        }

        let name = args[0]
            .as_symbol()
            .ok_or("def expects a symbol as first argument")?;

        // Check if we're defining a function: (def name (fn (params) body))
        if let Some(fn_expr) = args[1].as_list() {
            if fn_expr.len() >= 3 {
                if let Some(fn_sym) = fn_expr[0].as_symbol() {
                    if fn_sym == "fn" {
                        if let Some(params_list) = fn_expr[1].as_list() {
                            // Get parameter names
                            let params: Option<Vec<String>> = params_list
                                .iter()
                                .map(|p| p.as_symbol().map(|s| s.to_string()))
                                .collect();

                            if let Some(params) = params {
                                // Get the body (handle multi-expression body)
                                let body = if fn_expr.len() == 3 {
                                    fn_expr[2].clone()
                                } else {
                                    let mut do_list = vec![Value::symbol("do")];
                                    do_list.extend(fn_expr[2..].iter().cloned());
                                    Value::list(do_list)
                                };

                                // Register ALL functions as inline candidates
                                // (inlining decision made at call site based on size/recursion)
                                self.inline_candidates.register(name, params.clone(), body.clone());

                                // Also register pure functions for constant folding
                                if is_pure_expr_with_fns(&body, &self.pure_fns) {
                                    self.pure_fns.register(name, params.clone(), body.clone());
                                }

                                // Check if this function is self-recursive (calls itself)
                                // If so, compile with the name for loop optimization
                                let is_self_recursive = contains_call_to(&body, name);

                                // Compile the function with name for self-recursion loop optimization
                                let proto = if is_self_recursive {
                                    Compiler::compile_function_named(&params, &body, Some(name))?
                                } else {
                                    Compiler::compile_function(&params, &body)?
                                };

                                let proto_idx = self.chunk.protos.len() as ConstIdx;
                                self.chunk.protos.push(proto);
                                self.emit(Op::closure(dest, proto_idx));

                                // Store to global
                                let name_idx = self.add_constant(Value::symbol(name));
                                self.emit(Op::set_global(name_idx, dest));

                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Compile value (non-function or anonymous function)
        self.compile_expr(&args[1], dest, false)?;

        // Store to global
        let name_idx = self.add_constant(Value::symbol(name));
        self.emit(Op::set_global(name_idx, dest));

        Ok(())
    }

    fn compile_let(&mut self, args: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if args.len() < 2 {
            return Err("let expects at least 2 arguments".to_string());
        }

        let bindings = args[0]
            .as_list()
            .ok_or("let expects a list of bindings")?;

        if bindings.len() % 2 != 0 {
            return Err("let bindings must be pairs".to_string());
        }

        self.scope_depth += 1;
        let locals_before = self.locals.len();

        // Compile bindings
        for chunk in bindings.chunks(2) {
            let name = chunk[0]
                .as_symbol()
                .ok_or("let binding name must be a symbol")?;

            let reg = self.alloc_reg();
            self.compile_expr(&chunk[1], reg, false)?;
            // Replace placeholder with actual name
            self.locals[reg as usize] = name.to_string();
        }

        // Compile body (all but last not in tail position)
        let body = &args[1..];
        if body.is_empty() {
            self.emit(Op::load_nil(dest));
        } else {
            for expr in &body[..body.len() - 1] {
                let temp = self.alloc_reg();
                self.compile_expr(expr, temp, false)?;
                self.free_reg();
            }
            self.compile_expr(&body[body.len() - 1], dest, tail_pos)?;
        }

        // Pop locals
        self.locals.truncate(locals_before);
        self.scope_depth -= 1;

        Ok(())
    }

    fn compile_fn(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        if args.len() < 2 {
            return Err("fn expects at least 2 arguments".to_string());
        }

        let params_list = args[0]
            .as_list()
            .ok_or("fn expects a list of parameters")?;

        let params: Result<Vec<String>, String> = params_list
            .iter()
            .map(|p| {
                p.as_symbol()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "Parameter must be a symbol".to_string())
            })
            .collect();
        let params = params?;

        // Handle multi-expression body as implicit do
        let body = if args.len() == 2 {
            args[1].clone()
        } else {
            let mut do_list = vec![Value::symbol("do")];
            do_list.extend(args[1..].iter().cloned());
            Value::list(do_list)
        };

        // Compile nested function
        let proto = Compiler::compile_function(&params, &body)?;
        let proto_idx = self.chunk.protos.len() as ConstIdx;
        self.chunk.protos.push(proto);

        self.emit(Op::closure(dest, proto_idx));
        Ok(())
    }

    fn compile_do(&mut self, args: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if args.is_empty() {
            self.emit(Op::load_nil(dest));
            return Ok(());
        }

        // All but last not in tail position
        for expr in &args[..args.len() - 1] {
            let temp = self.alloc_reg();
            self.compile_expr(expr, temp, false)?;
            self.free_reg();
        }

        // Last in tail position
        self.compile_expr(&args[args.len() - 1], dest, tail_pos)
    }

    /// Try to compile a binary operation using specialized opcodes
    /// Returns Some(true) if compiled, Some(false) if not applicable, Err on error
    fn try_compile_binary_op(&mut self, op: &str, args: &[Value], dest: Reg) -> Result<Option<bool>, String> {
        // Binary arithmetic/comparison operators
        if args.len() == 2 {
            // Check if both operands are known integers for type-specialized opcodes
            let left_type = self.infer_type(&args[0]);
            let right_type = self.infer_type(&args[1]);
            let both_int = left_type == StaticType::Int && right_type == StaticType::Int;

            // Check for immediate value optimization (+ n 1) or (- n 1)
            // Only applies to + and - where one operand is a small constant
            if op == "+" || op == "-" {
                // Check if second arg is a small constant integer
                if let Some(imm) = args[1].as_int() {
                    if imm >= i8::MIN as i64 && imm <= i8::MAX as i64 {
                        // Optimization: compile first arg directly into dest
                        self.compile_expr(&args[0], dest, false)?;
                        // Use integer-specialized opcodes if first operand is known int
                        if left_type == StaticType::Int {
                            if op == "+" {
                                self.emit(Op::add_int_imm(dest, dest, imm as i8));
                            } else {
                                self.emit(Op::sub_int_imm(dest, dest, imm as i8));
                            }
                            self.set_reg_type(dest, StaticType::Int);
                        } else {
                            if op == "+" {
                                self.emit(Op::add_imm(dest, dest, imm as i8));
                            } else {
                                self.emit(Op::sub_imm(dest, dest, imm as i8));
                            }
                        }
                        return Ok(Some(true));
                    }
                }
                // Check if first arg is a small constant integer (for + only, since + is commutative)
                if op == "+" {
                    if let Some(imm) = args[0].as_int() {
                        if imm >= i8::MIN as i64 && imm <= i8::MAX as i64 {
                            // Optimization: compile second arg directly into dest
                            self.compile_expr(&args[1], dest, false)?;
                            // Use integer-specialized opcode if second operand is known int
                            if right_type == StaticType::Int {
                                self.emit(Op::add_int_imm(dest, dest, imm as i8));
                                self.set_reg_type(dest, StaticType::Int);
                            } else {
                                self.emit(Op::add_imm(dest, dest, imm as i8));
                            }
                            return Ok(Some(true));
                        }
                    }
                }
            }

            // Check for comparison with immediate value: (< x 0), (<= n 10), etc.
            if op == "<" || op == "<=" || op == ">" || op == ">=" {
                // Check if second arg is a small constant integer
                if let Some(imm) = args[1].as_int() {
                    if imm >= i8::MIN as i64 && imm <= i8::MAX as i64 {
                        // Optimization: compile first arg directly into dest
                        self.compile_expr(&args[0], dest, false)?;
                        match op {
                            "<" => self.emit(Op::lt_imm(dest, dest, imm as i8)),
                            "<=" => self.emit(Op::le_imm(dest, dest, imm as i8)),
                            ">" => self.emit(Op::gt_imm(dest, dest, imm as i8)),
                            ">=" => self.emit(Op::ge_imm(dest, dest, imm as i8)),
                            _ => unreachable!(),
                        };
                        self.set_reg_type(dest, StaticType::Bool);
                        return Ok(Some(true));
                    }
                }
            }

            // Integer-specialized register-register operations
            if both_int {
                let make_int_op: Option<fn(Reg, Reg, Reg) -> Op> = match op {
                    "+" => Some(Op::add_int),
                    "-" => Some(Op::sub_int),
                    "*" => Some(Op::mul_int),
                    "<" => Some(Op::lt_int),
                    "<=" => Some(Op::le_int),
                    ">" => Some(Op::gt_int),
                    ">=" => Some(Op::ge_int),
                    _ => None,
                };

                if let Some(make_op) = make_int_op {
                    // Compile operands
                    self.compile_expr(&args[0], dest, false)?;
                    let b_reg = self.alloc_reg();
                    self.compile_expr(&args[1], b_reg, false)?;

                    // Emit the integer-specialized operation
                    self.emit(make_op(dest, dest, b_reg));

                    // Set result type
                    match op {
                        "+" | "-" | "*" => self.set_reg_type(dest, StaticType::Int),
                        "<" | "<=" | ">" | ">=" => self.set_reg_type(dest, StaticType::Bool),
                        _ => {}
                    }

                    self.free_reg();
                    return Ok(Some(true));
                }
            }

            // Generic binary operations (fallback for unknown types)
            let make_binary_op: Option<fn(Reg, Reg, Reg) -> Op> = match op {
                "+" => Some(Op::add),
                "-" => Some(Op::sub),
                "*" => Some(Op::mul),
                "/" => Some(Op::div),
                "mod" => Some(Op::modulo),
                "<" => Some(Op::lt),
                "<=" => Some(Op::le),
                ">" => Some(Op::gt),
                ">=" => Some(Op::ge),
                "=" => Some(Op::eq),
                "!=" => Some(Op::ne),
                _ => None,
            };

            if let Some(make_op) = make_binary_op {
                // Optimization: compile first arg directly into dest to save a register
                self.compile_expr(&args[0], dest, false)?;
                let b_reg = self.alloc_reg();
                self.compile_expr(&args[1], b_reg, false)?;

                // Emit the operation (dest = dest op b_reg)
                self.emit(make_op(dest, dest, b_reg));

                // Free temp register
                self.free_reg();

                return Ok(Some(true));
            }
        }

        // Unary operators
        if args.len() == 1 {
            if op == "not" {
                // Optimization: compile arg directly into dest
                self.compile_expr(&args[0], dest, false)?;
                self.emit(Op::not(dest, dest));
                self.set_reg_type(dest, StaticType::Bool);
                return Ok(Some(true));
            }

            // Unary minus: (- x)
            if op == "-" {
                // Optimization: compile arg directly into dest
                self.compile_expr(&args[0], dest, false)?;
                self.emit(Op::neg(dest, dest));
                return Ok(Some(true));
            }
        }

        // Not a specialized operation
        Ok(None)
    }

    fn compile_call(&mut self, items: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        // Try constant folding for the entire call expression (including pure user functions)
        let call_expr = Value::list(items.to_vec());
        if let Some(folded) = try_const_eval_with_fns(&call_expr, &self.pure_fns) {
            let idx = self.add_constant(folded);
            self.emit(Op::load_const(dest, idx));
            return Ok(());
        }

        // Try to compile as specialized binary operation
        if let Some(op) = items[0].as_symbol() {
            if let Some(result) = self.try_compile_binary_op(op, &items[1..], dest)? {
                if result {
                    return Ok(());
                }
            }

            // Try specialized car/cdr opcodes (single argument)
            let args = &items[1..];
            if args.len() == 1 {
                match op {
                    "car" => {
                        // Compile arg into dest, then emit Car
                        self.compile_expr(&args[0], dest, false)?;
                        self.emit(Op::car(dest, dest));
                        return Ok(());
                    }
                    "cdr" => {
                        // Compile arg into dest, then emit Cdr
                        self.compile_expr(&args[0], dest, false)?;
                        self.emit(Op::cdr(dest, dest));
                        return Ok(());
                    }
                    _ => {}
                }
            }

            // Try specialized cons opcode (two arguments)
            if args.len() == 2 && op == "cons" {
                // Compile car into dest
                self.compile_expr(&args[0], dest, false)?;
                // Compile cdr into temp register
                let cdr_reg = self.alloc_reg();
                self.compile_expr(&args[1], cdr_reg, false)?;
                // Emit Cons
                self.emit(Op::cons(dest, dest, cdr_reg));
                self.free_reg(); // free cdr_reg
                return Ok(());
            }

            // Try inlining small non-recursive functions
            // This eliminates call overhead for thin wrappers like:
            //   (def reverse (fn (lst) (reverse-acc lst (list))))
            if let Some(fn_def) = self.inline_candidates.get(op).cloned() {
                // Check: correct number of arguments
                if args.len() == fn_def.params.len() {
                    // Check: function is small enough to inline
                    if is_small_body(&fn_def.body) {
                        // Check: function doesn't call itself (direct recursion)
                        if !contains_call_to(&fn_def.body, op) {
                            // Check: function doesn't call any function in the inline stack (mutual recursion)
                            if !contains_call_to_any(&fn_def.body, &self.inline_stack) {
                                // Inline: substitute parameters with arguments and compile
                                let inlined = substitute(&fn_def.body, &fn_def.params, args);
                                // Track that we're inlining this function (for mutual recursion detection)
                                self.inline_stack.push(op.to_string());
                                let result = self.compile_expr(&inlined, dest, tail_pos);
                                self.inline_stack.pop();
                                return result;
                            }
                        }
                    }
                }
            }
        }

        // Check if calling a global symbol (optimization: use CallGlobal/TailCallGlobal)
        let is_global_call = if let Some(name) = items[0].as_symbol() {
            // It's a global if it's not in our local variables
            !self.locals.iter().any(|local| local == name)
        } else {
            false
        };

        if is_global_call {
            let name = items[0].as_symbol().unwrap();
            let args = &items[1..];
            let nargs = args.len() as u8;

            // LOOP OPTIMIZATION: Detect self-recursive tail calls and compile as loops
            // This eliminates function call overhead for tail-recursive functions
            if tail_pos {
                if let Some(ref self_name) = self.self_name {
                    if name == self_name && args.len() == self.self_arity {
                        // Self-recursive tail call detected!
                        // Compile arguments to temporary registers
                        let first_temp = self.locals.len() as Reg;
                        for arg in args.iter() {
                            let arg_reg = self.alloc_reg();
                            self.compile_expr(arg, arg_reg, false)?;
                        }

                        // Copy temporaries back to parameter registers (0, 1, 2, ...)
                        for i in 0..nargs as usize {
                            self.emit(Op::mov(i as Reg, first_temp + i as Reg));
                        }

                        // Free the temp registers
                        for _ in 0..nargs {
                            self.free_reg();
                        }

                        // Jump back to the start of the function (instruction 0)
                        // Current position is chunk.code.len(), target is 0
                        // offset = target - (current + 1) = 0 - (current + 1) = -(current + 1)
                        let current_pos = self.chunk.current_pos() as i16;
                        let offset = -(current_pos + 1);
                        self.emit(Op::jump(offset));

                        // The LoadNil is unreachable but keeps the dest semantics consistent
                        self.emit(Op::load_nil(dest));
                        return Ok(());
                    }
                }
            }

            let name_idx = self.add_constant(Value::symbol(name));

            // CallGlobal and TailCallGlobal use 8-bit constant index
            // If name_idx > 255, fall back to regular GetGlobal + Call
            if name_idx <= 255 {
                let name_idx_u8 = name_idx as u8;

                if tail_pos {
                    // For TailCallGlobal, compile args to temp registers first
                    let first_arg = self.locals.len() as Reg;
                    for arg in args.iter() {
                        let arg_reg = self.alloc_reg();
                        self.compile_expr(arg, arg_reg, false)?;
                    }
                    self.emit(Op::tail_call_global(name_idx_u8, first_arg, nargs));
                    self.emit(Op::load_nil(dest));
                    // Free the temp registers
                    for _ in 0..nargs {
                        self.free_reg();
                    }
                } else {
                    // For CallGlobal, args go in dest+1, dest+2, ...
                    // BUG FIX: If dest+1 would overlap with locals, use a temp dest
                    // to prevent arguments from overwriting let-bound variables.
                    let start_locals = self.locals.len();
                    let actual_dest = if dest + 1 < start_locals as Reg {
                        // Use a destination after all locals to avoid register clash
                        start_locals as Reg
                    } else {
                        dest
                    };

                    for (i, arg) in args.iter().enumerate() {
                        let arg_reg = actual_dest + 1 + i as Reg;
                        while (self.locals.len() as Reg) <= arg_reg {
                            self.alloc_reg();
                        }
                        self.compile_expr(arg, arg_reg, false)?;
                    }
                    self.emit(Op::call_global(actual_dest, name_idx_u8, nargs));

                    // If we used a temp dest, move result to original dest
                    if actual_dest != dest {
                        self.emit(Op::mov(dest, actual_dest));
                    }

                    while self.locals.len() > start_locals {
                        self.free_reg();
                    }
                }
                return Ok(());
            }
            // Fall through to regular call path if name_idx > 255
        }

        // Regular call path (or fallback for large name_idx)
        let func_reg = self.alloc_reg();

        // Compile function
        self.compile_expr(&items[0], func_reg, false)?;

        // Compile arguments into consecutive registers after func
        let args = &items[1..];
        let nargs = args.len() as u8;

        for arg in args {
            let arg_reg = self.alloc_reg();
            self.compile_expr(arg, arg_reg, false)?;
        }

        // Emit call
        if tail_pos {
            self.emit(Op::tail_call(func_reg, nargs));
            // After tail call, we still need to put something in dest for non-tail paths
            // But if it's truly a tail call, we won't reach here
            self.emit(Op::load_nil(dest));
        } else {
            self.emit(Op::call(dest, func_reg, nargs));
        }

        // Free arg registers and func register
        for _ in 0..=nargs {
            self.free_reg();
        }

        Ok(())
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn compile_str(input: &str) -> Result<Chunk, String> {
        let expr = parse(input)?;
        Compiler::compile(&expr)
    }

    #[test]
    fn test_compile_literals() {
        let chunk = compile_str("42").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.code[0].a(), 0);
        assert_eq!(chunk.code[1].opcode(), Op::RETURN);
    }

    #[test]
    fn test_compile_nil() {
        let chunk = compile_str("nil").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_NIL);
    }

    #[test]
    fn test_compile_bool() {
        let chunk = compile_str("true").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_TRUE);

        let chunk = compile_str("false").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_FALSE);
    }

    #[test]
    fn test_compile_quote() {
        let chunk = compile_str("'foo").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::symbol("foo"));
    }

    #[test]
    fn test_compile_def() {
        let chunk = compile_str("(def x 42)").unwrap();
        // Should have LoadConst, SetGlobal, Return
        assert!(chunk.code.len() >= 2);
    }

    #[test]
    fn test_compile_fn() {
        let chunk = compile_str("(fn (x) x)").unwrap();
        // Should emit Closure instruction
        assert_eq!(chunk.code[0].opcode(), Op::CLOSURE);
        assert_eq!(chunk.code[0].a(), 0);
        assert_eq!(chunk.code[0].bx(), 0);
        // Should have a proto
        assert_eq!(chunk.protos.len(), 1);
        assert_eq!(chunk.protos[0].num_params, 1);
    }

    #[test]
    fn test_compile_if() {
        let chunk = compile_str("(if true 1 2)").unwrap();
        // Should have jumps
        let has_jump = chunk.code.iter().any(|op| {
            let opc = op.opcode();
            opc == Op::JUMP || opc == Op::JUMP_IF_FALSE
        });
        assert!(has_jump);
    }

    #[test]
    fn test_compile_let() {
        let chunk = compile_str("(let (x 10) x)").unwrap();
        // Should compile without error
        assert!(!chunk.code.is_empty());
    }

    #[test]
    fn test_compile_call() {
        // Top-level call is in tail position, so it's a TailCall or TailCallGlobal
        let chunk = compile_str("(foo 1 2)").unwrap();
        let has_tail_call = chunk.code.iter().any(|op| {
            let opc = op.opcode();
            opc == Op::TAIL_CALL || opc == Op::TAIL_CALL_GLOBAL
        });
        assert!(has_tail_call);

        // Non-tail call (if condition is not in tail position)
        let chunk = compile_str("(if (foo) 1 2)").unwrap();
        let has_call = chunk.code.iter().any(|op| {
            let opc = op.opcode();
            opc == Op::CALL || opc == Op::CALL_GLOBAL
        });
        assert!(has_call);
    }

    #[test]
    fn test_constant_folding_add() {
        let chunk = compile_str("(+ 1 2 3)").unwrap();
        // Should fold to LoadConst 6, not a call
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::int(6));
        // No function calls
        let has_call = chunk.code.iter().any(|op| {
            let opc = op.opcode();
            opc == Op::CALL || opc == Op::TAIL_CALL
        });
        assert!(!has_call);
    }

    #[test]
    fn test_constant_folding_nested() {
        let chunk = compile_str("(* 2 (+ 3 4))").unwrap();
        // Should fold to LoadConst 14
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::int(14));
    }

    #[test]
    fn test_constant_folding_comparison() {
        let chunk = compile_str("(< 1 2)").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::bool(true));

        let chunk = compile_str("(>= 5 10)").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::bool(false));
    }

    #[test]
    fn test_constant_folding_not() {
        let chunk = compile_str("(not false)").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::bool(true));
    }

    #[test]
    fn test_no_fold_with_variables() {
        // If any arg is not a constant, don't fold to LoadConst
        let chunk = compile_str("(+ x 1)").unwrap();
        // Should use Op::AddImm (specialized for small constants), not fold to LoadConst
        let has_add_imm = chunk.code.iter().any(|op| op.opcode() == Op::ADD_IMM);
        assert!(has_add_imm, "Expected Op::AddImm for (+ x 1)");
        // Should NOT have LoadConst as first instruction (that would mean folding)
        // First instruction should be GetGlobal for 'x'
        let first_is_get_global = chunk.code[0].opcode() == Op::GET_GLOBAL;
        assert!(first_is_get_global, "First op should be GetGlobal for variable x");
    }

    #[test]
    fn test_constant_folding_division() {
        let chunk = compile_str("(/ 10 2)").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::float(5.0));
    }

    #[test]
    fn test_constant_folding_mod() {
        let chunk = compile_str("(mod 17 5)").unwrap();
        assert_eq!(chunk.code[0].opcode(), Op::LOAD_CONST);
        assert_eq!(chunk.constants[0], Value::int(2));
    }

    #[test]
    fn test_pure_function_folding() {
        use crate::parser::parse_all;

        // Define a pure function and call it with constants
        let exprs = parse_all("(def square (fn (n) (* n n))) (square 5)").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        // The call (square 5) should be folded to 25
        // Look for LoadConst 25 in the chunk
        let has_25 = chunk.constants.iter().any(|c| *c == Value::int(25));
        assert!(has_25, "Expected constant 25 from folding (square 5)");

        // Should NOT have a function call for (square 5)
        // (there may be a call for def thougheverbeit, so just check we have the constant)
    }

    #[test]
    fn test_pure_function_nested() {
        use crate::parser::parse_all;

        // Define two pure functions
        let exprs = parse_all("(def double (fn (x) (* x 2))) (def quad (fn (x) (double (double x)))) (quad 3)").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        // (quad 3) = (double (double 3)) = (double 6) = 12
        let has_12 = chunk.constants.iter().any(|c| *c == Value::int(12));
        assert!(has_12, "Expected constant 12 from folding (quad 3)");
    }

    #[test]
    fn test_impure_function_not_folded() {
        use crate::parser::parse_all;

        // A function that calls println is not pure
        let exprs = parse_all("(def greet (fn (x) (println x))) (greet 5)").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        // Should have a function call (not folded)
        let has_call = chunk.code.iter().any(|op| {
            let opc = op.opcode();
            opc == Op::CALL || opc == Op::TAIL_CALL || opc == Op::CALL_GLOBAL || opc == Op::TAIL_CALL_GLOBAL
        });
        assert!(has_call, "Impure function should not be folded");
    }

    #[test]
    fn test_loop_optimization_self_recursive() {
        use crate::parser::parse_all;

        // Define a self-recursive tail-call function (sum)
        let exprs = parse_all("(def sum (fn (n acc) (if (<= n 0) acc (sum (- n 1) (+ acc n)))))").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        // The function body should be in the first proto
        let sum_proto = &chunk.protos[0];

        // Should have a JUMP instruction (for the loop-back)
        let has_jump = sum_proto.code.iter().any(|op| op.opcode() == Op::JUMP);
        assert!(has_jump, "Self-recursive tail call should be compiled as JUMP (loop)");

        // Should NOT have TAIL_CALL_GLOBAL (that would mean no loop optimization)
        let has_tail_call = sum_proto.code.iter().any(|op| op.opcode() == Op::TAIL_CALL_GLOBAL);
        assert!(!has_tail_call, "Self-recursive function should not use TAIL_CALL_GLOBAL");
    }

    #[test]
    fn test_loop_optimization_non_recursive_unchanged() {
        use crate::parser::parse_all;

        // A non-recursive function should not get loop optimization
        let exprs = parse_all("(def square (fn (n) (* n n)))").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        let square_proto = &chunk.protos[0];

        // Should NOT have a JUMP (no loop needed)
        let has_jump_backward = square_proto.code.iter().any(|op| {
            op.opcode() == Op::JUMP && op.sbx() < 0
        });
        assert!(!has_jump_backward, "Non-recursive function should not have backward JUMP");
    }

    #[test]
    fn test_loop_optimization_mutual_recursion_unchanged() {
        use crate::parser::parse_all;

        // Mutual recursion (even/odd) should NOT get loop optimization
        // because the tail call is to a different function
        let exprs = parse_all("(def is-even (fn (n) (if (<= n 0) true (is-odd (- n 1))))) (def is-odd (fn (n) (if (<= n 0) false (is-even (- n 1)))))").unwrap();
        let chunk = Compiler::compile_all(&exprs).unwrap();

        let is_even_proto = &chunk.protos[0];

        // Should have TAIL_CALL_GLOBAL (no self-recursion loop optimization)
        let has_tail_call = is_even_proto.code.iter().any(|op| op.opcode() == Op::TAIL_CALL_GLOBAL);
        assert!(has_tail_call, "Mutual recursion should use TAIL_CALL_GLOBAL, not JUMP");
    }
}
