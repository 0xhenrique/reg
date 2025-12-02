use crate::bytecode::{Chunk, ConstIdx, Op, Reg};
use crate::value::Value;

pub struct Compiler {
    chunk: Chunk,
    locals: Vec<String>,   // local variable names, index = register
    scope_depth: usize,
}

/// Try to evaluate a constant expression recursively
fn try_const_eval(expr: &Value) -> Option<Value> {
    match expr {
        // Literals are constants
        Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Nil | Value::String(_) => {
            Some(expr.clone())
        }
        // Try to fold function calls
        Value::List(items) if !items.is_empty() => {
            let op = items[0].as_symbol()?;
            let args = &items[1..];
            // Recursively evaluate arguments
            let const_args: Option<Vec<Value>> = args.iter().map(try_const_eval).collect();
            let const_args = const_args?;
            let const_refs: Vec<&Value> = const_args.iter().collect();
            fold_op(op, &const_refs)
        }
        _ => None,
    }
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
        match arg {
            Value::Int(n) => {
                if is_float {
                    sum_float += *n as f64;
                } else {
                    sum_int += n;
                }
            }
            Value::Float(n) => {
                if !is_float {
                    is_float = true;
                    sum_float = sum_int as f64;
                }
                sum_float += n;
            }
            _ => return None,
        }
    }

    Some(if is_float {
        Value::Float(sum_float)
    } else {
        Value::Int(sum_int)
    })
}

fn fold_sub(args: &[&Value]) -> Option<Value> {
    if args.is_empty() {
        return None;
    }
    if args.len() == 1 {
        return match args[0] {
            Value::Int(n) => Some(Value::Int(-n)),
            Value::Float(n) => Some(Value::Float(-n)),
            _ => None,
        };
    }

    let mut result = match args[0] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return None,
    };
    let mut is_float = matches!(args[0], Value::Float(_));

    for arg in &args[1..] {
        match arg {
            Value::Int(n) => result -= *n as f64,
            Value::Float(n) => {
                is_float = true;
                result -= n;
            }
            _ => return None,
        }
    }

    Some(if is_float {
        Value::Float(result)
    } else {
        Value::Int(result as i64)
    })
}

fn fold_mul(args: &[&Value]) -> Option<Value> {
    let mut prod_int: i64 = 1;
    let mut prod_float: f64 = 1.0;
    let mut is_float = false;

    for arg in args {
        match arg {
            Value::Int(n) => {
                if is_float {
                    prod_float *= *n as f64;
                } else {
                    prod_int *= n;
                }
            }
            Value::Float(n) => {
                if !is_float {
                    is_float = true;
                    prod_float = prod_int as f64;
                }
                prod_float *= n;
            }
            _ => return None,
        }
    }

    Some(if is_float {
        Value::Float(prod_float)
    } else {
        Value::Int(prod_int)
    })
}

fn fold_div(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    let a = match args[0] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return None,
    };
    let b = match args[1] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return None,
    };
    if b == 0.0 {
        return None; // Don't fold division by zero
    }
    Some(Value::Float(a / b))
}

fn fold_mod(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    match (args[0], args[1]) {
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                return None;
            }
            Some(Value::Int(a % b))
        }
        _ => None,
    }
}

fn to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(n) => Some(*n),
        _ => None,
    }
}

fn fold_cmp<F: Fn(f64, f64) -> bool>(args: &[&Value], cmp: F) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    let a = to_f64(args[0])?;
    let b = to_f64(args[1])?;
    Some(Value::Bool(cmp(a, b)))
}

fn fold_eq(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    Some(Value::Bool(args[0] == args[1]))
}

fn fold_ne(args: &[&Value]) -> Option<Value> {
    if args.len() != 2 {
        return None;
    }
    Some(Value::Bool(args[0] != args[1]))
}

fn fold_not(args: &[&Value]) -> Option<Value> {
    if args.len() != 1 {
        return None;
    }
    Some(Value::Bool(!args[0].is_truthy()))
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            chunk: Chunk::new(),
            locals: Vec::new(),
            scope_depth: 0,
        }
    }

    pub fn compile(expr: &Value) -> Result<Chunk, String> {
        let mut compiler = Compiler::new();
        let dest = compiler.alloc_reg();
        compiler.compile_expr(expr, dest, true)?;
        compiler.emit(Op::Return(dest));
        compiler.chunk.num_registers = compiler.locals.len().max(1) as u8 + 16; // extra for temps
        Ok(compiler.chunk)
    }

    pub fn compile_function(params: &[String], body: &Value) -> Result<Chunk, String> {
        let mut compiler = Compiler::new();
        compiler.chunk.num_params = params.len() as u8;

        // Parameters occupy the first registers
        for param in params {
            compiler.locals.push(param.clone());
        }

        let dest = compiler.alloc_reg();
        compiler.compile_expr(body, dest, true)?;
        compiler.emit(Op::Return(dest));
        compiler.chunk.num_registers = compiler.locals.len().max(1) as u8 + 16;
        Ok(compiler.chunk)
    }

    fn alloc_reg(&mut self) -> Reg {
        let reg = self.locals.len() as Reg;
        self.locals.push(String::new()); // placeholder
        reg
    }

    fn free_reg(&mut self) {
        if !self.locals.is_empty() && self.locals.last().map(|s| s.is_empty()).unwrap_or(false) {
            self.locals.pop();
        }
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
        match expr {
            Value::Nil => {
                self.emit(Op::LoadNil(dest));
            }
            Value::Bool(true) => {
                self.emit(Op::LoadTrue(dest));
            }
            Value::Bool(false) => {
                self.emit(Op::LoadFalse(dest));
            }
            Value::Int(_) | Value::Float(_) | Value::String(_) => {
                let idx = self.add_constant(expr.clone());
                self.emit(Op::LoadConst(dest, idx));
            }
            Value::Symbol(name) => {
                if let Some(reg) = self.resolve_local(name) {
                    if reg != dest {
                        self.emit(Op::Move(dest, reg));
                    }
                } else {
                    let idx = self.add_constant(Value::symbol(name));
                    self.emit(Op::GetGlobal(dest, idx));
                }
            }
            Value::List(items) => {
                if items.is_empty() {
                    let idx = self.add_constant(Value::list(vec![]));
                    self.emit(Op::LoadConst(dest, idx));
                    return Ok(());
                }

                let first = &items[0];
                if let Some(sym) = first.as_symbol() {
                    match sym {
                        "quote" => return self.compile_quote(&items[1..], dest),
                        "if" => return self.compile_if(&items[1..], dest, tail_pos),
                        "def" => return self.compile_def(&items[1..], dest),
                        "let" => return self.compile_let(&items[1..], dest, tail_pos),
                        "fn" => return self.compile_fn(&items[1..], dest),
                        "do" => return self.compile_do(&items[1..], dest, tail_pos),
                        _ => {}
                    }
                }

                // Function call
                self.compile_call(items, dest, tail_pos)?;
            }
            Value::Function(_) | Value::NativeFunction(_) | Value::CompiledFunction(_) => {
                return Err("Cannot compile function value directly".to_string());
            }
        }
        Ok(())
    }

    fn compile_quote(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        if args.len() != 1 {
            return Err("quote expects exactly 1 argument".to_string());
        }
        let idx = self.add_constant(args[0].clone());
        self.emit(Op::LoadConst(dest, idx));
        Ok(())
    }

    fn compile_if(&mut self, args: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if args.len() < 2 || args.len() > 3 {
            return Err("if expects 2 or 3 arguments".to_string());
        }

        // Compile condition into dest
        self.compile_expr(&args[0], dest, false)?;

        // Jump to else if false
        let jump_to_else = self.emit(Op::JumpIfFalse(dest, 0));

        // Compile then branch
        self.compile_expr(&args[1], dest, tail_pos)?;

        if args.len() == 3 {
            // Jump over else
            let jump_over_else = self.emit(Op::Jump(0));

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
            let jump_over_nil = self.emit(Op::Jump(0));

            let else_start = self.chunk.current_pos();
            self.chunk.patch_jump(jump_to_else, else_start);
            self.emit(Op::LoadNil(dest));

            let end = self.chunk.current_pos();
            self.chunk.patch_jump(jump_over_nil, end);
        }

        Ok(())
    }

    fn compile_def(&mut self, args: &[Value], dest: Reg) -> Result<(), String> {
        if args.len() != 2 {
            return Err("def expects exactly 2 arguments".to_string());
        }

        let name = args[0]
            .as_symbol()
            .ok_or("def expects a symbol as first argument")?;

        // Compile value
        self.compile_expr(&args[1], dest, false)?;

        // Store to global
        let name_idx = self.add_constant(Value::symbol(name));
        self.emit(Op::SetGlobal(name_idx, dest));

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
            self.emit(Op::LoadNil(dest));
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

        self.emit(Op::Closure(dest, proto_idx));
        Ok(())
    }

    fn compile_do(&mut self, args: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        if args.is_empty() {
            self.emit(Op::LoadNil(dest));
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

    fn compile_call(&mut self, items: &[Value], dest: Reg, tail_pos: bool) -> Result<(), String> {
        // Try constant folding for the entire call expression
        let call_expr = Value::list(items.to_vec());
        if let Some(folded) = try_const_eval(&call_expr) {
            let idx = self.add_constant(folded);
            self.emit(Op::LoadConst(dest, idx));
            return Ok(());
        }

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
            self.emit(Op::TailCall(func_reg, nargs));
            // After tail call, we still need to put something in dest for non-tail paths
            // But if it's truly a tail call, we won't reach here
            self.emit(Op::LoadNil(dest));
        } else {
            self.emit(Op::Call(dest, func_reg, nargs));
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
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert!(matches!(chunk.code[1], Op::Return(0)));
    }

    #[test]
    fn test_compile_nil() {
        let chunk = compile_str("nil").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadNil(0)));
    }

    #[test]
    fn test_compile_bool() {
        let chunk = compile_str("true").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadTrue(0)));

        let chunk = compile_str("false").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadFalse(0)));
    }

    #[test]
    fn test_compile_quote() {
        let chunk = compile_str("'foo").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
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
        assert!(matches!(chunk.code[0], Op::Closure(0, 0)));
        // Should have a proto
        assert_eq!(chunk.protos.len(), 1);
        assert_eq!(chunk.protos[0].num_params, 1);
    }

    #[test]
    fn test_compile_if() {
        let chunk = compile_str("(if true 1 2)").unwrap();
        // Should have jumps
        let has_jump = chunk.code.iter().any(|op| matches!(op, Op::Jump(_) | Op::JumpIfFalse(_, _)));
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
        // Top-level call is in tail position, so it's a TailCall
        let chunk = compile_str("(foo 1 2)").unwrap();
        let has_tail_call = chunk.code.iter().any(|op| matches!(op, Op::TailCall(_, _)));
        assert!(has_tail_call);

        // Non-tail call (if condition is not in tail position)
        let chunk = compile_str("(if (foo) 1 2)").unwrap();
        let has_call = chunk.code.iter().any(|op| matches!(op, Op::Call(_, _, _)));
        assert!(has_call);
    }

    #[test]
    fn test_constant_folding_add() {
        let chunk = compile_str("(+ 1 2 3)").unwrap();
        // Should fold to LoadConst 6, not a call
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Int(6));
        // No function calls
        let has_call = chunk.code.iter().any(|op| matches!(op, Op::Call(_, _, _) | Op::TailCall(_, _)));
        assert!(!has_call);
    }

    #[test]
    fn test_constant_folding_nested() {
        let chunk = compile_str("(* 2 (+ 3 4))").unwrap();
        // Should fold to LoadConst 14
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Int(14));
    }

    #[test]
    fn test_constant_folding_comparison() {
        let chunk = compile_str("(< 1 2)").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Bool(true));

        let chunk = compile_str("(>= 5 10)").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Bool(false));
    }

    #[test]
    fn test_constant_folding_not() {
        let chunk = compile_str("(not false)").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Bool(true));
    }

    #[test]
    fn test_no_fold_with_variables() {
        // If any arg is not a constant, don't fold
        let chunk = compile_str("(+ x 1)").unwrap();
        // Should have a call, not just LoadConst
        let has_call = chunk.code.iter().any(|op| matches!(op, Op::Call(_, _, _) | Op::TailCall(_, _)));
        assert!(has_call);
    }

    #[test]
    fn test_constant_folding_division() {
        let chunk = compile_str("(/ 10 2)").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Float(5.0));
    }

    #[test]
    fn test_constant_folding_mod() {
        let chunk = compile_str("(mod 17 5)").unwrap();
        assert!(matches!(chunk.code[0], Op::LoadConst(0, _)));
        assert_eq!(chunk.constants[0], Value::Int(2));
    }
}
