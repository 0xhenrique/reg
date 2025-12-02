use crate::value::{Function, NativeFunction, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Environment: a chain of scopes for variable lookup
#[derive(Clone, Debug)]
pub struct Env {
    bindings: Rc<RefCell<HashMap<String, Value>>>,
    parent: Option<Box<Env>>,
}

impl Env {
    /// Create a new empty environment
    pub fn new() -> Self {
        Env {
            bindings: Rc::new(RefCell::new(HashMap::new())),
            parent: None,
        }
    }

    /// Create a child environment (for function calls, let bindings)
    pub fn extend(&self) -> Self {
        Env {
            bindings: Rc::new(RefCell::new(HashMap::new())),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Define a new binding in the current scope
    pub fn define(&self, name: &str, value: Value) {
        self.bindings
            .borrow_mut()
            .insert(name.to_string(), value);
    }

    /// Look up a variable, searching parent scopes
    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.bindings.borrow().get(name) {
            Some(value.clone())
        } else if let Some(parent) = &self.parent {
            parent.get(name)
        } else {
            None
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

/// Trampoline: either a final value or a tail call to execute
enum Trampoline {
    Value(Value),
    TailCall {
        func: Rc<Function>,
        args: Vec<Value>,
    },
}

/// Evaluate an expression (public API - handles trampolining)
pub fn eval(expr: &Value, env: &Env) -> Result<Value, String> {
    let mut result = eval_inner(expr, env, false)?;

    // Trampoline loop - handles tail calls without growing Rust stack
    loop {
        match result {
            Trampoline::Value(v) => return Ok(v),
            Trampoline::TailCall { func, args } => {
                // Set up call environment
                let call_env = func.env.extend();
                for (param, arg) in func.params.iter().zip(args.iter()) {
                    call_env.define(param, arg.clone());
                }
                // Evaluate body in tail position (it's always the last thing)
                result = eval_inner(&func.body, &call_env, true)?;
            }
        }
    }
}

/// Inner eval that tracks tail position and returns Trampoline
fn eval_inner(expr: &Value, env: &Env, tail_pos: bool) -> Result<Trampoline, String> {
    // Self-evaluating forms
    if expr.is_nil() || expr.is_bool() || expr.is_int() || expr.is_float() || expr.as_string().is_some() {
        return Ok(Trampoline::Value(expr.clone()));
    }

    // Functions evaluate to themselves
    if expr.as_function().is_some() || expr.as_native_function().is_some() || expr.as_compiled_function().is_some() {
        return Ok(Trampoline::Value(expr.clone()));
    }

    // Symbol lookup
    if let Some(name) = expr.as_symbol() {
        return env
            .get(name)
            .map(Trampoline::Value)
            .ok_or_else(|| format!("Undefined variable: {}", name));
    }

    // List: function call or special form
    if let Some(items) = expr.as_list() {
        if items.is_empty() {
            return Ok(Trampoline::Value(Value::list(vec![]))); // () evaluates to ()
        }

        let first = &items[0];

        // Check for special forms
        if let Some(sym) = first.as_symbol() {
            match sym {
                "quote" => return eval_quote(&items[1..]),
                "if" => return eval_if(&items[1..], env, tail_pos),
                "def" => return eval_def(&items[1..], env),
                "let" => return eval_let(&items[1..], env, tail_pos),
                "fn" => return eval_fn(&items[1..], env),
                "do" => return eval_do(&items[1..], env, tail_pos),
                _ => {}
            }
        }

        // Regular function call - evaluate function and arguments
        let func = eval(first, env)?;
        let args: Result<Vec<Value>, String> =
            items[1..].iter().map(|arg| eval(arg, env)).collect();
        let args = args?;

        return apply(&func, args, tail_pos);
    }

    Err("Unknown expression type".to_string())
}

/// Apply a function to arguments
fn apply(func: &Value, args: Vec<Value>, tail_pos: bool) -> Result<Trampoline, String> {
    if let Some(f) = func.as_function() {
        if f.params.len() != args.len() {
            return Err(format!(
                "Expected {} arguments, got {}",
                f.params.len(),
                args.len()
            ));
        }

        if tail_pos {
            // In tail position: return thunk for trampoline
            Ok(Trampoline::TailCall {
                func: f.clone(),
                args,
            })
        } else {
            // Not in tail position: evaluate now (will trampoline internally)
            let call_env = f.env.extend();
            for (param, arg) in f.params.iter().zip(args.iter()) {
                call_env.define(param, arg.clone());
            }
            eval_inner(&f.body, &call_env, true)
        }
    } else if let Some(nf) = func.as_native_function() {
        let result = (nf.func)(&args)?;
        Ok(Trampoline::Value(result))
    } else {
        Err(format!("Not a function: {}", func))
    }
}

// Special form implementations

fn eval_quote(args: &[Value]) -> Result<Trampoline, String> {
    if args.len() != 1 {
        return Err("quote expects exactly 1 argument".to_string());
    }
    Ok(Trampoline::Value(args[0].clone()))
}

fn eval_if(args: &[Value], env: &Env, tail_pos: bool) -> Result<Trampoline, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("if expects 2 or 3 arguments".to_string());
    }

    // Condition is NOT in tail position
    let cond = eval(&args[0], env)?;

    // Branches ARE in tail position (if the if itself is)
    if cond.is_truthy() {
        eval_inner(&args[1], env, tail_pos)
    } else if args.len() == 3 {
        eval_inner(&args[2], env, tail_pos)
    } else {
        Ok(Trampoline::Value(Value::Nil))
    }
}

fn eval_def(args: &[Value], env: &Env) -> Result<Trampoline, String> {
    if args.len() != 2 {
        return Err("def expects exactly 2 arguments".to_string());
    }

    let name = args[0]
        .as_symbol()
        .ok_or("def expects a symbol as first argument")?;

    // def value is NOT in tail position
    let value = eval(&args[1], env)?;
    env.define(name, value.clone());
    Ok(Trampoline::Value(value))
}

fn eval_let(args: &[Value], env: &Env, tail_pos: bool) -> Result<Trampoline, String> {
    if args.len() < 2 {
        return Err("let expects at least 2 arguments (bindings and body)".to_string());
    }

    let bindings = args[0]
        .as_list()
        .ok_or("let expects a list of bindings")?;

    if bindings.len() % 2 != 0 {
        return Err("let bindings must be pairs".to_string());
    }

    let let_env = env.extend();

    // Process bindings - NOT in tail position
    for chunk in bindings.chunks(2) {
        let name = chunk[0]
            .as_symbol()
            .ok_or("let binding name must be a symbol")?;
        let value = eval(&chunk[1], &let_env)?;
        let_env.define(name, value);
    }

    // Evaluate body expressions - only LAST is in tail position
    let body = &args[1..];
    if body.is_empty() {
        return Ok(Trampoline::Value(Value::Nil));
    }

    // All but last: not in tail position
    for expr in &body[..body.len() - 1] {
        eval(expr, &let_env)?;
    }

    // Last expression: in tail position (if let itself is)
    eval_inner(&body[body.len() - 1], &let_env, tail_pos)
}

fn eval_fn(args: &[Value], env: &Env) -> Result<Trampoline, String> {
    if args.len() < 2 {
        return Err("fn expects at least 2 arguments (params and body)".to_string());
    }

    let params_list = args[0]
        .as_list()
        .ok_or("fn expects a list of parameters")?;

    let params: Result<Vec<String>, String> = params_list
        .iter()
        .map(|p| {
            p.as_symbol()
                .map(|s| s.to_string())
                .ok_or_else(|| "Function parameter must be a symbol".to_string())
        })
        .collect();
    let params = params?;

    // If there are multiple body expressions, wrap in (do ...)
    let body = if args.len() == 2 {
        args[1].clone()
    } else {
        let mut do_list = vec![Value::symbol("do")];
        do_list.extend(args[1..].iter().cloned());
        Value::list(do_list)
    };

    Ok(Trampoline::Value(Value::Function(Rc::new(Function {
        params,
        body,
        env: env.clone(),
    }))))
}

fn eval_do(args: &[Value], env: &Env, tail_pos: bool) -> Result<Trampoline, String> {
    if args.is_empty() {
        return Ok(Trampoline::Value(Value::Nil));
    }

    // All but last: not in tail position
    for expr in &args[..args.len() - 1] {
        eval(expr, env)?;
    }

    // Last expression: in tail position (if do itself is)
    eval_inner(&args[args.len() - 1], env, tail_pos)
}

/// Create an environment with standard built-in functions
pub fn standard_env() -> Env {
    let env = Env::new();

    // Arithmetic
    env.define("+", native_fn("+", builtin_add));
    env.define("-", native_fn("-", builtin_sub));
    env.define("*", native_fn("*", builtin_mul));
    env.define("/", native_fn("/", builtin_div));
    env.define("mod", native_fn("mod", builtin_mod));

    // Comparison
    env.define("<", native_fn("<", builtin_lt));
    env.define("<=", native_fn("<=", builtin_le));
    env.define(">", native_fn(">", builtin_gt));
    env.define(">=", native_fn(">=", builtin_ge));
    env.define("=", native_fn("=", builtin_eq));
    env.define("!=", native_fn("!=", builtin_ne));

    // Logic
    env.define("not", native_fn("not", builtin_not));

    // Type predicates
    env.define("nil?", native_fn("nil?", builtin_is_nil));
    env.define("int?", native_fn("int?", builtin_is_int));
    env.define("float?", native_fn("float?", builtin_is_float));
    env.define("string?", native_fn("string?", builtin_is_string));
    env.define("list?", native_fn("list?", builtin_is_list));
    env.define("fn?", native_fn("fn?", builtin_is_fn));

    // List operations
    env.define("list", native_fn("list", builtin_list));
    env.define("cons", native_fn("cons", builtin_cons));
    env.define("car", native_fn("car", builtin_car));
    env.define("cdr", native_fn("cdr", builtin_cdr));
    env.define("length", native_fn("length", builtin_length));

    // I/O
    env.define("print", native_fn("print", builtin_print));
    env.define("println", native_fn("println", builtin_println));

    // Symbol operations (useful for macros)
    env.define("symbol?", native_fn("symbol?", builtin_is_symbol));
    env.define("gensym", native_fn("gensym", builtin_gensym));

    env
}

fn native_fn(name: &str, func: fn(&[Value]) -> Result<Value, String>) -> Value {
    Value::NativeFunction(Rc::new(NativeFunction {
        name: name.to_string(),
        func,
    }))
}

// Built-in function implementations

fn builtin_add(args: &[Value]) -> Result<Value, String> {
    let mut int_sum: i64 = 0;
    let mut float_sum: f64 = 0.0;
    let mut is_float = false;

    for arg in args {
        if let Some(n) = arg.as_int() {
            if is_float {
                float_sum += n as f64;
            } else {
                int_sum += n;
            }
        } else if let Some(n) = arg.as_float() {
            if !is_float {
                is_float = true;
                float_sum = int_sum as f64;
            }
            float_sum += n;
        } else {
            return Err(format!("+ expects numbers, got {}", arg.type_name()));
        }
    }

    if is_float {
        Ok(Value::float(float_sum))
    } else {
        Ok(Value::int(int_sum))
    }
}

fn builtin_sub(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("- expects at least 1 argument".to_string());
    }

    if args.len() == 1 {
        // Unary minus
        if let Some(n) = args[0].as_int() {
            return Ok(Value::int(-n));
        } else if let Some(n) = args[0].as_float() {
            return Ok(Value::float(-n));
        } else {
            return Err(format!("- expects numbers, got {}", args[0].type_name()));
        }
    }

    let mut is_float = args[0].is_float();
    let mut result = if let Some(n) = args[0].as_int() {
        n as f64
    } else if let Some(n) = args[0].as_float() {
        n
    } else {
        return Err(format!("- expects numbers, got {}", args[0].type_name()));
    };

    for arg in &args[1..] {
        if let Some(n) = arg.as_int() {
            result -= n as f64;
        } else if let Some(n) = arg.as_float() {
            is_float = true;
            result -= n;
        } else {
            return Err(format!("- expects numbers, got {}", arg.type_name()));
        }
    }

    if is_float {
        Ok(Value::float(result))
    } else {
        Ok(Value::int(result as i64))
    }
}

fn builtin_mul(args: &[Value]) -> Result<Value, String> {
    let mut int_prod: i64 = 1;
    let mut float_prod: f64 = 1.0;
    let mut is_float = false;

    for arg in args {
        if let Some(n) = arg.as_int() {
            if is_float {
                float_prod *= n as f64;
            } else {
                int_prod *= n;
            }
        } else if let Some(n) = arg.as_float() {
            if !is_float {
                is_float = true;
                float_prod = int_prod as f64;
            }
            float_prod *= n;
        } else {
            return Err(format!("* expects numbers, got {}", arg.type_name()));
        }
    }

    if is_float {
        Ok(Value::float(float_prod))
    } else {
        Ok(Value::int(int_prod))
    }
}

fn builtin_div(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("/ expects exactly 2 arguments".to_string());
    }

    let a = if let Some(n) = args[0].as_int() {
        n as f64
    } else if let Some(n) = args[0].as_float() {
        n
    } else {
        return Err(format!("/ expects numbers, got {}", args[0].type_name()));
    };

    let b = if let Some(n) = args[1].as_int() {
        n as f64
    } else if let Some(n) = args[1].as_float() {
        n
    } else {
        return Err(format!("/ expects numbers, got {}", args[1].type_name()));
    };

    if b == 0.0 {
        return Err("Division by zero".to_string());
    }

    Ok(Value::float(a / b))
}

fn builtin_mod(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mod expects exactly 2 arguments".to_string());
    }

    if let (Some(a), Some(b)) = (args[0].as_int(), args[1].as_int()) {
        if b == 0 {
            Err("Division by zero".to_string())
        } else {
            Ok(Value::int(a % b))
        }
    } else {
        Err("mod expects integers".to_string())
    }
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

fn builtin_lt(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("< expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(
        compare_values(&args[0], &args[1])? == std::cmp::Ordering::Less,
    ))
}

fn builtin_le(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("<= expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(
        compare_values(&args[0], &args[1])? != std::cmp::Ordering::Greater,
    ))
}

fn builtin_gt(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("> expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(
        compare_values(&args[0], &args[1])? == std::cmp::Ordering::Greater,
    ))
}

fn builtin_ge(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(">= expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(
        compare_values(&args[0], &args[1])? != std::cmp::Ordering::Less,
    ))
}

fn builtin_eq(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("= expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(args[0] == args[1]))
}

fn builtin_ne(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("!= expects exactly 2 arguments".to_string());
    }
    Ok(Value::bool(args[0] != args[1]))
}

fn builtin_not(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("not expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(!args[0].is_truthy()))
}

fn builtin_is_nil(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nil? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].is_nil()))
}

fn builtin_is_int(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("int? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].is_int()))
}

fn builtin_is_float(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("float? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].is_float()))
}

fn builtin_is_string(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("string? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].as_string().is_some()))
}

fn builtin_is_list(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("list? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].as_list().is_some()))
}

fn builtin_is_fn(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fn? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(
        args[0].as_function().is_some() || args[0].as_native_function().is_some()
    ))
}

fn builtin_list(args: &[Value]) -> Result<Value, String> {
    Ok(Value::list(args.to_vec()))
}

fn builtin_cons(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("cons expects exactly 2 arguments".to_string());
    }
    let head = args[0].clone();
    let tail = args[1]
        .as_list()
        .ok_or("cons expects a list as second argument")?;

    let mut new_list = vec![head];
    new_list.extend(tail.iter().cloned());
    Ok(Value::list(new_list))
}

fn builtin_car(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("car expects exactly 1 argument".to_string());
    }
    let list = args[0].as_list().ok_or("car expects a list")?;
    list.first()
        .cloned()
        .ok_or_else(|| "car on empty list".to_string())
}

fn builtin_cdr(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cdr expects exactly 1 argument".to_string());
    }
    let list = args[0].as_list().ok_or("cdr expects a list")?;
    if list.is_empty() {
        return Err("cdr on empty list".to_string());
    }
    Ok(Value::list(list[1..].to_vec()))
}

fn builtin_length(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("length expects exactly 1 argument".to_string());
    }
    if let Some(items) = args[0].as_list() {
        return Ok(Value::int(items.len() as i64));
    }
    if let Some(s) = args[0].as_string() {
        return Ok(Value::int(s.len() as i64));
    }
    Err(format!(
        "length expects list or string, got {}",
        args[0].type_name()
    ))
}

fn builtin_print(args: &[Value]) -> Result<Value, String> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        if let Some(s) = arg.as_string() {
            print!("{}", s); // Print strings without quotes
        } else {
            print!("{}", arg);
        }
    }
    Ok(Value::nil())
}

fn builtin_println(args: &[Value]) -> Result<Value, String> {
    builtin_print(args)?;
    println!();
    Ok(Value::nil())
}

fn builtin_is_symbol(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("symbol? expects exactly 1 argument".to_string());
    }
    Ok(Value::bool(args[0].as_symbol().is_some()))
}

fn builtin_gensym(args: &[Value]) -> Result<Value, String> {
    let prefix = if args.is_empty() {
        "g"
    } else if let Some(s) = args[0].as_string() {
        s
    } else if let Some(s) = args[0].as_symbol() {
        s
    } else {
        return Err("gensym: argument must be a string or symbol".to_string());
    };
    Ok(Value::symbol(&crate::macros::gensym(prefix)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn eval_str(input: &str) -> Result<Value, String> {
        let expr = parse(input)?;
        let env = standard_env();
        eval(&expr, &env)
    }

    #[test]
    fn test_eval_literals() {
        assert_eq!(eval_str("42").unwrap(), Value::int(42));
        assert_eq!(eval_str("3.14").unwrap(), Value::float(3.14));
        assert_eq!(eval_str("true").unwrap(), Value::bool(true));
        assert_eq!(eval_str("nil").unwrap(), Value::nil());
        assert_eq!(eval_str("\"hello\"").unwrap(), Value::string("hello"));
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), Value::int(3));
        assert_eq!(eval_str("(+ 1 2 3)").unwrap(), Value::int(6));
        assert_eq!(eval_str("(- 5 3)").unwrap(), Value::int(2));
        assert_eq!(eval_str("(* 4 5)").unwrap(), Value::int(20));
        assert_eq!(eval_str("(/ 10 4)").unwrap(), Value::float(2.5));
        assert_eq!(eval_str("(mod 10 3)").unwrap(), Value::int(1));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_str("(< 1 2)").unwrap(), Value::bool(true));
        assert_eq!(eval_str("(< 2 1)").unwrap(), Value::bool(false));
        assert_eq!(eval_str("(<= 2 2)").unwrap(), Value::bool(true));
        assert_eq!(eval_str("(> 3 2)").unwrap(), Value::bool(true));
        assert_eq!(eval_str("(>= 2 2)").unwrap(), Value::bool(true));
        assert_eq!(eval_str("(= 1 1)").unwrap(), Value::bool(true));
        assert_eq!(eval_str("(!= 1 2)").unwrap(), Value::bool(true));
    }

    #[test]
    fn test_eval_if() {
        assert_eq!(eval_str("(if true 1 2)").unwrap(), Value::int(1));
        assert_eq!(eval_str("(if false 1 2)").unwrap(), Value::int(2));
        assert_eq!(eval_str("(if nil 1 2)").unwrap(), Value::int(2));
        assert_eq!(eval_str("(if 0 1 2)").unwrap(), Value::int(1)); // 0 is truthy!
        assert_eq!(eval_str("(if false 1)").unwrap(), Value::nil());
    }

    #[test]
    fn test_eval_quote() {
        let result = eval_str("(quote foo)").unwrap();
        assert_eq!(result, Value::symbol("foo"));

        let result = eval_str("'foo").unwrap();
        assert_eq!(result, Value::symbol("foo"));

        let result = eval_str("'(1 2 3)").unwrap();
        assert_eq!(
            result,
            Value::list(vec![Value::int(1), Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_def() {
        let env = standard_env();
        let expr = parse("(def x 42)").unwrap();
        eval(&expr, &env).unwrap();
        assert_eq!(env.get("x").unwrap(), Value::int(42));
    }

    #[test]
    fn test_eval_let() {
        assert_eq!(eval_str("(let (x 10) x)").unwrap(), Value::int(10));
        assert_eq!(eval_str("(let (x 10 y 20) (+ x y))").unwrap(), Value::int(30));
        // Let bindings are sequential
        assert_eq!(
            eval_str("(let (x 10 y (+ x 5)) y)").unwrap(),
            Value::int(15)
        );
    }

    #[test]
    fn test_eval_fn() {
        assert_eq!(
            eval_str("((fn (x) (* x x)) 5)").unwrap(),
            Value::int(25)
        );
        assert_eq!(
            eval_str("((fn (x y) (+ x y)) 3 4)").unwrap(),
            Value::int(7)
        );
    }

    #[test]
    fn test_eval_do() {
        assert_eq!(eval_str("(do 1 2 3)").unwrap(), Value::int(3));
    }

    #[test]
    fn test_eval_nested() {
        assert_eq!(eval_str("(+ (* 2 3) 4)").unwrap(), Value::int(10));
        assert_eq!(
            eval_str("(if (< 1 2) (+ 3 4) (- 5 6))").unwrap(),
            Value::int(7)
        );
    }

    #[test]
    fn test_eval_list_ops() {
        assert_eq!(
            eval_str("(list 1 2 3)").unwrap(),
            Value::list(vec![Value::int(1), Value::int(2), Value::int(3)])
        );
        assert_eq!(eval_str("(car (list 1 2 3))").unwrap(), Value::int(1));
        assert_eq!(
            eval_str("(cdr (list 1 2 3))").unwrap(),
            Value::list(vec![Value::int(2), Value::int(3)])
        );
        assert_eq!(
            eval_str("(cons 0 (list 1 2))").unwrap(),
            Value::list(vec![Value::int(0), Value::int(1), Value::int(2)])
        );
        assert_eq!(eval_str("(length (list 1 2 3))").unwrap(), Value::int(3));
    }

    #[test]
    fn test_milestone_square() {
        let env = standard_env();
        eval(&parse("(def x 10)").unwrap(), &env).unwrap();
        eval(&parse("(def square (fn (n) (* n n)))").unwrap(), &env).unwrap();
        let result = eval(&parse("(square x)").unwrap(), &env).unwrap();
        assert_eq!(result, Value::int(100));
    }

    #[test]
    fn test_recursion() {
        let env = standard_env();
        eval(&parse("(def factorial (fn (n) (if (<= n 1) 1 (* n (factorial (- n 1))))))").unwrap(), &env).unwrap();
        let result = eval(&parse("(factorial 5)").unwrap(), &env).unwrap();
        assert_eq!(result, Value::int(120));
    }

    #[test]
    fn test_closure() {
        let env = standard_env();
        eval(&parse("(def make-adder (fn (x) (fn (y) (+ x y))))").unwrap(), &env).unwrap();
        eval(&parse("(def add5 (make-adder 5))").unwrap(), &env).unwrap();
        let result = eval(&parse("(add5 10)").unwrap(), &env).unwrap();
        assert_eq!(result, Value::int(15));
    }

    #[test]
    fn test_tail_call_sum() {
        let env = standard_env();
        eval(&parse("(def sum (fn (n acc) (if (<= n 0) acc (sum (- n 1) (+ acc n)))))").unwrap(), &env).unwrap();
        // 100k iterations - would stack overflow without TCO
        let result = eval(&parse("(sum 100000 0)").unwrap(), &env).unwrap();
        assert_eq!(result, Value::int(5000050000));
    }

    #[test]
    fn test_mutual_recursion() {
        let env = standard_env();
        eval(&parse("(def even? (fn (n) (if (= n 0) true (odd? (- n 1)))))").unwrap(), &env).unwrap();
        eval(&parse("(def odd? (fn (n) (if (= n 0) false (even? (- n 1)))))").unwrap(), &env).unwrap();
        // 10k mutual calls - would stack overflow without TCO
        assert_eq!(eval(&parse("(even? 10000)").unwrap(), &env).unwrap(), Value::bool(true));
        assert_eq!(eval(&parse("(odd? 10001)").unwrap(), &env).unwrap(), Value::bool(true));
    }
}
