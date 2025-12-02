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

/// Evaluate an expression in an environment
pub fn eval(expr: &Value, env: &Env) -> Result<Value, String> {
    match expr {
        // Self-evaluating forms
        Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::String(_) => {
            Ok(expr.clone())
        }

        // Functions evaluate to themselves
        Value::Function(_) | Value::NativeFunction(_) => Ok(expr.clone()),

        // Symbol lookup
        Value::Symbol(name) => env
            .get(name)
            .ok_or_else(|| format!("Undefined variable: {}", name)),

        // List: function call or special form
        Value::List(items) => {
            if items.is_empty() {
                return Ok(Value::list(vec![])); // () evaluates to ()
            }

            let first = &items[0];

            // Check for special forms
            if let Some(sym) = first.as_symbol() {
                match sym {
                    "quote" => return eval_quote(&items[1..]),
                    "if" => return eval_if(&items[1..], env),
                    "def" => return eval_def(&items[1..], env),
                    "let" => return eval_let(&items[1..], env),
                    "fn" => return eval_fn(&items[1..], env),
                    "do" => return eval_do(&items[1..], env),
                    _ => {}
                }
            }

            // Regular function call
            let func = eval(first, env)?;
            let args: Result<Vec<Value>, String> =
                items[1..].iter().map(|arg| eval(arg, env)).collect();
            let args = args?;

            apply(&func, &args)
        }
    }
}

/// Apply a function to arguments
fn apply(func: &Value, args: &[Value]) -> Result<Value, String> {
    match func {
        Value::Function(f) => {
            if f.params.len() != args.len() {
                return Err(format!(
                    "Expected {} arguments, got {}",
                    f.params.len(),
                    args.len()
                ));
            }

            // Create new environment extending the closure's environment
            let call_env = f.env.extend();

            // Bind parameters to arguments
            for (param, arg) in f.params.iter().zip(args.iter()) {
                call_env.define(param, arg.clone());
            }

            // Evaluate body
            eval(&f.body, &call_env)
        }

        Value::NativeFunction(nf) => (nf.func)(args),

        _ => Err(format!("Not a function: {}", func)),
    }
}

// Special form implementations

fn eval_quote(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("quote expects exactly 1 argument".to_string());
    }
    Ok(args[0].clone())
}

fn eval_if(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("if expects 2 or 3 arguments".to_string());
    }

    let cond = eval(&args[0], env)?;

    if cond.is_truthy() {
        eval(&args[1], env)
    } else if args.len() == 3 {
        eval(&args[2], env)
    } else {
        Ok(Value::Nil)
    }
}

fn eval_def(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("def expects exactly 2 arguments".to_string());
    }

    let name = args[0]
        .as_symbol()
        .ok_or("def expects a symbol as first argument")?;

    let value = eval(&args[1], env)?;
    env.define(name, value.clone());
    Ok(value)
}

fn eval_let(args: &[Value], env: &Env) -> Result<Value, String> {
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

    // Process bindings in pairs
    for chunk in bindings.chunks(2) {
        let name = chunk[0]
            .as_symbol()
            .ok_or("let binding name must be a symbol")?;
        let value = eval(&chunk[1], &let_env)?;
        let_env.define(name, value);
    }

    // Evaluate body expressions (implicit do)
    let mut result = Value::Nil;
    for body_expr in &args[1..] {
        result = eval(body_expr, &let_env)?;
    }
    Ok(result)
}

fn eval_fn(args: &[Value], env: &Env) -> Result<Value, String> {
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

    Ok(Value::Function(Rc::new(Function {
        params,
        body,
        env: env.clone(),
    })))
}

fn eval_do(args: &[Value], env: &Env) -> Result<Value, String> {
    let mut result = Value::Nil;
    for expr in args {
        result = eval(expr, env)?;
    }
    Ok(result)
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
        match arg {
            Value::Int(n) => {
                if is_float {
                    float_sum += *n as f64;
                } else {
                    int_sum += n;
                }
            }
            Value::Float(n) => {
                if !is_float {
                    is_float = true;
                    float_sum = int_sum as f64;
                }
                float_sum += n;
            }
            _ => return Err(format!("+ expects numbers, got {}", arg.type_name())),
        }
    }

    if is_float {
        Ok(Value::Float(float_sum))
    } else {
        Ok(Value::Int(int_sum))
    }
}

fn builtin_sub(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("- expects at least 1 argument".to_string());
    }

    if args.len() == 1 {
        // Unary minus
        return match &args[0] {
            Value::Int(n) => Ok(Value::Int(-n)),
            Value::Float(n) => Ok(Value::Float(-n)),
            _ => Err(format!("- expects numbers, got {}", args[0].type_name())),
        };
    }

    let mut is_float = false;
    let mut result = match &args[0] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => {
            is_float = true;
            *n
        }
        _ => return Err(format!("- expects numbers, got {}", args[0].type_name())),
    };

    for arg in &args[1..] {
        match arg {
            Value::Int(n) => result -= *n as f64,
            Value::Float(n) => {
                is_float = true;
                result -= n;
            }
            _ => return Err(format!("- expects numbers, got {}", arg.type_name())),
        }
    }

    if is_float {
        Ok(Value::Float(result))
    } else {
        Ok(Value::Int(result as i64))
    }
}

fn builtin_mul(args: &[Value]) -> Result<Value, String> {
    let mut int_prod: i64 = 1;
    let mut float_prod: f64 = 1.0;
    let mut is_float = false;

    for arg in args {
        match arg {
            Value::Int(n) => {
                if is_float {
                    float_prod *= *n as f64;
                } else {
                    int_prod *= n;
                }
            }
            Value::Float(n) => {
                if !is_float {
                    is_float = true;
                    float_prod = int_prod as f64;
                }
                float_prod *= n;
            }
            _ => return Err(format!("* expects numbers, got {}", arg.type_name())),
        }
    }

    if is_float {
        Ok(Value::Float(float_prod))
    } else {
        Ok(Value::Int(int_prod))
    }
}

fn builtin_div(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("/ expects exactly 2 arguments".to_string());
    }

    let a = match &args[0] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return Err(format!("/ expects numbers, got {}", args[0].type_name())),
    };

    let b = match &args[1] {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return Err(format!("/ expects numbers, got {}", args[1].type_name())),
    };

    if b == 0.0 {
        return Err("Division by zero".to_string());
    }

    Ok(Value::Float(a / b))
}

fn builtin_mod(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mod expects exactly 2 arguments".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err("Division by zero".to_string())
            } else {
                Ok(Value::Int(a % b))
            }
        }
        _ => Err("mod expects integers".to_string()),
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

fn builtin_lt(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("< expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(
        compare_values(&args[0], &args[1])? == std::cmp::Ordering::Less,
    ))
}

fn builtin_le(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("<= expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(
        compare_values(&args[0], &args[1])? != std::cmp::Ordering::Greater,
    ))
}

fn builtin_gt(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("> expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(
        compare_values(&args[0], &args[1])? == std::cmp::Ordering::Greater,
    ))
}

fn builtin_ge(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(">= expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(
        compare_values(&args[0], &args[1])? != std::cmp::Ordering::Less,
    ))
}

fn builtin_eq(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("= expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(args[0] == args[1]))
}

fn builtin_ne(args: &[Value]) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("!= expects exactly 2 arguments".to_string());
    }
    Ok(Value::Bool(args[0] != args[1]))
}

fn builtin_not(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("not expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(!args[0].is_truthy()))
}

fn builtin_is_nil(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nil? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::Nil)))
}

fn builtin_is_int(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("int? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::Int(_))))
}

fn builtin_is_float(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("float? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::Float(_))))
}

fn builtin_is_string(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("string? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::String(_))))
}

fn builtin_is_list(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("list? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::List(_))))
}

fn builtin_is_fn(args: &[Value]) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fn? expects exactly 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(
        args[0],
        Value::Function(_) | Value::NativeFunction(_)
    )))
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
    match &args[0] {
        Value::List(items) => Ok(Value::Int(items.len() as i64)),
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        _ => Err(format!(
            "length expects list or string, got {}",
            args[0].type_name()
        )),
    }
}

fn builtin_print(args: &[Value]) -> Result<Value, String> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        match arg {
            Value::String(s) => print!("{}", s), // Print strings without quotes
            other => print!("{}", other),
        }
    }
    Ok(Value::Nil)
}

fn builtin_println(args: &[Value]) -> Result<Value, String> {
    builtin_print(args)?;
    println!();
    Ok(Value::Nil)
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
        assert_eq!(eval_str("42").unwrap(), Value::Int(42));
        assert_eq!(eval_str("3.14").unwrap(), Value::Float(3.14));
        assert_eq!(eval_str("true").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("nil").unwrap(), Value::Nil);
        assert_eq!(eval_str("\"hello\"").unwrap(), Value::string("hello"));
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), Value::Int(3));
        assert_eq!(eval_str("(+ 1 2 3)").unwrap(), Value::Int(6));
        assert_eq!(eval_str("(- 5 3)").unwrap(), Value::Int(2));
        assert_eq!(eval_str("(* 4 5)").unwrap(), Value::Int(20));
        assert_eq!(eval_str("(/ 10 4)").unwrap(), Value::Float(2.5));
        assert_eq!(eval_str("(mod 10 3)").unwrap(), Value::Int(1));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_str("(< 1 2)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(< 2 1)").unwrap(), Value::Bool(false));
        assert_eq!(eval_str("(<= 2 2)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(> 3 2)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(>= 2 2)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(= 1 1)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(!= 1 2)").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_if() {
        assert_eq!(eval_str("(if true 1 2)").unwrap(), Value::Int(1));
        assert_eq!(eval_str("(if false 1 2)").unwrap(), Value::Int(2));
        assert_eq!(eval_str("(if nil 1 2)").unwrap(), Value::Int(2));
        assert_eq!(eval_str("(if 0 1 2)").unwrap(), Value::Int(1)); // 0 is truthy!
        assert_eq!(eval_str("(if false 1)").unwrap(), Value::Nil);
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
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );
    }

    #[test]
    fn test_eval_def() {
        let env = standard_env();
        let expr = parse("(def x 42)").unwrap();
        eval(&expr, &env).unwrap();
        assert_eq!(env.get("x").unwrap(), Value::Int(42));
    }

    #[test]
    fn test_eval_let() {
        assert_eq!(eval_str("(let (x 10) x)").unwrap(), Value::Int(10));
        assert_eq!(eval_str("(let (x 10 y 20) (+ x y))").unwrap(), Value::Int(30));
        // Let bindings are sequential
        assert_eq!(
            eval_str("(let (x 10 y (+ x 5)) y)").unwrap(),
            Value::Int(15)
        );
    }

    #[test]
    fn test_eval_fn() {
        assert_eq!(
            eval_str("((fn (x) (* x x)) 5)").unwrap(),
            Value::Int(25)
        );
        assert_eq!(
            eval_str("((fn (x y) (+ x y)) 3 4)").unwrap(),
            Value::Int(7)
        );
    }

    #[test]
    fn test_eval_do() {
        assert_eq!(eval_str("(do 1 2 3)").unwrap(), Value::Int(3));
    }

    #[test]
    fn test_eval_nested() {
        assert_eq!(eval_str("(+ (* 2 3) 4)").unwrap(), Value::Int(10));
        assert_eq!(
            eval_str("(if (< 1 2) (+ 3 4) (- 5 6))").unwrap(),
            Value::Int(7)
        );
    }

    #[test]
    fn test_eval_list_ops() {
        assert_eq!(
            eval_str("(list 1 2 3)").unwrap(),
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );
        assert_eq!(eval_str("(car (list 1 2 3))").unwrap(), Value::Int(1));
        assert_eq!(
            eval_str("(cdr (list 1 2 3))").unwrap(),
            Value::list(vec![Value::Int(2), Value::Int(3)])
        );
        assert_eq!(
            eval_str("(cons 0 (list 1 2))").unwrap(),
            Value::list(vec![Value::Int(0), Value::Int(1), Value::Int(2)])
        );
        assert_eq!(eval_str("(length (list 1 2 3))").unwrap(), Value::Int(3));
    }

    #[test]
    fn test_milestone_square() {
        // This is the Phase 1 milestone test
        let env = standard_env();

        // (def x 10)
        let expr = parse("(def x 10)").unwrap();
        eval(&expr, &env).unwrap();

        // (def square (fn (n) (* n n)))
        let expr = parse("(def square (fn (n) (* n n)))").unwrap();
        eval(&expr, &env).unwrap();

        // (square x) => 100
        let expr = parse("(square x)").unwrap();
        let result = eval(&expr, &env).unwrap();
        assert_eq!(result, Value::Int(100));
    }

    #[test]
    fn test_recursion() {
        // Test that recursion works
        let env = standard_env();

        // Define factorial
        let expr = parse(
            "(def factorial (fn (n) (if (<= n 1) 1 (* n (factorial (- n 1))))))",
        )
        .unwrap();
        eval(&expr, &env).unwrap();

        // factorial(5) = 120
        let expr = parse("(factorial 5)").unwrap();
        let result = eval(&expr, &env).unwrap();
        assert_eq!(result, Value::Int(120));
    }

    #[test]
    fn test_closure() {
        // Test that closures capture their environment
        let env = standard_env();

        // (def make-adder (fn (x) (fn (y) (+ x y))))
        let expr = parse("(def make-adder (fn (x) (fn (y) (+ x y))))").unwrap();
        eval(&expr, &env).unwrap();

        // (def add5 (make-adder 5))
        let expr = parse("(def add5 (make-adder 5))").unwrap();
        eval(&expr, &env).unwrap();

        // (add5 10) => 15
        let expr = parse("(add5 10)").unwrap();
        let result = eval(&expr, &env).unwrap();
        assert_eq!(result, Value::Int(15));
    }
}
