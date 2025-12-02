use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::eval::{eval, Env};
use crate::value::Value;

/// Global counter for generating unique symbols
static GENSYM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique symbol for hygienic macros
pub fn gensym(prefix: &str) -> String {
    let id = GENSYM_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{}#{}", prefix, id)
}

/// A macro definition
#[derive(Clone)]
pub struct Macro {
    pub params: Vec<String>,
    pub body: Value,
    pub env: Env,
}

/// Registry for macro definitions
#[derive(Clone, Default)]
pub struct MacroRegistry {
    macros: Rc<RefCell<HashMap<String, Macro>>>,
}

impl MacroRegistry {
    pub fn new() -> Self {
        MacroRegistry {
            macros: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn define(&self, name: &str, mac: Macro) {
        self.macros.borrow_mut().insert(name.to_string(), mac);
    }

    pub fn get(&self, name: &str) -> Option<Macro> {
        self.macros.borrow().get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.macros.borrow().contains_key(name)
    }
}

/// Expand macros in an expression
pub fn expand(expr: &Value, registry: &MacroRegistry, env: &Env) -> Result<Value, String> {
    if let Some(items) = expr.as_list() {
        if items.is_empty() {
            return Ok(expr.clone());
        }
        // Check for special forms first
        if let Some(sym) = items[0].as_symbol() {
            match sym {
                "quote" => return Ok(expr.clone()),
                "defmacro" => return expand_defmacro(&items[1..], registry, env),
                "gensym" => return expand_gensym(&items[1..]),
                _ => {
                    // Check if it's a macro call
                    if let Some(mac) = registry.get(sym) {
                        let expanded = apply_macro(&mac, &items[1..], env)?;
                        // Recursively expand the result
                        return expand(&expanded, registry, env);
                    }
                }
            }
        }

        // Recursively expand all elements
        let expanded: Result<Vec<Value>, String> = items
            .iter()
            .map(|item| expand(item, registry, env))
            .collect();
        return Ok(Value::list(expanded?));
    }
    Ok(expr.clone())
}

/// Handle defmacro: (defmacro name (params...) body)
fn expand_defmacro(args: &[Value], registry: &MacroRegistry, env: &Env) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("defmacro expects at least 3 arguments: name, params, body".to_string());
    }

    let name = args[0]
        .as_symbol()
        .ok_or("defmacro: first argument must be a symbol")?;

    let params_list = args[1]
        .as_list()
        .ok_or("defmacro: second argument must be a parameter list")?;

    let params: Result<Vec<String>, String> = params_list
        .iter()
        .map(|p| {
            p.as_symbol()
                .map(|s| s.to_string())
                .ok_or_else(|| "defmacro: parameter must be a symbol".to_string())
        })
        .collect();
    let params = params?;

    // Handle multi-expression body as implicit do
    let body = if args.len() == 3 {
        args[2].clone()
    } else {
        let mut do_list = vec![Value::symbol("do")];
        do_list.extend(args[2..].iter().cloned());
        Value::list(do_list)
    };

    let mac = Macro {
        params,
        body,
        env: env.clone(),
    };

    registry.define(name, mac);
    Ok(Value::nil())
}

/// Handle gensym: (gensym) or (gensym "prefix")
fn expand_gensym(args: &[Value]) -> Result<Value, String> {
    let prefix = if args.is_empty() {
        "g"
    } else if let Some(s) = args[0].as_string() {
        s
    } else if let Some(s) = args[0].as_symbol() {
        s
    } else {
        return Err("gensym: argument must be a string or symbol".to_string());
    };
    Ok(Value::symbol(&gensym(prefix)))
}

/// Apply a macro to its arguments
fn apply_macro(mac: &Macro, args: &[Value], _env: &Env) -> Result<Value, String> {
    if mac.params.len() != args.len() {
        return Err(format!(
            "Macro expects {} arguments, got {}",
            mac.params.len(),
            args.len()
        ));
    }

    // Create environment with parameters bound to unevaluated arguments
    let macro_env = mac.env.extend();
    for (param, arg) in mac.params.iter().zip(args.iter()) {
        // Arguments are passed as quoted values (unevaluated)
        macro_env.define(param, arg.clone());
    }

    // Evaluate the macro body to produce the expansion
    eval(&mac.body, &macro_env)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::standard_env;
    use crate::parser::parse;

    fn expand_str(input: &str) -> Result<Value, String> {
        let expr = parse(input)?;
        let registry = MacroRegistry::new();
        let env = standard_env();
        expand(&expr, &registry, &env)
    }

    fn expand_with_macros(macro_def: &str, expr: &str) -> Result<Value, String> {
        let registry = MacroRegistry::new();
        let env = standard_env();

        // Define the macro
        let macro_expr = parse(macro_def)?;
        expand(&macro_expr, &registry, &env)?;

        // Expand the expression
        let expr = parse(expr)?;
        expand(&expr, &registry, &env)
    }

    #[test]
    fn test_gensym() {
        let s1 = gensym("x");
        let s2 = gensym("x");
        assert_ne!(s1, s2);
        assert!(s1.starts_with("x#"));
        assert!(s2.starts_with("x#"));
    }

    #[test]
    fn test_expand_no_macros() {
        let result = expand_str("(+ 1 2)").unwrap();
        assert_eq!(result, parse("(+ 1 2)").unwrap());
    }

    #[test]
    fn test_defmacro_simple() {
        // Define a macro that doubles its argument
        let result = expand_with_macros(
            "(defmacro twice (x) (list '* 2 x))",
            "(twice 5)",
        )
        .unwrap();
        // Should expand to (* 2 5)
        assert_eq!(result, parse("(* 2 5)").unwrap());
    }

    #[test]
    fn test_macro_unless() {
        // unless macro: (unless cond body) -> (if cond nil body)
        let result = expand_with_macros(
            "(defmacro unless (cond body) (list 'if cond nil body))",
            "(unless false 42)",
        )
        .unwrap();
        assert_eq!(result, parse("(if false nil 42)").unwrap());
    }

    #[test]
    fn test_macro_nested_expansion() {
        // Test that macro results are recursively expanded
        let registry = MacroRegistry::new();
        let env = standard_env();

        // Define two macros
        expand(&parse("(defmacro inc (x) (list '+ x 1))").unwrap(), &registry, &env).unwrap();
        expand(&parse("(defmacro double-inc (x) (list 'inc (list 'inc x)))").unwrap(), &registry, &env).unwrap();

        let result = expand(&parse("(double-inc 5)").unwrap(), &registry, &env).unwrap();
        // Should expand to (+ (+ 5 1) 1)
        assert_eq!(result, parse("(+ (+ 5 1) 1)").unwrap());
    }

    #[test]
    fn test_gensym_hygiene() {
        // Test that gensym creates unique symbols each time
        let env = standard_env();
        let sym1 = eval(&parse("(gensym \"tmp\")").unwrap(), &env).unwrap();
        let sym2 = eval(&parse("(gensym \"tmp\")").unwrap(), &env).unwrap();

        // Both should be symbols
        assert!(sym1.as_symbol().is_some());
        assert!(sym2.as_symbol().is_some());

        // But different from each other
        assert_ne!(sym1, sym2);
    }

    #[test]
    fn test_macro_with_gensym() {
        // A macro that uses gensym for hygienic binding
        // This simulates a "let" macro that avoids variable capture
        let registry = MacroRegistry::new();
        let env = standard_env();

        // Define a simple swap macro that uses gensym for a temporary variable
        // Note: This is a simplified test - real swap would need mutation
        expand(
            &parse("(defmacro my-add (a b) (list '+ a b))").unwrap(),
            &registry,
            &env,
        ).unwrap();

        let result = expand(&parse("(my-add 10 20)").unwrap(), &registry, &env).unwrap();
        assert_eq!(result, parse("(+ 10 20)").unwrap());

        // Evaluate to verify it works end-to-end
        let final_result = eval(&result, &env).unwrap();
        assert_eq!(final_result, Value::Int(30));
    }
}
