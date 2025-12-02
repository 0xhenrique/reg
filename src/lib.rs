pub mod eval;
pub mod parser;
pub mod value;

pub use eval::{eval, standard_env, Env};
pub use parser::{parse, parse_all};
pub use value::Value;

/// Convenience function to evaluate a string in a standard environment
pub fn run(input: &str) -> Result<Value, String> {
    let expr = parse(input)?;
    let env = standard_env();
    eval(&expr, &env)
}

/// Convenience function to evaluate multiple expressions
pub fn run_all(input: &str) -> Result<Value, String> {
    let exprs = parse_all(input)?;
    let env = standard_env();
    let mut result = Value::Nil;
    for expr in exprs {
        result = eval(&expr, &env)?;
    }
    Ok(result)
}
