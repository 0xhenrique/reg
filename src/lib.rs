pub mod bytecode;
pub mod compiler;
pub mod eval;
pub mod macros;
pub mod parser;
pub mod value;
pub mod vm;

pub use compiler::Compiler;
pub use eval::{eval, standard_env, Env};
pub use macros::{expand, MacroRegistry};
pub use parser::{parse, parse_all};
pub use value::Value;
pub use vm::{standard_vm, VM};

/// Convenience function to evaluate a string using the bytecode VM
pub fn run(input: &str) -> Result<Value, String> {
    let expr = parse(input)?;
    let chunk = Compiler::compile(&expr)?;
    let mut vm = standard_vm();
    vm.run(chunk)
}

/// Convenience function to evaluate multiple expressions using the bytecode VM
pub fn run_all(input: &str) -> Result<Value, String> {
    let exprs = parse_all(input)?;
    let chunk = Compiler::compile_all(&exprs)?;
    let mut vm = standard_vm();
    vm.run(chunk)
}
