pub mod bytecode;
pub mod compiler;
pub mod eval;
pub mod jit;
pub mod macros;
pub mod parser;
pub mod value;
pub mod vm;

pub use compiler::Compiler;
pub use eval::{eval, standard_env, Env};
pub use macros::{expand, MacroRegistry};
pub use parser::{parse, parse_all};
pub use value::{clear_arena, set_arena_enabled, ConversionError, Value};
pub use vm::{standard_vm, VM};

use std::fmt;

// Re-export embedding API types after they're defined
// (LispError and LispResult are defined below)

//=============================================================================
// Embedding API - Error Types
//=============================================================================

/// Error type for Lisp VM operations
#[derive(Debug, Clone)]
pub enum LispError {
    /// Parse error - invalid syntax
    ParseError(String),
    /// Compilation error - invalid bytecode generation
    CompileError(String),
    /// Runtime error - error during execution
    RuntimeError(String),
    /// Conversion error - type mismatch when converting Value to Rust type
    ConversionError(ConversionError),
}

impl fmt::Display for LispError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LispError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LispError::CompileError(msg) => write!(f, "Compile error: {}", msg),
            LispError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            LispError::ConversionError(err) => write!(f, "Conversion error: {}", err),
        }
    }
}

impl std::error::Error for LispError {}

impl From<ConversionError> for LispError {
    fn from(err: ConversionError) -> Self {
        LispError::ConversionError(err)
    }
}

/// Type alias for Results in the embedding API
pub type LispResult<T> = std::result::Result<T, LispError>;

//=============================================================================
// Embedding API - VM Extensions
//=============================================================================

impl VM {
    /// Evaluate a Lisp expression and return the result as a Value
    ///
    /// # Example
    /// ```
    /// use lisp_vm::{VM, standard_vm};
    ///
    /// let mut vm = standard_vm();
    /// let result = vm.eval("(+ 1 2 3)").unwrap();
    /// assert_eq!(result.as_int(), Some(6));
    /// ```
    pub fn eval(&mut self, input: &str) -> LispResult<Value> {
        let expr = parse(input).map_err(LispError::ParseError)?;
        let chunk = Compiler::compile(&expr).map_err(LispError::CompileError)?;
        let result = self.run(chunk).map_err(LispError::RuntimeError)?;
        let promoted = result.promote();
        clear_arena();
        Ok(promoted)
    }

    /// Evaluate a Lisp expression and convert the result to a Rust type
    ///
    /// # Example
    /// ```
    /// use lisp_vm::{VM, standard_vm};
    ///
    /// let mut vm = standard_vm();
    /// let result: i64 = vm.eval_to("(+ 1 2 3)").unwrap();
    /// assert_eq!(result, 6);
    /// ```
    pub fn eval_to<T>(&mut self, input: &str) -> LispResult<T>
    where
        T: TryFrom<Value, Error = ConversionError>,
    {
        let value = self.eval(input)?;
        T::try_from(value).map_err(LispError::from)
    }

    /// Evaluate multiple expressions and return the result of the last one
    ///
    /// # Example
    /// ```
    /// use lisp_vm::{VM, standard_vm};
    ///
    /// let mut vm = standard_vm();
    /// let result = vm.eval_all("(def x 10) (def y 20) (+ x y)").unwrap();
    /// assert_eq!(result.as_int(), Some(30));
    /// ```
    pub fn eval_all(&mut self, input: &str) -> LispResult<Value> {
        let exprs = parse_all(input).map_err(LispError::ParseError)?;
        let chunk = Compiler::compile_all(&exprs).map_err(LispError::CompileError)?;
        let result = self.run(chunk).map_err(LispError::RuntimeError)?;
        let promoted = result.promote();
        clear_arena();
        Ok(promoted)
    }

    /// Set a global variable to a Rust value
    ///
    /// # Example
    /// ```
    /// use lisp_vm::{VM, standard_vm};
    ///
    /// let mut vm = standard_vm();
    /// vm.set("answer", 42i64);
    /// let result: i64 = vm.eval_to("answer").unwrap();
    /// assert_eq!(result, 42);
    /// ```
    pub fn set<T: Into<Value>>(&mut self, name: &str, value: T) {
        self.define_global(name, value.into());
    }

    /// Get a global variable as a Rust type
    ///
    /// # Example
    /// ```
    /// use lisp_vm::{VM, standard_vm};
    ///
    /// let mut vm = standard_vm();
    /// vm.eval("(def x 42)").unwrap();
    /// let x: i64 = vm.get("x").unwrap();
    /// assert_eq!(x, 42);
    /// ```
    pub fn get<T>(&self, name: &str) -> LispResult<T>
    where
        T: TryFrom<Value, Error = ConversionError>,
    {
        let value = self
            .get_global(name)
            .ok_or_else(|| LispError::RuntimeError(format!("Undefined variable: {}", name)))?;
        T::try_from(value).map_err(LispError::from)
    }
}

/// Debugging function to evaluate a string using the bytecode VM
pub fn run(input: &str) -> std::result::Result<Value, String> {
    let expr = parse(input)?;
    let chunk = Compiler::compile(&expr)?;
    let mut vm = standard_vm();
    let result = vm.run(chunk)?;
    // Promote result to Rc if needed (escaping arena scope)
    let promoted = result.promote();
    // Clear arena after each expression
    clear_arena();
    Ok(promoted)
}

/// Debugging function to evaluate multiple expressions using the bytecode VM
pub fn run_all(input: &str) -> std::result::Result<Value, String> {
    let exprs = parse_all(input)?;
    let chunk = Compiler::compile_all(&exprs)?;
    let mut vm = standard_vm();
    let result = vm.run(chunk)?;
    // Promote result to Rc if needed (escaping arena scope)
    let promoted = result.promote();
    // Clear arena after execution
    clear_arena();
    Ok(promoted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_eval() {
        let mut vm = standard_vm();
        let result = vm.eval("(+ 1 2 3)").unwrap();
        assert_eq!(result.as_int(), Some(6));
    }

    #[test]
    fn test_embedding_eval_to() {
        let mut vm = standard_vm();
        let result: i64 = vm.eval_to("(+ 1 2 3)").unwrap();
        assert_eq!(result, 6);

        let result: f64 = vm.eval_to("(/ 22 7)").unwrap();
        assert!((result - 3.142857).abs() < 0.01);

        let result: String = vm.eval_to(r#"(string-append "hello" " " "world")"#).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_embedding_eval_all() {
        let mut vm = standard_vm();
        let result = vm.eval_all("(def x 10) (def y 20) (+ x y)").unwrap();
        assert_eq!(result.as_int(), Some(30));
    }

    #[test]
    fn test_embedding_set_get() {
        let mut vm = standard_vm();

        // Set from Rust
        vm.set("answer", 42i64);

        // Get from Rust
        let answer: i64 = vm.get("answer").unwrap();
        assert_eq!(answer, 42);

        // Access from Lisp
        let result: i64 = vm.eval_to("answer").unwrap();
        assert_eq!(result, 42);

        // Modify from Lisp
        vm.eval("(def answer 100)").unwrap();
        let new_answer: i64 = vm.get("answer").unwrap();
        assert_eq!(new_answer, 100);
    }

    #[test]
    fn test_embedding_conversions() {
        let mut vm = standard_vm();

        // Integer
        vm.set("int_val", 42i64);
        assert_eq!(vm.get::<i64>("int_val").unwrap(), 42);

        // Float
        vm.set("float_val", 3.14f64);
        assert!((vm.get::<f64>("float_val").unwrap() - 3.14).abs() < 0.001);

        // String
        vm.set("string_val", "hello");
        assert_eq!(vm.get::<String>("string_val").unwrap(), "hello");

        // Bool
        vm.set("bool_val", true);
        assert_eq!(vm.get::<bool>("bool_val").unwrap(), true);

        // Vec
        vm.set("list_val", vec![1i64, 2i64, 3i64]);
        let list: Vec<Value> = vm.get("list_val").unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(1));
    }

    #[test]
    fn test_embedding_error_handling() {
        let mut vm = standard_vm();

        // Parse error
        let result = vm.eval("(def x");
        assert!(matches!(result, Err(LispError::ParseError(_))));

        // Runtime error
        let result = vm.eval("(/ 1 0)");
        assert!(matches!(result, Err(LispError::RuntimeError(_))));

        // Conversion error
        vm.set("string_val", "hello");
        let result: std::result::Result<i64, _> = vm.get("string_val");
        assert!(matches!(result, Err(LispError::ConversionError(_))));

        // Undefined variable
        let result: std::result::Result<i64, _> = vm.get("undefined");
        assert!(matches!(result, Err(LispError::RuntimeError(_))));
    }

    #[test]
    fn test_embedding_complex_workflow() {
        let mut vm = standard_vm();

        // Define a function in Lisp
        vm.eval("(def square (fn (x) (* x x)))").unwrap();

        // Set a variable from Rust
        vm.set("n", 5i64);

        // Call the function with the variable
        let result: i64 = vm.eval_to("(square n)").unwrap();
        assert_eq!(result, 25);

        // Define another function using the first
        vm.eval("(def sum-of-squares (fn (a b) (+ (square a) (square b))))").unwrap();

        // Call with Rust values
        vm.set("x", 3i64);
        vm.set("y", 4i64);
        let result: i64 = vm.eval_to("(sum-of-squares x y)").unwrap();
        assert_eq!(result, 25); // 3^2 + 4^2 = 9 + 16 = 25
    }

    #[test]
    fn test_embedding_multiple_vms() {
        // Test that multiple VMs are independent
        let mut vm1 = standard_vm();
        let mut vm2 = standard_vm();

        vm1.set("x", 10i64);
        vm2.set("x", 20i64);

        assert_eq!(vm1.get::<i64>("x").unwrap(), 10);
        assert_eq!(vm2.get::<i64>("x").unwrap(), 20);
    }

    #[test]
    fn test_lisp_error_display() {
        let err = LispError::ParseError("unexpected EOF".to_string());
        assert_eq!(format!("{}", err), "Parse error: unexpected EOF");

        let err = LispError::RuntimeError("division by zero".to_string());
        assert_eq!(format!("{}", err), "Runtime error: division by zero");

        let conv_err = ConversionError {
            expected: "int",
            got: "string".to_string(),
        };
        let err = LispError::ConversionError(conv_err);
        assert_eq!(format!("{}", err), "Conversion error: expected int, got string");
    }
}
