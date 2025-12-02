use std::fmt;
use std::rc::Rc;

use crate::bytecode::Chunk;

/// The core value type for our Lisp.
/// Uses Rc for reference counting (not Arc - single-threaded by default).
#[derive(Clone, Debug)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Symbol(Rc<str>),
    String(Rc<str>),
    List(Rc<[Value]>),
    Function(Rc<Function>),
    NativeFunction(Rc<NativeFunction>),
    CompiledFunction(Rc<Chunk>),
}

/// A user-defined function
#[derive(Debug)]
pub struct Function {
    pub params: Vec<String>,
    pub body: Value, // The body expression
    pub env: crate::eval::Env,
}

/// A native (Rust) function
pub struct NativeFunction {
    pub name: String,
    pub func: fn(&[Value]) -> Result<Value, String>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<native fn {}>", self.name)
    }
}

impl Value {
    /// Check if a value is truthy.
    /// Only nil and false are falsy, everything else is truthy (like Lua).
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Bool(false))
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    /// Get the type name as a string.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Symbol(_) => "symbol",
            Value::String(_) => "string",
            Value::List(_) => "list",
            Value::Function(_) => "function",
            Value::NativeFunction(_) => "native-function",
            Value::CompiledFunction(_) => "function",
        }
    }

    /// Create a symbol from a string slice
    pub fn symbol(s: &str) -> Value {
        Value::Symbol(Rc::from(s))
    }

    /// Create a string value
    pub fn string(s: &str) -> Value {
        Value::String(Rc::from(s))
    }

    /// Create a list from a Vec
    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::from(items))
    }

    /// Get as a symbol string, if this is a symbol
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Value::Symbol(s) => Some(s),
            _ => None,
        }
    }

    /// Get as a list slice, if this is a list
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(items) => Some(items),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(true) => write!(f, "true"),
            Value::Bool(false) => write!(f, "false"),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::List(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            Value::Function(_) => write!(f, "<function>"),
            Value::NativeFunction(nf) => write!(f, "<native fn {}>", nf.name),
            Value::CompiledFunction(_) => write!(f, "<function>"),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Int(a), Value::Float(b)) => (*a as f64) == *b,
            (Value::Float(a), Value::Int(b)) => *a == (*b as f64),
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            // Functions are never equal (identity comparison would require Rc::ptr_eq)
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_truthy() {
        assert!(!Value::Nil.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(Value::Int(0).is_truthy()); // 0 is truthy!
        assert!(Value::Int(42).is_truthy());
        assert!(Value::string("").is_truthy()); // empty string is truthy
        assert!(Value::list(vec![]).is_truthy()); // empty list is truthy
    }

    #[test]
    fn test_type_name() {
        assert_eq!(Value::Nil.type_name(), "nil");
        assert_eq!(Value::Bool(true).type_name(), "bool");
        assert_eq!(Value::Int(42).type_name(), "int");
        assert_eq!(Value::Float(3.14).type_name(), "float");
        assert_eq!(Value::symbol("foo").type_name(), "symbol");
        assert_eq!(Value::string("hello").type_name(), "string");
        assert_eq!(Value::list(vec![]).type_name(), "list");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::Nil), "nil");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
        assert_eq!(format!("{}", Value::Int(42)), "42");
        assert_eq!(format!("{}", Value::Float(3.14)), "3.14");
        assert_eq!(format!("{}", Value::symbol("foo")), "foo");
        assert_eq!(format!("{}", Value::string("hello")), "\"hello\"");
        assert_eq!(
            format!("{}", Value::list(vec![Value::Int(1), Value::Int(2)])),
            "(1 2)"
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(Value::Nil, Value::Nil);
        assert_eq!(Value::Bool(true), Value::Bool(true));
        assert_eq!(Value::Int(42), Value::Int(42));
        assert_eq!(Value::Float(3.14), Value::Float(3.14));
        assert_eq!(Value::Int(42), Value::Float(42.0)); // int/float comparison
        assert_eq!(Value::symbol("foo"), Value::symbol("foo"));
        assert_eq!(Value::string("hello"), Value::string("hello"));

        assert_ne!(Value::Nil, Value::Bool(false));
        assert_ne!(Value::Int(1), Value::Int(2));
    }
}
