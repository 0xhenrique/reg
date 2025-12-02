use std::fmt;
use std::rc::Rc;

use crate::bytecode::Chunk;

//=============================================================================
// NaN-Boxing Implementation
//=============================================================================
//
// We pack all values into 8 bytes using NaN-boxing:
//
// 64-bit Value Layout:
// ┌────────────────────────────────────────────────────────────────────┐
// │ If NOT a quiet NaN (bits 62:52 ≠ 0x7FF or bit 51 = 0):             │
// │   → It's a regular IEEE 754 double (float)                         │
// ├────────────────────────────────────────────────────────────────────┤
// │ If quiet NaN, use upper bits as type tag:                          │
// │                                                                    │
// │ 0x7FFC_0000_0000_0000 = Nil                                        │
// │ 0x7FFC_0000_0000_0001 = False                                      │
// │ 0x7FFC_0000_0000_0002 = True                                       │
// │ 0x7FFD_XXXX_XXXX_XXXX = Integer (48-bit signed in lower bits)      │
// │ 0x7FFE_XXXX_XXXX_XXXX = Heap pointer (48-bit address)              │
// └────────────────────────────────────────────────────────────────────┘
//
// Heap objects (String, Symbol, List, Function, etc.) use tagged pointers.

// Bit patterns for NaN-boxing
const QNAN: u64 = 0x7FFC_0000_0000_0000;  // Quiet NaN base
const TAG_NIL: u64 = 0x7FFC_0000_0000_0000;
const TAG_FALSE: u64 = 0x7FFC_0000_0000_0001;
const TAG_TRUE: u64 = 0x7FFC_0000_0000_0002;
const TAG_INT: u64 = 0x7FFD_0000_0000_0000;
const TAG_PTR: u64 = 0x7FFE_0000_0000_0000;

const TAG_MASK: u64 = 0xFFFF_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;  // 48 bits

// Sign extension for 48-bit integers
const INT_SIGN_BIT: u64 = 0x0000_8000_0000_0000;  // Bit 47

/// Heap object types - what the pointer points to
#[derive(Debug)]
pub enum HeapObject {
    String(Rc<str>),
    Symbol(Rc<str>),
    List(Rc<[Value]>),
    Function(Rc<Function>),
    NativeFunction(Rc<NativeFunction>),
    CompiledFunction(Rc<Chunk>),
}

impl Clone for HeapObject {
    fn clone(&self) -> Self {
        match self {
            HeapObject::String(s) => HeapObject::String(s.clone()),
            HeapObject::Symbol(s) => HeapObject::Symbol(s.clone()),
            HeapObject::List(l) => HeapObject::List(l.clone()),
            HeapObject::Function(f) => HeapObject::Function(f.clone()),
            HeapObject::NativeFunction(f) => HeapObject::NativeFunction(f.clone()),
            HeapObject::CompiledFunction(c) => HeapObject::CompiledFunction(c.clone()),
        }
    }
}

/// A user-defined function (for tree-walking interpreter)
#[derive(Debug)]
pub struct Function {
    pub params: Vec<String>,
    pub body: Value,
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

/// The core value type for our Lisp - NaN-boxed into 8 bytes.
pub struct Value(u64);

impl Value {
    //-------------------------------------------------------------------------
    // Constructors
    //-------------------------------------------------------------------------

    /// Create a Nil value
    #[inline]
    pub fn nil() -> Value {
        Value(TAG_NIL)
    }

    /// Create a boolean value
    #[inline]
    pub fn bool(b: bool) -> Value {
        Value(if b { TAG_TRUE } else { TAG_FALSE })
    }

    /// Create an integer value (48-bit signed)
    #[inline]
    pub fn int(n: i64) -> Value {
        // Store as 48-bit value (truncate if larger)
        let payload = (n as u64) & PAYLOAD_MASK;
        Value(TAG_INT | payload)
    }

    /// Create a float value
    #[inline]
    pub fn float(n: f64) -> Value {
        Value(n.to_bits())
    }

    /// Create a string value
    pub fn string(s: &str) -> Value {
        let heap = Box::new(HeapObject::String(Rc::from(s)));
        Value::from_heap(heap)
    }

    /// Create a symbol value
    pub fn symbol(s: &str) -> Value {
        let heap = Box::new(HeapObject::Symbol(Rc::from(s)));
        Value::from_heap(heap)
    }

    /// Create a list value
    pub fn list(items: Vec<Value>) -> Value {
        let heap = Box::new(HeapObject::List(Rc::from(items)));
        Value::from_heap(heap)
    }

    /// Create a heap-allocated value from a Box<HeapObject>
    fn from_heap(heap: Box<HeapObject>) -> Value {
        let ptr = Box::into_raw(heap) as u64;
        debug_assert!(ptr & TAG_MASK == 0, "Pointer uses more than 48 bits");
        Value(TAG_PTR | ptr)
    }

    //-------------------------------------------------------------------------
    // Type checks
    //-------------------------------------------------------------------------

    #[inline]
    pub fn is_nil(&self) -> bool {
        self.0 == TAG_NIL
    }

    #[inline]
    pub fn is_bool(&self) -> bool {
        self.0 == TAG_TRUE || self.0 == TAG_FALSE
    }

    #[inline]
    pub fn is_int(&self) -> bool {
        (self.0 & TAG_MASK) == TAG_INT
    }

    #[inline]
    pub fn is_float(&self) -> bool {
        // It's a float if it's not a special NaN value
        // A special NaN has bits 62:52 = 0x7FF and bit 51 = 1
        // Our tags use 0x7FFC, 0x7FFD, 0x7FFE which all have bit 51 = 1
        let is_quiet_nan = (self.0 & 0x7FFC_0000_0000_0000) == QNAN;
        !is_quiet_nan
    }

    #[inline]
    pub fn is_ptr(&self) -> bool {
        (self.0 & TAG_MASK) == TAG_PTR
    }

    //-------------------------------------------------------------------------
    // Value extraction
    //-------------------------------------------------------------------------

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self.0 {
            TAG_TRUE => Some(true),
            TAG_FALSE => Some(false),
            _ => None,
        }
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        if !self.is_int() {
            return None;
        }
        let payload = self.0 & PAYLOAD_MASK;
        // Sign-extend from 48 bits to 64 bits
        if payload & INT_SIGN_BIT != 0 {
            // Negative: set upper 16 bits
            Some((payload | !PAYLOAD_MASK) as i64)
        } else {
            Some(payload as i64)
        }
    }

    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        if self.is_float() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    /// Get the heap object if this is a heap pointer
    #[inline]
    fn as_heap(&self) -> Option<&HeapObject> {
        if !self.is_ptr() {
            return None;
        }
        let ptr = (self.0 & PAYLOAD_MASK) as *const HeapObject;
        // Safety: we only create these pointers from Box::into_raw
        Some(unsafe { &*ptr })
    }

    /// Get the heap object mutably (for cloning the inner Rc)
    #[inline]
    fn as_heap_mut(&self) -> Option<&mut HeapObject> {
        if !self.is_ptr() {
            return None;
        }
        let ptr = (self.0 & PAYLOAD_MASK) as *mut HeapObject;
        // Safety: we only create these pointers from Box::into_raw
        Some(unsafe { &mut *ptr })
    }

    /// Get as a symbol string, if this is a symbol
    pub fn as_symbol(&self) -> Option<&str> {
        match self.as_heap() {
            Some(HeapObject::Symbol(s)) => Some(s),
            _ => None,
        }
    }

    /// Get as a string, if this is a string
    pub fn as_string(&self) -> Option<&str> {
        match self.as_heap() {
            Some(HeapObject::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get as a list slice, if this is a list
    pub fn as_list(&self) -> Option<&[Value]> {
        match self.as_heap() {
            Some(HeapObject::List(items)) => Some(items),
            _ => None,
        }
    }

    /// Get as a function reference
    pub fn as_function(&self) -> Option<&Rc<Function>> {
        match self.as_heap() {
            Some(HeapObject::Function(f)) => Some(f),
            _ => None,
        }
    }

    /// Get as a native function reference
    pub fn as_native_function(&self) -> Option<&Rc<NativeFunction>> {
        match self.as_heap() {
            Some(HeapObject::NativeFunction(f)) => Some(f),
            _ => None,
        }
    }

    /// Get as a compiled function reference
    pub fn as_compiled_function(&self) -> Option<&Rc<Chunk>> {
        match self.as_heap() {
            Some(HeapObject::CompiledFunction(c)) => Some(c),
            _ => None,
        }
    }

    //-------------------------------------------------------------------------
    // Lisp semantics
    //-------------------------------------------------------------------------

    /// Check if a value is truthy.
    /// Only nil and false are falsy, everything else is truthy (like Lua).
    pub fn is_truthy(&self) -> bool {
        self.0 != TAG_NIL && self.0 != TAG_FALSE
    }

    /// Get the type name as a string.
    pub fn type_name(&self) -> &'static str {
        if self.is_nil() {
            "nil"
        } else if self.is_bool() {
            "bool"
        } else if self.is_int() {
            "int"
        } else if self.is_float() {
            "float"
        } else if let Some(heap) = self.as_heap() {
            match heap {
                HeapObject::String(_) => "string",
                HeapObject::Symbol(_) => "symbol",
                HeapObject::List(_) => "list",
                HeapObject::Function(_) => "function",
                HeapObject::NativeFunction(_) => "native-function",
                HeapObject::CompiledFunction(_) => "function",
            }
        } else {
            "unknown"
        }
    }

    //-------------------------------------------------------------------------
    // Backwards compatibility constructors
    //-------------------------------------------------------------------------

    /// Create Nil (backwards compat)
    pub const Nil: Value = Value(TAG_NIL);

    /// Create Bool (backwards compat)
    #[inline]
    pub fn Bool(b: bool) -> Value {
        Value::bool(b)
    }

    /// Create Int (backwards compat)
    #[inline]
    pub fn Int(n: i64) -> Value {
        Value::int(n)
    }

    /// Create Float (backwards compat)
    #[inline]
    pub fn Float(n: f64) -> Value {
        Value::float(n)
    }

    /// Create String (backwards compat) - from Rc<str>
    pub fn String(s: Rc<str>) -> Value {
        let heap = Box::new(HeapObject::String(s));
        Value::from_heap(heap)
    }

    /// Create Symbol (backwards compat) - from Rc<str>
    pub fn Symbol(s: Rc<str>) -> Value {
        let heap = Box::new(HeapObject::Symbol(s));
        Value::from_heap(heap)
    }

    /// Create List (backwards compat) - from Rc<[Value]>
    pub fn List(items: Rc<[Value]>) -> Value {
        let heap = Box::new(HeapObject::List(items));
        Value::from_heap(heap)
    }

    /// Create Function (backwards compat)
    pub fn Function(f: Rc<Function>) -> Value {
        let heap = Box::new(HeapObject::Function(f));
        Value::from_heap(heap)
    }

    /// Create NativeFunction (backwards compat)
    pub fn NativeFunction(f: Rc<NativeFunction>) -> Value {
        let heap = Box::new(HeapObject::NativeFunction(f));
        Value::from_heap(heap)
    }

    /// Create CompiledFunction (backwards compat)
    pub fn CompiledFunction(c: Rc<Chunk>) -> Value {
        let heap = Box::new(HeapObject::CompiledFunction(c));
        Value::from_heap(heap)
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        // If this is a heap pointer, we need to drop the HeapObject
        if self.is_ptr() {
            let ptr = (self.0 & PAYLOAD_MASK) as *mut HeapObject;
            // Safety: we only create these from Box::into_raw, and we set
            // the tag to something else after dropping
            unsafe {
                drop(Box::from_raw(ptr));
            }
            // Mark as nil to prevent double-free
            self.0 = TAG_NIL;
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        if self.is_ptr() {
            // Clone the heap object (which clones the inner Rc)
            if let Some(heap) = self.as_heap() {
                let cloned = Box::new(heap.clone());
                Value::from_heap(cloned)
            } else {
                Value::nil()
            }
        } else {
            // Primitives: just copy the bits
            Value(self.0)
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nil() {
            write!(f, "Nil")
        } else if let Some(b) = self.as_bool() {
            write!(f, "Bool({})", b)
        } else if let Some(n) = self.as_int() {
            write!(f, "Int({})", n)
        } else if let Some(n) = self.as_float() {
            write!(f, "Float({})", n)
        } else if let Some(heap) = self.as_heap() {
            write!(f, "{:?}", heap)
        } else {
            write!(f, "Value(0x{:016x})", self.0)
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nil() {
            write!(f, "nil")
        } else if let Some(true) = self.as_bool() {
            write!(f, "true")
        } else if let Some(false) = self.as_bool() {
            write!(f, "false")
        } else if let Some(n) = self.as_int() {
            write!(f, "{}", n)
        } else if let Some(n) = self.as_float() {
            write!(f, "{}", n)
        } else if let Some(heap) = self.as_heap() {
            match heap {
                HeapObject::String(s) => write!(f, "\"{}\"", s),
                HeapObject::Symbol(s) => write!(f, "{}", s),
                HeapObject::List(items) => {
                    write!(f, "(")?;
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "{}", item)?;
                    }
                    write!(f, ")")
                }
                HeapObject::Function(_) => write!(f, "<function>"),
                HeapObject::NativeFunction(nf) => write!(f, "<native fn {}>", nf.name),
                HeapObject::CompiledFunction(_) => write!(f, "<function>"),
            }
        } else {
            write!(f, "<unknown>")
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: if bits are equal, values are equal (for primitives)
        if self.0 == other.0 && !self.is_ptr() {
            return true;
        }

        // Check type-by-type
        if self.is_nil() && other.is_nil() {
            return true;
        }
        if let (Some(a), Some(b)) = (self.as_bool(), other.as_bool()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.as_int(), other.as_int()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.as_float(), other.as_float()) {
            return a == b;
        }
        // Int/Float comparison
        if let (Some(a), Some(b)) = (self.as_int(), other.as_float()) {
            return (a as f64) == b;
        }
        if let (Some(a), Some(b)) = (self.as_float(), other.as_int()) {
            return a == (b as f64);
        }

        // Heap object comparison
        match (self.as_heap(), other.as_heap()) {
            (Some(HeapObject::String(a)), Some(HeapObject::String(b))) => a == b,
            (Some(HeapObject::Symbol(a)), Some(HeapObject::Symbol(b))) => a == b,
            (Some(HeapObject::List(a)), Some(HeapObject::List(b))) => a == b,
            // Functions are never equal
            _ => false,
        }
    }
}

//=============================================================================
// Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil() {
        let v = Value::nil();
        assert!(v.is_nil());
        assert!(!v.is_truthy());
        assert_eq!(v.type_name(), "nil");
        assert_eq!(format!("{}", v), "nil");
    }

    #[test]
    fn test_bool() {
        let t = Value::bool(true);
        let f = Value::bool(false);

        assert!(t.is_bool());
        assert!(f.is_bool());
        assert!(t.is_truthy());
        assert!(!f.is_truthy());
        assert_eq!(t.as_bool(), Some(true));
        assert_eq!(f.as_bool(), Some(false));
        assert_eq!(format!("{}", t), "true");
        assert_eq!(format!("{}", f), "false");
    }

    #[test]
    fn test_int() {
        let v = Value::int(42);
        assert!(v.is_int());
        assert!(v.is_truthy());
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(format!("{}", v), "42");

        // Negative number
        let neg = Value::int(-100);
        assert_eq!(neg.as_int(), Some(-100));
        assert_eq!(format!("{}", neg), "-100");

        // Zero
        let zero = Value::int(0);
        assert_eq!(zero.as_int(), Some(0));
        assert!(zero.is_truthy()); // 0 is truthy in our Lisp!
    }

    #[test]
    fn test_int_large() {
        // Test larger integers within 48-bit range
        let large = Value::int(1_000_000_000_000);
        assert_eq!(large.as_int(), Some(1_000_000_000_000));

        let large_neg = Value::int(-1_000_000_000_000);
        assert_eq!(large_neg.as_int(), Some(-1_000_000_000_000));
    }

    #[test]
    fn test_float() {
        let v = Value::float(3.14);
        assert!(v.is_float());
        assert!(v.is_truthy());
        assert_eq!(v.as_float(), Some(3.14));
        assert_eq!(format!("{}", v), "3.14");

        // Zero
        let zero = Value::float(0.0);
        assert_eq!(zero.as_float(), Some(0.0));

        // Negative
        let neg = Value::float(-2.5);
        assert_eq!(neg.as_float(), Some(-2.5));
    }

    #[test]
    fn test_string() {
        let v = Value::string("hello");
        assert_eq!(v.as_string(), Some("hello"));
        assert_eq!(v.type_name(), "string");
        assert_eq!(format!("{}", v), "\"hello\"");
    }

    #[test]
    fn test_symbol() {
        let v = Value::symbol("foo");
        assert_eq!(v.as_symbol(), Some("foo"));
        assert_eq!(v.type_name(), "symbol");
        assert_eq!(format!("{}", v), "foo");
    }

    #[test]
    fn test_list() {
        let v = Value::list(vec![Value::int(1), Value::int(2), Value::int(3)]);
        assert_eq!(v.type_name(), "list");
        let items = v.as_list().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].as_int(), Some(1));
        assert_eq!(format!("{}", v), "(1 2 3)");
    }

    #[test]
    fn test_is_truthy() {
        assert!(!Value::nil().is_truthy());
        assert!(!Value::bool(false).is_truthy());
        assert!(Value::bool(true).is_truthy());
        assert!(Value::int(0).is_truthy()); // 0 is truthy!
        assert!(Value::int(42).is_truthy());
        assert!(Value::string("").is_truthy()); // empty string is truthy
        assert!(Value::list(vec![]).is_truthy()); // empty list is truthy
    }

    #[test]
    fn test_type_name() {
        assert_eq!(Value::nil().type_name(), "nil");
        assert_eq!(Value::bool(true).type_name(), "bool");
        assert_eq!(Value::int(42).type_name(), "int");
        assert_eq!(Value::float(3.14).type_name(), "float");
        assert_eq!(Value::symbol("foo").type_name(), "symbol");
        assert_eq!(Value::string("hello").type_name(), "string");
        assert_eq!(Value::list(vec![]).type_name(), "list");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::nil()), "nil");
        assert_eq!(format!("{}", Value::bool(true)), "true");
        assert_eq!(format!("{}", Value::int(42)), "42");
        assert_eq!(format!("{}", Value::float(3.14)), "3.14");
        assert_eq!(format!("{}", Value::symbol("foo")), "foo");
        assert_eq!(format!("{}", Value::string("hello")), "\"hello\"");
        assert_eq!(
            format!("{}", Value::list(vec![Value::int(1), Value::int(2)])),
            "(1 2)"
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(Value::nil(), Value::nil());
        assert_eq!(Value::bool(true), Value::bool(true));
        assert_eq!(Value::int(42), Value::int(42));
        assert_eq!(Value::float(3.14), Value::float(3.14));
        assert_eq!(Value::int(42), Value::float(42.0)); // int/float comparison
        assert_eq!(Value::symbol("foo"), Value::symbol("foo"));
        assert_eq!(Value::string("hello"), Value::string("hello"));

        assert_ne!(Value::nil(), Value::bool(false));
        assert_ne!(Value::int(1), Value::int(2));
    }

    #[test]
    fn test_clone() {
        // Primitives
        let a = Value::int(42);
        let b = a.clone();
        assert_eq!(a, b);

        // Heap objects
        let list = Value::list(vec![Value::int(1), Value::int(2)]);
        let list2 = list.clone();
        assert_eq!(list, list2);

        // String
        let s = Value::string("hello");
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_size() {
        // Value should be 8 bytes (u64)
        assert_eq!(std::mem::size_of::<Value>(), 8);
    }
}
