use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::bytecode::Chunk;

//=============================================================================
// Arena Allocator - EXPERIMENTAL!
//=============================================================================
//
// Arena allocation eliminates per-object reference counting overhead by:
// 1. Allocating objects in a growable Vec using bump allocation
// 2. Using indices instead of pointers for stable references
// 3. Cloning is free (just copy the index)
// 4. Dropping is a no-op (arena is cleared in bulk)
// 5. Values that escape (returned from top-level) are promoted to Rc
//
// Thread-local arena is used so native functions can allocate without
// explicit arena parameter passing.
//
// IMPORTANT: Arena has a size limit to prevent unbounded memory growth in
// long-running programs. When the limit is reached, new allocations fall
// back to Rc allocation. The ideia is fast allocation for short-lived programs

/// Maximum number of objects in arena before falling back to Rc allocation.
/// This prevents memory growth in long-running programs
/// 64K objects * ~40 bytes/cons = ~2.5MB max arena size
const ARENA_SIZE_LIMIT: usize = 64 * 1024;

/// Arena for heap object allocation without per-object reference counting.
/// Uses a simple bump allocator with Vec backing storage.
pub struct Arena {
    /// Storage for heap objects, indexed by arena index
    objects: Vec<HeapObject>,
    /// High water mark for statistics
    #[allow(dead_code)]
    max_size: usize,
}

impl Arena {
    /// Create a new empty arena
    pub fn new() -> Self {
        Arena {
            objects: Vec::with_capacity(1024), // Pre-allocation
            max_size: 0,
        }
    }

    /// Try to allocate a heap object in the arena.
    /// Returns Some(index) if successful, None if arena is full.
    #[inline]
    pub fn try_alloc(&mut self, obj: HeapObject) -> Option<u32> {
        if self.objects.len() >= ARENA_SIZE_LIMIT {
            return None; // Arena full, caller should use Rc
        }
        let idx = self.objects.len() as u32;
        self.objects.push(obj);
        Some(idx)
    }

    /// Allocate a heap object in the arena, returning its index
    /// Note: Use try_alloc() for size-limited allocation
    #[inline]
    pub fn alloc(&mut self, obj: HeapObject) -> u32 {
        let idx = self.objects.len() as u32;
        self.objects.push(obj);
        idx
    }

    /// Get a reference to an object by index
    #[inline]
    pub fn get(&self, idx: u32) -> &HeapObject {
        unsafe { self.objects.get_unchecked(idx as usize) }
    }

    /// Clear the arena, freeing all objects
    /// This is called after each top-level expression
    pub fn clear(&mut self) {
        if self.objects.len() > self.max_size {
            // Update high water mark (for future tuning)
        }
        self.objects.clear();
    }

    /// Check if arena is at size limit
    #[inline]
    pub fn is_full(&self) -> bool {
        self.objects.len() >= ARENA_SIZE_LIMIT
    }

    /// Check if arena is empty (for testing purposes)
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Number of objects in arena
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.objects.len()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local arena for allocation from native functions
thread_local! {
    static ARENA: RefCell<Arena> = RefCell::new(Arena::new());
    // Flag to enable/disable arena allocation (disabled by default, enable with --arena CLI flag)
    static ARENA_ENABLED: RefCell<bool> = const { RefCell::new(false) };
}

/// Enable or disable arena allocation (for testing)
pub fn set_arena_enabled(enabled: bool) {
    ARENA_ENABLED.with(|e| *e.borrow_mut() = enabled);
}

/// Check if arena allocation is enabled
#[inline]
fn arena_enabled() -> bool {
    ARENA_ENABLED.with(|e| *e.borrow())
}

/// Clear the thread-local arena
pub fn clear_arena() {
    ARENA.with(|a| a.borrow_mut().clear());
}

/// Get the current arena size (for testing/debugging)
#[allow(dead_code)]
pub fn arena_size() -> usize {
    ARENA.with(|a| a.borrow().len())
}

/// Get a reference to an arena object (unsafe - caller must ensure index is valid)
#[inline]
unsafe fn arena_get(idx: u32) -> *const HeapObject {
    ARENA.with(|a| {
        let arena = a.borrow();
        arena.get(idx) as *const HeapObject
    })
}

//=============================================================================
// Symbol Interner
//=============================================================================
//
// All symbols are interned for O(1) comparison. The interner maintains a
// mapping from string content to Rc<str>, ensuring identical symbols share
// the same Rc. Symbol comparison then uses Rc::ptr_eq instead of string comparison

thread_local! {
    static SYMBOL_INTERNER: RefCell<SymbolInterner> = RefCell::new(SymbolInterner::new());
}

struct SymbolInterner {
    symbols: HashMap<Box<str>, Rc<str>>,
}

impl SymbolInterner {
    fn new() -> Self {
        SymbolInterner {
            symbols: HashMap::new(),
        }
    }

    /// Intern a symbol string, returning a shared Rc<str>
    /// If the symbol already exists, returns the existing Rc
    /// Otherwise, creates a new Rc and stores it
    fn intern(&mut self, s: &str) -> Rc<str> {
        if let Some(rc) = self.symbols.get(s) {
            return rc.clone();
        }
        let rc: Rc<str> = Rc::from(s);
        self.symbols.insert(s.into(), rc.clone());
        rc
    }
}

/// Public function to intern a symbol string.
/// Returns an Rc<str> that can be used for pointer-based caching
/// Equal symbol strings will return the same Rc (pointer equality)
pub fn intern_symbol(s: &str) -> Rc<str> {
    SYMBOL_INTERNER.with(|interner| interner.borrow_mut().intern(s))
}

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
// Heap objects (String, Symbol, List, Function, etc.) use tagged pointers

// Bit patterns for NaN-boxing
const QNAN: u64 = 0x7FFC_0000_0000_0000;  // Quiet NaN base
const TAG_NIL: u64 = 0x7FFC_0000_0000_0000;
const TAG_FALSE: u64 = 0x7FFC_0000_0000_0001;
const TAG_TRUE: u64 = 0x7FFC_0000_0000_0002;
const TAG_INT: u64 = 0x7FFD_0000_0000_0000;
const TAG_PTR: u64 = 0x7FFE_0000_0000_0000;
// Arena-allocated objects use TAG_ARENA with index in lower 32 bits
// This distinguishes them from Rc-allocated objects (TAG_PTR)
// Arena values have free clone (copy bits) and no-op drop
const TAG_ARENA: u64 = 0x7FFF_0000_0000_0000;

const TAG_MASK: u64 = 0xFFFF_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;  // 48 bits

// Sign extension for 48-bit integers
const INT_SIGN_BIT: u64 = 0x0000_8000_0000_0000;  // Bit 47

/// A cons cell
/// (cons car cdr) creates a pair where car is the head and cdr is the tail
///
/// NOTE: This struct is now stored INLINE in HeapObject, not behind an Rc.
/// This eliminates one pointer dereference for every car/cdr operation
#[derive(Debug, Clone)]
pub struct ConsCell {
    pub car: Value,
    pub cdr: Value,
}

/// Heap object types - what the pointer points to
///
/// NOTE: Data is now stored inline (no double Rc indirection)
#[derive(Debug)]
pub enum HeapObject {
    /// String data stored inline (Box, not Rc)
    String(Box<str>),
    /// Symbol - keeps Rc<str> for interning (multiple Values share same string)
    Symbol(Rc<str>),
    /// List stored inline (Box, not Rc)
    List(Box<[Value]>),
    /// Cons cell stored INLINE - for list performance
    Cons(ConsCell),
    /// Function for tree-walking interpreter
    Function(Box<Function>),
    /// Native function stored inline
    NativeFunction(NativeFunction),
    /// Compiled function - keeps Rc<Chunk> for sharing in tail calls
    CompiledFunction(Rc<Chunk>),
    /// Thread handle for spawned threads - wrapped in Mutex because JoinHandle can only be joined once
    /// The Option allows us to take the handle when joining (join consumes the handle)
    ThreadHandle(Arc<Mutex<Option<JoinHandle<Result<SharedValue, String>>>>>),
}

impl Clone for HeapObject {
    fn clone(&self) -> Self {
        match self {
            HeapObject::String(s) => HeapObject::String(s.clone()),
            HeapObject::Symbol(s) => HeapObject::Symbol(s.clone()),
            HeapObject::List(l) => HeapObject::List(l.clone()),
            HeapObject::Cons(c) => HeapObject::Cons(c.clone()),
            HeapObject::Function(f) => HeapObject::Function(f.clone()),
            HeapObject::NativeFunction(f) => HeapObject::NativeFunction(f.clone()),
            HeapObject::CompiledFunction(c) => HeapObject::CompiledFunction(c.clone()),
            HeapObject::ThreadHandle(h) => HeapObject::ThreadHandle(h.clone()),
        }
    }
}

//=============================================================================
// Thread-Safe Shared Values (Arc-based)
//=============================================================================
//
// SharedValue and SharedHeapObject provide Arc-based alternatives to the
// Rc-based Value and HeapObject types. These are used when values need to
// be shared across thread boundaries.
//
// Conversion:
//   Value (Rc) -> SharedValue (Arc) via make_shared()
//   SharedValue (Arc) -> Value (Rc) via from_shared()

/// Thread-safe cons cell using SharedValue instead of Value
#[derive(Debug, Clone)]
pub struct SharedConsCell {
    pub car: SharedValue,
    pub cdr: SharedValue,
}

/// Thread-safe heap objects using Arc instead of Rc
/// Mirrors HeapObject but is safe to send across threads
/// Note: Functions are NOT included because closures capture environments with Rc
/// and cannot be shared across threads. Use compiled functions or native functions instead.
#[derive(Debug, Clone)]
pub enum SharedHeapObject {
    /// String data
    String(Box<str>),
    /// Symbol - shared across threads
    Symbol(Arc<str>),
    /// List of shared values
    List(Box<[SharedValue]>),
    /// Cons cell with shared values
    Cons(SharedConsCell),
    /// Native functions are just function pointers, safe to share
    NativeFunction(NativeFunction),
    /// Compiled functions share their chunk via Arc
    CompiledFunction(Arc<Chunk>),
}

/// A thread-safe value that can be sent across thread boundaries
/// Uses Arc internally instead of Rc
#[derive(Clone)]
pub struct SharedValue {
    /// The underlying Arc-allocated heap object
    /// For primitives (int, float, bool, nil), we just clone the Value bits
    inner: Arc<SharedHeapObject>,
}

impl fmt::Debug for SharedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedValue({:?})", self.inner)
    }
}

impl fmt::Display for SharedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.inner {
            SharedHeapObject::String(s) => write!(f, "\"{}\"", s),
            SharedHeapObject::Symbol(s) => write!(f, "{}", s),
            SharedHeapObject::List(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            SharedHeapObject::Cons(_) => {
                // Print cons cells as lists
                write!(f, "(")?;
                let mut current = self.clone();
                let mut first = true;
                loop {
                    if let SharedHeapObject::Cons(cons) = &*current.inner {
                        if !first {
                            write!(f, " ")?;
                        }
                        first = false;
                        write!(f, "{}", cons.car)?;
                        current = cons.cdr.clone();
                    } else if matches!(&*current.inner, SharedHeapObject::List(items) if items.is_empty()) {
                        break;
                    } else if let SharedHeapObject::List(items) = &*current.inner {
                        for item in items.iter() {
                            write!(f, " {}", item)?;
                        }
                        break;
                    } else {
                        write!(f, " . {}", current)?;
                        break;
                    }
                }
                write!(f, ")")
            }
            SharedHeapObject::NativeFunction(nf) => write!(f, "<native fn {}>", nf.name),
            SharedHeapObject::CompiledFunction(_) => write!(f, "<function>"),
        }
    }
}

/// A user-defined function (for tree-walking interpreter)
#[derive(Debug, Clone)]
pub struct Function {
    pub params: Vec<String>,
    pub body: Value,
    pub env: crate::eval::Env,
}

/// A native (Rust) function
#[derive(Clone)]
pub struct NativeFunction {
    pub name: String,
    pub func: fn(&[Value]) -> Result<Value, String>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<native fn {}>", self.name)
    }
}

/// The core value type - NaN-boxed into 8 bytes.
pub struct Value(u64);

impl Value {
    // Constructors

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
        let heap = Rc::new(HeapObject::String(s.into()));
        Value::from_heap(heap)
    }

    /// Create a symbol value (interned for O(1) comparison)
    pub fn symbol(s: &str) -> Value {
        let interned = SYMBOL_INTERNER.with(|interner| interner.borrow_mut().intern(s));
        let heap = Rc::new(HeapObject::Symbol(interned));
        Value::from_heap(heap)
    }

    /// Create a list value
    pub fn list(items: Vec<Value>) -> Value {
        let heap = Rc::new(HeapObject::List(items.into_boxed_slice()));
        Value::from_heap(heap)
    }

    /// Create a cons cell - O(1) operation
    /// (cons car cdr) creates a pair where car is the head and cdr is the tail
    ///
    /// NOTE: Uses arena allocation when enabled for zero-refcount cloning
    /// When arena is enabled, cons cells are allocated in the thread-local arena
    /// When arena is full or disabled, falls back to Rc allocation
    pub fn cons(car: Value, cdr: Value) -> Value {
        if arena_enabled() && !ARENA.with(|a| a.borrow().is_full()) {
            // Arena allocation - zero overhead clone/drop
            let idx = ARENA.with(|a| {
                a.borrow_mut().alloc(HeapObject::Cons(ConsCell { car, cdr }))
            });
            Value(TAG_ARENA | idx as u64)
        } else {
            // Rc allocation fallback (arena disabled or full)
            let heap = Rc::new(HeapObject::Cons(ConsCell { car, cdr }));
            Value::from_heap(heap)
        }
    }

    /// Create a cons cell using Rc (for values that need to persist beyond arena lifetime)
    pub fn cons_rc(car: Value, cdr: Value) -> Value {
        let heap = Rc::new(HeapObject::Cons(ConsCell { car, cdr }));
        Value::from_heap(heap)
    }

    /// Create a user-defined function value
    pub fn function(f: Function) -> Value {
        let heap = Rc::new(HeapObject::Function(Box::new(f)));
        Value::from_heap(heap)
    }

    /// Create a native function value
    pub fn native_function(name: &str, func: fn(&[Value]) -> Result<Value, String>) -> Value {
        let heap = Rc::new(HeapObject::NativeFunction(NativeFunction {
            name: name.to_string(),
            func,
        }));
        Value::from_heap(heap)
    }

    /// Create a heap-allocated value from an Rc<HeapObject>
    /// Uses Rc::into_raw to store the pointer - refcount is NOT decremented
    pub fn from_heap(heap: Rc<HeapObject>) -> Value {
        let ptr = Rc::into_raw(heap) as u64;
        debug_assert!(ptr & TAG_MASK == 0, "Pointer uses more than 48 bits");
        Value(TAG_PTR | ptr)
    }

    // Type checks

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

    /// Check if this is an arena-allocated value
    #[inline]
    pub fn is_arena(&self) -> bool {
        (self.0 & TAG_MASK) == TAG_ARENA
    }

    /// Check if this is any heap value (either Rc or arena)
    #[inline]
    pub fn is_heap(&self) -> bool {
        let tag = self.0 & TAG_MASK;
        tag == TAG_PTR || tag == TAG_ARENA
    }

    // Value extraction

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

    /// Unchecked integer extraction - ONLY call when you're certain this is an integer
    /// Skips the is_int check for performance in specialized opcodes
    /// Safety: Caller must ensure this Value is an integer (is_int() == true)
    #[inline(always)]
    pub unsafe fn as_int_unchecked(&self) -> i64 {
        let payload = self.0 & PAYLOAD_MASK;
        // Sign-extend from 48 bits to 64 bits
        if payload & INT_SIGN_BIT != 0 {
            (payload | !PAYLOAD_MASK) as i64
        } else {
            payload as i64
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

    /// Get the heap object if this is a heap pointer (Rc or arena)
    #[inline]
    pub fn as_heap(&self) -> Option<&HeapObject> {
        if self.is_ptr() {
            let ptr = (self.0 & PAYLOAD_MASK) as *const HeapObject;
            // Safety: we only create these pointers from Rc::into_raw
            Some(unsafe { &*ptr })
        } else if self.is_arena() {
            // Arena allocation - get from thread-local arena
            let idx = (self.0 & PAYLOAD_MASK) as u32;
            // Safety: arena values are only created via arena_alloc
            Some(unsafe { &*arena_get(idx) })
        } else {
            None
        }
    }

    /// Get as a symbol string, if this is a symbol
    pub fn as_symbol(&self) -> Option<&str> {
        match self.as_heap() {
            Some(HeapObject::Symbol(s)) => Some(s),
            _ => None,
        }
    }

    /// Get the symbol's Rc<str> for pointer-based caching
    /// Returns the interned Rc<str> which can be used for O(1) hash/compare
    pub fn as_symbol_rc(&self) -> Option<Rc<str>> {
        match self.as_heap() {
            Some(HeapObject::Symbol(s)) => Some(s.clone()),
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

    /// Get as a cons cell reference (now directly from HeapObject, no Rc deref)
    pub fn as_cons(&self) -> Option<&ConsCell> {
        match self.as_heap() {
            Some(HeapObject::Cons(c)) => Some(c),
            _ => None,
        }
    }

    /// Check if this is a cons cell
    pub fn is_cons(&self) -> bool {
        matches!(self.as_heap(), Some(HeapObject::Cons(_)))
    }

    /// Get as a function reference
    pub fn as_function(&self) -> Option<&Function> {
        match self.as_heap() {
            Some(HeapObject::Function(f)) => Some(f),
            _ => None,
        }
    }

    /// Get as a native function reference
    pub fn as_native_function(&self) -> Option<&NativeFunction> {
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

    // Lisp semantics

    /// Check if a value is truthy
    /// Only nil and false are falsy, everything else is truthy (like Lua)
    #[inline]
    pub fn is_truthy(&self) -> bool {
        self.0 != TAG_NIL && self.0 != TAG_FALSE
    }

    // Move semantics support

    /// Take the value, leaving Nil in its place
    /// This is O(1) and avoids Rc increment/decrement entirely
    /// Use this when you know the source won't be used again
    #[inline]
    pub fn take(&mut self) -> Value {
        let bits = self.0;
        self.0 = TAG_NIL;
        Value(bits)
    }

    /// Take and get car of a cons cell, consuming the cons cell
    /// Returns None if not a cons cell
    ///
    /// After some benchmarking we discovered that this is more
    /// efficient than clone + car probably because:
    /// >no Rc increment for the cons cell
    /// >cons cell is freed immediately if refcount was 1
    /// >if refcount is 1, we can move car out without cloning
    #[inline]
    pub fn take_car(self) -> Option<Value> {
        // Get the cons cell reference
        if let Some(cons) = self.as_cons() {
            // Clone car (we need to because cons is borrowed)
            let car = cons.car.clone();
            // self drops here, decrementing cons cell refcount
            Some(car)
        } else if let Some(list) = self.as_list() {
            // Array list fallback
            list.first().cloned()
        } else {
            None
        }
    }

    /// Take and get cdr of a cons cell, consuming the cons cell
    /// Returns None if not a cons cell
    #[inline]
    pub fn take_cdr(self) -> Option<Value> {
        if let Some(cons) = self.as_cons() {
            let cdr = cons.cdr.clone();
            Some(cdr)
        } else if let Some(list) = self.as_list() {
            if list.is_empty() {
                None
            } else {
                // Convert array tail to cons chain for O(1) following CDR operations
                Some(Value::slice_to_cons(&list[1..]))
            }
        } else {
            None
        }
    }

    /// Convert an array slice to a cons chain
    /// Used when CDR is called on an array-backed list to avoid O(n) copies
    /// The resulting cons chain enables O(1) subsequent CDR operations
    pub fn slice_to_cons(slice: &[Value]) -> Value {
        let mut result = Value::nil();
        for item in slice.iter().rev() {
            result = Value::cons(item.clone(), result);
        }
        result
    }

    /// Get the type name as a string
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
                HeapObject::Cons(_) => "list", // cons cells are also lists
                HeapObject::Function(_) => "function",
                HeapObject::NativeFunction(_) => "native-function",
                HeapObject::CompiledFunction(_) => "function",
                HeapObject::ThreadHandle(_) => "thread-handle",
            }
        } else {
            "unknown"
        }
    }

    // Backwards compatibility constructors

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
        let heap = Rc::new(HeapObject::String(Box::from(&*s)));
        Value::from_heap(heap)
    }

    /// Create Symbol (backwards compat) - from Rc<str>
    /// Note: For proper interning, prefer Value::symbol(&str) instead
    pub fn Symbol(s: Rc<str>) -> Value {
        // Re-intern to ensure pointer equality works correctly
        let interned = SYMBOL_INTERNER.with(|interner| interner.borrow_mut().intern(&s));
        let heap = Rc::new(HeapObject::Symbol(interned));
        Value::from_heap(heap)
    }

    /// Create List (backwards compat) - from Rc<[Value]>
    pub fn List(items: Rc<[Value]>) -> Value {
        let heap = Rc::new(HeapObject::List(items.to_vec().into_boxed_slice()));
        Value::from_heap(heap)
    }

    /// Create Function (backwards compat)
    pub fn Function(f: Rc<Function>) -> Value {
        // Convert Rc to Box by cloning the inner value
        let heap = Rc::new(HeapObject::Function(Box::new((*f).clone())));
        Value::from_heap(heap)
    }

    /// Create NativeFunction (backwards compat)
    pub fn NativeFunction(f: Rc<NativeFunction>) -> Value {
        // Clone the inner value since NativeFunction is now stored inline
        let heap = Rc::new(HeapObject::NativeFunction((*f).clone()));
        Value::from_heap(heap)
    }

    /// Create CompiledFunction (backwards compat)
    pub fn CompiledFunction(c: Rc<Chunk>) -> Value {
        let heap = Rc::new(HeapObject::CompiledFunction(c));
        Value::from_heap(heap)
    }

    // JIT support methods

    /// Get the raw bits of this value (for JIT compilation)
    /// This returns the NaN-boxed u64 representation
    #[inline]
    pub fn to_bits(&self) -> u64 {
        self.0
    }

    /// Create a Value from raw bits (for JIT compilation)
    /// # Safety
    /// The caller must ensure the bits represent a valid NaN-boxed value
    /// For primitive types (nil, bool, int, float), this is always safe
    /// For heap types (TAG_PTR), the caller must ensure the pointer is valid
    #[inline]
    pub unsafe fn from_bits(bits: u64) -> Value {
        Value(bits)
    }

    /// Create a Value from raw bits, with a reference count increment for heap values
    /// This is the safe version for JIT use when the value might be a heap pointer
    #[inline]
    pub fn from_bits_safe(bits: u64) -> Value {
        let tag = bits & TAG_MASK;
        if tag == TAG_PTR {
            // Heap pointer - need to increment reference count
            let ptr = (bits & PAYLOAD_MASK) as *const HeapObject;
            unsafe {
                Rc::increment_strong_count(ptr);
            }
        }
        // For primitives, arena values, and the newly-cloned ptr, just return
        Value(bits)
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        // Only Rc-allocated values need refcount management
        // Arena values are freed in bulk when the arena is cleared
        if self.is_ptr() {
            let ptr = (self.0 & PAYLOAD_MASK) as *const HeapObject;
            // Safety: we only create these from Rc::into_raw
            // Rc::from_raw will decrement the refcount and free if it hits 0
            unsafe {
                drop(Rc::from_raw(ptr));
            }
            // Mark as nil to prevent double-free
            self.0 = TAG_NIL;
        }
        // Arena values: no-op drop (the arena will free them all at once)
    }
}

impl Clone for Value {
    #[inline(always)]
    fn clone(&self) -> Self {
        if self.is_ptr() {
            // Rc-allocated: increment reference count
            let ptr = (self.0 & PAYLOAD_MASK) as *const HeapObject;
            // Safety: we only create these from Rc::into_raw
            // increment_strong_count increases refcount without creating an Rc
            unsafe {
                Rc::increment_strong_count(ptr);
            }
            // Return a Value with the same pointer (both now own a reference)
            Value(self.0)
        } else if self.is_arena() {
            // Arena-allocated: FREE CLONE - just copy the bits!
            // No refcount increment, arena owns all objects
            Value(self.0)
        } else {
            // Primitives: just copy the bits
            Value(self.0)
        }
    }
}

impl Value {
    /// Promote an arena-allocated value to Rc-allocated for escape from arena scope
    /// This is called when a value needs to outlive the arena (for exemplo returned from VM)
    /// Non-arena values are returned unchanged
    ///
    /// This recursively promotes nested values (cons cells) to ensure the entire
    /// structure is Rc-allocated
    pub fn promote(&self) -> Value {
        if self.is_arena() {
            // Get the heap object and create an Rc copy
            match self.as_heap() {
                Some(HeapObject::Cons(cons)) => {
                    // Recursively promote car and cdr
                    let car = cons.car.promote();
                    let cdr = cons.cdr.promote();
                    Value::cons_rc(car, cdr)
                }
                Some(HeapObject::String(s)) => Value::string(s),
                Some(HeapObject::Symbol(s)) => {
                    // Re-intern the symbol
                    Value::symbol(s)
                }
                Some(HeapObject::List(items)) => {
                    // Recursively promote list elements
                    let promoted: Vec<Value> = items.iter().map(|v| v.promote()).collect();
                    Value::list(promoted)
                }
                Some(HeapObject::Function(f)) => Value::function((**f).clone()),
                Some(HeapObject::NativeFunction(f)) => {
                    let heap = Rc::new(HeapObject::NativeFunction(f.clone()));
                    Value::from_heap(heap)
                }
                Some(HeapObject::CompiledFunction(c)) => {
                    Value::CompiledFunction(c.clone())
                }
                Some(HeapObject::ThreadHandle(h)) => {
                    // ThreadHandles are already Arc-based, just clone the Value
                    let heap = Rc::new(HeapObject::ThreadHandle(h.clone()));
                    Value::from_heap(heap)
                }
                None => Value::nil(),
            }
        } else if self.is_ptr() {
            // Already Rc-allocated, but may contain arena children (cons cells)
            // Need to check and promote nested arena values
            match self.as_heap() {
                Some(HeapObject::Cons(cons)) => {
                    if cons.car.is_arena() || cons.cdr.is_arena() {
                        let car = cons.car.promote();
                        let cdr = cons.cdr.promote();
                        Value::cons_rc(car, cdr)
                    } else {
                        self.clone()
                    }
                }
                Some(HeapObject::List(items)) => {
                    if items.iter().any(|v| v.is_arena()) {
                        let promoted: Vec<Value> = items.iter().map(|v| v.promote()).collect();
                        Value::list(promoted)
                    } else {
                        self.clone()
                    }
                }
                _ => self.clone(),
            }
        } else {
            // Primitive values don't need promotion
            self.clone()
        }
    }

    /// Convert an Rc-based Value to an Arc-based SharedValue for thread sharing
    /// This deep-converts the entire value tree, recursively converting all nested values
    ///
    /// # Example
    /// ```ignore
    /// let value = Value::list(vec![Value::int(1), Value::int(2)]);
    /// let shared = value.make_shared().unwrap();
    /// // shared can now be sent across thread boundaries
    /// ```
    pub fn make_shared(&self) -> Result<SharedValue, String> {
        // First promote arena values to Rc (if any)
        let promoted = self.promote();

        // Then convert Rc to Arc
        if promoted.is_nil() {
            // Nil is represented as an empty list
            return Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::List(Box::new([]))),
            });
        }

        if let Some(b) = promoted.as_bool() {
            // Bools need to be wrapped
            let bool_str = if b { "true" } else { "false" };
            return Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::Symbol(Arc::from(bool_str))),
            });
        }

        if let Some(i) = promoted.as_int() {
            // Integers as symbols for now (could be optimized later)
            return Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::Symbol(Arc::from(i.to_string().as_str()))),
            });
        }

        if let Some(f) = promoted.as_float() {
            // Floats as symbols for now (could be optimized later)
            return Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::Symbol(Arc::from(f.to_string().as_str()))),
            });
        }

        // Handle heap objects
        match promoted.as_heap() {
            Some(HeapObject::String(s)) => Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::String(s.to_string().into_boxed_str())),
            }),
            Some(HeapObject::Symbol(s)) => Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::Symbol(Arc::from(&**s))),
            }),
            Some(HeapObject::List(items)) => {
                let shared_items: Result<Vec<SharedValue>, String> =
                    items.iter().map(|v| v.make_shared()).collect();
                Ok(SharedValue {
                    inner: Arc::new(SharedHeapObject::List(
                        shared_items?.into_boxed_slice(),
                    )),
                })
            }
            Some(HeapObject::Cons(cons)) => {
                let shared_car = cons.car.make_shared()?;
                let shared_cdr = cons.cdr.make_shared()?;
                Ok(SharedValue {
                    inner: Arc::new(SharedHeapObject::Cons(SharedConsCell {
                        car: shared_car,
                        cdr: shared_cdr,
                    })),
                })
            }
            Some(HeapObject::Function(_)) => {
                Err("Functions cannot be shared across threads".to_string())
            }
            Some(HeapObject::NativeFunction(nf)) => Ok(SharedValue {
                inner: Arc::new(SharedHeapObject::NativeFunction(nf.clone())),
            }),
            Some(HeapObject::CompiledFunction(chunk)) => {
                // Convert Rc<Chunk> to Arc<Chunk>
                let arc_chunk = Arc::new((**chunk).clone());
                Ok(SharedValue {
                    inner: Arc::new(SharedHeapObject::CompiledFunction(arc_chunk)),
                })
            }
            Some(HeapObject::ThreadHandle(_)) => {
                Err("Thread handles cannot be shared across threads".to_string())
            }
            None => Err("Cannot convert unknown value to SharedValue".to_string()),
        }
    }

    /// Convert an Arc-based SharedValue back to an Rc-based Value
    /// This is used when bringing values back from threads into single-threaded context
    ///
    /// # Example
    /// ```ignore
    /// let shared = value.make_shared().unwrap();
    /// // ... pass to thread ...
    /// let value_back = Value::from_shared(&shared);
    /// ```
    pub fn from_shared(shared: &SharedValue) -> Value {
        match &*shared.inner {
            SharedHeapObject::String(s) => Value::string(s),
            SharedHeapObject::Symbol(s) => {
                // Try to parse as primitives first
                if s.as_ref() == "true" {
                    return Value::bool(true);
                }
                if s.as_ref() == "false" {
                    return Value::bool(false);
                }
                if s.as_ref() == "nil" {
                    return Value::nil();
                }
                // Try parsing as number
                if let Ok(i) = s.parse::<i64>() {
                    return Value::int(i);
                }
                if let Ok(f) = s.parse::<f64>() {
                    return Value::float(f);
                }
                // Otherwise treat as symbol
                Value::symbol(s)
            }
            SharedHeapObject::List(items) => {
                if items.is_empty() {
                    return Value::nil();
                }
                let values: Vec<Value> = items.iter().map(Value::from_shared).collect();
                Value::list(values)
            }
            SharedHeapObject::Cons(cons) => {
                let car = Value::from_shared(&cons.car);
                let cdr = Value::from_shared(&cons.cdr);
                Value::cons_rc(car, cdr)
            }
            SharedHeapObject::NativeFunction(nf) => {
                Value::native_function(&nf.name, nf.func)
            }
            SharedHeapObject::CompiledFunction(chunk) => {
                // Convert Arc<Chunk> back to Rc<Chunk>
                let rc_chunk = Rc::new((**chunk).clone());
                Value::CompiledFunction(rc_chunk)
            }
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
                HeapObject::Cons(_) => {
                    // Print cons cells as lists by traversing the chain
                    write!(f, "(")?;
                    let mut current = self;
                    let mut first = true;
                    loop {
                        if let Some(cons) = current.as_cons() {
                            if !first {
                                write!(f, " ")?;
                            }
                            first = false;
                            write!(f, "{}", cons.car)?;
                            current = &cons.cdr;
                        } else if current.is_nil() {
                            // Proper list ending with nil
                            break;
                        } else if let Some(items) = current.as_list() {
                            // Cons chain ending with an array list
                            for item in items.iter() {
                                write!(f, " {}", item)?;
                            }
                            break;
                        } else {
                            // Improper list (dotted pair)
                            write!(f, " . {}", current)?;
                            break;
                        }
                    }
                    write!(f, ")")
                }
                HeapObject::Function(_) => write!(f, "<function>"),
                HeapObject::NativeFunction(nf) => write!(f, "<native fn {}>", nf.name),
                HeapObject::CompiledFunction(_) => write!(f, "<function>"),
                HeapObject::ThreadHandle(_) => write!(f, "<thread-handle>"),
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
            // Symbol comparison uses Rc::ptr_eq for O(1) - symbols are interned!
            (Some(HeapObject::Symbol(a)), Some(HeapObject::Symbol(b))) => Rc::ptr_eq(a, b),
            (Some(HeapObject::List(a)), Some(HeapObject::List(b))) => a == b,
            (Some(HeapObject::Cons(a)), Some(HeapObject::Cons(b))) => {
                a.car == b.car && a.cdr == b.cdr
            }
            // Functions are never equal
            _ => false,
        }
    }
}

//=============================================================================
// Value Conversion Traits - FFI Support
//=============================================================================
// These traits enable ergonomic conversion between Rust types and Values,
// making FFI and native function implementation more convenient.

// From<T> for Value - infallible conversions (always succeed)

impl From<()> for Value {
    fn from(_: ()) -> Self {
        Value::nil()
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::bool(b)
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::int(n)
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::int(n as i64)
    }
}

impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Value::int(n as i64)
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::float(n)
}
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::string(s)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::string(&s)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        let items: Vec<Value> = v.into_iter().map(|x| x.into()).collect();
        Value::list(items)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::nil(),
        }
    }
}

// TryFrom<Value> for T - fallible conversions (can fail)

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionError {
    pub expected: &'static str,
    pub got: String,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected {}, got {}", self.expected, self.got)
    }
}

impl std::error::Error for ConversionError {}

impl TryFrom<Value> for bool {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value.as_bool().ok_or_else(|| ConversionError {
            expected: "bool",
            got: value.type_name().to_string(),
        })
    }
}

impl TryFrom<Value> for i64 {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value.as_int().ok_or_else(|| ConversionError {
            expected: "int",
            got: value.type_name().to_string(),
        })
    }
}

impl TryFrom<Value> for i32 {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let n = value.as_int().ok_or_else(|| ConversionError {
            expected: "int",
            got: value.type_name().to_string(),
        })?;
        i32::try_from(n).map_err(|_| ConversionError {
            expected: "int32",
            got: format!("int64 out of range: {}", n),
        })
    }
}

impl TryFrom<Value> for usize {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let n = value.as_int().ok_or_else(|| ConversionError {
            expected: "int",
            got: value.type_name().to_string(),
        })?;
        if n < 0 {
            return Err(ConversionError {
                expected: "non-negative int",
                got: format!("{}", n),
            });
        }
        Ok(n as usize)
    }
}

impl TryFrom<Value> for f64 {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        // Allow both float and int -> float conversion
        if let Some(f) = value.as_float() {
            Ok(f)
        } else if let Some(i) = value.as_int() {
            Ok(i as f64)
        } else {
            Err(ConversionError {
                expected: "float or int",
                got: value.type_name().to_string(),
            })
        }
    }
}

impl TryFrom<Value> for String {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value
            .as_string()
            .map(|s| s.to_string())
            .ok_or_else(|| ConversionError {
                expected: "string",
                got: value.type_name().to_string(),
            })
    }
}

impl TryFrom<Value> for Vec<Value> {
    type Error = ConversionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if value.is_nil() {
            return Ok(Vec::new());
        }

        // Handle array lists
        if let Some(items) = value.as_list() {
            return Ok(items.to_vec());
        }

        // Handle cons cells - traverse and collect
        if value.as_cons().is_some() {
            let mut result = Vec::new();
            let mut current = value;
            while let Some(cons) = current.as_cons() {
                result.push(cons.car.clone());
                current = cons.cdr.clone();
            }
            // If ended on array list, append it
            if let Some(items) = current.as_list() {
                result.extend_from_slice(items);
            }
            return Ok(result);
        }

        Err(ConversionError {
            expected: "list",
            got: value.type_name().to_string(),
        })
    }
}

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

    #[test]
    fn test_cons_cell() {
        // Test that cons cells work with the flattened structure
        let cell = Value::cons(Value::int(1), Value::nil());
        assert!(cell.is_cons());
        let cons = cell.as_cons().unwrap();
        assert_eq!(cons.car.as_int(), Some(1));
        assert!(cons.cdr.is_nil());
    }

    #[test]
    fn test_cons_list_display() {
        // Build a cons list: (1 2 3)
        let list = Value::cons(
            Value::int(1),
            Value::cons(
                Value::int(2),
                Value::cons(Value::int(3), Value::nil()),
            ),
        );
        assert_eq!(format!("{}", list), "(1 2 3)");
    }

    // Arena Allocation Tests

    #[test]
    fn test_arena_cons_is_arena() {
        // Ensure arena is enabled for this test
        super::set_arena_enabled(true);
        super::clear_arena();

        let cell = Value::cons(Value::int(1), Value::nil());
        assert!(cell.is_arena(), "cons should be arena-allocated");
        assert!(!cell.is_ptr(), "cons should not be Rc-allocated");

        // Cleanup
        super::clear_arena();
    }

    #[test]
    fn test_arena_cons_clone_is_free() {
        super::set_arena_enabled(true);
        super::clear_arena();

        let cell = Value::cons(Value::int(1), Value::nil());
        assert!(cell.is_arena());

        // Clone should just copy bits (no refcount increment)
        let cell2 = cell.clone();
        assert!(cell2.is_arena());
        assert_eq!(cell.0, cell2.0); // Same bits

        // Both should access the same cons cell
        assert_eq!(cell.as_cons().unwrap().car.as_int(), Some(1));
        assert_eq!(cell2.as_cons().unwrap().car.as_int(), Some(1));

        super::clear_arena();
    }

    #[test]
    fn test_arena_promote() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Create arena-allocated cons
        let cell = Value::cons(Value::int(42), Value::nil());
        assert!(cell.is_arena());

        // Promote to Rc
        let promoted = cell.promote();
        assert!(promoted.is_ptr(), "promoted should be Rc-allocated");
        assert!(!promoted.is_arena());

        // Contents should be preserved
        assert_eq!(promoted.as_cons().unwrap().car.as_int(), Some(42));

        super::clear_arena();
    }

    #[test]
    fn test_arena_promote_nested() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Create nested arena-allocated cons list: (1 2 3)
        let list = Value::cons(
            Value::int(1),
            Value::cons(
                Value::int(2),
                Value::cons(Value::int(3), Value::nil()),
            ),
        );
        assert!(list.is_arena());

        // Promote should recursively promote all nested cons cells
        let promoted = list.promote();
        assert!(promoted.is_ptr());

        // All nested cells should now be Rc
        let cons1 = promoted.as_cons().unwrap();
        assert!(cons1.cdr.is_ptr() || cons1.cdr.is_nil());

        // Contents should be preserved
        assert_eq!(format!("{}", promoted), "(1 2 3)");

        super::clear_arena();
    }

    #[test]
    fn test_arena_disable_falls_back_to_rc() {
        super::set_arena_enabled(false);

        let cell = Value::cons(Value::int(1), Value::nil());
        assert!(cell.is_ptr(), "cons should be Rc-allocated when arena disabled");
        assert!(!cell.is_arena());

        // Re-enable for other tests
        super::set_arena_enabled(true);
    }

    #[test]
    fn test_arena_clear() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Create some arena-allocated values
        let _c1 = Value::cons(Value::int(1), Value::nil());
        let _c2 = Value::cons(Value::int(2), Value::nil());
        let _c3 = Value::cons(Value::int(3), Value::nil());

        assert!(super::arena_size() >= 3);

        // Clear should reset
        super::clear_arena();
        assert_eq!(super::arena_size(), 0);
    }

    #[test]
    fn test_arena_many_allocations() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Allocate many cons cells
        let mut list = Value::nil();
        for i in 0..1000 {
            list = Value::cons(Value::int(i), list);
        }

        // All should be arena-allocated
        assert!(list.is_arena());

        // Traverse to verify structure
        let mut current = &list;
        let mut count = 0;
        while let Some(cons) = current.as_cons() {
            count += 1;
            current = &cons.cdr;
        }
        assert_eq!(count, 1000);

        super::clear_arena();
    }

    #[test]
    fn test_arena_size_limit_fallback() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Fill the arena to the limit
        let limit = super::ARENA_SIZE_LIMIT;
        for i in 0..limit {
            let cell = Value::cons(Value::int(i as i64), Value::nil());
            if i < limit {
                // Should be arena-allocated until we hit the limit
                assert!(cell.is_arena() || cell.is_ptr());
            }
        }

        // Arena should now be full
        assert!(super::ARENA.with(|a| a.borrow().is_full()));

        // Next allocation should fall back to Rc
        let overflow = Value::cons(Value::int(999999), Value::nil());
        assert!(overflow.is_ptr(), "should fall back to Rc when arena is full");
        assert!(!overflow.is_arena());

        // Verify the Rc-allocated value still works
        assert_eq!(overflow.as_cons().unwrap().car.as_int(), Some(999999));

        super::clear_arena();
    }

    #[test]
    fn test_arena_clear_allows_reuse() {
        super::set_arena_enabled(true);
        super::clear_arena();

        // Fill the arena
        for i in 0..super::ARENA_SIZE_LIMIT {
            let _ = Value::cons(Value::int(i as i64), Value::nil());
        }
        assert!(super::ARENA.with(|a| a.borrow().is_full()));

        // Clear arena
        super::clear_arena();
        assert_eq!(super::arena_size(), 0);

        // Should be able to allocate in arena again
        let cell = Value::cons(Value::int(42), Value::nil());
        assert!(cell.is_arena(), "should be arena-allocated after clear");

        super::clear_arena();
    }

    // Conversion trait tests
    #[test]
    fn test_from_bool() {
        let v: Value = true.into();
        assert_eq!(v.as_bool(), Some(true));
        let v: Value = false.into();
        assert_eq!(v.as_bool(), Some(false));
    }

    #[test]
    fn test_from_integers() {
        let v: Value = 42i64.into();
        assert_eq!(v.as_int(), Some(42));

        let v: Value = 42i32.into();
        assert_eq!(v.as_int(), Some(42));

        let v: Value = 42usize.into();
        assert_eq!(v.as_int(), Some(42));
    }

    #[test]
    fn test_from_float() {
        let v: Value = 3.14f64.into();
        assert_eq!(v.as_float(), Some(3.14));
    }

    #[test]
    fn test_from_string() {
        let v: Value = "hello".into();
        assert_eq!(v.as_string(), Some("hello"));

        let v: Value = String::from("world").into();
        assert_eq!(v.as_string(), Some("world"));
    }

    #[test]
    fn test_from_vec() {
        let v: Value = vec![1i64, 2i64, 3i64].into();
        let list = v.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(1));
        assert_eq!(list[2].as_int(), Some(3));
    }

    #[test]
    fn test_from_option() {
        let v: Value = Some(42i64).into();
        assert_eq!(v.as_int(), Some(42));

        let v: Value = None::<i64>.into();
        assert!(v.is_nil());
    }

    #[test]
    fn test_from_unit() {
        let v: Value = ().into();
        assert!(v.is_nil());
    }

    #[test]
    fn test_try_from_bool() {
        use std::convert::TryFrom;

        let v = Value::bool(true);
        assert_eq!(bool::try_from(v), Ok(true));

        let v = Value::int(42);
        assert!(bool::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_i64() {
        use std::convert::TryFrom;

        let v = Value::int(42);
        assert_eq!(i64::try_from(v), Ok(42));

        let v = Value::bool(true);
        assert!(i64::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_i32() {
        use std::convert::TryFrom;

        let v = Value::int(42);
        assert_eq!(i32::try_from(v), Ok(42));

        // Test overflow - use value definitely outside i32 range
        let v = Value::int(i32::MAX as i64 + 1);
        assert!(i32::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_usize() {
        use std::convert::TryFrom;

        let v = Value::int(42);
        assert_eq!(usize::try_from(v), Ok(42));

        // Test negative
        let v = Value::int(-1);
        assert!(usize::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_f64() {
        use std::convert::TryFrom;

        let v = Value::float(3.14);
        assert_eq!(f64::try_from(v), Ok(3.14));

        // Should also work for int -> float
        let v = Value::int(42);
        assert_eq!(f64::try_from(v), Ok(42.0));

        let v = Value::bool(true);
        assert!(f64::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_string() {
        use std::convert::TryFrom;

        let v = Value::string("hello");
        assert_eq!(String::try_from(v), Ok("hello".to_string()));

        let v = Value::int(42);
        assert!(String::try_from(v).is_err());
    }

    #[test]
    fn test_try_from_vec() {
        use std::convert::TryFrom;

        let v = Value::list(vec![Value::int(1), Value::int(2), Value::int(3)]);
        let vec = Vec::<Value>::try_from(v).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0].as_int(), Some(1));

        // Test nil -> empty vec
        let v = Value::nil();
        let vec = Vec::<Value>::try_from(v).unwrap();
        assert_eq!(vec.len(), 0);

        // Test cons cells
        let v = Value::cons(Value::int(1), Value::cons(Value::int(2), Value::nil()));
        let vec = Vec::<Value>::try_from(v).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0].as_int(), Some(1));
        assert_eq!(vec[1].as_int(), Some(2));

        // Test error case
        let v = Value::int(42);
        assert!(Vec::<Value>::try_from(v).is_err());
    }

    #[test]
    fn test_conversion_error_display() {
        let err = super::ConversionError {
            expected: "int",
            got: "string".to_string(),
        };
        assert_eq!(format!("{}", err), "expected int, got string");
    }

    // Arc/Rc Conversion Tests

    #[test]
    fn test_make_shared_primitives() {
        // Test nil
        let v = Value::nil();
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert!(back.is_nil());

        // Test bool
        let v = Value::bool(true);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_bool(), Some(true));

        let v = Value::bool(false);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_bool(), Some(false));

        // Test int
        let v = Value::int(42);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_int(), Some(42));

        // Test float
        let v = Value::float(3.14);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_float(), Some(3.14));
    }

    #[test]
    fn test_make_shared_string() {
        let v = Value::string("hello world");
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_string(), Some("hello world"));
    }

    #[test]
    fn test_make_shared_symbol() {
        let v = Value::symbol("foo");
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);
        assert_eq!(back.as_symbol(), Some("foo"));
    }

    #[test]
    fn test_make_shared_list() {
        let v = Value::list(vec![
            Value::int(1),
            Value::int(2),
            Value::int(3),
        ]);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);

        let items = back.as_list().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].as_int(), Some(1));
        assert_eq!(items[1].as_int(), Some(2));
        assert_eq!(items[2].as_int(), Some(3));
    }

    #[test]
    fn test_make_shared_cons() {
        let v = Value::cons_rc(
            Value::int(1),
            Value::cons_rc(Value::int(2), Value::nil()),
        );
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);

        let cons1 = back.as_cons().unwrap();
        assert_eq!(cons1.car.as_int(), Some(1));

        let cons2 = cons1.cdr.as_cons().unwrap();
        assert_eq!(cons2.car.as_int(), Some(2));
        assert!(cons2.cdr.is_nil());
    }

    #[test]
    fn test_make_shared_nested_list() {
        let v = Value::list(vec![
            Value::int(1),
            Value::list(vec![Value::int(2), Value::int(3)]),
            Value::int(4),
        ]);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);

        let items = back.as_list().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].as_int(), Some(1));

        let nested = items[1].as_list().unwrap();
        assert_eq!(nested.len(), 2);
        assert_eq!(nested[0].as_int(), Some(2));
        assert_eq!(nested[1].as_int(), Some(3));

        assert_eq!(items[2].as_int(), Some(4));
    }

    #[test]
    fn test_make_shared_mixed_types() {
        let v = Value::list(vec![
            Value::int(42),
            Value::string("hello"),
            Value::symbol("foo"),
            Value::bool(true),
        ]);
        let shared = v.make_shared().unwrap();
        let back = Value::from_shared(&shared);

        let items = back.as_list().unwrap();
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].as_int(), Some(42));
        assert_eq!(items[1].as_string(), Some("hello"));
        assert_eq!(items[2].as_symbol(), Some("foo"));
        assert_eq!(items[3].as_bool(), Some(true));
    }

    #[test]
    fn test_make_shared_function_errors() {
        use crate::eval::Env;

        let env = Env::new();
        let func = Function {
            params: vec!["x".to_string()],
            body: Value::symbol("x"),
            env: env.clone(),
        };
        let v = Value::function(func);

        // Functions cannot be shared across threads
        let result = v.make_shared();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be shared"));
    }

    #[test]
    fn test_shared_value_display() {
        let v = Value::list(vec![Value::int(1), Value::int(2), Value::int(3)]);
        let shared = v.make_shared().unwrap();
        let display = format!("{}", shared);

        // Should display as a list
        assert!(display.contains("1"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_shared_value_clone() {
        let v = Value::int(42);
        let shared1 = v.make_shared().unwrap();
        let shared2 = shared1.clone();

        // Both should convert back to the same value
        let back1 = Value::from_shared(&shared1);
        let back2 = Value::from_shared(&shared2);
        assert_eq!(back1.as_int(), Some(42));
        assert_eq!(back2.as_int(), Some(42));
    }
}
