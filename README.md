# Register based VM

A Lisp interpreter with a register-based bytecode virtual machine.

## Features

- **Register-based bytecode VM** - Inspired by Lua's design
- **Compile-time optimizations** - Constant folding and pure function inlining
- **Tail call optimization** - Write recursive functions without stack overflow (citation needed)
- **Macro system** - Compile-time code generation with `defmacro`
- **Reference counting** - Predictable memory management without garbage collection pauses*
- **Immutable by default** - Values are immutable, inspired by FP
- **Fast startup** - Minimal overhead, just feed the VM with code

## Installation

### Building from Source

```bash
~$ git clone https://github.com/0xhenrique/code-reg.git
~$ cd reg
~$ cargo build --release
```

## How to use

### Running a Lisp File

```bash
./target/release/lisp-vm your-program.lisp
```

### REPL

```bash
./target/release/lisp-vm
```

Try to experiment a bit with it. Type `:q` or `:quit` to exit.

### CLI Options

```bash
lisp-vm [OPTIONS] [FILE]

Options:
  --arena    Enable arena allocation for cons cells (experimental)
```

## Examples

### Basic Arithmetic

```lisp
; Addition, subtraction, multiplication, division
(+ 1 2 3)        ; => 6
(* 4 5)          ; => 20
(- 10 3)         ; => 7
(/ 20 4)         ; => 5

; Comparisons
(< 5 10)         ; => true
(>= 5 5)         ; => true
(= 2 2)          ; => true
```

### Variables and Functions

```lisp
; Define a variable
(def x 42)

; Define a function
(def square (fn (n) (* n n)))
(square 5)       ; => 25

; Functions with multiple statements using 'do'
(def greet (fn (name)
  (do
    (println "Hello")
    name)))
(greet "World")  ; prints "Hello", returns "World"
```

### Let Bindings

```lisp
; Create local bindings
(let (x 10 y 20)
  (+ x y))       ; => 30

; Nested let bindings
(let (x 5)
  (let (y (* x 2))
    (+ x y)))    ; => 15
```

### Conditionals

```lisp
; if expressions
(if (< 5 10)
    "yes"
    "no")        ; => "yes"

; if without else returns nil
(if false "won't happen")  ; => nil
```

### Recursion

The VM supports kinda recursion with TCO. This is not battle tested:

```lisp
; Regular recursion
(def factorial (fn (n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1))))))
(factorial 5)    ; => 120
(factorial 10)   ; => 3628800

; Tail-recursive function
(def sum (fn (n acc)
  (if (<= n 0)
      acc
      (sum (- n 1) (+ acc n)))))
(sum 100 0)      ; => 5050
(sum 10000 0)    ; => 50005000 (no stack overflow!)
```

### Lists

```lisp
; Create a list
(list 1 2 3 4 5)           ; => (1 2 3 4 5)

; car and cdr (first and rest)
(car (list 1 2 3))         ; => 1
(cdr (list 1 2 3))         ; => (2 3)

; cons (prepend element)
(cons 0 (list 1 2 3))      ; => (0 1 2 3)

; length
(length (list 1 2 3 4 5))  ; => 5

; Quoting
'(1 2 3)                   ; => (1 2 3)
```

### Higher-Order Functions

```lisp
; Functions are first-class values
(def apply-twice (fn (f x)
  (f (f x))))

(def add-one (fn (x) (+ x 1)))
(apply-twice add-one 5)    ; => 7

; Functions can be returned
(def make-adder (fn (n)
  (fn (x) (+ x n))))
(def add-ten (make-adder 10))
(add-ten 5)                ; => 15
```

### Macros

Macros enable compile-time code generation:

```lisp
; Define a macro
(defmacro unless (cond body)
  (list 'if cond nil body))

; Use the macro
(unless false (println "This prints!"))
(unless true (println "This does not print"))

; Macros expand at compile time
(defmacro when (cond body)
  (list 'if cond body nil))

(when (> 5 3) (println "5 is greater than 3"))
```

### Built-in Functions

#### Arithmetic
- `+`, `-`, `*`, `/`, `mod` - Arithmetic operations (support multiple arguments)

#### Comparison
- `<`, `<=`, `>`, `>=`, `=`, `!=` - Comparison operations

#### Logic
- `not` - Logical negation

#### List Operations
- `list` - Create a list
- `cons` - Prepend element to list
- `car` - Get first element (`first` or `head`)
- `cdr` - Get rest of list (`rest` or `tail`)
- `length` - Get list length

#### Type Predicates
- `nil?`, `int?`, `float?`, `string?`, `list?`, `fn?`, `symbol?` - Type checking

#### I/O
- `print` - Print value without newline
- `println` - Print value with newline

#### Special Forms
- `def` - Define global variable
- `fn` - Create anonymous function
- `if` - Conditional expression
- `let` - Create local bindings
- `do` - Execute multiple expressions, return last one
- `quote` (or `'`) - Quote expression without evaluation
- `defmacro` - Define compile-time macro
- `gensym` - Generate unique symbol (for hygienic macros)

## Example Programs

### Fibonacci (Tree Recursion)

```lisp
(def fib (fn (n)
  (if (<= n 1)
      n
      (+ (fib (- n 1)) (fib (- n 2))))))

(println (fib 10))  ; => 55
```

### Sum of List

```lisp
(def sum-list (fn (lst)
  (if (nil? lst)
      0
      (+ (car lst) (sum-list (cdr lst))))))

(sum-list (list 1 2 3 4 5))  ; => 15
```

### Map Function

```lisp
(def map (fn (f lst)
  (if (nil? lst)
      nil
      (cons (f (car lst)) (map f (cdr lst))))))

(def double (fn (x) (* x 2)))
(map double (list 1 2 3 4))  ; => (2 4 6 8)
```

## Performance

The VM uses some common optimization techniques:

- **Compile-time constant folding**: Expressions like `(+ 1 2 3)` compile directly to `6`
- **Pure function inlining**: Pure functions with constant arguments are evaluated at compile time
- **Register-based bytecode**: Reduced memory traffic
- **Specialized opcodes**: Direct arithmetic operations bypass function call overhead
- **Tail call optimization**: Recursive tail calls reuse stack frames

This is a rabbit hole by itself. I'm not expecting to beat something like Lua.

## Acknowledgments

Inspired by:
- **Lua** - For the register-based VM design
- **Crafting Interpreters** - For VM implementation techniques
- **[Tsoding - Lisp in C](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VYbTFO5tBwxG4Q20BJuqXD_)** - The series of videos that inspired me into learning more about VM
- **A ton of blogs posts, articles, videos and Github issues** - Thank you a lot!
