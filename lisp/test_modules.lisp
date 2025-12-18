; Test module system functionality
; This test demonstrates multi-file module loading and interop

; ============================================================================
; Load standard library and modules from separate files
; ============================================================================
(load "lisp/stdlib.lisp")
(load "lisp/modules/math.lisp")
(load "lisp/modules/predicates.lisp")
(load "lisp/modules/greetings.lisp")

; ============================================================================
; Test 1: Basic qualified access
; ============================================================================
(println "Test 1: Qualified access (module/function)")

(if (= (math/square 5) 25)
    (println "  math/square: passed")
    (println "  math/square: FAILED"))

(if (= (math/cube 3) 27)
    (println "  math/cube: passed")
    (println "  math/cube: FAILED"))

(if (= (math/abs -42) 42)
    (println "  math/abs: passed")
    (println "  math/abs: FAILED"))

(if (= (math/double 7) 14)
    (println "  math/double: passed")
    (println "  math/double: FAILED"))

(if (= (math/negate 5) -5)
    (println "  math/negate: passed")
    (println "  math/negate: FAILED"))

; ============================================================================
; Test 2: Selective import
; ============================================================================
(println "")
(println "Test 2: Selective import")

(import math (square cube))

(if (= (square 6) 36)
    (println "  imported square: passed")
    (println "  imported square: FAILED"))

(if (= (cube 4) 64)
    (println "  imported cube: passed")
    (println "  imported cube: FAILED"))

; Original qualified access still works after import
(if (= (math/abs -10) 10)
    (println "  qualified still works: passed")
    (println "  qualified still works: FAILED"))

; ============================================================================
; Test 3: Predicates module
; ============================================================================
(println "")
(println "Test 3: Predicates module")

(if (predicates/positive? 5)
    (println "  predicates/positive?: passed")
    (println "  predicates/positive?: FAILED"))

(if (predicates/negative? -3)
    (println "  predicates/negative?: passed")
    (println "  predicates/negative?: FAILED"))

(if (predicates/zero? 0)
    (println "  predicates/zero?: passed")
    (println "  predicates/zero?: FAILED"))

(if (predicates/even? 4)
    (println "  predicates/even?: passed")
    (println "  predicates/even?: FAILED"))

(if (predicates/odd? 7)
    (println "  predicates/odd?: passed")
    (println "  predicates/odd?: FAILED"))

; ============================================================================
; Test 4: Import from multiple modules
; ============================================================================
(println "")
(println "Test 4: Import from multiple modules")

(import predicates (even? positive?))

(if (even? 10)
    (println "  imported even?: passed")
    (println "  imported even?: FAILED"))

(if (positive? 42)
    (println "  imported positive?: passed")
    (println "  imported positive?: FAILED"))

; ============================================================================
; Test 5: Composing functions from different modules
; ============================================================================
(println "")
(println "Test 5: Composing module functions")

; square of absolute value
(def square-abs (fn (x) (math/square (math/abs x))))

(if (= (square-abs -5) 25)
    (println "  square-abs: passed")
    (println "  square-abs: FAILED"))

; check if double is positive
(def double-positive? (fn (x) (predicates/positive? (math/double x))))

(if (double-positive? 3)
    (println "  double-positive?: passed")
    (println "  double-positive?: FAILED"))

; ============================================================================
; Test 6: Module values
; ============================================================================
(println "")
(println "Test 6: Module as value")

; Modules are first-class values
(def m math)
(println (list "  module value: " m))

; ============================================================================
; Test 7: Nested function calls with modules
; ============================================================================
(println "")
(println "Test 7: Nested calls")

(if (= (math/square (math/abs -5)) 25)
    (println "  nested calls: passed")
    (println "  nested calls: FAILED"))

(if (= (math/cube (math/double 2)) 64)  ; cube(double(2)) = cube(4) = 64
    (println "  cube of double: passed")
    (println "  cube of double: FAILED"))

; ============================================================================
; Test 8: Using modules with higher-order functions (from stdlib)
; ============================================================================
(println "")
(println "Test 8: Modules with higher-order functions (using stdlib)")

; Map with module function - check individual elements
(def squares (map math/square (list 1 2 3 4 5)))
(if (and (= (car squares) 1)
         (= (car (cdr squares)) 4)
         (= (car (cdr (cdr squares))) 9))
    (println "  map with math/square: passed")
    (println "  map with math/square: FAILED"))

; Filter with module predicate - check result
(def evens (filter predicates/even? (list 1 2 3 4 5 6)))
(if (and (= (car evens) 2)
         (= (car (cdr evens)) 4)
         (= (car (cdr (cdr evens))) 6))
    (println "  filter with even?: passed")
    (println "  filter with even?: FAILED"))

; Filter with positive?
(def positives (filter predicates/positive? (list -2 -1 0 1 2)))
(if (and (= (car positives) 1)
         (= (car (cdr positives)) 2))
    (println "  filter with positive?: passed")
    (println "  filter with positive?: FAILED"))

; Fold with module function
(def sum-of-squares (fold (fn (acc x) (+ acc (math/square x))) 0 (list 1 2 3 4)))
(if (= sum-of-squares 30)  ; 1 + 4 + 9 + 16 = 30
    (println "  fold with math/square: passed")
    (println "  fold with math/square: FAILED"))

; ============================================================================
; Test 9: Greetings module (side effects)
; ============================================================================
(println "")
(println "Test 9: Greetings module")
(greetings/hello "World")
(greetings/goodbye "World")
(greetings/welcome "Alice" "Lisp Land")
(println "  greetings module: passed (check output above)")

; ============================================================================
; Summary
; ============================================================================
(println "All module tests completed!")
(println "This test loaded modules from separate files:")
(println "  - lisp/stdlib.lisp (map, filter, fold)")
(println "  - lisp/modules/math.lisp")
(println "  - lisp/modules/predicates.lisp")
(println "  - lisp/modules/greetings.lisp")
(println "")
(println "Note: Private functions are not accessible.")
(println "Trying (math/internal-constant) would error:")
(println "  'internal-constant' is not exported from module 'math'")
