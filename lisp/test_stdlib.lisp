; Test Standard Library Functions
; Tests for logic operators, list functions, string functions, and I/O

; ============================================
; Test short-circuiting logic operators
; ============================================

(println "Testing logic operators...")

; Test: and with no arguments
(def test1 (and))
(if (= test1 true)
    (println "✓ (and) => true")
    (println "✗ (and) failed"))

; Test: and with one argument
(def test2 (and 42))
(if (= test2 42)
    (println "✓ (and 42) => 42")
    (println "✗ (and 42) failed"))

; Test: and short-circuits on false
(def test3 (and true false (/ 1 0)))  ; Should not divide by zero
(if (= test3 false)
    (println "✓ (and true false ...) short-circuits")
    (println "✗ (and) short-circuit failed"))

; Test: and returns last value when all true
(def test4 (and 1 2 3))
(if (= test4 3)
    (println "✓ (and 1 2 3) => 3")
    (println "✗ (and 1 2 3) failed"))

; Test: or with no arguments
(def test5 (or))
(if (= test5 false)
    (println "✓ (or) => false")
    (println "✗ (or) failed"))

; Test: or with one argument
(def test6 (or 42))
(if (= test6 42)
    (println "✓ (or 42) => 42")
    (println "✗ (or 42) failed"))

; Test: or short-circuits on true
(def test7 (or false true (/ 1 0)))  ; Should not divide by zero
(if (= test7 true)
    (println "✓ (or false true ...) short-circuits")
    (println "✗ (or) short-circuit failed"))

; Test: or returns last value when all false
(def test8 (or false nil false))
(if (= test8 false)
    (println "✓ (or false nil false) => false")
    (println "✗ (or false nil false) failed"))

; ============================================
; Test list functions
; ============================================

(println "\nTesting list functions...")

; Test: nth on list
(def lst (list 10 20 30 40 50))
(def test9 (nth lst 0))
(if (= test9 10)
    (println "✓ (nth lst 0) => 10")
    (println "✗ (nth lst 0) failed"))

(def test10 (nth lst 2))
(if (= test10 30)
    (println "✓ (nth lst 2) => 30")
    (println "✗ (nth lst 2) failed"))

; Test: append
(def test11 (append (list 1 2) (list 3 4) (list 5 6)))
(def test11-len (length test11))
(if (= test11-len 6)
    (println "✓ (append ...) => list of length 6")
    (println "✗ (append) failed"))

; Test: reverse
(def test12 (reverse (list 1 2 3 4 5)))
(def test12-first (nth test12 0))
(if (= test12-first 5)
    (println "✓ (reverse (list 1 2 3 4 5)) => (5 4 3 2 1)")
    (println "✗ (reverse) failed"))

; ============================================
; Test string functions
; ============================================

(println "\nTesting string functions...")

; Test: string-length
(def test13 (string-length "hello"))
(if (= test13 5)
    (println "✓ (string-length \"hello\") => 5")
    (println "✗ (string-length) failed"))

; Test: string-append
(def test14 (string-append "hello" " " "world"))
(if (= test14 "hello world")
    (println "✓ (string-append ...) => \"hello world\"")
    (println "✗ (string-append) failed"))

; Test: substring
(def test15 (substring "hello world" 0 5))
(if (= test15 "hello")
    (println "✓ (substring \"hello world\" 0 5) => \"hello\"")
    (println "✗ (substring) failed"))

(def test16 (substring "hello world" 6 11))
(if (= test16 "world")
    (println "✓ (substring \"hello world\" 6 11) => \"world\"")
    (println "✗ (substring) failed"))

; Test: string->list
(def test17 (string->list "abc"))
(def test17-len (length test17))
(if (= test17-len 3)
    (println "✓ (string->list \"abc\") => list of length 3")
    (println "✗ (string->list) failed"))

; Test: list->string
(def test18 (list->string (list "h" "e" "l" "l" "o")))
(if (= test18 "hello")
    (println "✓ (list->string ...) => \"hello\"")
    (println "✗ (list->string) failed"))

; Test: format
(def test19 (format "Hello, ~a!" "world"))
(if (= test19 "Hello, world!")
    (println "✓ (format \"Hello, ~a!\" \"world\") => \"Hello, world!\"")
    (println "✗ (format) failed"))

(def test20 (format "The answer is ~a" 42))
(if (= test20 "The answer is 42")
    (println "✓ (format \"The answer is ~a\" 42) => \"The answer is 42\"")
    (println "✗ (format) failed"))

; ============================================
; Test I/O functions
; ============================================

(println "\nTesting I/O functions...")

; Test: write-file and read-file
(def test-content "Hello from Lisp VM!\nLine 2\nLine 3")
(write-file "/tmp/lisp-test.txt" test-content)
(def read-content (read-file "/tmp/lisp-test.txt"))
(if (= read-content test-content)
    (println "✓ write-file and read-file work correctly")
    (println "✗ write-file or read-file failed"))

(println "\nAll standard library tests completed!")
