(println "Testing macros...")

; Defining an 'unless' macro
(defmacro unless (cond body)
    (list 'if cond nil body))

(if (= (unless false 42) 42)
    (println "  unless macro: passed")
    (println "  unless macro: FAILED"))

; Defining an 'inc' macro
(defmacro inc (x)
    (list '+ x 1))

(if (= (inc 5) 6)
    (println "  inc macro: passed")
    (println "  inc macro: FAILED"))

; Defining a 'double' macro
(defmacro double (x)
    (list '* x 2))

(if (= (double 10) 20)
    (println "  double macro: passed")
    (println "  double macro: FAILED"))

; Nested macro expansion
(defmacro inc-twice (x)
    (list 'inc (list 'inc x)))

(if (= (inc-twice 5) 7)
    (println "  nested macros: passed")
    (println "  nested macros: FAILED"))

(println "")
(println "All macro tests completed!")
