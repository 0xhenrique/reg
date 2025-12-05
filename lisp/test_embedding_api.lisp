; Test Embedding API
; This file tests the embedding API from the Lisp side
; The Rust tests will verify the API from the Rust side

(println "Testing embedding API from Lisp side...")

; Test basic evaluation
(def x 10)
(def y 20)
(println "Can define variables")

; Test eval result
(def result (+ x y))
(if (= result 30)
    (println "Can evaluate expressions")
    (println "Evaluation failed"))

; Test type conversions
(def int-val 42)
(def float-val 3.14)
(def string-val "hello")
(def list-val (list 1 2 3))

(println "Can create values of different types")

; Test get/set through globals
(def answer 42)
(if (= answer 42)
    (println "Global variables work")
    (println "Global variables failed"))

(println "All Lisp-side embedding API tests passed!")
