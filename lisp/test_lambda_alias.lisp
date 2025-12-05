; Test lambda as an alias for fn

(println "Testing lambda alias...")

; Test 1: lambda and fn produce identical functions
(def square-fn (fn (n) (* n n)))
(def square-lambda (lambda (n) (* n n)))

(if (= (square-fn 5) (square-lambda 5))
    (println "lambda and fn produce identical results")
    (println "lambda and fn differ"))

; Test 2: lambda with multiple parameters
(def add (lambda (a b) (+ a b)))
(if (= (add 3 4) 7)
    (println "lambda with multiple parameters works")
    (println "lambda with multiple parameters failed"))

; Test 3: lambda in higher-order functions
(def apply-twice (lambda (f x) (f (f x))))
(def double (lambda (n) (* n 2)))
(if (= (apply-twice double 3) 12)
    (println "lambda works in higher-order functions")
    (println "lambda in higher-order functions failed"))

; Test 4: lambda with tail recursion
(def sum-lambda (lambda (n acc)
  (if (<= n 0)
      acc
      (sum-lambda (- n 1) (+ acc n)))))

(if (= (sum-lambda 100 0) 5050)
    (println "lambda supports tail recursion")
    (println "lambda tail recursion failed"))

; Test 5: Mixed usage - fn calling lambda and vice versa
(def fn-func (fn (x) (* x 2)))
(def lambda-func (lambda (x) (fn-func (+ x 1))))
(if (= (lambda-func 4) 10)
    (println "fn and lambda interoperate seamlessly")
    (println "fn/lambda interop failed"))

(println "All lambda alias tests passed!")
