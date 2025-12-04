(println "Testing pure function detection...")

; Defining a pure function
(def square (fn (n) (* n n)))

; This should work (pure function with constant arg could be folded)
(if (= (square 5) 25)
    (println "  square(5) = 25: passed")
    (println "  square(5) = 25: FAILED"))

; Nested pure functions
(def double (fn (x) (* x 2)))
(def quad (fn (x) (double (double x))))

(if (= (quad 3) 12)
    (println "  quad(3) = 12: passed")
    (println "  quad(3) = 12: FAILED"))

; Pure function with conditional
(def abs (fn (n) (if (< n 0) (- n) n)))

(if (= (abs -5) 5)
    (println "  abs(-5) = 5: passed")
    (println "  abs(-5) = 5: FAILED"))

(if (= (abs 5) 5)
    (println "  abs(5) = 5: passed")
    (println "  abs(5) = 5: FAILED"))

; Pure function with let
(def sum-squares (fn (a b)
    (let (a2 (* a a)
          b2 (* b b))
        (+ a2 b2))))

(if (= (sum-squares 3 4) 25)
    (println "  sum-squares(3,4) = 25: passed")
    (println "  sum-squares(3,4) = 25: FAILED"))

(println "")
(println "All pure function tests completed!")
