; Test basic functionality

; Arithmetic
(println "Testing arithmetic...")
(def result (+ 1 2 3))
(if (= result 6)
    (println "  + works: passed")
    (println "  + works: FAILED"))

(if (= (* 4 5) 20)
    (println "  * works: passed")
    (println "  * works: FAILED"))

(if (= (- 10 3) 7)
    (println "  - works: passed")
    (println "  - works: FAILED"))

; Comparison
(println "Testing comparison...")
(if (< 1 2)
    (println "  < works: passed")
    (println "  < works: FAILED"))

(if (>= 5 5)
    (println "  >= works: passed")
    (println "  >= works: FAILED"))

; Functions
(println "Testing functions...")
(def square (fn (n) (* n n)))
(if (= (square 5) 25)
    (println "  fn works: passed")
    (println "  fn works: FAILED"))

; Let bindings
(println "Testing let...")
(def let-result (let (x 10 y 20) (+ x y)))
(if (= let-result 30)
    (println "  let works: passed")
    (println "  let works: FAILED"))

; Recursion
(println "Testing recursion...")
(def factorial (fn (n)
    (if (<= n 1)
        1
        (* n (factorial (- n 1))))))
(if (= (factorial 5) 120)
    (println "  recursion works: passed")
    (println "  recursion works: FAILED"))

(println "")
(println "All basic tests completed!")
