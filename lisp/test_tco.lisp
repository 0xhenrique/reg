; Test tail call optimization

(println "Testing tail call optimization...")

; Sum function with accumulator (tail recursive)
(def sum (fn (n acc)
    (if (<= n 0)
        acc
        (sum (- n 1) (+ acc n)))))

; This would stack overflow without TCO
(def result (sum 10000 0))
(if (= result 50005000)
    (println "  tail call sum(10000): passed")
    (println "  tail call sum(10000): FAILED"))

; Even/odd mutual recursion
(def even? (fn (n)
    (if (= n 0)
        true
        (odd? (- n 1)))))

(def odd? (fn (n)
    (if (= n 0)
        false
        (even? (- n 1)))))

(if (even? 100)
    (println "  mutual recursion even?(100): passed")
    (println "  mutual recursion even?(100): FAILED"))

(if (odd? 101)
    (println "  mutual recursion odd?(101): passed")
    (println "  mutual recursion odd?(101): FAILED"))

(println "")
(println "All TCO tests completed!")
