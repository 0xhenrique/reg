(println "Testing atoms...")

;; Test 1: Basic atom creation and deref
(println "\n=== Test 1: Create and deref ===")
(def counter (atom 0))
(println "Initial value:" (deref counter))

;; Test 2: reset! to change value
(println "\n=== Test 2: reset! ===")
(reset! counter 42)
(println "After reset:" (deref counter))
(reset! counter 100)
(println "After second reset:" (deref counter))

;; Test 3: swap! with increment
(println "\n=== Test 3: swap! with increment ===")
(def num (atom 10))
(println "Initial:" (deref num))
(swap! num + 5)
(println "After (swap! num + 5):" (deref num))
(swap! num + 10)
(println "After (swap! num + 10):" (deref num))

;; Test 4: swap! with multiply
(println "\n=== Test 4: swap! with multiply ===")
(def val (atom 3))
(println "Initial:" (deref val))
(swap! val * 2)
(println "After (swap! val * 2):" (deref val))
(swap! val * 5)
(println "After (swap! val * 5):" (deref val))

;; Test 5: Atoms with strings
(println "\n=== Test 5: Atoms with strings ===")
(def name (atom "Alice"))
(println "Name:" (deref name))
(reset! name "Bob")
(println "After reset:" (deref name))

;; Test 6: Atoms with lists
(println "\n=== Test 6: Atoms with lists ===")
(def items (atom (list 1 2 3)))
(println "Items:" (deref items))
(reset! items (list 4 5 6))
(println "After reset:" (deref items))

;; Test 7: Multiple atoms
(println "\n=== Test 7: Multiple atoms ===")
(def x (atom 1))
(def y (atom 2))
(def z (atom 3))
(println "x:" (deref x) "y:" (deref y) "z:" (deref z))
(swap! x + 10)
(swap! y + 20)
(swap! z + 30)
(println "After swaps - x:" (deref x) "y:" (deref y) "z:" (deref z))
(def sum (+ (deref x) (deref y) (deref z)))
(println "Sum:" sum)

;; Test 8: Atom shared between threads (each thread gets its own reference)
(println "\n=== Test 8: Thread-local atom usage ===")
(def worker (fn ()
  (def local-atom (atom 0))
  (swap! local-atom + 1)
  (swap! local-atom + 2)
  (swap! local-atom + 3)
  (deref local-atom)))

(def t1 (spawn worker))
(def result (join t1))
(println "Thread result:" result)

;; Test 9: swap! with subtraction
(println "\n=== Test 9: swap! with subtraction ===")
(def balance (atom 100))
(println "Balance:" (deref balance))
(swap! balance - 30)
(println "After withdrawal:" (deref balance))
(swap! balance + 50)
(println "After deposit:" (deref balance))

;; Test 10: Displaying atoms
(println "\n=== Test 10: Display atoms ===")
(def a1 (atom 42))
(def a2 (atom "hello"))
(def a3 (atom (list 1 2 3)))
(println "Atom with int:" a1)
(println "Atom with string:" a2)
(println "Atom with list:" a3)

(println "\n=== All atom tests passed! ===")
