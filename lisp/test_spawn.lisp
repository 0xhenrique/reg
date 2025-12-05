;; Test spawn and join - Phase 9 Concurrency

(println "Testing spawn and join...")

;; Test 1: Simple spawn/join - just return a value
(println "\n=== Test 1: Simple constant ===")
(def worker1 (fn () 42))
(def handle1 (spawn worker1))
(def result1 (join handle1))
(println "Result:" result1)

;; Test 2: Simple arithmetic
(println "\n=== Test 2: Simple arithmetic ===")
(def worker2 (fn () (+ (* 6 7) 10)))
(def handle2 (spawn worker2))
(def result2 (join handle2))
(println "Result:" result2)

;; Test 3: Working with lists (built-in functions only)
(println "\n=== Test 3: List operations ===")
(def worker3 (fn ()
  (def nums (list 1 2 3 4 5))
  (def first-item (car nums))
  (def rest-items (cdr nums))
  (+ first-item (car rest-items))))
(def handle3 (spawn worker3))
(def result3 (join handle3))
(println "Result:" result3)

;; Test 4: Multiple independent threads
(println "\n=== Test 4: Three parallel computations ===")
(def compute1 (fn () (+ 10 20 30)))
(def compute2 (fn () (* 5 7)))
(def compute3 (fn () (- 100 25)))

(def t1 (spawn compute1))
(def t2 (spawn compute2))
(def t3 (spawn compute3))

(println "Threads spawned, joining...")
(def r1 (join t1))
(def r2 (join t2))
(def r3 (join t3))

(println "Thread 1:" r1)
(println "Thread 2:" r2)
(println "Thread 3:" r3)
(println "Sum:" (+ r1 r2 r3))

;; Test 5: Thread returning a list
(println "\n=== Test 5: Thread returning list ===")
(def list-worker (fn ()
  (list 1 2 3 4 5)))
(def list-handle (spawn list-worker))
(def list-result (join list-handle))
(println "List from thread:" list-result)
(println "Length:" (length list-result))

(println "\n=== All spawn/join tests passed! ===")
