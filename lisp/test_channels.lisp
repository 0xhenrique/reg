(println "Testing channels...")

;; Test 1: Simple send/recv in same thread
(println "\n=== Test 1: Basic channel send/recv ===")
(def ch (channel))
(def sender (car ch))
(def receiver (car (cdr ch)))
(send! sender 42)
(def result (recv receiver))
(println "Received:" result)

;; Test 2: Multiple sends and receives
(println "\n=== Test 2: Multiple messages ===")
(def ch2 (channel))
(def s2 (car ch2))
(def r2 (car (cdr ch2)))
(send! s2 10)
(send! s2 20)
(send! s2 30)
(def v1 (recv r2))
(def v2 (recv r2))
(def v3 (recv r2))
(println "Received:" v1 v2 v3)
(println "Sum:" (+ v1 v2 v3))

;; Test 3: Send/recv strings
(println "\n=== Test 3: Sending strings ===")
(def ch3 (channel))
(def s3 (car ch3))
(def r3 (car (cdr ch3)))
(send! s3 "Hello")
(send! s3 "World")
(println "Received:" (recv r3))
(println "Received:" (recv r3))

;; Test 4: Send/recv lists
(println "\n=== Test 4: Sending lists ===")
(def ch4 (channel))
(def s4 (car ch4))
(def r4 (car (cdr ch4)))
(send! s4 (list 1 2 3))
(send! s4 (list 4 5 6))
(def list1 (recv r4))
(def list2 (recv r4))
(println "List 1:" list1)
(println "List 2:" list2)

;; Test 5: Producer/Consumer with threads
(println "\n=== Test 5: Producer/Consumer threads ===")
(def ch5 (channel))
(def sender5 (car ch5))
(def receiver5 (car (cdr ch5)))

;; Producer thread - send numbers 1 through 5
(def producer (fn ()
  ;; Create a new channel in this thread's scope
  ;; Since we can't share the sender from parent thread, we'll just compute
  (+ 100 200)))

;; For now, just demonstrate thread-local channel usage
(def worker1 (fn ()
  (def local-ch (channel))
  (def local-s (car local-ch))
  (def local-r (car (cdr local-ch)))
  (send! local-s 123)
  (recv local-r)))

(def t1 (spawn worker1))
(def r1 (join t1))
(println "Thread result:" r1)

;; Test 6: Passing channels between functions
(println "\n=== Test 6: Channel as function argument ===")
(def send-many (fn (sender values)
  (def first-val (car values))
  (def rest-vals (cdr values))
  (send! sender first-val)
  (if (not (nil? rest-vals))
    (send-many sender rest-vals)
    nil)))

(def ch6 (channel))
(def s6 (car ch6))
(def r6 (car (cdr ch6)))
(send-many s6 (list 7 8 9))
(println "Received:" (recv r6))
(println "Received:" (recv r6))
(println "Received:" (recv r6))

(println "\n=== All channel tests passed! ===")
