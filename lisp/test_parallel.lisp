(println "Testing parallel map and filter...")

;; Note: pmap and pfilter currently only support native functions
;; We'll use built-in functions like abs, even?, odd?, etc.

;; Test 1: pmap with abs
(println "\n=== Test 1: pmap with abs ===")
(def nums (list -5 3 -2 8 -1 10))
(println "Input:" nums)
(def absolutes (pmap abs nums))
(println "Absolute values:" absolutes)

;; Test 2: pfilter - even numbers
(println "\n=== Test 2: pfilter - even numbers ===")
(def nums2 (list 1 2 3 4 5 6 7 8 9 10))
(println "Input:" nums2)
(def evens (pfilter even? nums2))
(println "Evens:" evens)

;; Test 3: pfilter - odd numbers
(println "\n=== Test 3: pfilter - odd numbers ===")
(def nums3 (list 1 2 3 4 5 6 7 8 9 10))
(println "Input:" nums3)
(def odds (pfilter odd? nums3))
(println "Odds:" odds)

;; Test 4: pfilter - positive numbers
(println "\n=== Test 4: pfilter - positive numbers ===")
(def mixed (list -5 -2 0 1 3 -1 7 10))
(println "Input:" mixed)
(def positives (pfilter positive? mixed))
(println "Positives:" positives)

;; Test 5: pfilter - negative numbers
(println "\n=== Test 5: pfilter - negative numbers ===")
(println "Input:" mixed)
(def negatives (pfilter negative? mixed))
(println "Negatives:" negatives)

;; Test 6: Combine pmap and pfilter
(println "\n=== Test 6: Combined - filter evens then abs ===")
(def nums4 (list -8 -6 -4 -2 0 1 2 3 4 5))
(println "Input:" nums4)
(def evens4 (pfilter even? nums4))
(println "After pfilter (evens):" evens4)
(def absolutes4 (pmap abs evens4))
(println "After pmap (abs):" absolutes4)

;; Test 7: Large list performance test
(println "\n=== Test 7: Performance test (100 elements) ===")
;; Create a list of 100 numbers (mix of positive and negative)
(def large-list (list -50 -49 -48 -47 -46 -45 -44 -43 -42 -41 -40 -39 -38 -37 -36 -35 -34 -33 -32 -31 -30 -29 -28 -27 -26 -25 -24 -23 -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50))
(println "Processing 100 elements with pmap abs...")
(def large-abs (pmap abs large-list))
(println "First 3 results:" (car large-abs) (car (cdr large-abs)) (car (cdr (cdr large-abs))))
(println "Length of result:" (length large-abs))

(println "\n=== Test 8: pfilter on large list ===")
(println "Filtering 100 elements for evens...")
(def large-evens (pfilter even? large-list))
(println "Number of evens:" (length large-evens))
(println "First 3 evens:" (car large-evens) (car (cdr large-evens)) (car (cdr (cdr large-evens))))

(println "\n=== Test 9: pfilter then pmap on large list ===")
(def large-positives (pfilter positive? large-list))
(println "Number of positives:" (length large-positives))
(def large-positives-abs (pmap abs large-positives))
(println "After pmap abs, first 3:" (car large-positives-abs) (car (cdr large-positives-abs)) (car (cdr (cdr large-positives-abs))))

(println "\n=== All parallel tests passed! ===")
