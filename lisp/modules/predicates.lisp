; Predicates module

(module predicates
  (export positive? negative? zero? even? odd?)

  (def positive? (fn (x) (> x 0)))

  (def negative? (fn (x) (< x 0)))

  (def zero? (fn (x) (= x 0)))

  (def even? (fn (x) (= (mod x 2) 0)))

  (def odd? (fn (x) (= (mod x 2) 1))))
