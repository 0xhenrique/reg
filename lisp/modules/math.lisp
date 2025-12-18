; Math module

(module math
  (export square cube abs double add negate)

  (def square (fn (x) (* x x)))

  (def cube (fn (x) (* x x x)))

  (def abs (fn (x) (if (< x 0) (- 0 x) x)))

  (def double (fn (x) (* x 2)))

  (def add (fn (a b) (+ a b)))

  (def negate (fn (x) (- 0 x)))

  ; Private helper - not exported
  (def internal-constant 42))
