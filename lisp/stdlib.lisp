; Standard Library for Lisp VM
; Higher-order functions that can't be efficiently implemented in native code

; Map: apply function to each element of a list
(def map (fn (f lst)
    (if (nil? lst)
        nil
        (cons (f (car lst)) (map f (cdr lst))))))

; Filter: keep only elements that satisfy the predicate
(def filter (fn (pred lst)
    (if (nil? lst)
        nil
        (if (pred (car lst))
            (cons (car lst) (filter pred (cdr lst)))
            (filter pred (cdr lst))))))

; Fold (reduce): combine elements from left to right
(def fold (fn (f init lst)
    (if (nil? lst)
        init
        (fold f (f init (car lst)) (cdr lst)))))

; Fold-right: combine elements from right to left
(def fold-right (fn (f init lst)
    (if (nil? lst)
        init
        (f (car lst) (fold-right f init (cdr lst))))))
