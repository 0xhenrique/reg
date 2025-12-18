; Greetings module

(module greetings
  (export hello goodbye welcome)

  (def hello (fn (name)
    (println (list "Hello, " name "!"))))

  (def goodbye (fn (name)
    (println (list "Goodbye, " name "!"))))

  (def welcome (fn (name place)
    (println (list "Welcome to " place ", " name "!")))))
