use lisp_vm::{eval, parse, standard_env};
use std::io::{self, BufRead, Write};

fn main() {
    println!("Lisp VM v0.1.0");
    println!("Type expressions to evaluate. Ctrl+D to exit.");
    println!();

    let env = standard_env();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match parse(line) {
                    Ok(expr) => match eval(&expr, &env) {
                        Ok(result) => println!("{}", result),
                        Err(e) => eprintln!("Error: {}", e),
                    },
                    Err(e) => eprintln!("Parse error: {}", e),
                }
            }
            Err(e) => {
                eprintln!("IO error: {}", e);
                break;
            }
        }
    }

    println!("\nGoodbye!");
}
