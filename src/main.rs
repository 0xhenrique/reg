use lisp_vm::{eval, expand, parse, standard_env, MacroRegistry};
use std::io::{self, BufRead, Write};

fn main() {
    println!("Lisp VM v0.1.0");
    println!("Type :q or :quit to exit.");
    println!();

    let env = standard_env();
    let macros = MacroRegistry::new();
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

                if line == ":q" || line == ":quit" {
                    break;
                }

                match parse(line) {
                    Ok(expr) => {
                        // First expand macros, then evaluate
                        match expand(&expr, &macros, &env) {
                            Ok(expanded) => match eval(&expanded, &env) {
                                Ok(result) => {
                                    // Don't display nil results (common REPL behavior)
                                    if !result.is_nil() {
                                        println!("{}", result);
                                    }
                                }
                                Err(e) => eprintln!("Error: {}", e),
                            },
                            Err(e) => eprintln!("Macro error: {}", e),
                        }
                    }
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
