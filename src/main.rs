use lisp_vm::{eval, expand, parse, parse_all, standard_env, MacroRegistry, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Run file mode
        let filename = &args[1];
        if let Err(e) = run_file(filename) {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    } else {
        // REPL mode
        run_repl();
    }
}

fn run_file(filename: &str) -> Result<Value, String> {
    let contents = fs::read_to_string(filename)
        .map_err(|e| format!("Could not read file '{}': {}", filename, e))?;

    let exprs = parse_all(&contents)?;
    let env = standard_env();
    let macros = MacroRegistry::new();

    let mut result = Value::Nil;
    for expr in exprs {
        let expanded = expand(&expr, &macros, &env)?;
        result = eval(&expanded, &env)?;
    }

    Ok(result)
}

fn run_repl() {
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
