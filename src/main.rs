use lisp_vm::{clear_arena, expand, parse, parse_all, standard_env, standard_vm, Compiler, MacroRegistry, Value};
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

    // For macro expansion, we still need the tree-walking environment
    let env = standard_env();
    let macros = MacroRegistry::new();

    // Expand macros on all expressions first
    let expanded: Result<Vec<Value>, String> = exprs
        .iter()
        .map(|expr| expand(expr, &macros, &env))
        .collect();
    let expanded = expanded?;

    // Compile all expressions to bytecode
    let chunk = Compiler::compile_all(&expanded)?;

    // Execute via bytecode VM
    // Note: Arena allocations accumulate during execution for maximum performance
    // (free clone/drop). The arena is cleared when the program exits.
    let mut vm = standard_vm();
    let result = vm.run(chunk)?;

    // Promote result if it contains arena values (for returning to caller)
    let promoted = result.promote();

    // Clear arena after execution (frees all temporary allocations at once)
    clear_arena();

    Ok(promoted)
}

fn run_repl() {
    println!("Lisp VM v0.1.0 (bytecode)");
    println!("Type :q or :quit to exit.");
    println!();

    // For macro expansion, we still need the tree-walking environment
    let env = standard_env();
    let macros = MacroRegistry::new();

    // Create a single VM instance to maintain globals across expressions
    let mut vm = standard_vm();

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
                        // First expand macros
                        match expand(&expr, &macros, &env) {
                            Ok(expanded) => {
                                // Compile to bytecode
                                match Compiler::compile(&expanded) {
                                    Ok(chunk) => {
                                        // Execute via VM
                                        match vm.run(chunk) {
                                            Ok(result) => {
                                                // Promote result before displaying (escapes arena)
                                                let result = result.promote();
                                                // Don't display nil results (common REPL behavior)
                                                if !result.is_nil() {
                                                    println!("{}", result);
                                                }
                                                // Clear arena after each REPL command
                                                clear_arena();
                                            }
                                            Err(e) => {
                                                clear_arena(); // Clear even on error
                                                eprintln!("Error: {}", e);
                                            }
                                        }
                                    }
                                    Err(e) => eprintln!("Compile error: {}", e),
                                }
                            }
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
