use lisp_vm::{clear_arena, expand, parse, parse_all, set_arena_enabled, standard_env, standard_vm, Compiler, MacroRegistry, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

struct Config {
    filename: Option<String>,
    arena_enabled: bool,
    jit_enabled: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut config = Config {
        filename: None,
        arena_enabled: false, // Arena is opt-in via --arena flag
        jit_enabled: false,   // JIT is opt-in via --jit flag
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--arena" => {
                config.arena_enabled = true;
            }
            "--jit" => {
                config.jit_enabled = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                print_usage();
                process::exit(1);
            }
            _ => {
                config.filename = Some(args[i].clone());
            }
        }
        i += 1;
    }

    config
}

fn print_usage() {
    eprintln!("Usage: lisp-vm [OPTIONS] [FILE]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --arena    Enable arena allocation for cons cells (experimental)");
    eprintln!("  --jit      Enable JIT compilation for hot functions (experimental)");
    eprintln!("  --help     Show this help message");
    eprintln!();
    eprintln!("If FILE is provided, execute it. Otherwise, start the REPL.");
}

fn main() {
    let config = parse_args();

    set_arena_enabled(config.arena_enabled);

    if let Some(filename) = config.filename {
        if let Err(e) = run_file(&filename, config.arena_enabled, config.jit_enabled) {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    } else {
        run_repl(config.arena_enabled, config.jit_enabled);
    }
}

fn run_file(filename: &str, arena_enabled: bool, jit_enabled: bool) -> Result<Value, String> {
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
    let mut vm = standard_vm();
    if jit_enabled {
        vm.enable_jit();
    }
    let result = vm.run(chunk)?;

    // Arena cleanup
    let final_result = if arena_enabled {
        // Promote result if it contains arena values (for returning to caller)
        let promoted = result.promote();
        clear_arena();
        promoted
    } else {
        result
    };

    Ok(final_result)
}

fn run_repl(arena_enabled: bool, jit_enabled: bool) {
    println!("Lisp VM v0.1.0 (bytecode)");
    if arena_enabled {
        println!("Arena allocation: enabled");
    }
    if jit_enabled {
        println!("JIT compilation: enabled");
    }
    println!("Type :q or :quit to exit.");
    println!();

    // For macro expansion, we still need the tree-walking
    let env = standard_env();
    let macros = MacroRegistry::new();

    // Single VM instance to maintain globals across expressions
    let mut vm = standard_vm();
    if jit_enabled {
        vm.enable_jit();
    }

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

                // Sorry for this
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
                                                // Handle arena
                                                let result = if arena_enabled {
                                                    result.promote()
                                                } else {
                                                    result
                                                };
                                                // Don't display nil results for REPL
                                                if !result.is_nil() {
                                                    println!("{}", result);
                                                }
                                                // Clear arena after each REPL command if enabled
                                                if arena_enabled {
                                                    clear_arena();
                                                }
                                            }
                                            Err(e) => {
                                                if arena_enabled {
                                                    clear_arena(); // Clear even on error
                                                }
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
