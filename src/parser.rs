use crate::value::Value;

/// Token types for the lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    Quote,
    Symbol(String),
    String(String),
    Int(i64),
    Float(f64),
}

/// Tokenize input string into tokens
pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            // Skip whitespace
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }

            // Skip comments (from ; to end of line)
            ';' => {
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }

            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }

            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }

            '\'' => {
                tokens.push(Token::Quote);
                chars.next();
            }

            // String literal
            '"' => {
                chars.next(); // consume opening quote
                let mut s = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => {
                            // Escape sequences
                            match chars.next() {
                                Some('n') => s.push('\n'),
                                Some('t') => s.push('\t'),
                                Some('r') => s.push('\r'),
                                Some('\\') => s.push('\\'),
                                Some('"') => s.push('"'),
                                Some(c) => return Err(format!("Unknown escape sequence: \\{}", c)),
                                None => return Err("Unterminated string".to_string()),
                            }
                        }
                        Some(c) => s.push(c),
                        None => return Err("Unterminated string".to_string()),
                    }
                }
                tokens.push(Token::String(s));
            }

            // Number or symbol starting with - or +
            '-' | '+' => {
                chars.next();
                if let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        // It's a number
                        let mut num_str = String::from(ch);
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_digit() || c == '.' {
                                num_str.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(parse_number(&num_str)?);
                    } else {
                        // It's a symbol like - or +
                        let mut sym = String::from(ch);
                        while let Some(&c) = chars.peek() {
                            if is_symbol_char(c) {
                                sym.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(Token::Symbol(sym));
                    }
                } else {
                    // Just - or + at end of input
                    tokens.push(Token::Symbol(ch.to_string()));
                }
            }

            // Number
            c if c.is_ascii_digit() => {
                let mut num_str = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(parse_number(&num_str)?);
            }

            // Symbol
            c if is_symbol_start(c) => {
                let mut sym = String::new();
                while let Some(&c) = chars.peek() {
                    if is_symbol_char(c) {
                        sym.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Symbol(sym));
            }

            _ => return Err(format!("Unexpected character: {}", ch)),
        }
    }

    Ok(tokens)
}

fn is_symbol_start(c: char) -> bool {
    c.is_alphabetic() || matches!(c, '_' | '*' | '/' | '<' | '>' | '=' | '!' | '?' | '+' | '-')
}

fn is_symbol_char(c: char) -> bool {
    is_symbol_start(c) || c.is_ascii_digit()
}

fn parse_number(s: &str) -> Result<Token, String> {
    if s.contains('.') {
        s.parse::<f64>()
            .map(Token::Float)
            .map_err(|_| format!("Invalid float: {}", s))
    } else {
        s.parse::<i64>()
            .map(Token::Int)
            .map_err(|_| format!("Invalid integer: {}", s))
    }
}

/// Parser for s-expressions
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        self.pos += 1;
        token
    }

    /// Parse a single expression
    pub fn parse_expr(&mut self) -> Result<Value, String> {
        let token = self.advance().ok_or("Unexpected end of input")?;

        match token.clone() {
            Token::LParen => self.parse_list(),
            Token::RParen => Err("Unexpected ')'".to_string()),
            Token::Quote => {
                let quoted = self.parse_expr()?;
                Ok(Value::list(vec![Value::symbol("quote"), quoted]))
            }
            Token::Symbol(s) => Ok(parse_atom(&s)),
            Token::String(s) => Ok(Value::String(s.into())),
            Token::Int(n) => Ok(Value::Int(n)),
            Token::Float(n) => Ok(Value::Float(n)),
        }
    }

    fn parse_list(&mut self) -> Result<Value, String> {
        let mut items = Vec::new();

        loop {
            match self.peek() {
                Some(Token::RParen) => {
                    self.advance();
                    return Ok(Value::list(items));
                }
                Some(_) => {
                    items.push(self.parse_expr()?);
                }
                None => return Err("Unclosed list".to_string()),
            }
        }
    }

    /// Check if there are more expressions to parse
    pub fn is_done(&self) -> bool {
        self.pos >= self.tokens.len()
    }
}

/// Parse an atom (symbol that might be nil, true, false, or a regular symbol)
fn parse_atom(s: &str) -> Value {
    match s {
        "nil" => Value::Nil,
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::symbol(s),
    }
}

/// Parse a string into a single expression
pub fn parse(input: &str) -> Result<Value, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err("Empty input".to_string());
    }
    let mut parser = Parser::new(tokens);
    parser.parse_expr()
}

/// Parse a string into multiple expressions
pub fn parse_all(input: &str) -> Result<Vec<Value>, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Ok(vec![]);
    }
    let mut parser = Parser::new(tokens);
    let mut exprs = Vec::new();
    while !parser.is_done() {
        exprs.push(parser.parse_expr()?);
    }
    Ok(exprs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("(+ 1 2)").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Symbol("+".to_string()),
                Token::Int(1),
                Token::Int(2),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_tokenize_string() {
        let tokens = tokenize("\"hello world\"").unwrap();
        assert_eq!(tokens, vec![Token::String("hello world".to_string())]);
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("3.14").unwrap();
        assert_eq!(tokens, vec![Token::Float(3.14)]);
    }

    #[test]
    fn test_tokenize_negative() {
        let tokens = tokenize("-42").unwrap();
        assert_eq!(tokens, vec![Token::Int(-42)]);
    }

    #[test]
    fn test_tokenize_comment() {
        let tokens = tokenize("1 ; this is a comment\n2").unwrap();
        assert_eq!(tokens, vec![Token::Int(1), Token::Int(2)]);
    }

    #[test]
    fn test_parse_atom() {
        assert_eq!(parse("nil").unwrap(), Value::Nil);
        assert_eq!(parse("true").unwrap(), Value::Bool(true));
        assert_eq!(parse("false").unwrap(), Value::Bool(false));
        assert_eq!(parse("42").unwrap(), Value::Int(42));
        assert_eq!(parse("3.14").unwrap(), Value::Float(3.14));
        assert_eq!(parse("foo").unwrap(), Value::symbol("foo"));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(parse("\"hello\"").unwrap(), Value::string("hello"));
    }

    #[test]
    fn test_parse_list() {
        let result = parse("(+ 1 2)").unwrap();
        let expected = Value::list(vec![Value::symbol("+"), Value::Int(1), Value::Int(2)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_nested_list() {
        let result = parse("(+ (* 2 3) 4)").unwrap();
        let inner = Value::list(vec![Value::symbol("*"), Value::Int(2), Value::Int(3)]);
        let expected = Value::list(vec![Value::symbol("+"), inner, Value::Int(4)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_quote() {
        let result = parse("'foo").unwrap();
        let expected = Value::list(vec![Value::symbol("quote"), Value::symbol("foo")]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_empty_list() {
        let result = parse("()").unwrap();
        assert_eq!(result, Value::list(vec![]));
    }

    #[test]
    fn test_parse_multiple() {
        let results = parse_all("1 2 3").unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Value::Int(1));
        assert_eq!(results[1], Value::Int(2));
        assert_eq!(results[2], Value::Int(3));
    }
}
