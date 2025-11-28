import re #regular expression (regex)
from typing import List, Tuple
Token = Tuple[str, str]

# PHASE 1 — LEXICAL ANALYZER

token_specification = [
    ('WHILE',      r'while\b'), # r make / not treated as a character
    ('PRINTF',     r'printf\b'), #\b word boundary
    # Operators
    ('EQ',         r'=='),
    ('NEQ',        r'!='),
    ('LESS',       r'<'),
    ('GREATER',    r'>'),
    ('INC',        r'\+\+'),

    # Symbols
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('COMMA',      r','),
    ('SEMICOLON',  r';'),

    # Literals
    ('STRING',     r'"[^"]*"'),
    ('NUM',        r'\d+'),
    ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'), #First character must be letter or _ //// Remaining characters can include letters, digits, or _

 # WHITESPACE (NEEDED!!)
    ('NEWLINE',    r'\n'),
    ('SKIP',       r'[ \t]+'),
    
    ('MISMATCH',   r'.'),
]

tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)

def tokenize(code):
    tokens = []
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()

        if kind in ('NEWLINE', 'SKIP'):
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Unexpected character {value}')
        else:
            tokens.append((kind, value))
    return tokens

# PHASE 2 — RIGHT-MOST DERIVATION PARSER

class ParserRMD:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t[0] for t in tokens]
        self.lexemes = [t[1] for t in tokens]
        self.pos = 0

        self.sentential = ['S']
        self.sent_forms = []
        self.record()

    def record(self):
        self.sent_forms.append(' '.join(self.sentential))

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else 'EOF'

    def accept(self, expected):
        if self.peek() == expected:
            self.pos += 1
            return True
        return False

    def expect(self, expected):
        if not self.accept(expected):
            raise SyntaxError(f"Expected {expected} but got {self.peek()}")

    def expand(self, nonterm, rhs):
        for i in range(len(self.sentential)-1, -1, -1):
            if self.sentential[i] == nonterm:
                self.sentential = self.sentential[:i] + rhs + self.sentential[i+1:]
                self.record()
                return
        raise RuntimeError(f"Nonterminal {nonterm} not found")

    # Parsing Logic

    def parse(self):
        ok = self.parse_S()
        if self.pos != len(self.tokens):
            raise SyntaxError("Extra tokens found after parsing.")
        return ok

    def parse_S(self):
        self.expand('S', ['WHILE','LPAREN','CND','RPAREN','LBRACE','STMTS','RBRACE'])

        self.expect('WHILE')
        self.expect('LPAREN')
        self.parse_CND()
        self.expect('RPAREN')
        self.expect('LBRACE')
        self.parse_STMTS()
        self.expect('RBRACE')
        return True

    def parse_CND(self):
        self.expand('CND', ['ID','OP','EXPR'])

        self.expect('ID')

        if self.peek() in ('LESS','GREATER','EQ','NEQ'):
            self.accept(self.peek())
        else:
            raise SyntaxError("Expected comparison operator")

        if self.peek() == 'NUM':
            self.accept('NUM')
        elif self.peek() == 'ID':
            self.accept('ID')
        else:
            raise SyntaxError("Expected NUM or ID in expression")

        return True

    def parse_STMTS(self):
        if self.peek() == 'RBRACE':
            self.expand('STMTS', [])
            return True

        self.expand('STMTS', ['STMT','STMTS'])
        self.parse_STMT()
        return self.parse_STMTS()

    def parse_STMT(self):
        if self.peek() == 'PRINTF':
            return self.parse_PRINT()
        elif self.peek() == 'ID':
            return self.parse_INC()
        else:
            raise SyntaxError("Expected a statement")

    def parse_PRINT(self):
        self.expand('STMT', ['PRINTSTMT'])
        self.expand('PRINTSTMT', ['PRINTF','LPAREN','STRING','COMMA','ID','RPAREN','SEMICOLON'])

        self.expect('PRINTF')
        self.expect('LPAREN')
        self.expect('STRING')
        self.expect('COMMA')
        self.expect('ID')
        self.expect('RPAREN')
        self.expect('SEMICOLON')
        return True

    def parse_INC(self):
        self.expand('STMT', ['ASSIGN'])
        self.expand('ASSIGN', ['ID','INC','SEMICOLON'])

        self.expect('ID')
        self.expect('INC')
        self.expect('SEMICOLON')
        return True

    def print_derivation(self):
        print("\n=== RIGHT-MOST DERIVATION STEPS ===")
        for i, s in enumerate(self.sent_forms):
            print(f"{i:02d}: {s}")
        print("====================================\n")


# LEFT-MOST DERIVATION PARSER
class ParserLMD(ParserRMD):
    def __init__(self, tokens: List[Token]):
        super().__init__(tokens)
        # Reset sentential forms for left-most parsing trace
        self.sentential = ['S']
        self.sent_forms = []
        self.record()

    def expand(self, nonterm, rhs):
        # left-most expansion: find the first occurrence
        for i in range(len(self.sentential)):
            if self.sentential[i] == nonterm:
                self.sentential = self.sentential[:i] + rhs + self.sentential[i+1:]
                self.record()
                return
        raise RuntimeError(f"Nonterminal {nonterm} not found")

    def print_derivation(self):
        print("\n=== LEFT-MOST DERIVATION STEPS ===")
        for i, s in enumerate(self.sent_forms):
            print(f"{i:02d}: {s}")
        print("====================================\n")

# PHASE 3 — SEMANTIC ANALYZER

class SemanticAnalyzer:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.symbols = {}
        self.messages = []

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF','')

    def advance(self):
        t = self.peek()
        self.pos += 1
        return t

    def expect(self, token_type):
        if self.peek()[0] != token_type:
            raise Exception(f"Semantic Error: expected {token_type}, got {self.peek()}")
        return self.advance()

    def analyze(self):
        self._parse_program()
        self.messages.append("Semantic Analysis Completed Successfully.")
        return self.messages

    def _add_symbol(self, name):
        if name not in self.symbols:
            self.symbols[name] = "int"
            self.messages.append(f"Symbol: '{name}' added to table as int")

    def _parse_program(self):
        self.expect('WHILE')
        self.expect('LPAREN')
        self._parse_condition()
        self.expect('RPAREN')
        self.expect('LBRACE')
        self._parse_statements()
        self.expect('RBRACE')

    def _parse_condition(self):
        left = self.expect('ID')[1]
        self._add_symbol(left)

        op = self.advance()[0]
        if op not in ('LESS','GREATER','EQ','NEQ'):
            raise Exception("Invalid operator in condition")

        rhs = self.peek()
        if rhs[0] == 'ID':
            right = self.expect('ID')[1]
            self._add_symbol(right)
        elif rhs[0] == 'NUM':
            right = self.expect('NUM')[1]
        else:
            raise Exception("Right side of condition invalid")

        self.messages.append(f"Condition '{left} {op} {right}' is semantically valid.")

    def _parse_statements(self):
        while self.peek()[0] != 'RBRACE':
            if self.peek()[0] == 'PRINTF':
                self._parse_printf()
            elif self.peek()[0] == 'ID':
                self._parse_increment()
            else:
                raise Exception(f"Unexpected token in block: {self.peek()}")

    def _parse_printf(self):
        self.expect('PRINTF')
        self.expect('LPAREN')
        fmt = self.expect('STRING')[1][1:-1]
        self.expect('COMMA')
        arg = self.expect('ID')[1]
        self._add_symbol(arg)
        self.expect('RPAREN')
        self.expect('SEMICOLON')

        if fmt.count('%d') != 1:
            raise Exception("Printf expects exactly one %d")
        self.messages.append(f"Printf call is semantically valid with arg '{arg}'.")

    def _parse_increment(self):
        name = self.expect('ID')[1]
        self._add_symbol(name)
        self.expect('INC')
        self.expect('SEMICOLON')
        self.messages.append(f"Increment '{name}++' is semantically valid.")

# MAIN DRIVER

if __name__ == "__main__":
    code = """
    while (i < 5) {
        printf("%d\\n", i);
        i++;
    }
    """

    print("=== SOURCE CODE ===")
    print(code)

    # PHASE 1: LEXING
    tokens = tokenize(code)
    print("\n=== TOKENS ===")
    for t in tokens:
        print(f"{t[0]:12} -> {t[1]}")

    # PHASE 2: PARSING (RMD)
    parser = ParserRMD(tokens)
    parser.parse()
    parser.print_derivation()

    # PHASE 2b: PARSING (LMD)
    parser_l = ParserLMD(tokens)
    parser_l.parse()
    parser_l.print_derivation()

    # PHASE 3: SEMANTICS
    sem = SemanticAnalyzer(tokens)
    messages = sem.analyze()
    print("=== SEMANTIC ANALYSIS ===")
    for msg in messages:
        print(msg)
