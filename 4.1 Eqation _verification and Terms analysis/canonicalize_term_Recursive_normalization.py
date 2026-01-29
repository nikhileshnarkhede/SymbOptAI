import re

def canonicalize_term(term: str, consider_coeff: bool = True) -> str:
    """
    Canonicalize a symbolic term into infix form, 
    merging algebraically equivalent terms.

    Supports: add, sub, mul, div, sqrt, log, abs, neg, inv, max, min, sin, cos, tan
    """
    term = term.strip()

    # Base case: variable or number
    if "(" not in term or not term.endswith(")"):
        if not consider_coeff and re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", term):
            return ""  # drop pure numbers
        return term

    # Extract operator and arguments
    op, args_str = term.split("(", 1)
    op = op.strip()
    args_str = args_str[:-1]  # remove trailing ')'
    args = [a.strip() for a in args_str.split(",")]

    # Recursively canonicalize arguments
    args = [canonicalize_term(a, consider_coeff) for a in args if a != ""]

    # --- Commutative ops ---
    if op == "mul":
        args = sorted(args)
        return " * ".join(args)
    if op == "add":
        args = sorted(args)
        return " + ".join(args)
    if op == "max":
        args = sorted(args)
        return "max(" + ", ".join(args) + ")"
    if op == "min":
        args = sorted(args)
        return "min(" + ", ".join(args) + ")"

    # --- Non-commutative ops ---
import re

import re

def canonicalize_term(term: str, consider_coeff: bool = True) -> str:
    """
    Canonicalize symbolic terms and output in infix style:
      - (X2 * X0) instead of mul(X2,X0)
      - (X2 + X3) instead of add(X2,X3)
      - Functions like tan, sqrt kept in tan(X2), sqrt((X2+X3)) form
    If consider_coeff=False -> drop numeric coefficients and ignore sign (-1 * ... == ...).
    """

    # ---------------- Tokenizer ----------------
    token_spec = [
        ("NUMBER",   r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'),
        ("NAME",     r'[A-Za-z_]\w*'),
        ("OP",       r'[\+\-\*/]'),
        ("LPAREN",   r'\('),
        ("RPAREN",   r'\)'),
        ("COMMA",    r','),
        ("SKIP",     r'\s+'),
        ("MISMATCH", r'.'),
    ]
    tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_spec)

    def tokenize(s):
        for mo in re.finditer(tok_regex, s):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "SKIP":
                continue
            if kind == "MISMATCH":
                raise SyntaxError(f"Unexpected character {value!r} in term: {s!r}")
            yield (kind, value)

    try:
        tokens = list(tokenize(term.strip()))
    except Exception:
        return term.strip()

    if not tokens:
        return term.strip()

    # ---------------- Parser ----------------
    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0

        def peek(self):
            return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

        def next(self):
            tok = self.peek()
            self.pos += 1
            return tok

        def parse(self):
            return self.parse_expression()

        def parse_expression(self):
            node = self.parse_term()
            while True:
                kind, val = self.peek()
                if kind == "OP" and val in ("+", "-"):
                    self.next()
                    right = self.parse_term()
                    node = {"type": "op", "op": val, "args": [node, right]}
                else:
                    break
            return node

        def parse_term(self):
            node = self.parse_factor()
            while True:
                kind, val = self.peek()
                if kind == "OP" and val in ("*", "/"):
                    self.next()
                    right = self.parse_factor()
                    node = {"type": "op", "op": val, "args": [node, right]}
                else:
                    break
            return node

        def parse_factor(self):
            kind, val = self.peek()
            if kind == "OP" and val in ("+", "-"):
                self.next()
                node = self.parse_factor()
                if val == "-":
                    return {"type": "op", "op": "neg", "args": [node]}
                return node
            if kind == "NUMBER":
                self.next()
                return {"type": "num", "value": float(val)}
            if kind == "NAME":
                self.next()
                name = val
                k2, _ = self.peek()
                if k2 == "LPAREN":
                    self.next()
                    args = []
                    k3, _ = self.peek()
                    if k3 == "RPAREN":
                        self.next()
                        return {"type": "func", "name": name, "args": []}
                    while True:
                        args.append(self.parse_expression())
                        kx, vx = self.peek()
                        if kx == "COMMA":
                            self.next()
                            continue
                        elif kx == "RPAREN":
                            self.next()
                            break
                    return {"type": "func", "name": name, "args": args}
                return {"type": "var", "name": name}
            if kind == "LPAREN":
                self.next()
                node = self.parse_expression()
                self.next()  # consume ')'
                return node
            raise SyntaxError(f"Unexpected {kind} {val}")

    try:
        ast_root = Parser(tokens).parse()
    except Exception:
        return term.strip()

    # ---------------- Pretty Printer ----------------
    def fmt_num(x):
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        return str(x)

    def pretty(node):
        if node["type"] == "num":
            return fmt_num(node["value"])
        if node["type"] == "var":
            return node["name"]
        if node["type"] == "op":
            if node["op"] == "+":
                return f"({pretty(node['args'][0])} + {pretty(node['args'][1])})"
            if node["op"] == "-":
                return f"({pretty(node['args'][0])} - {pretty(node['args'][1])})"
            if node["op"] == "*":
                return f"({pretty(node['args'][0])} * {pretty(node['args'][1])})"
            if node["op"] == "/":
                return f"({pretty(node['args'][0])} / {pretty(node['args'][1])})"
            if node["op"] == "neg":
                return f"neg({pretty(node['args'][0])})"
        if node["type"] == "func":
            args_str = ", ".join(pretty(a) for a in node["args"])
            return f"{node['name']}({args_str})"
        return str(node)

    # ---------------- Helpers ----------------
    def flatten_mul(node):
        if node["type"] == "op" and node["op"] == "*":
            a, b = node["args"]
            ca, fa = flatten_mul(a)
            cb, fb = flatten_mul(b)
            return (ca * cb, fa + fb)
        if node["type"] == "num":
            return (node["value"], [])
        return (1, [node])

    # ---------------- Canonicalize ----------------
    coeff, factors = flatten_mul(ast_root)
    if not consider_coeff:
        coeff = 1
        factors = [f for f in factors if f.get("type") != "num"]

    # drop -1 if ignoring coeff
    if not consider_coeff and coeff < 0:
        coeff = -coeff

    # sort factors for commutativity
    factors_strs = sorted([pretty(f) for f in factors])

    # ---------------- Build Final ----------------
    if coeff != 1 or not factors_strs:
        parts = [fmt_num(coeff)] if coeff != 1 else []
        parts.extend(factors_strs)
        if len(parts) == 1:
            return parts[0]
        return "(" + " * ".join(parts) + ")"
    if len(factors_strs) == 1:
        return factors_strs[0]
    return "(" + " * ".join(factors_strs) + ")"
