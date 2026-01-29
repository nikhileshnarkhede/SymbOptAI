import pandas as pd
from Terms_collection import extract_terms
from Equation_Terms import analyze
from sklearn.metrics import r2_score
from Simple_terms_breaker import simplify
from Term_wise_calculation import build_formula
from Str_to_formula_converter import evaluate_formula
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def canonicalize_term(term: str, consider_coeff: bool = True) -> str:
    """
    Robust, recursive canonicalizer. Handles both infix and function-style terms.

    - Normalizes commutative ops: add, mul, max, min (sorts args).
    - Flattens multiplication/division, collects numeric coefficients.
    - When consider_coeff == False: numeric coefficients (including -1) are ignored
      and top-level negation is removed, so -1*(X2*X0) == (X0*X2).
    - Produces function-style canonical strings: add(...), mul(...), neg(...), tan(...), sqrt(...), etc.

    Note: relies on `re` being already imported in the file.
    """
    term = term.strip()
    if not term:
        return term

    # ---------- tokenizer ----------
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
    import re as _re

    def tokenize(s):
        for mo in _re.finditer(tok_regex, s):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "SKIP":
                continue
            if kind == "MISMATCH":
                # treat stray character as itself (fallback)
                raise SyntaxError(f"Unexpected character {value!r} in term: {s!r}")
            yield (kind, value)

    try:
        tokens = list(tokenize(term))
    except Exception:
        # fallback: return stripped term if tokenizer fails
        return term

    if not tokens:
        return term

    # ---------- parser (recursive descent) ----------
    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0
            self.N = len(tokens)

        def peek(self):
            return self.tokens[self.pos] if self.pos < self.N else (None, None)

        def next(self):
            t = self.peek()
            self.pos += 1
            return t

        def parse(self):
            node = self.parse_expression()
            return node

        # expression := term (('+'|'-') term)*
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

        # term := factor (('*'|'/') factor)*
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

        # factor := NUMBER | NAME ( '(' args ')' )? | '(' expression ')' | unary +/- factor
        def parse_factor(self):
            kind, val = self.peek()
            # unary + / -
            if kind == "OP" and val in ("+", "-"):
                self.next()
                f = self.parse_factor()
                if val == "-":
                    return {"type": "func", "name": "neg", "args": [f]}
                return f
            if kind == "NUMBER":
                self.next()
                return {"type": "num", "value": float(val)}
            if kind == "NAME":
                name = val
                self.next()
                k2, v2 = self.peek()
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
                        else:
                            raise SyntaxError("Expected ',' or ')' in function args")
                    return {"type": "func", "name": name, "args": args}
                else:
                    return {"type": "var", "name": name}
            if kind == "LPAREN":
                self.next()
                node = self.parse_expression()
                k, v = self.peek()
                if k != "RPAREN":
                    raise SyntaxError("Expected ')'")
                self.next()
                return node
            raise SyntaxError(f"Unexpected token {kind} {val}")

    try:
        ast_root = Parser(tokens).parse()
    except Exception:
        # fallback: return input term
        return term

    # ---------- helpers ----------
    def fmt_num(x):
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        return format(x, ".6g")

    # flatten multiplication and collect coefficient
    def get_mul_factors(node):
        """
        Return (coeff, factors_list).
        - Numbers multiply into coeff
        - divisions with non-numeric right become inv(right) factor
        - function-style 'mul' and infix '*' are both handled
        """
        if node["type"] == "num":
            return (node["value"], [])
        if node["type"] == "var" or node["type"] == "func":
            # func may be e.g. sin(...); treat as factor (unless it's mul/div)
            if node["type"] == "func" and node["name"] == "mul":
                # combine args
                tot_coeff = 1.0
                tot_factors = []
                for a in node["args"]:
                    ca, fa = get_mul_factors(a)
                    tot_coeff *= ca
                    tot_factors.extend(fa)
                return (tot_coeff, tot_factors)
            if node["type"] == "func" and node["name"] == "div":
                # div(a,b) -> a * inv(b)
                a, b = node["args"][0], node["args"][1] if len(node["args"]) > 1 else {"type":"num","value":1.0}
                ca, fa = get_mul_factors(a)
                if b["type"] == "num":
                    return (ca / (b["value"] if b["value"] != 0 else 1.0), fa)
                inv_node = {"type":"func", "name": "inv", "args":[b]}
                return (ca, fa + [inv_node])
            return (1.0, [node])

        if node["type"] == "op":
            op = node["op"]
            a, b = node["args"][0], node["args"][1]
            if op == "*":
                ca, fa = get_mul_factors(a)
                cb, fb = get_mul_factors(b)
                return (ca * cb, fa + fb)
            if op == "/":
                # if right is numeric reduce coeff; else add inv(right)
                ca, fa = get_mul_factors(a)
                if b["type"] == "num":
                    return (ca / (b["value"] if b["value"] != 0 else 1.0), fa)
                inv_node = {"type":"func", "name":"inv", "args":[b]}
                return (ca, fa + [inv_node])
            # other ops treated as single factor
            return (1.0, [node])
        # fallback
        return (1.0, [node])

    COMMUTATIVE_FUNCS = {"add", "mul", "max", "min"}

    def canonicalize_atom(node):
        # node -> canonical string
        if node["type"] == "num":
            return fmt_num(node["value"])
        if node["type"] == "var":
            return node["name"]
        if node["type"] == "func":
            name = node["name"]
            arg_strs = [canonicalize_atom(a) for a in node["args"]]
            if name in COMMUTATIVE_FUNCS:
                arg_strs = sorted(arg_strs)
            return f"{name}({','.join(arg_strs)})"
        if node["type"] == "op":
            op = node["op"]
            L = node["args"][0]; R = node["args"][1]
            if op == "+":
                l = canonicalize_atom(L); r = canonicalize_atom(R)
                parts = sorted([l, r])
                return f"add({parts[0]},{parts[1]})"
            if op == "-":
                l = canonicalize_atom(L); r = canonicalize_atom(R)
                return f"sub({l},{r})"
            if op == "*":
                c, f = get_mul_factors(node)
                fstrs = [canonicalize_atom(x) for x in f]
                fstrs = sorted(fstrs)
                if abs(c - 1.0) < 1e-12 and len(fstrs) == 1:
                    return fstrs[0]
                if len(fstrs) == 0:
                    return fmt_num(c)
                if abs(c - 1.0) < 1e-12:
                    return f"mul({','.join(fstrs)})"
                return f"mul({fmt_num(c)},{','.join(fstrs)})"
            if op == "/":
                l = canonicalize_atom(L); r = canonicalize_atom(R)
                return f"div({l},{r})"
        return str(node)

    # ---------- top-level canonicalization ----------
    coeff, factor_nodes = get_mul_factors(ast_root)

    # If consider_coeff is False -> drop numeric coefficients and top-level negation
    if not consider_coeff:
        coeff = 1.0
    neg_flag = False
    if coeff < 0:
        neg_flag = True
        coeff = abs(coeff)
    # If not considering coeffs, also ignore neg_flag
    if not consider_coeff:
        neg_flag = False

    # Remove numeric factors (shouldn't be present because get_mul_factors folds them to coeff),
    # but just in case, drop nodes which are pure numbers when not considering coeffs.
    if not consider_coeff:
        new_factors = []
        for n in factor_nodes:
            # unwrap neg(...) if present and user requested ignore coefficients/sign
            def unwrap_neg(x):
                if x["type"] == "func" and x["name"] == "neg" and x["args"]:
                    return unwrap_neg(x["args"][0])
                return x
            n_un = unwrap_neg(n)
            if n_un["type"] == "num":
                continue
            new_factors.append(n_un)
        factor_nodes = new_factors

    # canonicalize factor strings and sort (commutative)
    factor_strs = [canonicalize_atom(n) for n in factor_nodes]
    factor_strs = sorted(factor_strs)

    # Build canonical body
    # purely numeric
    if len(factor_strs) == 0:
        body = fmt_num(coeff)
        if neg_flag:
            return f"neg({body})"
        return body

    # single factor
    if len(factor_strs) == 1:
        single = factor_strs[0]
        if abs(coeff - 1.0) < 1e-12:
            body = single
        else:
            body = f"mul({fmt_num(coeff)},{single})"
        if neg_flag:
            return f"neg({body})"
        return body

    # multiple factors
    if abs(coeff - 1.0) < 1e-12:
        body = f"mul({','.join(factor_strs)})"
    else:
        body = f"mul({fmt_num(coeff)},{','.join(factor_strs)})"

    if neg_flag:
        return f"neg({body})"
    return body

