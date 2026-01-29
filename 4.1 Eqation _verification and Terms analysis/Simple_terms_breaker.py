from collections import defaultdict

def split_args(expr):
    """Split into arguments at top-level, handling nested parentheses."""
    depth = 0
    args = []
    start = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(expr[start:i].strip())
            start = i + 1
    args.append(expr[start:].strip())
    return args

def parse_formula(expr, coeff=1):
    """Recursively extract additive/subtractive terms with coefficients."""
    expr = expr.strip()
    terms = defaultdict(float)

    if expr.startswith("add("):
        inner = expr[4:-1]
        left, right = split_args(inner)
        for k,v in parse_formula(left, coeff).items():
            terms[k] += v
        for k,v in parse_formula(right, coeff).items():
            terms[k] += v

    elif expr.startswith("sub("):
        inner = expr[4:-1]
        left, right = split_args(inner)
        for k,v in parse_formula(left, coeff).items():
            terms[k] += v
        for k,v in parse_formula(right, -coeff).items():  # subtract
            terms[k] += v

    else:
        # Base case: just a variable/constant/expression
        terms[expr] += coeff

    return terms
    
def simplify(expr):
    """Simplify formula and return terms as formatted list."""
    terms = parse_formula(expr)
    # remove near-zero terms
    terms = {k: v for k, v in terms.items() if abs(v) > 1e-9}
    for i, (t, c) in enumerate(terms.items(), 1):
        print(f"Term {i}: {c} * {t}")

    formatted_terms = []
    for t, c in terms.items():
        if c == 1:
            formatted_terms.append(f"{t}")
        elif c == -1:
            formatted_terms.append(f"-{t}")
        else:
            formatted_terms.append(f"{c} * {t}")

    return formatted_terms

