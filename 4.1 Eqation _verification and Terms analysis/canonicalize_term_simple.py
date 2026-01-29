from Str_to_formula_converter import evaluate_formula
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def canonicalize_term(term: str, consider_coeff: bool = True) -> str:
    """
    Canonicalize a term string so that algebraically equivalent
    terms are treated the same.

    Supports operators: add, sub, mul, div, sqrt, log, abs, neg,
                        inv, max, min, sin, cos, tan

    Args:
        term (str): raw term like 'mul(X1, X2)', 'add(0.5, X3)', etc.
        consider_coeff (bool): whether to include numeric coefficients

    Returns:
        str: canonicalized term
    """
    term = term.strip()

    # --- Multiplication canonicalization ---
    if term.startswith("mul("):
        inner = term[4:-1]
        factors = [f.strip() for f in inner.split(",")]

        if not consider_coeff:
            # Drop purely numeric factors
            factors = [f for f in factors if not re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", f)]

        factors.sort()
        return "mul(" + ",".join(factors) + ")"

    # --- Addition canonicalization ---
    if term.startswith("add("):
        inner = term[4:-1]
        parts = [f.strip() for f in inner.split(",")]
        parts.sort()
        return "add(" + ",".join(parts) + ")"

    # --- Max/Min canonicalization (commutative) ---
    if term.startswith("max(") or term.startswith("min("):
        op = term[:3]
        inner = term[4:-1]
        parts = [f.strip() for f in inner.split(",")]
        parts.sort()
        return f"{op}(" + ",".join(parts) + ")"

    # --- Other unary ops (keep structure) ---
    for op in ["sub", "div", "sqrt", "log", "abs", "neg", "inv", "sin", "cos", "tan"]:
        if term.startswith(op + "("):
            return term  # leave as-is

    # --- Fallback (unchanged) ---
    return term