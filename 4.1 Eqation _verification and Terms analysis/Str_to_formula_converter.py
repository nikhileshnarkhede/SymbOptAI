import pandas as pd
import numpy as np

# Primitive operators
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return a / b
def sqrt(a): return np.sqrt(np.absolute(a))   # safe sqrt
def inv(a): return 1 / a
def log(a): return np.log(np.absolute(a) + 1e-12)  # avoid log(0)
def neg(a): return -a
def abs_val(a): return np.abs(a)
def max_op(a, b): return np.maximum(a, b)
def min_op(a, b): return np.minimum(a, b)
def sin(a): return np.sin(a)
def cos(a): return np.cos(a)
def tan(a): return np.tan(a)

def evaluate_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    Evaluate a symbolic regression style formula string over a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Data with columns X0, X1, ...
        formula (str): Expression string like 'add(X0, mul(X1, 2))'

    Returns:
        pd.Series: Result of formula evaluated row-wise.
    """
    # Environment = only columns that exist in df + operators
    env = {col: df[col] for col in df.columns if col in formula}
    env.update({
        "add": add,
        "sub": sub,
        "mul": mul,
        "div": div,
        "sqrt": sqrt,
        "log": log,
        "inv": inv,
        "neg": neg,
        "abs": abs_val,
        "max": max_op,
        "min": min_op,
        "sin": sin,
        "cos": cos,
        "tan": tan
    })

    # Safe eval
    return eval(formula, {"__builtins__": {}}, env)
