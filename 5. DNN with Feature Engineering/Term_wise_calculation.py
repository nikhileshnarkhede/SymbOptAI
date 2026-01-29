import pandas as pd
import numpy as np

# ---------- Operators ----------
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return a / b
def sqrt(a): return np.sqrt(np.abs(a))               # Safe sqrt
def inv(a): return 1 / a
def log(a): return np.log(np.where(a > 0, a, 1e-9))  # Safe log (avoid <=0)
def neg(a): return -a
def abs_val(a): return np.abs(a)
def max_op(a, b): return np.maximum(a, b)
def min_op(a, b): return np.minimum(a, b)
def sin(a): return np.sin(a)
def cos(a): return np.cos(a)
def tan(a): return np.tan(a)   # could clip near pi/2 if needed

# ---------- Formula builder ----------
def build_formula(terms):
    """
    Build a function that takes a DataFrame and evaluates each term + final result.
    terms: list of expressions like ['add(X0, -0.141)', 'mul(X1, X2)']
    """
    def formula_df(df):
        results = pd.DataFrame(index=df.index)

        # Environment: operators + dataframe columns
        env = {
            "add": add, "sub": sub, "mul": mul, "div": div,
            "sqrt": sqrt, "inv": inv, "log": log, "neg": neg,
            "abs": abs_val, "max": max_op, "min": min_op,
            "sin": sin, "cos": cos, "tan": tan,
            **{col: df[col] for col in df.columns}
        }

        for i, t in enumerate(terms, 1):
            # Ensure env includes any new DataFrame column referenced
            env.update({col: df[col] for col in df.columns if col in t})

            # Evaluate each term safely
            results[f"Term_{i}"] = eval(t, {"__builtins__": {}}, env)

        # Final result = sum of all terms
        results["Final_Result"] = results.sum(axis=1)
        return results

    return formula_df
