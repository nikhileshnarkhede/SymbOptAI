# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import itertools
from doepy import build
import warnings

# -------------------------------
# Suppress warnings
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Taguchi generator function
# -------------------------------
def create_taguchi_matrix(factors: dict) -> pd.DataFrame:
    """
    Generate a Taguchi experimental design matrix.

    Parameters:
    factors : dict
        Keys are factor names, values are lists of levels for each factor.

    Returns:
    pd.DataFrame
        Taguchi design matrix with selected experiments.
    """
    factor_names = list(factors.keys())
    factor_levels = [factors[f] for f in factor_names]

    # Full factorial design
    full_factorial_df = pd.DataFrame(list(itertools.product(*factor_levels)), columns=factor_names)

    # Determine Taguchi L-array size
    num_factors = len(factors)
    max_factor_levels = max(len(levels) for levels in factors.values())

    if max_factor_levels == 2:
        l_array_size = 4 if num_factors <= 3 else 8
    elif max_factor_levels == 3:
        l_array_size = 9
    elif max_factor_levels == 4:
        l_array_size = 16
    else:
        l_array_size = 27

    # Select subset for Taguchi design if needed
    if len(full_factorial_df) <= l_array_size:
        taguchi_df = full_factorial_df.copy()
    else:
        np.random.seed(42)
        selected_indices = np.linspace(0, len(full_factorial_df) - 1, l_array_size, dtype=int)
        taguchi_df = full_factorial_df.iloc[selected_indices].reset_index(drop=True)

    print(f"\n              Generated Taguchi Design (L{l_array_size}) with {len(taguchi_df)} experiments")
    return taguchi_df



# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    # ---- Define design space ----
    design_space = {
        'Layer thickness (mm)': [0.15, 0.20, 0.25],
        'Printing speed (mm/s)': [40, 50, 60],
        'Bed temperature (°C)': [95, 100, 105],
        'Nozzle temperature (°C)': [230, 245, 260]
    }
    design_space1 = design_space


    # Custom Taguchi design
    df_taguchi_l9_mapped = create_taguchi_matrix(factors=design_space1)

    # -------------------------------
    # Combine all designs in dictionary
    # -------------------------------
    dfs = {        
        "taguchi_l9_mapped": df_taguchi_l9_mapped
    }

    # ---- Save all DOE designs to one Excel file ----
    output_file = "designs_output.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for name, df in dfs.items():
            # Excel sheet name max length = 31 chars
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    
