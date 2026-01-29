import pandas as pd
from Terms_collection import extract_terms
from Equation_Terms import analyze
from sklearn.metrics import r2_score
from Simple_terms_breaker import simplify
from Term_wise_calculation import build_formula
from Str_to_formula_converter import evaluate_formula
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs("Actual_vs_Predicted", exist_ok=True)

def r2_score(X, Y):
    """
    Calculate R² (coefficient of determination) like Excel's =r2_score().
    """
    X = np.array(X)
    Y = np.array(Y)

    # Fit linear regression line
    coeffs = np.polyfit(X, Y, 1)   # degree 1 polynomial (linear fit)
    Y_pred = np.polyval(coeffs, X)

    # Calculate R²
    ss_res = np.sum((Y - Y_pred) ** 2)        # residual sum of squares
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)    # total sum of squares
    r2 = 1 - (ss_res / ss_tot)

    return r2







df_excel = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name='Mean_with_std')
df_excel.columns = [
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness','Fracture_toughness','Flexure_modulus','Flexure_strength',
    'Tensile_strength_STD', 'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD','Flexure_modulus_STD','Flexure_strength_STD'
]

new_columns = {col: f'X{i}' for i, col in enumerate(df_excel.columns)}
df_renamed = df_excel.rename(columns=new_columns)



df = df_renamed[['X0','X1',	'X2', 'X3']]
df.columns = ['X0','X1', 'X2', 'X3']


print('\n\n\n Y_1 \n\n\n\n\n')
with open('Eq_Y_1.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_1_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_1_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 1
# Calculate the R2 score
Y_1= result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X4'], result_df['Final_Result']/Scale)

#df_renamed['X4'].to_excel('Acutal_Y_1.xlsx', index=False)
#Y_1.to_excel('predicted_Y_1.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}") 

print('\n\n\n Y_2 \n\n\n\n\n')
with open('Eq_Y_2.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_2_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_2_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 100
# Calculate the R2 score
Y_2 = result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X5'], result_df['Final_Result']/Scale)

#df_renamed['X5'].to_excel('Acutal_Y_2.xlsx', index=False)
#Y_2.to_excel('predicted_Y_2.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}")


print('\n\n\n Y_3 \n\n\n\n\n')
with open('Eq_Y_3.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_3_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_3_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 100
# Calculate the R2 score
Y_3 = result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X6'], result_df['Final_Result']/Scale)

#df_renamed['X6'].to_excel('Acutal_Y_3.xlsx', index=False)
#Y_3.to_excel('predicted_Y_3.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}")


print('\n\n\n Y_4 \n\n\n\n\n')
with open('Eq_Y_4.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_4_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_4_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 10
# Calculate the R2 score
Y_4 = result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X7'], result_df['Final_Result']/Scale)

#df_renamed['X7'].to_excel('Acutal_Y_4.xlsx', index=False)
#Y_4.to_excel('predicted_Y_4.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}")


print('\n\n\n Y_5 \n\n\n\n\n')
with open('Eq_Y_5.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_5_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_5_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 1
# Calculate the R2 score
Y_5 = result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X8'], result_df['Final_Result']/Scale)

#df_renamed['X8'].to_excel('Acutal_Y_5.xlsx', index=False)
#Y_5.to_excel('predicted_Y_5.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}")


print('\n\n\n Y_6 \n\n\n\n\n')
with open('Eq_Y_6.txt', 'r') as f:
    expr = f.read()



analyze(expr,verbose = False, save_path="Actual_vs_Predicted/output_EQ_Y_6_Terms_Details.txt")

#simplify(expr)

terms = extract_terms('Actual_vs_Predicted/output_EQ_Y_6_Terms_Details.txt')
#terms = simplify(expr)

# Build formula function
formula_df = build_formula(terms)

# Apply formula to dataframe
result_df = formula_df(df)
result_df.head(10)

Scale = 1
# Calculate the R2 score
Y_6 = result_df['Final_Result']/Scale
r2 = r2_score(df_renamed['X9'], result_df['Final_Result']/Scale)

#df_renamed['X9'].to_excel('Acutal_Y_6.xlsx', index=False)
#Y_6.to_excel('predicted_Y_6.xlsx', index=False)

# Print the R2 score
print(f"R2 Score: {r2}")


os.makedirs("Actual_vs_Predicted", exist_ok=True)
# Columns for actuals and predictions
actual_cols = ['X4','X5','X6','X7','X8','X9']
predicted_cols = ['Y_1','Y_2','Y_3','Y_4','Y_5','Y_6']

# If Y_1, Y_2, ... Y_6 are separate DataFrames/Series, combine them
df_predicted = pd.concat([Y_1, Y_2, Y_3, Y_4, Y_5, Y_6], axis=1)
df_predicted.columns = predicted_cols  # ensure column names match

# Combine actuals and predicted side by side
df_actual_pred = pd.concat([df_renamed[actual_cols], df_predicted], axis=1)

# Save to Excel
df_actual_pred.to_excel('Actual_vs_Predicted/df_actual_pred.xlsx', index=False)


# Create 2 rows x 3 columns of square scatter plots
fig, axes = plt.subplots(2, 3, figsize=(12,10))  # 2 rows, 3 columns

for i in range(6):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    x = df_actual_pred[actual_cols[i]]
    y = df_actual_pred[predicted_cols[i]]
    
    ax.scatter(x, y, alpha=0.6)
    
    # 45-degree line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    
    # Labels and title
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(predicted_cols[i])
    ax.grid(True)
    
    # Make square
    ax.set_aspect('equal', adjustable='box')
    
    # R² score
    r2 = r2_score(x, y)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('Actual_vs_Predicted/actual_vs_predicted_scatter.png', dpi=300)
plt.show()