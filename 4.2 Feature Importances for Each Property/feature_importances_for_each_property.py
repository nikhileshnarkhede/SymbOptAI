from Equation_Terms import analyze
from Term_wise_calculation import build_formula
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow import keras
import joblib
from sklearn.metrics import r2_score

from re import X
import matplotlib.pyplot as plt
from warnings import filterwarnings

filterwarnings('ignore')

Eq_1 = open("Best_EQ/Eq_Y_1.txt").read().strip()
Eq_2 = open("Best_EQ/Eq_Y_2.txt").read().strip()
Eq_3 = open("Best_EQ/Eq_Y_3.txt").read().strip()
Eq_4 = open("Best_EQ/Eq_Y_4.txt").read().strip()
Eq_5 = open("Best_EQ/Eq_Y_5.txt").read().strip()
Eq_6 = open("Best_EQ/Eq_Y_6.txt").read().strip()

EQ_1_Terms = analyze(Eq_1,verbose = False,save_path="Output/output1.txt")
EQ_2_Terms = analyze(Eq_2,verbose = False,save_path="Output/output2.txt")
EQ_3_Terms = analyze(Eq_3,verbose = False,save_path="Output/output3.txt")
EQ_4_Terms = analyze(Eq_4,verbose = False,save_path="Output/output4.txt")
EQ_5_Terms = analyze(Eq_5,verbose = False,save_path="Output/output5.txt")
EQ_6_Terms = analyze(Eq_6,verbose = False,save_path="Output/output6.txt")

EQ_1_Terms = EQ_1_Terms['terms_out']
EQ_2_Terms = EQ_2_Terms['terms_out']
EQ_3_Terms = EQ_3_Terms['terms_out']
EQ_4_Terms = EQ_4_Terms['terms_out']
EQ_5_Terms = EQ_5_Terms['terms_out']
EQ_6_Terms = EQ_6_Terms['terms_out']


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, EQ_Terms_List):
        self.EQ_Terms_List = EQ_Terms_List

    def fit(self, X, y=None):
        return self  # No fitting required

    def predict(self, X):
        return self.transform(X)

    def transform(self, X):
        # Make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['X0','X1','X2','X3'])
        else:
            X = X.copy()
            X.columns = ['X0','X1','X2','X3']

        engineered_features_all = pd.DataFrame(index=X.index)

        for eq_index, EQ_Terms in enumerate(self.EQ_Terms_List, start=1):
            formula_func = build_formula(EQ_Terms)
            result_df = formula_func(X)

            # Drop Final_Result if exists
            Featrue_Eng_df = result_df.drop(columns=['Final_Result'], errors='ignore')

            # Rename columns systematically
            Featrue_Eng_df.columns = [f"EQ_{eq_index}_Term_{i+1}" for i in range(len(Featrue_Eng_df.columns))]

            engineered_features_all = pd.concat([engineered_features_all, Featrue_Eng_df], axis=1)

        return engineered_features_all.values  # Return as ndarray



class KerasModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self  # Already trained

    def predict(self, X):
        return self.model.predict(X)



# Redefine this class exactly as it was during saving
class FullPredictionPipeline:
    def __init__(self, fe_pipeline, model):
        self.fe_pipeline = fe_pipeline
        self.model = model

    def predict(self, X):
        X_fe = self.fe_pipeline.transform(X)
        return self.model.predict(X_fe)

# Now safely load
pipeline = joblib.load('final_FE_DNN_pipeline.joblib')

print("Pipeline loaded successfully!")


df_main = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name='Mean_with_std')
df_main.columns = [
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength', 'Tensile_strength_STD',
    'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD',
    'Flexure_modulus_STD', 'Flexure_strength_STD'
]
X = df_main.drop(columns=['Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength', 'Tensile_strength_STD',
    'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD',
    'Flexure_modulus_STD', 'Flexure_strength_STD'])
y = df_main[['Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness','Flexure_modulus', 'Flexure_strength']]

df4 = df_main.copy()
df4["Tensile_strength"] = df4["Tensile_strength"] * 1
df4["Elongation_Break"] = df4["Elongation_Break"] * 100
df4["Toughness"] = df4["Toughness"] * 100
df4["Fracture_toughness"] = df4["Fracture_toughness"] * 10
df4["Flexure_modulus"] = df4["Flexure_modulus"] / 100
df4["Flexure_strength"] = df4["Flexure_strength"] * 1
y_test = df4[['Tensile_strength', 'Elongation_Break', 'Toughness',
              'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']]

y_pred = pipeline.predict(X)

for i, col in enumerate(y_test.columns):
    r2 = r2_score(y_test[col], y_pred[:, i])
    print(f"R2 score for {col}: {r2:.4f}")

# ============================
# 1. Prepare Engineered Features
# ============================
EQ_Terms_List = [
    EQ_1_Terms,
    EQ_2_Terms,
    EQ_3_Terms,
    EQ_4_Terms,
    EQ_5_Terms,
    EQ_6_Terms
]

# Transform original features using your feature engineering pipeline
X_engineered_array = pipeline.fe_pipeline.transform(X)

# Build feature names like EQ_1_Term_1, EQ_2_Term_1, etc.
engineered_feature_names = []
for eq_idx, terms in enumerate(EQ_Terms_List, start=1):
    for term_idx in range(len(terms)):
        engineered_feature_names.append(f"EQ_{eq_idx}_Term_{term_idx + 1}")

# Create DataFrame for easy manipulation
X_engineered = pd.DataFrame(X_engineered_array, columns=engineered_feature_names)

print("Engineered features extracted and stored in X_engineered DataFrame.")
print(X_engineered.head())

# ============================
# 2. Robust Permutation Importances
# ============================
rng = np.random.default_rng(seed=42)  # reproducible RNG
n_repeats = 10  # average over multiple permutations

permutation_importances = {}
target_cols = y_test.columns
feature_cols = X_engineered.columns

for i, target_col in enumerate(target_cols):
    print(f"Calculating permutation importance for target: {target_col}")

    y_true_target = y_test[target_col].values
    y_pred_full_pipeline = pipeline.predict(X)
    y_pred_target = y_pred_full_pipeline[:, i]

    baseline_r2 = r2_score(y_true_target, y_pred_target)

    importances_for_target = {feature: 0.0 for feature in feature_cols}

    for repeat in range(n_repeats):
        for feature in feature_cols:
            X_permuted = X_engineered.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature])
            y_pred_permuted_full = pipeline.model.predict(X_permuted.values)
            y_pred_permuted_target = y_pred_permuted_full[:, i]
            permuted_r2 = r2_score(y_true_target, y_pred_permuted_target)
            importance = baseline_r2 - permuted_r2
            importances_for_target[feature] += importance / n_repeats

    permutation_importances[target_col] = importances_for_target

# Convert to DataFrame
permutation_importances_df = pd.DataFrame(permutation_importances)
print("\nPermutation Importances (first 5 rows):")
print(permutation_importances_df.head())

# ============================
# 3. Map Features to Formulas
# ============================
# Build a feature-term mapping DataFrame
feature_term_mapping = []
for eq_idx, terms in enumerate(EQ_Terms_List, start=1):
    for term_idx, term_formula in enumerate(terms):
        feature_term_mapping.append({
            'feature': f"EQ_{eq_idx}_Term_{term_idx + 1}",
            'formula': term_formula
        })
feature_term_mapping = pd.DataFrame(feature_term_mapping)

# ============================
# 4. Extract Top-5 Features per Target
# ============================
top5_dict = {}

for target in permutation_importances_df.columns:
    top5_features = permutation_importances_df[target].sort_values(ascending=False).head(5)

    formulas = []
    for feat in top5_features.index:
        formula_row = feature_term_mapping.loc[feature_term_mapping['feature']==feat, 'formula']
        formula = formula_row.values[0] if not formula_row.empty else ''
        formulas.append(formula)

    top5_dict[target] = pd.DataFrame({
        'Feature': top5_features.index,
        'Formula': formulas,
        'Importance': top5_features.values
    })

# ============================
# 5. Save Top-5 Summary
# ============================
with pd.ExcelWriter('Output/Top5_Features_per_Target.xlsx') as writer:
    for target, df_top5 in top5_dict.items():
        df_top5.to_excel(writer, sheet_name=target, index=False)

print("Saved Top5_Features_per_Target.xlsx")

# ============================
# 6. Optional: Bar Plots for Visualization
# ============================
for target, df_top5 in top5_dict.items():
    plt.figure(figsize=(8,4))
    plt.barh(df_top5['Feature'][::-1], df_top5['Importance'][::-1], color='skyblue')
    plt.xlabel('Permutation Importance (drop in R2)')
    plt.title(f'Top-5 Features for {target}')
    plt.tight_layout()
    plt.savefig(f'Output/top5_{target}.png', dpi=200)
    plt.close()
    print(f"Saved plot: top5_{target}.png")

num_top_features = 5 # Display top 5 features for each target

print("\n--- Top Influential Engineered Features ---")
for target_col in permutation_importances_df.columns:
    print(f"\nTarget: {target_col}")
    # Sort features by importance for the current target in descending order
    sorted_features = permutation_importances_df[target_col].sort_values(ascending=False)

    # Display the top N features
    for feature, importance in sorted_features.head(num_top_features).items():
        print(f"  {feature}: {importance:.4f}")



