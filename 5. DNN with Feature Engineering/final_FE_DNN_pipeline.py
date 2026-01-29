# -*- coding: utf-8 -*-
"""FE_DNN_model.ipynb"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from Equation_Terms import analyze
from Term_wise_calculation import build_formula
import joblib

# ==============================
# Load data and prepare datasets
# ==============================
df_main = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name='Mean_with_std')
df_main.columns = [
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength', 'Tensile_strength_STD',
    'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD',
    'Flexure_modulus_STD', 'Flexure_strength_STD'
]

df = pd.read_excel('Bootstrapped_Data.xlsx')


# Load equations
Eq_files = [f"Eq_Y_{i}.txt" for i in range(1, 7)]
Eq_terms = [analyze(open(eq).read().strip(), verbose=False)['terms_out'] for eq in Eq_files]

# ===========================
# Scale and preprocess dataset
# ===========================
df["Tensile_strength"] = df["Tensile_strength"] * 1
df["Elongation_Break"] = df["Elongation_Break"] * 100
df["Toughness"] = df["Toughness"] * 100
df["Fracture_toughness"] = df["Fracture_toughness"] * 10
df["Flexure_modulus"] = df["Flexure_modulus"] / 100
df["Flexure_strength"] = df["Flexure_strength"] * 1

# Feature-engineered input (precomputed)
df1 = df.drop(columns=[
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength'
])

scaler = MinMaxScaler(feature_range=(-100, 100))
X_train = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
y_train = df[['Tensile_strength', 'Elongation_Break', 'Toughness', 
              'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']]

# Test data (for evaluation)
df4 = df_main.copy()
df4["Tensile_strength"] = df4["Tensile_strength"] * 1
df4["Elongation_Break"] = df4["Elongation_Break"] * 100
df4["Toughness"] = df4["Toughness"] * 100
df4["Fracture_toughness"] = df4["Fracture_toughness"] * 10
df4["Flexure_modulus"] = df4["Flexure_modulus"] / 100
df4["Flexure_strength"] = df4["Flexure_strength"] * 1

X_test = df4[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']]
y_test = df4[['Tensile_strength', 'Elongation_Break', 'Toughness', 
              'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']]

# ==================================
# Feature Engineering Transformer
# ==================================
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, EQ_Terms_List):
        self.EQ_Terms_List = EQ_Terms_List

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['X0','X1','X2','X3'])
        else:
            X = X.copy()
            X.columns = ['X0','X1','X2','X3']

        engineered_features_all = pd.DataFrame(index=X.index)

        for eq_index, EQ_Terms in enumerate(self.EQ_Terms_List, start=1):
            formula_func = build_formula(EQ_Terms)
            result_df = formula_func(X)
            Featrue_Eng_df = result_df.drop(columns=['Final_Result'], errors='ignore')
            Featrue_Eng_df.columns = [f"EQ_{eq_index}_Term_{i+1}" for i in range(len(Featrue_Eng_df.columns))]
            engineered_features_all = pd.concat([engineered_features_all, Featrue_Eng_df], axis=1)

        return engineered_features_all.values

# ==================================
# Build Feature Engineering + Model
# ==================================
feature_eng_transformer = FeatureEngineeringTransformer(EQ_Terms_List=Eq_terms)
fe_pipeline = Pipeline([
    ('feature_eng', feature_eng_transformer),
    ('scaler', MinMaxScaler(feature_range=(-100, 100)))
])

# Prepare transformed test data for validation
fe_pipeline.fit(X_test)
X_test_fe = fe_pipeline.transform(X_test)

# ===============================
# Deep Neural Network Architecture
# ===============================
x_input = layers.Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation='relu')(x_input)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(14, activation='relu')(x)
y_output = layers.Dense(6)(x)
model = Model(x_input, y_output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test_fe, y_test))

# Save DNN
model.save('model_FE_DNN.keras')

# ===============================
# Evaluate model
# ===============================
y_pred = pd.DataFrame(model.predict(X_test_fe),
                      columns=['Tensile_strength','Elongation_Break','Toughness',
                               'Fracture_toughness','Flexure_modulus','Flexure_strength'])
r2_scores = {col: r2_score(y_test[col], y_pred[col]) for col in y_test.columns}
print("R2 Scores for each target variable:")
for k, v in r2_scores.items():
    print(f"{k}: {v:.3f}")

# ===============================
# Create unified pipeline (FE + DNN)
# ===============================
class FullPredictionPipeline:
    def __init__(self, fe_pipeline, model):
        self.fe_pipeline = fe_pipeline
        self.model = model

    def predict(self, X):
        X_fe = self.fe_pipeline.transform(X)
        return self.model.predict(X_fe)

# Create and save unified pipeline
unified_pipeline = FullPredictionPipeline(fe_pipeline, model)
joblib.dump(unified_pipeline, 'final_FE_DNN_pipeline.joblib')

print("\n Unified pipeline saved as 'final_FE_DNN_pipeline.joblib'")
