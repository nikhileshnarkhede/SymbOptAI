# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Equation_Terms import analyze
from Term_wise_calculation import build_formula
import matplotlib.pyplot as plt

Eq_1 = open("Eq_Y_1.txt").read().strip()
Eq_2 = open("Eq_Y_2.txt").read().strip()
Eq_3 = open("Eq_Y_3.txt").read().strip()
Eq_4 = open("Eq_Y_4.txt").read().strip()
Eq_5 = open("Eq_Y_5.txt").read().strip()
Eq_6 = open("Eq_Y_6.txt").read().strip()

EQ_1_Terms = analyze(Eq_1,verbose = False,save_path="output1.txt")
EQ_2_Terms = analyze(Eq_2,verbose = False,save_path="output2.txt")
EQ_3_Terms = analyze(Eq_3,verbose = False,save_path="output3.txt")
EQ_4_Terms = analyze(Eq_4,verbose = False,save_path="output4.txt")
EQ_5_Terms = analyze(Eq_5,verbose = False,save_path="output5.txt")
EQ_6_Terms = analyze(Eq_6,verbose = False,save_path="output6.txt")

EQ_1_Terms = EQ_1_Terms['terms_out']
EQ_2_Terms = EQ_2_Terms['terms_out']
EQ_3_Terms = EQ_3_Terms['terms_out']
EQ_4_Terms = EQ_4_Terms['terms_out']
EQ_5_Terms = EQ_5_Terms['terms_out']
EQ_6_Terms = EQ_6_Terms['terms_out']

bootstrap1 = True
bootstrap = 100

# ==============================
# Load Data (original df only)
# ==============================
df = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name='Mean_with_std')
df.columns = [
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength',
    'Tensile_strength_STD', 'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD',
    'Flexure_modulus_STD', 'Flexure_strength_STD'
]

# Empty base DataFrames
df1 = pd.DataFrame(columns=[
    'Layer_thickness', 'Printing_speed', 'Bed_temperature',
    'Nozzle_temperature', 'Tensile_strength', 'Elongation_Break',
    'Toughness', 'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength'
])
df2 = pd.DataFrame(columns=df1.columns)

# Store all EQ term sets in a list for iteration
EQ_Terms_List = [EQ_1_Terms, EQ_2_Terms, EQ_3_Terms, EQ_4_Terms, EQ_5_Terms, EQ_6_Terms]

STD_Scale = 0.75

# ====================================
# BOOTSTRAP LOOP with MULTI-FORMULA FEATURE ENGINEERING
# ====================================
for j in range(len(df)):
    # -----------------------------
    # Bootstrap Sampling
    # -----------------------------
    df1['Nozzle_temperature'] = [df['Nozzle_temperature'][j]] * bootstrap
    df1['Nozzle_temperature'] = [df['Nozzle_temperature'][j]]*bootstrap
    df1['Bed_temperature'] = [df['Bed_temperature'][j]]*bootstrap
    df1['Printing_speed'] = [df['Printing_speed'][j]]*bootstrap
    df1['Layer_thickness'] = [df['Layer_thickness'][j]]*bootstrap
    df1['Tensile_strength'] = np.random.normal(loc=df['Tensile_strength'][j], scale=STD_Scale*df['Tensile_strength_STD'][j], size = bootstrap)
    df1['Elongation_Break'] = np.random.normal(loc=df['Elongation_Break'][j], scale=STD_Scale*df['Elongation_Break_STD'][j], size = bootstrap)
    df1['Toughness'] = np.random.normal(loc=df['Toughness'][j], scale=0.6*df['Toughness_STD'][j], size = bootstrap)
    df1['Fracture_toughness'] = np.random.normal(loc=df['Fracture_toughness'][j], scale=STD_Scale*df['Fracture_toughness_STD'][j], size = bootstrap)
    df1['Flexure_modulus'] = np.random.normal(loc=df['Flexure_modulus'][j], scale=STD_Scale*df['Flexure_modulus_STD'][j], size = bootstrap)
    df1['Flexure_strength'] = np.random.normal(loc=df['Flexure_strength'][j], scale=STD_Scale*df['Flexure_strength_STD'][j], size = bootstrap)
  



    # -----------------------------
    # Prepare input (df3) for formula calculation
    # -----------------------------
    df3 = df1[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']].copy()
    df3.columns = ['X0', 'X1', 'X2', 'X3']  # required input variable names

    # -----------------------------
    # Apply All Formula Functions
    # -----------------------------
    engineered_features_all = pd.DataFrame(index=df3.index)

    for eq_index, EQ_Terms in enumerate(EQ_Terms_List, start=1):
        formula_func = build_formula(EQ_Terms)
        result_df = formula_func(df3)

        # Drop Final_Result and rename engineered features systematically
        Featrue_Eng_df = result_df.drop(columns=['Final_Result'], errors='ignore')
        Featrue_Eng_df.columns = [f"EQ_{eq_index}_Term_{i+1}" for i in range(len(Featrue_Eng_df.columns))]

        # Merge all engineered feature groups
        engineered_features_all = pd.concat([engineered_features_all, Featrue_Eng_df], axis=1)

    # -----------------------------
    # Combine Bootstrapped + Engineered Features
    # -----------------------------
    df1_full = pd.concat([df1.reset_index(drop=True), engineered_features_all.reset_index(drop=True)], axis=1)
    df2 = pd.concat([df2, df1_full], ignore_index=True)

# Reset index after all bootstraps
df2.reset_index(drop=True, inplace=True)

# ====================================
# Select dataset based on bootstrap flag
# ====================================
if not bootstrap1:
    X = df[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']]
    y = df['Tensile_strength']
    X_train, y_train = X, y
    X_test = X
    y_test = y
else:
    X = df2[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']]
    y = df2['Tensile_strength']
    X_train, y_train = X, y
    X_test = df[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']]
    y_test = df['Tensile_strength']

df2.to_excel('Bootstrapped_Data.xlsx', index=False)
df3 = df2.drop(columns=['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength'])
    
df3.to_excel('Featrue_Eng.xlsx', index=False)

