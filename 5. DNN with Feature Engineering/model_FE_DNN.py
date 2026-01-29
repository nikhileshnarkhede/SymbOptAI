
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from Equation_Terms import analyze
from Term_wise_calculation import build_formula

df_main = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name = 'Mean_with_std' )
df_main.columns = ['Layer_thickness', 'Printing_speed', 'Bed_temperature',
       'Nozzle_temperature', 'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength', 'Tensile_strength_STD', 'Elongation_Break_STD',
       'Toughness_STD', 'Fracture_toughness_STD', 'Flexure_modulus_STD',
       'Flexure_strength_STD']

df = pd.read_excel('Bootstrapped_Data.xlsx')
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

df.T

df["Tensile_strength"] = df["Tensile_strength"] * 1
df["Elongation_Break"] = df["Elongation_Break"] * 100
df["Toughness"] = df["Toughness"] * 100
df["Fracture_toughness"] = df["Fracture_toughness"] * 10
df["Flexure_modulus"] = df["Flexure_modulus"] / 100
df["Flexure_strength"] = df["Flexure_strength"] * 1

df

df1 = df.drop(columns=['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness',
    'Flexure_modulus', 'Flexure_strength'])

df1

scaler = MinMaxScaler(feature_range=(-100, 100))
scaled_data = scaler.fit_transform(df1)
scaled_data_FE = pd.DataFrame(scaled_data, columns=df1.columns)

scaled_data_FE

X_train = scaled_data_FE
y_train  = df[['Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness','Flexure_modulus', 'Flexure_strength']]

df4 = df_main.copy()
df4["Tensile_strength"] = df4["Tensile_strength"] * 1
df4["Elongation_Break"] = df4["Elongation_Break"] * 100
df4["Toughness"] = df4["Toughness"] * 100
df4["Fracture_toughness"] = df4["Fracture_toughness"] * 10
df4["Flexure_modulus"] = df4["Flexure_modulus"] / 100
df4["Flexure_strength"] = df4["Flexure_strength"] * 1
X_test = df4[['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']]
y_test = df4[['Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']]

X_mean = df_main[['Layer_thickness',	'Printing_speed',	'Bed_temperature',	'Nozzle_temperature']]
y_mean = df_main[['Tensile_strength', 'Elongation_Break', 'Toughness', 'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']]



from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, EQ_Terms_List):
        self.EQ_Terms_List = EQ_Terms_List

    def fit(self, X, y=None):
        return self  # No fitting required

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

from sklearn.pipeline import Pipeline

EQ_Terms_List = [EQ_1_Terms, EQ_2_Terms, EQ_3_Terms, EQ_4_Terms, EQ_5_Terms, EQ_6_Terms]
feature_eng_transformer = FeatureEngineeringTransformer(EQ_Terms_List=EQ_Terms_List)
pipeline = Pipeline([
    ('feature_eng', feature_eng_transformer),
    ('scaler', MinMaxScaler(feature_range=(-100, 100)))

])

pipeline.fit(X_test)

X_test_fe = pipeline.transform(X_test)



x_input = layers.Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation='relu')(x_input)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(14, activation='relu')(x)
y_output = layers.Dense(6)(x)

model = Model(x_input, y_output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train,y_train, epochs=500, batch_size=64, validation_data=(X_test_fe, y_test))


model.save('model_FE_DNN.keras')

y_pred = model.predict(X_test_fe)

y_pred = pd.DataFrame(y_pred, columns = ['Tensile_strength','Elongation_Break','Toughness','Fracture_toughness','Flexure_modulus','Flexure_strength'])

r2_scores = {}
for i in range(y_test.shape[1]):
  r2_scores[y_test.columns[i]] = r2_score(y_test.iloc[:, i], y_pred.iloc[:, i])

print("R2 Scores for each target variable:")
for target_variable, score in r2_scores.items():
  print(f"{target_variable}: {score}")
