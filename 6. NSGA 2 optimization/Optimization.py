import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from warnings import filterwarnings
filterwarnings('ignore')

from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import joblib

from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------
# Protected Binary Operators
# ---------------------------------------------

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    # Protected division (similar to gplearn)
    # If denominator is near zero -> return 1.0
    safe_b = np.where(np.abs(b) < 1e-3, 1.0, b)
    return a / safe_b

def max_op(a, b):
    return np.maximum(a, b)

def min_op(a, b):
    return np.minimum(a, b)

# ---------------------------------------------
# Protected Unary Operators
# ---------------------------------------------

def sqrt(a):
    # sqrt(|a|)
    return np.sqrt(np.abs(a))

def inv(a):
    # Protected inverse
    # If a is near zero -> return 0.0
    safe_a = np.where(np.abs(a) < 1e-3, 0.0, a)
    return 1.0 / safe_a

def log(a):
    # Protected logarithm:
    # log(|a|), but if |a| < 0.001 return 0.0
    abs_a = np.abs(a)
    safe_a = np.where(abs_a < 1e-3, 1e-3, abs_a)
    return np.log(safe_a)

def neg(a):
    return -a

def abs_val(a):
    return np.abs(a)

def sin(a):
    return np.sin(a)

def cos(a):
    return np.cos(a)

def tan(a):
    # Avoid tan blow-up near pi/2 + k*pi by clipping
    # Clip inputs to prevent extremely large numbers
    clipped = np.clip(a, -1e6, 1e6)
    return np.tan(clipped)
    
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


class InverseModelProblem(ElementwiseProblem):
    def __init__(self, pipeline):
        super().__init__(
            n_var=4,
            n_obj=6,
            n_constr=0,
            xl=np.array([0.15, 40, 95, 230]),
            xu=np.array([0.25, 60, 105, 260])
        )
        self.pipeline = pipeline

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(x).reshape(1, -1)
        y_pred = self.pipeline.predict(x)
        out["F"] = -y_pred.flatten()   # NSGA-II maximizes objectives



    
model = keras.models.load_model('model_FE_DNN.keras')

problem = InverseModelProblem(pipeline)

algorithm = NSGA2(
   pop_size=10,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

problem = InverseModelProblem(pipeline=pipeline)

termination = get_termination("n_gen", 200)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)
# ============================
# Load the saved unified pipeline
# ============================
pipeline = joblib.load('final_FE_DNN_pipeline.joblib')

# ============================
# Get optimized input values from NSGA-II
# ============================
X_optimized = res.X  # shape: (n_solutions, 4)

# Convert to DataFrame with correct column names
X_df = pd.DataFrame(
    X_optimized,
    columns=['Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature']
)

# ============================
# Predict outputs using the unified pipeline
# (includes feature engineering + scaling + trained DNN)
# ============================
Y_optimized = pipeline.predict(X_df)  # shape: (n_solutions, 6)
Y_optimized = Y_optimized/1000  # Reshape to (n_solutions, 6)

# Convert predictions to DataFrame
Y_df = pd.DataFrame(
    Y_optimized,
    columns=['Tensile_strength', 'Elongation_Break', 'Toughness',
             'Fracture_toughness', 'Flexure_modulus', 'Flexure_strength']
)

# ============================
# Combine inputs and outputs
# ============================
df_result = pd.concat([X_df, Y_df], axis=1)
df_result["Elongation_Break"] /= 100
df_result["Toughness"] /= 100
df_result["Fracture_toughness"] /= 10
df_result["Flexure_modulus"] *= 100

# Display first few optimized solutions
print(df_result.head())
# Get the optimal inputs and corresponding predictions
optimal_inputs = res.X  # shape (n_points, 4)
optimal_outputs = -res.F  # negate back to original direction (maximized)


df = pd.DataFrame(optimal_inputs, columns=["Layer_thickness", "Printing_speed", "Bed_temperature", "Nozzle_temperature"])
df["Tensile_strength"] = optimal_outputs[:, 0]
df["Elongation_Break"] = optimal_outputs[:, 1]
df["Toughness"] = optimal_outputs[:, 2]
df["Fracture_toughness"] = optimal_outputs[:, 3]
df["Flexure_modulus"] = optimal_outputs[:, 4]
df["Flexure_strength"] = optimal_outputs[:, 5]



