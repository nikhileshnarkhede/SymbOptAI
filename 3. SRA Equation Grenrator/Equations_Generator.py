# ==============================
# Imports
# ==============================
import os
import datetime
import numpy as np
import pandas as pd
import graphviz
from warnings import filterwarnings

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

filterwarnings('ignore')

# ==============================
# Flags
# ==============================
bootstrap1 = True
bootstrap = 10


df = pd.read_csv('df_mean_with_std.csv')
df2 = pd.read_csv('df_bootstrap.csv')


# ==============================
# Train / Test
# ==============================
if not bootstrap1:
    X_train = df[['Layer_thickness', 'Printing_speed',
                  'Bed_temperature', 'Nozzle_temperature']]
    y_train = df['Tensile_strength']
else:
    X_train = df2[['Layer_thickness', 'Printing_speed',
                   'Bed_temperature', 'Nozzle_temperature']]
    y_train = df2['Tensile_strength']

X_test = df[['Layer_thickness', 'Printing_speed',
             'Bed_temperature', 'Nozzle_temperature']]
y_test = df['Tensile_strength']

# ==============================
# Output directories
# ==============================
os.makedirs("set_of_formula", exist_ok=True)
os.makedirs("set_of_formula_only", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# ==============================
# Run state
# ==============================
ct = 0
results = []

# ==============================
# Anti-bloat probability controls
# ==============================
BASE_PC = 0.18
STEP_PC = 0.02
PS_SHARE = 0.18
PH_SHARE = 0.40
STEP_PS = 0.005
STEP_PH = 0.02

I_STEPS = 6
J_STEPS = 3
T_STEPS = 6
CC_STEPS = 4

# ==============================
# GP Hyperparameters
# ==============================
POP_SIZE = 3000
GENERATIONS = 100
INIT_DEPTH = (1, 3)
INIT_METHOD = 'grow'
CONST_RANGE = (-10, 10)

TOURNAMENT_SIZE = 30
STOPPING_CRITERIA = 0.01
METRIC = 'mean absolute error'
BASE_RANDOM_STATE = 42
VERBOSE = 1

# ==============================
# Custom safe functions (UNCHANGED)
# ==============================
# Keep minimal; required for gplearn if not built-in on your version
neg = make_function(function=lambda x: -x, name='neg', arity=1)
inv = make_function(function=lambda x: np.where(np.abs(x) > 1e-12, 1.0/x, 0.0), name='inv', arity=1)
fmax = make_function(function=lambda x, y: np.maximum(x, y), name='max', arity=2)
fmin = make_function(function=lambda x, y: np.minimum(x, y), name='min', arity=2)
# Tanh is safer, but if you want real tan with protection:
tanp = make_function(function=lambda x: np.tan(np.clip(x, -3.0, 3.0)), name='tan', arity=1)

# ==============================
# FUNCTION SET (NO CHANGE)
# ==============================
FUNCTION_SET = [
    'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
    'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan'
]

# ==============================
# Symbolic Regression Sweep
# ==============================
for i in range(I_STEPS):
    for j in range(J_STEPS):
        for t in range(T_STEPS):
            for cc in range(CC_STEPS):

                ct += 1

                p_crossover = BASE_PC + STEP_PC * i
                p_subtree_mutation = PS_SHARE * (1 - p_crossover) + STEP_PS * j
                p_hoist_mutation = PH_SHARE * (1 - p_crossover) + STEP_PH * t
                p_point_mutation = 1.0 - (
                    p_crossover + p_subtree_mutation + p_hoist_mutation
                )

                if p_point_mutation <= 0:
                    continue

                time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                est_gp_y1 = SymbolicRegressor(
                    population_size=POP_SIZE,
                    generations=GENERATIONS,
                    init_depth=INIT_DEPTH,
                    init_method=INIT_METHOD,
                    function_set=FUNCTION_SET,
                    const_range=CONST_RANGE,
                    parsimony_coefficient='auto',
                    p_crossover=p_crossover,
                    p_subtree_mutation=p_subtree_mutation,
                    p_hoist_mutation=p_hoist_mutation,
                    p_point_mutation=p_point_mutation,
                    tournament_size=TOURNAMENT_SIZE,
                    stopping_criteria=STOPPING_CRITERIA,
                    metric=METRIC,
                    random_state=BASE_RANDOM_STATE + cc,
                    verbose=VERBOSE
                )

                est_gp_y1.fit(X_train, y_train)

                r2_train = est_gp_y1.score(X_train, y_train)
                r2_test = est_gp_y1.score(X_test, y_test)

                equation = str(est_gp_y1)
                length = est_gp_y1.run_details_['best_length'][-1]

                results.append({
                    'ID': ct,
                    'time': time2,
                    'r2_train': r2_train,
                    'r2_test': r2_test,
                    'pc': p_crossover,
                    'ps': p_subtree_mutation,
                    'ph': p_hoist_mutation,
                    'pp': p_point_mutation,
                    'parsimony': 'auto',
                    'length': length,
                    'equation': equation
                })

                with open(f"set_of_formula/EQ_{ct}.txt", "w") as f:
                    f.write(equation)

                with open(f"set_of_formula_only/EQ_{ct}.txt", "w") as f:
                    f.write(equation)

                try:
                    dot = est_gp_y1._program.export_graphviz()
                    graphviz.Source(dot).render(f"graphs/EQ_{ct}", format="pdf")
                except Exception:
                    pass

                print(
                    f"ID={ct:04d} | R2 train={r2_train:.3f} "
                    f"test={r2_test:.3f} | len={length}"
                )

# ==============================
# Save Results
# ==============================
results_df = pd.DataFrame(results).sort_values(
    by=['r2_test', 'length'], ascending=[False, True]
)

results_df.to_csv("symbolic_regression_results.csv", index=False)

best = results_df.iloc[0]

with open("best_equation.txt", "w") as f:
    f.write(best['equation'])

print("\n=== Done ===")
print("Best equation saved to best_equation.txt")
