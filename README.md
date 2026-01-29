# SymbOptAI: Symbolic Regression-Based Multi-Objective Optimization Framework for Additive Manufacturing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-Pending-green.svg)]()

## Overview

**SymbOptAI** is an end-to-end computational framework that integrates **Symbolic Regression Analysis (SRA)**, **Deep Neural Networks (DNN)**, and **Multi-Objective Optimization (NSGA-II)** for predicting and optimizing mechanical properties of 3D printed materials. This repository accompanies the research paper submitted to the journal *Additive Manufacturing*.

The framework discovers **interpretable mathematical equations** that relate FFF (Fused Filament Fabrication) printing parameters to six critical mechanical properties, offering both predictive accuracy and physical interpretability—bridging the gap between black-box machine learning and explainable models in materials science.

---

## Research Highlights

- **Interpretable Equations**: Discovers closed-form mathematical relationships using genetic programming-based symbolic regression
- **Multi-Property Prediction**: Simultaneously predicts six mechanical properties (tensile strength, elongation at break, toughness, fracture toughness, flexure modulus, flexure strength)
- **Feature Engineering from Symbolic Terms**: Extracts equation terms as engineered features for enhanced DNN performance
- **Inverse Design**: Employs NSGA-II for multi-objective optimization to identify optimal printing parameters
- **Statistical Validation**: Rigorous bootstrap resampling for robust model training and validation

---

## Repository Structure

```
SymbOptAI/
│
├── 1. DOE/
│   ├── doe.py                          # Taguchi L9 Design of Experiments generator
│   └── requirements.txt
│
├── 2. Synthetic Data Generation and Validation/
│   └── synthetic_data_generation_and_validation.py
│                                        # Bootstrap resampling & statistical validation
│
├── 3. SRA Equation Generator/
│   ├── Equations_Generator.py          # Symbolic regression with gplearn
│   ├── main_job.sh                     # HPC job submission script
│   ├── submit_all_jobs.sh              # Batch job submission
│   └── requirements.txt
│
├── 4.1 Equation Verification and Terms Analysis/
│   ├── Verification_of_equations.py    # R² validation for discovered equations
│   ├── Equation_Terms.py               # AST parser for equation decomposition
│   ├── Terms_counter.py                # Term frequency analysis
│   ├── canonicalize_term_*.py          # Term normalization utilities
│   ├── Term_wise_calculation.py        # Term-wise computation engine
│   └── Str_to_formula_converter.py     # String to executable formula converter
│
├── 4.2 Feature Importances for Each Property/
│   ├── feature_importances_for_each_property.py
│   │                                   # Permutation importance analysis
│   ├── Best_EQ/                        # Best equations for each property
│   └── Output/                         # Feature importance results
│
├── 5. DNN with Feature Engineering/
│   ├── final_FE_DNN_pipeline.py        # Complete FE + DNN training pipeline
│   ├── FE_DNN.py                       # DNN model architecture
│   ├── model_FE_DNN.py                 # Model utilities
│   └── requirements.txt
│
├── 6. NSGA 2 Optimization/
│   ├── Optimization.py                 # NSGA-II inverse design optimization
│   └── requirements.txt
│
└── README.md
```

---

## Methodology

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SymbOptAI Framework                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │                  │
│  │     DOE      │───▶│  Bootstrap   │───▶│    SRA       │                  │
│  │  (Taguchi)   │    │  Resampling  │    │  (gplearn)   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                 │                           │
│                                                 ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Stage 6    │    │   Stage 5    │    │   Stage 4    │                  │
│  │   NSGA-II    │◀───│    DNN +     │◀───│    Term      │                  │
│  │ Optimization │    │Feature Eng.  │    │  Extraction  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage Descriptions

| Stage | Component | Description |
|-------|-----------|-------------|
| **1** | Design of Experiments | Taguchi L9 orthogonal array for systematic parameter space exploration |
| **2** | Data Augmentation | Bootstrap resampling with experimental standard deviations (n=10 per condition) |
| **3** | Symbolic Regression | Genetic programming with exhaustive hyperparameter sweep (1500 configurations) |
| **4** | Equation Analysis | AST parsing, term extraction, and frequency analysis across all equations |
| **5** | Feature-Enhanced DNN | Terms from symbolic equations used as engineered features for DNN |
| **6** | Multi-Objective Optimization | NSGA-II for inverse design with 6 simultaneous objectives |

---

## Input Parameters and Output Properties

### Input Variables (Printing Parameters)

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Layer Thickness | X₀ | 0.15 – 0.25 | mm |
| Printing Speed | X₁ | 40 – 60 | mm/s |
| Bed Temperature | X₂ | 95 – 105 | °C |
| Nozzle Temperature | X₃ | 230 – 260 | °C |

### Output Variables (Mechanical Properties)

| Property | Symbol | Unit |
|----------|--------|------|
| Tensile Strength | Y₁ | MPa |
| Elongation at Break | Y₂ | % |
| Toughness | Y₃ | MPa |
| Fracture Toughness | Y₄ | MPa√m |
| Flexure Modulus | Y₅ | GPa |
| Flexure Strength | Y₆ | MPa |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for accelerated DNN training)

### Setup

```bash
# Clone the repository
git clone https://github.com/nikhileshnarkhede/SymbOptAI.git
cd SymbOptAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
numpy>=1.26.4
pandas>=2.2.3
matplotlib>=3.10.0
scikit-learn>=1.5.2
gplearn>=0.4.2
graphviz>=0.20.3
tensorflow>=2.15.0
pymoo>=0.6.0
scipy>=1.16.0
openpyxl>=3.1.5
joblib>=1.5.2
seaborn>=0.13.0
```

---

## Usage

### 1. Design of Experiments

```bash
cd "1. DOE"
python doe.py
# Output: designs_output.xlsx (Taguchi L9 matrix)
```

### 2. Synthetic Data Generation

```bash
cd "2. Synthetic Data Generation and Validation"
python synthetic_data_generation_and_validation.py
# Output: df_bootstrap.csv, validation plots
```

### 3. Symbolic Regression Analysis

```bash
cd "3. SRA Equation Generator"
python Equations_Generator.py
# Output: symbolic_regression_results.csv, best_equation.txt
```

**Note**: For HPC environments, use the provided SLURM scripts:
```bash
sbatch main_job.sh
```

### 4. Equation Verification

```bash
cd "4.1 Equation Verification and Terms Analysis"
python Verification_of_equations.py
# Output: Actual_vs_Predicted/*.xlsx, scatter plots
```

### 5. DNN Training with Feature Engineering

```bash
cd "5. DNN with Feature Engineering"
python final_FE_DNN_pipeline.py
# Output: final_FE_DNN_pipeline.joblib, model_FE_DNN.keras
```

### 6. Multi-Objective Optimization

```bash
cd "6. NSGA 2 Optimization"
python Optimization.py
# Output: Pareto-optimal solutions for printing parameters
```

---

## Symbolic Regression Configuration

The symbolic regression employs an exhaustive hyperparameter sweep across 1500 unique configurations:

```python
# GP Hyperparameters
POP_SIZE = 3000              # Population size
GENERATIONS = 100            # Number of generations
TOURNAMENT_SIZE = 30         # Tournament selection size
PARSIMONY = 'auto'          # Automatic parsimony coefficient

# Function Set
FUNCTION_SET = [
    'add', 'sub', 'mul', 'div',  # Arithmetic
    'sqrt', 'log', 'abs',        # Unary
    'neg', 'inv',                # Inverse operations
    'max', 'min',                # Comparison
    'sin', 'cos', 'tan'          # Trigonometric
]

# Probability Grid (Anti-bloat controls)
p_crossover:         0.18 → 0.28 (6 steps)
p_subtree_mutation:  variable (3 steps)
p_hoist_mutation:    variable (6 steps)
random_state:        42 → 45 (4 seeds)
```

---

## DNN Architecture

```
Input Layer (n_features from symbolic terms)
    │
    ▼
Dense(128, activation='relu')
    │
    ▼
Dense(64, activation='relu')
    │
    ▼
Dense(32, activation='relu')
    │
    ▼
Dense(14, activation='relu')
    │
    ▼
Output Layer (6 properties)
```

---

## Results Reproduction

To reproduce the results presented in the paper:

1. Ensure all experimental data files are placed in the appropriate directories
2. Execute scripts in sequential order (Stage 1 → Stage 6)
3. For exact reproduction, use the same random seeds specified in the code



---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{narkhede2025symboptai,
  title={Symbolic Regression-Based Multi-Objective Optimization Framework for Predicting Mechanical Properties of 3D Printed Materials},
  author={Narkhede, Nikhilesh and [Co-Author Names]},
  journal={Additive Manufacturing},
  year={2025},
  publisher={Elsevier},
  doi={[DOI pending]}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Computational resources provided by the **Unity High-Performance Computing Cluster** at UMass


---

## Contact

For questions or collaborations, please contact:

- **Nikhilesh Narkhede** - University of Massachusetts Dartmouth
- M.S. Data Science | Research Assistant
- GitHub: [@nikhileshnarkhede](https://github.com/nikhileshnarkhede)

---

## Supplementary Materials

Additional materials including:
- Complete equation set for all properties
- Detailed hyperparameter sweep results
- Extended validation plots

are available in the supplementary materials of the published article.
