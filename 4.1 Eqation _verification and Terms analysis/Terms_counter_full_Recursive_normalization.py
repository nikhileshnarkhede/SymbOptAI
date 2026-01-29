import pandas as pd
from Terms_collection import extract_terms
from Equation_Terms import analyze
from sklearn.metrics import r2_score
from Simple_terms_breaker import simplify
from Term_wise_calculation import build_formula
from Str_to_formula_converter import evaluate_formula
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from canonicalize_term_Recursive_normalization  import canonicalize_term

        

Formulas_dir = "D:\\MECH-DNN PAPER\\VC SIR PAPER\\SRA_EQUATION\\SRA\\Condition 1-9 analysis\\Y_2\\set_of_formula_only"
formulas = []

# Get all .txt files sorted (so EQ_1.txt comes before EQ_10.txt)
txt_files = sorted([f for f in os.listdir(Formulas_dir) if f.endswith(".txt")])

print(f"Found {len(txt_files)} txt files")

for file in txt_files:
    filename = os.path.join(Formulas_dir, file)
    with open(filename, "r") as f:
        formula = f.read().strip()
        formulas.append(formula)

# Print loaded formulas
for i, fml in enumerate(formulas, 1):
    print(f"EQ_{i}: {fml}")

simplified_formulas = {}
for i, formula in enumerate(formulas):
    simplified_formulas[f'formula_{i+1}'] = analyze(formula,verbose = False)['terms_out']


simplified_formulas

all_terms = []
for terms_list in simplified_formulas.values():
    for term in terms_list:
        all_terms.append(term)
        
term_counts = {}
for term in all_terms:
    canon = canonicalize_term(term, consider_coeff=True)
    term_counts[canon] = term_counts.get(canon, 0) + 1
    
    
term_counts_df = pd.DataFrame(list(term_counts.items()), columns=['Term', 'Frequency'])
term_counts_df = term_counts_df.sort_values(by='Frequency', ascending=False)
print(term_counts_df)
term_counts_df.to_csv("D:\\MECH-DNN PAPER\\VC SIR PAPER\\SRA_EQUATION\\Equation Terms_upadate\\Set of formula\\formulas_txt\\Comman_Terms\\term_counts_with_consider_coeff.csv", index=False)


# plt.figure(figsize=(12, 6))
# sns.barplot(x='Term', y='Frequency', data=term_counts_df)
# plt.xticks(rotation=90)
# plt.title('Frequency of Terms in Simplified Formulas')
# plt.xlabel('Term')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('D:\\MECH-DNN PAPER\\VC SIR PAPER\\SRA_EQUATION\\Equation Terms_upadate\\Set of formula\\formulas_txt\\Comman_Terms\\term_counts_with_consider_coeff_.png')
# #plt.show()

term_counts = {}
for term in all_terms:
    canon = canonicalize_term(term, consider_coeff=False)
    term_counts[canon] = term_counts.get(canon, 0) + 1
    
term_counts_df = pd.DataFrame(list(term_counts.items()), columns=['Term', 'Frequency'])
term_counts_df = term_counts_df.sort_values(by='Frequency', ascending=False)
print(term_counts_df)
term_counts_df.to_csv('D:\\MECH-DNN PAPER\\VC SIR PAPER\\SRA_EQUATION\\Equation Terms_upadate\\Set of formula\\formulas_txt\\Comman_Terms\\term_counts_without_consider_coeff.csv', index=False)

# plt.figure(figsize=(12, 6))
# sns.barplot(x='Term', y='Frequency', data=term_counts_df)
# plt.xticks(rotation=90)
# plt.title('Frequency of Terms in Simplified Formulas')
# plt.xlabel('Term')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('D:\\MECH-DNN PAPER\\VC SIR PAPER\\SRA_EQUATION\\Equation Terms_upadate\\Set of formula\\formulas_txt\\Comman_Terms\\term_counts_without_consider_coeff_.png')
# #plt.show()


