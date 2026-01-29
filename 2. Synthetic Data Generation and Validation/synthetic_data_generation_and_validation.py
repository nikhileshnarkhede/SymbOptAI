# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, ttest_ind
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

df = pd.read_excel('Condition 1-9 analysis.xlsx', sheet_name='Mean_with_std')
df.columns = [
    'Layer_thickness', 'Printing_speed', 'Bed_temperature', 'Nozzle_temperature',
    'Tensile_strength', 'Elongation_Break', 'Toughness','Fracture_toughness','Flexure_modulus','Flexure_strength',
    'Tensile_strength_STD', 'Elongation_Break_STD', 'Toughness_STD', 'Fracture_toughness_STD','Flexure_modulus_STD','Flexure_strength_STD'
]
df1 = pd.DataFrame(columns = ['Layer_thickness', 'Printing_speed', 'Bed_temperature',
       'Nozzle_temperature', 'Tensile_strength', 'Elongation_Break',
       'Toughness','Fracture_toughness','Flexure_modulus','Flexure_strength'])
df2 = pd.DataFrame(columns = ['Layer_thickness', 'Printing_speed', 'Bed_temperature',
       'Nozzle_temperature', 'Tensile_strength', 'Elongation_Break',
       'Toughness','Fracture_toughness','Flexure_modulus','Flexure_strength'])
       
       
bootstrap = 10
STD_Scale = 0.8
Tog_STD = 0.65


for j in range(len(df)):
  df1['Nozzle_temperature'] = [df['Nozzle_temperature'][j]]*bootstrap
  df1['Bed_temperature'] = [df['Bed_temperature'][j]]*bootstrap
  df1['Printing_speed'] = [df['Printing_speed'][j]]*bootstrap
  df1['Layer_thickness'] = [df['Layer_thickness'][j]]*bootstrap
  df1['Tensile_strength'] = np.random.normal(loc=df['Tensile_strength'][j], scale=STD_Scale*df['Tensile_strength_STD'][j], size = bootstrap)
  df1['Elongation_Break'] = np.random.normal(loc=df['Elongation_Break'][j], scale=STD_Scale*df['Elongation_Break_STD'][j], size = bootstrap)
  df1['Toughness'] = np.random.normal(loc=df['Toughness'][j], scale= Tog_STD * df['Toughness_STD'][j], size = bootstrap)
  df1['Fracture_toughness'] = np.random.normal(loc=df['Fracture_toughness'][j], scale=STD_Scale*df['Fracture_toughness_STD'][j], size = bootstrap)
  df1['Flexure_modulus'] = np.random.normal(loc=df['Flexure_modulus'][j], scale=STD_Scale*df['Flexure_modulus_STD'][j], size = bootstrap)
  df1['Flexure_strength'] = np.random.normal(loc=df['Flexure_strength'][j], scale=STD_Scale*df['Flexure_strength_STD'][j], size = bootstrap)
  df2 = pd.concat([df2,df1])

df2.reset_index(inplace= True,drop='index')

df2.to_csv('df_bootstrap.csv', index=False)

df.to_csv('df_mean_with_std.csv', index=False)




# ============================================================================
# GLOBAL STYLE SETTINGS
# ============================================================================

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data(data, prop_name=None):
    """Extract numeric array from DataFrame or array-like input"""
    if isinstance(data, pd.DataFrame):
        if prop_name is None:
            raise ValueError("prop_name required when passing DataFrame")
        if prop_name not in data.columns:
            raise ValueError(f"'{prop_name}' not found in DataFrame columns")
        arr = data[prop_name].values
    elif isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = np.array(data)

    arr = arr.flatten().astype(float)
    arr = arr[~np.isnan(arr)]
    return arr

def format_axis(ax, xlabel=None, ylabel=None, title=None):
    """Apply formatting with tick marks inside on all 4 sides"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color='black')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='black')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='black', pad=10)

    ax.tick_params(axis='both', which='major',
                   direction='in', length=6, width=1.2,
                   colors='black', labelsize=12,
                   top=True, bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor',
                   direction='in', length=3, width=1,
                   top=True, bottom=True, left=True, right=True)

    for label in ax.get_xticklabels():
        label.set_color('black')
        label.set_fontsize(12)
    for label in ax.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(12)

def format_legend(ax, loc='best'):
    """Format legend - black text, black border"""
    legend = ax.legend(loc=loc, frameon=True, edgecolor='black',
                       facecolor='white', fontsize=12, framealpha=1)
    if legend:
        legend.get_frame().set_linewidth(1.5)
        for text in legend.get_texts():
            text.set_color('black')
            text.set_fontsize(12)

# ============================================================================
# 1. HISTOGRAM WITH KDE
# ============================================================================

def plot_histogram(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single histogram + KDE plot with smart legend placement"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    sns.histplot(exp, kde=True, stat='density', alpha=0.4, color='steelblue',
                 label=f'Experimental', ax=ax, bins=10,
                 edgecolor='black', linewidth=0.8)
    sns.histplot(syn, kde=True, stat='density', alpha=0.4, color='coral',
                 label=f'Synthetic', ax=ax, bins=20,
                 edgecolor='black', linewidth=0.8)

    ax.axvline(exp.mean(), color='steelblue', linestyle='--', linewidth=2,
               label=f'Exp μ = {exp.mean():.2f}')
    ax.axvline(syn.mean(), color='coral', linestyle='--', linewidth=2,
               label=f'Syn μ = {syn.mean():.2f}')

    format_axis(ax, xlabel=display_name, ylabel='Density', title=f'Distribution: {display_name}')

    # Smart legend placement: check where data is concentrated
    all_data = np.concatenate([exp, syn])
    data_min, data_max = all_data.min(), all_data.max()
    data_center = (data_min + data_max) / 2
    mean_position = np.mean([exp.mean(), syn.mean()])

    # Place legend on opposite side of data concentration
    if mean_position > data_center:
        legend_loc = 'upper left'
    else:
        legend_loc = 'upper right'

    format_legend(ax, loc=legend_loc)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 2. BOXPLOT
# ============================================================================

def plot_boxplot(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single boxplot"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    data_df = pd.DataFrame({
        'Value': np.concatenate([exp, syn]),
        'Type': ['Experimental']*len(exp) + ['Synthetic']*len(syn)
    })

    sns.boxplot(data=data_df, x='Type', y='Value', ax=ax,
                palette={'Experimental': 'steelblue', 'Synthetic': 'coral'},
                width=0.5, showfliers=False, linewidth=1.5)

    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    sns.stripplot(data=data_df, x='Type', y='Value', ax=ax,
                  color='black', alpha=0.5, size=5, jitter=0.15)

    format_axis(ax, xlabel='', ylabel=display_name, title=f'Boxplot: {display_name}')
    ax.set_xticklabels(['Experimental', 'Synthetic'], fontsize=12,
                       fontweight='bold', color='black')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 3. VIOLIN PLOT
# ============================================================================

def plot_violin(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single violin plot"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    data_df = pd.DataFrame({
        'Value': np.concatenate([exp, syn]),
        'Type': ['Experimental']*len(exp) + ['Synthetic']*len(syn)
    })

    sns.violinplot(data=data_df, x='Type', y='Value', ax=ax,
                   palette={'Experimental': 'steelblue', 'Synthetic': 'coral'},
                   inner='box', cut=0, linewidth=1.5)

    for collection in ax.collections:
        collection.set_edgecolor('black')
        collection.set_linewidth(1.5)

    format_axis(ax, xlabel='', ylabel=display_name, title=f'Violin Plot: {display_name}')
    ax.set_xticklabels(['Experimental', 'Synthetic'], fontsize=12,
                       fontweight='bold', color='black')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 4. Q-Q PLOT
# ============================================================================

def plot_qq(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single Q-Q plot with R² in text box"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    min_len = min(len(exp), len(syn))
    quantiles = np.linspace(0, 100, min_len)
    exp_q = np.percentile(exp, quantiles)
    syn_q = np.percentile(syn, quantiles)

    ax.scatter(exp_q, syn_q, alpha=0.7, s=50, color='darkblue',
               edgecolors='black', linewidth=1)

    min_val = min(exp_q.min(), syn_q.min())
    max_val = max(exp_q.max(), syn_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    r_squared = np.corrcoef(exp_q, syn_q)[0, 1]**2

    format_axis(ax, xlabel=f'Experimental {display_name}', ylabel=f'Synthetic {display_name}',
                title=f'Q-Q Plot: {display_name}')

    # R² text box inside plot
    textstr = f'R² = {r_squared:.4f}'
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5, alpha=1)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', color='black', verticalalignment='top',
            horizontalalignment='left', bbox=props)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 5. MEAN COMPARISON BAR
# ============================================================================

def plot_mean_bar(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single mean comparison bar chart"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    x = ['Experimental', 'Synthetic']
    means = [exp.mean(), syn.mean()]
    stds = [exp.std(), syn.std()]

    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=['steelblue', 'coral'], edgecolor='black',
                  linewidth=1.5, width=0.5,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05*max(means),
                f'{val:.2f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='black')

    format_axis(ax, xlabel='', ylabel=display_name, title=f'Mean ± Std: {display_name}')
    ax.set_xticklabels(x, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 6. STD COMPARISON BAR
# ============================================================================

def plot_std_bar(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single std comparison bar chart"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    x = ['Experimental', 'Synthetic']
    stds = [exp.std(), syn.std()]

    bars = ax.bar(x, stds, color=['steelblue', 'coral'],
                  edgecolor='black', linewidth=1.5, width=0.5)

    for bar, val in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(stds),
                f'{val:.2f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='black')

    format_axis(ax, xlabel='', ylabel='Standard Deviation', title=f'Std: {display_name}')
    ax.set_xticklabels(x, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

# ============================================================================
# 7. ERROR ANALYSIS BAR
# ============================================================================

def plot_error_bar(df_exp, df_syn, prop_name, filename=None, figsize=(3, 3), label=None):
    """Single error analysis bar"""
    fig, ax = plt.subplots(figsize=figsize)

    exp = get_data(df_exp, prop_name)
    syn = get_data(df_syn, prop_name)

    display_name = label if label else prop_name

    mean_err = abs(exp.mean() - syn.mean()) / exp.mean() * 100
    std_err = abs(exp.std() - syn.std()) / exp.std() * 100

    x = ['Mean Error (%)', 'Std Error (%)']
    errors = [mean_err, std_err]
    colors = ['#2ecc71' if e < 5 else '#e74c3c' for e in errors]

    bars = ax.bar(x, errors, color=colors, edgecolor='black', linewidth=1.5, width=0.5)

    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='black')

    format_axis(ax, xlabel='', ylabel='Relative Error (%)', title=f'Error: {display_name}')
    ax.set_xticklabels(x, fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()

import os
os.mkdir('plots_syn_data')

# =============================================================================
# GENERATE ALL PLOTS FOR ALL PROPERTIES WITH UNITS
# =============================================================================

# Property names in DataFrame (must match your column names exactly)
properties = [
    'Tensile strength',
    'Elongation at break',
    'Toughness',
    'Fracture toughness',
    'Flexure modulus',
    'Flexure strength'
]

# Property names with units for display labels
labels = [
    'Tensile Strength (MPa)',
    'Elongation at Break (%)',
    'Toughness (MPa)',
    'Fracture Toughness (MPa√m)',
    'Flexure Modulus (GPa)',
    'Flexure Strength (MPa)'
]

# Generate all plots
for prop, lbl in zip(properties, labels):
    safe_name = prop.replace(" ", "_")

    print(f"\n{'='*60}")
    print(f"Generating plots for: {lbl}")
    print('='*60)

    # Histogram
    plot_histogram(df, df2, prop, f'plots_syn_data/Histogram_{safe_name}.png', figsize=(5, 5), label=lbl)

    # Boxplot
    plot_boxplot(df, df2, prop, f'plots_syn_data/Boxplot_{safe_name}.png', figsize=(5, 5), label=lbl)

    # Violin
    plot_violin(df, df2, prop, f'plots_syn_data/Violin_{safe_name}.png', figsize=(3.5, 6), label=lbl)

    # Q-Q
    plot_qq(df, df2, prop, f'plots_syn_data/QQ_{safe_name}.png', figsize=(5, 5), label=lbl)

print(f"\n{'='*60}")
print("✓ ALL PLOTS GENERATED!")
print(f"Total: {len(properties) * 4} plots")
print('='*60)

