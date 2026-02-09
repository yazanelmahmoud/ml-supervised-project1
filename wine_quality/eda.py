"""
Exploratory Data Analysis for Wine Quality.
Functions to compute class distribution, basic stats, and EDA plots.
Used to ground hypotheses (e.g. imbalance, separability, dimensionality).
"""

import os
import sys
from contextlib import contextmanager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import TARGET_COLUMN, OUTPUT_DIR, PROJECT_DIR


class TeeOutput:
    """Context manager that writes to both stdout and a file."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self.stdout = sys.stdout
        
    def __enter__(self):
        self.file = open(self.file_path, 'w', encoding='utf-8')
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()
            
    def write(self, data):
        self.stdout.write(data)
        if self.file:
            self.file.write(data)
            
    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()


def _get_numeric_and_categorical(df):
    """Return lists of numeric and categorical column names (excluding target)."""
    exclude = [TARGET_COLUMN]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in numeric if c not in exclude]
    categorical = [c for c in categorical if c not in exclude]
    return numeric, categorical


def class_distribution(df):
    """Compute and print class distribution and imbalance ratio."""
    counts = df[TARGET_COLUMN].value_counts()
    total = len(df)
    pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    minor = counts.min()
    major = counts.max()
    ratio = major / minor if minor > 0 else float("inf")
    print("Class distribution (target: quality)")
    print(counts.to_string())
    print(f"\nPercentages:\n{pct.to_string()}")
    print(f"\nImbalance ratio (majority/minority): {ratio:.2f}")
    return {"counts": counts, "ratio": ratio, "pct": pct}


def missing_and_dtypes(df):
    """Report missing values and dtypes."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("Missing values:")
    if missing.empty:
        print("  None.")
    else:
        print(missing.to_string())
    print("\nDtypes:")
    print(df.dtypes.to_string())
    return missing


def numeric_summary(df, numeric_cols=None):
    """Describe numeric features."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    sub = df[numeric_cols] if numeric_cols else pd.DataFrame()
    if sub.empty:
        print("No numeric columns.")
        return None
    print("Numeric features — describe:")
    print(sub.describe().to_string())
    return sub.describe()


def categorical_summary(df, categorical_cols=None, top=None):
    """Value counts for categoricals (all values per column)."""
    if categorical_cols is None:
        _, categorical_cols = _get_numeric_and_categorical(df)
    if not categorical_cols:
        print("No categorical columns.")
        return {}
    out = {}
    for col in categorical_cols:
        vc = df[col].value_counts()
        if top is not None:
            print(f"\n{col} (top {top}):\n{vc.head(top).to_string()}")
        else:
            print(f"\n{col} (all values):\n{vc.to_string()}")
        out[col] = vc
    return out


def plot_class_balance(df, save=True):
    """Bar plot of target class counts."""
    counts = df[TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Target distribution (quality)")
    ax.set_xlabel(TARGET_COLUMN)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_class_balance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_numeric_distributions(df, numeric_cols=None, save=True):
    """Distributions of all numeric features (overall, not by class)."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    if not numeric_cols:
        return
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(numeric_cols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        df[col].hist(ax=ax, bins=30, edgecolor="black", color="steelblue", alpha=0.7)
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.set_xlabel("Value")
    for idx in range(len(numeric_cols), axes.size):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_numeric_distributions.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_type_by_quality(df, type_col="type", save=True):
    """
    Visualize type (red/white) showing percentage of each quality class for each type.
    Creates a grouped bar chart showing quality distribution for each wine type.
    """
    if type_col not in df.columns:
        print(f"Warning: Column '{type_col}' not found in dataset. Skipping type visualization.")
        return
    
    # Create cross-tabulation
    ct = pd.crosstab(df[type_col], df[TARGET_COLUMN], normalize='index') * 100
    
    # Print summary
    print(f"\nQuality distribution by {type_col} (percentages):")
    print("-" * 60)
    print(ct.to_string())
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique types and quality classes
    types = ct.index.tolist()
    quality_classes = ct.columns.tolist()
    x = np.arange(len(types))
    width = 0.8 / len(quality_classes)  # Width of bars
    
    # Create bars for each quality class
    colors = plt.cm.Set3(np.linspace(0, 1, len(quality_classes)))
    bars = []
    for i, quality in enumerate(quality_classes):
        values = ct[quality].values
        bar = ax.bar(x + i * width, values, width, label=f'Quality {quality}', 
                     color=colors[i], edgecolor='black', alpha=0.8)
        bars.append(bar)
        
        # Add percentage labels on bars
        for j, (val, rect) in enumerate(zip(values, bar)):
            if val > 0:  # Only show label if value > 0
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel(type_col.capitalize(), fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Quality Distribution by {type_col.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(quality_classes) - 1) / 2)
    ax.set_xticklabels(types)
    ax.legend(title='Quality Class', loc='upper right')
    ax.set_ylim([0, max(ct.max(axis=1)) * 1.2])  # Add some space at top for labels
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_type_by_quality.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_correlation_with_target(df, numeric_cols=None, save=True):
    """Correlation of numeric features with target."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    if not numeric_cols:
        return
    y_numeric = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    raw = df[numeric_cols].corrwith(y_numeric)
    corrs = raw.reindex(raw.abs().sort_values().index)
    
    print("\nCorrelation of numeric features with target:")
    print("-" * 60)
    for feature, corr_val in corrs.items():
        print(f"  {feature:20s}: {corr_val:7.4f}")
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(corrs) * 0.35)))
    corrs.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Correlation of numeric features with target")
    ax.set_xlabel("Correlation")
    ax.axvline(0, color="gray", linestyle="--")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_with_target.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return corrs


def plot_numeric_correlation_matrix(df, numeric_cols=None, save=True):
    """Heatmap of correlations between numeric features."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    if not numeric_cols:
        return
    corr = df[numeric_cols].corr()
    
    print("\nCorrelation matrix (numeric features):")
    print("-" * 60)
    print(corr.to_string())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix (numeric features)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return corr


def run_eda(df, save_figures=True, save_results_to_file=True):
    """
    Run full EDA: class distribution, missing/dtypes, numeric and categorical
    summaries, and all plots. Figures saved to config.OUTPUT_DIR if save_figures.
    
    Args:
        df: DataFrame to analyze
        save_figures: If True, save plots to OUTPUT_DIR
        save_results_to_file: If True, save all printed output to EDA_RESULTS.txt
    """
    results_file = os.path.join(PROJECT_DIR, "EDA_RESULTS.txt")
    
    if save_results_to_file:
        with TeeOutput(results_file):
            _run_eda_internal(df, save_figures)
        print(f"\nEDA results saved to: {results_file}")
    else:
        _run_eda_internal(df, save_figures)
    
    numeric_cols, categorical_cols = _get_numeric_and_categorical(df)
    return {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}


def _run_eda_internal(df, save_figures=True):
    """Internal function that performs the actual EDA analysis."""
    print("\nFirst 5 rows:\n", df.head())
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS — Wine Quality")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    numeric_cols, categorical_cols = _get_numeric_and_categorical(df)
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    print("\n" + "-" * 40)
    class_distribution(df)

    print("\n" + "-" * 40)
    missing_and_dtypes(df)

    print("\n" + "-" * 40)
    numeric_summary(df, numeric_cols)

    print("\n" + "-" * 40)
    categorical_summary(df, categorical_cols, top=None)

    print("\n" + "-" * 40)
    print("Plots (saving to outputs/):")
    plot_class_balance(df, save=save_figures)
    plot_numeric_distributions(df, numeric_cols=numeric_cols, save=save_figures)
    plot_type_by_quality(df, type_col="type", save=save_figures)
    plot_correlation_with_target(df, numeric_cols=numeric_cols, save=save_figures)
    plot_numeric_correlation_matrix(df, numeric_cols=numeric_cols, save=save_figures)

    print("\nEDA complete.")
