"""
Exploratory Data Analysis for Adult Income.
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
    print("Class distribution (target: income <=50K vs >50K)")
    print(counts.to_string())
    print(f"\nPercentages:\n{pct.to_string()}")
    print(f"\nImbalance ratio (majority/minority): {ratio:.2f} (expect ~3:1)")
    return {"counts": counts, "ratio": ratio, "pct": pct}


def missing_and_dtypes(df):
    """Report missing values and dtypes. Treat '?' as missing for object columns."""
    df_check = df.copy()
    # Common missing marker in Adult dataset
    for col in df_check.select_dtypes(include=["object"]).columns:
        if col in df_check.columns:
            df_check[col] = df_check[col].replace("?", np.nan)
    missing = df_check.isna().sum()
    missing = missing[missing > 0]
    print("Missing values (after treating '?' as NaN in object columns):")
    if missing.empty:
        print("  None.")
    else:
        print(missing.to_string())
    print("\nDtypes:")
    print(df_check.dtypes.to_string())
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
    counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_title("Target distribution (income)")
    ax.set_xlabel(TARGET_COLUMN)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_class_balance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_numeric_by_class(df, numeric_cols=None, max_cols=6, save=True):
    """Distributions of numeric features by target class (overlay or subplots)."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    if not numeric_cols:
        return
    cols = numeric_cols[:max_cols]
    n = len(cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(cols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        for label in df[TARGET_COLUMN].unique():
            subset = df.loc[df[TARGET_COLUMN] == label, col]
            subset.hist(ax=ax, bins=30, alpha=0.5, label=str(label), edgecolor="black")
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.legend()
    for idx in range(len(cols), axes.size):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "eda_numeric_by_class.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def categorical_percentage_by_class(df, categorical_cols=None):
    """Print percentage of >50K for each category, sorted from highest to lowest."""
    if categorical_cols is None:
        _, categorical_cols = _get_numeric_and_categorical(df)
    if not categorical_cols:
        print("No categorical columns.")
        return {}
    
    results = {}
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[TARGET_COLUMN])
        
        # Find the >50K column
        target_values = ct.columns.tolist()
        high_income_col = None
        for val in target_values:
            if ">50K" in str(val):
                high_income_col = val
                break
        
        if high_income_col is None:
            high_income_col = target_values[-1] if len(target_values) > 1 else target_values[0]
        
        # Calculate percentage
        total_counts = ct.sum(axis=1)
        high_income_counts = ct[high_income_col]
        percentages = (high_income_counts / total_counts * 100).fillna(0)
        
        # Sort by percentage descending
        percentages_sorted = percentages.sort_values(ascending=False)
        
        # Print results
        print(f"\n{col} - Percentage of >50K (sorted from highest to lowest):")
        print("-" * 60)
        for idx, pct in percentages_sorted.items():
            total = total_counts[idx]
            high = high_income_counts[idx]
            print(f"  {idx:30s}: {pct:6.2f}% (n={int(total)}, >50K={int(high)})")
        
        results[col] = percentages_sorted
    
    return results


def plot_categorical_by_class(df, categorical_cols=None, top_n=None, save=True):
    """Bar plot showing percentage of >50K for each category, with count labels."""
    if categorical_cols is None:
        _, categorical_cols = _get_numeric_and_categorical(df)
    if not categorical_cols:
        return
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[TARGET_COLUMN])
        # Use all categories if top_n is None, otherwise use top N
        if top_n is not None:
            top_cats = ct.sum(axis=1).nlargest(top_n).index.tolist()
            ct_plot = ct.loc[ct.index.isin(top_cats)]
        else:
            ct_plot = ct
        
        # Calculate percentage of >50K for each category
        # Handle both string and numeric representations of >50K
        target_values = ct_plot.columns.tolist()
        # Find the >50K column (could be ">50K", ">50K.", etc.)
        high_income_col = None
        for val in target_values:
            if ">50K" in str(val):
                high_income_col = val
                break
        
        if high_income_col is None:
            # Fallback: assume second column or last column is >50K
            high_income_col = target_values[-1] if len(target_values) > 1 else target_values[0]
        
        # Calculate percentage
        total_counts = ct_plot.sum(axis=1)
        high_income_counts = ct_plot[high_income_col]
        percentages = (high_income_counts / total_counts * 100).fillna(0)
        
        # Sort by percentage descending (highest to lowest)
        percentages_sorted = percentages.sort_values(ascending=False)
        total_counts_sorted = total_counts[percentages_sorted.index]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(10, len(percentages_sorted) * 0.5), 4))
        bars = ax.bar(range(len(percentages_sorted)), percentages_sorted.values, color='steelblue', edgecolor='black')
        
        # Add count labels on top of bars (without affecting bar height)
        for i, (idx, pct) in enumerate(percentages_sorted.items()):
            total = total_counts_sorted[idx]
            # Show total count as label
            ax.text(i, pct, f'n={int(total)}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f"{col} by income class")
        ax.set_xlabel(col)
        ax.set_ylabel("Percentage of >50K (%)")
        ax.set_xticks(range(len(percentages_sorted)))
        ax.set_xticklabels(percentages_sorted.index, rotation=45, ha="right")
        ax.set_ylim([0, max(percentages_sorted.max() * 1.15, 10)])  # Add some space for labels
        plt.tight_layout()
        if save:
            safe_name = col.replace("-", "_")[:30]
            plt.savefig(os.path.join(OUTPUT_DIR, f"eda_cat_by_class_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


def plot_correlation_with_target(df, numeric_cols=None, save=True):
    """Correlation of numeric features with binary target (encoded 0/1)."""
    if numeric_cols is None:
        numeric_cols, _ = _get_numeric_and_categorical(df)
    if not numeric_cols:
        return
    y_bin = (df[TARGET_COLUMN].astype(str).str.strip() == ">50K").astype(int)
    raw = df[numeric_cols].corrwith(y_bin)
    corrs = raw.reindex(raw.abs().sort_values().index)
    
    # Print correlation results
    print("\nCorrelation of numeric features with target (income >50K = 1):")
    print("-" * 60)
    for feature, corr_val in corrs.items():
        print(f"  {feature:20s}: {corr_val:7.4f}")
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(corrs) * 0.35)))
    corrs.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Correlation of numeric features with target (income >50K = 1)")
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
    
    # Print correlation matrix
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


def print_education_mapping(df):
    """Print the mapping between education-num and education categories."""
    if 'education' not in df.columns or 'education-num' not in df.columns:
        print("Warning: 'education' or 'education-num' columns not found in dataset.")
        return None
    
    # Group by education-num and get unique education values for each
    mapping = df.groupby('education-num')['education'].unique()
    
    # Check if mapping is one-to-one (each education-num maps to exactly one education category)
    is_one_to_one = all(len(vals) == 1 for vals in mapping.values)
    
    print("\n" + "=" * 60)
    print("Education Number to Education Category Mapping")
    print("=" * 60)
    
    if is_one_to_one:
        print("\nMapping (one-to-one):")
        print("-" * 60)
        for edu_num in sorted(mapping.index):
            edu_cat = mapping[edu_num][0]
            count = len(df[df['education-num'] == edu_num])
            print(f"  {edu_num:6.1f} -> {edu_cat:20s} (n={count})")
    else:
        print("\nMapping (may have multiple categories per number):")
        print("-" * 60)
        for edu_num in sorted(mapping.index):
            edu_cats = mapping[edu_num]
            count = len(df[df['education-num'] == edu_num])
            if len(edu_cats) == 1:
                print(f"  {edu_num:6.1f} -> {edu_cats[0]:20s} (n={count})")
            else:
                print(f"  {edu_num:6.1f} -> {edu_cats} (n={count})")
                print(f"         Warning: Multiple categories for this number!")
    
    # Also print reverse mapping (education category to education-num)
    print("\n" + "-" * 60)
    print("Education Category to Education Number Mapping")
    print("-" * 60)
    reverse_mapping = df.groupby('education')['education-num'].unique()
    for edu_cat in sorted(reverse_mapping.index):
        edu_nums = reverse_mapping[edu_cat]
        count = len(df[df['education'] == edu_cat])
        if len(edu_nums) == 1:
            print(f"  {edu_cat:20s} -> {edu_nums[0]:6.1f} (n={count})")
        else:
            print(f"  {edu_cat:20s} -> {edu_nums} (n={count})")
            print(f"                     Warning: Multiple numbers for this category!")
    
    return mapping


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
    print("EXPLORATORY DATA ANALYSIS — Adult Income")
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
    print_education_mapping(df)

    print("\n" + "-" * 40)
    numeric_summary(df, numeric_cols)

    print("\n" + "-" * 40)
    categorical_summary(df, categorical_cols, top=None)

    print("\n" + "-" * 40)
    categorical_percentage_by_class(df, categorical_cols)

    print("\n" + "-" * 40)
    print("Plots (saving to outputs/):")
    plot_class_balance(df, save=save_figures)
    plot_numeric_by_class(df, numeric_cols=numeric_cols, save=save_figures)
    plot_correlation_with_target(df, numeric_cols=numeric_cols, save=save_figures)
    plot_numeric_correlation_matrix(df, numeric_cols=numeric_cols, save=save_figures)
    plot_categorical_by_class(df, categorical_cols=categorical_cols, save=save_figures)

    print("\nEDA complete.")
