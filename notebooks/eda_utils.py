"""
eda_utils.py
============
Simple reusable EDA functions for quick data checks and visualizations.

Functions:
    - load_data()           : Load CSV with smart sampling for large files
    - check_structure()     : Shape, column names, data types, sample rows
    - check_quality()       : Missing values and duplicate rows
    - check_statistics()    : Numeric summary and category frequencies
    - plot_distributions()  : Key charts and visualizations
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")

# Output folder for saved plots
OUTPUT_DIR = "outputs/eda_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# LOAD DATA — smart sampling for large files
# ──────────────────────────────────────────────────────────────
def load_data(filepath, sample_size=50_000, stratify_col=None, date_col=None):
    """
    Load a CSV file and return a sample safe for EDA.
    Always reads the full file first to get row count,
    then returns a stratified or random sample.
    """
    print(f"\n📂 Loading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    total_rows = len(df)
    print(f"   Total rows : {total_rows:,}")

    if total_rows > sample_size:
        if stratify_col and stratify_col in df.columns:
            n_groups  = df[stratify_col].nunique()
            per_group = max(1, sample_size // n_groups)
            df = (
                df.groupby(stratify_col, group_keys=False)
                .apply(lambda x: x.sample(min(len(x), per_group), random_state=42))
                .reset_index(drop=True)
            )
            print(f"   Sampled    : {len(df):,} rows (stratified by '{stratify_col}')")
        else:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"   Sampled    : {len(df):,} rows (random)")
    else:
        print(f"   Small file — using all {total_rows:,} rows")

    return df


# ──────────────────────────────────────────────────────────────
# MODULE 1 — STRUCTURE
# ──────────────────────────────────────────────────────────────
def check_structure(df, name="Dataset"):
    """
    What do I have?
    Shows shape, column names, data types, and sample rows.
    """
    display(Markdown(f"---\n## 🗂️ Structure — {name}"))

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}\n")

    dtype_df = pd.DataFrame({
        "Column"   : df.columns,
        "Type"     : df.dtypes.values,
        "Non-Null" : df.notnull().sum().values,
        "Null"     : df.isnull().sum().values,
        "Example"  : [df[c].dropna().iloc[0] if df[c].notna().any() else "—" for c in df.columns],
    })
    display(dtype_df)

    display(Markdown("**First 3 rows:**"))
    display(df.head(3))


# ──────────────────────────────────────────────────────────────
# MODULE 2 — QUALITY
# ──────────────────────────────────────────────────────────────
def check_quality(df, name="Dataset"):
    """
    What's broken?
    Shows missing values per column and duplicate row count.
    """
    display(Markdown(f"---\n## 🔍 Quality — {name}"))

    # Missing values
    display(Markdown("**Missing Values:**"))
    missing = pd.DataFrame({
        "Column"    : df.columns,
        "Missing"   : df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values,
    }).sort_values("Missing %", ascending=False)

    has_missing = missing[missing["Missing"] > 0]
    if has_missing.empty:
        print("  ✅ No missing values found")
    else:
        print(f"  ⚠️  {len(has_missing)} columns have missing values")
        display(has_missing)

    # Duplicates
    display(Markdown("**Duplicate Rows:**"))
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        print("  ✅ No duplicate rows found")
    else:
        print(f"  ⚠️  {n_dupes:,} duplicate rows ({n_dupes/len(df)*100:.2f}%)")


# ──────────────────────────────────────────────────────────────
# MODULE 3 — STATISTICS
# ──────────────────────────────────────────────────────────────
def check_statistics(df, name="Dataset"):
    """
    What does it look like numerically?
    Shows numeric summary and top values for categorical columns.
    """
    display(Markdown(f"---\n## 📊 Statistics — {name}"))

    # Numeric summary
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        display(Markdown("**Numeric Summary:**"))
        display(num_df.describe().T.round(2))

    # Categorical frequencies
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        display(Markdown("**Categorical Columns — Top Values:**"))
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"\n  '{col}' — {n_unique:,} unique values")
            if n_unique <= 30:
                display(df[col].value_counts().head(5).to_frame())


# ──────────────────────────────────────────────────────────────
# MODULE 4 — PLOTS
# ──────────────────────────────────────────────────────────────
def plot_distributions(df, name="Dataset", date_col=None, target_col=None):
    """
    What does the data look like visually?
    Plots histograms for numeric columns, bar charts for categoricals,
    and a time trend if date + target columns are provided.
    """
    display(Markdown(f"---\n## 📈 Plots — {name}"))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Numeric histograms ─────────────────────────────────────────────────
    if num_cols:
        n = min(len(num_cols), 4)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        axes = [axes] if n == 1 else axes
        for ax, col in zip(axes, num_cols[:n]):
            ax.hist(df[col].dropna(), bins=30, color="#2E86AB", alpha=0.8, edgecolor="white")
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        fig.suptitle(f"Distributions — {name}", fontweight="bold", y=1.02)
        plt.tight_layout()
        path = f"{OUTPUT_DIR}/{name.replace(' ', '_')}_histograms.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.show()
        print(f"  💾 Saved → {path}")

    # ── Categorical bar charts ─────────────────────────────────────────────
    low_cardinality = [c for c in cat_cols if df[c].nunique() <= 20]
    if low_cardinality:
        n = min(len(low_cardinality), 3)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        axes = [axes] if n == 1 else axes
        for ax, col in zip(axes, low_cardinality[:n]):
            counts = df[col].value_counts().head(10)
            ax.barh(counts.index.astype(str), counts.values, color="#E07B39", alpha=0.85)
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel("Count")
            ax.invert_yaxis()
        fig.suptitle(f"Category Frequencies — {name}", fontweight="bold", y=1.02)
        plt.tight_layout()
        path = f"{OUTPUT_DIR}/{name.replace(' ', '_')}_categories.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.show()
        print(f"  💾 Saved → {path}")

    # ── Time trend ─────────────────────────────────────────────────────────
    if date_col and target_col and date_col in df.columns and target_col in df.columns:
        trend = (
            df.groupby(pd.to_datetime(df[date_col]).dt.to_period("W"))[target_col]
            .sum()
            .reset_index()
        )
        trend[date_col] = trend[date_col].astype(str)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trend[date_col], trend[target_col], color="#2E86AB", linewidth=2)
        ax.fill_between(range(len(trend)), trend[target_col], alpha=0.15, color="#2E86AB")
        step = max(1, len(trend) // 8)
        ax.set_xticks(range(0, len(trend), step))
        ax.set_xticklabels(trend[date_col].iloc[::step], rotation=45, ha="right")
        ax.set_title(f"Weekly {target_col} Over Time — {name}", fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylabel(target_col)
        plt.tight_layout()
        path = f"{OUTPUT_DIR}/{name.replace(' ', '_')}_time_trend.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.show()
        print(f"  💾 Saved → {path}")
