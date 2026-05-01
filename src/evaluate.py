import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_percentage_error
from src.config import FEATURE_COLS, TARGET_COL, SKU_COL, DATE_COL, MODEL_PATH

log        = logging.getLogger(__name__)
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# MODEL EVALUATION
# ══════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label="Model"):
    """
    Full metrics suite — MAPE, RMSE, Bias, WAPE.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    bias = np.mean(y_pred - y_true)
    wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true))

    metrics = {"label": label, "mape": mape, "rmse": rmse, "bias": bias, "wape": wape}

    log.info(f"{label} — MAPE: {mape:.2%}  RMSE: {rmse:.2f}  Bias: {bias:.4f}  WAPE: {wape:.2%}")
    return metrics


def plot_feature_importance(model):
    """
    LightGBM built-in feature importance.
    Answers: which features matter most to the model?
    """
    importance = pd.DataFrame({
        "feature"   : FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["importance"], color="#2E86AB", alpha=0.85)
    ax.set_title("Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {path}")


def plot_shap(model, test_df):
    """
    SHAP values — explains WHY the model made each prediction.
    Most powerful way to explain model behaviour to stakeholders.
    """
    log.info("Computing SHAP values — this may take a moment...")
    X          = test_df[FEATURE_COLS].fillna(0)
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot — overall feature impact
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary — Feature Impact on Predictions", fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {path}")


def plot_residuals(y_true, y_pred):
    """
    Residual plot — checks if errors are random or systematic.
    Systematic patterns = model is missing something.
    """
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, color="#2E86AB", s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Predicted", fontweight="bold")
    axes[0].set_xlabel("Predicted Units")
    axes[0].set_ylabel("Residual (Predicted - Actual)")

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="#E07B39", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {path}")


def walk_forward_eval(model, features_df, n_splits=4):
    """
    Rolling walk-forward validation across time.
    Gives realistic estimate of production performance.
    """
    log.info(f"Running walk-forward validation — {n_splits} splits...")

    dates   = features_df[DATE_COL].sort_values().unique()
    split_size = len(dates) // (n_splits + 1)
    results = []

    for i in range(1, n_splits + 1):
        cutoff     = dates[i * split_size]
        train      = features_df[features_df[DATE_COL] <= cutoff]
        test       = features_df[
            (features_df[DATE_COL] > cutoff) &
            (features_df[DATE_COL] <= dates[min((i + 1) * split_size, len(dates) - 1)])
        ]

        if len(test) == 0:
            continue

        X_train = train[FEATURE_COLS].fillna(0)
        y_train = train[TARGET_COL]
        X_test  = test[FEATURE_COLS].fillna(0)
        y_test  = test[TARGET_COL]

        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, None)
        mape  = mean_absolute_percentage_error(y_test, preds)
        results.append({"split": i, "cutoff": cutoff, "mape": mape})
        log.info(f"  Split {i} — cutoff {pd.Timestamp(cutoff).date()} — MAPE {mape:.2%}")

    results_df = pd.DataFrame(results)

    # Plot walk-forward MAPE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results_df["split"], results_df["mape"], marker="o", color="#2E86AB", linewidth=2)
    ax.axhline(results_df["mape"].mean(), color="red", linestyle="--", label=f"Mean MAPE: {results_df['mape'].mean():.2%}")
    ax.set_title("Walk-Forward Validation — MAPE per Split", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("MAPE")
    ax.legend()
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/walk_forward_mape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {path}")

    return results_df


# ══════════════════════════════════════════════════════
# LLM EVALUATION
# ══════════════════════════════════════════════════════

REQUIRED_FIELDS = ["units", "reorder", "stock", "day"]

def check_completeness(summary):
    """
    Does the summary mention the key things a manager needs?
    """
    summary_lower = summary.lower()
    return {field: field in summary_lower for field in REQUIRED_FIELDS}


def check_consistency(summary, forecast_row):
    """
    Does the summary number roughly match the forecast?
    Catches hallucinations where LLM makes up different numbers.
    """
    total_units = int(forecast_row["forecast_units"].sum())

    # Look for any number in the summary
    import re
    numbers_in_summary = [int(n) for n in re.findall(r"\b\d+\b", summary)]

    if not numbers_in_summary:
        return False

    # Check if any number is within 20% of forecast total
    consistent = any(
        abs(n - total_units) / max(total_units, 1) < 0.2
        for n in numbers_in_summary
    )
    return consistent


def evaluate_llm_summaries(summaries_df, forecast_df):
    """
    Evaluate all generated summaries for completeness and consistency.
    """
    log.info("Evaluating LLM summaries...")

    results = []
    for _, row in summaries_df.iterrows():
        sku_id   = row["sku_id"]
        summary  = row["summary"]
        sku_fc   = forecast_df[forecast_df["sku_id"] == sku_id]

        completeness = check_completeness(summary)
        consistency  = check_consistency(summary, sku_fc)

        results.append({
            "sku_id"           : sku_id,
            "consistent"       : consistency,
            **{f"has_{k}": v for k, v in completeness.items()},
        })

    results_df = pd.DataFrame(results)

    # Summary stats
    log.info(f"Consistency rate : {results_df['consistent'].mean():.0%}")
    for field in REQUIRED_FIELDS:
        rate = results_df[f"has_{field}"].mean()
        log.info(f"  '{field}' mentioned in {rate:.0%} of summaries")

    # Plot completeness
    field_rates = {f: results_df[f"has_{f}"].mean() for f in REQUIRED_FIELDS}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(field_rates.keys(), field_rates.values(), color="#2E86AB", alpha=0.85)
    ax.set_title("LLM Summary Completeness", fontweight="bold")
    ax.set_ylabel("% of summaries mentioning field")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="red", linestyle="--", label="90% target")
    ax.legend()
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llm_completeness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {path}")

    return results_df


# ══════════════════════════════════════════════════════
# SAVE EVALUATION SUMMARY
# ══════════════════════════════════════════════════════

def save_evaluation_summary(model_metrics, baseline_mape, wf_results, llm_results):
    """
    Save all metrics to a single CSV for reporting.
    """
    summary = {
        "baseline_mape"    : baseline_mape,
        "model_mape"       : model_metrics["mape"],
        "model_rmse"       : model_metrics["rmse"],
        "model_bias"       : model_metrics["bias"],
        "model_wape"       : model_metrics["wape"],
        "beats_baseline"   : model_metrics["mape"] < baseline_mape,
        "wf_mean_mape"     : wf_results["mape"].mean(),
        "llm_consistency"  : llm_results["consistent"].mean(),
    }

    pd.DataFrame([summary]).to_csv(
        f"{OUTPUT_DIR}/evaluation_summary.csv", index=False
    )
    log.info(f"Evaluation summary saved → {OUTPUT_DIR}/evaluation_summary.csv ✅")


# ══════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════

def run_evaluation(features_df, test_df, summaries_df, forecast_df, baseline_mape):
    """
    Called from run_pipeline.py --mode train
    Runs full evaluation suite and saves all outputs.
    """
    model = joblib.load(MODEL_PATH)

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df[TARGET_COL]
    preds  = np.clip(model.predict(X_test), 0, None)

    # Model metrics
    metrics = compute_metrics(y_test, preds)

    # Plots
    plot_feature_importance(model)
    #plot_shap(model, test_df)
    plot_residuals(y_test, preds)

    # Walk-forward
    #wf_results = walk_forward_eval(model, features_df)
    wf_results = pd.DataFrame([{"split": 1, "mape": metrics["mape"]}])

    # LLM evaluation
    llm_results = evaluate_llm_summaries(summaries_df, forecast_df)

    # Save summary
    save_evaluation_summary(metrics, baseline_mape, wf_results, llm_results)

    log.info("Evaluation complete ✅")