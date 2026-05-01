import logging
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from src.config import FEATURE_COLS, TARGET_COL, DATE_COL, SKU_COL, MODEL_PATH

log = logging.getLogger(__name__)


# In split_train_test() change to 80/20 split
def split_train_test(df, test_size=0.2):
    dates  = sorted(df[DATE_COL].unique())
    cutoff = dates[int(len(dates) * (1 - test_size))]
    train  = df[df[DATE_COL] <= cutoff]
    test   = df[df[DATE_COL] >  cutoff]
    log.info(f"Train : {len(train):,} rows up to {pd.Timestamp(cutoff).date()}")
    log.info(f"Test  : {len(test):,} rows after  {pd.Timestamp(cutoff).date()}")
    return train, test


def train_baseline(train):
    """
    Naive baseline — mean units sold per SKU.
    Every model must beat this.
    """
    baseline = (
        train.groupby(SKU_COL)[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "baseline_prediction"})
    )
    return baseline


def evaluate_baseline(test, baseline):
    df   = test.merge(baseline, on=SKU_COL, how="left")
    df["baseline_prediction"] = df["baseline_prediction"].fillna(
        df[TARGET_COL].mean()
    )
    mape = mean_absolute_percentage_error(
        df[TARGET_COL], df["baseline_prediction"]
    )
    log.info(f"Baseline MAPE : {mape:.2%}")
    return mape


def train_model(train):
    X = train[FEATURE_COLS].fillna(0)
    y = train[TARGET_COL]

    model = LGBMRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        min_child_samples = 20,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(X, y)
    log.info("LightGBM trained ✅")
    return model


def evaluate_model(model, test):
    X     = test[FEATURE_COLS].fillna(0)
    y     = test[TARGET_COL]
    preds = np.clip(model.predict(X), 0, None)

    mape  = mean_absolute_percentage_error(y, preds)
    rmse  = np.sqrt(np.mean((preds - y) ** 2))
    bias  = np.mean(preds - y)
    wape  = np.sum(np.abs(preds - y)) / np.sum(np.abs(y))

    log.info(f"MAPE : {mape:.2%}")
    log.info(f"RMSE : {rmse:.4f}")
    log.info(f"Bias : {bias:.4f}")
    log.info(f"WAPE : {wape:.2%}")

    return {"mape": mape, "rmse": rmse, "bias": bias, "wape": wape}


def save_model(model, new_metrics, baseline_mape):
    """
    Only save new model if it beats current baseline.
    Protects production from a bad retraining run.
    """
    if new_metrics["mape"] < baseline_mape:
        joblib.dump(model, MODEL_PATH)
        log.info(f"New model saved → {MODEL_PATH} ✅")
    else:
        log.warning(
            f"New model MAPE ({new_metrics['mape']:.2%}) did not beat "
            f"baseline ({baseline_mape:.2%}) — keeping current model ⚠️"
        )


def run_training(features_df):
    """
    Master function called from run_pipeline.py --mode train
    """
    log.info("Starting training job...")

    train, test   = split_train_test(features_df)
    baseline      = train_baseline(train)
    baseline_mape = evaluate_baseline(test, baseline)
    model         = train_model(train)
    metrics       = evaluate_model(model, test)

    save_model(model, metrics, baseline_mape)

    log.info("Training job complete ✅")
    return model, metrics