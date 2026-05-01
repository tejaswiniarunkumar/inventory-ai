import logging
import joblib
import numpy as np
import pandas as pd
from src.config import FEATURE_COLS, SKU_COL, DATE_COL, MODEL_PATH, HORIZON_DAYS

log = logging.getLogger(__name__)


def load_model():
    """
    Load saved model from disk.
    Fails loudly if model does not exist — never silently.
    """
    try:
        model = joblib.load(MODEL_PATH)
        log.info(f"Model loaded from {MODEL_PATH} ✅")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run pipeline in --mode train first."
        )


def generate_forecast(model, features_df):
    """
    Generate HORIZON_DAYS forward forecast for every SKU.
    Uses last known feature values per SKU as the base.
    """
    log.info(f"Generating {HORIZON_DAYS}-day forecast...")

    # Last known row per SKU = starting point for future predictions
    latest = (
        features_df.sort_values(DATE_COL)
        .groupby(SKU_COL)
        .last()
        .reset_index()
    )

    forecasts = []
    for day in range(1, HORIZON_DAYS + 1):
        temp                  = latest.copy()
        temp["forecast_date"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=day)
        temp["day_of_week"]   = temp["forecast_date"].dt.dayofweek
        temp["week_of_year"]  = temp["forecast_date"].dt.isocalendar().week.astype(int)
        temp["month"]         = temp["forecast_date"].dt.month
        temp["is_weekend"]    = temp["day_of_week"].isin([5, 6]).astype(int)

        X                          = temp[FEATURE_COLS].fillna(0)
        temp["forecast_units"]     = np.clip(model.predict(X), 0, None).round()
        forecasts.append(temp[[SKU_COL, "forecast_date", "forecast_units"]])

    forecast_df = pd.concat(forecasts, ignore_index=True)
    log.info(f"Forecast complete — {len(forecast_df):,} rows ✅")
    return forecast_df


def run_prediction(features_df):
    """
    Master function called from run_pipeline.py --mode predict
    """
    log.info("Starting prediction job...")
    model       = load_model()
    forecast_df = generate_forecast(model, features_df)
    log.info("Prediction job complete ✅")
    return forecast_df