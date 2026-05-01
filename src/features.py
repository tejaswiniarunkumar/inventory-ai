import pandas as pd
import numpy as np
import logging

from src.config import (
    FEATURE_COLS, TARGET_COL, SKU_COL, DATE_COL
)

log = logging.getLogger(__name__)


def build_features(sales, inventory, catalog):
    log.info("Building features...")

    df = sales.copy()

    # ── 1. Sort — critical for time series lag features 
    df = df.sort_values([SKU_COL, DATE_COL]).reset_index(drop=True)

    # ── 2. Calendar features 
    df["day_of_week"]  = df[DATE_COL].dt.dayofweek
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["month"]        = df[DATE_COL].dt.month
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)

    # ── 3. Lag features 
    for lag in [7, 14, 21]:
        df[f"units_sold_lag_{lag}d"] = (
            df.groupby(SKU_COL)[TARGET_COL].shift(lag)
        )

    # ── 4. Rolling averages 
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}d"] = (
            df.groupby(SKU_COL)[TARGET_COL]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # ── 5. Rolling std — demand volatility 
    df["rolling_std_14d"] = (
        df.groupby(SKU_COL)[TARGET_COL]
        .transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
    )

    # ── 6. Inventory features 
    # Rename snapshot_date to date for merging
    inventory = inventory.rename(columns={"snapshot_date": DATE_COL})
    inventory[DATE_COL] = pd.to_datetime(inventory[DATE_COL])

    inventory_daily = (
        inventory.groupby([SKU_COL, DATE_COL])[["stock_on_hand", "reorder_point"]]
        .mean()
        .reset_index()
    )

    df = df.merge(inventory_daily, on=[SKU_COL, DATE_COL], how="left")

    # Stock vs reorder point — negative means already needs reorder
    df["stock_vs_reorder"] = df["stock_on_hand"] - df["reorder_point"]

    # How far below reorder point as a percentage
    df["reorder_urgency"] = (
        df["reorder_point"] - df["stock_on_hand"]
    ).clip(lower=0) / df["reorder_point"].replace(0, np.nan)

    # Days of stock remaining based on recent average demand
    df["days_of_stock_left"] = (
        df["stock_on_hand"] / df["rolling_mean_7d"].replace(0, np.nan)
    ).clip(upper=365)  # cap at 1 year to avoid outliers

    # Low stock flag — stock below 2 weeks of average demand
    df["low_stock_flag"] = (
        df["stock_on_hand"] < df["rolling_mean_14d"] * 2
    ).astype(int)

    # ── 7. Catalog features 
    df = df.merge(
        catalog[[SKU_COL, "category", "is_promotional", "product_description"]],
        on=SKU_COL, how="left"
    )

    # Category as numeric code
    df["category_code"] = df["category"].astype("category").cat.codes

    # Promotional flag — direct feature
    df["is_promotional"] = df["is_promotional"].fillna(0).astype(int)

    # Promo on weekend — interaction feature
    # Promotions during weekends tend to cause bigger demand spikes
    df["promo_x_weekend"] = df["is_promotional"] * df["is_weekend"]

    # Description length — proxy for product complexity
    df["description_length"] = (
        df["product_description"].fillna("").str.split().str.len()
    )

    # Is it a bundle or multi-unit product?
    df["is_bundle"] = (
        df["product_description"].fillna("").str.lower()
        .str.contains("bundle|pack|set|kit", regex=True)
        .astype(int)
    )

    # ── 8. Product age features 
    # Approximate — based on first appearance in sales data
    sku_first_seen = (
        df.groupby(SKU_COL)[DATE_COL].min()
        .reset_index()
        .rename(columns={DATE_COL: "sku_first_seen"})
    )
    df = df.merge(sku_first_seen, on=SKU_COL, how="left")

    df["sku_age_days"] = (
        df[DATE_COL] - df["sku_first_seen"]
    ).dt.days

    # New product flag — less than 30 days in sales data
    # New products have limited history so model treats them differently
    df["is_new_product"] = (df["sku_age_days"] < 30).astype(int)

    # Drop helper column
    df = df.drop(columns=["sku_first_seen"])

    # ── 9. Price features 
    df["price_point"] = pd.to_numeric(df["price_point"], errors="coerce")
    df["price_rolling_mean_7d"] = (
        df.groupby(SKU_COL)["price_point"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )

    # Price change flag — did price change recently?
    df["price_changed"] = (
        df["price_point"] != df["price_rolling_mean_7d"]
    ).astype(int)

    # ── 10. Drop rows with nulls in lag columns 
    lag_cols = [c for c in df.columns if "lag" in c]
    before   = len(df)
    df       = df.dropna(subset=lag_cols)
    log.info(f"Rows dropped due to lag nulls : {before - len(df):,}")

    log.info(f"Features built — {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df