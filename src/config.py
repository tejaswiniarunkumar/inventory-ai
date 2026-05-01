# pipeline/config.py

# ── Column names 
SKU_COL    = "sku_id"
DATE_COL   = "date"
TARGET_COL = "units_sold"

# ── Model 
MODEL_PATH   = "models/lgbm_forecast.pkl"
HORIZON_DAYS = 14

# ── Features 
FEATURE_COLS = [
    # Calendar
    "day_of_week", "week_of_year", "month", "is_weekend",

    # Lag features
    "units_sold_lag_7d", "units_sold_lag_14d", "units_sold_lag_21d",

    # Rolling averages
    "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_28d",
    "rolling_std_14d",

    # Inventory
    "stock_on_hand", "reorder_point", "stock_vs_reorder",
    "reorder_urgency", "days_of_stock_left", "low_stock_flag",

    # Catalog
    "category_code", "is_promotional", "promo_x_weekend",
    "description_length", "is_bundle",

    # Product age
    "sku_age_days", "is_new_product",

    # Price
    "price_point", "price_rolling_mean_7d", "price_changed",
]

# ── Data paths 
RAW_SALES_PATH     = "data/raw_sales_transactions.csv"
RAW_INVENTORY_PATH = "data/daily_inventory_snapshots.csv"
RAW_CATALOG_PATH   = "data/product_catalog_metadata.csv"

# ── Output paths 
FORECAST_OUTPUT_PATH = "outputs/forecasts/forecasts.csv"
SUMMARY_OUTPUT_PATH  = "outputs/summaries/summaries.csv"
EVAL_OUTPUT_DIR      = "outputs/evaluation"