import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)


def load_raw_data():
    log.info("Loading raw files...")
    sales     = pd.read_csv("data/raw_sales_transactions.csv")
    inventory = pd.read_csv("data/daily_inventory_snapshots.csv")
    catalog   = pd.read_csv("data/product_catalog_metadata.csv")
    return sales, inventory, catalog


def clean_sales(sales, inventory):

    # ── Standardise SKU IDs first
    sales['sku_id'] = sales['sku_id'].str.strip().str.lower()

    # ── Fix data types 
    sales['date']        = pd.to_datetime(sales['date'], errors='coerce')
    sales['units_sold']  = pd.to_numeric(sales['units_sold'], errors='coerce')
    sales['price_point'] = pd.to_numeric(sales['price_point'], errors='coerce')

    # ── Drop rows missing date or sku_id — empty legacy rows 
    before = len(sales)
    sales  = sales.dropna(subset=['date', 'sku_id'])
    log.info(f"Empty rows dropped : {before - len(sales):,}")

    # ── Drop negative units 
    neg = sales['units_sold'] < 0
    log.info(f"Negative units dropped : {neg.sum():,}")
    sales = sales[~neg]

    # ── Drop duplicates 
    before = len(sales)
    sales  = sales.drop_duplicates()
    log.info(f"Duplicates dropped : {before - len(sales):,}")

    log.info(f"Sales rows after cleaning : {len(sales):,}")
    return sales


def clean_catalog(catalog):
    # Standardise string columns — lowercase, strip whitespace
    catalog['sku_id'] = catalog['sku_id'].str.strip().str.lower()
    str_cols = catalog.select_dtypes(include="object").columns
    for col in str_cols:
        catalog[col] = catalog[col].str.strip().str.lower()
    return catalog


def clean_inventory(inventory):
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce")
    inventory["stock_on_hand"] = pd.to_numeric(inventory["stock_on_hand"], errors="coerce")
    inventory['sku_id'] = inventory['sku_id'].str.strip().str.lower()
    inventory = inventory.dropna(subset=["snapshot_date"])
    inventory = inventory.drop_duplicates()
    return inventory


if __name__ == "__main__":
    sales, inventory, catalog = load_raw_data()

    sales     = clean_sales(sales, inventory)
    inventory = clean_inventory(inventory)
    catalog   = clean_catalog(catalog)
    
    log.info("Cleaning complete ✅")