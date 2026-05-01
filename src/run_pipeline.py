import argparse
import logging
import sys

from src.clean    import load_raw_data, clean_sales, clean_inventory, clean_catalog
from src.features import build_features
from src.train    import run_training, split_train_test, train_baseline, evaluate_baseline
from src.predict  import run_prediction
from src.summarize import generate_all_summaries
from src.evaluate import run_evaluation

# ── Logging setup 
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s — %(levelname)s — %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log"),
    ]
)
log = logging.getLogger(__name__)


def load_and_prepare():
    """
    Shared across both modes — always runs.
    Load raw data → clean → build features
    Everything stays in memory.
    """
    log.info("━━━━ STEP 1 — Loading raw data ━━━━")
    sales, inventory, catalog = load_raw_data()

    log.info("━━━━ STEP 2 — Cleaning ━━━━")
    sales     = clean_sales(sales, inventory)
    inventory = clean_inventory(inventory)
    catalog   = clean_catalog(catalog)

    log.info("━━━━ STEP 3 — Building features ━━━━")
    features_df = build_features(sales, inventory, catalog)

    return features_df, catalog


def run_train_mode(features_df, catalog):
    """
    --mode train
    Runs once a month.
    Trains model, evaluates fully, saves only if better than baseline.
    """
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("         TRAINING JOB STARTED        ")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    log.info("━━━━ STEP 4 — Training model ━━━━")
    model, metrics = run_training(features_df)

    log.info("━━━━ STEP 5 — Generating forecasts ━━━━")
    forecast_df = run_prediction(features_df)

    log.info("━━━━ STEP 6 — Generating LLM summaries ━━━━")
    summaries_df = generate_all_summaries(forecast_df, catalog, features_df)

    log.info("━━━━ STEP 7 — Running full evaluation ━━━━")
    train_df, test_df = split_train_test(features_df)
    baseline          = train_baseline(train_df)
    baseline_mape     = evaluate_baseline(test_df, baseline)

    run_evaluation(
        features_df  = features_df,
        test_df      = test_df,
        summaries_df = summaries_df,
        forecast_df  = forecast_df,
        baseline_mape= baseline_mape,
    )

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("       TRAINING JOB COMPLETE ✅      ")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def run_predict_mode(features_df, catalog):
    """
    --mode predict
    Runs every 2 weeks.
    Loads saved model, generates forecasts and summaries only.
    Fast and lightweight — no training, no evaluation.
    """
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("        PREDICTION JOB STARTED       ")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    log.info("━━━━ STEP 4 — Loading model + forecasting ━━━━")
    forecast_df = run_prediction(features_df)

    log.info("━━━━ STEP 5 — Generating LLM summaries ━━━━")
    summaries_df = generate_all_summaries(forecast_df, catalog, features_df)

    # Save final outputs
    log.info("━━━━ STEP 6 — Saving outputs ━━━━")
    forecast_df.to_csv("outputs/forecasts/forecasts.csv", index=False)
    summaries_df.to_csv("outputs/summaries/summaries.csv", index=False)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("      PREDICTION JOB COMPLETE ✅     ")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demand Signal Intelligence Pipeline")
    parser.add_argument(
        "--mode",
        choices = ["train", "predict"],
        required= True,
        help    = "train — monthly training job | predict — bi-weekly prediction job"
    )
    args = parser.parse_args()

    log.info(f"Pipeline starting in [{args.mode.upper()}] mode")

    # Always runs — shared between both modes
    features_df, catalog = load_and_prepare()

    # Mode specific
    if args.mode == "train":
        run_train_mode(features_df, catalog)
    elif args.mode == "predict":
        run_predict_mode(features_df, catalog)