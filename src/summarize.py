"""
summarise.py
============
Generates scenario-aware forecast summaries for store managers using Claude Haiku.
Different prompts are used depending on the SKU's situation:
    - Low stock      → urgent reorder advice
    - Demand spike   → prepare for high volume
    - Low demand     → consider promotion or markdown
    - New product    → cautious forecast warning
    - Normal         → standard summary
"""

import logging
import pandas as pd
import anthropic
from dotenv import load_dotenv

from src.config import SKU_COL

# Load API key from .env file at project root
load_dotenv()

log = logging.getLogger(__name__)

# Anthropic client — automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Default model — Haiku is fast and cost efficient for high volume runs
LLM_MODEL = "claude-haiku-4-5-20251001"


# ══════════════════════════════════════════════════════════════════
# SCENARIO DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_scenario(sku_forecast, sku_features):
    """
    Decide which scenario applies based on forecast + recent features.
    Returns scenario name to pick the right prompt template.
    """
    avg_daily = sku_forecast["forecast_units"].mean()

    if sku_features.empty:
        return "normal"

    latest          = sku_features.iloc[-1]
    stock_on_hand   = latest.get("stock_on_hand", 0)
    days_left       = latest.get("days_of_stock_left", 999)
    is_new_product  = latest.get("is_new_product", 0)
    rolling_mean_7d = latest.get("rolling_mean_7d", 0)

    # Decision tree — order matters, most urgent scenarios first
    if days_left < 7 or stock_on_hand < avg_daily * 7:
        return "low_stock"

    if is_new_product == 1:
        return "new_product"

    if avg_daily > rolling_mean_7d * 1.5 and rolling_mean_7d > 0:
        return "demand_spike"

    if avg_daily < rolling_mean_7d * 0.5 and rolling_mean_7d > 0:
        return "low_demand"

    return "normal"


# ══════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════

PROMPT_INTRO = (
    "You are an assistant helping retail store managers understand stock needs. "
    "Write a short, clear summary (3 to 4 sentences max). "
    "Use plain English — no technical jargon. "
    "Be direct and actionable."
)


def _scenario_instruction(scenario):
    """Return the scenario specific guidance text."""
    instructions = {
        "low_stock": (
            "URGENT: Stock is critically low. The summary must clearly tell the manager "
            "to reorder immediately, mention the risk of stockout, and suggest priority action."
        ),
        "demand_spike": (
            "Demand is forecast to be unusually high — significantly above recent averages. "
            "Tell the manager to prepare for higher than usual volumes and consider extra "
            "inventory ahead of the busiest day."
        ),
        "low_demand": (
            "Demand is forecast to drop below recent averages. "
            "Suggest the manager consider a promotion, markdown, or reduced reorder quantity "
            "to avoid overstock."
        ),
        "new_product": (
            "This is a new product with limited sales history. "
            "Tell the manager the forecast is an early estimate, encourage them to monitor "
            "actual sales closely, and suggest a conservative reorder approach."
        ),
        "normal": (
            "This is a standard forecast — no unusual patterns detected. "
            "Provide a brief overview of expected sales and highlight the busiest day."
        ),
    }
    return instructions.get(scenario, instructions["normal"])


def build_prompt(sku_id, forecast_df, catalog_df, features_df):
    """
    Build a scenario-aware prompt for one SKU.
    Returns (prompt, scenario) tuple.
    """
    sku_forecast = forecast_df[forecast_df[SKU_COL] == sku_id].copy()
    sku_catalog  = catalog_df[catalog_df[SKU_COL] == sku_id]
    sku_features = features_df[features_df[SKU_COL] == sku_id]


    # Detect which scenario applies
    scenario = detect_scenario(sku_forecast, sku_features)

    # Product context
    if not sku_catalog.empty:
        category       = sku_catalog["category"].values[0]
        description    = sku_catalog.get("product_description", pd.Series([""])).values[0]
        is_promotional = sku_catalog.get("is_promotional", pd.Series([0])).values[0]
    else:
        category, description, is_promotional = "Unknown", "", 0

    # Forecast numbers
    total_units = int(sku_forecast["forecast_units"].fillna(0).sum())
    avg_daily   = round(float(sku_forecast["forecast_units"].fillna(0).mean()), 1)
    peak_row    = sku_forecast.loc[sku_forecast["forecast_units"].fillna(0).idxmax()]
    peak_day    = peak_row["forecast_date"].strftime("%A %d %B")
    peak_units  = 0 if pd.isna(peak_row["forecast_units"]) else int(peak_row["forecast_units"])

    # Stock context
    if not sku_features.empty:
        stock_on_hand = int(sku_features["stock_on_hand"].dropna().iloc[-1] if not sku_features["stock_on_hand"].dropna().empty else 0)
        days_left     = int(sku_features["days_of_stock_left"].dropna().iloc[-1] if not sku_features["days_of_stock_left"].dropna().empty else 0)
    else:
        stock_on_hand, days_left = 0, 0

    prompt = f"""
{PROMPT_INTRO}

SCENARIO: {scenario.upper().replace("_", " ")}
SCENARIO GUIDANCE:
{_scenario_instruction(scenario)}

Product Information:
- SKU ID         : {sku_id}
- Category       : {category}
- Description    : {description}
- On Promotion   : {"Yes" if is_promotional else "No"}

Current Stock Position:
- Stock on hand        : {stock_on_hand} units
- Estimated days left  : {days_left} days

14-Day Forecast:
- Total units expected : {total_units}
- Average per day      : {avg_daily}
- Busiest day          : {peak_day} ({peak_units} units)

Write the summary now:
""".strip()

    return prompt, scenario


# ══════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_summary(sku_id, forecast_df, catalog_df, features_df):
    """
    Call Claude Haiku for a single SKU summary.
    Returns (summary_text, scenario) tuple.
    """
    prompt, scenario = build_prompt(sku_id, forecast_df, catalog_df, features_df)

    response = client.messages.create(
        model      = LLM_MODEL,
        max_tokens = 300,
        messages   = [{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip(), scenario


def generate_all_summaries(forecast_df, catalog_df, features_df):
    """
    Generate one summary per SKU. Failures logged but never stop the pipeline.
    Master function called from run_pipeline.py
    """
    skus    = forecast_df[SKU_COL].unique()
    results = []

    log.info(f"Generating summaries for {len(skus):,} SKUs...")

    # Log progress every 10% of total SKUs
    log_interval = max(1, len(skus) // 10)

    for i, sku_id in enumerate(skus):
        try:
            summary, scenario = generate_summary(sku_id, forecast_df, catalog_df, features_df)
            results.append({
                "sku_id"  : sku_id,
                "scenario": scenario,
                "summary" : summary,
            })
        except Exception as e:
            log.warning(f"  Summary failed for SKU {sku_id} : {e}")
            results.append({
                "sku_id"  : sku_id,
                "scenario": "error",
                "summary" : "Summary unavailable for this SKU.",
            })

        if (i + 1) % log_interval == 0:
            log.info(f"  Progress : {i + 1}/{len(skus)} summaries done ({(i+1)/len(skus)*100:.0f}%)")

    summaries_df = pd.DataFrame(results)

    # Log scenario distribution
    scenario_counts = summaries_df["scenario"].value_counts()
    log.info("Scenario distribution:")
    for scenario, count in scenario_counts.items():
        log.info(f"  {scenario:15s} : {count:,}")

    log.info(f"Summaries complete — {len(summaries_df):,} generated ✅")
    return summaries_df