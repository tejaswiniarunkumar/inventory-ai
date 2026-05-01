# Inventory AI

AI-powered pipeline that forecasts SKU-level retail demand and generates automated store manager summaries using Claude AI.

---

## Folder Structure

```
demand-signal-intelligence/
├── pipeline/          ← all pipeline scripts
├── notebooks/         ← EDA and model comparison notebooks
├── data/raw/          ← place raw CSV files here
├── models/            ← saved trained model
├── outputs/           ← forecasts, summaries, evaluation plots
├── docs/              ← design document and architecture diagram
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API key — create a .env file at project root
ANTHROPIC_API_KEY=your_key_here
```

---

## How to Run

**Training** — run once a month:
```bash
python -m pipeline.run_pipeline --mode train
```

**Prediction** — run every 2 weeks:
```bash
python -m pipeline.run_pipeline --mode predict
```

**Docker:**
```bash
docker build -t demand-pipeline .
docker run -e ANTHROPIC_API_KEY=your_key demand-pipeline --mode predict
```
