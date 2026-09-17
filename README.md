# Fraud Detection ML Platform

End-to-end ML platform for real-time credit card fraud detection. Covers data ingestion, model training with experiment tracking, a versioned serving API, and a live monitoring dashboard.

## Quick Start

```bash
cd ml-platform

# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest synthetic fraud data into SQLite
python data/ingest.py

# 3. Train models (v1 RandomForest, v2 GradientBoosting) — logs to MLflow
python training/train.py

# 4. Start the API + dashboard
uvicorn api.main:app --reload --port 8000
```

Then open:
- **Dashboard**: http://localhost:8000
- **API docs**: http://localhost:8000/docs
- **MLflow UI**: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`

## Architecture

```
data/ingest.py          → SQLite (transactions table)
training/train.py       → scikit-learn models + MLflow experiment tracking
api/main.py             → FastAPI (/v1/predict, /v2/predict, /v2/batch)
api/registry.py         → in-process model cache, hot-swap support
monitoring/monitor.py   → PSI drift detection, rolling accuracy
frontend/               → dashboard (served by FastAPI at /)
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health + active model |
| `GET` | `/models` | List all registered model versions |
| `PATCH` | `/models/{version}/activate` | Promote a model (hot-swap, no restart needed) |
| `POST` | `/v1/predict` | Simple prediction (fraud flag + probability) |
| `POST` | `/v2/predict` | Enriched prediction (risk band + top risk factors) |
| `POST` | `/v2/batch` | Batch predictions (up to 500 rows) |
| `GET` | `/monitoring/snapshot` | Current drift + performance snapshot |
| `GET` | `/monitoring/history` | Historical metrics for charting |

### Example: v2 prediction

```bash
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 850.0,
    "hour_of_day": 2,
    "day_of_week": 6,
    "merchant_category": 8,
    "distance_from_home": 450.0,
    "is_weekend": true,
    "velocity_1h": 7,
    "velocity_24h": 18,
    "amount_z_score": 3.2
  }'
```

Response:
```json
{
  "transaction_id": 42,
  "model_version": "v1",
  "predicted_fraud": true,
  "fraud_probability": 0.9312,
  "risk_band": "critical",
  "top_risk_factors": [
    {"feature": "amount_z_score", "importance": 0.183},
    {"feature": "velocity_ratio", "importance": 0.161},
    {"feature": "distance_from_home", "importance": 0.147}
  ],
  "latency_ms": 1.24
}
```

## Model Performance

| Model | ROC-AUC | F1 | p50 Latency |
|---|---|---|---|
| v1 (RandomForest 200) | ~0.985 | ~0.72 | ~0.9 ms |
| v2 (GradientBoosting 300) | ~0.992 | ~0.78 | ~1.8 ms |

## Key Design Decisions

See [design-doc.md](design-doc.md) for full architectural rationale. Short version:

- **URL versioning** controls response schema (clients never see breaking changes)
- **Registry versioning** controls which model artifact runs (promotes without restart)
- **PSI > 0.2** triggers a drift alert — catches both covariate shift and concept drift
- **`actual_fraud` nullable** in prediction logs — chargeback feedback arrives days later
- **Same `engineer_features()` for training and serving** — prevents train/serve skew

## Project Structure

```
ml-platform/
├── config.py                  # Central config (DB URLs, thresholds, seeds)
├── requirements.txt
├── design-doc.md
├── db/
│   ├── database.py            # SQLAlchemy engine + session
│   └── models.py              # Transaction, ModelVersion, PredictionLog, PerformanceMetric
├── data/
│   └── ingest.py              # Synthetic fraud data generation + DB ingest
├── training/
│   └── train.py               # Train v1+v2, MLflow logging, latency benchmarks
├── api/
│   ├── main.py                # FastAPI app, all endpoints
│   ├── schemas.py             # Pydantic request/response models
│   └── registry.py            # Model cache + hot-swap logic
├── monitoring/
│   └── monitor.py             # PSI computation, snapshot persistence
├── frontend/
│   ├── index.html             # Dashboard
│   └── static/
│       ├── app.js             # Chart.js charts, API calls
│       └── style.css          # Dark theme
└── models/                    # Trained .joblib artifacts (created by training)
```
