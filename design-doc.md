# Fraud Detection ML Platform — Design Document

## Problem Statement

Credit card fraud costs the industry ~$30 billion/year. Detecting it requires sub-100ms decisions on high-velocity transaction streams while maintaining low false-positive rates (every false positive is a blocked legitimate transaction and a frustrated customer).

**Constraints we designed for:**
- p99 latency < 10 ms (card-present flow SLA)
- Fraud rate ~1.5% (severe class imbalance)
- Model must be hot-swappable (can't take the endpoint down to redeploy)
- Drift must be detectable before it causes revenue loss

---

## Architecture Overview

```
[Transaction Input]
       │
       ▼
  FastAPI Endpoint ──────────────── Model Registry (disk + DB)
  /v1/predict                           │
  /v2/predict (+ risk factors)          │ loads on first request
  /v2/batch                             │ hot-swap via PATCH /models/{v}/activate
       │
       ▼
  PredictionLog (SQLite)
       │
       ▼
  Monitoring Worker
  PSI drift detection
  Rolling accuracy
  Alert thresholds
       │
       ▼
  Frontend Dashboard (served by same FastAPI process)
```

---

## Data Layer

**Storage: SQLite** — chosen for zero-ops local deployment. In production this would be Postgres with connection pooling (PgBouncer), a separate read replica for monitoring queries, and partitioning on `created_at` for the prediction log (which grows at ~1M rows/day at scale).

**Tables:**
| Table | Purpose |
|---|---|
| `transactions` | Training data. Frozen after initial ingest. Indexed on `split` and `is_fraud`. |
| `model_versions` | Registry of all trained artifacts. Only one row has `is_active=True`. |
| `prediction_logs` | Every inference. `actual_fraud` is nullable — filled in via chargeback feedback loop. |
| `performance_metrics` | Monitoring snapshots. Append-only; used for the history chart. |

**Why not a feature store?** At this scale, raw feature columns in `prediction_logs` are sufficient. A real feature store (Feast, Tecton) makes sense when feature computation needs to be shared across training and serving to prevent train/serve skew — not needed when features are computed from raw request fields.

---

## Feature Engineering

All features are computed at request time from the raw transaction fields:

| Feature | Rationale |
|---|---|
| `amount_log` | Raw amount is log-normal; log-transform makes it more Gaussian and helps tree splits |
| `velocity_ratio` | 1h/24h velocity ratio catches burst patterns missed by either count alone |
| `night_flag` | 00:00–05:59 and 22:00–23:59 — fraud peaks in low-oversight hours |
| `high_distance_flag` | Binary threshold at 100 km — large distance is a strong fraud signal |

**Train/serve symmetry:** The exact same `engineer_features()` function is used in both `training/train.py` and `api/registry.py`. Shared via import, not copy-paste — this prevents the most common source of production ML bugs.

---

## Model Versioning

**Strategy: path-based versioning + DB registry**

Two independent mechanisms:
1. **URL versioning** (`/v1/predict`, `/v2/predict`) — controls the *response schema*. v1 returns a minimal payload; v2 adds risk band, top risk factors, and feature importances. Clients can pin to a version and never receive breaking schema changes.
2. **Model registry** (`model_versions` table) — controls *which artifact* handles predictions. The active model is promoted via `PATCH /models/{version}/activate`. Both `/v1` and `/v2` endpoints use whichever model is active in the registry.

**Why separate these two concerns?** You might want to:
- Serve v2 schema with model v1 (gradual schema rollout)
- Serve v1 schema with model v2 (upgrade the model without forcing client updates)
- A/B test model v2 on v1 schema traffic

**Hot-swap mechanism:** `ModelRegistry` is a singleton in-process cache. When `promote()` is called, it updates the DB flag and pre-loads the new artifact. The next request picks up the new model with zero downtime.

---

## Model Selection

| | v1: RandomForest | v2: GradientBoosting |
|---|---|---|
| **Strength** | Parallelizable training, easy to interpret | Higher AUC, better on imbalanced classes |
| **Weakness** | Lower AUC, larger artifact | Slower to train, serial boosting |
| **Inference** | ~1 ms p50 | ~2 ms p50 |
| **Use case** | Explainability-first (regulatory) | Performance-first (pure fraud reduction) |

Both use `class_weight="balanced"` / `subsample` tuning to handle the 1.5% fraud rate without manual over/undersampling (which leaks in validation if done carelessly).

---

## Monitoring & Drift Detection

**Population Stability Index (PSI)** measures how much the distribution of model output probabilities has shifted relative to a baseline window:

```
PSI = Σ (actual_bucket% − expected_bucket%) × ln(actual_bucket% / expected_bucket%)
```

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | No change | Monitor normally |
| 0.10–0.20 | Moderate drift | Investigate feature distributions |
| > 0.20 | Major drift | Retrain immediately |

PSI on the *output probability distribution* is a proxy for upstream feature drift — it catches both covariate shift (features changed) and concept drift (fraud patterns changed) in a single metric, without needing to compute PSI on every individual feature.

**Ground-truth latency problem:** Chargebacks arrive days to weeks after a transaction. The `actual_fraud` column in `prediction_logs` is nullable precisely for this reason — accuracy metrics are computed only on the labeled subset, while PSI runs on all predictions immediately.

---

## Latency Budget

Target SLA: p99 < 10 ms for a single prediction.

| Component | Typical latency |
|---|---|
| FastAPI routing + Pydantic validation | ~0.3 ms |
| Feature engineering | ~0.1 ms |
| RandomForest inference (200 trees, 14 features) | ~0.8 ms |
| GradientBoosting inference (300 trees) | ~1.5 ms |
| SQLite write (prediction log) | ~0.5 ms |
| Total p50 | ~2 ms |
| Total p99 (GC, disk flush) | ~6 ms |

SQLite write is async-friendly but we call it synchronously here — in production this would be an async fire-and-forget write to a message queue (Kafka/Kinesis) consumed by a separate logging service, removing the write latency from the critical path entirely.

---

## What's Missing for Production

| Gap | Production Solution |
|---|---|
| SQLite | Postgres + PgBouncer |
| In-process model cache | Triton or TorchServe with dedicated GPU instances |
| Synchronous prediction logging | Async Kafka producer |
| Manual drift monitoring | Scheduled Airflow DAG + PagerDuty alerts |
| No auth | JWT + API key middleware |
| Single instance | Kubernetes Deployment with HPA on p99 latency |
| No retraining trigger | MLflow webhook → training job on drift detection |
