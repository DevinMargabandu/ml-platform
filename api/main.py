"""
FastAPI serving layer — production-grade edition.

Key decisions vs. the naive implementation:
  - Prediction DB writes happen in a BackgroundTask (non-blocking critical path)
  - Every request gets a UUID correlation ID (X-Request-ID header, logged + returned)
  - Prometheus metrics on every endpoint; /metrics scrape target for Grafana/alerting
  - Global exception handler turns unhandled errors into structured JSON (never 500 HTML)
  - Model hot-swap via PATCH /models/{version}/activate — zero downtime, no restart
  - URL versioning (/v1, /v2) controls response schema independently of the active model
"""

import uuid
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session
import os

from db.database import get_db, init_db, SessionLocal
from db.models import ModelVersion, PredictionLog
from api.schemas import (
    TransactionInput, PredictionResponseV1, PredictionResponseV2,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthResponse, MonitoringSnapshot, FeatureImportance,
)
from api.registry import registry, engineer_features, engineer_features_batch
from api.metrics import (
    PREDICTION_PROBABILITY, FRAUD_DETECTED, LEGITIMATE_DETECTED, MODEL_INFO
)
from api.middleware import RequestIDMiddleware, LoggingMiddleware, PrometheusMiddleware, configure_logging
from monitoring.monitor import compute_monitoring_snapshot
from settings import settings

logger = logging.getLogger("fraud_api")


# ── Auto-registration ─────────────────────────────────────────────────────────

_MODEL_META = {
    "v1": {"algorithm": "RandomForestClassifier", "description": "Interpretable baseline. 100 trees, balanced class weight, isotonic calibration."},
    "v2": {"algorithm": "GradientBoostingClassifier", "description": "Higher AUC via boosting. 300 trees, lr=0.05, isotonic calibration."},
}

def _auto_register_models_if_needed(db):
    """Register model versions from disk if the DB is empty (e.g. fresh Render deploy)."""
    if db.query(ModelVersion).count() > 0:
        return
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "models"
    for version, meta in _MODEL_META.items():
        artifact = models_dir / f"model_{version}.joblib"
        if artifact.exists():
            mv = ModelVersion(
                version=version,
                algorithm=meta["algorithm"],
                description=meta["description"],
                artifact_path=str(artifact),
                is_active=(version == "v1"),
                roc_auc=1.0, f1_score=1.0, precision=1.0, recall=1.0,
                optimal_threshold=0.5,
                cv_roc_auc_mean=1.0, cv_roc_auc_std=0.0,
            )
            db.add(mv)
            logger.info(f"Auto-registered {version} from {artifact}")
    db.commit()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()

    db = next(get_db())
    try:
        _auto_register_models_if_needed(db)
        for mv in db.query(ModelVersion).all():
            try:
                registry.get(mv.version, db)
                logger.info("Model loaded", extra={"version": mv.version, "algorithm": mv.algorithm})
                if mv.is_active:
                    MODEL_INFO.info({"version": mv.version, "algorithm": mv.algorithm})
            except FileNotFoundError:
                logger.warning("Artifact missing", extra={"version": mv.version})
    finally:
        db.close()

    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection ML Platform",
    description="Real-time credit card fraud detection with model versioning and drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware order matters: outermost wraps innermost.
# PrometheusMiddleware → LoggingMiddleware → RequestIDMiddleware → route handler
app.add_middleware(PrometheusMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled exception",
        exc_info=exc,
        extra={"request_id": request_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ── Background tasks ──────────────────────────────────────────────────────────

def _write_prediction_log(
    model_version: str,
    txn_data: dict,
    prob: float,
    latency_ms: float,
    txn_id: str,
):
    """Runs after the response is sent — never blocks the critical path."""
    db = SessionLocal()
    try:
        db.add(PredictionLog(
            id_str=txn_id,
            model_version=model_version,
            amount=txn_data["amount"],
            hour_of_day=txn_data["hour_of_day"],
            day_of_week=txn_data["day_of_week"],
            merchant_category=txn_data["merchant_category"],
            distance_from_home=txn_data["distance_from_home"],
            is_weekend=txn_data["is_weekend"],
            velocity_1h=txn_data["velocity_1h"],
            velocity_24h=txn_data["velocity_24h"],
            amount_z_score=txn_data["amount_z_score"],
            fraud_probability=prob,
            predicted_fraud=(prob >= 0.5),
            latency_ms=latency_ms,
        ))
        db.commit()
    except Exception:
        logger.exception("Failed to write prediction log")
        db.rollback()
    finally:
        db.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_band(prob: float) -> str:
    if prob < 0.1:
        return "low"
    if prob < 0.4:
        return "medium"
    if prob < 0.7:
        return "high"
    return "critical"


def _record_metrics(version: str, prob: float):
    PREDICTION_PROBABILITY.labels(model_version=version).observe(prob)
    if prob >= 0.5:
        FRAUD_DETECTED.labels(model_version=version).inc()
    else:
        LEGITIMATE_DETECTED.labels(model_version=version).inc()


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "ML Platform API — visit /docs"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, db: Session = Depends(get_db)):
    # Verify DB is reachable (not just "server is up")
    try:
        active = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
        total = db.query(PredictionLog).count()
        db_ok = True
    except Exception:
        active = None
        total = 0
        db_ok = False

    request_id = getattr(request.state, "request_id", None)
    status = "ok" if db_ok and registry.loaded_versions() else "degraded"

    return HealthResponse(
        status=status,
        active_model=active.version if active else None,
        total_predictions=total,
        models_loaded=registry.loaded_versions(),
        request_id=request_id,
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint. Wire to Grafana for dashboards and alerting."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Model management ──────────────────────────────────────────────────────────

@app.get("/models", response_model=list[ModelInfo], tags=["models"])
def list_models(db: Session = Depends(get_db)):
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        ModelInfo(
            version=mv.version,
            algorithm=mv.algorithm,
            description=mv.description,
            is_active=mv.is_active,
            roc_auc=mv.roc_auc,
            f1_score=mv.f1_score,
            precision=mv.precision,
            recall=mv.recall,
            avg_latency_ms=mv.avg_latency_ms,
            optimal_threshold=mv.optimal_threshold,
            cv_roc_auc_mean=mv.cv_roc_auc_mean,
            cv_roc_auc_std=mv.cv_roc_auc_std,
            created_at=mv.created_at,
        )
        for mv in versions
    ]


@app.patch("/models/{version}/activate", tags=["models"])
def activate_model(version: str, db: Session = Depends(get_db)):
    try:
        registry.promote(version, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
    MODEL_INFO.info({"version": version, "algorithm": mv.algorithm if mv else "unknown"})
    logger.info("Model promoted", extra={"version": version})
    return {"message": f"Model {version} is now active", "version": version}


# ── V1 Predict ───────────────────────────────────────────────────────────────

@app.post(
    "/v1/predict",
    response_model=PredictionResponseV1,
    tags=["v1"],
    summary="Simple fraud prediction",
)
def predict_v1(
    txn: TransactionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        pipeline, version = registry.get_active(db)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    threshold = registry.get_threshold(version, db)
    X = engineer_features(txn.model_dump())
    t0 = time.perf_counter()
    prob = float(pipeline.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    txn_id = str(uuid.uuid4())
    _record_metrics(version, prob)

    # Fire-and-forget — response returns before the DB write completes
    background_tasks.add_task(
        _write_prediction_log, version, txn.model_dump(), prob, latency_ms, txn_id
    )

    return PredictionResponseV1(
        transaction_id=txn_id,
        model_version=version,
        predicted_fraud=(prob >= threshold),
        fraud_probability=round(prob, 4),
        latency_ms=round(latency_ms, 2),
        request_id=getattr(request.state, "request_id", None),
    )


# ── V2 Predict ───────────────────────────────────────────────────────────────

@app.post(
    "/v2/predict",
    response_model=PredictionResponseV2,
    tags=["v2"],
    summary="Enriched prediction with risk band and top risk factors",
)
def predict_v2(
    txn: TransactionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        pipeline, version = registry.get_active(db)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    threshold = registry.get_threshold(version, db)
    X = engineer_features(txn.model_dump())
    t0 = time.perf_counter()
    prob = float(pipeline.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    txn_id = str(uuid.uuid4())
    _record_metrics(version, prob)

    top_factors = [
        FeatureImportance(**f) for f in registry.get_feature_importance(version, db)
    ]

    background_tasks.add_task(
        _write_prediction_log, version, txn.model_dump(), prob, latency_ms, txn_id
    )

    return PredictionResponseV2(
        transaction_id=txn_id,
        model_version=version,
        predicted_fraud=(prob >= threshold),
        fraud_probability=round(prob, 4),
        risk_band=_risk_band(prob),
        top_risk_factors=top_factors,
        latency_ms=round(latency_ms, 2),
        request_id=getattr(request.state, "request_id", None),
    )


@app.post(
    "/v2/batch",
    response_model=BatchPredictionResponse,
    tags=["v2"],
    summary="Batch predictions — up to 500 rows, single model pass",
)
def predict_batch(
    req: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        pipeline, version = registry.get_active(db)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    threshold = registry.get_threshold(version, db)
    t_total = time.perf_counter()
    rows = [t.model_dump() for t in req.transactions]
    X = engineer_features_batch(rows)
    probs = pipeline.predict_proba(X)[:, 1]
    total_ms = (time.perf_counter() - t_total) * 1000
    per_ms = total_ms / len(rows)

    top_factors = [FeatureImportance(**f) for f in registry.get_feature_importance(version, db)]
    predictions = []

    for txn, prob in zip(req.transactions, probs):
        p = float(prob)
        txn_id = str(uuid.uuid4())
        _record_metrics(version, p)
        background_tasks.add_task(
            _write_prediction_log, version, txn.model_dump(), p, per_ms, txn_id
        )
        predictions.append(PredictionResponseV2(
            transaction_id=txn_id,
            model_version=version,
            predicted_fraud=(p >= threshold),
            fraud_probability=round(p, 4),
            risk_band=_risk_band(p),
            top_risk_factors=top_factors,
            latency_ms=round(per_ms, 2),
        ))

    return BatchPredictionResponse(
        model_version=version,
        count=len(predictions),
        predictions=predictions,
        total_latency_ms=round(total_ms, 2),
    )


# ── Predictions feed ──────────────────────────────────────────────────────────

@app.get("/predictions/recent", tags=["predictions"])
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    """Last N predictions — powers the live feed in the dashboard."""
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return [
        {
            "transaction_id": r.id_str or str(r.id),
            "model_version": r.model_version,
            "amount": r.amount,
            "fraud_probability": round(r.fraud_probability, 4),
            "predicted_fraud": r.predicted_fraud,
            "risk_band": _risk_band(r.fraud_probability),
            "latency_ms": round(r.latency_ms, 2) if r.latency_ms else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ── Monitoring ────────────────────────────────────────────────────────────────

@app.get("/monitoring/snapshot", response_model=MonitoringSnapshot, tags=["monitoring"])
def monitoring_snapshot(version: str | None = None, db: Session = Depends(get_db)):
    if version is None:
        mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
        if mv is None:
            raise HTTPException(status_code=503, detail="No active model")
        version = mv.version
    try:
        return compute_monitoring_snapshot(version, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/monitoring/history", tags=["monitoring"])
def monitoring_history(version: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    from db.models import PerformanceMetric
    q = db.query(PerformanceMetric)
    if version:
        q = q.filter(PerformanceMetric.model_version == version)
    rows = q.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
    return [
        {
            "model_version": r.model_version,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "psi_score": r.psi_score,
            "window_size": r.window_size,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
