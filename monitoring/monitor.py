"""
Model monitoring — drift detection and performance tracking.

Population Stability Index (PSI) measures distributional shift:
  PSI < 0.10  → stable
  0.10–0.20  → moderate drift — investigate
  > 0.20     → major drift — retrain

Why PSI on the output distribution?
  Feature-level PSI requires one chart per feature; output PSI is a single
  early-warning signal. When it fires, THEN you drill into feature distributions.
  This keeps the alert surface manageable while still catching drift early.

Ground-truth latency:
  Chargebacks arrive 30–90 days after a transaction. The `actual_fraud` column
  is nullable. Accuracy metrics only compute when labeled feedback is available;
  PSI runs immediately on all predictions.
"""

import numpy as np
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from db.models import PredictionLog, ModelVersion, PerformanceMetric
from api.schemas import MonitoringSnapshot
from settings import settings

# Training-time fraud rate (matches data/ingest.py design)
BASELINE_FRAUD_RATE = settings.fraud_rate


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two probability distributions.

    Uses equal-width bins over [0, 1] — appropriate for model output probabilities.
    Epsilon clipping prevents log(0); normalization handles different sample sizes.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    eps = 1e-6

    def _bucket(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / (len(arr) + eps)
        pct = np.clip(pct, eps, None)
        return pct / pct.sum()

    e = _bucket(expected)
    a = _bucket(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_monitoring_snapshot(version: str, db: Session) -> MonitoringSnapshot:
    mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
    if mv is None:
        raise ValueError(f"Version '{version}' not registered")

    # Most recent 500 predictions for this model
    recent_logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.model_version == version)
        .order_by(PredictionLog.created_at.desc())
        .limit(500)
        .all()
    )

    n = len(recent_logs)
    if n < settings.min_samples_for_monitoring:
        return MonitoringSnapshot(
            model_version=version,
            window_size=n,
            fraud_rate_observed=0.0,
            fraud_rate_baseline=BASELINE_FRAUD_RATE,
            psi_score=0.0,
            drift_detected=False,
            accuracy=None,
            f1=None,
            alert_message=(
                f"Only {n} predictions logged (need ≥{settings.min_samples_for_monitoring}). "
                "Run more predictions to enable monitoring."
            ),
            computed_at=datetime.now(timezone.utc),
        )

    probs = np.array([r.fraud_probability for r in recent_logs])
    observed_fraud_rate = float((probs >= 0.5).mean())

    # Baseline: first N predictions (represents the training distribution)
    # In production: compare against a stored reference distribution artifact.
    baseline_rows = (
        db.query(PredictionLog.fraud_probability)
        .filter(PredictionLog.model_version == version)
        .order_by(PredictionLog.created_at.asc())
        .limit(min(1000, n))
        .all()
    )
    baseline_probs = np.array([r.fraud_probability for r in baseline_rows])

    # Need sufficient baseline samples for PSI to be meaningful
    psi = _psi(baseline_probs, probs) if len(baseline_probs) >= 20 else 0.0

    # Ground-truth metrics (only when labeled)
    labeled = [r for r in recent_logs if r.actual_fraud is not None]
    accuracy = f1 = None
    if len(labeled) >= 10:
        y_true = np.array([int(r.actual_fraud) for r in labeled])
        y_pred = np.array([int(r.predicted_fraud) for r in labeled])
        accuracy = float((y_true == y_pred).mean())

        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

    drift_detected = psi > settings.psi_alert_threshold
    alert = None
    if drift_detected:
        alert = (
            f"DRIFT: PSI={psi:.3f} exceeds threshold {settings.psi_alert_threshold}. "
            "Investigate feature distributions and consider retraining."
        )
    elif mv.roc_auc and accuracy is not None:
        drop = mv.roc_auc - accuracy
        if drop > settings.accuracy_drop_threshold:
            alert = (
                f"ACCURACY DROP: baseline={mv.roc_auc:.3f} "
                f"current≈{accuracy:.3f} (Δ={drop:.3f}). Possible concept drift."
            )

    _persist(db, version, n, observed_fraud_rate, psi, accuracy, f1)

    return MonitoringSnapshot(
        model_version=version,
        window_size=n,
        fraud_rate_observed=round(observed_fraud_rate, 4),
        fraud_rate_baseline=BASELINE_FRAUD_RATE,
        psi_score=round(psi, 4),
        drift_detected=drift_detected,
        accuracy=round(accuracy, 4) if accuracy is not None else None,
        f1=round(f1, 4) if f1 is not None else None,
        alert_message=alert,
        computed_at=datetime.now(timezone.utc),
    )


def _persist(
    db: Session, version: str, window_size: int,
    fraud_rate: float, psi: float, accuracy, f1,
):
    rows = [
        PerformanceMetric(model_version=version, metric_name="fraud_rate",
                          metric_value=fraud_rate, window_size=window_size, psi_score=psi),
        PerformanceMetric(model_version=version, metric_name="psi_score",
                          metric_value=psi, window_size=window_size, psi_score=psi),
    ]
    if accuracy is not None:
        rows.append(PerformanceMetric(model_version=version, metric_name="accuracy",
                                      metric_value=accuracy, window_size=window_size, psi_score=psi))
    if f1 is not None:
        rows.append(PerformanceMetric(model_version=version, metric_name="f1",
                                      metric_value=f1, window_size=window_size, psi_score=psi))
    db.bulk_save_objects(rows)
    db.commit()
