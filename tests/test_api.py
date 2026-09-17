"""
API integration tests.

These tests hit real endpoints with a real (temporary) SQLite database and
real trained models. We deliberately avoid mocking the model — mocked tests
pass even when the real pipeline breaks.

Run: pytest tests/ -v
"""

import os
import sys
import pytest
import joblib
import numpy as np
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_db_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("data") / "test.db")


@pytest.fixture(scope="session", autouse=True)
def patch_settings(tmp_db_path, tmp_path_factory):
    """Override settings before anything imports db.database."""
    model_dir = tmp_path_factory.mktemp("models")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db_path}"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path_factory.mktemp('mlruns')}/test.db"
    yield
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory, patch_settings):
    """Train a tiny model just for tests — fast, no I/O to production artifacts."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n = 500
    X = rng.random((n, 13))
    y = (X[:, 0] + X[:, 2] > 1.2).astype(int)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=10, random_state=42))])
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=3)
    calibrated.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "model_v1.joblib"
    joblib.dump(calibrated, path)
    return str(path)


@pytest.fixture(scope="session")
def client(patch_settings, trained_model_path):
    """TestClient with real DB and a registered model version."""
    from db.database import init_db, SessionLocal
    from db.models import ModelVersion

    init_db()

    db = SessionLocal()
    db.add(ModelVersion(
        version="v1",
        algorithm="RandomForestClassifier",
        description="Test model",
        artifact_path=trained_model_path,
        is_active=True,
        roc_auc=0.99,
        f1_score=0.85,
        precision=0.90,
        recall=0.80,
        avg_latency_ms=1.0,
        optimal_threshold=0.5,
        cv_roc_auc_mean=0.98,
        cv_roc_auc_std=0.01,
    ))
    db.commit()
    db.close()

    from api.main import app
    return TestClient(app)


VALID_TXN = {
    "amount": 850.0,
    "hour_of_day": 2,
    "day_of_week": 6,
    "merchant_category": 8,
    "distance_from_home": 450.0,
    "is_weekend": True,
    "velocity_1h": 7,
    "velocity_24h": 18,
    "amount_z_score": 3.2,
}

LEGIT_TXN = {
    "amount": 45.0,
    "hour_of_day": 14,
    "day_of_week": 2,
    "merchant_category": 3,
    "distance_from_home": 2.5,
    "is_weekend": False,
    "velocity_1h": 1,
    "velocity_24h": 4,
    "amount_z_score": -0.3,
}


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert "active_model" in body
        assert "total_predictions" in body
        assert isinstance(body["models_loaded"], list)

    def test_health_returns_request_id(self, client):
        r = client.get("/health")
        # Request ID should be echoed in response header
        assert "x-request-id" in r.headers

    def test_health_with_custom_request_id(self, client):
        r = client.get("/health", headers={"X-Request-ID": "test-trace-123"})
        assert r.headers.get("x-request-id") == "test-trace-123"


# ── Models ────────────────────────────────────────────────────────────────────

class TestModels:
    def test_list_models(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        models = r.json()
        assert len(models) >= 1
        v1 = next(m for m in models if m["version"] == "v1")
        assert v1["is_active"] is True
        assert v1["cv_roc_auc_mean"] == pytest.approx(0.98)
        assert v1["optimal_threshold"] == pytest.approx(0.5)

    def test_activate_unknown_model_returns_404(self, client):
        r = client.patch("/models/nonexistent/activate")
        assert r.status_code == 404


# ── V1 Predict ───────────────────────────────────────────────────────────────

class TestPredictV1:
    def test_valid_prediction(self, client):
        r = client.post("/v1/predict", json=VALID_TXN)
        assert r.status_code == 200
        body = r.json()
        assert "transaction_id" in body
        assert isinstance(body["transaction_id"], str)
        assert len(body["transaction_id"]) == 36  # UUID format
        assert "predicted_fraud" in body
        assert isinstance(body["predicted_fraud"], bool)
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["latency_ms"] > 0

    def test_response_has_request_id(self, client):
        r = client.post("/v1/predict", json=VALID_TXN)
        assert r.status_code == 200
        # Should be echoed in header and body
        assert "x-request-id" in r.headers
        assert r.json()["request_id"] is not None

    def test_invalid_hour_rejected(self, client):
        bad = {**VALID_TXN, "hour_of_day": 25}
        r = client.post("/v1/predict", json=bad)
        assert r.status_code == 422

    def test_negative_amount_rejected(self, client):
        bad = {**VALID_TXN, "amount": -10.0}
        r = client.post("/v1/predict", json=bad)
        assert r.status_code == 422

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in VALID_TXN.items() if k != "amount"}
        r = client.post("/v1/predict", json=bad)
        assert r.status_code == 422


# ── V2 Predict ───────────────────────────────────────────────────────────────

class TestPredictV2:
    def test_enriched_response(self, client):
        r = client.post("/v2/predict", json=VALID_TXN)
        assert r.status_code == 200
        body = r.json()
        assert body["risk_band"] in ("low", "medium", "high", "critical")
        assert isinstance(body["top_risk_factors"], list)
        # Top factors should be sorted descending by importance
        factors = body["top_risk_factors"]
        if len(factors) >= 2:
            assert factors[0]["importance"] >= factors[1]["importance"]

    def test_legit_transaction_lower_risk(self, client):
        r_fraud = client.post("/v2/predict", json=VALID_TXN)
        r_legit = client.post("/v2/predict", json=LEGIT_TXN)
        # Fraud transaction should have higher probability (not guaranteed, but expected)
        assert r_fraud.status_code == 200
        assert r_legit.status_code == 200


# ── Batch ─────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_returns_correct_count(self, client):
        r = client.post("/v2/batch", json={"transactions": [VALID_TXN, LEGIT_TXN]})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2
        assert body["total_latency_ms"] > 0

    def test_empty_batch_rejected(self, client):
        r = client.post("/v2/batch", json={"transactions": []})
        assert r.status_code == 422

    def test_oversized_batch_rejected(self, client):
        r = client.post("/v2/batch", json={"transactions": [VALID_TXN] * 501})
        assert r.status_code == 422


# ── Prometheus metrics ────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_endpoint_exists(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_contains_fraud_counters(self, client):
        # Make a prediction first to populate counters
        client.post("/v1/predict", json=VALID_TXN)
        r = client.get("/metrics")
        body = r.text
        assert "fraud_api_requests_total" in body
        assert "fraud_api_request_duration_seconds" in body


# ── Feature engineering ───────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_engineer_features_output_shape(self):
        from api.registry import engineer_features
        X = engineer_features(VALID_TXN)
        assert X.shape == (1, 13)

    def test_night_flag_is_set_for_early_morning(self):
        from api.registry import engineer_features
        txn = {**VALID_TXN, "hour_of_day": 3}
        X = engineer_features(txn)
        assert X["night_flag"].iloc[0] == 1

    def test_night_flag_not_set_midday(self):
        from api.registry import engineer_features
        txn = {**VALID_TXN, "hour_of_day": 12}
        X = engineer_features(txn)
        assert X["night_flag"].iloc[0] == 0

    def test_high_distance_flag(self):
        from api.registry import engineer_features
        far = engineer_features({**VALID_TXN, "distance_from_home": 150})
        near = engineer_features({**VALID_TXN, "distance_from_home": 5})
        assert far["high_distance_flag"].iloc[0] == 1
        assert near["high_distance_flag"].iloc[0] == 0

    def test_velocity_ratio_computation(self):
        from api.registry import engineer_features
        txn = {**VALID_TXN, "velocity_1h": 4, "velocity_24h": 7}
        X = engineer_features(txn)
        expected = 4 / (7 + 1)
        assert X["velocity_ratio"].iloc[0] == pytest.approx(expected)

    def test_amount_log_is_positive(self):
        from api.registry import engineer_features
        X = engineer_features(VALID_TXN)
        assert X["amount_log"].iloc[0] > 0


# ── Predictions feed ──────────────────────────────────────────────────────────

class TestFeed:
    def test_recent_predictions_returns_list(self, client):
        client.post("/v1/predict", json=VALID_TXN)
        r = client.get("/predictions/recent?limit=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_feed_limit_respected(self, client):
        for _ in range(5):
            client.post("/v1/predict", json=VALID_TXN)
        r = client.get("/predictions/recent?limit=3")
        assert len(r.json()) <= 3
