"""
Lightweight training script for Render — no mlflow dependency.
Trains v1 (RandomForest) and v2 (GradientBoosting) and writes
both artifacts to models/ and registers them in the SQLite DB.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve,
)

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from db.database import SessionLocal, init_db
from db.models import Transaction, ModelVersion
from settings import settings

FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category",
    "distance_from_home", "is_weekend", "velocity_1h", "velocity_24h",
    "amount_z_score",
    "amount_log", "velocity_ratio", "night_flag", "high_distance_flag",
]

BASE_MODELS = {
    "v1": {
        "algorithm": "RandomForestClassifier",
        "description": "Interpretable baseline. 100 trees, balanced class weight, isotonic calibration.",
        "clf": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=settings.random_seed,
        ),
    },
    "v2": {
        "algorithm": "GradientBoostingClassifier",
        "description": "Higher AUC via boosting. 200 trees, lr=0.05, isotonic calibration.",
        "clf": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=settings.random_seed,
        ),
    },
}


def load_data(db):
    def _query(split):
        rows = db.query(Transaction).filter(Transaction.split == split).all()
        return pd.DataFrame([{
            "amount": r.amount,
            "hour_of_day": r.hour_of_day,
            "day_of_week": r.day_of_week,
            "merchant_category": r.merchant_category,
            "distance_from_home": r.distance_from_home,
            "is_weekend": int(r.is_weekend),
            "velocity_1h": r.velocity_1h,
            "velocity_24h": r.velocity_24h,
            "amount_z_score": r.amount_z_score,
            "is_fraud": int(r.is_fraud),
        } for r in rows])
    return _query("train"), _query("test")


def engineer_features(df):
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    df["velocity_ratio"] = df["velocity_1h"] / (df["velocity_24h"] + 1)
    df["night_flag"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["high_distance_flag"] = (df["distance_from_home"] > 100).astype(int)
    return df


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax()
    return float(thresholds[best_idx])


def train_version(version, train_df, test_df, db):
    meta = BASE_MODELS[version]
    print(f"\nTraining {version} — {meta['algorithm']}")

    X_train = engineer_features(train_df)[FEATURE_COLS]
    y_train = train_df["is_fraud"]
    X_test = engineer_features(test_df)[FEATURE_COLS]
    y_test = test_df["is_fraud"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_seed)
    raw_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", meta["clf"]),
    ])
    cv_scores = cross_val_score(raw_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    print(f"  CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")

    t0 = time.perf_counter()
    calibrated_clf = CalibratedClassifierCV(estimator=raw_pipeline, method="isotonic", cv=3)
    calibrated_clf.fit(X_train, y_train)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    y_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test.values, y_prob)
    y_pred = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}  F1: {metrics['f1']:.4f}  "
          f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")

    artifact_path = settings.model_artifacts_dir / f"model_{version}.joblib"
    joblib.dump(calibrated_clf, artifact_path)
    print(f"  Saved → {artifact_path}")

    existing = db.query(ModelVersion).filter(ModelVersion.version == version).first()
    if existing:
        db.delete(existing)
        db.commit()

    mv = ModelVersion(
        version=version,
        algorithm=meta["algorithm"],
        description=meta["description"],
        artifact_path=str(artifact_path),
        is_active=(version == "v1"),
        roc_auc=metrics["roc_auc"],
        f1_score=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        optimal_threshold=optimal_threshold,
        cv_roc_auc_mean=cv_mean,
        cv_roc_auc_std=cv_std,
    )
    db.add(mv)
    db.commit()
    print(f"  Registered {version} in DB (active={version == 'v1'})")


def run():
    print("Initializing DB …")
    init_db()
    db = SessionLocal()
    try:
        n = db.query(Transaction).count()
        print(f"  Found {n:,} transactions in DB")
        if n == 0:
            print("ERROR: No data in DB — run data/ingest.py first")
            sys.exit(1)

        train_df, test_df = load_data(db)
        print(f"  Train: {len(train_df):,} rows  Test: {len(test_df):,} rows")

        for version in ["v1", "v2"]:
            train_version(version, train_df, test_df, db)

        print("\nAll models trained and registered.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
