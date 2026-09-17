"""
Model registry — loads versioned sklearn pipelines from disk and caches them.

The registry is a singleton in-process cache. On promote(), the new artifact
is pre-loaded so the first post-swap request doesn't pay a cold-start penalty.

Thread safety: SQLite serializes writes; the in-memory cache dict is safe for
CPython (GIL), but a multi-process deployment should use a shared model store
(e.g. Redis + S3) instead.
"""

import joblib
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from db.models import ModelVersion


FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category",
    "distance_from_home", "is_weekend", "velocity_1h", "velocity_24h",
    "amount_z_score", "amount_log", "velocity_ratio", "night_flag", "high_distance_flag",
]


def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df["amount_log"] = np.log1p(df["amount"])
    df["velocity_ratio"] = df["velocity_1h"] / (df["velocity_24h"] + 1)
    df["night_flag"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["high_distance_flag"] = (df["distance_from_home"] > 100).astype(int)
    return df[FEATURE_COLS]


def engineer_features_batch(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["amount_log"] = np.log1p(df["amount"])
    df["velocity_ratio"] = df["velocity_1h"] / (df["velocity_24h"] + 1)
    df["night_flag"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["high_distance_flag"] = (df["distance_from_home"] > 100).astype(int)
    return df[FEATURE_COLS]


class ModelRegistry:
    def __init__(self):
        self._cache: dict[str, object] = {}

    def _load(self, artifact_path: str) -> object:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        return joblib.load(path)

    def get(self, version: str, db: Session) -> object:
        if version not in self._cache:
            mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
            if mv is None:
                raise ValueError(f"Model version '{version}' not registered")
            self._cache[version] = self._load(mv.artifact_path)
        return self._cache[version]

    def get_active(self, db: Session) -> tuple[object, str]:
        mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
        if mv is None:
            raise ValueError("No active model registered. Run the training pipeline first.")
        return self.get(mv.version, db), mv.version

    def get_threshold(self, version: str, db: Session) -> float:
        """Return the F1-optimal threshold for this model version (default 0.5)."""
        mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
        if mv and mv.optimal_threshold:
            return mv.optimal_threshold
        return 0.5

    def promote(self, version: str, db: Session):
        """Hot-swap the active model. Pre-loads the artifact to avoid cold-start on first call."""
        db.query(ModelVersion).update({ModelVersion.is_active: False})
        mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
        if mv is None:
            raise ValueError(f"Version '{version}' not found")
        mv.is_active = True
        db.commit()
        self.get(version, db)  # warm the cache

    def loaded_versions(self) -> list[str]:
        return list(self._cache.keys())

    def get_feature_importance(self, version: str, db: Session) -> list[dict]:
        model = self.get(version, db)

        # Unwrap CalibratedClassifierCV → Pipeline → base classifier
        # Structure: CalibratedClassifierCV.calibrated_classifiers_[0].estimator → Pipeline
        if hasattr(model, "calibrated_classifiers_"):
            inner = model.calibrated_classifiers_[0].estimator
            clf = inner.named_steps["clf"] if hasattr(inner, "named_steps") else inner
        elif hasattr(model, "named_steps"):
            clf = model.named_steps["clf"]
        else:
            clf = model

        if not hasattr(clf, "feature_importances_"):
            return []

        importances = clf.feature_importances_
        return sorted(
            [{"feature": f, "importance": float(i)} for f, i in zip(FEATURE_COLS, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )[:5]


registry = ModelRegistry()
