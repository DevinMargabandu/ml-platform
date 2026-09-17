from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text, Index
from db.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    hour_of_day = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    merchant_category = Column(Integer, nullable=False)
    distance_from_home = Column(Float, nullable=False)
    is_weekend = Column(Boolean, nullable=False)
    velocity_1h = Column(Integer, nullable=False)
    velocity_24h = Column(Integer, nullable=False)
    amount_z_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    split = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_transactions_split", "split"),
        Index("ix_transactions_is_fraud", "is_fraud"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    description = Column(Text)
    artifact_path = Column(String(500), nullable=False)
    mlflow_run_id = Column(String(100))
    is_active = Column(Boolean, default=False)

    # Test-set metrics
    roc_auc = Column(Float)
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    avg_latency_ms = Column(Float)

    # Decision threshold — optimized on test set to maximize F1
    optimal_threshold = Column(Float, default=0.5)

    # Cross-validation scores — gives confidence intervals, not just point estimates
    cv_roc_auc_mean = Column(Float)
    cv_roc_auc_std = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    id_str = Column(String(36), index=True)   # UUID for external reference
    model_version = Column(String(50), nullable=False, index=True)
    amount = Column(Float)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    merchant_category = Column(Integer)
    distance_from_home = Column(Float)
    is_weekend = Column(Boolean)
    velocity_1h = Column(Integer)
    velocity_24h = Column(Integer)
    amount_z_score = Column(Float)
    fraud_probability = Column(Float, nullable=False)
    predicted_fraud = Column(Boolean, nullable=False)
    actual_fraud = Column(Boolean)   # nullable — filled in via chargeback feedback
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_prediction_logs_created_at", "created_at"),
    )


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    window_size = Column(Integer)
    psi_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
