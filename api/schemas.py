from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")
    merchant_category: int = Field(..., ge=0, le=9)
    distance_from_home: float = Field(..., ge=0, description="Distance from cardholder home in km")
    is_weekend: bool
    velocity_1h: int = Field(..., ge=0, description="Card transactions in the last 1 hour")
    velocity_24h: int = Field(..., ge=0, description="Card transactions in the last 24 hours")
    amount_z_score: float = Field(..., description="Z-score of amount vs cardholder history")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 850.00,
                "hour_of_day": 2,
                "day_of_week": 6,
                "merchant_category": 8,
                "distance_from_home": 450.0,
                "is_weekend": True,
                "velocity_1h": 7,
                "velocity_24h": 18,
                "amount_z_score": 3.2,
            }
        }
    }


class PredictionResponseV1(BaseModel):
    transaction_id: str       # UUID — use this for feedback / chargeback correlation
    model_version: str
    predicted_fraud: bool
    fraud_probability: float
    latency_ms: float
    request_id: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class PredictionResponseV2(BaseModel):
    transaction_id: str
    model_version: str
    predicted_fraud: bool
    fraud_probability: float
    risk_band: str                             # "low" | "medium" | "high" | "critical"
    top_risk_factors: list[FeatureImportance]
    latency_ms: float
    request_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    model_version: str
    count: int
    predictions: list[PredictionResponseV2]
    total_latency_ms: float


class ModelInfo(BaseModel):
    version: str
    algorithm: str
    description: Optional[str]
    is_active: bool
    roc_auc: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    avg_latency_ms: Optional[float]
    optimal_threshold: Optional[float]
    cv_roc_auc_mean: Optional[float]
    cv_roc_auc_std: Optional[float]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str                    # "ok" | "degraded"
    active_model: Optional[str]
    total_predictions: int
    models_loaded: list[str]
    request_id: Optional[str] = None


class MonitoringSnapshot(BaseModel):
    model_version: str
    window_size: int
    fraud_rate_observed: float
    fraud_rate_baseline: float
    psi_score: float
    drift_detected: bool
    accuracy: Optional[float]
    f1: Optional[float]
    alert_message: Optional[str]
    computed_at: datetime
