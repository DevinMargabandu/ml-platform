"""
Prometheus metrics definitions.

Exposes:
  fraud_api_requests_total        — request count by method/path/status
  fraud_api_request_duration_seconds — latency histogram by path
  fraud_prediction_probability    — histogram of model output probabilities
  fraud_detections_total          — count of predicted frauds by model version
  fraud_model_info                — info gauge (labels carry version/algorithm)
"""

from prometheus_client import Counter, Histogram, Info, REGISTRY

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Distribution of model output fraud probabilities",
    ["model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

FRAUD_DETECTED = Counter(
    "fraud_detections_total",
    "Total transactions predicted as fraud",
    ["model_version"],
)

LEGITIMATE_DETECTED = Counter(
    "legitimate_detections_total",
    "Total transactions predicted as legitimate",
    ["model_version"],
)

MODEL_INFO = Info(
    "fraud_active_model",
    "Currently active fraud detection model",
)
