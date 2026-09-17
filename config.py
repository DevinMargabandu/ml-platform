# Backward-compatibility shim — prefer importing from settings directly.
from settings import settings

DATABASE_URL = settings.resolved_database_url
MLFLOW_TRACKING_URI = settings.resolved_mlflow_tracking_uri
MLFLOW_EXPERIMENT_NAME = settings.mlflow_experiment_name
MODEL_ARTIFACTS_DIR = settings.model_artifacts_dir
N_TRAIN_SAMPLES = settings.n_train_samples
N_TEST_SAMPLES = settings.n_test_samples
FRAUD_RATE = settings.fraud_rate
RANDOM_SEED = settings.random_seed
PSI_ALERT_THRESHOLD = settings.psi_alert_threshold
ACCURACY_DROP_THRESHOLD = settings.accuracy_drop_threshold
MIN_SAMPLES_FOR_MONITORING = settings.min_samples_for_monitoring
API_V1_PREFIX = "/v1"
API_V2_PREFIX = "/v2"
