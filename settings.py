from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_dir: Path = Path(__file__).parent

    # Storage
    database_url: str = ""
    mlflow_tracking_uri: str = ""

    # MLflow
    mlflow_experiment_name: str = "fraud-detection"

    # Data generation
    n_train_samples: int = 50_000
    n_test_samples: int = 10_000
    fraud_rate: float = 0.015
    random_seed: int = 42

    # Monitoring thresholds
    psi_alert_threshold: float = 0.2
    accuracy_drop_threshold: float = 0.05
    min_samples_for_monitoring: int = 50

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    @property
    def model_artifacts_dir(self) -> Path:
        d = self.base_dir / "models"
        d.mkdir(exist_ok=True)
        return d

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.base_dir}/data/fraud.db"

    @property
    def resolved_mlflow_tracking_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{self.base_dir}/mlruns/mlflow.db"


settings = Settings()
