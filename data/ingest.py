"""
Synthetic fraud dataset generator and DB ingestion pipeline.

Fraud signal design (realistic patterns):
  - High amount + unusual hour → elevated risk
  - High velocity (many txns in short window) → elevated risk
  - Large distance from home + online/international merchant → elevated risk
  - High amount z-score → elevated risk
"""

import os

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from db.database import init_db, SessionLocal
from db.models import Transaction
from config import N_TRAIN_SAMPLES, N_TEST_SAMPLES, FRAUD_RATE, RANDOM_SEED


def generate_transactions(n: int, fraud_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def _legit(size):
        hours = rng.integers(0, 24, size=size)  # uniform — legit txns happen any time
        return pd.DataFrame({
            "amount": rng.lognormal(mean=3.8, sigma=1.3, size=size).clip(1, 5000),
            "hour_of_day": hours,
            "day_of_week": rng.integers(0, 7, size=size),
            "merchant_category": rng.integers(0, 10, size=size),
            "distance_from_home": rng.lognormal(mean=2.5, sigma=1.5, size=size).clip(0, 2000),
            "is_weekend": rng.integers(0, 7, size=size) >= 5,
            "velocity_1h": rng.integers(1, 6, size=size),
            "velocity_24h": rng.integers(1, 20, size=size),
            "is_fraud": False,
        })

    def _fraud(size):
        # Fraud has weak signals — mostly overlapping with legit, slight skew
        hours = rng.integers(0, 24, size=size)  # fraud also happens any time
        return pd.DataFrame({
            "amount": rng.lognormal(mean=4.5, sigma=1.4, size=size).clip(1, 10000),
            "hour_of_day": hours,
            "day_of_week": rng.integers(0, 7, size=size),
            "merchant_category": rng.integers(0, 10, size=size),
            "distance_from_home": rng.lognormal(mean=3.5, sigma=1.5, size=size).clip(0, 5000),
            "is_weekend": rng.integers(0, 7, size=size) >= 5,
            "velocity_1h": rng.integers(2, 10, size=size),
            "velocity_24h": rng.integers(5, 30, size=size),
            "is_fraud": True,
        })

    df = pd.concat([_legit(n_legit), _fraud(n_fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    # Compute amount_z_score from the full population — avoids label leakage
    df["amount_z_score"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = df["amount"].clip(lower=0.01)
    df["hour_of_day"] = df["hour_of_day"].clip(0, 23).astype(int)
    df["day_of_week"] = df["day_of_week"].clip(0, 6).astype(int)
    df["merchant_category"] = df["merchant_category"].clip(0, 9).astype(int)
    df["distance_from_home"] = df["distance_from_home"].clip(lower=0)
    df["velocity_1h"] = df["velocity_1h"].clip(lower=0).astype(int)
    df["velocity_24h"] = df["velocity_24h"].clip(lower=0).astype(int)
    df["is_weekend"] = df["is_weekend"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df = df.dropna()
    return df


def ingest(db: Session, df: pd.DataFrame, split: str) -> int:
    records = [
        Transaction(
            amount=float(row.amount),
            hour_of_day=int(row.hour_of_day),
            day_of_week=int(row.day_of_week),
            merchant_category=int(row.merchant_category),
            distance_from_home=float(row.distance_from_home),
            is_weekend=bool(row.is_weekend),
            velocity_1h=int(row.velocity_1h),
            velocity_24h=int(row.velocity_24h),
            amount_z_score=float(row.amount_z_score),
            is_fraud=bool(row.is_fraud),
            split=split,
        )
        for row in df.itertuples(index=False)
    ]
    db.bulk_save_objects(records)
    db.commit()
    return len(records)


def run():
    print("Initializing database …")
    init_db()

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Generating {N_TRAIN_SAMPLES:,} training transactions …")
    train_df = clean(generate_transactions(N_TRAIN_SAMPLES, FRAUD_RATE, rng))

    print(f"Generating {N_TEST_SAMPLES:,} test transactions …")
    test_df = clean(generate_transactions(N_TEST_SAMPLES, FRAUD_RATE, rng))

    db = SessionLocal()
    try:
        existing = db.query(Transaction).count()
        if existing > 0:
            print(f"Database already contains {existing:,} rows — skipping ingest.")
            return
        n_train = ingest(db, train_df, "train")
        n_test = ingest(db, test_df, "test")
        print(f"Ingested {n_train:,} train + {n_test:,} test rows.")
        print(f"Fraud rate — train: {train_df.is_fraud.mean():.2%}  test: {test_df.is_fraud.mean():.2%}")
    finally:
        db.close()


if __name__ == "__main__":
    run()
