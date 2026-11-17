"""Feature engineering. All customer-history features use only PAST transactions (no look-ahead leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["log_amount", "hour", "is_night", "is_weekend", "distance_from_home_km", "is_foreign",
                    "amount_to_customer_mean", "amount_zscore", "tx_count_1h", "tx_count_24h",
                    "minutes_since_prev", "customer_tx_index"]
CATEGORICAL_FEATURES = ["merchant_category", "channel"]


def engineer_features(df: pd.DataFrame, home_country: str = "IN") -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["is_foreign"] = (df["country"] != home_country).astype(int)

    g = df.groupby("customer_id", sort=False)
    # expanding stats over previous transactions only (shift(1))
    prev_mean = g["amount"].transform(lambda s: s.shift(1).expanding().mean())
    prev_std = g["amount"].transform(lambda s: s.shift(1).expanding().std())
    global_mean = df["amount"].median()
    df["amount_to_customer_mean"] = (df["amount"] / prev_mean.fillna(global_mean)).clip(upper=100)
    df["amount_zscore"] = ((df["amount"] - prev_mean) / prev_std.replace(0, np.nan)).fillna(0).clip(-10, 10)
    df["customer_tx_index"] = g.cumcount()

    prev_ts = g["timestamp"].shift(1)
    df["minutes_since_prev"] = ((ts - prev_ts).dt.total_seconds() / 60).fillna(7 * 24 * 60).clip(upper=7 * 24 * 60)

    # velocity: number of the customer's transactions in the previous 1h / 24h (excluding current)
    df["_one"] = 1
    for window, col in (("1h", "tx_count_1h"), ("24h", "tx_count_24h")):
        counts = (df.set_index("timestamp").groupby("customer_id", sort=False)["_one"]
                  .rolling(window).sum().reset_index(level=0, drop=True))
        df[col] = (counts.values - 1)
    df = df.drop(columns="_one")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier transactions, test on later ones (mirrors real deployment)."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    df = df.sort_values("timestamp")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
