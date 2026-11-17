"""Loading and schema validation for transaction datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO

import pandas as pd

REQUIRED_COLUMNS = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "channel", "country",
                    "distance_from_home_km"}
LABEL = "is_fraud"
DEFAULT_DATA = Path(os.getenv("FRAUD_DATA_PATH") or Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv")


class DataValidationError(ValueError):
    """Raised when an uploaded/loaded dataset does not match the expected schema."""


def validate(df: pd.DataFrame, require_label: bool = False) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError("The dataset is empty.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if require_label and LABEL not in df.columns:
        missing.add(LABEL)
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["distance_from_home_km"] = pd.to_numeric(df["distance_from_home_km"], errors="coerce")
    bad = df["timestamp"].isna() | df["amount"].isna()
    if bad.all():
        raise DataValidationError("No rows have a valid timestamp and amount.")
    df = df.loc[~bad]
    if (df["amount"] < 0).any():
        raise DataValidationError("Amounts must be non-negative.")
    df["distance_from_home_km"] = df["distance_from_home_km"].fillna(df["distance_from_home_km"].median())
    for col in ("merchant_category", "channel", "country", "customer_id"):
        df[col] = df[col].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
    if LABEL in df.columns:
        lab = pd.to_numeric(df[LABEL], errors="coerce")
        if not lab.dropna().isin([0, 1]).all():
            raise DataValidationError("is_fraud must contain only 0/1.")
        df[LABEL] = lab.fillna(0).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_csv(source: str | Path | IO, require_label: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(source)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse CSV: {exc}") from exc
    return validate(df, require_label)
