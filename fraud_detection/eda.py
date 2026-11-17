"""Exploratory-data-analysis helpers returning tidy DataFrames for plotting."""
from __future__ import annotations

import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    out = {"transactions": len(df), "customers": df["customer_id"].nunique(),
           "total_amount": float(df["amount"].sum()), "date_from": df["timestamp"].min(), "date_to": df["timestamp"].max(),
           "missing_values": int(df.isna().sum().sum())}
    if "is_fraud" in df.columns:
        out["fraud_count"] = int(df["is_fraud"].sum())
        out["fraud_rate"] = float(df["is_fraud"].mean())
        out["fraud_amount"] = float(df.loc[df["is_fraud"] == 1, "amount"].sum())
    return out


def fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["is_fraud"].agg(transactions="count", frauds="sum").reset_index()
    g["fraud_rate"] = g["frauds"] / g["transactions"]
    return g.sort_values("fraud_rate", ascending=False)


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(hour=df["timestamp"].dt.hour)
    return d.groupby(["hour", "is_fraud"]).size().rename("count").reset_index()


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(date=df["timestamp"].dt.date)
    return d.groupby("date").agg(transactions=("amount", "size"), frauds=("is_fraud", "sum"), amount=("amount", "sum")).reset_index()
