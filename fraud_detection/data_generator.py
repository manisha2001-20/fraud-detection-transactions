"""Generates a SYNTHETIC, labelled financial-transaction dataset with realistic fraud patterns.

This is not real customer data. Fraud is injected with known patterns (night-time spending,
foreign transactions, bursts of rapid transactions, unusual merchant categories, amounts far above
the customer's normal) plus label noise, so the classification problem is not trivially separable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "restaurant", "fuel", "utilities", "clothing", "travel", "electronics", "gift_cards", "online_gaming", "pharmacy"]
NORMAL_CAT_P = [0.25, 0.17, 0.12, 0.10, 0.10, 0.05, 0.06, 0.02, 0.03, 0.10]
FRAUD_CAT_P = [0.05, 0.05, 0.03, 0.01, 0.08, 0.12, 0.25, 0.22, 0.14, 0.05]
CHANNELS = ["pos", "online", "atm"]
COUNTRIES = ["IN", "IN", "IN", "IN", "AE", "US", "GB", "SG", "NG", "RU"]


def generate_transactions(n_customers: int = 400, days: int = 60, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be between 0 and 0.5")
    if n_customers < 10 or days < 1:
        raise ValueError("need at least 10 customers and 1 day")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    rows = []

    for c in range(n_customers):
        cid = f"C{c:05d}"
        typical = rng.lognormal(mean=6.2, sigma=0.6)  # customer's typical spend (INR)
        rate = rng.uniform(0.4, 3.0)                   # transactions per day
        n = max(1, rng.poisson(rate * days))
        home_lat, home_lon = rng.uniform(8, 30), rng.uniform(70, 88)
        times = start + pd.to_timedelta(np.sort(rng.uniform(0, days * 24 * 3600, n)), unit="s")
        for t in times:
            hour_shift = rng.normal(14, 4)  # daytime-centred
            t = t.normalize() + pd.Timedelta(hours=float(np.clip(hour_shift, 0, 23.9)))
            rows.append({
                "customer_id": cid, "timestamp": t,
                "amount": round(float(typical * rng.lognormal(0, 0.5)), 2),
                "merchant_category": rng.choice(CATEGORIES, p=NORMAL_CAT_P),
                "channel": rng.choice(CHANNELS, p=[0.6, 0.35, 0.05]),
                "country": "IN" if rng.random() > 0.03 else rng.choice(COUNTRIES[4:]),
                # ~6% of legitimate spending happens while travelling
                "distance_from_home_km": round(float(abs(rng.normal(5, 8)) if rng.random() > 0.06 else rng.uniform(100, 2500)), 1),
                "is_fraud": 0,
                "_home": (home_lat, home_lon), "_typical": typical,
            })

    df = pd.DataFrame(rows)
    n_fraud_events = int(len(df) * fraud_rate / 1.8)  # bursts add ~1.8 tx per event on average
    victims = rng.choice(df["customer_id"].unique(), size=n_fraud_events, replace=True)
    typical_by_c = df.groupby("customer_id")["_typical"].first()
    fraud_rows = []
    for cid in victims:
        t0 = start + pd.Timedelta(seconds=float(rng.uniform(0, days * 24 * 3600)))
        if rng.random() < 0.40:
            t0 = t0.normalize() + pd.Timedelta(hours=float(rng.uniform(0, 5)))  # night
        burst = 1 + rng.poisson(0.8)
        country = rng.choice(COUNTRIES[4:]) if rng.random() < 0.30 else "IN"
        channel = "online" if rng.random() < 0.6 else rng.choice(["pos", "atm"])
        local_skim = country == "IN" and rng.random() < 0.5  # cloned card used near the victim's home
        for k in range(burst):
            mult = rng.lognormal(1.0, 0.8) if rng.random() < 0.6 else rng.lognormal(0, 0.5)  # much fraud is small
            fraud_rows.append({
                "customer_id": cid, "timestamp": t0 + pd.Timedelta(minutes=float(k * rng.uniform(1, 12))),
                "amount": round(float(typical_by_c[cid] * mult), 2),
                "merchant_category": rng.choice(CATEGORIES, p=FRAUD_CAT_P if rng.random() < 0.6 else NORMAL_CAT_P),
                "channel": channel, "country": country,
                "distance_from_home_km": round(float(abs(rng.normal(5, 8)) if local_skim else abs(rng.normal(400 if country != "IN" else 120, 300))), 1),
                "is_fraud": 1, "_home": None, "_typical": typical_by_c[cid],
            })
    df = pd.concat([df, pd.DataFrame(fraud_rows)], ignore_index=True)

    # label noise: a few legitimate transactions look suspicious (travel, big purchases)
    legit = df.index[df["is_fraud"] == 0]
    odd = rng.choice(legit, size=int(len(legit) * 0.02), replace=False)
    df.loc[odd, "amount"] = (df.loc[odd, "amount"] * rng.uniform(3, 8, len(odd))).round(2)
    df.loc[odd[: len(odd) // 2], "country"] = rng.choice(COUNTRIES[4:], len(odd) // 2)

    # a small share of labels is wrong, as in real chargeback data (missed or mis-reported fraud)
    flip = rng.choice(df.index, size=int(len(df) * 0.002), replace=False)
    df.loc[flip, "is_fraud"] = 1 - df.loc[flip, "is_fraud"]

    df = df.drop(columns=["_home", "_typical"]).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    return df


if __name__ == "__main__":  # pragma: no cover
    d = generate_transactions()
    print(d.shape, d["is_fraud"].mean())
