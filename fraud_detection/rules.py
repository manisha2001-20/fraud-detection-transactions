"""Transparent rule-based flags for suspicious transactions (complements the ML models)."""
from __future__ import annotations

import pandas as pd

RULES = {
    "high_amount_vs_history": ("Amount ≥ 5× customer's past average", lambda d: d["amount_to_customer_mean"] >= 5),
    "night_foreign": ("Foreign transaction between 00:00–05:59", lambda d: (d["is_night"] == 1) & (d["is_foreign"] == 1)),
    "rapid_burst": ("≥ 3 transactions by the same customer within 1 hour", lambda d: d["tx_count_1h"] >= 2),
    "far_from_home": ("More than 500 km from usual location", lambda d: d["distance_from_home_km"] > 500),
    "risky_category_online": ("Online gift cards / gaming purchase ≥ 3× average",
                              lambda d: d["merchant_category"].isin(["gift_cards", "online_gaming"]) & (d["channel"] == "online") & (d["amount_to_customer_mean"] >= 3)),
}


def apply_rules(features: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean frame (one column per rule) plus `rule_hits` and `rule_reasons`."""
    out = pd.DataFrame(index=features.index)
    for name, (_, fn) in RULES.items():
        out[name] = fn(features).astype(bool)
    out["rule_hits"] = out[list(RULES)].sum(axis=1)
    out["rule_reasons"] = out[list(RULES)].apply(lambda r: ", ".join(RULES[k][0] for k in RULES if r[k]), axis=1)
    return out
