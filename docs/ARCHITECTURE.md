# Architecture — Fraud Detection in Financial Transactions

```
CSV / synthetic generator
        │
        ▼
 data_loader.validate ──► DataValidationError (shown in the UI)
        │
        ▼
 preprocessing.engineer_features   (per-customer history uses shift(1): past only)
        │
        ├──► rules.apply_rules ───────────────► Suspicious-transactions page
        ├──► eda.* ───────────────────────────► Overview / EDA pages
        ▼
 time_based_split (train = earlier 75 %, test = later 25 %)
        │
        ├─ train → fit-part (80 %) + validation-part (20 %, later in time)
        │     ├─ fit each classifier on fit-part → pick F1-optimal threshold on validation-part
        │     └─ refit on full train
        ├─ IsolationForest fit on train features WITHOUT labels
        ▼
 evaluation.evaluate_scores on test → PR/ROC curves, confusion, PR-AUC …
        │
        ▼
 Streamlit dashboard (app.py)  ·  scripts/train.py → models/fraud_model.joblib
```

## Key design decisions

| Decision | Reason |
|---|---|
| **Time-based split** instead of a random split | Fraud models are deployed on future data. A random split leaks future behaviour into training and inflates scores. |
| **Past-only customer features** (`shift(1)` + expanding stats, rolling windows minus the current transaction) | Prevents target leakage. `tests/test_fraud_detection.py::test_features_have_no_lookahead` checks this. |
| **PR-AUC as the headline metric** | With about 2% positives, ROC-AUC and accuracy look high even for weak models. |
| **Threshold tuned on a validation slice** | Picking it on the test set would report optimistic numbers. |
| **Class weighting** instead of oversampling | Simple and leakage-free, and needs no extra dependency (e.g. imbalanced-learn). |
| **Rules + ML + anomaly detection** | Rules are explainable, supervised models catch known patterns, and Isolation Forest needs no labels (useful for new fraud types). |
| **Synthetic data with deliberate hard cases and label noise** | Lets the project run anywhere without sharing real financial data, while avoiding an unrealistically separable problem. |

## Features

| Feature | Meaning |
|---|---|
| `log_amount` | log(1 + amount) |
| `hour`, `is_night`, `is_weekend` | Time context |
| `is_foreign`, `distance_from_home_km` | Location context |
| `amount_to_customer_mean`, `amount_zscore` | Deviation from the customer's own past spending |
| `tx_count_1h`, `tx_count_24h`, `minutes_since_prev` | Velocity / burst behaviour |
| `customer_tx_index` | How much history the customer has |
| `merchant_category`, `channel` | One-hot encoded |

## Error handling

- Invalid uploads (missing columns, empty file, negative amounts, non-0/1 labels, unparseable CSV) raise `DataValidationError` and show a clear message in the UI.
- Rows with unparseable timestamps or amounts are dropped. Missing distances are filled with the median.
- Training refuses data where either split lacks a class.
- Unknown categories at scoring time are ignored by `OneHotEncoder(handle_unknown="ignore")`.

## Limitations

- Results on synthetic data do not transfer to real data. The public Kaggle "Credit Card Fraud" dataset uses PCA-anonymised features with a different schema; supporting it would need a separate feature path.
- Thresholds are optimised for F1. A real deployment would weigh the cost of a missed fraud against the cost of a false alarm.
