"""Supervised classifiers + unsupervised anomaly detection, trained and evaluated on a time-based split."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.evaluation import EvaluationResult, best_f1_threshold, evaluate_scores
from fraud_detection.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor,
                                           engineer_features, time_based_split)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_models(seed: int = 42) -> dict[str, Pipeline]:
    pre = build_preprocessor
    return {
        "Logistic Regression": Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", pre()), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))]),
        "Gradient Boosting": Pipeline([("pre", pre()), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed))]),
    }


@dataclass
class TrainingReport:
    results: dict[str, EvaluationResult]
    test_frame: pd.DataFrame                       # test rows with per-model scores
    feature_importance: pd.Series | None
    train_size: int
    test_size: int
    train_fraud_rate: float
    test_fraud_rate: float
    models: dict = field(default_factory=dict)

    @property
    def best_model(self) -> str:
        return max(self.results, key=lambda k: self.results[k].pr_auc)

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row(name) for name, r in self.results.items()]).set_index("Model")


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42,
                       model_names: list[str] | None = None) -> TrainingReport:
    if "is_fraud" not in df.columns:
        raise ValueError("Training requires an is_fraud label column.")
    feats = engineer_features(df)
    train, test = time_based_split(feats, test_fraction)
    if train["is_fraud"].nunique() < 2 or test["is_fraud"].nunique() < 2:
        raise ValueError("Both train and test splits need fraud and non-fraud examples; use more data.")

    candidates = make_models(seed)
    if model_names:
        candidates = {k: v for k, v in candidates.items() if k in model_names}
    results, fitted = {}, {}
    test = test.copy()
    # decision threshold is tuned on a later validation slice of the TRAINING period, never on test
    fit_part, val_part = time_based_split(train, 0.2)
    if val_part["is_fraud"].nunique() < 2:
        fit_part, val_part = train, train
    for name, model in candidates.items():
        model.fit(fit_part[FEATURES], fit_part["is_fraud"])
        thr = best_f1_threshold(val_part["is_fraud"].to_numpy(), model.predict_proba(val_part[FEATURES])[:, 1])
        model.fit(train[FEATURES], train["is_fraud"])  # refit on the full training period
        scores = model.predict_proba(test[FEATURES])[:, 1]
        results[name] = evaluate_scores(test["is_fraud"].to_numpy(), scores, thr)
        test[f"score::{name}"] = scores
        fitted[name] = model

    # unsupervised anomaly detection: trained WITHOUT labels on (mostly legitimate) training data
    iso = Pipeline([("pre", build_preprocessor()), ("iso", IsolationForest(n_estimators=300, contamination="auto", random_state=seed))])
    iso.fit(train[FEATURES])
    raw = -iso.decision_function(test[FEATURES])  # higher = more anomalous
    anomaly = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    train_raw = -iso.decision_function(train[FEATURES])
    contamination = float(train["is_fraud"].mean())
    iso_thr_raw = np.quantile(train_raw, 1 - contamination)
    iso_thr = float((iso_thr_raw - raw.min()) / (raw.max() - raw.min() + 1e-12))
    results["Isolation Forest (unsupervised)"] = evaluate_scores(test["is_fraud"].to_numpy(), anomaly, iso_thr)
    test["score::Isolation Forest (unsupervised)"] = anomaly
    fitted["Isolation Forest (unsupervised)"] = iso

    importance = None
    if "Random Forest" in fitted:
        rf = fitted["Random Forest"]
        names = rf.named_steps["pre"].get_feature_names_out()
        importance = pd.Series(rf.named_steps["clf"].feature_importances_, index=[n.split("__", 1)[-1] for n in names]).sort_values(ascending=False)

    return TrainingReport(results, test, importance, len(train), len(test), float(train["is_fraud"].mean()),
                          float(test["is_fraud"].mean()), fitted)


def save_model(model, threshold: float, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES}, path)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def score_transactions(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    feats = engineer_features(df)
    feats["fraud_probability"] = bundle["model"].predict_proba(feats[bundle["features"]])[:, 1]
    feats["predicted_fraud"] = (feats["fraud_probability"] >= bundle["threshold"]).astype(int)
    return feats
