import io

import numpy as np
import pandas as pd
import pytest

from fraud_detection import eda
from fraud_detection.data_generator import generate_transactions
from fraud_detection.data_loader import DataValidationError, load_csv, validate
from fraud_detection.evaluation import best_f1_threshold, evaluate_scores
from fraud_detection.models import FEATURES, load_model, save_model, score_transactions, train_and_evaluate
from fraud_detection.preprocessing import engineer_features, time_based_split
from fraud_detection.rules import RULES, apply_rules


@pytest.fixture(scope="module")
def data():
    return validate(generate_transactions(n_customers=120, days=30, fraud_rate=0.04, seed=7), require_label=True)


@pytest.fixture(scope="module")
def report(data):
    return train_and_evaluate(data, model_names=["Logistic Regression", "Random Forest"])


def test_generator_is_deterministic_and_labelled():
    a = generate_transactions(n_customers=20, days=5, seed=1)
    b = generate_transactions(n_customers=20, days=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["is_fraud"].unique()) <= {0, 1}
    assert a["transaction_id"].is_unique


@pytest.mark.parametrize("kwargs", [{"fraud_rate": 0}, {"fraud_rate": 0.9}, {"n_customers": 2}])
def test_generator_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**kwargs)


def test_validate_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate(pd.DataFrame({"amount": [1.0]}))


def test_validate_rejects_negative_amount(data):
    bad = data.head(5).copy()
    bad.loc[bad.index[0], "amount"] = -5
    with pytest.raises(DataValidationError, match="non-negative"):
        validate(bad)


def test_validate_rejects_bad_labels(data):
    bad = data.head(5).copy()
    bad["is_fraud"] = 3
    with pytest.raises(DataValidationError, match="0/1"):
        validate(bad)


def test_validate_drops_unparseable_rows(data):
    d = data.head(10).copy().astype({"timestamp": str})
    d.loc[d.index[0], "timestamp"] = "not a date"
    assert len(validate(d)) == 9


def test_load_csv_roundtrip_and_empty():
    df = generate_transactions(n_customers=15, days=3, seed=2)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    assert len(load_csv(buf, require_label=True)) == len(df)
    with pytest.raises(DataValidationError):
        validate(df.iloc[0:0])


def test_features_have_no_lookahead():
    df = pd.DataFrame({
        "transaction_id": ["a", "b", "c"], "customer_id": ["x", "x", "x"],
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:20", "2026-01-02 11:00"]),
        "amount": [100.0, 100.0, 1000.0], "merchant_category": ["grocery"] * 3, "channel": ["pos"] * 3,
        "country": ["IN", "IN", "US"], "distance_from_home_km": [1.0, 1.0, 900.0]})
    f = engineer_features(df).set_index("transaction_id")
    assert f.loc["a", "customer_tx_index"] == 0
    assert f.loc["b", "tx_count_1h"] == 1          # one previous tx within the hour
    assert f.loc["c", "tx_count_24h"] == 0         # previous ones are >24h earlier
    assert f.loc["c", "amount_to_customer_mean"] == pytest.approx(10.0)  # uses only past mean (100)
    assert f.loc["c", "is_foreign"] == 1
    assert f.loc["b", "minutes_since_prev"] == pytest.approx(20)


def test_time_split_is_chronological(data):
    feats = engineer_features(data)
    train, test = time_based_split(feats, 0.25)
    assert train["timestamp"].max() <= test["timestamp"].min()
    with pytest.raises(ValueError):
        time_based_split(feats, 1.5)


def test_rules_flag_expected_cases():
    f = pd.DataFrame({"amount_to_customer_mean": [6, 1], "is_night": [1, 0], "is_foreign": [1, 0], "tx_count_1h": [0, 0],
                      "distance_from_home_km": [800, 2], "merchant_category": ["gift_cards", "grocery"], "channel": ["online", "pos"]})
    r = apply_rules(f)
    assert r.loc[0, "rule_hits"] == 4 and r.loc[1, "rule_hits"] == 0
    assert r.loc[1, "rule_reasons"] == ""
    assert set(RULES) <= set(r.columns)


def test_metrics_on_known_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    r = evaluate_scores(y, s, 0.3)
    assert r.roc_auc == pytest.approx(0.75)
    assert r.recall == 1.0 and r.precision == pytest.approx(2 / 3)
    assert r.confusion.tolist() == [[1, 1], [0, 2]]
    assert 0 < best_f1_threshold(y, s) <= 0.8
    with pytest.raises(ValueError):
        evaluate_scores(np.zeros(3), np.zeros(3), 0.5)


def test_models_beat_random_baseline(report):
    # sanity check, not a performance claim: learned models must rank fraud above chance
    for name, res in report.results.items():
        assert res.roc_auc > 0.6, name
        assert res.pr_auc > res.base_rate, name
    assert {"Logistic Regression", "Random Forest", "Isolation Forest (unsupervised)"} == set(report.results)
    assert report.feature_importance is not None and len(report.feature_importance) > 5
    assert report.summary_table().shape[0] == 3


def test_training_requires_label(data):
    with pytest.raises(ValueError):
        train_and_evaluate(data.drop(columns="is_fraud"))


def test_save_load_and_score(tmp_path, report, data):
    path = tmp_path / "m.joblib"
    save_model(report.models["Random Forest"], 0.5, path)
    bundle = load_model(path)
    scored = score_transactions(bundle, data.drop(columns="is_fraud").head(200))
    assert scored["fraud_probability"].between(0, 1).all()
    assert set(scored["predicted_fraud"].unique()) <= {0, 1}
    assert bundle["features"] == FEATURES


def test_eda_helpers(data):
    ov = eda.overview(data)
    assert ov["transactions"] == len(data) and 0 < ov["fraud_rate"] < 1
    by_cat = eda.fraud_rate_by(data, "merchant_category")
    assert by_cat["transactions"].sum() == len(data)
    assert eda.hourly_profile(data)["count"].sum() == len(data)
    assert eda.daily_series(data)["frauds"].sum() == data["is_fraud"].sum()


def test_profile(data):
    prof = eda.profile(data).set_index("column")
    assert set(data.columns) == set(prof.index)
    assert prof.loc["amount", "missing"] == 0 and prof.loc["amount", "min"] >= 0
    assert prof.loc["is_fraud", "unique"] == 2
    assert prof.loc["merchant_category", "top_value"] in set(data["merchant_category"])
    d = data.head(10).copy()
    d.loc[d.index[0], "distance_from_home_km"] = None
    assert eda.profile(d).set_index("column").loc["distance_from_home_km", "missing"] == 1
