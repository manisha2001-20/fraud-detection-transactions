"""Streamlit dashboard: Fraud Detection in Financial Transactions.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_detection import eda
from fraud_detection.data_generator import generate_transactions
from fraud_detection.data_loader import DEFAULT_DATA, DataValidationError, load_csv
from fraud_detection.models import FEATURES, score_transactions, train_and_evaluate
from fraud_detection.preprocessing import engineer_features
from fraud_detection.rules import RULES, apply_rules

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

PALETTE = {"legit": "#3b82f6", "fraud": "#ef4444", "accent": "#8b5cf6", "muted": "#94a3b8"}
MODEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1300px;}
  div[data-testid="stMetric"] {background: var(--secondary-background-color); border-radius: 12px; padding: 14px 16px;
     border: 1px solid rgba(128,128,128,.18);}
  div[data-testid="stMetricLabel"] p {font-size: .85rem; opacity: .75;}
  h1 {font-weight: 700; letter-spacing: -.01em;}
  .note {font-size: .85rem; opacity: .75;}
</style>""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      font=dict(family="Inter, Segoe UI, sans-serif"), template="plotly_white")
    return fig


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading dataset…")
def load_default() -> pd.DataFrame:
    if DEFAULT_DATA.exists():
        return load_csv(DEFAULT_DATA, require_label=True)
    return generate_transactions()


@st.cache_data(show_spinner="Engineering features…")
def features_for(df: pd.DataFrame) -> pd.DataFrame:
    f = engineer_features(df)
    return pd.concat([f, apply_rules(f)], axis=1)


@st.cache_resource(show_spinner="Training models on the earlier 75% of transactions…")
def train(df: pd.DataFrame, test_fraction: float):
    return train_and_evaluate(df, test_fraction=test_fraction)


with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    PAGES = ["Overview", "Exploratory analysis", "Suspicious transactions", "Model evaluation", "Score new transactions"]
    # deep links: ?page=2 opens "Suspicious transactions"
    qp = st.query_params.get("page", "0")
    start = int(qp) if qp.isdigit() and int(qp) < len(PAGES) else 0
    page = st.radio("Navigate", PAGES, index=start, label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset**")
    upload = st.file_uploader("Upload labelled CSV (optional)", type="csv",
                              help="Columns: transaction_id, customer_id, timestamp, amount, merchant_category, channel, "
                                   "country, distance_from_home_km, is_fraud")
    st.caption("Default: synthetic sample data (`data/sample_transactions.csv`). Not real customer data.")

try:
    data = load_csv(upload, require_label=True) if upload else load_default()
except DataValidationError as exc:
    st.error(f"Uploaded file rejected: {exc}")
    st.stop()

feats = features_for(data)
ov = eda.overview(data)

# ------------------------------------------------------------- pages
if page == "Overview":
    st.title("Fraud Detection in Financial Transactions")
    st.markdown('<p class="note">Data analysis, rule-based screening, supervised classification and unsupervised '
                'anomaly detection on transaction data.</p>', unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Transactions", f"{ov['transactions']:,}")
    c[1].metric("Customers", f"{ov['customers']:,}")
    c[2].metric("Fraud cases", f"{ov['fraud_count']:,}")
    c[3].metric("Fraud rate", f"{ov['fraud_rate']:.2%}")
    c[4].metric("Fraud amount", f"₹{ov['fraud_amount']:,.0f}")

    daily = eda.daily_series(data)
    left, right = st.columns([2, 1])
    fig = go.Figure()
    fig.add_bar(x=daily["date"], y=daily["transactions"], name="Transactions", marker_color=PALETTE["legit"], opacity=.55)
    fig.add_scatter(x=daily["date"], y=daily["frauds"], name="Fraud cases", yaxis="y2", line=dict(color=PALETTE["fraud"], width=2.5))
    fig.update_layout(title="Daily volume and fraud cases", yaxis=dict(title="Transactions"),
                      yaxis2=dict(title="Fraud", overlaying="y", side="right", showgrid=False))
    left.plotly_chart(style_fig(fig), use_container_width=True)
    pie = px.pie(names=["Legitimate", "Fraud"], values=[ov["transactions"] - ov["fraud_count"], ov["fraud_count"]],
                 hole=.62, color_discrete_sequence=[PALETTE["legit"], PALETTE["fraud"]], title="Class balance")
    right.plotly_chart(style_fig(pie), use_container_width=True)
    st.info("Fraud is rare, so accuracy is misleading: a model that predicts *legitimate* for everything would be "
            f"{1 - ov['fraud_rate']:.1%} accurate. This dashboard reports PR-AUC, precision and recall instead.")

elif page == "Exploratory analysis":
    st.title("Exploratory data analysis")
    t1, t2, t3 = st.tabs(["Patterns", "Distributions", "Data quality"])
    with t1:
        a, b = st.columns(2)
        cat = eda.fraud_rate_by(data, "merchant_category")
        a.plotly_chart(style_fig(px.bar(cat, x="fraud_rate", y="merchant_category", orientation="h", title="Fraud rate by merchant category",
                                        color="fraud_rate", color_continuous_scale="Reds", labels={"fraud_rate": "Fraud rate", "merchant_category": ""})
                                 .update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))), use_container_width=True)
        hourly = eda.hourly_profile(data)
        hourly["share"] = hourly["count"] / hourly.groupby("is_fraud")["count"].transform("sum")
        hourly["class"] = hourly["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        b.plotly_chart(style_fig(px.line(hourly, x="hour", y="share", color="class", markers=True, title="Share of transactions by hour",
                                         color_discrete_map={"Legitimate": PALETTE["legit"], "Fraud": PALETTE["fraud"]},
                                         labels={"share": "Share of class", "hour": "Hour of day", "class": ""})), use_container_width=True)
        a, b = st.columns(2)
        ch = eda.fraud_rate_by(data, "channel")
        a.plotly_chart(style_fig(px.bar(ch, x="channel", y="fraud_rate", title="Fraud rate by channel", color_discrete_sequence=[PALETTE["accent"]],
                                        labels={"fraud_rate": "Fraud rate", "channel": ""})), use_container_width=True)
        f2 = feats.assign(foreign=feats["is_foreign"].map({0: "Domestic", 1: "Foreign"}))
        fr = f2.groupby("foreign")["is_fraud"].mean().reset_index()
        b.plotly_chart(style_fig(px.bar(fr, x="foreign", y="is_fraud", title="Fraud rate: domestic vs foreign", color_discrete_sequence=[PALETTE["fraud"]],
                                        labels={"is_fraud": "Fraud rate", "foreign": ""})), use_container_width=True)
    with t2:
        a, b = st.columns(2)
        d = feats.assign(cls=feats["is_fraud"].map({0: "Legitimate", 1: "Fraud"}))
        a.plotly_chart(style_fig(px.histogram(d, x="log_amount", color="cls", histnorm="probability density", nbins=60, barmode="overlay",
                                              opacity=.6, title="log(1 + amount)", color_discrete_map={"Legitimate": PALETTE["legit"], "Fraud": PALETTE["fraud"]},
                                              labels={"cls": ""})), use_container_width=True)
        b.plotly_chart(style_fig(px.box(d, x="cls", y="amount_to_customer_mean", color="cls", points=False, log_y=True,
                                        title="Amount ÷ customer's past average (log scale)", color_discrete_map={"Legitimate": PALETTE["legit"], "Fraud": PALETTE["fraud"]},
                                        labels={"cls": "", "amount_to_customer_mean": "Ratio"})), use_container_width=True)
        num = ["log_amount", "hour", "distance_from_home_km", "amount_to_customer_mean", "amount_zscore", "tx_count_1h",
               "tx_count_24h", "minutes_since_prev", "is_foreign", "is_night", "is_fraud"]
        corr = feats[num].corr()
        st.plotly_chart(style_fig(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                            title="Feature correlation matrix"), 520), use_container_width=True)
    with t3:
        st.write("Rows:", len(data), "· Missing values after validation:", ov["missing_values"])
        st.markdown("**Column profile**")
        st.dataframe(eda.profile(data).astype(str), use_container_width=True, hide_index=True)
        st.dataframe(data.head(50), use_container_width=True, hide_index=True)

elif page == "Suspicious transactions":
    st.title("Rule-based suspicious transaction screening")
    st.caption("Transparent rules that an analyst can read, used alongside the ML models.")
    rows = []
    for name, (desc, _) in RULES.items():
        hit = feats[name]
        rows.append({"Rule": desc, "Flagged": int(hit.sum()), "Precision": (feats.loc[hit, "is_fraud"].mean() if hit.any() else np.nan),
                     "Fraud caught": feats.loc[hit, "is_fraud"].sum() / max(1, feats["is_fraud"].sum())})
    rt = pd.DataFrame(rows)
    st.dataframe(rt.style.format({"Precision": "{:.1%}", "Fraud caught": "{:.1%}"}), use_container_width=True, hide_index=True)
    min_hits = st.slider("Minimum number of rules triggered", 1, len(RULES), 2)
    flagged = feats[feats["rule_hits"] >= min_hits]
    c = st.columns(3)
    c[0].metric("Flagged transactions", f"{len(flagged):,}")
    c[1].metric("Of which actually fraud", f"{flagged['is_fraud'].mean():.1%}" if len(flagged) else "—")
    c[2].metric("Share of all fraud caught", f"{flagged['is_fraud'].sum() / max(1, feats['is_fraud'].sum()):.1%}")
    sample = feats.sample(min(len(feats), 8000), random_state=1)
    sample = sample.assign(rules=sample["rule_hits"].clip(upper=3).astype(str)).sort_values("rules")
    st.plotly_chart(style_fig(px.scatter(sample, x="hour", y="amount", log_y=True,
                                         color="rules", title="Transactions by hour and amount (coloured by rules triggered)",
                                         color_discrete_sequence=["#cbd5e1", "#fbbf24", "#f97316", "#dc2626"], labels={"rules": "Rules triggered"}, opacity=.6), 420),
                    use_container_width=True)
    st.dataframe(flagged.sort_values(["rule_hits", "amount"], ascending=False)[
        ["transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "channel", "country", "rule_hits", "rule_reasons", "is_fraud"]].head(300),
        use_container_width=True, hide_index=True)

elif page == "Model evaluation":
    st.title("Machine-learning models")
    test_fraction = st.select_slider("Test set = most recent share of transactions", [0.2, 0.25, 0.3], value=0.25)
    try:
        report = train(data, test_fraction)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"Time-based split: train on {report.train_size:,} earlier transactions (fraud rate {report.train_fraud_rate:.2%}), "
               f"test on {report.test_size:,} later ones (fraud rate {report.test_fraud_rate:.2%}). Thresholds are tuned on a validation "
               "slice of the training period, never on the test set. All numbers below are computed live.")
    table = report.summary_table()
    st.dataframe(table.style.highlight_max(subset=["PR-AUC", "ROC-AUC", "F1"], color="#d1fae5")
                 .format({c: "{:.3f}" for c in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Threshold"]}), use_container_width=True)
    a, b = st.columns(2)
    pr, roc = go.Figure(), go.Figure()
    for (name, r), col in zip(report.results.items(), MODEL_COLORS):
        pr.add_scatter(x=r.pr[0], y=r.pr[1], name=f"{name} (AP {r.pr_auc:.3f})", line=dict(color=col))
        roc.add_scatter(x=r.roc[0], y=r.roc[1], name=f"{name} ({r.roc_auc:.3f})", line=dict(color=col))
    pr.add_hline(y=report.test_fraud_rate, line_dash="dot", line_color=PALETTE["muted"], annotation_text="random baseline", annotation_position="top left")
    roc.add_scatter(x=[0, 1], y=[0, 1], name="random", line=dict(dash="dot", color=PALETTE["muted"]))
    pr.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    a.plotly_chart(style_fig(pr, 420), use_container_width=True)
    b.plotly_chart(style_fig(roc, 420), use_container_width=True)

    a, b = st.columns(2)
    chosen = a.selectbox("Confusion matrix for", list(report.results))
    cm = report.results[chosen].confusion
    a.plotly_chart(style_fig(px.imshow(cm, text_auto=True, x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                                       color_continuous_scale="Blues", title=f"{chosen} @ threshold {report.results[chosen].threshold:.3f}")
                             .update_layout(coloraxis_showscale=False)), use_container_width=True)
    if report.feature_importance is not None:
        fi = report.feature_importance.head(12)[::-1]
        b.plotly_chart(style_fig(px.bar(x=fi.values, y=fi.index, orientation="h", title="Random Forest feature importance",
                                        color_discrete_sequence=[PALETTE["accent"]], labels={"x": "Importance", "y": ""})), use_container_width=True)

    st.subheader("Threshold trade-off")
    model_name = st.selectbox("Model", [m for m in report.results if "Isolation" not in m])
    scores = report.test_frame[f"score::{model_name}"].to_numpy()
    y = report.test_frame["is_fraud"].to_numpy()
    thr = st.slider("Decision threshold", 0.01, 0.99, float(round(report.results[model_name].threshold, 2)), 0.01)
    pred = scores >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    c = st.columns(4)
    c[0].metric("Alerts raised", f"{pred.sum():,}")
    c[1].metric("Fraud caught (recall)", f"{tp / max(1, y.sum()):.1%}")
    c[2].metric("Alert precision", f"{tp / max(1, pred.sum()):.1%}")
    c[3].metric("False alarms", f"{fp:,}")

else:  # Score new transactions
    st.title("Score new transactions")
    st.write("Upload transactions **without** labels. The best supervised model (by PR-AUC on the held-out test set) scores each one.")
    report = train(data, 0.25)
    supervised = {k: v for k, v in report.results.items() if "Isolation" not in k}
    best = max(supervised, key=lambda k: supervised[k].pr_auc)
    st.caption(f"Using **{best}** at threshold {supervised[best].threshold:.3f}.")
    new_file = st.file_uploader("Transactions CSV", type="csv", key="score")
    if st.button("Or use a fresh synthetic batch (different random seed)"):
        st.session_state["batch"] = generate_transactions(n_customers=60, days=7, seed=int(np.random.SeedSequence().entropy % 10_000)).drop(columns="is_fraud")
    batch = None
    try:
        if new_file:
            batch = load_csv(new_file)
        elif "batch" in st.session_state:
            batch = st.session_state["batch"]
    except DataValidationError as exc:
        st.error(f"File rejected: {exc}")
    if batch is not None:
        scored = score_transactions({"model": report.models[best], "threshold": supervised[best].threshold,
                                     "features": FEATURES}, batch)
        scored = pd.concat([scored, apply_rules(scored)[["rule_hits", "rule_reasons"]]], axis=1)
        c = st.columns(3)
        c[0].metric("Transactions scored", f"{len(scored):,}")
        c[1].metric("Predicted fraud", f"{int(scored['predicted_fraud'].sum()):,}")
        c[2].metric("Mean fraud probability", f"{scored['fraud_probability'].mean():.2%}")
        st.plotly_chart(style_fig(px.histogram(scored, x="fraud_probability", nbins=40, log_y=True, title="Fraud probability distribution",
                                               color_discrete_sequence=[PALETTE["fraud"]])), use_container_width=True)
        cols = ["transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "channel", "country",
                "fraud_probability", "predicted_fraud", "rule_reasons"]
        out = scored.sort_values("fraud_probability", ascending=False)[cols]
        st.dataframe(out, use_container_width=True, hide_index=True,
                     column_config={"fraud_probability": st.column_config.ProgressColumn("Fraud probability", min_value=0.0, max_value=1.0, format="%.3f")})
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download scored CSV", buf.getvalue(), "scored_transactions.csv", "text/csv")
