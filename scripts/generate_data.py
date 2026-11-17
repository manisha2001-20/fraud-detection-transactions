"""Regenerate the synthetic sample dataset: python scripts/generate_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fraud_detection.data_generator import generate_transactions  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"
    df = generate_transactions(n_customers=400, days=60, fraud_rate=0.02, seed=42)
    df.to_csv(out, index=False)
    print(f"wrote {out} rows={len(df)} fraud_rate={df['is_fraud'].mean():.4f}")
