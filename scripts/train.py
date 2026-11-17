"""Train all models on a CSV, print the evaluation table and save the best model.

Usage: python scripts/train.py [path/to/transactions.csv]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraud_detection.data_loader import DEFAULT_DATA, load_csv  # noqa: E402
from fraud_detection.models import save_model, train_and_evaluate  # noqa: E402

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    df = load_csv(path, require_label=True)
    report = train_and_evaluate(df)
    print(f"train={report.train_size} test={report.test_size} test fraud rate={report.test_fraud_rate:.4f}")
    print(report.summary_table().to_string())
    supervised = {k: v for k, v in report.results.items() if "Isolation" not in k}
    best = max(supervised, key=lambda k: supervised[k].pr_auc)
    save_model(report.models[best], report.results[best].threshold, ROOT / "models" / "fraud_model.joblib")
    print(f"saved best supervised model by PR-AUC: {best}")
