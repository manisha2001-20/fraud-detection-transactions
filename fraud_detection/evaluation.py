"""Evaluation metrics suited to heavily imbalanced fraud data (PR-AUC is the headline metric)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


@dataclass
class EvaluationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray  # [[tn, fp], [fn, tp]]
    roc: tuple[np.ndarray, np.ndarray]
    pr: tuple[np.ndarray, np.ndarray]
    base_rate: float

    def as_row(self, name: str) -> dict:
        tn, fp, fn, tp = self.confusion.ravel()
        return {"Model": name, "PR-AUC": round(self.pr_auc, 4), "ROC-AUC": round(self.roc_auc, 4),
                "Precision": round(self.precision, 4), "Recall": round(self.recall, 4), "F1": round(self.f1, 4),
                "Threshold": round(self.threshold, 4), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> EvaluationResult:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation needs both classes in y_true.")
    pred = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    p, r, _ = precision_recall_curve(y_true, scores)
    return EvaluationResult(
        threshold=float(threshold),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        confusion=confusion_matrix(y_true, pred, labels=[0, 1]),
        roc=(fpr, tpr), pr=(r, p), base_rate=float(y_true.mean()),
    )
