"""
evaluator.py
------------
Computes evaluation metrics and renders the confusion matrix / ROC curve
plots used in the project's evaluation report.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Evaluator:
    """Computes and stores evaluation metrics for a trained model."""

    def __init__(self, y_true, y_pred, y_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def compute_metrics(self) -> dict:
        return {
            "accuracy": round(accuracy_score(self.y_true, self.y_pred), 4),
            "precision": round(precision_score(self.y_true, self.y_pred), 4),
            "recall": round(recall_score(self.y_true, self.y_pred), 4),
            "f1_score": round(f1_score(self.y_true, self.y_pred), 4),
            "roc_auc": round(roc_auc_score(self.y_true, self.y_proba), 4),
        }

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self, out_path: str | Path) -> None:
        cm = self.confusion_matrix()
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticklabels(["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_roc_curve(self, out_path: str | Path) -> None:
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#2563eb", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
