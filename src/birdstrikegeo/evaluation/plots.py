"""
evaluation/plots.py
-----------------------
ROC curve, precision-recall curve, calibration curve, and training-loss
plots. Every plot generated from sample data gets a prominent
"SAMPLE DATA — NOT A REAL RESULT" watermark, per project requirements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def _watermark(fig, sample_data_flag: bool) -> None:
    if sample_data_flag:
        fig.text(
            0.5, 0.5, "SAMPLE DATA — NOT A REAL RESULT",
            fontsize=22, color="red", alpha=0.25, ha="center", va="center", rotation=30,
        )


def plot_roc_curve(y_true, y_proba, output_path: str | Path, sample_data_flag: bool, model_name: str = "") -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve{f' — {model_name}' if model_name else ''}")
    ax.legend()
    _watermark(fig, sample_data_flag)
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)


def plot_precision_recall_curve(y_true, y_proba, output_path: str | Path, sample_data_flag: bool, model_name: str = "") -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve{f' — {model_name}' if model_name else ''}")
    ax.legend()
    _watermark(fig, sample_data_flag)
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)


def plot_calibration_curve(calibration: dict, output_path: str | Path, sample_data_flag: bool, model_name: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(calibration["prob_pred"], calibration["prob_true"], marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve{f' — {model_name}' if model_name else ''}")
    ax.legend()
    _watermark(fig, sample_data_flag)
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)


def plot_training_loss(train_losses: list[float], val_losses: list[float], output_path: str | Path,
                        sample_data_flag: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Training loss")
    ax.plot(val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("DamageNet Training Loss")
    ax.legend()
    _watermark(fig, sample_data_flag)
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)
