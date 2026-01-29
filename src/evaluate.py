import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from src.data import make_generators
from src.utils import ensure_dir

def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="artifacts/model.keras")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    ensure_dir("reports")

    img_size = (args.img_size, args.img_size)
    _, _, test_gen = make_generators(args.data_dir, img_size=img_size, batch_size=args.batch_size)

    model = tf.keras.models.load_model(args.model_path)

    y_true = test_gen.classes
    probs = model.predict(test_gen, verbose=0).ravel()
    y_pred = (probs >= 0.5).astype(int)

    # Confusion matrix and clinical metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)   # recall positive
    specificity = tn / (tn + fp + 1e-9)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "sensitivity_recall_pos": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    save_plot("reports/roc_curve.png")

    # Plot PR
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={pr_auc:.3f})")
    save_plot("reports/pr_curve.png")

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    save_plot("reports/confusion_matrix.png")

    print("Saved reports to /reports:")
    print("- metrics.json")
    print("- roc_curve.png")
    print("- pr_curve.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main()
