from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import MODEL_DIR, MODEL_SPECS, RANDOM_SEED, RESULTS_DIR
from data_loader import build_tf_dataset, prepare_splits, resolve_dataset_root
from models import is_compatible_xception_model


def _check_model(tf, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required trained model is missing: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    if not is_compatible_xception_model(model):
        raise ValueError(f"Model checkpoint is incompatible: {path}")
    return model


def _plot_training_history(history: dict, model_key: str, output_dir: Path) -> None:
    model_history = history.get(model_key, {})
    if "accuracy" not in model_history or "loss" not in model_history:
        return
    for metric in ("accuracy", "loss"):
        values = model_history[metric]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(values) + 1), values, label=f"Training {metric}")
        validation = model_history.get(f"val_{metric}")
        if validation:
            plt.plot(
                range(1, len(validation) + 1),
                validation,
                label=f"Validation {metric}",
            )
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(f"{MODEL_SPECS[model_key]['display_name']} {metric.title()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_key}_{metric}.png", dpi=150)
        plt.close()


def evaluate_one_model(
    tf, model_key: str, test_samples, batch_size: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    spec = MODEL_SPECS[model_key]
    model_path = MODEL_DIR / spec["filename"]
    model = _check_model(tf, model_path)
    test_data = build_tf_dataset(
        test_samples, model_key, batch_size, training=False, seed=RANDOM_SEED
    )
    started = time.monotonic()
    real_probabilities = np.asarray(model.predict(test_data, verbose=1)).reshape(-1)
    fake_probabilities = 1.0 - real_probabilities
    evaluation_duration = time.monotonic() - started
    truth = np.asarray([sample.label for sample in test_samples], dtype=np.int32)
    if len(fake_probabilities) != len(truth):
        raise RuntimeError(
            f"{spec['display_name']} returned {len(fake_probabilities)} predictions "
            f"for {len(truth)} test images."
        )
    predictions = (fake_probabilities >= 0.5).astype(np.int32)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    report = classification_report(
        truth,
        predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": model_key,
        "display_name": spec["display_name"],
        "test_accuracy": float(accuracy_score(truth, predictions)),
        "precision_fake": float(precision_score(truth, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(truth, predictions, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(truth, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, fake_probabilities))
        if len(np.unique(truth)) == 2
        else None,
        "training_duration_seconds": _training_duration(model_key),
        "evaluation_duration_seconds": evaluation_duration,
        "test_images": len(truth),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return metrics, truth, fake_probabilities


def _training_duration(model_key: str) -> float | None:
    times_path = RESULTS_DIR / "training_times.json"
    if not times_path.is_file():
        return None
    return json.loads(times_path.read_text(encoding="utf-8")).get(model_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Xception on the held-out test set.")
    parser.add_argument("--data-root", type=Path, help="Use a local dataset directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    import tensorflow as tf

    dataset_root = resolve_dataset_root(args.data_root)
    splits = prepare_splits(dataset_root, seed=args.seed)
    print(f"Dataset root: {dataset_root} | split source: {splits.source}")
    missing = [
        str(MODEL_DIR / spec["filename"])
        for spec in MODEL_SPECS.values()
        if not (MODEL_DIR / spec["filename"]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing trained model file(s):\n" + "\n".join(missing))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = RESULTS_DIR / "classification_reports"
    plot_dir = RESULTS_DIR / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    all_metrics = []
    roc_entries = []
    for model_key, spec in MODEL_SPECS.items():
        print(f"\nEvaluating {spec['display_name']}...")
        metrics, truth, probabilities = evaluate_one_model(
            tf, model_key, splits.test, args.batch_size
        )
        all_metrics.append(metrics)
        (report_dir / f"{model_key}.json").write_text(
            json.dumps(metrics["classification_report"], indent=2), encoding="utf-8"
        )
        matrix = np.asarray(metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            xlabel="Predicted label",
            ylabel="True label",
            title=f"{spec['display_name']} confusion matrix",
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{model_key}_confusion_matrix.png", dpi=150)
        plt.close(figure)
        _plot_training_history(history, model_key, plot_dir)
        if len(np.unique(truth)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(truth, probabilities)
            roc_entries.append((model_key, spec["display_name"], false_positive_rate, true_positive_rate))

    if roc_entries:
        plt.figure(figsize=(7, 6))
        for _, name, false_positive_rate, true_positive_rate in roc_entries:
            plt.plot(false_positive_rate, true_positive_rate, label=name)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curves on the held-out test set")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "roc_curves.png", dpi=150)
        plt.close()

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    flattened = [
        {key: value for key, value in item.items() if key not in {"confusion_matrix", "classification_report"}}
        for item in all_metrics
    ]
    metrics_csv = RESULTS_DIR / "metrics.csv"
    pd.DataFrame(flattened).to_csv(metrics_csv, index=False)
    print("\nComparison:")
    print(pd.DataFrame(flattened).to_string(index=False))
    print(f"\nSaved actual metrics: {metrics_csv} and {metrics_path}")


if __name__ == "__main__":
    main()
