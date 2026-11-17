"""Evaluate disease-classifier checkpoints and export validation error analysis."""

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PROJECT_ROOT, PlantDiseaseDataset, get_dataloaders
from src.feature_engineering import FEATURE_FIELDS
from src.model import (
    build_disease_classifier,
    build_feature_model,
    load_feature_table,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_error_analysis(
    results_dir: Path,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    true_positives = np.diag(confusion)
    precision = np.divide(
        true_positives,
        confusion.sum(axis=0),
        out=np.zeros(class_count, dtype=float),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(axis=1),
        out=np.zeros(class_count, dtype=float),
        where=confusion.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) != 0,
    )
    accuracy = float(np.mean(predictions == labels))
    per_class = [
        {
            "class_name": class_names[index],
            "support": int(confusion[index].sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(class_count)
    ]
    metrics: dict[str, Any] = {
        "validation_images": len(rows),
        "classes": class_count,
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "misclassified_images": int(np.count_nonzero(predictions != labels)),
        "per_class": per_class,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_csv(
        results_dir / "per_class_metrics.csv",
        ["class_name", "support", "precision", "recall", "f1"],
        per_class,
    )
    confusion_rows = [
        {
            "true_class": class_names[true_index],
            **{
                f"predicted::{class_names[predicted_index]}": int(
                    confusion[true_index, predicted_index]
                )
                for predicted_index in range(class_count)
            },
        }
        for true_index in range(class_count)
    ]
    _write_csv(
        results_dir / "confusion_matrix.csv",
        ["true_class", *(f"predicted::{name}" for name in class_names)],
        confusion_rows,
    )
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    display_names = [name.split("___", maxsplit=1)[-1] for name in class_names]
    positions = np.arange(class_count)
    axis.set_xticks(positions, display_names, rotation=90, fontsize=7)
    axis.set_yticks(positions, display_names, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Validation confusion matrix")
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    prediction_rows = [
        {
            "path": row["path"],
            "true_class": class_names[int(truth)],
            "predicted_class": class_names[int(prediction)],
            "confidence": f"{float(score):.8f}",
            "correct": str(int(truth) == int(prediction)).lower(),
        }
        for row, truth, prediction, score in zip(
            rows, labels, predictions, confidence
        )
    ]
    prediction_fields = [
        "path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]
    _write_csv(
        results_dir / "validation_predictions.csv", prediction_fields, prediction_rows
    )
    _write_csv(
        results_dir / "validation_errors.csv",
        prediction_fields,
        [row for row in prediction_rows if row["correct"] == "false"],
    )
    print(
        f"Validation accuracy: {accuracy:.4f}; "
        f"macro F1: {metrics['macro_f1']:.4f}; "
        f"errors: {metrics['misclassified_images']:,}/{len(rows):,}"
    )
    print(f"Error analysis written to {results_dir}")
    return metrics


def run_error_analysis(
    checkpoint_path: str | Path | None = None,
    validation_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else PROJECT_ROOT / "models" / "feature_baseline.pt"
    ).resolve()
    validation_path = (
        Path(validation_csv)
        if validation_csv is not None
        else PROJECT_ROOT / "dataset" / "processed" / "valid.csv"
    ).resolve()
    results_dir = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "dataset" / "processed" / "error_analysis"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("feature_fields") != list(FEATURE_FIELDS):
        raise ValueError("Checkpoint feature schema does not match the current feature CSV schema.")
    table = load_feature_table(validation_path, "valid")
    class_names = checkpoint_data.get("class_names")
    if table.class_names != class_names:
        raise ValueError("Validation CSV classes do not match the checkpoint classes.")

    mean = np.asarray(checkpoint_data["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint_data["feature_std"], dtype=np.float32)
    if mean.shape != (len(FEATURE_FIELDS),) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("Checkpoint contains invalid feature normalization values.")
    normalized = (table.features - mean) / std
    model = build_feature_model(
        input_size=int(checkpoint_data["input_size"]),
        num_classes=int(checkpoint_data["num_classes"]),
    )
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy(normalized)), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    rows = [{"path": row["path"]} for row in table.rows]
    return _write_error_analysis(
        results_dir, rows, table.labels, predictions, confidence, class_names
    )


def run_classifier_error_analysis(
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the image-based classifier against the generated validation split."""
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else PROJECT_ROOT / "models" / "plant_disease_resnet50.pt"
    ).resolve()
    results_dir = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "dataset" / "processed" / "error_analysis"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("architecture") != "resnet50":
        raise ValueError(f"Unsupported checkpoint architecture in {checkpoint}.")
    train_loader, valid_loader, class_to_idx = get_dataloaders()
    del train_loader
    if not isinstance(valid_loader.dataset, PlantDiseaseDataset):
        raise TypeError("Validation loader does not contain a PlantDiseaseDataset.")

    class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    if checkpoint_data.get("class_names") != class_names:
        raise ValueError("Validation classes do not match the CNN checkpoint classes.")
    if checkpoint_data.get("num_classes") != len(class_names):
        raise ValueError("CNN checkpoint class count does not match the class mapping.")

    model = build_disease_classifier(
        num_classes=len(class_names), pretrained=False
    )
    model.load_state_dict(checkpoint_data["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model.to(device)
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    total_batches = len(valid_loader)
    progress_interval = max(1, total_batches // 20)
    evaluation_start = time.perf_counter()
    print(
        f"Evaluating {len(valid_loader.dataset)} images on {device} "
        f"in {total_batches} batches...",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(valid_loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                probabilities = torch.softmax(model(images), dim=1)
            all_labels.append(labels.numpy())
            all_probabilities.append(probabilities.float().cpu().numpy())
            if batch_index % progress_interval == 0 or batch_index == total_batches:
                print(
                    f"Evaluated {batch_index}/{total_batches} batches "
                    f"({batch_index / total_batches:.0%})",
                    flush=True,
                )

    print(
        f"Inference completed in "
        f"{(time.perf_counter() - evaluation_start) / 60:.1f} minutes.",
        flush=True,
    )
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows = [
        {"path": path.resolve().relative_to(PROJECT_ROOT).as_posix()}
        for path in valid_loader.dataset.image_paths
    ]
    return _write_error_analysis(
        results_dir, rows, labels, predictions, confidence, class_names
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validation_csv is not None:
        run_error_analysis(
            checkpoint_path=args.checkpoint,
            validation_csv=args.validation_csv,
            output_dir=args.output_dir,
        )
    else:
        run_classifier_error_analysis(
            checkpoint_path=args.checkpoint, output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()