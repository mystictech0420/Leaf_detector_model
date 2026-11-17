"""Image-based disease classifier and legacy feature-baseline utilities."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.feature_engineering import FEATURE_FIELDS


@dataclass(frozen=True)
class FeatureTable:
    features: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, str]]
    class_names: list[str]


def build_disease_classifier(
    num_classes: int = 38, pretrained: bool = True
) -> nn.Module:
    """Build a ResNet-50 classifier, fine-tuning its final residual stage."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def build_feature_model(input_size: int, num_classes: int, hidden_size: int = 128) -> nn.Module:
    if input_size <= 0 or num_classes <= 1 or hidden_size <= 0:
        raise ValueError("Model input, class, and hidden sizes must be positive.")
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )


def load_feature_table(path: str | Path, expected_split: str) -> FeatureTable:
    path = Path(path)
    required_fields = ["path", "split", "class_name", "class_idx", *FEATURE_FIELDS]
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    rows: list[dict[str, str]] = []
    names_by_index: dict[int, str] = {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != required_fields:
            raise ValueError(f"Unexpected columns in {path}.")
        for line_number, row in enumerate(reader, start=2):
            if row["split"] != expected_split:
                raise ValueError(
                    f"Unexpected split '{row['split']}' on line {line_number} of {path}."
                )
            try:
                class_index = int(row["class_idx"])
                values = [float(row[name]) for name in FEATURE_FIELDS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label/features on line {line_number} of {path}.") from exc
            if class_index < 0 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid label/features on line {line_number} of {path}.")
            class_name = row["class_name"]
            if class_index in names_by_index and names_by_index[class_index] != class_name:
                raise ValueError(f"Conflicting class names for class index {class_index} in {path}.")
            names_by_index[class_index] = class_name
            feature_rows.append(values)
            labels.append(class_index)
            rows.append(row)

    if not rows:
        raise ValueError(f"Feature CSV contains no rows: {path}")
    expected_indices = list(range(len(names_by_index)))
    if sorted(names_by_index) != expected_indices:
        raise ValueError(f"Class indices in {path} must be contiguous from zero.")

    class_names = [names_by_index[index] for index in expected_indices]
    return FeatureTable(
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        rows=rows,
        class_names=class_names,
    )