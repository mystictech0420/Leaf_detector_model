"""Train the image-based disease classifier and legacy feature baseline."""

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PROJECT_ROOT, get_dataloaders
from src.feature_engineering import FEATURE_FIELDS, generate_feature_csvs
from src.model import build_disease_classifier, build_feature_model, load_feature_table


def _save_checkpoint(checkpoint_data: dict[str, Any], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint_data, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def train_baseline(
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    processed_dir = PROJECT_ROOT / "dataset" / "processed"
    feature_paths = {split: processed_dir / f"{split}.csv" for split in ("train", "valid")}
    expected_fields = ["path", "split", "class_name", "class_idx", *FEATURE_FIELDS]
    feature_files_ready = True
    for path in feature_paths.values():
        if not path.is_file():
            feature_files_ready = False
            break
        with path.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file), []) != expected_fields:
                feature_files_ready = False
                break
    if not feature_files_ready:
        feature_paths = generate_feature_csvs()
    train_table = load_feature_table(feature_paths["train"], "train")
    valid_table = load_feature_table(feature_paths["valid"], "valid")
    if train_table.class_names != valid_table.class_names:
        raise ValueError("Train and validation CSVs do not contain the same class mapping.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_mean = train_table.features.mean(axis=0)
    feature_std = train_table.features.std(axis=0)
    feature_std[feature_std == 0] = 1
    train_features = (train_table.features - feature_mean) / feature_std

    training_data = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_table.labels),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = build_feature_model(
        input_size=len(FEATURE_FIELDS),
        num_classes=len(train_table.class_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            total_seen += count
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        epoch_metrics = {
            "epoch": epoch,
            "loss": total_loss / total_seen,
            "accuracy": total_correct / total_seen,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={epoch_metrics['loss']:.4f}, "
            f"accuracy={epoch_metrics['accuracy']:.4f}"
        )

    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else PROJECT_ROOT / "models" / "feature_baseline.pt"
    ).resolve()
    _save_checkpoint(
        {
            "state_dict": model.cpu().state_dict(),
            "feature_fields": list(FEATURE_FIELDS),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "class_names": train_table.class_names,
            "input_size": len(FEATURE_FIELDS),
            "num_classes": len(train_table.class_names),
            "history": history,
            "seed": seed,
        },
        destination,
    )

    result = {
        "checkpoint": destination,
        "device": str(device),
        "train_images": len(train_table.rows),
        "validation_images": len(valid_table.rows),
        "classes": len(train_table.class_names),
        "history": history,
    }
    print(f"Baseline checkpoint written to {destination}")
    return result


def train_classifier(
    epochs: int = 10,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Train and validate the pretrained image classifier."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    train_loader, valid_loader, class_to_idx = get_dataloaders()
    class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    model = build_disease_classifier(
        num_classes=len(class_names), pretrained=pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": 1e-3},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if name.startswith("layer4.") and parameter.requires_grad
                ],
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else PROJECT_ROOT / "models" / "plant_disease_resnet50.pt"
    )
    history: list[dict[str, float]] = []
    epoch_durations: list[float] = []
    best_validation_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count = labels.size(0)
            train_seen += count
            train_loss += loss.item() * count
            train_correct += (logits.argmax(dim=1) == labels).sum().item()

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_seen = 0
        with torch.inference_mode():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_cuda,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                count = labels.size(0)
                validation_seen += count
                validation_loss += loss.item() * count
                validation_correct += (logits.argmax(dim=1) == labels).sum().item()

        epoch_metrics = {
            "epoch": float(epoch),
            "loss": train_loss / train_seen,
            "accuracy": train_correct / train_seen,
            "validation_loss": validation_loss / validation_seen,
            "validation_accuracy": validation_correct / validation_seen,
        }
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics["epoch_seconds"] = epoch_seconds
        history.append(epoch_metrics)
        scheduler.step()
        epoch_durations.append(epoch_seconds)
        average_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_seconds = average_epoch_seconds * (epochs - epoch)
        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={epoch_metrics['loss']:.4f}, "
            f"accuracy={epoch_metrics['accuracy']:.4f}, "
            f"validation_loss={epoch_metrics['validation_loss']:.4f}, "
            f"validation_accuracy={epoch_metrics['validation_accuracy']:.4f}, "
            f"time={epoch_seconds / 60:.1f} min, "
            f"ETA={remaining_seconds / 60:.1f} min"
        )

        if epoch_metrics["validation_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = epoch_metrics["validation_accuracy"]
            state_dict = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            _save_checkpoint(
                {
                    "architecture": "resnet50",
                    "state_dict": state_dict,
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "pretrained": pretrained,
                    "history": history,
                    "seed": seed,
                },
                destination,
            )

    result = {
        "checkpoint": destination.resolve(),
        "device": str(device),
        "train_images": train_seen,
        "validation_images": validation_seen,
        "classes": len(class_names),
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
    }
    print(f"CNN checkpoint written to {destination.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet-50 disease classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    train_classifier(
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()