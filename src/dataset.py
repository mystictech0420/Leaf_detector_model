"""Dataset discovery, CSV manifests, and PyTorch data loaders."""

import csv
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
MANIFEST_FIELDS = ("path", "split", "class_name", "class_idx")
_AUGMENTATION_SUFFIX = re.compile(
    r"_(?:\d{1,3}deg|new\d+deg(?:FlipLR|FlipUD)?|Flip(?:LR|UD)|rot\d*)$",
    re.IGNORECASE,
)


def _source_stem(filename: str) -> str:
    stem = Path(filename).stem
    while True:
        match = _AUGMENTATION_SUFFIX.search(stem)
        if match is None:
            return stem
        stem = stem[: match.start()]


def _group_split_by_source(
    class_rows: Sequence[dict[str, str | int]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Split images without separating offline-augmented source variants."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one.")

    stem_groups: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in class_rows:
        stem_groups[_source_stem(Path(str(row["source_path"])).name)].append(row)
    if len(stem_groups) < 2:
        raise ValueError("Each class needs at least two source images to create both splits.")

    stems = sorted(stem_groups)
    random.Random(seed).shuffle(stems)
    split_idx = min(max(int(len(stems) * train_ratio), 1), len(stems) - 1)
    train_stems = set(stems[:split_idx])

    train_rows: list[dict[str, str | int]] = []
    valid_rows: list[dict[str, str | int]] = []
    for stem, rows in stem_groups.items():
        target = train_rows if stem in train_stems else valid_rows
        split = "train" if stem in train_stems else "valid"
        target.extend({**row, "split": split} for row in rows)
    return train_rows, valid_rows


class PlantDiseaseDataset(Dataset[tuple[Tensor, int]]):
    """Load labeled plant-disease images from class-organized directories."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        class_to_idx: Mapping[str, int],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if not class_to_idx:
            raise ValueError("class_to_idx must contain at least one class.")

        self.image_paths = tuple(Path(path) for path in image_paths)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

        unknown_classes = sorted(
            {path.parent.name for path in self.image_paths} - self.class_to_idx.keys()
        )
        if unknown_classes:
            raise ValueError(
                "Image paths contain classes missing from class_to_idx: "
                + ", ".join(unknown_classes)
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            sample = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = transforms.ToTensor()(sample)
        if not isinstance(sample, Tensor):
            raise TypeError("Image transform must return a PyTorch tensor.")

        label = self.class_to_idx[image_path.parent.name]
        return sample, label


def _read_config(config_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    project_root = path.parent.parent if path.parent.name == "config" else path.parent
    return config, project_root


def _required_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return section


def _discover_split(data_dir: Path, split: str) -> Path:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    matches = sorted(path for path in data_dir.rglob(split) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find a '{split}' directory under {data_dir}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Found multiple '{split}' directories under {data_dir}: {locations}"
        )
    return matches[0]


def _load_class_mapping(path: Path, expected_count: int | None) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping file not found: {path}")

    with path.open(encoding="utf-8") as file:
        mapping_data = json.load(file)
    if not isinstance(mapping_data, dict):
        raise ValueError(f"Class mapping file must contain a JSON object: {path}")
    class_to_idx = mapping_data.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError(f"'class_to_idx' must be a non-empty object in {path}")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in class_to_idx.values()):
        raise ValueError(f"Class indices in {path} must be integers.")

    indices = sorted(class_to_idx.values())
    if indices != list(range(len(class_to_idx))):
        raise ValueError(f"Class indices in {path} must be unique and contiguous from zero.")
    if expected_count is not None and len(class_to_idx) != expected_count:
        raise ValueError(
            f"Configuration expects {expected_count} classes, but {path} defines "
            f"{len(class_to_idx)}."
        )
    return class_to_idx


def _image_paths_by_class(
    split_dir: Path, split: str, class_to_idx: Mapping[str, int]
) -> list[dict[str, str | int]]:
    class_dirs = {path.name: path for path in split_dir.iterdir() if path.is_dir()}
    if set(class_dirs) != set(class_to_idx):
        missing = sorted(set(class_to_idx) - set(class_dirs))
        unexpected = sorted(set(class_dirs) - set(class_to_idx))
        details = []
        if missing:
            details.append("missing classes: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected classes: " + ", ".join(unexpected))
        raise ValueError(f"Class folders in {split_dir} do not match mapping ({'; '.join(details)}).")

    rows: list[dict[str, str | int]] = []
    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda item: item[1]):
        images = sorted(
            path
            for path in class_dirs[class_name].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(
                f"No supported images found for class '{class_name}' in {split_dir}."
            )
        rows.extend(
            {
                "source_path": str(path.resolve()),
                "split": split,
                "class_name": class_name,
                "class_idx": class_idx,
            }
            for path in images
        )
    return rows


def _write_manifest(rows: Sequence[Mapping[str, str | int]], path: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                image_path = Path(str(row["source_path"])).resolve()
                try:
                    relative_path = image_path.relative_to(root).as_posix()
                except ValueError as exc:
                    raise ValueError(
                        f"Image path is outside the project root and cannot be made "
                        f"portable: {image_path}"
                    ) from exc
                writer.writerow(
                    {
                        "path": relative_path,
                        "split": row["split"],
                        "class_name": row["class_name"],
                        "class_idx": row["class_idx"],
                    }
                )
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def generate_dataset_manifests(
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Create portable per-image CSV manifests without copying source images."""
    config, project_root = _read_config(config_path)
    dataset_config = _required_section(config, "dataset")
    paths_config = _required_section(config, "paths")

    data_dir_value = paths_config.get("data_dir")
    mapping_value = paths_config.get("class_mapping")
    if not isinstance(data_dir_value, str) or not data_dir_value:
        raise ValueError("Configuration 'paths.data_dir' must be a non-empty string.")
    if not isinstance(mapping_value, str) or not mapping_value:
        raise ValueError("Configuration 'paths.class_mapping' must be a non-empty string.")

    data_dir = (project_root / data_dir_value).resolve()
    mapping_path = (project_root / mapping_value).resolve()
    expected_count = dataset_config.get("num_classes")
    if expected_count is not None and (
        not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0
    ):
        raise ValueError("Configuration 'dataset.num_classes' must be a positive integer.")
    class_to_idx = _load_class_mapping(mapping_path, expected_count)

    all_rows = [
        row
        for split in ("train", "valid")
        for row in _image_paths_by_class(
            _discover_split(data_dir, split), split, class_to_idx
        )
    ]
    split_rows: dict[str, list[dict[str, str | int]]] = {
        "train": [],
        "valid": [],
    }
    for class_name in class_to_idx:
        class_rows = [
            row for row in all_rows if row["class_name"] == class_name
        ]
        train_rows, valid_rows = _group_split_by_source(class_rows)
        split_rows["train"].extend(train_rows)
        split_rows["valid"].extend(valid_rows)

    output_dir = project_root / "dataset" / "processed"
    manifests = {split: output_dir / f"{split}_manifest.csv" for split in split_rows}
    for split, rows in split_rows.items():
        _write_manifest(rows, manifests[split], project_root)
    return manifests


def get_dataloaders(
    config: str | Path | None = None,
) -> tuple[DataLoader[tuple[Tensor, int]], DataLoader[tuple[Tensor, int]], dict[str, int]]:
    """Build train and validation loaders from a YAML config file."""
    config_data, project_root = _read_config(config)
    dataset_config = _required_section(config_data, "dataset")

    image_size = dataset_config.get("img_size", 224)
    batch_size = dataset_config.get("batch_size", 64)
    validation_batch_size = dataset_config.get("validation_batch_size", batch_size)
    num_workers = dataset_config.get("num_workers", 0)
    mean = dataset_config.get("mean")
    std = dataset_config.get("std")
    for key, value in (
        ("img_size", image_size),
        ("batch_size", batch_size),
        ("validation_batch_size", validation_batch_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Configuration 'dataset.{key}' must be a positive integer.")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("Configuration 'dataset.num_workers' must be a non-negative integer.")
    if (
        not isinstance(mean, Sequence)
        or isinstance(mean, (str, bytes))
        or not isinstance(std, Sequence)
        or isinstance(std, (str, bytes))
        or len(mean) != 3
        or len(std) != 3
    ):
        raise ValueError("Configuration 'dataset.mean' and 'dataset.std' must each have 3 values.")
    if any(not isinstance(value, (int, float)) for value in (*mean, *std)):
        raise ValueError("Normalization values must be numeric.")
    if any(value <= 0 for value in std):
        raise ValueError("Normalization standard deviations must be positive.")

    manifest_paths = generate_dataset_manifests(config)
    paths_config = _required_section(config_data, "paths")
    class_mapping_path = (project_root / str(paths_config["class_mapping"])).resolve()
    class_to_idx = _load_class_mapping(
        class_mapping_path, dataset_config.get("num_classes")
    )

    def load_manifest(split: str) -> list[Path]:
        paths: list[Path] = []
        with manifest_paths[split].open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(MANIFEST_FIELDS):
                raise ValueError(f"Unexpected columns in {manifest_paths[split]}.")
            for row in reader:
                if row["split"] != split:
                    raise ValueError(
                        f"Unexpected split '{row['split']}' in {manifest_paths[split]}."
                    )
                if row["class_name"] not in class_to_idx:
                    raise ValueError(
                        f"Unknown class '{row['class_name']}' in {manifest_paths[split]}."
                    )
                if int(row["class_idx"]) != class_to_idx[row["class_name"]]:
                    raise ValueError(
                        f"Class index mismatch for '{row['class_name']}' in "
                        f"{manifest_paths[split]}."
                    )
                image_path = (project_root / row["path"]).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image listed in {manifest_paths[split]} was not found: {image_path}"
                    )
                paths.append(image_path)
        if not paths:
            raise ValueError(f"Manifest contains no images: {manifest_paths[split]}")
        return paths

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = PlantDiseaseDataset(
        load_manifest("train"), class_to_idx, transform=train_transform
    )
    valid_dataset = PlantDiseaseDataset(
        load_manifest("valid"), class_to_idx, transform=valid_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, valid_loader, class_to_idx