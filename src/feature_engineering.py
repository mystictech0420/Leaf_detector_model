"""Extract deterministic, interpretable image features into CSV datasets."""

import csv
import os
import tempfile
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageOps, ImageStat

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import MANIFEST_FIELDS, generate_dataset_manifests


_CHANNELS = ("r", "g", "b")
_HSV_CHANNELS = ("h", "s", "v")
_HISTOGRAM_BINS = 8
_BIN_WIDTH = 256 // _HISTOGRAM_BINS
FEATURE_FIELDS = (
    "width",
    "height",
    "aspect_ratio",
    *(f"rgb_mean_{channel}" for channel in _CHANNELS),
    *(f"rgb_std_{channel}" for channel in _CHANNELS),
    *(f"hsv_mean_{channel}" for channel in _HSV_CHANNELS),
    *(f"hsv_std_{channel}" for channel in _HSV_CHANNELS),
    *(
        f"{color}_hist_{channel}_{bin_index}"
        for color, channels in (("rgb", _CHANNELS), ("hsv", _HSV_CHANNELS))
        for channel_index, channel in enumerate(channels)
        for bin_index in range(_HISTOGRAM_BINS)
    ),
)
OUTPUT_FIELDS = (*MANIFEST_FIELDS, *FEATURE_FIELDS)


def extract_image_features(image_path: str | Path) -> dict[str, int | float]:
    """Return size, normalized color statistics, and per-channel histograms."""
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")

    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")

    features: dict[str, int | float] = {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
    }

    hsv = rgb.convert("HSV")
    for color, image, channels in (
        ("rgb", rgb, _CHANNELS),
        ("hsv", hsv, _HSV_CHANNELS),
    ):
        statistics = ImageStat.Stat(image)
        for channel_index, channel in enumerate(channels):
            features[f"{color}_mean_{channel}"] = statistics.mean[channel_index] / 255
            features[f"{color}_std_{channel}"] = statistics.stddev[channel_index] / 255

        histogram = image.histogram()
        pixel_count = width * height
        for channel_index, channel in enumerate(channels):
            channel_histogram = histogram[channel_index * 256 : (channel_index + 1) * 256]
            for bin_index in range(_HISTOGRAM_BINS):
                start = bin_index * _BIN_WIDTH
                count = sum(channel_histogram[start : start + _BIN_WIDTH])
                features[f"{color}_hist_{channel}_{bin_index}"] = count / pixel_count

    return features


def _read_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(f"Unexpected columns in {path}.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest contains no images: {path}")
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(
                f"Unexpected split '{row['split']}' in {path}; expected '{expected_split}'."
            )
    return rows


def _feature_row(
    row: dict[str, str], project_root: Path
) -> dict[str, Any]:
    image_path = (project_root / row["path"]).resolve()
    try:
        image_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Image path is outside the project root: {image_path}") from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Image listed in manifest was not found: {image_path}")

    try:
        features = extract_image_features(image_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to extract features from {image_path}: {exc}") from exc
    return {**row, **features}


def generate_feature_csvs(
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate feature-enriched train/validation CSVs without changing images."""
    manifests = generate_dataset_manifests(config_path)
    project_root = manifests["train"].parent.parent.parent.resolve()
    output_paths = {
        split: project_root / "dataset" / "processed" / f"{split}.csv"
        for split in ("train", "valid")
    }
    temporary_paths: dict[str, Path] = {}

    try:
        for split, manifest_path in manifests.items():
            rows = _read_manifest(manifest_path, split)
            output_path = output_paths[split]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent
            )
            temporary_path = Path(temporary_name)
            temporary_paths[split] = temporary_path
            with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                for index, row in enumerate(rows, start=1):
                    writer.writerow(_feature_row(row, project_root))
                    if index % 5000 == 0:
                        print(f"{split}: processed {index:,}/{len(rows):,} images")
            print(f"{split}: processed {len(rows):,} images")

        for split, temporary_path in temporary_paths.items():
            os.replace(temporary_path, output_paths[split])
    except Exception:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
        raise

    return output_paths


if __name__ == "__main__":
    for split, path in generate_feature_csvs().items():
        print(f"{split} features written to {path}")
