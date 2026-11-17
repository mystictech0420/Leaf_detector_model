"""Dataset integrity and exploratory analysis for the leaf image manifests."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import _source_stem, generate_dataset_manifests


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = ["path", "split", "class_name", "class_idx"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns in {path}; expected {expected}.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest contains no images: {path}")
    return rows


def _write_plots(
    output_dir: Path,
    split_counts: dict[str, Counter[str]],
    dimensions: Counter[tuple[int, int]],
) -> None:
    class_names = sorted(
        {name for counts in split_counts.values() for name in counts}
    )
    figure, axis = plt.subplots(figsize=(12, max(8, len(class_names) * 0.32)))
    positions = list(range(len(class_names)))
    train_values = [split_counts.get("train", Counter())[name] for name in class_names]
    valid_values = [split_counts.get("valid", Counter())[name] for name in class_names]
    axis.barh(positions, train_values, label="train")
    axis.barh(positions, valid_values, left=train_values, label="valid")
    axis.set_yticks(positions, class_names)
    axis.set_xlabel("Image count")
    axis.set_title("Class counts by split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "class_counts.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    most_common = dimensions.most_common(15)
    labels = [f"{width}x{height}" for (width, height), _ in most_common]
    counts = [count for _, count in most_common]
    axis.bar(labels, counts)
    axis.set_ylabel("Image count")
    axis.set_title("Most common image dimensions")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "image_dimensions.png", dpi=150)
    plt.close(figure)


def run_eda(config_path: str | Path | None = None) -> dict[str, Any]:
    """Audit every manifest image and write a compact summary plus plots."""
    manifests = generate_dataset_manifests(config_path)
    project_root = manifests["train"].parent.parent.parent.resolve()
    output_dir = project_root / "dataset" / "processed" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, Counter[str]] = {}
    dimensions: Counter[tuple[int, int]] = Counter()
    source_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    unreadable: list[str] = []
    missing: list[str] = []
    total_images = 0

    for split, manifest_path in manifests.items():
        rows = _read_rows(manifest_path)
        counts: Counter[str] = Counter()
        split_counts[split] = counts
        for index, row in enumerate(rows, start=1):
            total_images += 1
            class_name = row["class_name"]
            counts[class_name] += 1
            source_splits[(class_name, _source_stem(Path(row["path"]).name))].add(split)
            image_path = (project_root / row["path"]).resolve()
            try:
                image_path.relative_to(project_root)
            except ValueError as exc:
                raise ValueError(f"Image path is outside the project root: {image_path}") from exc
            if not image_path.is_file():
                missing.append(row["path"])
                continue
            try:
                with Image.open(image_path) as image:
                    size = image.size
                    image.verify()
                dimensions[size] += 1
            except (OSError, UnidentifiedImageError, ValueError):
                unreadable.append(row["path"])
            if index % 10000 == 0:
                print(f"{split}: audited {index:,}/{len(rows):,} images")
        print(f"{split}: audited {len(rows):,} images")

    overlapping_sources = [
        {"class_name": class_name, "source_stem": stem}
        for (class_name, stem), splits in source_splits.items()
        if len(splits) > 1
    ]
    all_classes = sorted(
        {class_name for counts in split_counts.values() for class_name in counts}
    )
    class_parts = [name.split("___", maxsplit=1) for name in all_classes]
    summary: dict[str, Any] = {
        "total_images": total_images,
        "classes": len(all_classes),
        "plant_species": len({parts[0] for parts in class_parts}),
        "split_counts": {
            split: {"images": sum(counts.values()), "classes": len(counts)}
            for split, counts in split_counts.items()
        },
        "class_counts": {
            split: dict(sorted(counts.items())) for split, counts in split_counts.items()
        },
        "dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in dimensions.most_common()
        ],
        "missing_images": missing,
        "unreadable_images": unreadable,
        "cross_split_source_overlap_count": len(overlapping_sources),
        "cross_split_source_overlap_examples": overlapping_sources[:25],
    }
    _write_plots(output_dir, split_counts, dimensions)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"EDA summary written to {summary_path}")
    print(f"Unreadable images: {len(unreadable)}; missing images: {len(missing)}")
    print(f"Potential source overlap across splits: {len(overlapping_sources)}")
    return summary


if __name__ == "__main__":
    run_eda()
