import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import (
    PlantDiseaseDataset,
    _source_stem,
    generate_dataset_manifests,
    get_dataloaders,
)
from src.evaluate import run_classifier_error_analysis, run_error_analysis
from src.feature_engineering import (
    FEATURE_FIELDS,
    extract_image_features,
    generate_feature_csvs,
)
from src.model import build_disease_classifier, build_feature_model, load_feature_table
from src.predict import classify_prediction


class ImageFeatureTests(unittest.TestCase):
    def test_feature_statistics_and_histograms_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "solid-red.png"
            Image.new("RGB", (4, 2), color=(255, 0, 0)).save(image_path)

            features = extract_image_features(image_path)

        self.assertEqual(features["width"], 4)
        self.assertEqual(features["height"], 2)
        self.assertEqual(features["aspect_ratio"], 2.0)
        self.assertEqual(features["rgb_mean_r"], 1.0)
        self.assertEqual(features["rgb_mean_g"], 0.0)
        self.assertEqual(features["rgb_std_r"], 0.0)
        self.assertAlmostEqual(
            sum(features[f"rgb_hist_r_{index}"] for index in range(8)), 1.0
        )
        self.assertAlmostEqual(
            sum(features[f"rgb_hist_g_{index}"] for index in range(8)), 1.0
        )
        self.assertEqual(len(features), len(FEATURE_FIELDS))


class DiseaseClassifierTests(unittest.TestCase):
    def test_prediction_label_reports_health_or_disease(self) -> None:
        self.assertEqual(
            classify_prediction("Apple___healthy"),
            ("Healthy", "Apple", None),
        )
        self.assertEqual(
            classify_prediction("Tomato___Early_blight"),
            ("Disease detected", "Tomato", "Early blight"),
        )
        self.assertEqual(
            classify_prediction("Corn_(maize)___Common_rust_"),
            ("Disease detected", "Corn (maize)", "Common rust"),
        )

    def test_resnet_classifier_has_expected_head_and_freezing(self) -> None:
        model = build_disease_classifier(num_classes=3, pretrained=False)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros((1, 3, 64, 64)))

        self.assertEqual(logits.shape, (1, 3))
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(model.fc[-1].weight.requires_grad)

    def test_classifier_evaluation_exports_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            image_paths = []
            class_mapping = {"healthy": 0, "diseased": 1}
            for class_name, color in (
                ("healthy", (20, 180, 40)),
                ("diseased", (180, 20, 40)),
            ):
                image_path = project_root / "data" / class_name / "leaf.png"
                image_path.parent.mkdir(parents=True)
                Image.new("RGB", (32, 32), color=color).save(image_path)
                image_paths.append(image_path)

            dataset = PlantDiseaseDataset(
                image_paths, class_mapping, transform=transforms.ToTensor()
            )
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            model = build_disease_classifier(num_classes=2, pretrained=False)
            checkpoint_path = project_root / "classifier.pt"
            torch.save(
                {
                    "architecture": "resnet50",
                    "state_dict": model.state_dict(),
                    "class_names": ["healthy", "diseased"],
                    "num_classes": 2,
                },
                checkpoint_path,
            )
            output_dir = project_root / "error_analysis"

            with patch(
                "src.evaluate.get_dataloaders",
                return_value=(loader, loader, class_mapping),
            ), patch("src.evaluate.PROJECT_ROOT", project_root.resolve()):
                metrics = run_classifier_error_analysis(
                    checkpoint_path=checkpoint_path, output_dir=output_dir
                )

            self.assertEqual(metrics["validation_images"], 2)
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "validation_predictions.csv").is_file())


class FeatureCsvTests(unittest.TestCase):
    def test_generation_preserves_splits_and_creates_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            data_root = project_root / "data"
            class_mapping = {"healthy": 0, "diseased": 1}
            for split in ("train", "valid"):
                for class_name, class_index in class_mapping.items():
                    class_directory = data_root / split / class_name
                    class_directory.mkdir(parents=True)
                    Image.new(
                        "RGB",
                        (3 + class_index, 2),
                        color=(class_index * 100, 25, 180),
                    ).save(class_directory / f"{split}-{class_name}.png")

            config_directory = project_root / "config"
            config_directory.mkdir()
            (config_directory / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"num_classes": 2},
                        "paths": {
                            "data_dir": "data",
                            "class_mapping": "config/class_mapping.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (config_directory / "class_mapping.json").write_text(
                json.dumps({"class_to_idx": class_mapping}),
                encoding="utf-8",
            )

            outputs = generate_feature_csvs(config_directory / "config.yaml")

            for split, output_path in outputs.items():
                with output_path.open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    self.assertEqual(
                        reader.fieldnames,
                        ["path", "split", "class_name", "class_idx", *FEATURE_FIELDS],
                    )
                    rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["split"] for row in rows}, {split})
                table = load_feature_table(output_path, split)
                self.assertEqual(table.features.shape, (2, len(FEATURE_FIELDS)))
                self.assertEqual(table.class_names, ["healthy", "diseased"])

            checkpoint_path = project_root / "models" / "test-checkpoint.pt"
            checkpoint_path.parent.mkdir()
            torch.save(
                {
                    "state_dict": build_feature_model(len(FEATURE_FIELDS), 2).state_dict(),
                    "feature_fields": list(FEATURE_FIELDS),
                    "feature_mean": np.zeros(len(FEATURE_FIELDS)).tolist(),
                    "feature_std": np.ones(len(FEATURE_FIELDS)).tolist(),
                    "class_names": ["healthy", "diseased"],
                    "input_size": len(FEATURE_FIELDS),
                    "num_classes": 2,
                },
                checkpoint_path,
            )
            metrics = run_error_analysis(
                checkpoint_path=checkpoint_path,
                validation_csv=outputs["valid"],
                output_dir=project_root / "error_analysis",
            )
            self.assertEqual(metrics["validation_images"], 2)
            self.assertEqual(metrics["classes"], 2)
            self.assertTrue((project_root / "error_analysis" / "confusion_matrix.csv").is_file())
            self.assertTrue((project_root / "error_analysis" / "confusion_matrix.png").is_file())


class SourceGroupedSplitTests(unittest.TestCase):
    def test_augmented_source_images_never_cross_generated_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            data_root = project_root / "data"
            class_mapping = {"healthy": 0, "diseased": 1}
            for split in ("train", "valid"):
                for class_name in class_mapping:
                    (data_root / split / class_name).mkdir(parents=True)

            for class_name in class_mapping:
                for index in range(10):
                    source_split = "train" if index % 2 == 0 else "valid"
                    alternate_split = "valid" if source_split == "train" else "train"
                    image = Image.new("RGB", (4, 4), color=(index, 20, 40))
                    image.save(
                        data_root / source_split / class_name / f"leaf-{index}.png"
                    )
                    image.save(
                        data_root / alternate_split / class_name / f"leaf-{index}_rot.png"
                    )

            config_directory = project_root / "config"
            config_directory.mkdir()
            (config_directory / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "num_classes": 2,
                            "img_size": 32,
                            "batch_size": 4,
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                        "paths": {
                            "data_dir": "data",
                            "class_mapping": "config/class_mapping.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (config_directory / "class_mapping.json").write_text(
                json.dumps({"class_to_idx": class_mapping}),
                encoding="utf-8",
            )

            manifests = generate_dataset_manifests(config_directory / "config.yaml")
            rows_by_split: dict[str, list[dict[str, str]]] = {}
            for split, manifest_path in manifests.items():
                with manifest_path.open(newline="", encoding="utf-8") as file:
                    rows_by_split[split] = list(csv.DictReader(file))

            for class_name in class_mapping:
                source_stems = {
                    split: {
                        _source_stem(Path(row["path"]).name)
                        for row in rows
                        if row["class_name"] == class_name
                    }
                    for split, rows in rows_by_split.items()
                }
                self.assertFalse(source_stems["train"] & source_stems["valid"])
                self.assertEqual(len(source_stems["train"] | source_stems["valid"]), 10)

            first_split_rows = {
                split: [(row["path"], row["split"]) for row in rows]
                for split, rows in rows_by_split.items()
            }
            repeated_manifests = generate_dataset_manifests(
                config_directory / "config.yaml"
            )
            for split, manifest_path in repeated_manifests.items():
                with manifest_path.open(newline="", encoding="utf-8") as file:
                    repeated_rows = list(csv.DictReader(file))
                self.assertEqual(
                    [(row["path"], row["split"]) for row in repeated_rows],
                    first_split_rows[split],
                )

            train_loader, valid_loader, loaded_mapping = get_dataloaders(
                config_directory / "config.yaml"
            )
            train_images, _ = next(iter(train_loader))
            valid_images, _ = next(iter(valid_loader))
            self.assertEqual(loaded_mapping, class_mapping)
            self.assertEqual(tuple(train_images.shape[1:]), (3, 32, 32))
            self.assertEqual(tuple(valid_images.shape[1:]), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
