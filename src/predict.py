"""Predict plant-disease classes for one leaf image using a trained ResNet-50."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PROJECT_ROOT
from src.model import build_disease_classifier


def classify_prediction(class_name: str) -> tuple[str, str, str | None]:
    """Split a model class label into health status, plant, and disease name."""
    plant_name, separator, condition = class_name.partition("___")
    if not separator or not plant_name or not condition.strip("_ "):
        raise ValueError(f"Unexpected plant-disease class label: {class_name!r}")

    condition_name = condition.strip("_ ")
    plant_display = plant_name.replace("_", " ").strip()
    if condition_name.casefold() == "healthy":
        return "Healthy", plant_display, None
    disease_display = condition_name.replace("_", " ").strip()
    return "Disease detected", plant_display, disease_display


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path = PROJECT_ROOT / "models" / "plant_disease_resnet50.pt",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the top predicted class names and their probabilities."""
    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet50":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint_path}")
    class_names = checkpoint.get("class_names")
    num_classes = checkpoint.get("num_classes")
    if (
        not isinstance(class_names, list)
        or not all(isinstance(name, str) for name in class_names)
        or not isinstance(num_classes, int)
        or isinstance(num_classes, bool)
        or len(class_names) != num_classes
        or num_classes <= 1
    ):
        raise ValueError(f"Checkpoint has an invalid class mapping: {checkpoint_path}")
    if not 1 <= top_k <= num_classes:
        raise ValueError(f"top_k must be between 1 and {num_classes}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_disease_classifier(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        scores, indices = probabilities.topk(top_k)

    return [
        (class_names[index], float(score))
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to a leaf image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "plant_disease_resnet50.pt",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictions = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
    )
    class_name, probability = predictions[0]
    status, plant_name, disease_name = classify_prediction(class_name)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Assessment: {status}")
    print(f"Plant: {plant_name}")
    if disease_name is None:
        print("Disease: None detected")
    else:
        print(f"Disease: {disease_name}")
    print(f"Model score: {probability:.1%}")
    if len(predictions) > 1:
        print("Other likely classes:")
        for alternative, score in predictions[1:]:
            print(f"  - {alternative}: {score:.1%}")
    print("This is an image-model prediction, not a definitive diagnosis.")


if __name__ == "__main__":
    main()
