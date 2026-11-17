# Leaves_Diseases_PR
Dự án phát hiện và phân loại các loại bệnh trên lá cây ăn quả

Nguồn dataset vui lòng tải tại trang này: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data

Hoặc có thể tự tải về bằng cách chạy file test_kaggle.ipynb (thời gian dự kiến là 5 - 9p)

## Dataset manifests and PyTorch loaders

The labeled Kaggle `train` and `valid` folders should each contain one folder per class. From the repository root, generate portable CSV manifests without copying the images:

```powershell
python -m pip install -r requirements.txt
python -c "from src.dataset import generate_dataset_manifests; print(generate_dataset_manifests())"
```

This writes `dataset/processed/train_manifest.csv` and `dataset/processed/valid_manifest.csv`. Each row contains a repository-relative image path, split, class name, and class index. The split folders are discovered recursively under the configured `data` directory.

The manifest generator combines the configured `train` and `valid` folders, groups offline-augmented files by their original source stem, then creates a deterministic, class-stratified 80/20 split. This prevents augmented copies of the same leaf from leaking across training and validation.

Run the EDA and, optionally, create feature-enriched datasets for the legacy baseline:

```powershell
python -m src.eda
python -m src.feature_engineering
```

EDA checks image readability and dimensions, class balance, and augmented-source overlap across splits. Feature engineering writes separate `dataset/processed/train.csv` and `valid.csv` files with dimensions, aspect ratio, normalized RGB/HSV summaries, and compact per-channel histograms. These CSVs are retained for analysis compatibility; the primary classifier trains directly from image pixels.

Train the pretrained ResNet-50 classifier and export validation error analysis. Validation inference uses CUDA with mixed precision when available and prints progress while processing the full validation split:

```powershell
python -m src.train --epochs 10
python -m src.evaluate --checkpoint models/plant_disease_resnet50.pt
```

The first run downloads the ImageNet-pretrained ResNet-50 weights. Use `--no-pretrained` to train without downloading weights. Training uses ImageNet normalization, randomized resized crops, flips and color jitter; the classification head uses label-smoothed cross entropy, AdamW and cosine annealing. The best validation-accuracy checkpoint is saved to `models/plant_disease_resnet50.pt`. Evaluation writes metrics, per-class results, a confusion matrix, predictions and misclassified examples under `dataset/processed/error_analysis/`.

Predict the top classes for one image:

```powershell
python -m src.predict --image "C:\path\to\leaf.jpg"
```

By default, the script loads `models/plant_disease_resnet50.pt` and reports whether the leaf is classified as healthy or diseased, the plant, the predicted disease (if any), and the model score. It also lists other likely classes. Override the checkpoint or number of results with `--checkpoint` or `--top-k`. This is a model prediction, not a definitive diagnosis.

To create PyTorch image loaders (manifests are refreshed automatically):

```python
from src.dataset import get_dataloaders

train_loader, valid_loader, class_to_idx = get_dataloaders()
```

`get_dataloaders()` reads `config/config.yaml`, applies random resized crops, flips and color jitter to training images, and normalizes both splits with ImageNet RGB statistics. Training and validation batch sizes can be configured independently with `batch_size` and `validation_batch_size`; `num_workers` controls parallel image loading. It returns the training loader, validation loader, and class mapping.