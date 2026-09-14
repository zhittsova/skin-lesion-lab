"""PyTorch helpers for the in-project deep learning baseline."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


class SkinLesionImageDataset(Dataset):
    """Image dataset backed by the local ISIC image folder."""

    def __init__(
        self,
        image_ids: Sequence[str],
        labels: Sequence[int],
        images_dir: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_ids = list(image_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label, image_id, str(image_path)


class SmallDropoutCnn(nn.Module):
    """Small CNN with dropout layers for fast local MC Dropout experiments."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout * 0.75),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(96, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return self.classifier(x).squeeze(1)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Build image transforms using ImageNet normalization for both models."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(
    architecture: str = "small_cnn",
    dropout: float = 0.3,
    pretrained: bool = False,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build one of the project-owned deep baselines."""
    if architecture == "small_cnn":
        return SmallDropoutCnn(dropout=dropout)

    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(labels: np.ndarray) -> float:
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    return negatives / max(positives, 1.0)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_probabilities(model: nn.Module, dataloader, device: torch.device) -> dict:
    model.eval()
    probabilities = []
    labels_all = []
    image_ids = []
    image_paths = []

    for images, labels, batch_image_ids, batch_image_paths in dataloader:
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        probabilities.append(probs)
        labels_all.append(labels.numpy())
        image_ids.extend(batch_image_ids)
        image_paths.extend(batch_image_paths)

    return {
        "probability": np.concatenate(probabilities),
        "label": np.concatenate(labels_all).astype(int),
        "image_id": image_ids,
        "image_path": image_paths,
    }


def set_dropout_layers_to_train(model: nn.Module) -> None:
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


@torch.no_grad()
def predict_with_mc_dropout(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_passes: int = 30,
) -> dict:
    """Run dropout-active inference and return all stochastic probabilities."""
    if n_passes < 2:
        raise ValueError("MC Dropout needs at least two stochastic passes")

    all_samples = []
    labels_all = None
    image_ids = None
    image_paths = None

    for _ in range(n_passes):
        model.eval()
        set_dropout_layers_to_train(model)

        pass_probabilities = []
        pass_labels = []
        pass_image_ids = []
        pass_image_paths = []

        for images, labels, batch_image_ids, batch_image_paths in dataloader:
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            pass_probabilities.append(probs)
            pass_labels.append(labels.numpy())
            pass_image_ids.extend(batch_image_ids)
            pass_image_paths.extend(batch_image_paths)

        all_samples.append(np.concatenate(pass_probabilities))

        if labels_all is None:
            labels_all = np.concatenate(pass_labels).astype(int)
            image_ids = list(pass_image_ids)
            image_paths = list(pass_image_paths)

    samples = np.stack(all_samples, axis=0)
    return {
        "all_probabilities": samples,
        "mean_probability": samples.mean(axis=0),
        "uncertainty": samples.std(axis=0, ddof=1),
        "label": labels_all,
        "image_id": image_ids,
        "image_path": image_paths,
    }
