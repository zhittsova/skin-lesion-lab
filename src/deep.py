"""PyTorch helpers for the in-project deep learning baseline."""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from src.image_cache import ResizedImageCache


class SkinLesionImageDataset(Dataset):
    """Image dataset backed by the local ISIC image folder."""

    def __init__(
        self,
        image_ids: Sequence[str],
        labels: Sequence[int],
        images_dir: Path,
        transform: transforms.Compose | None = None,
        resized_cache: ResizedImageCache | None = None,
    ) -> None:
        self.image_ids = list(image_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.images_dir = Path(images_dir)
        self.transform = transform
        if resized_cache is not None and not isinstance(
            resized_cache, ResizedImageCache
        ):
            raise ValueError("resized_cache must be a ResizedImageCache")
        self.resized_cache = resized_cache
        self._after_resize_transform = (
            resized_cache.prepare_transform(
                self.image_ids, self.images_dir, self.transform
            )
            if resized_cache is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = self.images_dir / f"{image_id}.jpg"
        image = (
            self.resized_cache.get(image_id)
            if self.resized_cache is not None
            else Image.open(image_path).convert("RGB")
        )

        if self._after_resize_transform is not None:
            image = self._after_resize_transform(image)
        elif self.transform is not None:
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
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
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
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if architecture == "small_cnn":
        if pretrained or not freeze_backbone:
            raise ValueError("small_cnn has no pretrained or backbone fine-tuning mode")
        return SmallDropoutCnn(dropout=dropout)

    if architecture == "efficientnet_b0":
        if freeze_backbone and not pretrained:
            raise ValueError(
                "a frozen EfficientNet backbone requires pretrained weights"
            )
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        model._frozen_backbone = freeze_backbone
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


def get_default_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError(f"device unavailable or invalid: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    if not 0 <= seed <= 2**32 - 4:
        raise ValueError("seed must be between 0 and 2**32 - 4")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from PyTorch's DataLoader worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_training_mode(model: nn.Module) -> None:
    """Train the classifier while keeping a frozen backbone's buffers fixed."""
    model.train()
    if getattr(model, "_frozen_backbone", False):
        model.features.eval()


def _validate_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[2:]) < 32:
        raise ValueError("images must be RGB tensors with at least 32 pixels per side")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0] or not labels.numel():
        raise ValueError("labels must have one value per image")
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be finite binary values")


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
    set_training_mode(model)
    loss_total = 0.0
    sample_count = 0

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        _validate_batch(images, labels)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        if logits.shape != labels.shape or not torch.isfinite(logits).all():
            raise ValueError("invalid training logits shape or non-finite values")
        loss = criterion(logits, labels)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("non-finite training loss or invalid loss shape")
        loss.backward()
        optimizer.step()

        batch_count = labels.numel()
        loss_total += float(loss.detach().cpu()) * batch_count
        sample_count += batch_count

    if not sample_count:
        raise ValueError("empty training dataloader")
    return loss_total / sample_count


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_total = 0.0
    sample_count = 0

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        _validate_batch(images, labels)
        logits = model(images).view(-1)
        if logits.shape != labels.shape or not torch.isfinite(logits).all():
            raise ValueError("invalid evaluation logits shape or non-finite values")
        loss = criterion(logits, labels)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("non-finite evaluation loss or invalid loss shape")
        batch_count = labels.numel()
        loss_total += float(loss.detach().cpu()) * batch_count
        sample_count += batch_count

    if not sample_count:
        raise ValueError("empty evaluation dataloader")
    return loss_total / sample_count


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
    if isinstance(n_passes, bool) or not isinstance(n_passes, int) or n_passes < 2:
        raise ValueError("MC dropout needs an integer pass count of at least two")
    modes = [(module, module.training) for module in model.modules()]
    all_samples = []
    identity = None
    try:
        model.eval()
        set_dropout_layers_to_train(model)
        for _ in range(n_passes):
            pass_probabilities, pass_labels, ids, paths = [], [], [], []
            for images, labels, batch_ids, batch_paths in dataloader:
                if not torch.isfinite(images).all():
                    raise ValueError("MC inputs must be finite")
                logits = model(images.to(device)).view(-1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
                if (
                    not np.isfinite(logits.detach().cpu().numpy()).all()
                    or targets.shape != probs.shape
                    or not np.isin(targets, [0, 1]).all()
                    or len(batch_ids) != len(probs)
                    or len(batch_paths) != len(probs)
                ):
                    raise ValueError("invalid MC batch outputs or identity")
                pass_probabilities.append(probs)
                pass_labels.extend(targets.tolist())
                ids.extend(batch_ids)
                paths.extend(batch_paths)
            if (
                not ids
                or len(set(ids)) != len(ids)
                or len(set(paths)) != len(paths)
                or any(not isinstance(i, str) or not i for i in ids + paths)
            ):
                raise ValueError("MC pass needs nonempty unique sample IDs and paths")
            current = (ids, paths, pass_labels)
            if identity is not None and current != identity:
                raise ValueError("MC pass identity/order differs across passes")
            identity = current
            all_samples.append(np.concatenate(pass_probabilities))
    finally:
        # Assign flags directly so a parent's train() cannot overwrite children.
        for module, training in modes:
            module.training = training
    samples = np.stack(all_samples, axis=0)
    return {
        "all_probabilities": samples,
        "mean_probability": samples.mean(axis=0),
        "uncertainty": samples.std(axis=0, ddof=1),
        "label": np.asarray(identity[2], dtype=int),
        "image_id": identity[0],
        "image_path": identity[1],
    }
