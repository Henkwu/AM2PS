from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class SharedViewTransform:
    """Apply one geometric/photometric transform, then create ResNet and CLIP-normalized views."""

    def __init__(
        self,
        image_size: int = 224,
        augment: bool = False,
        rotation_degrees: float = 10.0,
        color_jitter: float = 0.10,
    ) -> None:
        ops: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
        if augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=rotation_degrees),
                    transforms.ColorJitter(
                        brightness=color_jitter,
                        contrast=color_jitter,
                        saturation=color_jitter,
                        hue=min(0.05, color_jitter / 2),
                    ),
                ]
            )
        self.shared = transforms.Compose(ops)
        self.to_tensor = transforms.ToTensor()
        self.backbone_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.clip_norm = transforms.Normalize(CLIP_MEAN, CLIP_STD)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.shared(image.convert("RGB"))
        tensor = self.to_tensor(image)
        return self.backbone_norm(tensor.clone()), self.clip_norm(tensor.clone())


class DualViewImageFolder(Dataset):
    def __init__(self, root: str | Path, transform: SharedViewTransform) -> None:
        base = ImageFolder(root=str(root))
        self.samples = base.samples
        self.targets = base.targets
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        path, target = self.samples[idx]
        with Image.open(path) as img:
            backbone_x, clip_x = self.transform(img)
        return {
            "backbone": backbone_x,
            "clip": clip_x,
            "target": torch.tensor(target, dtype=torch.long),
            "path": path,
        }


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


def _split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
    return train_subset, val_subset


def _assert_same_classes(reference: list[str], ds: DualViewImageFolder, split: str) -> None:
    if list(reference) != list(ds.classes):
        raise ValueError(
            f"Class mismatch for {split}: expected {reference}, found {ds.classes}. "
            "All splits must use the same class folders."
        )


def make_dataloaders(cfg: dict, seed: int = 42) -> DataBundle:
    dcfg = cfg["data"]
    root = Path(dcfg["root"]).expanduser()
    image_size = int(dcfg.get("image_size", 224))
    use_da = bool(cfg.get("components", {}).get("data_augmentation", True))

    train_tf = SharedViewTransform(
        image_size=image_size,
        augment=use_da,
        rotation_degrees=float(dcfg.get("rotation_degrees", 10.0)),
        color_jitter=float(dcfg.get("color_jitter", 0.10)),
    )
    eval_tf = SharedViewTransform(image_size=image_size, augment=False)

    train_dir = root / dcfg.get("train_dir", "train")
    val_dir = root / dcfg.get("val_dir", "val")
    test_dir = root / dcfg.get("test_dir", "test")

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected at least train and test directories under {root}. "
            f"Missing: {[str(p) for p in (train_dir, test_dir) if not p.exists()]}"
        )

    train_full = DualViewImageFolder(train_dir, train_tf)
    class_names = list(train_full.classes)

    if val_dir.exists():
        val_ds = DualViewImageFolder(val_dir, eval_tf)
        _assert_same_classes(class_names, val_ds, "val")
        train_ds: Dataset = train_full
    else:
        # The paper does not specify a validation split for ChestXRay-Covid19.
        # We create one deterministically from training data to avoid tuning on the test set.
        train_ds, val_indices = _split_dataset(train_full, float(dcfg.get("val_fraction", 0.10)), seed)
        eval_train_full = DualViewImageFolder(train_dir, eval_tf)
        val_ds = Subset(eval_train_full, val_indices.indices)

    test_ds = DualViewImageFolder(test_dir, eval_tf)
    _assert_same_classes(class_names, test_ds, "test")

    loader_cfg = cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 16))
    workers = int(loader_cfg.get("num_workers", 4))
    pin = bool(loader_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    def loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin,
            drop_last=False,
        )

    return DataBundle(
        train=loader(train_ds, True),
        val=loader(val_ds, False),
        test=loader(test_ds, False),
        class_names=class_names,
    )
