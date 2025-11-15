import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ColorFolderDataset(Dataset):
    """Dataset that reads images from a folder-of-folders structure.

    root/
      0N05 Fawn/
        key.png
        brightness_1_0.3.png
        ...
      0P13 Pink Plume/
        ...
    """

    def __init__(self, root: Path, image_size: int = 64):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.image_paths: List[Path] = []
        self.labels: List[str] = []

        # Gather all PNG images and their labels
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir():
                label = folder.name
                pngs = sorted(folder.glob("*.png"))
                for img_path in pngs:
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        if not self.image_paths:
            raise RuntimeError(f"No PNG images found under {self.root}")

        unique_labels = sorted(set(self.labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.targets = [self.label2idx[l] for l in self.labels]

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # We keep values in [0,1]; no mean/std normalization is necessary for flat colors
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = self.targets[idx]
        return x, y
