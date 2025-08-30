import os
from typing import Callable, Optional, Tuple, List, Dict

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedImageMaskListDataset(Dataset):
    """
    Dataset from an explicit list of (image_path, sem_path) items.
    Items may also carry meta like tissue; it is not returned by __getitem__.
    """

    def __init__(
        self,
        items: List[Dict[str, str]],
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.items = items
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        entry = self.items[index]
        img_path = entry["image_path"]
        sem_path = entry["sem_path"]

        image = Image.open(img_path).convert('RGB')
        sem_mask = Image.open(sem_path)

        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = self.default_image_transform(image)

        if self.target_transform:
            target = self.target_transform(sem_mask)
        else:
            target = self.default_target_transform(sem_mask)

        return image, target

    @staticmethod
    def default_image_transform(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def default_target_transform(mask_img: Image.Image) -> torch.Tensor:
        arr = np.array(mask_img, dtype=np.uint8)
        mask = torch.from_numpy(arr).long()
        return mask


def gather_all_items_for_tissue(tissue_root: str) -> List[Dict[str, str]]:
    """
    Scan train/val/test under a tissue root and return items with full paths.
    """
    items: List[Dict[str, str]] = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(tissue_root, split, "images")
        sem_dir = os.path.join(tissue_root, split, "sem_masks")
        if not (os.path.isdir(img_dir) and os.path.isdir(sem_dir)):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.endswith('.png'):
                continue
            img_path = os.path.join(img_dir, name)
            sem_name = name.replace('img_', 'sem_', 1)
            sem_path = os.path.join(sem_dir, sem_name)
            if not os.path.isfile(sem_path):
                sem_path = os.path.join(sem_dir, name)
                if not os.path.isfile(sem_path):
                    continue
            items.append({"image_path": img_path, "sem_path": sem_path})
    return items


