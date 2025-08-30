import os
from typing import Callable, Optional, Tuple, List

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class PanNukeTissueDataset(Dataset):
    """
    Per-tissue PanNuke dataset.

    Expects directory layout:
      root_dir/
        images/
        sem_masks/
        inst_masks/  (optional, not required for training)

    Images are RGB PNGs. Semantic masks are single-channel PNGs with class indices [0..num_classes-1].
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_instance_boundaries: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, split, "images")
        self.sem_dir = os.path.join(root_dir, split, "sem_masks")
        self.inst_dir = os.path.join(root_dir, split, "inst_masks")
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.use_instance_boundaries = use_instance_boundaries

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.sem_dir):
            raise FileNotFoundError(f"Missing images/sem_masks under {root_dir}/{split}")

        self.image_files = sorted(os.listdir(self.images_dir))

        # Verify pairs exist by mapping stem suffixes
        self.pairs: List[Tuple[str, str, Optional[str]]] = []
        for img_name in self.image_files:
            if not img_name.endswith('.png'):
                continue
            sem_name = img_name.replace('img_', 'sem_', 1)
            inst_name = img_name.replace('img_', 'inst_', 1)
            sem_path = os.path.join(self.sem_dir, sem_name)
            if not os.path.isfile(sem_path):
                # Fallback: try same basename
                sem_name = img_name
                sem_path = os.path.join(self.sem_dir, sem_name)
                if not os.path.isfile(sem_path):
                    raise FileNotFoundError(f"Semantic mask not found for {img_name}")
            inst_path = os.path.join(self.inst_dir, inst_name) if os.path.isdir(self.inst_dir) else None
            self.pairs.append((img_name, sem_name, inst_path if (inst_path and os.path.isfile(inst_path)) else None))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_name, sem_name, _ = self.pairs[index]
        img_path = os.path.join(self.images_dir, img_name)
        sem_path = os.path.join(self.sem_dir, sem_name)

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
        # Convert to float tensor in [0,1] and normalize with ImageNet mean/std (reasonable default)
        arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 3
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def default_target_transform(mask_img: Image.Image) -> torch.Tensor:
        arr = np.array(mask_img, dtype=np.uint8)  # H, W
        mask = torch.from_numpy(arr).long()
        return mask



