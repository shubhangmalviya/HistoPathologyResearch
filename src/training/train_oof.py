import os
import argparse
from typing import List

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.datasets.pannuke_tissue_dataset import PanNukeTissueDataset
from src.models.unet import UNet
from src.utils.paths import dataset_root_tissues, checkpoints_oof_dir, ensure_dirs_exist


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    preds = torch.softmax(logits, dim=1)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(preds * targets_onehot, dim=(0, 2, 3))
    union = torch.sum(preds + targets_onehot, dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = ce_criterion(logits, targets) + dice_loss(logits, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tissues_root", type=str, default=dataset_root_tissues())
    parser.add_argument("--tissue", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--n_splits", type=int, default=3)
    args = parser.parse_args()

    ensure_dirs_exist()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Build index list from all splits for this tissue
    tissue_root = os.path.join(args.dataset_tissues_root, args.tissue)
    full_train = PanNukeTissueDataset(tissue_root, split="train")
    full_val = PanNukeTissueDataset(tissue_root, split="val")
    # Use combined train+val to define OOF folds
    combined_items = list(range(len(full_train))) + [len(full_train) + i for i in range(len(full_val))]
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(combined_items)):
        # Map indices back to datasets
        def map_idx(idx: List[int]) -> List[int]:
            mapped: List[int] = []
            for i in idx:
                if i < len(full_train):
                    mapped.append(i)
                else:
                    mapped.append(i - len(full_train) + len(full_train))
            return mapped

        # Create subsets from concatenation of train/val
        concat_images = torch.utils.data.ConcatDataset([full_train, full_val])
        train_subset = Subset(concat_images, train_idx.tolist())
        val_subset = Subset(concat_images, val_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        best_val = float("inf")
        for epoch in range(args.epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            # quick val pass to save ckpt
            model.eval()
            ce = nn.CrossEntropyLoss()
            vloss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    vloss += (ce(logits, y) + dice_loss(logits, y)).item() * x.size(0)
            vloss /= len(val_loader.dataset)
            if vloss < best_val:
                best_val = vloss
                out_dir = os.path.join(checkpoints_oof_dir("experts"), args.tissue)
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, f"unet_{args.tissue}_fold{fold}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": vloss,
                    "fold": fold,
                    "tissue": args.tissue,
                }, ckpt_path)
                print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()


