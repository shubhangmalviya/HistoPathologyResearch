import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.pannuke_tissue_dataset import PanNukeTissueDataset
from src.models.unet import UNet
from src.utils.paths import checkpoints_dir, dataset_root_tissues, ensure_dirs_exist


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    preds = torch.softmax(logits, dim=1)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(preds * targets_onehot, dim=(0, 2, 3))
    union = torch.sum(preds + targets_onehot, dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train_one_epoch(model, loader, optimizers, device):
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizers.zero_grad(set_to_none=True)
        logits = model(images)
        ce = ce_criterion(logits, targets)
        d = dice_loss(logits, targets)
        loss = ce + d
        loss.backward()
        optimizers.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ce_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            ce = ce_criterion(logits, targets)
            d = dice_loss(logits, targets)
            loss = ce + d
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def get_concat_loaders(dataset_root: str, tissues: List[str], batch_size: int = 16, num_workers: int = 2):
    train_datasets = [PanNukeTissueDataset(os.path.join(dataset_root, t), split="train") for t in tissues]
    val_datasets = [PanNukeTissueDataset(os.path.join(dataset_root, t), split="val") for t in tissues]
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tissues_root", type=str, default=dataset_root_tissues())
    parser.add_argument("--tissues", type=str, nargs="+", default=["Breast", "Colon", "Adrenal_gland", "Esophagus", "Bile-duct"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default=checkpoints_dir("unified"))
    args = parser.parse_args()

    ensure_dirs_exist()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, val_loader = get_concat_loaders(args.dataset_tissues_root, args.tissues, batch_size=args.batch_size)

    model = UNet(in_channels=3, num_classes=args.num_classes)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.save_dir, f"unet_unified.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "tissues": args.tissues,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()


