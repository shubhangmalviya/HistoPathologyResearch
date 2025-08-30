import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.pannuke_tissue_dataset import PanNukeTissueDataset
from src.models.unet import UNet
from src.metrics.seg_metrics import dice_coefficient, reconstruct_instances, aji_aggregated_jaccard, pq_panoptic, f1_object
from src.utils.paths import checkpoints_oof_dir, results_dir, dataset_root_tissues, ensure_dirs_exist


def load_checkpoint(ckpt_path: str, num_classes: int = 7) -> UNet:
    model = UNet(in_channels=3, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def compute_metrics_for_loader(model: UNet, loader: DataLoader, device: torch.device) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    model.to(device)
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts = targets.numpy()
        bs = preds.shape[0]
        for i in range(bs):
            pred_sem = preds[i]
            gt_sem = gts[i]
            boundary = (pred_sem != np.pad(pred_sem, 1, mode='edge')[1:-1, 1:-1]).astype(np.uint8)
            pred_inst = reconstruct_instances(pred_sem, boundary)
            gt_boundary = (gt_sem != np.pad(gt_sem, 1, mode='edge')[1:-1, 1:-1]).astype(np.uint8)
            gt_inst = reconstruct_instances(gt_sem, gt_boundary)
            rows.append({
                "Dice": dice_coefficient(pred_sem, gt_sem),
                "AJI": aji_aggregated_jaccard(gt_inst, pred_inst),
                "PQ": pq_panoptic(gt_inst, pred_inst),
                "F1": f1_object(gt_inst, pred_inst),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tissues_root", type=str, default=dataset_root_tissues())
    parser.add_argument("--tissues", type=str, nargs="+", default=["Breast", "Colon", "Adrenal_gland", "Esophagus", "Bile-duct"])
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_csv", type=str, default=os.path.join(results_dir(), "per_image_metrics_oof.csv"))
    args = parser.parse_args()

    ensure_dirs_exist()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    records: List[Dict[str, object]] = []

    for tissue in args.tissues:
        # For each fold, load the expert checkpoint for this tissue
        oof_dir = os.path.join(checkpoints_oof_dir("experts"), tissue)
        for fold in range(args.n_splits):
            ckpt_path = os.path.join(oof_dir, f"unet_{tissue}_fold{fold}.pt")
            if not os.path.isfile(ckpt_path):
                continue
            model = load_checkpoint(ckpt_path)
            # Evaluate on the validation fold images: use val split as a proxy for fold
            # In a real OOF, we'd persist the exact indices; here we evaluate on val split to avoid leakage from train.
            val_ds = PanNukeTissueDataset(os.path.join(args.dataset_tissues_root, tissue), split="val")
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            rows = compute_metrics_for_loader(model, loader, device)
            for idx, row in enumerate(rows):
                image_id = f"{tissue}/fold{fold}/val/{idx:06d}"
                records.append({
                    "image_id": image_id,
                    "tissue": tissue,
                    "model": "expert",
                    "fold": fold,
                    **row,
                })

    # Note: unified OOF not applicable; unified model is trained on all tissues. For pairing,
    # evaluate unified on the same val splits for comparability
    unified_ckpt_dir = checkpoints_oof_dir("unified")
    unified_ckpt = os.path.join(unified_ckpt_dir, "unet_unified.pt")
    if os.path.isfile(unified_ckpt):
        unified_model = load_checkpoint(unified_ckpt)
        for tissue in args.tissues:
            val_ds = PanNukeTissueDataset(os.path.join(args.dataset_tissues_root, tissue), split="val")
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            rows = compute_metrics_for_loader(unified_model, loader, device)
            for idx, row in enumerate(rows):
                image_id = f"{tissue}/val/{idx:06d}"
                records.append({
                    "image_id": image_id,
                    "tissue": tissue,
                    "model": "unified",
                    "fold": -1,
                    **row,
                })

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote OOF per-image metrics to {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()


