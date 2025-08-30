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
from src.utils.paths import checkpoints_dir, results_dir, dataset_root_tissues, ensure_dirs_exist


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
    for batch_idx, (images, targets) in enumerate(loader):
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
    parser.add_argument("--experts_ckpts_root", type=str, default=checkpoints_dir("experts"))
    parser.add_argument("--unified_ckpt", type=str, default=os.path.join(checkpoints_dir("unified"), "unet_unified.pt"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_csv", type=str, default=os.path.join(results_dir(), "per_image_metrics.csv"))
    parser.add_argument("--fold", type=int, default=-1, help="Optional fold id for OOF tagging (-1 if not OOF)")
    args = parser.parse_args()

    ensure_dirs_exist()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    unified_model = load_checkpoint(args.unified_ckpt)

    records: List[Dict[str, object]] = []

    for tissue in args.tissues:
        expert_ckpt = os.path.join(args.experts_ckpts_root, f"unet_{tissue}.pt")
        expert_model = load_checkpoint(expert_ckpt)

        test_ds = PanNukeTissueDataset(os.path.join(args.dataset_tissues_root, tissue), split="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        expert_rows = compute_metrics_for_loader(expert_model, test_loader, device)
        unified_rows = compute_metrics_for_loader(unified_model, test_loader, device)

        n = min(len(expert_rows), len(unified_rows))
        for idx in range(n):
            image_id = f"{tissue}/test/{idx:06d}"
            e = expert_rows[idx]
            u = unified_rows[idx]
            for model_name, row in (("expert", e), ("unified", u)):
                rec = {
                    "image_id": image_id,
                    "tissue": tissue,
                    "model": model_name,
                    "fold": int(args.fold),
                    "PQ": row["PQ"],
                    "Dice": row["Dice"],
                    "AJI": row["AJI"],
                    "F1": row["F1"],
                }
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote per-image metrics to {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()


