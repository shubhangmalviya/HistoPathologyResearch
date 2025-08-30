import os
import argparse
from typing import List, Dict

import numpy as np
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
def eval_on_loader(model: UNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    dices: List[float] = []
    ajis: List[float] = []
    pqs: List[float] = []
    f1s: List[float] = []

    model.to(device)
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts = targets.numpy()

        for pred_sem, gt_sem in zip(preds, gts):
            boundary = (pred_sem != np.pad(pred_sem, 1, mode='edge')[1:-1, 1:-1]).astype(np.uint8)
            pred_inst = reconstruct_instances(pred_sem, boundary)
            gt_boundary = (gt_sem != np.pad(gt_sem, 1, mode='edge')[1:-1, 1:-1]).astype(np.uint8)
            gt_inst = reconstruct_instances(gt_sem, gt_boundary)

            dices.append(dice_coefficient(pred_sem, gt_sem))
            ajis.append(aji_aggregated_jaccard(gt_inst, pred_inst))
            pqs.append(pq_panoptic(gt_inst, pred_inst))
            f1s.append(f1_object(gt_inst, pred_inst))

    return {
        "Dice": float(np.mean(dices)) if dices else 0.0,
        "AJI": float(np.mean(ajis)) if ajis else 0.0,
        "PQ": float(np.mean(pqs)) if pqs else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tissues_root", type=str, default=dataset_root_tissues())
    parser.add_argument("--tissues", type=str, nargs="+", default=["Breast", "Colon", "Adrenal_gland", "Esophagus", "Bile-duct"])
    parser.add_argument("--experts_ckpts_root", type=str, default=checkpoints_dir("experts"))
    parser.add_argument("--unified_ckpt", type=str, default=os.path.join(checkpoints_dir("unified"), "unet_unified.pt"))
    parser.add_argument("--save_csv", type=str, default=os.path.join(results_dir(), "metrics.csv"))
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs_exist()
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    expert_ckpts = {t: os.path.join(args.experts_ckpts_root, f"unet_{t}.pt") for t in args.tissues}

    rows: List[str] = ["model,tissue,Dice,AJI,PQ,F1"]

    for tissue in args.tissues:
        test_ds = PanNukeTissueDataset(os.path.join(args.dataset_tissues_root, tissue), split="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        expert_model = load_checkpoint(expert_ckpts[tissue])
        metrics = eval_on_loader(expert_model, test_loader, device)
        rows.append(f"expert,{tissue},{metrics['Dice']:.4f},{metrics['AJI']:.4f},{metrics['PQ']:.4f},{metrics['F1']:.4f}")

    unified_model = load_checkpoint(args.unified_ckpt)
    for tissue in args.tissues:
        test_ds = PanNukeTissueDataset(os.path.join(args.dataset_tissues_root, tissue), split="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        metrics = eval_on_loader(unified_model, test_loader, device)
        rows.append(f"unified,{tissue},{metrics['Dice']:.4f},{metrics['AJI']:.4f},{metrics['PQ']:.4f},{metrics['F1']:.4f}")

    with open(args.save_csv, "w") as f:
        f.write("\n".join(rows))
    print(f"Saved metrics to {args.save_csv}")


if __name__ == "__main__":
    main()


