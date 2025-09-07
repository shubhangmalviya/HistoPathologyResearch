### RQ1: SAM Variants vs Established Models on PanNuke

This study evaluates whether Segment Anything Model (SAM) variants (ViT-B/L/H, optional PathoSAM) achieve competitive or superior nuclei instance segmentation on PanNuke compared to established models and a U-Net baseline.

#### Notebook
- Run `src/rq1_sam_variants_comparison_v2.ipynb`

#### Data layout
- `dataset_tissues/<Tissue>/{train,val,test}/images/*.png`
- `dataset_tissues/<Tissue>/{train,val,test}/sem_masks/*.png`

Instance masks are reconstructed from semantic masks with simple boundary-based splitting for fair, model-agnostic instance metrics.

#### Models
- U-Net baseline checkpoint (optional): `artifacts/rq3_enhanced/checkpoints/unet_original_enhanced_best.pth`
- SAM variants require `segment-anything` installed; checkpoints are optional (registry defaults used otherwise).
- HoVer-Net (official): install `tiatoolbox` (`pip install tiatoolbox`) to enable the TIAToolbox HoVer‑Net integration (pretrained `hovernet_fast-pannuke`).

#### Metrics
- PQ (panoptic quality), object F1, AJI, binary Dice

#### Outputs
- Per-image metrics CSV: `reports/rq1/tables/per_image_instance_metrics.csv`
- Pairwise stats (BH-corrected): `reports/rq1/tables/pairwise_stats_bh.csv`
- Figures: `reports/rq1/figures/*.png`
- HTML summary: `reports/rq1/RQ1_SAM_Variants_Report.html`

#### Notes
- The notebook gates unavailable models gracefully.
- To add HoVer-Net/CellViT/LKCell, plug in their inference to produce instance maps and append to `MODELS`.


