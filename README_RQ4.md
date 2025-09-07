### RQ4: Do lightweight explainability techniques (Grad-CAM) enhance interpretability of U-Net-based nuclei segmentation on PanNuke, and does stain normalization improve this further?

#### Overview
This study evaluates whether Grad-CAM can yield spatially meaningful explanations for U-Net-based nuclei segmentation on PanNuke and whether Vahadane stain normalization improves those explanations.

- **Null Hypothesis (H0)**: Stain normalization does not improve Grad-CAM alignment.
- **Alternative Hypothesis (H1)**: Stain normalization significantly improves Grad-CAM alignment.

The analysis compares Grad-CAM alignment on the same images under two conditions: original images vs normalized images. We quantify alignment against ground-truth nuclei masks and test H1 with paired non-parametric statistics.

#### What is Grad-CAM and how we use it here
Grad-CAM highlights image regions that most influence a model’s prediction by backpropagating gradients to a chosen convolutional layer and aggregating channel-wise importance.

- **Backbone**: U-Net (encoder–decoder) segmentation network.
- **Target layer**: The last convolutional block(s) in the decoder (closest to the output), where spatial detail is rich.
- **Procedure**:
  - Forward pass: get activations from the target layer.
  - Backward pass: compute gradients of the prediction wrt those activations.
  - Aggregate gradients via global average pooling to obtain per-channel weights.
  - Weighted sum of activations → ReLU → normalize to [0, 1] to obtain a Grad-CAM heatmap.
- **For segmentation**: We follow a lightweight adaptation—targeting the foreground logit (nuclei vs background) to produce a single saliency map per image and compare this map to nuclei masks.

#### What we show with EDA (and why)
EDA provides quick sanity checks that the evaluation setting is appropriate for explainability measurement.

- **Per-tissue counts**: Images available in `train/val/test` across tissues to ensure sampling is broad and balanced enough.
- **Sample visuals**: Random examples with masks for face-validity and to corroborate correct pairings and naming conventions.
- **Nuclei coverage**: Distribution of foreground (nuclei) fraction; Grad-CAM alignment is only meaningful if sufficient foreground exists.

These checks prevent biased conclusions due to data sparsity, path issues, or mismatched pairs.

#### How RQ3 checkpoints help RQ4
RQ4 uses the trained U-Net weights from RQ3 to ensure explanations are assessed on competent models rather than randomly initialized ones.

- **Checkpoints**: `artifacts/rq3_enhanced/checkpoints/best_model_original.pth` and `best_model_normalized.pth`.
- **Rationale**: RQ3 optimized nuclei segmentation under original and normalized pipelines. Reusing these checkpoints allows a direct comparison of explainability under both preprocessing settings without re-training.
- **Outcome**: Any observed difference in Grad-CAM alignment can be tied to the stain normalization preprocessing rather than training instability.

#### Metrics for Grad-CAM alignment
We quantify how well Grad-CAM heatmaps align with ground-truth nuclei masks using complementary criteria:

- **Energy-in-mask**: Proportion of Grad-CAM energy within nuclei regions. Higher is better.
- **Pointing game**: 1 if the heatmap’s peak lies inside nuclei, else 0. Higher is better.
- **IoU@0.5**: IoU between binarized heatmap (threshold=0.5) and nuclei mask. Higher is better.

These capture distributional alignment (energy), precision of the most salient point (pointing), and overlap when treated as a mask (IoU).

#### Statistical testing and why
We evaluate the same images under both conditions (paired setting). Distributional assumptions may not hold for alignment scores, so we choose a non-parametric paired test.

- **Test**: Paired Wilcoxon signed-rank test with one-sided alternative (H1: normalized > original).
- **Why Wilcoxon**: Robust to non-normality; appropriate for paired differences; sensitive to median shifts.
- **Effect size**: Report r = Z / sqrt(N) for interpretability alongside p-values.
- **Interpretation**: Significant p-values with positive mean/median deltas support rejecting H0 in favor of H1.

#### Data format and directory layout
The code expects PanNuke-style splits with consistent naming:

- Root datasets:
  - `dataset/` for unified splits.
  - `dataset_tissues/<Tissue>/(train|val|test)/` for per-tissue splits.
- Inside each split:
  - `images/`: RGB `.png` images named like `img_<...>.png`.
  - `sem_masks/`: semantic masks named `sem_<...>.png` or same basename if `sem_` prefix not present.
  - `inst_masks/`: optional instance masks (not required for RQ4).
- Pairing rule: `img_*.png` ↔ `sem_*.png` (fallback to same basename if needed).

#### How to run
- Entry point notebook: `src/rq4_explainability_analysis.ipynb`.
- Dependencies: see `requirements.txt` (GPU recommended). The notebook auto-detects CUDA; it will still run on CPU, albeit slower.
- Checkpoints: place RQ3 weights under `artifacts/rq3_enhanced/checkpoints/` with names used above (or adjust the paths in the notebook).

Typical workflow in the notebook:
1) Setup and EDA (counts, samples, nuclei coverage)
2) Load models (original vs normalized), initialize GPU Vahadane normalizer
3) Generate Grad-CAM maps for sampled test images
4) Compute alignment metrics and aggregate results
5) Run paired Wilcoxon tests and report effect sizes
6) Visualize overlays and summarize findings

#### Outputs
- Tables and plots summarizing per-metric deltas and by-tissue aggregates.
- Paired statistical test results (p-values, effect sizes).
- Qualitative overlays comparing Grad-CAM on original vs normalized inputs.

#### Notes and limitations
- Grad-CAM layer choice affects spatial specificity; we use the last decoder conv where features are high-resolution.
- For multi-class outputs, we target the foreground logit to align with nuclei presence.
- Thresholds (e.g., IoU@0.5) are conventional but can be sensitivity-tested.
- Conclusions are specific to PanNuke tiles and the RQ3-trained U-Net; generalization to other datasets/models should be validated.


