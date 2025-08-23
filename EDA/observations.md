# EDA Observations

### Observations: Image Counts per Tissue Type
- Breast dominates the dataset size, followed by Colon and HeadNeck.
- Several tissues (e.g., Kidney, Bladder, Stomach, Ovarian) have far fewer samples, indicating potential imbalance.
- Consider stratified sampling or class-balanced sampling for training/validation splits.


### Observations: Fold Distribution
- Folds are generally well represented across tissues after fixing fold parsing.
- Some tissues show skewed fold counts (e.g., Uterus), suggesting careful attention during cross-validation setup.
- For fair benchmarking, ensure tissue-wise fold balance when evaluating models.


### Observations: Semantic Class Distribution (Background Excluded)
- Epithelial and Inflammatory classes are prevalent across samples, with Neoplastic varying by tissue.
- Rare classes (e.g., Dead) have significantly fewer pixels, indicating class imbalance that may require loss re-weighting or augmentation.
- For RQ2, consider whether class-specialized experts can better model rare class morphology.


### Observations: Instance Count Distribution
- Mean instances per image ~ reported in the figure; high variance suggests heterogeneous cell densities across tissues.
- For RQ1/RQ2 comparisons, normalize metrics by density where appropriate (e.g., per-instance metrics alongside per-image).


### Observations: RQ2 Per-Tissue Class Ratios
- Clear tissue-specific differences in class composition (e.g., higher Epithelial in Breast), reinforcing the case for expert models.
- Unified models may be biased toward dominant classes/tissues; ensembling experts could mitigate this.


### Observations: RQ3 Color Variability (HED/RGB proxy)
- Notable inter-tissue variability suggests stain normalization (e.g., Vahadane) could stabilize color distributions.
- For RQ3 evaluation, compute mPQ/Dice per tissue before vs. after normalization to measure gains.


