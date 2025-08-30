# PanNuke Preprocessing Module

This module provides a logically structured preprocessing pipeline for the PanNuke dataset, suitable for ML lifecycle workflows.

## Structure
- `pannuke_preprocessor.py`: Main class for loading, processing, and saving images and masks.
- `__main__.py`: CLI entry point for running preprocessing from the command line.
- `__init__.py`: Makes this folder a package.

## Usage

### As a Python Module
```python
from preprocessing.pannuke_preprocessor import PanNukePreprocessor
preprocessor = PanNukePreprocessor(data_dir='path/to/data', output_dir='path/to/output')
preprocessor.process()
```

### From the Command Line
```bash
python -m preprocessing --data_dir path/to/data --output_dir path/to/output
```

```bash
python -m src.preprocessing --data_dir './data' --output_dir './output'
```

## Output Structure
The processed data will be saved in the following format:
```
output_dir/
  tissue_type_1/
    images/
    sem_masks/
    inst_masks/
  tissue_type_2/
    ...
```

- `images/`: Raw images
- `sem_masks/`: Semantic masks (class labels)
- `inst_masks/`: Instance boundary masks

## Notes
- The code is inspired by the original scripts in `pre-processing/scripts` but is refactored for clarity, modularity, and maintainability.
- All major steps are encapsulated in the `PanNukePreprocessor` class for easy reuse and extension.


Ran tool
Here’s a detailed explanation of what the process is doing, with inline documentation now added to the code:

### What the Process Does

- **Iterates over all folds** (e.g., Fold 1, Fold 2, Fold 3) in your PanNuke dataset.
- **Loads images, masks, and tissue type labels** from `.npy` files for each fold.
- For each sample in each fold, it generates:
  - The **raw image**.
  - The **semantic mask** (class labels, with background always at channel 0).
  - The **instance boundary mask** (useful for instance segmentation).
- **Saves each output grouped by tissue type** (for ML convenience), but filenames include the fold number and sample index for full lineage traceability (e.g., `sem_{tissue_type}_{fold_num}_{k:05d}.png`).

### Why Merge All Folds into One Output Directory?

- **ML pipelines benefit from a unified structure**: It’s easier to split, shuffle, and use the data for training/validation/testing when all samples are accessible in a single place, grouped by tissue type.
- **Lineage is preserved**: The fold number and sample index are encoded in every output filename, so you can always trace any image/mask back to its original fold and index. No information is lost.

### Where Did the Lineage Information Go?

- The **fold number** is extracted from the folder name (e.g., `Fold 1` → `1`) and is included in every output filename.
- The **sample index** within the fold is also included in the filename.
- This means you can always reconstruct which fold and which sample any output file came from, even though the files are grouped by tissue type for ML convenience.

### Example Output Structure

```
output/
  0/
    images/
      img_0_1_00000.png  # tissue_type 0, fold 1, sample 0
      img_0_2_00001.png  # tissue_type 0, fold 2, sample 1
    sem_masks/
      sem_0_1_00000.png
    inst_masks/
      inst_0_1_00000.png
  1/
    ...
```

### Code Improvements

- **Inline comments** now explain each step and the rationale behind it.
- The **docstrings** clarify why certain design decisions were made (e.g., merging, output structure, and how lineage is preserved).


## Standardized Commands

Use the scripts or Makefile to run the end-to-end workflow.

### Using scripts/

```bash
# 1) Create unified + per-tissue splits
./scripts/split_all.sh

# 2) Train 5 expert models (override with EPOCHS, BATCH_SIZE, LR)
EPOCHS=5 BATCH_SIZE=16 LR=5e-4 ./scripts/train_experts.sh

# 3) Train unified model over the 5 tissues
EPOCHS=5 BATCH_SIZE=16 LR=5e-4 ./scripts/train_unified.sh

# 4) Evaluate experts vs unified and run statistical tests
./scripts/eval_and_stats.sh

# 5) Individual helpers
./scripts/eval.sh
./scripts/stats.sh
```

### Using Makefile

```bash
make splits   # runs scripts/split_all.sh
make experts  # runs scripts/train_experts.sh
make unified  # runs scripts/train_unified.sh
make eval     # runs scripts/eval_and_stats.sh
make stats    # runs the stats module only
make all      # splits + experts + unified + eval
```

### Direct Python invocations (RQ2 layout)

```bash
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/datasets/split_pannuke.py
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/datasets/split_by_tissue.py
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/training/train_expert_unet.py --tissue Breast --epochs 1 --batch_size 16 --lr 5e-4
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/training/train_unified_unet.py --epochs 1 --batch_size 16 --lr 5e-4 --tissues Breast Colon Adrenal_gland Esophagus Bile-duct
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/evaluation/eval_rq2.py --batch_size 16
PYTHONPATH="$(pwd)/src" "$(pwd)/venv/bin/python" src/stats/compare_expert_vs_unified.py
```

Artifacts (RQ2):
- Checkpoints: `artifacts/rq2/checkpoints/unified/`, `artifacts/rq2/checkpoints/experts/`
- Datasets: `dataset/`, `dataset_tissues/`
- Metrics/Stats: `artifacts/rq2/results/metrics.csv`, `artifacts/rq2/results/wilcoxon.csv`, `artifacts/rq2/results/ttest.csv`, `artifacts/rq2/results/stats_report.md`
