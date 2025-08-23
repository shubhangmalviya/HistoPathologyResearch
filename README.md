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

If you need further clarification or want to see the updated code, let me know!