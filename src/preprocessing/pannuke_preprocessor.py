import os
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
from tqdm import tqdm

class PanNukePreprocessor:
    """
    Preprocessor for the PanNuke dataset. Handles loading, processing, and saving images and masks
    in a structure suitable for ML pipelines.

    This class processes all folds of the PanNuke dataset, extracts images and masks, and saves them
    in a unified output directory, grouped by tissue type. The fold number is preserved in the output
    filenames for full lineage traceability.
    """
    def __init__(self, data_dir, output_dir):
        """
        Args:
            data_dir (str): Path to the root directory containing PanNuke data (with Fold 1, Fold 2, ...)
            output_dir (str): Path to the directory where processed images and masks will be saved
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_folds(self):
        """
        Return a list of fold directories in the data directory.
        Only directories starting with 'Fold' are considered as valid folds.
        """
        return [d for d in os.listdir(self.data_dir) if d.startswith('Fold')]

    @staticmethod
    def get_boundaries(raw_mask):
        """
        Extract instance boundaries from the ground truth mask.
        Args:
            raw_mask (np.ndarray): Raw mask array (H, W, C)
        Returns:
            np.ndarray: Boundary mask (H, W)
        Why: Instance boundaries are useful for instance segmentation tasks and are commonly used as
        additional supervision in ML pipelines.
        """
        bdr = np.zeros(shape=raw_mask.shape)
        # The last channel is background, so we skip it
        for i in range(raw_mask.shape[-1] - 1):
            bdr[:, :, i] = find_boundaries(raw_mask[:, :, i], connectivity=1, mode='thick', background=0)
        bdr = np.sum(bdr, axis=-1)
        return bdr.astype(np.uint8)

    def process(self):
        """
        Main method to process all folds and save images/masks in a unified, ML-friendly structure.

        What it does:
        - Iterates over all folds (e.g., Fold 1, Fold 2, Fold 3)
        - Loads images, masks, and tissue type labels from .npy files for each fold
        - For each sample, generates:
            - The raw image
            - The semantic mask (class labels)
            - The instance boundary mask
        - Saves each output grouped by tissue type, but filenames include the fold number and sample index
          to preserve full lineage (e.g., sem_{tissue_type}_{fold_num}_{k:05d}.png)

        Why merge all folds into one output directory?
        - For most ML pipelines, it's more convenient to have all data accessible in a single structure,
          grouped by tissue type, rather than split by fold. This enables easier random splitting,
          cross-validation, and downstream processing.
        - The fold number is NOT lost: it is encoded in every output filename, so you can always trace
          any image/mask back to its original fold and index.
        """
        folds = self.get_folds()
        for fold in folds:
            # Extract fold number (e.g., Fold 1 -> 1)
            fold_num = ''.join(filter(str.isdigit, fold))
            fold_subfolder = f'fold{fold_num}'
            print(f"Processing {fold}...")
            img_path = os.path.join(self.data_dir, fold, 'images', fold_subfolder, 'images.npy')
            type_path = os.path.join(self.data_dir, fold, 'images', fold_subfolder, 'types.npy')
            mask_path = os.path.join(self.data_dir, fold, 'masks', fold_subfolder, 'masks.npy')

            # Load the data for this fold
            images = np.load(img_path, mmap_mode='r')
            masks = np.load(mask_path, mmap_mode='r')
            types = np.load(type_path)

            # Create output directories for each tissue type (if not already present)
            tissue_types = np.unique(types)
            for ttype in tissue_types:
                ttype_dir = os.path.join(self.output_dir, str(ttype))
                os.makedirs(os.path.join(ttype_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(ttype_dir, 'sem_masks'), exist_ok=True)
                os.makedirs(os.path.join(ttype_dir, 'inst_masks'), exist_ok=True)

            # Process and save each sample in this fold
            for k in tqdm(range(images.shape[0]), desc=f"Saving {fold}"):
                raw_image = images[k].astype(np.uint8)
                raw_mask = masks[k]
                # Semantic mask: class with highest probability/channel
                sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
                # Swap channels so background is always at channel 0
                sem_mask = np.where(sem_mask == 5, 6, sem_mask)
                sem_mask = np.where(sem_mask == 0, 5, sem_mask)
                sem_mask = np.where(sem_mask == 6, 0, sem_mask)
                tissue_type = types[k]
                # Instance mask: boundaries of each instance
                instances = self.get_boundaries(raw_mask)

                ttype_dir = os.path.join(self.output_dir, str(tissue_type))
                # Save with fold number and sample index for full traceability
                Image.fromarray(sem_mask).save(os.path.join(ttype_dir, 'sem_masks', f'sem_{tissue_type}_{fold_num}_{k:05d}.png'))
                Image.fromarray(instances).save(os.path.join(ttype_dir, 'inst_masks', f'inst_{tissue_type}_{fold_num}_{k:05d}.png'))
                Image.fromarray(raw_image).save(os.path.join(ttype_dir, 'images', f'img_{tissue_type}_{fold_num}_{k:05d}.png'))
