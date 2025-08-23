import argparse
from .pannuke_preprocessor import PanNukePreprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PanNuke dataset for ML pipeline.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to PanNuke data directory (with Fold 1, Fold 2, etc.)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save processed images and masks')
    args = parser.parse_args()

    preprocessor = PanNukePreprocessor(args.data_dir, args.output_dir)
    preprocessor.process()
