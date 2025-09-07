"""
Custom Vahadane Stain Normalization Implementation
=================================================

This module provides a Vahadane stain normalization implementation that doesn't
require the problematic 'spams' package. Instead, it uses scikit-learn's 
dictionary learning which is more reliable and easier to install.

Based on:
A. Vahadane et al., 'Structure-Preserving Color Normalization and Sparse Stain 
Separation for Histological Images', IEEE Transactions on Medical Imaging, 
vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
"""

import numpy as np
import cv2
from sklearn.decomposition import DictionaryLearning
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class VahadaneNormalizer:
    """
    Vahadane stain normalization using scikit-learn dictionary learning
    """
    
    def __init__(self, threshold=0.8, lambda1=0.1, n_components=2, max_iter=1000):
        """
        Initialize Vahadane normalizer
        
        Args:
            threshold: Threshold for tissue mask (default: 0.8)
            lambda1: Sparsity regularization parameter (default: 0.1)
            n_components: Number of dictionary components (default: 2 for H&E)
            max_iter: Maximum iterations for dictionary learning (default: 1000)
        """
        self.threshold = threshold
        self.lambda1 = lambda1
        self.n_components = n_components
        self.max_iter = max_iter
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.fitted = False
    
    def standardize_brightness(self, I):
        """
        Standardize brightness as in StainTools
        """
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)
    
    def notwhite_mask(self, I, thresh=0.8):
        """
        Get tissue mask by excluding white/bright pixels
        """
        I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0
        return L < thresh
    
    def RGB_to_OD(self, I):
        """
        Convert RGB to optical density
        """
        I = I.astype(np.float64)
        I = np.maximum(I, 1)  # Avoid log(0)
        return -np.log(I / 255.0)
    
    def OD_to_RGB(self, OD):
        """
        Convert optical density to RGB
        """
        return (255 * np.exp(-OD)).astype(np.uint8)
    
    def normalize_rows(self, A):
        """
        Normalize rows of matrix A
        """
        return A / np.linalg.norm(A, axis=1, keepdims=True)
    
    def get_stain_matrix(self, I):
        """
        Extract stain matrix using dictionary learning
        """
        # Standardize brightness
        I = self.standardize_brightness(I)
        
        # Get tissue mask
        mask = self.notwhite_mask(I, self.threshold).reshape((-1,))
        
        # Convert to optical density
        OD = self.RGB_to_OD(I).reshape((-1, 3))
        OD = OD[mask]
        
        if len(OD) == 0:
            # Fallback if no tissue detected
            print("Warning: No tissue detected, using default stain matrix")
            return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        
        # Use scikit-learn DictionaryLearning instead of spams
        dict_learner = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.lambda1,
            max_iter=self.max_iter,
            fit_algorithm='cd',  # Use coordinate descent instead of lars for positive constraints
            transform_algorithm='lasso_cd',
            positive_dict=True,
            positive_code=True,
            random_state=42
        )
        
        try:
            # Fit dictionary learning on the transposed OD data
            # OD is (N, 3), we need to fit on (3, N) to learn 2 components of size 3
            dict_learner.fit(OD)  # Fit on (N, 3) to learn dictionary of shape (2, 3)
            dictionary = dict_learner.components_  # This should be (2, 3)
            
            # Ensure hematoxylin is first (darker stain)
            if dictionary[0, 0] < dictionary[1, 0]:
                dictionary = dictionary[[1, 0], :]
            
            # Normalize rows
            dictionary = self.normalize_rows(dictionary)
            
            return dictionary
            
        except Exception as e:
            print(f"Dictionary learning failed: {e}")
            # Fallback to default H&E stain matrix
            return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    
    def get_concentrations(self, I, stain_matrix):
        """
        Get concentration matrix using least squares
        """
        # Convert to optical density
        OD = self.RGB_to_OD(I).reshape((-1, 3))
        
        # Solve for concentrations using least squares
        # stain_matrix is (2, 3), OD is (N, 3), we want concentrations (N, 2)
        # We solve: OD = concentrations @ stain_matrix
        # This is equivalent to: concentrations = OD @ stain_matrix.T @ inv(stain_matrix @ stain_matrix.T)
        
        try:
            # Use Moore-Penrose pseudoinverse for robust solution
            # stain_matrix has shape (2, 3), pinv gives (3, 2)
            stain_matrix_pinv = np.linalg.pinv(stain_matrix)  # (2, 3) -> pinv -> (3, 2)
            concentrations = OD @ stain_matrix_pinv  # (N, 3) @ (3, 2) = (N, 2)
        except Exception as e:
            # Ultimate fallback - use simple projection
            print(f"Warning: Using simplified concentration estimation due to: {e}")
            concentrations = np.zeros((OD.shape[0], stain_matrix.shape[0]))
            # Simple approximation: project OD onto stain vectors
            for i in range(stain_matrix.shape[0]):
                stain_vec = stain_matrix[i, :]
                # Compute dot product for each pixel
                concentrations[:, i] = np.sum(OD * stain_vec, axis=1) / (np.sum(stain_vec * stain_vec) + 1e-8)
        
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, 0)
        
        return concentrations
    
    def fit(self, target_image):
        """
        Fit normalizer to target image
        
        Args:
            target_image: RGB target image (numpy array)
        """
        # Ensure uint8 format
        if target_image.dtype != np.uint8:
            target_image = (np.clip(target_image, 0, 255)).astype(np.uint8)
        
        # Extract target stain matrix
        self.stain_matrix_target = self.get_stain_matrix(target_image)
        
        # Get target concentrations
        self.target_concentrations = self.get_concentrations(target_image, self.stain_matrix_target)
        
        # Get 99th percentile of concentrations for normalization
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        
        self.fitted = True
        print(f"✓ Vahadane normalizer fitted successfully")
        print(f"  Target stain matrix shape: {self.stain_matrix_target.shape}")
        print(f"  Hematoxylin vector: {self.stain_matrix_target[0]}")
        print(f"  Eosin vector: {self.stain_matrix_target[1]}")
    
    def transform(self, source_image):
        """
        Transform source image to match target staining
        
        Args:
            source_image: RGB source image (numpy array)
            
        Returns:
            Normalized RGB image
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transformation")
        
        # Ensure uint8 format
        if source_image.dtype != np.uint8:
            source_image = (np.clip(source_image, 0, 255)).astype(np.uint8)
        
        # Extract source stain matrix
        stain_matrix_source = self.get_stain_matrix(source_image)
        
        # Get source concentrations
        source_concentrations = self.get_concentrations(source_image, stain_matrix_source)
        
        # Normalize concentrations
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        
        # Avoid division by zero
        maxC_source = np.maximum(maxC_source, 1e-6)
        
        # Scale concentrations
        source_concentrations *= (self.maxC_target / maxC_source)
        
        # Reconstruct image using target stain matrix
        normalized_OD = np.dot(source_concentrations, self.stain_matrix_target)
        normalized_RGB = 255 * np.exp(-normalized_OD)
        
        # Reshape and clip
        normalized_RGB = normalized_RGB.reshape(source_image.shape)
        normalized_RGB = np.clip(normalized_RGB, 0, 255).astype(np.uint8)
        
        return normalized_RGB
    
    def get_stain_colors(self):
        """
        Get RGB colors of extracted stains for visualization
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted first")
        
        # Convert stain matrix to RGB colors
        stain_colors = self.OD_to_RGB(self.stain_matrix_target)
        return stain_colors


def create_normalizer(target_image_path=None, **kwargs):
    """
    Factory function to create and optionally fit normalizer
    
    Args:
        target_image_path: Path to target image (optional)
        **kwargs: Additional arguments for VahadaneNormalizer
        
    Returns:
        VahadaneNormalizer instance
    """
    normalizer = VahadaneNormalizer(**kwargs)
    
    if target_image_path is not None:
        from PIL import Image
        target_image = np.array(Image.open(target_image_path).convert('RGB'))
        normalizer.fit(target_image)
    
    return normalizer


if __name__ == "__main__":
    # Example usage
    print("Vahadane Normalizer (Custom Implementation)")
    print("==========================================")
    
    # Create dummy test images
    test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    target_image = np.random.randint(80, 220, (256, 256, 3), dtype=np.uint8)
    
    # Test normalizer
    normalizer = VahadaneNormalizer()
    normalizer.fit(target_image)
    normalized = normalizer.transform(test_image)
    
    print(f"✓ Test completed successfully")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {normalized.shape}")
    print(f"  Output dtype: {normalized.dtype}")
