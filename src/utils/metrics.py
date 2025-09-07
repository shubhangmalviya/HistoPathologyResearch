"""
Segmentation Metrics for Nuclei Analysis
=======================================

Comprehensive metrics calculation for nuclei instance segmentation evaluation.
Includes per-image and per-class metrics with statistical analysis support.
"""

import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient for binary or multi-class segmentation.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask  
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        float: IoU score
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
    
    Returns:
        float: Pixel accuracy
    """
    correct_pixels = np.sum(pred == target)
    total_pixels = pred.size
    return correct_pixels / total_pixels


def calculate_class_wise_metrics(pred: np.ndarray, target: np.ndarray, 
                                num_classes: int = 6) -> Dict[str, Dict[int, float]]:
    """
    Calculate metrics for each class separately.
    
    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth segmentation mask (H, W)
        num_classes: Number of classes (default 6 for PanNuke)
    
    Returns:
        Dict containing metrics for each class
    """
    metrics = {
        'dice': {},
        'iou': {},
        'precision': {},
        'recall': {},
        'f1': {}
    }
    
    for class_id in range(num_classes):
        # Create binary masks for current class
        pred_binary = (pred == class_id).astype(np.uint8)
        target_binary = (target == class_id).astype(np.uint8)
        
        # Skip if class not present in ground truth
        if np.sum(target_binary) == 0:
            metrics['dice'][class_id] = np.nan
            metrics['iou'][class_id] = np.nan
            metrics['precision'][class_id] = np.nan
            metrics['recall'][class_id] = np.nan
            metrics['f1'][class_id] = np.nan
            continue
        
        # Calculate metrics
        metrics['dice'][class_id] = calculate_dice_coefficient(pred_binary, target_binary)
        metrics['iou'][class_id] = calculate_iou(pred_binary, target_binary)
        
        # Use sklearn for precision, recall, f1
        try:
            metrics['precision'][class_id] = precision_score(
                target_binary.flatten(), pred_binary.flatten(), zero_division=0
            )
            metrics['recall'][class_id] = recall_score(
                target_binary.flatten(), pred_binary.flatten(), zero_division=0
            )
            metrics['f1'][class_id] = f1_score(
                target_binary.flatten(), pred_binary.flatten(), zero_division=0
            )
        except:
            metrics['precision'][class_id] = 0.0
            metrics['recall'][class_id] = 0.0
            metrics['f1'][class_id] = 0.0
    
    return metrics


def calculate_segmentation_metrics(pred: np.ndarray, target: np.ndarray, 
                                 num_classes: int = 6) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics for a single image.
    
    Args:
        pred: Predicted segmentation mask (H, W) or (H, W, C)
        target: Ground truth segmentation mask (H, W) or (H, W, C)
        num_classes: Number of classes
    
    Returns:
        Dict containing all calculated metrics
    """
    # Handle different input formats
    if len(pred.shape) == 3:
        pred = np.argmax(pred, axis=-1)
    if len(target.shape) == 3:
        target = np.argmax(target, axis=-1)
    
    # Ensure same shape
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
    
    # Overall metrics
    overall_dice = calculate_dice_coefficient(pred, target)
    overall_iou = calculate_iou(pred, target)
    pixel_accuracy = calculate_pixel_accuracy(pred, target)
    
    # Class-wise metrics
    class_metrics = calculate_class_wise_metrics(pred, target, num_classes)
    
    # Average class-wise metrics (excluding NaN values)
    avg_dice = np.nanmean(list(class_metrics['dice'].values()))
    avg_iou = np.nanmean(list(class_metrics['iou'].values()))
    avg_precision = np.nanmean(list(class_metrics['precision'].values()))
    avg_recall = np.nanmean(list(class_metrics['recall'].values()))
    avg_f1 = np.nanmean(list(class_metrics['f1'].values()))
    
    # Compile results
    results = {
        # Overall metrics
        'overall_dice': overall_dice,
        'overall_iou': overall_iou,
        'pixel_accuracy': pixel_accuracy,
        
        # Average class-wise metrics
        'avg_dice': avg_dice,
        'avg_iou': avg_iou,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        
        # Individual class metrics
        **{f'dice_class_{i}': class_metrics['dice'].get(i, np.nan) for i in range(num_classes)},
        **{f'iou_class_{i}': class_metrics['iou'].get(i, np.nan) for i in range(num_classes)},
        **{f'precision_class_{i}': class_metrics['precision'].get(i, np.nan) for i in range(num_classes)},
        **{f'recall_class_{i}': class_metrics['recall'].get(i, np.nan) for i in range(num_classes)},
        **{f'f1_class_{i}': class_metrics['f1'].get(i, np.nan) for i in range(num_classes)},
    }
    
    return results


def calculate_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                          num_classes: int = 6) -> Dict[str, float]:
    """
    Calculate metrics for a batch of predictions.
    
    Args:
        predictions: Batch of predicted segmentation masks (B, H, W) or (B, C, H, W)
        targets: Batch of ground truth segmentation masks (B, H, W) or (B, C, H, W)
        num_classes: Number of classes
    
    Returns:
        Dict containing averaged metrics across the batch
    """
    batch_size = predictions.shape[0]
    all_metrics = []
    
    for i in range(batch_size):
        pred = predictions[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        
        metrics = calculate_segmentation_metrics(pred, target, num_classes)
        all_metrics.append(metrics)
    
    # Average metrics across batch
    averaged_metrics = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        averaged_metrics[key] = np.mean(values) if values else np.nan
    
    return averaged_metrics


def evaluate_model_on_dataset(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                            device: str = 'cuda', num_classes: int = 6) -> List[Dict[str, float]]:
    """
    Evaluate model on entire dataset and return per-image metrics.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation dataset
        device: Device to run evaluation on
        num_classes: Number of classes
    
    Returns:
        List of per-image metrics
    """
    model.eval()
    all_image_metrics = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for each image in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred = predictions[i].cpu().numpy()
                target = targets[i].cpu().numpy()
                
                metrics = calculate_segmentation_metrics(pred, target, num_classes)
                metrics['batch_idx'] = batch_idx
                metrics['image_idx'] = i
                all_image_metrics.append(metrics)
    
    return all_image_metrics


def create_metrics_summary(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Create summary statistics from list of per-image metrics.
    
    Args:
        metrics_list: List of per-image metrics
    
    Returns:
        Dict containing mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}
    
    summary = {}
    metric_keys = [k for k in metrics_list[0].keys() if k not in ['batch_idx', 'image_idx']]
    
    for key in metric_keys:
        values = [m[key] for m in metrics_list if not np.isnan(m[key])]
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        else:
            summary[key] = {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
    
    return summary


if __name__ == "__main__":
    # Test the metrics functions
    print("Testing segmentation metrics...")
    
    # Create dummy data
    pred = np.random.randint(0, 6, (256, 256))
    target = np.random.randint(0, 6, (256, 256))
    
    # Test single image metrics
    metrics = calculate_segmentation_metrics(pred, target, num_classes=6)
    print(f"✅ Single image metrics calculated: {len(metrics)} metrics")
    
    # Test batch metrics
    pred_batch = torch.randint(0, 6, (4, 256, 256))
    target_batch = torch.randint(0, 6, (4, 256, 256))
    
    batch_metrics = calculate_batch_metrics(pred_batch, target_batch, num_classes=6)
    print(f"✅ Batch metrics calculated: {len(batch_metrics)} metrics")
    
    print("✅ Metrics module test successful!")
