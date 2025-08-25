import numpy as np
from typing import Tuple, Dict
from skimage.measure import label


def reconstruct_instances(sem_mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """
    Reconstruct instance labels from a semantic mask and a boundary mask.
    - sem_mask: HxW uint8, 0 is background, >0 are nuclei classes
    - boundary: HxW uint8, 0 background, >0 boundary pixels
    Returns: HxW int32 label image where 0 is background and 1..N are instances.
    """
    foreground = (sem_mask > 0).astype(np.uint8)
    # Remove boundary pixels to separate connected components
    separated = np.where(boundary > 0, 0, foreground)
    inst_labels = label(separated, connectivity=1)
    return inst_labels.astype(np.int32)


def dice_coefficient(pred_sem: np.ndarray, gt_sem: np.ndarray, num_classes: int = 7, ignore_background: bool = True) -> float:
    """
    Macro-averaged Dice over classes.
    pred_sem, gt_sem: HxW uint8 labels.
    """
    classes = range(1 if ignore_background else 0, num_classes)
    dice_scores = []
    for c in classes:
        pred_c = (pred_sem == c)
        gt_c = (gt_sem == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        if denom == 0:
            continue
        dice = 2 * inter / denom
        dice_scores.append(dice)
    if len(dice_scores) == 0:
        return 1.0
    return float(np.mean(dice_scores))


def _pair_instances_by_iou(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> Tuple[Dict[int, int], int, int, float]:
    """
    Match gt and pred instance ids by IoU.
    Returns: matches dict {gt_id: pred_id}, FP, FN, sum_iou_matched
    """
    gt_ids = [i for i in np.unique(gt) if i != 0]
    pred_ids = [i for i in np.unique(pred) if i != 0]

    matches = {}
    sum_iou = 0.0
    used_pred = set()
    for gid in gt_ids:
        gmask = (gt == gid)
        best_iou = 0.0
        best_pid = None
        for pid in pred_ids:
            if pid in used_pred:
                continue
            pmask = (pred == pid)
            inter = np.logical_and(gmask, pmask).sum()
            union = np.logical_or(gmask, pmask).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_pid = pid
        if best_pid is not None and best_iou >= iou_threshold:
            matches[gid] = best_pid
            sum_iou += best_iou
            used_pred.add(best_pid)

    tp = len(matches)
    fp = len(pred_ids) - len(used_pred)
    fn = len(gt_ids) - tp
    return matches, fp, fn, float(sum_iou)


def f1_object(gt_inst: np.ndarray, pred_inst: np.ndarray, iou_threshold: float = 0.5) -> float:
    matches, fp, fn, _ = _pair_instances_by_iou(gt_inst, pred_inst, iou_threshold)
    tp = len(matches)
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def pq_panoptic(gt_inst: np.ndarray, pred_inst: np.ndarray, iou_threshold: float = 0.5) -> float:
    matches, fp, fn, sum_iou = _pair_instances_by_iou(gt_inst, pred_inst, iou_threshold)
    tp = len(matches)
    denom = (tp + 0.5 * fp + 0.5 * fn)
    if denom == 0:
        return 1.0
    return float(sum_iou / denom)


def aji_aggregated_jaccard(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    """
    Aggregated Jaccard Index (AJI) per Kumar et al.
    Greedy matching by IoU; counts unmatched preds in denominator.
    """
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]

    matched_pred = set()
    inter_sum = 0
    union_sum = 0

    for gid in gt_ids:
        gmask = gt_inst == gid
        best_iou = 0.0
        best_pid = None
        best_inter = 0
        best_union = 0
        for pid in pred_ids:
            if pid in matched_pred:
                continue
            pmask = pred_inst == pid
            inter = np.logical_and(gmask, pmask).sum()
            union = np.logical_or(gmask, pmask).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou, best_pid = iou, pid
                best_inter, best_union = inter, union
        if best_pid is not None and best_iou > 0:
            matched_pred.add(best_pid)
            inter_sum += best_inter
            union_sum += best_union
        else:
            # unmatched GT contributes its area to union
            union_sum += gmask.sum()

    # add all unmatched predicted instance areas to denominator
    for pid in pred_ids:
        if pid not in matched_pred:
            union_sum += (pred_inst == pid).sum()

    if union_sum == 0:
        return 1.0
    return float(inter_sum / union_sum)


