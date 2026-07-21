import numpy as np


def area(box):
    """
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def compute_intersection(box1, box2):
    """
    retourne l'aire d'intersection entre deux boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    return w * h


def compute_iou(box1, box2):
    """
    Intersection over Union
    """
    inter = compute_intersection(box1, box2)

    area1 = area(box1)
    area2 = area(box2)

    union = area1 + area2 - inter

    if union == 0:
        return 0.0

    return inter / union



def is_match(box_pred, box_gt, iou_thresh=0.5):
    """
    Décide si une prédiction match une GT  
    Retourne:
    - match (bool)
    - iou
    - coverage_pred
    - coverage_gt
    """

    iou = compute_iou(box_pred, box_gt)
    inter = compute_intersection(box_pred, box_gt)

    pred_area = area(box_pred)
    gt_area = area(box_gt)

    coverage_pred = inter / pred_area if pred_area > 0 else 0.0
    coverage_gt = inter / gt_area if gt_area > 0 else 0.0

   
    match = (
        iou >= iou_thresh
        or coverage_pred >= 0.3
        or coverage_gt >= 0.5
    )

    return match, iou, coverage_pred, coverage_gt