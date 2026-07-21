import numpy as np
from scipy.optimize import linear_sum_assignment
from metrics import compute_iou, compute_intersection, area


def hungarian_match(preds, gts):
    """
    Associe preds ↔ gts via Hungarian algorithm

    Retourne :
    - row_ind (indices preds)
    - col_ind (indices gts)
    - cost_matrix
    """

    if len(preds) == 0 or len(gts) == 0:
        return [], [], np.array([])

    cost_matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)

    for i, p in enumerate(preds):
        for j, g in enumerate(gts):

            iou = compute_iou(p, g)
            inter = compute_intersection(p, g)

            pred_area = area(p)
            coverage = inter / pred_area if pred_area > 0 else 0.0

            #  score hybride 
            score = 0.3 * iou + 0.7 * coverage

            cost_matrix[i, j] = 1.0 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind, cost_matrix