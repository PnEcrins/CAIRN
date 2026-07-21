from hungarian_matching import hungarian_match
from metrics import is_match

class Evaluator:
    def __init__(self, iou_thresh):
        self.iou_thresh = iou_thresh
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, preds, gts):
        tp_boxes, fp_boxes, fn_boxes = [], [], []

        if len(preds) == 0:
            self.FN += len(gts)
            return tp_boxes, fp_boxes, fn_boxes

        if len(gts) == 0:
            self.FP += len(preds)
            return tp_boxes, fp_boxes, fn_boxes

        row_ind, col_ind, _ = hungarian_match(preds, gts)

        matched_preds = set()
        matched_gts = set()

        for r, c in zip(row_ind, col_ind):
            p = preds[r]
            g = gts[c]
            match, iou, cov, cov_gt = is_match(p, g, self.iou_thresh)
            if match:
                self.TP += 1
                tp_boxes.append(p)
                matched_preds.add(r)
                matched_gts.add(c)

        for i, pred in enumerate(preds):
            if i not in matched_preds:
                self.FP += 1
                fp_boxes.append(pred)

        for j, gt in enumerate(gts):
            if j not in matched_gts:
                self.FN += 1
                fn_boxes.append(gt)

        return tp_boxes, fp_boxes, fn_boxes

    def summary(self):
        precision = self.TP / (self.TP + self.FP + 1e-6)
        recall    = self.TP / (self.TP + self.FN + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        print("\n===== RESULTS =====")
        print("TP:", self.TP)
        print("FP:", self.FP)
        print("FN:", self.FN)
        print("Precision:", round(precision, 4))
        print("Recall:",    round(recall, 4))
        print("F1-score:",  round(f1, 4))
