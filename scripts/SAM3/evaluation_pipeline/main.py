import argparse
import json
import numpy as np
import pandas as pd
from data import CocoDataset
from SAM3_tilling import SAM3Model
from evaluator import Evaluator
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True, help= "chemin vers le fichier JSON")
parser.add_argument("--images", required=True, help= "chemin vers le dossier contenant les images")
parser.add_argument("--output", required=True)
parser.add_argument("--conf", type=float, default=0.6)
parser.add_argument("--iou", type=float, default=0.5)
parser.add_argument("--prompt", action="append", default=["swimmer"])

parser.add_argument("--tiling", action="store_true")
parser.add_argument("--tile_ratio", type=float, default=0.2, help="Taille des tuiles en pourcentage de l'image (ex: 0.3 pour 30%)")
parser.add_argument("--tile_overlap", type=int, default=50)
parser.add_argument("--nms_thresh", type=float, default=0.5)

args = parser.parse_args()



def to_list(boxes):
        return [list(box)for box in boxes]


TENT_CLASS_ID = 1

dataset = CocoDataset(args.labels, args.images)
model = SAM3Model(conf=args.conf)
evaluator_global = Evaluator(iou_thresh=args.iou)

results = []

for i, (image, boxes, labels, file_name) in enumerate(dataset):

    gt_boxes = [
        box.tolist()
        for box, label in zip(boxes, labels)
        if label.item() == TENT_CLASS_ID
    ]

    preds = model.predict(
        image, 
        text_prompts=args.prompt,
        use_tiling=args.tiling,
        tile_ratio=args.tile_ratio,
        tile_overlap=args.tile_overlap,
        nms_thresh=args.nms_thresh
    )

    preds = [
        [int(v) for v in box]
        for box in preds
    ]

    evaluator = Evaluator(iou_thresh=args.iou)
    tp_boxes, fp_boxes, fn_boxes = evaluator.update(preds, gt_boxes)
    
    TP, FP, FN = evaluator.TP, evaluator.FP, evaluator.FN

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    evaluator_global.TP += TP
    evaluator_global.FP += FP
    evaluator_global.FN += FN

    parts= Path(file_name).parts


    results.append({
        "image_name": file_name,
        "image_id": i,
        "lake_name": parts[parts.index("weksteenl-ext") + 1],
        "nb_preds": len(preds),
        "nb_gts": len(gt_boxes),
        "TP": TP, 
        "FP": FP, 
        "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "conf": args.conf,
        "pred_boxes": to_list(preds),
        "gt_boxes": gt_boxes,
        "tp_boxes": to_list(tp_boxes),
        "fp_boxes": to_list(fp_boxes),
        "fn_boxes": to_list(fn_boxes),
    })

    if i % 100 == 0:
        print(f"[{i}] done | GT: {len(gt_boxes)} | Pred: {len(preds)}")

# Sauvegarde csv 
df = pd.DataFrame(results)
df.to_csv(args.output, index=False)


# Résumé global
print("\n===== GLOBAL RESULTS =====")

TP = evaluator_global.TP
FP = evaluator_global.FP
FN = evaluator_global.FN

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
