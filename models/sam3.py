"""
Wrapper SAM3 avec option tiling.
"""

import math
import numpy as np
import cv2
from pathlib import Path
from .base import BaseModel, Detection

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    raise ImportError("pip install ultralytics")

DEFAULT_MODEL  = "sam3.pt"
TILE_OVERLAP   = 50


class SAM3Model(BaseModel):

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = 0.4, device: str = "cpu"):
        self.conf = conf
        overrides = dict(
            conf=conf, task="segment", mode="predict",
            model=model_path, half=False, save=False, verbose=False,
            device=device
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def detect(self, image_path: str, targets: list[str],
               use_tiling: bool = False) -> list[Detection]:

        img_bgr = cv2.imread(image_path)
        detections = []

        for target in targets:
            if use_tiling:
                bboxes = self._predict_tiled(img_bgr, [target])
            else:
                bboxes = self._predict_full(image_path, [target])

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                detections.append(Detection(
                    image_name=Path(image_path).name, label=target,
                    bbox=[round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    confidence=self.conf,
                    year=None, month=None, day=None, hour=None,
                ))
        return detections
    
    def __del__(self):
        try:
            del self.predictor
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── mode pleine image ──────────────────────────────────────────────────────

    def _predict_full(self, image_path: str, prompts: list[str]) -> list[list]:
        self.predictor.set_image(image_path)
        results = self.predictor(text=prompts)
        boxes = []
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                bbox = self._mask_to_bbox(mask)
                if bbox:
                    boxes.append(bbox)
        return boxes

    # ── mode tiling ───────────────────────────────────────────────────────────

    def _predict_tiled(self, img_bgr: np.ndarray, prompts: list[str]) -> list[list]:
        import tempfile, os
        h, w   = img_bgr.shape[:2]
        tile   = max(int(min(h, w) * 0.5), TILE_OVERLAP + 10)
        slices = self._get_slices(h, w, tile, TILE_OVERLAP)

        all_boxes, all_scores = [], []

        for (x1, y1, x2, y2) in slices:
            tile_img = img_bgr[y1:y2, x1:x2]
            _, buf   = cv2.imencode(".jpg", tile_img)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(buf.tobytes())
                tmp_path = f.name

            self.predictor.set_image(tmp_path)
            results = self.predictor(text=prompts)
            os.unlink(tmp_path)

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    all_boxes.append([bx1+x1, by1+y1, bx2+x1, by2+y1])
                    all_scores.append(float(box.conf))

        if not all_boxes:
            return []

        nms_input = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in all_boxes]
        scores    = [float(s) for s in all_scores]   

        indices = cv2.dnn.NMSBoxes(
            bboxes          = nms_input,
            scores          = scores,
            score_threshold = 0.0,    
            nms_threshold   = 0.3,    # IoU max pour considérer deux boxes comme distinctes
        )
        return [all_boxes[i] for i in indices.flatten()] if len(indices) else []

    # ── utilitaires ───────────────────────────────────────────────────────────

    @staticmethod
    def _mask_to_bbox(mask) -> list | None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()),
                float(xs.max()), float(ys.max())]

    @staticmethod
    def _get_slices(h, w, size, overlap) -> list[tuple]:
        cols   = math.ceil((w - overlap) / (size - overlap))
        rows   = math.ceil((h - overlap) / (size - overlap))
        slices = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * (size - overlap)
                y1 = r * (size - overlap)
                x2 = min(x1 + size, w)
                y2 = min(y1 + size, h)
                if x2 == w: x1 = max(0, w - size)
                if y2 == h: y1 = max(0, h - size)
                slices.append((int(x1), int(y1), int(x2), int(y2)))
        return slices