import numpy as np
import math
import cv2
from ultralytics.models.sam import SAM3SemanticPredictor


class SAM3Model:
    def __init__(self, conf):
        self.conf = conf
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model="sam3.pt",
            half=True,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def mask_to_bbox(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [xs.min(), ys.min(), xs.max(), ys.max()]

    def get_slices(self, img_h, img_w, size, overlap):
        """Découpe l'image en tuiles."""
        cols = math.ceil((img_w - overlap) / (size - overlap))
        rows = math.ceil((img_h - overlap) / (size - overlap))

        slices = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * (size - overlap)
                y1 = r * (size - overlap)
                x2 = min(x1 + size, img_w)
                y2 = min(y1 + size, img_h)

                if x2 == img_w:
                    x1 = max(0, img_w - size)
                if y2 == img_h:
                    y1 = max(0, img_h - size)

                slices.append((int(x1), int(y1), int(x2), int(y2)))
        return slices

    def predict(
        self,
        image,
        text_prompts=["tent"],
        use_tiling=False,
        tile_ratio=0.2,
        tile_overlap=50,
        nms_thresh=0.5,
    ):

        # --- MODE CLASSIQUE (Image entière) ---
        if not use_tiling:
            self.predictor.set_image(image)
            pred_boxes = []
            results = self.predictor(text=text_prompts)

            if results and len(results) > 0:
                r = results[0]
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    for mask in masks:
                        bbox = self.mask_to_bbox(mask)
                        if bbox:
                            pred_boxes.append(bbox)
            return pred_boxes

        # --- MODE TILING (Slicing + NMS) ---
        h, w = image.shape[:2]
        tile_size = int(min(h, w) * tile_ratio)
        tile_size_px = max(tile_size, tile_overlap + 10)
        slices = self.get_slices(h, w, tile_size, tile_overlap)

        all_global_boxes = []
        all_global_scores = []

        for x1, y1, x2, y2 in slices:
            tile = image[y1:y2, x1:x2]
            self.predictor.set_image(tile)
            results = self.predictor(text=text_prompts)

            if results and len(results) > 0 and results[0].boxes is not None:
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                for box, score in zip(bboxes, scores):
                    # Recalcul dans le repère global
                    global_box = [
                        box[0] + x1,  # xmin
                        box[1] + y1,  # ymin
                        box[2] + x1,  # xmax
                        box[3] + y1,  # ymax
                    ]
                    all_global_boxes.append(global_box)
                    all_global_scores.append(float(score))

        # Application du NMS
        if not all_global_boxes:
            return []

        # Format requis par OpenCV: [x, y, width, height]
        boxes_for_nms = [
            [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
            for b in all_global_boxes
        ]

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_nms,
            scores=all_global_scores,
            score_threshold=self.conf,
            nms_threshold=nms_thresh,
        )

        final_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                final_boxes.append(all_global_boxes[idx])

        return final_boxes
