"""
Wrapper YOLO avec tiling via SAHI.
"""

from pathlib import Path
from .base import BaseModel, Detection

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("pip install ultralytics")

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

DEFAULT_MODEL = "last.pt"

class YoloModel(BaseModel):

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = 0.4, device: str = "cpu"):
        self.conf       = conf
        self.model_path = model_path
        self.model      = YOLO(model_path)
        self.device     = device

        # Modèle SAHI 
        if SAHI_AVAILABLE:
            # On extrait les classes directement depuis le modèle YOLO chargé
            cat_mapping = {str(k): str(v) for k, v in self.model.names.items()}
            
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=conf,
                device=self.device,
                category_mapping=cat_mapping
            )
        else:
            self.sahi_model = None

    def detect(self, image_path: str, targets: list[str],
               use_tiling: bool = False) -> list[Detection]:

        if use_tiling and self.sahi_model:
            return self._detect_tiled(image_path, targets)
        else:
            return self._detect_full(image_path, targets)

    def _detect_full(self, image_path: str, targets: list[str]) -> list[Detection]:
        results = self.model(image_path, conf=self.conf, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                
                # Le modèle renvoyant les bons noms, on vérifie directement la cible
                if label not in targets:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    image_name=Path(image_path).name, label=label,
                    bbox=[round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    confidence=round(float(box.conf), 3),
                    year=None, month=None, day=None, hour=None,
                ))
        return detections

    def _detect_tiled(self, image_path: str, targets: list[str]) -> list[Detection]:
        from PIL import Image as PILImage
        w, h  = PILImage.open(image_path).size
        tile  = int(min(w, h) * 0.5)

        result = get_sliced_prediction(
            image_path,
            self.sahi_model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0,
        )
        detections = []
        for obj in result.object_prediction_list:
            label = obj.category.name
            
            # Pareil ici, obj.category.name contient directement le bon label grâce au category_mapping
            if label not in targets:
                continue
                
            b = obj.bbox
            detections.append(Detection(
                image_name=Path(image_path).name, label=label,
                # On force la conversion en float classique Python avant d'arrondir
                bbox=[
                    round(float(b.minx), 1), 
                    round(float(b.miny), 1), 
                    round(float(b.maxx), 1), 
                    round(float(b.maxy), 1)
                ],
                confidence=round(float(obj.score.value), 3),
                year=None, month=None, day=None, hour=None,
            ))
        return detections