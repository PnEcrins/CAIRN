import cv2
import numpy as np
import math
from ultralytics.models.sam import SAM3SemanticPredictor

# Paramètres
IMAGE_PATH = 'Images test/Brévent 2.JPG'
OUTPUT_PATH = 'resultat_final.jpg'
TEXT_PROMPT = [ "tent"]  
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.5

TILE_SIZE = 644 
TILE_OVERLAP = 50 #chevauchement, 50 pixels correspond en gros à la taille d'une tente donc on est large


# Initialisation de SAM
overrides = dict(
    conf=CONF_THRESHOLD,
    task="segment",
    mode="predict",
    model="sam3.pt",
    device="cpu",
    half=False,
    save=False,  
    verbose=False
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Fonction de slicing
def get_slices(img_h, img_w, size, overlap):
    cols = math.ceil((img_w - overlap) / (size - overlap))
    rows = math.ceil((img_h - overlap) / (size - overlap))
    
    slices = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * (size - overlap)
            y1 = r * (size - overlap)
            x2 = min(x1 + size, img_w)
            y2 = min(y1 + size, img_h)
            
            if x2 == img_w: x1 = max(0, img_w - size)
            if y2 == img_h: y1 = max(0, img_h - size)
            
            slices.append((x1, y1, x2, y2))
    return slices

# Exécution sur l'image
img = cv2.imread(IMAGE_PATH)

h, w, _ = img.shape
slices = get_slices(h, w, TILE_SIZE, TILE_OVERLAP)

all_global_boxes = []
all_global_scores = []

print(f"Lancement de l'inférence sur {len(slices)} tuiles")

for i, (x1, y1, x2, y2) in enumerate(slices):
    
    tile = img[y1:y2, x1:x2]
    
    
    predictor.set_image(tile)
    
    
    results = predictor(text=TEXT_PROMPT)
    
    
    if results and len(results) > 0 and results[0].boxes is not None:
        # Conversion des tenseurs en tableaux NumPy
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        for box, score in zip(bboxes, scores):
            # Recalcul des coordonnées dans le repère de l'image globale
            global_box = [
                box[0] + x1,  # xmin
                box[1] + y1,  # ymin
                box[2] + x1,  # xmax
                box[3] + y1   # ymax
            ]
            all_global_boxes.append(global_box)
            all_global_scores.append(score)
            
    print(f"Tuile {i+1}/{len(slices)} traitée.")

#Application du NMS de OpenCV-
if not all_global_boxes:
    print("Aucune détection trouvée sur l'ensemble de l'image.")
else:
           
    boxes_for_nms = []
    for b in all_global_boxes:
        width = b[2] - b[0]
        height = b[3] - b[1]
        boxes_for_nms.append([int(b[0]), int(b[1]), int(width), int(height)])
        
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_for_nms,
        scores=all_global_scores,
        score_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )
    
   #Regroupement des réusltats et dessin des boîtes finales
    if len(indices) > 0:
        indices = indices.flatten()
        
    for idx in indices:
        box = all_global_boxes[idx]
        score = all_global_scores[idx]
        
        
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        
        cv2.rectangle(img, p1, p2, (0, 165, 255), 2)
        texte = f"{score:.2f}"
        cv2.putText(img, texte, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Terminé. {len(indices)} objets conservés après filtrage.")
    print(f"Image sauvegardée sous : {OUTPUT_PATH}")