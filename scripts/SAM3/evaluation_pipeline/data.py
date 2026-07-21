import os
import json
import torch
import cv2
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, json_path, image_dir=None):
        """
        json_path : chemin vers le fichier COCO
        image_dir : dossier racine des images
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        self.image_dir = image_dir

        
        self.cat_id_to_label = {
            cat["id"]: i for i, cat in enumerate(self.categories)
        }

        self.label_to_cat = {
            i: cat["name"] for i, cat in enumerate(self.categories)
        }

        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]

            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []

            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # 1. Charger image
     
        img_info = self.images[idx]

        img_id = img_info["id"]
        file_name = img_info["file_name"]

        
        if self.image_dir is not None:
            img_path = os.path.join(self.image_dir, file_name)
        else:
            img_path = file_name

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image non trouvée: {img_path}")

       
        # 2. Récupérer annotations
        
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # COCO → [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])

            label = self.cat_id_to_label[ann["category_id"]]
            labels.append(label)

    
        # 3. Convertir en tensors
      
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, boxes, labels, file_name

   
    def get_class_name(self, label):
        return self.label_to_cat[label]
