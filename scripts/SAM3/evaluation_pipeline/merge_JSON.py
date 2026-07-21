import json
import os
import argparse
import unicodedata

# ==============================
# CONFIG
# ==============================

FINAL_CATEGORIES = [
    "Autre animal",
    "Baigneur",
    "Bateau",
    "Chien",
    "Feu de camp",
    "Paddle",
    "Tente",
    "Voir la référence",
    "Véhicule motorisé"
]

CATEGORY_MAPPING = {
    
    "baigneur": "Baigneur",
    "bateau": "Bateau",
    "chien": "Chien",
    "tente": "Tente",

    
    "animal indetermine": "Autre animal",
    "animal indéterminé": "Autre animal",

    
    "randonneur": None,  

    
    "feu de camp": "Feu de camp",
    "paddle": "Paddle",
    "vehicule motorise": "Véhicule motorisé",
    "véhicule motorisé": "Véhicule motorisé",
    "voir la reference": "Voir la référence",
    "voir la référence": "Voir la référence",
}



def normalize(text):
    text = text.strip().lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_global_categories():
    return {name: i for i, name in enumerate(FINAL_CATEGORIES)}

def remap_annotations(data, global_categories, stats):
    local_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    new_annotations = []

    for ann in data.get("annotations", []):
        old_cat_id = ann["category_id"]
        raw_name = local_id_to_name.get(old_cat_id, "")

        norm_name = normalize(raw_name)
        mapped = CATEGORY_MAPPING.get(norm_name)

        if mapped is None:
            stats["removed"] += 1
            continue

        ann["category_id"] = global_categories[mapped]
        new_annotations.append(ann)
        stats["kept"] += 1

    data["annotations"] = new_annotations

def update_ids(data, image_offset, annotation_offset):
    for img in data.get("images", []):
        img["id"] += image_offset

    for ann in data.get("annotations", []):
        ann["id"] += annotation_offset
        ann["image_id"] += image_offset


def merge_coco(files):
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    global_categories = build_global_categories()

    image_offset = 0
    annotation_offset = 0

    stats = {"kept": 0, "removed": 0}

    for file in files:
        print(f"Processing: {file}")
        data = load_json(file)

        remap_annotations(data, global_categories, stats)
        update_ids(data, image_offset, annotation_offset)

        merged["images"].extend(data.get("images", []))
        merged["annotations"].extend(data.get("annotations", []))

        image_offset += len(data.get("images", []))
        annotation_offset += len(data.get("annotations", []))

    merged["categories"] = [
        {"id": i, "name": name}
        for name, i in global_categories.items()
    ]

    print("\n=== STATS ===")
    print(f"Annotations gardées : {stats['kept']}")
    print(f"Annotations supprimées : {stats['removed']}")

    return merged



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dossier contenant les JSON")
    parser.add_argument("--output", required=True, help="Fichier JSON de sortie")

    args = parser.parse_args()

    files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".json")
    ]

    merged = merge_coco(files)
    save_json(merged, args.output)

    print(f"\n Merge terminé : {args.output}")

if __name__ == "__main__":
    main()