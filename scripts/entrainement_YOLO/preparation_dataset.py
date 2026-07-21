from pathlib import Path
import random
import os
# ============ CONFIG ============
BASE = Path("/bettik/PROJECTS/pr-loupe/weksteenl-ext")
HDD = Path("/media/lweksteen/HDD 500 Go/Timelapses")  
LABELS_DIR = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/labels")
DATASET_DIR = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Lacs avec sous-dossiers année
LACS_ANNEES = ["Pormenaz", "Lauvitel"]
# Lacs plats
LACS_PLATS  = ["Lauzon", "Anterne", "Brevent", "Cornu", "Muzelle", "Jovet"]

# Split par lac (stratifié) — on garde 1 lac pour val, 1 pour test
LAC_VAL  = "Anterne"      # ← choisis selon tes préférences
LAC_TEST = "Lauvitel"    # ← idem, idéalement un lac "différent" visuellement
# ================================

def find_image(stem):
    """Retrouve le chemin réel de l'image depuis le nom du fichier label."""
    lac = stem.split("_")[0]  # ex: "Anterne"
    filename = stem + ".jpg"
    
    if lac in LACS_ANNEES:
        for year in ["2024", "2025"]:
            p = HDD / lac / year / filename
            if p.exists():
                return BASE / lac /year / filename  # on retourne le chemin "normalisé" pour la config
    else:
        p = HDD / lac / filename
        if p.exists():
            return BASE / lac / filename  # chemin "normalisé" pour la config
    return None

# Parcourt tous les labels et construit les listes
train_imgs, val_imgs, test_imgs = [], [], []
missing = []

for txt in LABELS_DIR.glob("*.txt"):
    img_path = find_image(txt.stem)
    if img_path is None:
        missing.append(txt.name)
        continue
    
    lac = txt.stem.split("_")[0]
    if lac == LAC_TEST:
        test_imgs.append(str(img_path))
    elif lac == LAC_VAL:
        val_imgs.append(str(img_path))
    else:
        train_imgs.append(str(img_path))

# Écriture des fichiers listes
for name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
    out = DATASET_DIR / f"{name}.txt"
    out.write_text("\n".join(imgs))
    print(f"{name}: {len(imgs)} images → {out}")

if missing:
    print(f"\n⚠️  {len(missing)} labels sans image trouvée : {missing[:5]}...")