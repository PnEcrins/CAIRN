"""
generate_tiles.py
Génère les tuiles depuis les images originales + labels adaptés.
Calcule dynamiquement la taille des tuiles en fonction de la largeur de chaque image.

Usage:
  python generate_tiles.py
  python generate_tiles.py --tile_ratio 0.25 --overlap 0.2
  python generate_tiles.py --exclude Muzelle,Jovet,Anterne
"""

import argparse
from pathlib import Path
from PIL import Image

# ============ CONFIG LOCAL ============
BETTIK     = Path("/bettik/PROJECTS/pr-loupe/weksteenl-ext")
LABELS_DIR = Path("/home/weksteenl-ext/MyPython/datasets/labels")
TILES_DIR  = Path("/home/weksteenl-ext/MyPython/datasets/tiled")

LACS_ANNEES = ["Pormenaz", "Lauvitel"]
LACS_PLATS  = ["Lauzon", "Anterne", "Brevent", "Cornu", "Muzelle", "Jovet"]
TOUS_LES_LACS = LACS_ANNEES + LACS_PLATS
# ======================================

def find_image_local(stem: str) -> Path | None:
    """Trouve l'image sur le HDD local."""
    lac = stem.split("_")[0]
    filename = stem + ".jpg"
    if lac in LACS_ANNEES:
        for year in ["2024", "2025"]:
            p = BETTIK / lac / year / filename
            if p.exists():
                return p
    elif lac in LACS_PLATS:
        p = BETTIK / lac / filename
        if p.exists():
            return p
    return None

def load_labels(label_path: Path) -> list:
    """Charge les labels YOLO normalisés."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    lines = label_path.read_text().splitlines()
    boxes = []
    for line in lines:
        if line.strip():
            parts = line.split()
            boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes

def clip_box_to_tile(box, tx, ty, tile_size, img_w, img_h, min_coverage=0.3):
    """
    Projette une bbox YOLO (normalisée image entière)
    en bbox YOLO normalisée sur la tuile.
    Retourne None si couverture insuffisante.
    """
    cls, cx, cy, bw, bh = box
    # Coordonnées absolues de la bbox
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h

    # Coordonnées absolues de la tuile
    tx2, ty2 = tx + tile_size, ty + tile_size

    # Intersection
    ix1, iy1 = max(x1, tx), max(y1, ty)
    ix2, iy2 = min(x2, tx2), min(y2, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # Couverture minimale
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    box_area   = (x2 - x1)   * (y2 - y1)
    if box_area == 0 or inter_area / box_area < min_coverage:
        return None

    # Coordonnées normalisées sur la tuile
    return [cls,
            ((ix1 + ix2) / 2 - tx) / tile_size,
            ((iy1 + iy2) / 2 - ty) / tile_size,
            (ix2 - ix1)            / tile_size,
            (iy2 - iy1)            / tile_size]

def tile_image(img_path: Path, label_path: Path,               
               tile_ratio: float, overlap: float,               
               out_images: Path, out_labels: Path) -> int:    
    """Découpe une image en tuiles dynamiques basées sur un ratio de la largeur."""    
    img = Image.open(img_path)    
    img_w, img_h = img.size    
    boxes = load_labels(label_path)    
    
    # Calcul de la taille de la tuile en pixels (arrondi au multiple de 32 pour YOLO)
    tile_size = int(round((img_w * tile_ratio) / 32) * 32)
    tile_size = min(tile_size, img_w, img_h)
    
    stride = int(tile_size * (1 - overlap))    
    
    # Coordonnées X et Y avec garantie d'inclure le bord final
    x_starts = list(range(0, img_w - tile_size + 1, stride))
    if not x_starts or x_starts[-1] != img_w - tile_size:
        x_starts.append(img_w - tile_size)
        
    y_starts = list(range(0, img_h - tile_size + 1, stride))
    if not y_starts or y_starts[-1] != img_h - tile_size:
        y_starts.append(img_h - tile_size)
        
    count = 0    
    for row, ty in enumerate(y_starts):        
        for col, tx in enumerate(x_starts):            
            tile_name = f"{img_path.stem}_tile_{row}_{col}"            
            
            # Projection des labels sur la tuile
            tile_boxes = [                
                clip_box_to_tile(b, tx, ty, tile_size, img_w, img_h)                
                for b in boxes            
            ]            
            tile_boxes = [b for b in tile_boxes if b is not None]            
            
            # Sauvegarde de la tuile image
            img.crop((tx, ty, tx + tile_size, ty + tile_size)).save(out_images / f"{tile_name}.jpg", quality=95)            
            
            # Sauvegarde du fichier label associé
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"                     
                     for b in tile_boxes]            
            (out_labels / f"{tile_name}.txt").write_text("\n".join(lines))            
            count += 1    
    return count

def main(tile_ratio: float, overlap: float, lacs_exclus: list):
    out_images = TILES_DIR / "images"
    out_labels = TILES_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    lacs_inclus = [l for l in TOUS_LES_LACS if l not in lacs_exclus]
    print(f"Lacs inclus : {', '.join(lacs_inclus)}")
    print(f"Tile ratio  : {tile_ratio*100:.0f}% de la largeur | Overlap : {overlap*100:.0f}%\n")

    total_images = 0
    total_tiles  = 0
    missing      = []

    for txt in sorted(LABELS_DIR.glob("*.txt")):
        lac = txt.stem.split("_")[0]
        if lac in lacs_exclus:
            continue

        img_path = find_image_local(txt.stem)
        if img_path is None:
            missing.append(txt.name)
            continue

        n = tile_image(img_path, txt, tile_ratio, overlap, out_images, out_labels)
        total_tiles  += n
        total_images += 1

        if total_images % 100 == 0:
            print(f"  {total_images} images traitées ({total_tiles} tuiles)...")

    print(f"\n✓ {total_images} images → {total_tiles} tuiles")
    print(f"  Dossier : {TILES_DIR}")
    
    size_go = total_tiles * 0.09 / 1024
    print(f"  Espace estimé : ~{size_go:.1f} Go")

    if missing:
        print(f"\n⚠️  {len(missing)} images non trouvées sur HDD : {missing[:5]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_ratio", type=float, default=0.3,
                        help="Taille de la tuile en ratio de la largeur (ex: 0.25 pour 25%)")
    parser.add_argument("--overlap",    type=float, default=0.2)
    parser.add_argument("--exclude",    default="", help="Lacs à exclure ex: Muzelle,Jovet")
    args = parser.parse_args()

    lacs_exclus = [l.strip() for l in args.exclude.split(",")] if args.exclude else []
    main(args.tile_ratio, args.overlap, lacs_exclus)