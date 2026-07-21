"""
make_split_tiled.py
Split stratifié par lac sur les tuiles, avec contrôle des backgrounds.

Usage:
  python make_split_tiled.py --name tentes_tiled_01 --val Lauzon,Brevent,Cornu,Anterne
  python make_split_tiled.py --name tentes_tiled_01 --val Jovet --test Muzelle --bg_ratio 0.3
"""

import argparse
import random
from pathlib import Path

# ============ CONFIG ============
TILES_LABELS    = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/dataset/tiled/labels")
TILES_IMAGES    = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/dataset/tiled/images")
DATASETS_LOCAL  = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/dataset")
DATASETS_REMOTE = Path("/home/weksteenl-ext/MyPython/datasets")
TILES_REMOTE    = DATASETS_REMOTE / "tiled" / "images"

LACS_ANNEES   = ["Pormenaz", "Lauvitel"]
LACS_PLATS    = ["Lauzon", "Anterne", "Brevent", "Cornu", "Muzelle", "Jovet"]
TOUS_LES_LACS = LACS_ANNEES + LACS_PLATS
# ================================


def make_split(name: str, lacs_val: list, lacs_test: list, bg_ratio: float):

    # Validation
    for lac in lacs_val + lacs_test:
        if lac not in TOUS_LES_LACS:
            raise ValueError(f"Lac '{lac}' inconnu. Disponibles : {TOUS_LES_LACS}")
    overlap = set(lacs_val) & set(lacs_test)
    if overlap:
        raise ValueError(f"Ces lacs sont à la fois en val et test : {overlap}")

    # Création des dossiers locaux
    split_dir = DATASETS_LOCAL / "splits" / name
    yaml_dir  = DATASETS_LOCAL / "yamls"
    split_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir.mkdir(parents=True, exist_ok=True)

    # Collecte toutes les tuiles par split, séparées positifs/négatifs
    train_pos, train_neg = [], []
    val_pos,   val_neg   = [], []
    test_pos,  test_neg  = [], []
    missing = []

    for txt in sorted(TILES_LABELS.glob("*.txt")):
        lac = txt.stem.split("_")[0]

        img_local  = TILES_IMAGES / (txt.stem + ".jpg")
        img_remote = TILES_REMOTE / (txt.stem + ".jpg")

        if not img_local.exists():
            missing.append(txt.name)
            continue

        is_positive = txt.stat().st_size > 0

        if lac in lacs_test:
            (test_pos if is_positive else test_neg).append(str(img_remote))
        elif lac in lacs_val:
            (val_pos if is_positive else val_neg).append(str(img_remote))
        else:
            (train_pos if is_positive else train_neg).append(str(img_remote))

    # Sous-échantillonnage backgrounds
    random.seed(42)
    def sample_neg(neg_list):
        n_keep = int(len(neg_list) * bg_ratio)
        return random.sample(neg_list, min(n_keep, len(neg_list)))

    train_imgs = train_pos + sample_neg(train_neg)
    val_imgs   = val_pos   + sample_neg(val_neg)
    test_imgs  = test_pos  + sample_neg(test_neg)

    # Résumé par split
    lacs_train = [l for l in TOUS_LES_LACS if l not in lacs_val + lacs_test]
    for split_name, pos, neg_orig, imgs in [
        ("train", train_pos, train_neg, train_imgs),
        ("val",   val_pos,   val_neg,   val_imgs),
        ("test",  test_pos,  test_neg,  test_imgs),
    ]:
        n_neg_kept = len(imgs) - len(pos)
        print(f"  {split_name:5s}: {len(pos):4d} positifs + {n_neg_kept:4d}/{len(neg_orig)} négatifs = {len(imgs):5d} tuiles")

    # Écriture des listes
    for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        f = split_dir / f"{split}.txt"
        random.shuffle(imgs)
        f.write_text("\n".join(imgs))

    # YAML
    test_line = f"test:  splits/{name}/test.txt" if lacs_test else ""
    yaml_content = f"""# Split : {name}
# Mode     : tiled stratifié par lac
# train    : {', '.join(lacs_train)}
# val      : {', '.join(lacs_val)}
# test     : {', '.join(lacs_test) if lacs_test else 'aucun'}
# bg_ratio : {bg_ratio*100:.0f}%
# Seed     : 42

path: {DATASETS_REMOTE}
train: splits/{name}/train.txt
val:   splits/{name}/val.txt
{test_line}

nc: 1
names:
  - tente
"""
    yaml_file = yaml_dir / f"{name}.yaml"
    yaml_file.write_text(yaml_content)

    print(f"\n  YAML → {yaml_file}")

    if missing:
        print(f"\n  ⚠️  {len(missing)} tuiles sans image : {missing[:5]}...")

    print(f"""
  Transfert sur bigfoot :
    scp -r {split_dir} weksteenl-ext@bigfoot.ciment:/home/weksteenl-ext/MyPython/datasets/splits/
    scp {yaml_file} weksteenl-ext@bigfoot.ciment:/home/weksteenl-ext/MyPython/datasets/yamls/
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",     required=True)
    parser.add_argument("--val",      required=True, help="Lacs val séparés par virgule")
    parser.add_argument("--test",     default="",    help="Lacs test séparés par virgule (optionnel)")
    parser.add_argument("--bg_ratio", type=float, default=1.0,
                        help="Proportion de backgrounds à garder. 0.3=30%%, 1.0=tous (défaut: 1.0)")
    args = parser.parse_args()

    lacs_val  = [l.strip() for l in args.val.split(",")]
    lacs_test = [l.strip() for l in args.test.split(",")] if args.test else []

    print(f"\n→ Génération du split tiled '{args.name}'\n")
    make_split(args.name, lacs_val, lacs_test, args.bg_ratio)
    print("✓ Done\n")