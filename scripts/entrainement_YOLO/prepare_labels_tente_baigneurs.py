"""
prepare_labels.py
Filtre tente + baigneur, remappe en 0=tente, 1=baigneur.
Gère 3 sources : dataset principal, B (tente=5), A (tente=6).

Usage: python prepare_labels.py
"""

from pathlib import Path

# ============ CONFIG ============
# Sources
LABELS_PRINCIPAL = Path("/home/lweksteen/Téléchargements/Export2000/labels")
LABELS_B         = Path("/home/lweksteen/Téléchargements/B/export YOLO first 400/labels")
LABELS_A = [
    Path("/home/lweksteen/Téléchargements/A/export YOLO banque images WEB/labels"),
    Path("/home/lweksteen/Téléchargements/A/export YOLO images Océanes/labels"),
]

# Sortie
OUT = Path("/home/lweksteen/CrowdShore/Export_2000_formatYOLO/labels_tentes_baigneurs")
OUT.mkdir(exist_ok=True)

# Classes source → classe cible
# Format : {id_source: id_cible}  — les autres sont ignorés
MAPPING_PRINCIPAL = {6: 0, 1: 1}  # tente=6→0, baigneur=1→1
MAPPING_B         = {5: 0, 1: 1}  # tente=5→0, baigneur=1→1
MAPPING_A         = {6: 0, 1: 1}  # tente=6→0, baigneur=1→1
# ================================


def fix_and_copy(labels_dir: Path, mapping: dict, source_name: str):
    copied, vides, doublons = 0, 0, 0

    for txt in sorted(labels_dir.glob("*.txt")):
        dest = OUT / txt.name
        if dest.exists():
            print(f"  ⚠️  DOUBLON ignoré : {txt.name}")
            doublons += 1
            continue

        lines = txt.read_text().splitlines()
        filtered = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            src_id = int(parts[0])
            if src_id in mapping:
                filtered.append(f"{mapping[src_id]} " + " ".join(parts[1:]))

        dest.write_text("\n".join(filtered))
        copied += 1
        if not filtered:
            vides += 1

    print(f"  {source_name:20s} : {copied} copiés, {vides} vides, {doublons} doublons")


print("=== Dataset principal (tente=6, baigneur=1) ===")
fix_and_copy(LABELS_PRINCIPAL, MAPPING_PRINCIPAL, "principal")

print("\n=== Dataset B (tente=5, baigneur=1) ===")
fix_and_copy(LABELS_B, MAPPING_B, "B")

print("\n=== Dataset A (tente=6, baigneur=1) ===")
for d in LABELS_A:
    fix_and_copy(d, MAPPING_A, d.name)

# Vérifie les classes présentes dans le résultat
print("\n=== Vérification classes dans OUT ===")
classes = {}
for txt in OUT.glob("*.txt"):
    for line in txt.read_text().splitlines():
        if line.strip():
            cls = int(line.split()[0])
            classes[cls] = classes.get(cls, 0) + 1

for cls, count in sorted(classes.items()):
    name = {0: "tente", 1: "baigneur"}.get(cls, f"inconnue({cls})")
    print(f"  classe {cls} ({name}) : {count} instances")

print(f"\nTotal fichiers : {len(list(OUT.glob('*.txt')))}")
print("✓ Done")