#!/usr/bin/env python3
"""
Pipeline de traitement de timelapses (équivalent Python du script R original).

Étapes :
1. Renommer les fichiers .TLS en .avi
2. Extraire les images des .avi avec ffmpeg
3. OCR sur la bande d'incrustation horodatage + écriture dans les métadonnées EXIF

Usage :
    python TLS_to_JPG.py /chemin/vers/le/dossier
"""

import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime

from PIL import Image
import pytesseract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXIFTOOL_PATH = "exiftool"      # adapter si exiftool n'est pas dans le PATH
FFMPEG_PATH = "ffmpeg"          # adapter si ffmpeg n'est pas dans le PATH
FRAMERATE = 5                   # nombre d'images extraites par seconde (équivalent -r 5)

# Zone de recadrage de la bande d'horodatage (largeur, hauteur, x, y)
# Reprend "1920x80+0+1000" du script R : crop(width, height, x_offset, y_offset)
CROP_BOX = (0, 1000, 1920, 1080)  # (left, top, right, bottom) pour PIL

TIMESTAMP_REGEX = re.compile(
    r"(\d{2}/\d{2}/\d{4})\s*(\d{1,2}:\d{2}\s?[AP]M)"
)


# ---------------------------------------------------------------------------
# Étape 1 : renommer les .TLS en .avi
# ---------------------------------------------------------------------------

def rename_tls_to_avi(root: Path) -> None:
    tls_files = sorted(root.glob("*.TLS"))
    for tls in tls_files:
        avi = tls.with_suffix(".avi")
        print(f"Renommage : {tls.name} -> {avi.name}")
        tls.rename(avi)


# ---------------------------------------------------------------------------
# Étape 2 : extraction des images avec ffmpeg
# ---------------------------------------------------------------------------

def extract_frames(root: Path) -> None:
    avi_files = sorted(root.glob("*.avi"))
    for avi in avi_files:
        base_name = avi.stem
        print(base_name)

        out_dir = root / base_name
        out_dir.mkdir(exist_ok=True)

        # NB : ajuster FRAMERATE selon le nombre d'images par seconde réel
        # de vos vidéos (à vérifier en lisant un .TLS renommé en .avi dans
        # un lecteur vidéo).
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-i", str(avi),
            "-r", str(FRAMERATE),
            str(out_dir / "images-%04d.jpg"),
        ]
        subprocess.run(ffmpeg_cmd, check=False)


# ---------------------------------------------------------------------------
# Étape 3 : OCR de l'horodatage + écriture EXIF
# ---------------------------------------------------------------------------

def extract_timestamp(image_path: Path) -> str | None:
    """Recadre le bas de l'image, effectue l'OCR et retourne un timestamp
    au format 'YYYY:MM:DD HH:MM:SS' (format attendu par ExifTool), ou None."""

    with Image.open(image_path) as img:
        cropped = img.crop(CROP_BOX)  # (left, top, right, bottom)
        text = pytesseract.image_to_string(cropped)

    # Nettoyage : ne garder que chiffres, ':', '/', 'A', 'P', 'M', espaces
    cleaned = re.sub(r"[^0-9:/APM ]", "", text).strip()

    matches = TIMESTAMP_REGEX.findall(cleaned)

    if len(matches) >= 2:
        date_str, time_str = matches[1]
    elif len(matches) == 1:
        date_str, time_str = matches[0]
    else:
        return None

    try:
        date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        time_str_clean = time_str.replace(" ", "")
        time_obj = datetime.strptime(time_str_clean, "%I:%M%p")
    except ValueError:
        return None

    return f"{date_obj.strftime('%Y:%m:%d')} {time_obj.strftime('%H:%M:%S')}"


def embed_timestamps(root: Path) -> list[Path]:
    """Parcourt récursivement les .jpg, extrait l'horodatage via OCR, et
    pour les échecs, interpole l'horodatage à partir des images voisines
    qui ont réussi (les images d'un même dossier sont prises à intervalle
    régulier, donc l'interpolation linéaire est fiable). Écrit le résultat
    dans le champ EXIF DateTimeOriginal. Retourne les fichiers toujours en
    échec après interpolation (aucune image voisine valide trouvée)."""

    still_failed: list[Path] = []

    # On traite dossier par dossier, car l'interpolation n'a de sens
    # qu'entre images d'une même séquence.
    subdirs = sorted({p.parent for p in root.rglob("*.jpg")})

    for folder in subdirs:
        images = sorted(folder.glob("*.jpg"))

        # 1) OCR sur chaque image, on garde les timestamps en objets datetime
        results: dict[Path, datetime | None] = {}
        for image in images:
            ts = extract_timestamp(image)
            if ts is not None:
                results[image] = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S")
                print(f"OCR OK pour {image.name} : {ts}")
            else:
                results[image] = None
                print(f"OCR échoué pour {image.name}")

        valid_indices = [i for i, img in enumerate(images) if results[img] is not None]

        if not valid_indices:
            print(f"Aucune image valide dans {folder} : impossible d'interpoler.")
            still_failed.extend(images)
            continue

        # 2) Interpolation/extrapolation pour les échecs
        for i, image in enumerate(images):
            if results[image] is not None:
                continue

            before = max((j for j in valid_indices if j < i), default=None)
            after = min((j for j in valid_indices if j > i), default=None)

            if before is not None and after is not None:
                t_before = results[images[before]]
                t_after = results[images[after]]
                fraction = (i - before) / (after - before)
                delta = (t_after - t_before) * fraction
                results[image] = t_before + delta
                print(f"Interpolé pour {image.name} : {results[image]} "
                      f"(entre {images[before].name} et {images[after].name})")
            elif before is not None:
                # Extrapolation en fin de séquence : intervalle moyen entre
                # les deux dernières images valides connues.
                ref_indices = [j for j in valid_indices if j <= before]
                avg_step = None
                if len(ref_indices) >= 2:
                    j0, j1 = ref_indices[-2], ref_indices[-1]
                    avg_step = (results[images[j1]] - results[images[j0]]) / (j1 - j0)
                results[image] = (
                    results[images[before]] + avg_step * (i - before)
                    if avg_step is not None else results[images[before]]
                )
                print(f"Extrapolé (fin) pour {image.name} : {results[image]}")
            elif after is not None:
                ref_indices = [j for j in valid_indices if j >= after]
                avg_step = None
                if len(ref_indices) >= 2:
                    j0, j1 = ref_indices[0], ref_indices[1]
                    avg_step = (results[images[j1]] - results[images[j0]]) / (j1 - j0)
                results[image] = (
                    results[images[after]] - avg_step * (after - i)
                    if avg_step is not None else results[images[after]]
                )
                print(f"Extrapolé (début) pour {image.name} : {results[image]}")

        # 3) Écriture EXIF pour toutes les images (OCR direct ou interpolées)
        for image in images:
            final_ts = results[image]
            if final_ts is None:
                still_failed.append(image)
                continue

            exif_str = final_ts.strftime("%Y:%m:%d %H:%M:%S")
            exif_cmd = [
                EXIFTOOL_PATH,
                "-overwrite_original",
                f"-EXIF:DateTimeOriginal={exif_str}",
                str(image),
            ]
            subprocess.run(exif_cmd, check=False, capture_output=True)

    return still_failed


# ---------------------------------------------------------------------------
# Débogage OCR (équivalent du bloc commenté en bas du script R)
# ---------------------------------------------------------------------------

def debug_ocr(image_path: Path, save_crop_to: Path = Path("test_strip.png")) -> None:
    with Image.open(image_path) as img:
        cropped = img.crop(CROP_BOX)
        cropped.save(save_crop_to)  # pour inspection visuelle

        text = pytesseract.image_to_string(cropped)
        print("Texte brut OCR :", text)

        cleaned = re.sub(r"[^0-9:/APM ]", "", text).strip()
        print("Texte nettoyé :", cleaned)

        matches = TIMESTAMP_REGEX.findall(cleaned)
        print("Correspondances :", matches)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage : python TLS_to_JPG.py /chemin/vers/le/dossier")
        sys.exit(1)

    root = Path(sys.argv[1])

    rename_tls_to_avi(root)
    extract_frames(root)
    failed = embed_timestamps(root)

    if failed:
        print(f"\n{len(failed)} fichier(s) sans horodatage (OCR et interpolation "
              f"ont tous les deux échoué — probablement aucune image OCR-valide "
              f"dans leur dossier) :")
        for f in failed:
            print(f" - {f}")
        print("Vous pouvez inspecter ces images avec debug_ocr(), ou ajuster CROP_BOX.")


if __name__ == "__main__":
    main()