#!/usr/bin/env python3
"""
Extrait le timestamp d'images webcam JPG via Tesseract OCR
et l'embède dans les métadonnées EXIF.

Usage :
  python3 méta_données_Muzelle.py <dossier>          # traite tous les JPG du dossier
  python3 méta_données_Muzelle.py <dossier> --recur  # parcours récursif
  python3 méta_données_Muzelle.py image.jpg          # fichier unique
"""

import os
import sys
import re
import subprocess
from PIL import Image

# ── Tags EXIF ─────────────────────────────────────────────────────────────────
TAG_DATETIME           = 0x0132
TAG_DATETIME_ORIGINAL  = 0x9003
TAG_DATETIME_DIGITIZED = 0x9004
TAG_IMAGE_DESCRIPTION  = 0x010E
TAG_SOFTWARE           = 0x0131


def find_jpgs(path, recursive=False):
    jpg_files = []
    extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")
    if os.path.isfile(path):
        if path.endswith(extensions):
            jpg_files.append(path)
    elif os.path.isdir(path):
        if recursive:
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for fname in sorted(files):
                    if fname.endswith(extensions):
                        jpg_files.append(os.path.join(root, fname))
        else:
            for fname in sorted(os.listdir(path)):
                if fname.endswith(extensions):
                    jpg_files.append(os.path.join(path, fname))
    return jpg_files


def parse_timestamp(text):
    clean = re.sub(r":\s+", ":", text)
    m = re.search(r"(\d{4})[-:](\d{2})[-:](\d{2})\s+(\d{2}):(\d{2}):(\d{2})", clean)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}"
    return None


def ocr_timestamp(image_path):
    img = Image.open(image_path)
    w, h = img.size
    crop = img.crop((0, 0, int(w * 0.45), int(h * 0.08)))
    crop.save("/tmp/_ts_crop.png")
    raw = ""
    exif_dt = None
    for psm in [7, 6, 13]:
        r = subprocess.run(["tesseract", "/tmp/_ts_crop.png", "stdout", "--psm", str(psm)],
                           capture_output=True, text=True)
        raw = r.stdout.strip()
        exif_dt = parse_timestamp(raw)
        if exif_dt:
            break
    return {"ocr_raw": raw, "exif_datetime": exif_dt}


def embed_exif(src, dst, metadata):
    img = Image.open(src)
    exif = img.getexif()
    if metadata.get("exif_datetime"):
        dt = metadata["exif_datetime"]
        exif[TAG_DATETIME]           = dt
        exif[TAG_DATETIME_ORIGINAL]  = dt
        exif[TAG_DATETIME_DIGITIZED] = dt
    exif[TAG_IMAGE_DESCRIPTION] = f"Timestamp OCR: {metadata.get('ocr_raw', '')}"
    exif[TAG_SOFTWARE]          = "tesseract-ocr-metadata-embedder"
    img.save(dst, "JPEG", exif=exif.tobytes(), quality=95, subsampling=0)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    recursive = "--recur" in args
    args = [a for a in args if a != "--recur"]
    target = args[0]
    output = args[1] if len(args) == 2 else None

    jpg_files = find_jpgs(target, recursive=recursive)
    if not jpg_files:
        print(f"Aucun fichier JPG trouvé dans : {target}")
        sys.exit(1)

    print(f"{'─'*55}")
    print(f"Dossier  : {os.path.abspath(target)}")
    print(f"Fichiers : {len(jpg_files)} JPG trouvé(s)")
    print(f"{'─'*55}")

    ok, echecs, errors = 0, 0, []
    for f in jpg_files:
        base, ext = os.path.splitext(f)
        dst = output if (len(jpg_files) == 1 and output) else base + "_meta" + ext
        try:
            meta = ocr_timestamp(f)
            print(f"  {os.path.basename(f)}", end=" … ")
            if meta.get("exif_datetime"):
                print(f"{meta['exif_datetime']}", end=" → ")
                ok += 1
            else:
                print(f"('{meta['ocr_raw'][:25]}')", end=" → ")
                echecs += 1
            embed_exif(f, dst, meta)
            print(os.path.basename(dst))
        except Exception as e:
            print(f"  ✗ Erreur sur {os.path.basename(f)} : {e}")
            errors.append(f)
