"""
Lit les EXIF d'une image et renvoie le nom final du fichier.
Si pas d'EXIF (ou EXIF illisibles) → nom générique.
"""

import re
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    raise ImportError("Installe Pillow : pip install Pillow")


GENERIC_PREFIX = "image_sans_date"


def _parse_exif_datetime(raw: str) -> datetime | None:
    """Convertit '2024:07:15 10:32:00' en objet datetime."""
    try:
        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def get_exif_datetime(image_path: str) -> datetime | None:
    """
    Tente d'extraire la date/heure depuis les EXIF de l'image.
    Retourne un datetime ou None si impossible.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None

        # Tags utiles dans l'ordre de préférence
        target_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
        tag_map = {TAGS.get(k, k): v for k, v in exif_data.items()}

        for tag in target_tags:
            if tag in tag_map:
                dt = _parse_exif_datetime(tag_map[tag])
                if dt:
                    return dt
    except Exception:
        pass

    return None


def build_output_name(image_path: str, dt: datetime | None, index: int) -> str:
    """
    Retourne le nom de fichier final (sans dossier).
    - Si datetime connue : AAAA-MM-JJ-HH-MM-SS.ext
    - Sinon             : image_sans_date_001.ext
    """
    ext = Path(image_path).suffix.lower()

    if dt:
        return f"{dt.strftime('%Y-%m-%d-%H-%M-%S')}{ext}"
    else:
        return f"{GENERIC_PREFIX}_{index:03d}{ext}"


def preprocess_images(image_paths: list[str]) -> list[dict]:
    """
    Pour chaque image, renvoie un dict :
      {
        "original_path": str,
        "output_name": str,   # nom final (pour le CSV)
        "datetime": datetime | None
      }
    """
    results = []
    generic_counter = 1

    for path in image_paths:
        dt = get_exif_datetime(path)
        name = build_output_name(path, dt, generic_counter)

        if dt is None:
            generic_counter += 1

        results.append(
            {
                "original_path": path,
                "output_name": name,
                "datetime": dt,
            }
        )

    return results
