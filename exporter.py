"""
Transforme une liste de Detection en fichier CSV via pandas.

Format : 1 ligne par image.
Colonnes fixes : image_name, bbox_baigneur, count_baigneur,
                 bbox_tente, count_tente, bbox_prompt, count_prompt,
                 year, month, day, hour
"""

import tempfile
from pathlib import Path
import pandas as pd
from models.base import Detection

FIXED_COLUMNS = [
    "image_name",
    "bbox_baigneur",
    "count_baigneur",
    "bbox_tente",
    "count_tente",
    "bbox_prompt",
    "count_prompt",
    "year",
    "month",
    "day",
    "hour",
]


def export_to_csv(
    preprocessed: list[dict], detections: list[Detection], free_prompt: str = ""
) -> str:
    """
    1 ligne par image, colonnes fixes par classe.
    free_prompt sert à labelliser la colonne bbox_prompt dans les données
    (le nom de colonne reste toujours bbox_prompt).
    """
    # Index : image_name → label → liste de bboxes
    index: dict[str, dict[str, list]] = {}
    for d in detections:
        index.setdefault(d.image_name, {}).setdefault(d.label, []).append(d.bbox)

    rows = []
    for item in preprocessed:
        name = item["output_name"]
        dt = item["datetime"]
        dets = index.get(name, {})

        baigneur_bboxes = dets.get("baigneur", [])
        tente_bboxes = dets.get("tente", [])
        # tout ce qui n'est ni baigneur ni tente → colonne prompt
        prompt_bboxes = []
        for label, bboxes in dets.items():
            if label not in ("baigneur", "tente"):
                prompt_bboxes.extend(bboxes)

        rows.append(
            {
                "image_name": name,
                "bbox_baigneur": baigneur_bboxes if baigneur_bboxes else "",
                "count_baigneur": len(baigneur_bboxes),
                "bbox_tente": tente_bboxes if tente_bboxes else "",
                "count_tente": len(tente_bboxes),
                "bbox_prompt": prompt_bboxes if prompt_bboxes else "",
                "count_prompt": len(prompt_bboxes),
                "year": dt.year if dt else "",
                "month": dt.month if dt else "",
                "day": dt.day if dt else "",
                "hour": dt.hour if dt else "",
            }
        )

    df = pd.DataFrame(rows, columns=FIXED_COLUMNS)

    output_path = Path(tempfile.gettempdir()) / "résultats_détections.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)
