# ── Gestion des Logos ──────────────────────────────────────────────────────────
import ast
import gc
import os
from pathlib import Path

import base64

import cv2
import pandas as pd


from config import config
from src.constant import (
    ABS_DIR,
    COLORS,
    DISPLAY_MAX_DIM,
    IMAGE_EXTS,
    _path_registry,
    _last_df,
    _active_columns,
)
import gradio as gr


def _logo_b64(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _get_logo_path(logo_key):
    filename = config.ui.logos.get(logo_key, "")
    if not filename:
        return None
    return os.path.join(ABS_DIR, filename)


def _default_conf_for(model_name: str) -> float:
    """Retourne le seuil de confiance par défaut associé à un modèle donné."""
    return getattr(config.models.default_confidence, model_name, 0.4)


def _resolve_image_paths(files, input_mode, folder_path) -> tuple[list[str], str | None]:
    """
    Retourne (image_paths, error_message).
    Gère les deux modes d'import : upload Gradio et dossier local.
    """
    if input_mode == "Dossier local":
        if not folder_path or not os.path.isdir(folder_path):
            return [], "Dossier introuvable ou invalide."
        paths = sorted(
            str(p) for p in Path(folder_path).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            return [], "Aucune image trouvée dans ce dossier."
        return paths, None
    else:
        if not files:
            return [], "Aucune image chargée."
        return [f.name if hasattr(f, "name") else f for f in files], None


# ── Fonctions Backend ──────────────────────────────────────────────────────────


def _get_day_df(day_label: str) -> pd.DataFrame:
    if _last_df.empty or not day_label or "image_name" not in _last_df.columns:
        return pd.DataFrame()
    y, m, d = map(int, day_label.split("-"))
    mask = (_last_df["year"] == y) & (_last_df["month"] == m) & (_last_df["day"] == d)
    return _last_df[mask][_active_columns].copy()


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def on_image_select(table_df, evt: gr.SelectData, current_model):
    global _path_registry, _last_df, _active_columns
    if not config.ui.show_visualization:
        return None, ""
    if "image_name" not in table_df.columns:
        return None, ""

    image_name = table_df.iloc[evt.index[0]]["image_name"]
    original_path = _path_registry.get(image_name)
    if not original_path:
        return None, f"Chemin introuvable pour '{image_name}'."

    img_bgr = cv2.imread(original_path)
    if img_bgr is None:
        return None, "Erreur lors de la lecture de l'image."

    orig_h, orig_w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionnement pour l'affichage uniquement
    if max(orig_h, orig_w) > DISPLAY_MAX_DIM:
        scale = DISPLAY_MAX_DIM / max(orig_h, orig_w)
        img = cv2.resize(
            img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        scale = 1.0

    rows = _last_df[_last_df["image_name"] == image_name]
    if not rows.empty:
        row = rows.iloc[0]
        for col, label in [
            ("bbox_baigneur", "baigneur"),
            ("bbox_tente", "tente"),
            ("bbox_prompt", "prompt"),
        ]:
            if col in row:
                try:
                    for box in ast.literal_eval(str(row[col])):
                        scaled_box = [
                            box[0] * scale,
                            box[1] * scale,
                            box[2] * scale,
                            box[3] * scale,
                        ]
                        draw_box(img, scaled_box, COLORS[label], label)
                except Exception:
                    pass

    counts = rows.iloc[0] if not rows.empty else {}
    info_parts = [f" {image_name}"]
    if "count_baigneur" in counts:
        info_parts.append(f" Baigneurs: {int(counts['count_baigneur'])}")
    if "count_tente" in counts:
        info_parts.append(f" Tentes: {int(counts['count_tente'])}")
    if "count_prompt" in counts:
        info_parts.append(f" Prompt: {int(counts['count_prompt'])}")

    return img, " | ".join(info_parts)
