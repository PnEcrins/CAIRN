"""
Application Gradio : Détection et Comptage BiodivTourAlps
Lancement : python app.py
"""

import os
import sys

# ── CONFIGURATION STRICTE DES CHEMINS D'IMPORTATION (SYS.PATH) ───────────────
ABS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_OFB_DIR = os.path.normpath(os.path.join(ABS_DIR, "ofb_attendance"))

if SCRIPTS_OFB_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_OFB_DIR)
if ABS_DIR not in sys.path:
    sys.path.insert(0, ABS_DIR)

import ast
import gc
import base64
import gradio as gr
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import csv

from preprocessor import preprocess_images
from models.yolo import YoloModel
from models.sam3 import SAM3Model
from exporter import export_to_csv
from config import Config

# ── Importation sécurisée du script OFB ───────────────────────────────────────
HAS_OFB_SCRIPT = False
classification_ofb = None
sequence_image = None
gathering_time = None
YOLO_ofb = None

try:
    from ofb_attendance.yolov8_attendance import (
        classification as classification_ofb,
        sequence_image,
        gathering_time,
    )
    from ultralytics import YOLO as YOLO_ofb

    HAS_OFB_SCRIPT = True
except ImportError as e:
    print(f" Impossible de charger le script ofb_attendance : {e}")

# ── CHARGEMENT DE LA CONFIGURATION ────────────────────────────────────────────
CONFIG_PATH = os.environ.get("BIODIV_APP_CONFIG", os.path.join(ABS_DIR, "config.yaml"))
config = Config(CONFIG_PATH)


COLORS = {"baigneur": (239, 51, 64), "tente": (0, 122, 94), "prompt": (244, 162, 97)}
SAM_ALIASES = {"baigneur": "bather", "tente": "tent"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DISPLAY_MAX_DIM = 1280

# ── Variables d'état globales ──────────────────────────────────────────────────
_path_registry: dict[str, str] = {}
_last_df: pd.DataFrame = pd.DataFrame()
_active_columns: list[str] = ["image_name"]


# ── Gestion des Logos ──────────────────────────────────────────────────────────
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
        return ""
    assets_path = os.path.join(ABS_DIR, "assets", filename)
    return assets_path if os.path.exists(assets_path) else os.path.join(ABS_DIR, filename)


LOGO_TL1 = _logo_b64(_get_logo_path("top_left_1"))
LOGO_TL2 = _logo_b64(_get_logo_path("top_left_2"))
LOGO_TR1 = _logo_b64(_get_logo_path("top_right_1"))
LOGO_TR2 = _logo_b64(_get_logo_path("top_right_2"))


def _logo_img(src, alt, h=60):
    if not src:
        return ""
    return f'<img src="{src}" alt="{alt}" style="height:{h}px; width:auto; display:inline-block; vertical-align:middle; pointer-events:none; user-select:none;" draggable="false">'


HEADER_HTML = f"""
<div style="display:flex;align-items:center;justify-content:space-between;background:#ffffff;border:1px solid #e2e4e7;border-radius:12px;padding:15px 25px;gap:12px;box-shadow:0 2px 4px rgba(0,0,0,0.05);margin-bottom:10px;">
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">{_logo_img(LOGO_TL1, "HG1", h=80)}{_logo_img(LOGO_TL2, "HG2", h=80)}</div>
    <h2 style="flex:1;text-align:center;margin:0;font-family:'Inter',sans-serif;font-size:1.6rem;font-weight:800;color:{config.ui.theme_color};line-height:1.2;">
        {config.ui.title}<br><span style="font-size:1.5rem;font-weight:500;color:#6b7280;">Analyse de la fréquentation des lacs de montagne</span>
    </h2>
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">{_logo_img(LOGO_TR1, "HD1", h=50)}{_logo_img(LOGO_TR2, "HD2", h=80)}</div>
</div>
"""

MODEL_INFO = {
    "YOLO": "**YOLOv26** — Modèle spécialisé et rapide. Entrainé sur 3000 images. Meilleures performances pour détection de tentes prises de loin. \n\n*Seuil conseillé : 0.4*",
    "SAM3": "**SAM3** — Attention ! Modèle très lourd qui nécéssite d'avoir un GPU pour tourner convenablement (plusieurs dizaines de secondes par images sinon.) Permet de détecter n'importe quoi (via l'option prompt libre ci-dessous) \n\n*Seuil conseillé : 0.4*",
    "YOLOv8_Squelette": "**YOLOv8** — Modèle Yolo permettant de détecter automatiquement des personnes, leurs directions de passages, leurs activités.",
}

# ── Thème Custom & CSS ─────────────────────────────────────────────────────────
custom_magenta = gr.themes.colors.Color(
    name="charte_magenta",
    c50="#fdf0fd",
    c100="#fae1fa",
    c200="#f3c3f3",
    c300="#eba5eb",
    c400="#db69db",
    c500=config.ui.theme_color,
    c600=config.ui.theme_color,
    c700=config.ui.theme_color,
    c800="#5b115a",
    c900="#4c0e4b",
    c950="#2e092d",
)
custom_theme = gr.themes.Soft(
    primary_hue=custom_magenta,
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)
CUSTOM_CSS = (
    f"#model-info {{ background-color: #f0f2f6; border-left: 4px solid {config.ui.theme_color}; padding: 15px; border-radius: 8px; }}"
    f"#tiling-info {{ background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px; color: #92400e; font-size: 0.9em; }}"
    f"#file-upload .file-preview-holder {{ display: none !important; }}"
    f"#file-upload .upload-container {{ min-height: unset !important; padding: 20px !important; }}"
    f"#run-btn {{ font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }}"
    f"#folder-input-info {{ background-color: #f0f7ff; border-left: 4px solid #3b82f6; padding: 10px 12px; border-radius: 8px; color: #1e40af; font-size: 0.85em; margin-top: 4px; }}"
)


# ── Utilitaires ────────────────────────────────────────────────────────────────
def _free_model(model):
    """Libère proprement un modèle ML de la mémoire."""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
def run_detection(
    files,
    folder_path,
    input_mode,
    analysis_type,
    timelapse_model,
    targets,
    free_prompt,
    conf,
    use_tiling,
):
    global _path_registry, _last_df, _active_columns

    image_paths, err = _resolve_image_paths(files, input_mode, folder_path)
    if err:
        return None, f" {err}", pd.DataFrame(), gr.update(visible=False, choices=[])

    # ── PIPELINE PIÈGE PHOTOS (OFB_ATTENDANCE) ─────────────────────────────────
    if analysis_type == "Analyse Piège Photos Randonneurs":
        if not HAS_OFB_SCRIPT:
            return (
                None,
                " Erreur : Les scripts du dossier 'ofb_attendance' sont introuvables.",
                pd.DataFrame(),
                gr.update(visible=False, choices=[]),
            )

        with tempfile.TemporaryDirectory() as temp_input_dir:
            for src_path in image_paths:
                dst_path = os.path.join(temp_input_dir, os.path.basename(src_path))
                try:
                    with open(src_path, "rb") as sf, open(dst_path, "wb") as df:
                        df.write(sf.read())
                except Exception as e:
                    return (
                        None,
                        f" Erreur lors de la préparation des fichiers : {e}",
                        pd.DataFrame(),
                        gr.update(visible=False, choices=[]),
                    )

            try:
                model_google = YOLO_ofb(os.path.join(ABS_DIR, "weights", "yolov8x-oiv7.pt"))
                model_pose = YOLO_ofb(os.path.join(ABS_DIR, "weights", "yolov8n-pose.pt"))
            except Exception as e:
                return (
                    None,
                    f" Erreur lors du chargement des poids : {e}",
                    pd.DataFrame(),
                    gr.update(visible=False, choices=[]),
                )

            try:
                results_data = classification_ofb(
                    folder_pics=temp_input_dir,
                    model_google=model_google,
                    model_pose=model_pose,
                    classfication_date_file=os.path.join(ABS_DIR, "last_classification_date.txt"),
                    classes_path=os.path.join(ABS_DIR, "ofb_attendance", "classes.json"),
                    classes_exception_path=os.path.join(
                        ABS_DIR, "ofb_attendance", "classes_exeptions_rules.json"
                    ),
                    blur=False,
                    conf_google=conf,
                    conf_pose=conf,
                    format=True,
                )
                results_data = sequence_image(
                    results_data, config.ofb_attendance.sequence_duration
                )
                results_data = gathering_time(results_data, config.ofb_attendance.time_step or "h")

                tmp_csv = tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".csv", encoding="utf-8"
                )
                with open(tmp_csv.name, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(results_data)
                return_csv_path = tmp_csv.name

            except Exception as e:
                return (
                    None,
                    f" Erreur durant le traitement : {e}",
                    pd.DataFrame(),
                    gr.update(visible=False, choices=[]),
                )

        summary = f" Analyse terminée. {len(image_paths)} image(s) traitée(s)."
        return return_csv_path, summary, pd.DataFrame(), gr.update(visible=False, choices=[])

    # ── PIPELINE TIMELAPSE (YOLO / SAM3) ──────────────────────────────────────
    model_name = timelapse_model
    targets_list = list(targets or [])
    model_targets = (
        [SAM_ALIASES.get(t, t) for t in targets_list]
        if model_name == "SAM3"
        else list(targets_list)
    )
    if free_prompt:
        model_targets.append(free_prompt.strip())
    if not model_targets:
        return (
            None,
            "Sélectionnez au moins une cible.",
            pd.DataFrame(),
            gr.update(visible=False, choices=[]),
        )

    _active_columns = ["image_name"]
    csv_columns_to_keep = ["image_name", "datetime", "year", "month", "day", "hour"]

    if "baigneur" in targets_list:
        _active_columns.append("count_baigneur")
        csv_columns_to_keep.extend(["count_baigneur", "bbox_baigneur"])
    if "tente" in targets_list:
        _active_columns.append("count_tente")
        csv_columns_to_keep.extend(["count_tente", "bbox_tente"])
    if free_prompt:
        _active_columns.append("count_prompt")
        csv_columns_to_keep.extend(["count_prompt", "bbox_prompt"])

    preprocessed = preprocess_images(image_paths)
    _path_registry = {item["output_name"]: item["original_path"] for item in preprocessed}

    device_setting = config.models.device
    model = (
        YoloModel(conf=conf, device=device_setting, model_path=config.features.model_path.YOLO)
        if model_name == "YOLO"
        else SAM3Model(
            conf=conf, device=device_setting, model_path=config.features.model_path.SAM3
        )
    )

    sam_en_to_fr = {v: k for k, v in SAM_ALIASES.items()}
    all_detections = []

    for item in preprocessed:
        dets = model.detect(item["original_path"], model_targets, use_tiling=use_tiling)
        dt = item["datetime"]
        for d in dets:
            if model_name == "SAM3" and d.label in sam_en_to_fr:
                d.label = sam_en_to_fr[d.label]
            d.image_name = item["output_name"]
            if dt:
                d.year, d.month, d.day, d.hour = dt.year, dt.month, dt.day, dt.hour
        all_detections.extend(dets)

    # Libération mémoire (critique pour SAM3)
    _free_model(model)

    raw_csv_path = export_to_csv(preprocessed, all_detections, free_prompt)
    full_df = pd.read_csv(raw_csv_path)

    existing_csv_cols = [c for c in csv_columns_to_keep if c in full_df.columns]
    _last_df = full_df[existing_csv_cols].copy()
    _last_df.to_csv(raw_csv_path, index=False)

    summary = f" {len(preprocessed)} image(s) analysée(s) — {len(all_detections)} objets détectés."

    if config.ui.show_visualization and len(preprocessed) > config.ui.page_threshold:
        days = (
            _last_df[["year", "month", "day"]]
            .dropna()
            .drop_duplicates()
            .astype(int)
            .apply(lambda r: f"{r.year:04d}-{r.month:02d}-{r.day:02d}", axis=1)
            .tolist()
        )
        if days:
            return (
                raw_csv_path,
                summary,
                _get_day_df(days[0]),
                gr.update(visible=True, choices=days, value=days[0]),
            )

    return (
        raw_csv_path,
        summary,
        _last_df[_active_columns].copy(),
        gr.update(visible=False, choices=[]),
    )


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


# ── Callbacks UI ───────────────────────────────────────────────────────────────
def on_upload_change(imgs):
    imgs = imgs or []
    return imgs, f" {len(imgs)} image(s) chargée(s)" if imgs else ""


def on_folder_change(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return [], "Dossier invalide ou introuvable."
    paths = sorted(str(p) for p in Path(folder_path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return paths, f"📁 {len(paths)} image(s) trouvée(s)"


def toggle_input_mode(mode):
    is_upload = mode == "Upload fichiers"
    return (
        gr.update(visible=is_upload),
        gr.update(visible=not is_upload),
        gr.update(visible=not is_upload),
    )


def update_ui_visibility(analysis_mode, timelapse_model):
    if analysis_mode == "Analyse Piège Photos Randonneurs":
        return (
            gr.update(visible=False),
            gr.update(value=MODEL_INFO["YOLOv8_Squelette"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    else:
        is_sam3 = timelapse_model == "SAM3"
        return (
            gr.update(visible=True),
            gr.update(value=MODEL_INFO.get(timelapse_model, ""), visible=True),
            gr.update(visible=True),
            gr.update(visible=is_sam3),
            gr.update(visible=config.models.show_confidence_slider),
            gr.update(visible=config.features.allow_tiling),
            gr.update(visible=config.ui.show_visualization),
        )


# ── Interface Gradio ───────────────────────────────────────────────────────────
with gr.Blocks(title=config.ui.title) as demo:

    stored_files = gr.State([])
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Colonne gauche : paramètres ────────────────────────────────────────
        with gr.Column(scale=1):

            with gr.Group():
                input_mode = gr.Radio(
                    choices=["Upload fichiers", "Dossier local"],
                    value="Upload fichiers",
                    label="Mode d'import des images",
                )
                images_input = gr.File(
                    label="Importer les images",
                    file_count="multiple",
                    file_types=["image"],
                    elem_id="file-upload",
                    visible=True,
                )
                folder_input = gr.Textbox(
                    label="Chemin du dossier",
                    placeholder="/chemin/vers/mon/dossier/images",
                    visible=False,
                )
                folder_info = gr.Markdown(
                    value=" Les images sont lues directement depuis le disque, sans copie.",
                    elem_id="folder-input-info",
                    visible=False,
                )
                images_count = gr.Textbox(
                    label="",
                    interactive=False,
                    value="",
                    show_label=False,
                    placeholder="En attente d'images...",
                )

            with gr.Accordion("2. Paramètres de l'IA", open=True):
                analysis_type = gr.Radio(
                    choices=["Analyse Timelapse", "Analyse Piège Photos Randonneurs"],
                    value="Analyse Timelapse",
                    label="Type d'analyse",
                )
                model_input = gr.Radio(
                    choices=["YOLO", "SAM3"],
                    value="YOLO",
                    label="Modèle Timelapse",
                )
                model_info = gr.Markdown(value="", elem_id="model-info", visible=False)

                targets_input = gr.CheckboxGroup(
                    choices=config.features.classes,
                    value=config.features.classes,
                    label="Cibles à rechercher",
                )
                free_prompt_input = gr.Textbox(
                    label="Prompt libre (SAM3 uniquement)",
                    placeholder="ex: sac à dos, chien...",
                    lines=1,
                    visible=False,
                )
                conf_slider = gr.Slider(
                    minimum=config.models.confidence_range[0],
                    maximum=config.models.confidence_range[1],
                    step=0.05,
                    value=config.models.default_confidence,
                    label="Seuil de confiance",
                    visible=config.models.show_confidence_slider,
                )
                with gr.Group(visible=config.features.allow_tiling) as tiling_group:
                    tiling_checkbox = gr.Checkbox(
                        label="Activer le tiling (Petits objets)", value=True
                    )
                    tiling_info = gr.Markdown(
                        value=" **Tiling** : Technique de découpage de l'image pour augmenter les performances pour la détection de petits objets. Fortement recommandé.",
                        elem_id="tiling-info",
                        visible=True,
                    )

            # Boutons alignés côte à côte
            with gr.Row():
                run_btn = gr.Button(
                    " Lancer l'analyse", variant="primary", size="lg", elem_id="run-btn"
                )
                stop_btn = gr.Button("Annuler", variant="stop", size="lg")

            with gr.Group():
                status_output = gr.Textbox(
                    label="Résultat de l'analyse",
                    interactive=False,
                    show_label=False,
                    placeholder="Statut...",
                )
                csv_output = gr.File(label="Enregistrer le rapport (CSV)")

        # ── Colonne droite : visualisation ─────────────────────────────────────
        with gr.Column(scale=2, visible=config.ui.show_visualization) as right_side:
            with gr.Group():
                day_selector = gr.Dropdown(label=" Filtrer par date", choices=[], visible=False)
                selector_table = gr.Dataframe(
                    label=" Tableau des résultats — Cliquez sur une ligne pour visualiser",
                    interactive=False,
                    wrap=True,
                    row_count=10,
                )
            with gr.Group():
                visu_info = gr.Textbox(
                    label="Détails de l'image",
                    interactive=False,
                    show_label=False,
                    placeholder="Sélectionnez une image ci-dessus...",
                )
                visu_image = gr.Image(
                    label="Visualisation de la détection", type="numpy", interactive=False
                )

    # ── Wiring des événements ──────────────────────────────────────────────────
    _visibility_outputs = [
        model_input,
        model_info,
        targets_input,
        free_prompt_input,
        conf_slider,
        tiling_group,
        right_side,
    ]

    input_mode.change(
        fn=toggle_input_mode,
        inputs=input_mode,
        outputs=[images_input, folder_input, folder_info],
    )
    images_input.change(
        fn=on_upload_change, inputs=images_input, outputs=[stored_files, images_count]
    )
    folder_input.change(
        fn=on_folder_change, inputs=folder_input, outputs=[stored_files, images_count]
    )

    analysis_type.change(
        fn=update_ui_visibility,
        inputs=[analysis_type, model_input],
        outputs=_visibility_outputs,
        show_progress="hidden",
    )
    model_input.change(
        fn=update_ui_visibility,
        inputs=[analysis_type, model_input],
        outputs=_visibility_outputs,
        show_progress="hidden",
    )

    if config.features.allow_tiling:
        tiling_checkbox.change(
            fn=lambda v: gr.update(visible=v), inputs=tiling_checkbox, outputs=tiling_info
        )

    # Capturer l'événement de lancement dans une variable
    run_event = run_btn.click(
        fn=run_detection,
        inputs=[
            stored_files,
            folder_input,
            input_mode,
            analysis_type,
            model_input,
            targets_input,
            free_prompt_input,
            conf_slider,
            tiling_checkbox if config.features.allow_tiling else gr.State(False),
        ],
        outputs=[csv_output, status_output, selector_table, day_selector],
    )

    # Associer le bouton stop à l'annulation de cet événement spécifique
    stop_btn.click(
        fn=lambda: gr.Info("Annulation en cours..."),
        inputs=None,
        outputs=None,
        cancels=[run_event],
    )

    if config.ui.show_visualization:
        day_selector.change(fn=_get_day_df, inputs=day_selector, outputs=selector_table)
        selector_table.select(
            fn=on_image_select,
            inputs=[selector_table, model_input],
            outputs=[visu_image, visu_info],
        )

    demo.load(
        fn=update_ui_visibility,
        inputs=[analysis_type, model_input],
        outputs=_visibility_outputs,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=custom_theme, css=CUSTOM_CSS)
