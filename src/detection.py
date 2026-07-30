import gc
import os
import sys
import tempfile

import pandas as pd
import gradio as gr
from src.utils import _get_day_df, _resolve_image_paths


import os

from src.constant import *
from src.utils import (
    _resolve_image_paths,
)

import gradio as gr
import pandas as pd
import tempfile
import csv

from src.models.yolo import YoloModel
from src.models.sam3 import SAM3Model
from src.exporter import export_to_csv
from src.preprocessor import preprocess_images
from src.config import config

# ── Importation sécurisée du script OFB ───────────────────────────────────────
HAS_OFB_SCRIPT = False
classification_ofb = None
sequence_image = None
gathering_time = None
YOLO_ofb = None

# ── CONFIGURATION STRICTE DES CHEMINS D'IMPORTATION (SYS.PATH) ───────────────
SCRIPTS_OFB_DIR = os.path.normpath(os.path.join(ABS_DIR, "ofb_attendance"))

if SCRIPTS_OFB_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_OFB_DIR)
if ABS_DIR not in sys.path:
    sys.path.insert(0, ABS_DIR)

try:
    from src.ofb_attendance.yolov8_attendance import (
        classification as classification_ofb,
        sequence_image,
        gathering_time,
    )
    from ultralytics import YOLO as YOLO_ofb

    HAS_OFB_SCRIPT = True
except ImportError as e:
    print(f" Impossible de charger le script ofb_attendance : {e}")


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
    path_registry,
    last_df,
    active_columns,
):
    image_paths, err = _resolve_image_paths(files, input_mode, folder_path)
    if err:
        return (
            None,
            f" {err}",
            pd.DataFrame(),
            gr.update(visible=False, choices=[]),
            path_registry,
            last_df,
            active_columns,
        )

    # ── PIPELINE PIÈGE PHOTOS (OFB_ATTENDANCE) ─────────────────────────────────
    if analysis_type == "auto":
        if not HAS_OFB_SCRIPT:
            return (
                None,
                " Erreur : Les scripts du dossier 'ofb_attendance' sont introuvables.",
                pd.DataFrame(),
                gr.update(visible=False, choices=[]),
                path_registry,
                last_df,
                active_columns,
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
                        path_registry,
                        last_df,
                        active_columns,
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
                    path_registry,
                    last_df,
                    active_columns,
                )

            try:
                results_data = classification_ofb(
                    folder_pics=temp_input_dir,
                    model_google=model_google,
                    model_pose=model_pose,
                    classfication_date_file=os.path.join(ABS_DIR, "last_classification_date.txt"),
                    classes_path=os.path.join(ABS_DIR, "src", "ofb_attendance", "classes.json"),
                    classes_exception_path=os.path.join(
                        ABS_DIR, "src", "ofb_attendance", "classes_exeptions_rules.json"
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
                    path_registry,
                    last_df,
                    active_columns,
                )

        summary = f" Analyse terminée. {len(image_paths)} image(s) traitée(s)."
        return (
            return_csv_path,
            summary,
            pd.DataFrame(),
            gr.update(visible=False, choices=[]),
            path_registry,
            last_df,
            active_columns,
        )

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
            path_registry,
            last_df,
            active_columns,
        )

    active_columns = ["image_name"]
    csv_columns_to_keep = ["image_name", "date", "datetime", "year", "month", "day", "hour"]

    if "baigneur" in targets_list:
        active_columns.append("count_baigneur")
        csv_columns_to_keep.extend(["count_baigneur", "bbox_baigneur"])
    if "tente" in targets_list:
        active_columns.append("count_tente")
        csv_columns_to_keep.extend(["count_tente", "bbox_tente"])
    if free_prompt:
        active_columns.append("count_prompt")
        csv_columns_to_keep.extend(["count_prompt", "bbox_prompt"])

    preprocessed = preprocess_images(image_paths)
    path_registry = {item["output_name"]: item["original_path"] for item in preprocessed}
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
    last_df = full_df[existing_csv_cols].copy()
    last_df.to_csv(raw_csv_path, index=False)

    summary = f" {len(preprocessed)} image(s) analysée(s) — {len(all_detections)} objets détectés."

    if config.ui.show_visualization and len(preprocessed) > config.ui.page_threshold:
        days = (
            last_df[["year", "month", "day"]]
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
                _get_day_df(days[0], last_df, active_columns),
                gr.update(visible=True, choices=days, value=days[0]),
                path_registry,
                last_df,
                active_columns,
            )

    return (
        raw_csv_path,
        summary,
        last_df[active_columns].copy(),
        gr.update(visible=False, choices=[]),
        path_registry,
        last_df,
        active_columns,
    )
