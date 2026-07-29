from pathlib import Path
import os
import gradio as gr
import pandas as pd

from src.constant import MODEL_INFO, IMAGE_EXTS
from src.detection import run_detection
from src.utils import _default_conf_for, _get_day_df, on_image_select
from config import config


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
    is_upload = mode == "upload"
    return (
        gr.update(visible=is_upload),
        gr.update(visible=not is_upload),
        gr.update(visible=not is_upload),
    )


def update_ui_visibility(analysis_mode, timelapse_model):
    if analysis_mode == "auto":
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
            gr.update(
                value=_default_conf_for(timelapse_model),
                visible=config.models.show_confidence_slider,
            ),
            gr.update(visible=config.features.allow_tiling),
            gr.update(visible=config.ui.show_visualization),
        )


def get_layout(config, HEADER_HTML, gradio_context_variable):
    stored_files = gr.State([])
    path_registry_state = gr.State({})
    last_df_state = gr.State(pd.DataFrame())
    active_columns_state = gr.State(["image_name"])
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Colonne gauche : paramètres ────────────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("<div class='section-title'>1. Mode d'import des images</div>")
            with gr.Group():
                input_mode = gr.Radio(
                    (
                        [("Upload fichiers", "upload"), ("Dossier local", "folder")]
                        if config.ui.use_local_storage
                        else [("Upload fichiers", "upload")]
                    ),
                    label="Source des images",
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

            gr.Markdown("<div class='section-title'>2. Paramètres de l'IA</div>")
            with gr.Group():
                analysis_type = gr.Radio(
                    choices=[("Timelapse", "timelapse"), ("Détection automatique", "auto")],
                    value="timelapse",
                    label="Type de données à analyser",
                )

                targets_input = gr.CheckboxGroup(
                    choices=config.features.classes,
                    value=config.features.classes,
                    label="Cibles à rechercher",
                )
                model_input = gr.Radio(
                    choices=config.models.available,
                    value=config.models.available[0] if config.models.available else "YOLO",
                    label="Modèle",
                )
                model_info = gr.Markdown(value="", elem_id="model-info", visible=False)

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
                    value=_default_conf_for("YOLO"),
                    label="Seuil de confiance",
                    visible=config.models.show_confidence_slider,
                )
                with gr.Group(visible=config.features.allow_tiling) as tiling_group:
                    tiling_checkbox = gr.Checkbox(
                        label="Activer le tiling (Petits objets)", value=True
                    )
                    tiling_info = gr.Markdown(
                        value=" **Tiling** : Technique de découpage de l'image pour augmenter les performances pour la détection de petits objets. Fortement recommandé.\n\n Augmente le temps de traitement d'un facteur 10. \n\n En savoir plus : [Documentation Tiling](https://docs.ultralytics.com/fr/guides/sahi-tiled-inference#introduction-%C3%A0-sahi)",
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

            with gr.Group():
                csv_output = gr.File(
                    label="Rapport CSV généré (à télécharger en bas à droite de ce cadre)"
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
                path_registry_state,
                last_df_state,
                active_columns_state,
            ],
            outputs=[
                csv_output,
                status_output,
                selector_table,
                day_selector,
                path_registry_state,
                last_df_state,
                active_columns_state,
            ],
        )

        # Associer le bouton stop à l'annulation de cet événement spécifique
        stop_btn.click(
            fn=lambda: gr.Info("Annulation en cours..."),
            inputs=None,
            outputs=None,
            cancels=[run_event],
        )

        if config.ui.show_visualization:
            day_selector.change(
                fn=_get_day_df,
                inputs=[day_selector, last_df_state, active_columns_state],
                outputs=selector_table,
            )
            selector_table.select(
                fn=on_image_select,
                inputs=[selector_table, model_input, path_registry_state, last_df_state],
                outputs=[visu_image, visu_info],
            )

        gradio_context_variable.load(
            fn=update_ui_visibility,
            inputs=[analysis_type, model_input],
            outputs=_visibility_outputs,
            show_progress="hidden",
        )
