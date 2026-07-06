"""
Application Gradio : Détection et Comptage BiodivTourAlps
Lancement : python app.py
"""

import ast
import base64
import os
import gradio as gr
import cv2
import pandas as pd

from preprocessor import preprocess_images
from models.yolo import YoloModel
from models.sam3 import SAM3Model
from exporter import export_to_csv

# ── CONFIGURATION CHARTE GRAPHIQUE ─────────────────────────────────────────────

# Couleurs Bounding Boxes & Tableaux
COLORS  = {"baigneur": (239, 51, 64), "tente": (0, 122, 94), "prompt": (244, 162, 97)}
SAM_ALIASES = {"baigneur": "bather", "tente": "tent"}
CLASSES = ["baigneur", "tente"]
PAGE_THRESHOLD = 100   # au-delà : pagination par jour

# Variables d'état globales
_path_registry: dict[str, str] = {}
_last_df: pd.DataFrame = pd.DataFrame()

# ── Logos & Header ─────────────────────────────────────────────────────────────
ABS_DIR = os.path.dirname(os.path.abspath(__file__))

def _logo_b64(path):
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

LOGO_biodiv = _logo_b64(os.path.join(ABS_DIR, "docs/images/BiodivTourAlps_logo_def.png"))
LOGO_ecrin  = _logo_b64(os.path.join(ABS_DIR, "docs/images/logo_ecrins.png"))
LOGO_leca   = _logo_b64(os.path.join(ABS_DIR, "docs/images/logo_leca.png"))
LOGO_lacs   = _logo_b64(os.path.join(ABS_DIR, "docs/images/logo_lacs.png"))

def _logo_img(src, alt, h=60):
    return f'<img src="{src}" alt="{alt}" style="height:{h}px; width:auto; display:inline-block; vertical-align:middle; pointer-events:none; user-select:none;" draggable="false">'

# Header complet avec le titre intégré au centre
HEADER_HTML = f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            background:#ffffff;border:1px solid #e2e4e7;border-radius:12px;
            padding:15px 25px;gap:12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;">
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">
        {_logo_img(LOGO_ecrin,"Parc national des Ecrins", h=80)}
        {_logo_img(LOGO_lacs,"Lacs de montagne", h=80)}
        
        
    </div>
    <h2 style="flex:1;text-align:center;margin:0;font-family:'Inter',system-ui,sans-serif;
               font-size:1.6rem;font-weight:800;color:#981d97;line-height:1.2;">
        Détection et comptage automatique<br>
        <span style="font-size:1.5rem;font-weight:500;color:#6b7280;">Analyse de la fréquentation des lacs de montagne</span>
    </h2>
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">
        {_logo_img(LOGO_leca,"LECA", h=50)}
        {_logo_img(LOGO_biodiv,"BiodivTourAlps", h=80)}
    </div>
</div>
"""

MODEL_INFO = {
    "YOLO": "**YOLO** — Spécialisé, rapide, léger.\n\n*Seuil conseillé : 0.4* \n*mAP50 0.85, mAP50-95 0.65.*",
    "SAM3": "**SAM3** — Lourd (GPU requis). Classes définies ou prompt libre.\n\n*Seuil conseillé : 0.4* \n*mAP50 0.80 (baigneurs), 0.75 (tentes).* ",
}

# ── Thème Custom & CSS ─────────────────────────────────────────────────────────

# Déclinaison de la couleur de base #981d97 (du PNE)
custom_magenta = gr.themes.colors.Color(
    name="charte_magenta",
    c50="#fdf0fd", c100="#fae1fa", c200="#f3c3f3", c300="#eba5eb",
    c400="#db69db",
    c500="#981d97", 
    c600="#891a88", c700="#721671", c800="#5b115a", c900="#4c0e4b", c950="#2e092d"
)

# Thème 
custom_theme = gr.themes.Soft(
    primary_hue=custom_magenta,
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

# CSS 
CUSTOM_CSS = """
/* Encadrés d'information */
#model-info  { background-color: #f0f2f6; border-left: 4px solid #981d97; padding: 15px; border-radius: 8px; }
#tiling-info { background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px; color: #92400e; font-size: 0.9em; }

/* Cache la liste  des fichiers uploadés */
#file-upload .file-preview-holder { display: none !important; }
#file-upload .upload-container { min-height: unset !important; padding: 20px !important; }

/* Bouton principal */
#run-btn { font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
"""

# ── Fonctions Backend ──────────────────────────────────────────────────────────

def run_detection(files, model_name, targets, free_prompt, conf, use_tiling):
    global _path_registry, _last_df
    free_prompt = (free_prompt or "").strip()
    if not files:
        return None, "⚠️ Aucune image chargée.", pd.DataFrame(), gr.update(visible=False, choices=[])
    if not model_name:
        return None, "⚠️ Sélectionnez un modèle.", pd.DataFrame(), gr.update(visible=False, choices=[])
    
    targets_list  = list(targets or [])
    model_targets = [SAM_ALIASES.get(t, t) for t in targets_list] if model_name == "SAM3" else list(targets_list)
    if free_prompt: model_targets.append(free_prompt)
    if not model_targets:
        return None, "⚠️ Sélectionnez au moins une cible.", pd.DataFrame(), gr.update(visible=False, choices=[])
    
    image_paths  = [f.name if hasattr(f, "name") else f for f in files]
    preprocessed = preprocess_images(image_paths)
    _path_registry = {item["output_name"]: item["original_path"] for item in preprocessed}
    
    model        = YoloModel(conf=conf) if model_name == "YOLO" else SAM3Model(conf=conf)
    sam_en_to_fr = {v: k for k, v in SAM_ALIASES.items()}
    all_detections = []
    
    for item in preprocessed:
        dets = model.detect(item["original_path"], model_targets, use_tiling=use_tiling)
        dt   = item["datetime"]
        for d in dets:
            if model_name == "SAM3" and d.label in sam_en_to_fr: d.label = sam_en_to_fr[d.label]
            d.image_name = item["output_name"]
            if dt: d.year, d.month, d.day, d.hour = dt.year, dt.month, dt.day, dt.hour
        all_detections.extend(dets)
        
    csv_path = export_to_csv(preprocessed, all_detections, free_prompt)
    _last_df  = pd.read_csv(csv_path)
    n       = len(preprocessed)
    summary = f"✅ {n} image(s) analysée(s) — {len(all_detections)} objets détectés."
    
    if n > PAGE_THRESHOLD:
        days = (_last_df[["year","month","day"]].dropna().drop_duplicates().astype(int).apply(lambda r: f"{r.year:04d}-{r.month:02d}-{r.day:02d}", axis=1).tolist())
        if days:
            first_df = _get_day_df(days[0])
            return csv_path, summary, first_df, gr.update(visible=True, choices=days, value=days[0])
            
    table_df = _last_df[["image_name","count_baigneur","count_tente","count_prompt"]].copy()
    return csv_path, summary, table_df, gr.update(visible=False, choices=[])

def _get_day_df(day_label: str) -> pd.DataFrame:
    if _last_df.empty or not day_label: return pd.DataFrame()
    y, m, d = map(int, day_label.split("-"))
    mask = (_last_df["year"]==y) & (_last_df["month"]==m) & (_last_df["day"]==d)
    return _last_df[mask][["image_name","count_baigneur","count_tente","count_prompt"]].copy()

def draw_box(img, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-h-8), (x1+w, y1), color, -1)
    cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def on_image_select(table_df, evt: gr.SelectData):
    image_name    = table_df.iloc[evt.index[0]]["image_name"]
    original_path = _path_registry.get(image_name)
    if not original_path: return None, f"Chemin introuvable pour '{image_name}'."
    
    img_bgr = cv2.imread(original_path)
    if img_bgr is None: return None, "Erreur lors de la lecture de l'image."
    img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    rows = _last_df[_last_df["image_name"] == image_name]
    if not rows.empty:
        row = rows.iloc[0]
        for col, label in [("bbox_baigneur","baigneur"),("bbox_tente","tente"),("bbox_prompt","prompt")]:
            try:
                for box in ast.literal_eval(str(row[col])): draw_box(img, box, COLORS[label], label)
            except Exception: pass
            
    counts = rows.iloc[0] if not rows.empty else {}
    info   = (f"📄 {image_name} | "
              f"🔻 Baigneurs: {int(counts.get('count_baigneur',0))} | "
              f"🔻 Tentes: {int(counts.get('count_tente',0))} | "
              f"🔻 Prompt: {int(counts.get('count_prompt',0))}")
    return img, info

# ── Interface  ──────────────────────────────────────────────────────────

with gr.Blocks(title="Détection Fréquentation - BiodivTourAlps", theme=custom_theme, css=CUSTOM_CSS) as demo:

    stored_files = gr.State([])
    
    # Intégration de la bannière avec logos et titre
    gr.HTML(HEADER_HTML)

    with gr.Row():

        # ── Panneau gauche : Configuration ──────────────────────────────────
        with gr.Column(scale=1):
            
            with gr.Group():
                images_input  = gr.File(label="1. Importer les images", file_count="multiple", file_types=["image"], elem_id="file-upload")
                images_count  = gr.Textbox(label="", interactive=False, value="", show_label=False, placeholder="En attente d'images...")

            with gr.Accordion("2. Paramètres de l'IA", open=True):
                model_input = gr.Radio(choices=["YOLO","SAM3"], value=None, label="Modèle de détection")
                model_info  = gr.Markdown(value="", elem_id="model-info", visible=False)

                targets_input     = gr.CheckboxGroup(choices=CLASSES, value=[], label="Cibles à rechercher")
                free_prompt_input = gr.Textbox(label="Prompt libre (SAM3 uniquement)", placeholder="ex: sac à dos, chien...", lines=1, visible=False)

                conf_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.4, label="Seuil de confiance")

                with gr.Group():
                    tiling_checkbox = gr.Checkbox(label="Activer le tiling (Petits objets)", value=False)
                    tiling_info     = gr.Markdown(
                        value="💡 **Tiling** : Découpe l'image pour mieux voir les petits objets, mais ralentit le traitement.",
                        elem_id="tiling-info", visible=False)

            run_btn = gr.Button("🔍 Lancer l'analyse", variant="primary", size="lg", elem_id="run-btn")
            
            with gr.Group():
                status_output = gr.Textbox(label="Résultat de l'analyse", interactive=False, show_label=False, placeholder="Statut...")
                csv_output    = gr.File(label="📥 Exporter les données (CSV)")

        # ── Panneau droit : Visualisation ───────────────────────────────────
        with gr.Column(scale=2):

            with gr.Group():
                day_selector   = gr.Dropdown(label="📅 Filtrer par date", choices=[], visible=False)
                selector_table = gr.Dataframe(label="📊 Tableau des résultats — Cliquez sur une ligne pour visualiser", interactive=False, wrap=True)
            
            with gr.Group():
                visu_info  = gr.Textbox(label="Détails de l'image", interactive=False, show_label=False, placeholder="Sélectionnez une image ci-dessus...")
                visu_image = gr.Image(label="Visualisation de la détection", type="numpy", interactive=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    images_input.change(
        fn=lambda imgs: (imgs or [], f"📂 {len(imgs)} image(s) chargée(s)" if imgs else ""),
        inputs=images_input, outputs=[stored_files, images_count],
    )

    model_input.change(
        fn=lambda m: (gr.update(value=MODEL_INFO.get(m,""), visible=bool(m)),
                      gr.update(visible=(m=="SAM3"))),
        inputs=model_input, outputs=[model_info, free_prompt_input],
        show_progress="hidden",
    )

    free_prompt_input.change(
        fn=lambda t, m: gr.update(value="SAM3") if t and m != "SAM3" else gr.update(),
        inputs=[free_prompt_input, model_input], outputs=model_input,
        show_progress="hidden",
    )

    tiling_checkbox.change(fn=lambda v: gr.update(visible=v), inputs=tiling_checkbox, outputs=tiling_info)

    run_btn.click(
        fn=run_detection,
        inputs=[stored_files, model_input, targets_input, free_prompt_input, conf_slider, tiling_checkbox],
        outputs=[csv_output, status_output, selector_table, day_selector],
    )

    day_selector.change(fn=_get_day_df, inputs=day_selector, outputs=selector_table)

    selector_table.select(fn=on_image_select, inputs=selector_table, outputs=[visu_image, visu_info])

if __name__ == "__main__":
    demo.queue()
    demo.launch()