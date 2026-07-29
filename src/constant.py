import os
import gradio as gr

ABS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_theme(config):
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
    return gr.themes.Soft(
        primary_hue=custom_magenta,
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    )


MODEL_INFO = {
    "YOLO": (
        "**YOLOv26** — Modèle spécialisé et rapide. Fine-tuné sur 3000 images et 2 classes (tente, baigneur).\n\n "
        "*Seuil conseillé : 0.3* \n\n"
        "**Performances (mAP)**\n"
        "- Sans tiling : 0.5\n"
        "- Avec tiling : 0.85\n\n"
        "**Temps de traitement moyen (CPU, 100 images)**\n"
        "- Sans tiling : ~30 secondes\n"
        "- Avec tiling : ~30 minutes\n\n"
        "En savoir plus : [Documentation YOLOv26](https://docs.ultralytics.com/fr/models/yolo26#pr%C3%A9sentation)"
    ),
    "SAM3": (
        "**SAM3** — Attention ! Modèle très lourd qui nécessite d'avoir un GPU pour tourner convenablement "
        "(plusieurs dizaines de secondes par image sinon.) Permet de détecter n'importe quoi "
        "(via l'option prompt libre ci-dessous). \n\n"
        "*Seuil conseillé : 0.6* \n\n"
        "**Performances (mAP)**\n"
        "- Sans tiling : 0.4\n"
        "- Avec tiling : 0.5\n\n"
        "**Temps de traitement moyen (CPU, 100 images)**\n"
        "- Sans tiling : ~30 minutes\n"
        "- Avec tiling : ~10 heures\n\n"
        "En savoir plus : [Documentation SAM3](https://docs.ultralytics.com/fr/models/sam-3)"
    ),
    "YOLOv8_Squelette": "**YOLOv8** — Modèle Yolo permettant de détecter automatiquement des personnes, leurs directions de passages, leurs activités. \n\n En savoir plus : [Git du projet](https://github.com/Attendance-PNE-OFB/yolov8-attendance/blob/main/README-FR.md)",
}

COLORS = {"baigneur": (239, 51, 64), "tente": (0, 122, 94), "prompt": (244, 162, 97)}
SAM_ALIASES = {"baigneur": "bather", "tente": "tent"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DISPLAY_MAX_DIM = 1280


def _logo_img(src, alt, h=60):
    if not src:
        return ""
    return f'<img src="{src}" alt="{alt}" style="height:{h}px; width:auto; display:inline-block; vertical-align:middle; pointer-events:none; user-select:none;" draggable="false">'


def get_html_header(config, logo_tl1, logo_tl2, logo_tr1, logo_tr2):
    return f"""
<div style="display:flex;align-items:center;justify-content:space-between;background:#ffffff;border:1px solid #e2e4e7;border-radius:12px;padding:15px 25px;gap:12px;box-shadow:0 2px 4px rgba(0,0,0,0.05);margin-bottom:10px;">
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">{_logo_img(logo_tl1, "HG1", h=80)}{_logo_img(logo_tl2, "HG2", h=80)}</div>
    <h2 style="flex:1;text-align:center;margin:0;font-family:'Inter',sans-serif;font-size:1.6rem;font-weight:800;color:{config.ui.theme_color};line-height:1.2;">
        {config.ui.title}<br><span style="font-size:1.5rem;font-weight:500;color:#6b7280;">{config.ui.subtitle}</span>
    </h2>
    <div style="display:flex;gap:15px;flex-shrink:0;align-items:center;">{_logo_img(logo_tr1, "HD1", h=50)}{_logo_img(logo_tr2, "HD2", h=80)}</div>
</div>"""
