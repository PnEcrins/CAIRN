import gradio as gr
import pandas as pd
from jinja2 import Template
from pathlib import Path
from src.config import config
from src.constant import generate_theme, ABS_DIR, get_html_header
from src.layout import get_layout
from src.utils import (
    _get_logo_path,
    to_b64,
)

custom_theme = generate_theme(config)

LOGO_TL1 = to_b64(_get_logo_path("top_left_1"))
LOGO_TL2 = to_b64(_get_logo_path("top_left_2"))
LOGO_TR1 = to_b64(_get_logo_path("top_right_1"))
LOGO_TR2 = to_b64(_get_logo_path("top_right_2"))


HEADER_HTML = get_html_header(config, LOGO_TL1, LOGO_TL2, LOGO_TR1, LOGO_TR2)
CUSTOM_CSS = Template(Path(ABS_DIR, "style.css").read_text()).render(config=config)

# ── Interface Gradio ───────────────────────────────────────────────────────────
with gr.Blocks(title=config.ui.title) as demo:
    get_layout(config, HEADER_HTML, demo)


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=custom_theme, css=CUSTOM_CSS, server_name="0.0.0.0")
