"""
app.py
======
Interface web com Gradio para identificação de otólitos.
Acesse em: http://localhost:7860

Funcionalidades:
  - Upload de amostra → exibe top-5 mais similares com scores
  - Aba para construir/reconstruir o banco de referências
  - Aba para executar fine-tuning via interface
  - Configurações: backbone, segmentação, realce de contraste
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.identifier import (
    build_reference_database,
    load_database,
    identify_from_pil,
)
from pipeline.preprocessor import preprocess_pil
from scripts.batch_identify import batch_identify
from scripts.evaluate import evaluate

# ── Paths padrão ─────────────────────────────────────────────────
DB_PATH         = "models/reference_db.json"
FINETUNED_PATH  = "models/finetuned_resnet50.pth"
IMAGES_DIR      = "data/referencias/imagens"
PDFS_DIR        = "data/referencias/pdfs"
SAMPLES_DIR     = "data/amostras"


# ════════════════════════════════════════════════════════════════════
# Estado global (carregado uma vez ao iniciar)
# ════════════════════════════════════════════════════════════════════

_extractor: FeatureExtractor | None = None
_db: ReferenceDatabase | None = None


def _get_extractor(backbone: str = "resnet50") -> FeatureExtractor:
    global _extractor
    if _extractor is None or _extractor.backbone_name != backbone:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _extractor = FeatureExtractor(backbone=backbone, device=device)
        # Carrega pesos fine-tuned se existirem
        if Path(FINETUNED_PATH).exists():
            _extractor.load_finetuned_weights(FINETUNED_PATH)
    return _extractor


def _get_db() -> ReferenceDatabase | None:
    global _db
    if _db is None and Path(DB_PATH).exists():
        _db = load_database(DB_PATH)
    return _db


# ════════════════════════════════════════════════════════════════════
# Funções dos botões da UI
# ════════════════════════════════════════════════════════════════════

def identify(
    sample_image: Image.Image,
    top_k: int,
    backbone: str,
    do_segment: bool,
    do_contrast: bool,
) -> tuple[Image.Image, str, plt.Figure]:
    """Callback principal: recebe imagem, retorna pré-processada + resultados."""

    if sample_image is None:
        return None, "⚠️  Faça upload de uma imagem de otólito.", None

    db = _get_db()
    if db is None:
        return None, "⚠️  Banco de referências não encontrado. Construa-o na aba 'Banco'.", None

    extractor = _get_extractor(backbone)

    # Pré-processa e exibe imagem resultante
    processed = preprocess_pil(sample_image, segment=do_segment, contrast=do_contrast)

    # Identifica
    results = identify_from_pil(
        sample_image, db, extractor,
        top_k=int(top_k),
        segment=do_segment,
        contrast=do_contrast,
    )

    # Texto de resultados
    lines = ["### Resultados\n"]
    for r in results:
        label_str = f" — *{r['label']}*" if r["label"] else ""
        lines.append(
            f"**{r['rank']}.** Score: `{r['score']:.4f}`{label_str}  \n"
            f"&nbsp;&nbsp;`{Path(r['path']).name}`"
        )
    result_text = "\n\n".join(lines)

    # Gráfico de barras com scores
    fig, ax = plt.subplots(figsize=(6, 2.5))
    names = [f"#{r['rank']} {Path(r['path']).stem[:20]}" for r in results]
    scores = [r["score"] for r in results]
    colors = ["#1D9E75" if i == 0 else "#9FE1CB" for i in range(len(scores))]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score de similaridade")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return processed, result_text, fig


def build_db(
    use_images: bool,
    use_pdfs: bool,
    backbone: str,
    do_segment: bool,
    do_contrast: bool,
) -> str:
    """Constrói o banco de referências."""
    global _db

    image_dirs = [IMAGES_DIR] if use_images and Path(IMAGES_DIR).exists() else None
    pdf_dirs   = [PDFS_DIR]   if use_pdfs   and Path(PDFS_DIR).exists()   else None

    if not image_dirs and not pdf_dirs:
        return (
            "⚠️  Nenhuma fonte encontrada.\n"
            f"  - Imagens: coloque em `{IMAGES_DIR}`\n"
            f"  - PDFs:    coloque em `{PDFS_DIR}`"
        )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    try:
        _db = build_reference_database(
            image_dirs=image_dirs,
            pdf_dirs=pdf_dirs,
            db_output=DB_PATH,
            segment=do_segment,
            contrast=do_contrast,
            backbone=backbone,
            device=device,
        )
        return (
            f"✅  Banco construído com {len(_db)} referências.\n"
            f"Salvo em: `{DB_PATH}`"
        )
    except Exception as exc:
        return f"❌  Erro: {exc}"


def run_finetune(
    backbone: str,
    epochs_head: int,
    epochs_full: int,
    batch_size: int,
    do_segment: bool,
    do_contrast: bool,
) -> str:
    """Executa fine-tuning via UI."""
    if not any(Path(IMAGES_DIR).rglob("*/*.jpg")) and \
       not any(Path(IMAGES_DIR).rglob("*/*.png")):
        return (
            f"⚠️  Nenhuma imagem rotulada encontrada em `{IMAGES_DIR}`.\n"
            "Organize em subpastas: `{IMAGES_DIR}/Nome_da_especie/foto.jpg`"
        )

    from training.finetune import finetune
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    output = f"models/finetuned_{backbone}.pth"

    try:
        finetune(
            data_dir=IMAGES_DIR,
            output_path=output,
            backbone=backbone,
            epochs_head=int(epochs_head),
            epochs_full=int(epochs_full),
            batch_size=int(batch_size),
            device_str=device,
            segment=do_segment,
            contrast=do_contrast,
        )
        global _extractor
        _extractor = None  # força reload com novos pesos
        return f"✅  Fine-tuning concluído! Pesos salvos em `{output}`"
    except Exception as exc:
        return f"❌  Erro no fine-tuning: {exc}"


# ════════════════════════════════════════════════════════════════════
# Interface Gradio
# ════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="Otolith ID", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🐟 Identificação de Espécies por Otólito\n"
            "Upload de uma foto de otólito → comparação com banco de referências"
        )

        # ── Configurações globais ─────────────────────────────────
        with gr.Accordion("⚙️  Configurações", open=False):
            with gr.Row():
                backbone_dd = gr.Dropdown(
                    ["resnet50", "efficientnet_b0", "efficientnet_b4"],
                    value="resnet50", label="Backbone CNN"
                )
                segment_cb = gr.Checkbox(value=True,  label="Segmentação GrabCut")
                contrast_cb = gr.Checkbox(value=True, label="Realce de contraste (CLAHE)")

        # ── Aba: Identificação ────────────────────────────────────
        with gr.Tab("🔍 Identificar amostra"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Foto do otólito")
                    top_k_sl  = gr.Slider(1, 10, value=5, step=1, label="Top-K resultados")
                    id_btn    = gr.Button("Identificar", variant="primary")

                with gr.Column(scale=1):
                    proc_img    = gr.Image(label="Imagem pré-processada", type="pil")
                    result_md   = gr.Markdown()
                    result_plot = gr.Plot(label="Scores de similaridade")

            id_btn.click(
                fn=identify,
                inputs=[input_img, top_k_sl, backbone_dd, segment_cb, contrast_cb],
                outputs=[proc_img, result_md, result_plot],
            )

        # ── Aba: Banco de referências ─────────────────────────────
        with gr.Tab("🗄️  Banco de referências"):
            gr.Markdown(
                "Construa o banco de referências a partir das suas imagens e PDFs.\n\n"
                f"- Imagens: `{IMAGES_DIR}/Nome_da_especie/foto.jpg`\n"
                f"- PDFs:    `{PDFS_DIR}/artigo.pdf`"
            )
            with gr.Row():
                use_imgs_cb = gr.Checkbox(value=True,  label="Usar imagens locais")
                use_pdfs_cb = gr.Checkbox(value=True,  label="Extrair imagens de PDFs")
            build_btn = gr.Button("Construir banco", variant="primary")
            build_out = gr.Textbox(label="Status", lines=3)
            build_btn.click(
                fn=build_db,
                inputs=[use_imgs_cb, use_pdfs_cb, backbone_dd, segment_cb, contrast_cb],
                outputs=build_out,
            )

        # ── Aba: Fine-tuning ──────────────────────────────────────
        with gr.Tab("🎯 Fine-tuning"):
            gr.Markdown(
                "Treina a CNN com suas imagens rotuladas para melhorar a acurácia.\n\n"
                f"Organize imagens em: `{IMAGES_DIR}/Nome_da_especie/foto.jpg`"
            )
            with gr.Row():
                ep_head = gr.Slider(1, 20, value=5,  step=1, label="Epochs (cabeça)")
                ep_full = gr.Slider(1, 30, value=10, step=1, label="Epochs (backbone)")
                bs      = gr.Slider(4, 64, value=16, step=4,  label="Batch size")
            ft_btn  = gr.Button("Iniciar fine-tuning", variant="primary")
            ft_out  = gr.Textbox(label="Status", lines=5)
            ft_btn.click(
                fn=run_finetune,
                inputs=[backbone_dd, ep_head, ep_full, bs, segment_cb, contrast_cb],
                outputs=ft_out,
            )

        # ── Aba: Batch ───────────────────────────────────────────
        with gr.Tab("📦 Processamento em lote"):
            gr.Markdown(
                "Identifica todas as fotos de uma pasta e exporta:\n"
                "`outputs/results.csv` · `outputs/report.html` · `outputs/results.json`"
            )
            with gr.Row():
                batch_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            batch_btn = gr.Button("Processar lote", variant="primary")
            batch_out = gr.Textbox(label="Status", lines=4)

            def run_batch(top_k, backbone, seg, contrast):
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if not Path(DB_PATH).exists():
                    return "⚠️  Construa o banco de referências primeiro."
                try:
                    batch_identify(
                        samples_dir=SAMPLES_DIR,
                        db_path=DB_PATH,
                        output_dir="outputs",
                        top_k=int(top_k),
                        backbone=backbone,
                        segment=seg,
                        contrast=contrast,
                        device=device,
                    )
                    csv_n = sum(1 for _ in Path("outputs/results.csv").read_text(
                        encoding="utf-8").splitlines()) - 1
                    return (
                        f"✅ Concluído — {csv_n} amostras processadas\n"
                        "Arquivos em outputs/:\n"
                        "  • results.csv\n  • report.html\n  • results.json"
                    )
                except Exception as exc:
                    return f"❌ Erro: {exc}"

            batch_btn.click(
                fn=run_batch,
                inputs=[batch_topk, backbone_dd, segment_cb, contrast_cb],
                outputs=batch_out,
            )

        # ── Aba: Avaliação ─────────────────────────────────────
        with gr.Tab("📊 Avaliação"):
            gr.Markdown(
                "Avalia a acurácia com imagens rotuladas (leave-one-out).\n"
                "Gera matriz de confusão e métricas por espécie em `outputs/evaluation/`."
            )
            with gr.Row():
                eval_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            eval_btn = gr.Button("Avaliar", variant="primary")
            eval_out = gr.Textbox(label="Resultado", lines=12)
            eval_plot = gr.Image(label="Matriz de Confusão", type="filepath")

            def run_evaluate(top_k, backbone, seg, contrast):
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if not any(Path(IMAGES_DIR).rglob("*/*")):
                    return ("⚠️  Nenhuma imagem rotulada encontrada.\n"
                            f"Organize em: {IMAGES_DIR}/Nome_especie/foto.jpg"), None
                try:
                    evaluate(
                        data_dir=IMAGES_DIR,
                        output_dir="outputs/evaluation",
                        top_k=int(top_k),
                        backbone=backbone,
                        segment=seg,
                        contrast=contrast,
                        device=device,
                    )
                    report = Path("outputs/evaluation/report.txt").read_text(encoding="utf-8")
                    matrix_path = "outputs/evaluation/confusion_matrix.png"
                    return report, matrix_path if Path(matrix_path).exists() else None
                except Exception as exc:
                    return f"❌ Erro: {exc}", None

            eval_btn.click(
                fn=run_evaluate,
                inputs=[eval_topk, backbone_dd, segment_cb, contrast_cb],
                outputs=[eval_out, eval_plot],
            )

        # ── Status do banco ao carregar ───────────────────────────
        db = _get_db()
        if db:
            gr.Markdown(f"✅ Banco carregado: **{len(db)} referências**")
        else:
            gr.Markdown("⚠️ Banco não encontrado — vá à aba **Banco de referências** para criá-lo.")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
    )
