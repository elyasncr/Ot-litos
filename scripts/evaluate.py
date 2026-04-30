"""
scripts/evaluate.py
===================
Avalia a acurácia do pipeline de identificação com imagens rotuladas.

Requer imagens organizadas por espécie (mesmo padrão do fine-tuning):
  data/referencias/imagens/
    ├── Micropogonias_furnieri/
    └── Mugil_liza/

Estratégia: leave-one-out por espécie
  Para cada imagem, retira-a do banco, consulta as restantes como
  referência, e verifica se a espécie correta aparece no top-K.

Saída:
  outputs/evaluation/confusion_matrix.png
  outputs/evaluation/metrics.csv
  outputs/evaluation/report.txt

Uso:
  python -m scripts.evaluate \
      --data_dir  data/referencias/imagens \
      --output    outputs/evaluation \
      --top_k     5 \
      --backbone  resnet50
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.preprocessor import preprocess_pil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


# ════════════════════════════════════════════════════════════════════
# Coleta de dados
# ════════════════════════════════════════════════════════════════════

def collect_samples(data_dir: str) -> tuple[list[str], list[str], list[str]]:
    """Retorna (paths, labels, classes) para todas as imagens rotuladas."""
    root = Path(data_dir)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())

    paths, labels = [], []
    for cls in classes:
        for p in (root / cls).iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(p))
                labels.append(cls)

    return paths, labels, classes


# ════════════════════════════════════════════════════════════════════
# Avaliação leave-one-out
# ════════════════════════════════════════════════════════════════════

def evaluate(
    data_dir: str,
    output_dir: str = "outputs/evaluation",
    top_k: int = 5,
    backbone: str = "resnet50",
    segment: bool = True,
    contrast: bool = True,
    device: str = "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, classes = collect_samples(data_dir)
    n = len(paths)
    if n < 2:
        print("❌  Mínimo de 2 imagens rotuladas necessário para avaliação.")
        return

    print(f"[Eval] {n} imagens | {len(classes)} espécies | top-K={top_k}")

    extractor = FeatureExtractor(backbone=backbone, device=device)

    # Pré-extrai todos os embeddings uma vez (eficiente)
    print("[Eval] Extraindo embeddings...")
    all_embeddings = []
    valid_mask = []
    for path in tqdm(paths, desc="  features"):
        try:
            img = Image.open(path).convert("RGB")
            img = preprocess_pil(img, segment=segment, contrast=contrast)
            vec = extractor.extract(img)
            all_embeddings.append(vec)
            valid_mask.append(True)
        except Exception as exc:
            print(f"  [WARN] {Path(path).name}: {exc}")
            all_embeddings.append(None)
            valid_mask.append(False)

    valid_paths  = [p for p, v in zip(paths,  valid_mask) if v]
    valid_labels = [l for l, v in zip(labels, valid_mask) if v]
    valid_embs   = np.array([e for e, v in zip(all_embeddings, valid_mask) if v],
                            dtype="float32")

    n_valid = len(valid_paths)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Matrizes de resultados
    conf_matrix   = np.zeros((len(classes), len(classes)), dtype=int)
    top1_correct  = 0
    topk_correct  = 0
    per_class_top1 = defaultdict(lambda: {"correct": 0, "total": 0})

    print("[Eval] Avaliando (leave-one-out)...")
    for i in tqdm(range(n_valid), desc="  queries"):
        query_emb   = valid_embs[i]
        query_label = valid_labels[i]

        # Monta banco sem a imagem atual
        ref_embs   = np.concatenate([valid_embs[:i], valid_embs[i+1:]], axis=0)
        ref_labels = valid_labels[:i] + valid_labels[i+1:]

        db = ReferenceDatabase()
        db.build(valid_paths[:i] + valid_paths[i+1:], ref_embs, ref_labels)
        results = db.search(query_emb, top_k=top_k)

        top1_pred = results[0]["label"] if results else ""
        topk_preds = [r["label"] for r in results]

        # Top-1
        if top1_pred == query_label:
            top1_correct += 1

        # Top-K
        if query_label in topk_preds:
            topk_correct += 1

        # Confusion matrix (top-1)
        true_idx = cls_to_idx.get(query_label, -1)
        pred_idx = cls_to_idx.get(top1_pred, -1)
        if true_idx >= 0 and pred_idx >= 0:
            conf_matrix[true_idx][pred_idx] += 1

        # Por classe
        per_class_top1[query_label]["total"] += 1
        if top1_pred == query_label:
            per_class_top1[query_label]["correct"] += 1

    top1_acc = top1_correct / n_valid
    topk_acc = topk_correct / n_valid

    print(f"\n── Resultados ──────────────────────")
    print(f"  Top-1 acurácia: {top1_acc:.3f} ({top1_correct}/{n_valid})")
    print(f"  Top-{top_k} acurácia: {topk_acc:.3f} ({topk_correct}/{n_valid})")
    print(f"  Imagens avaliadas: {n_valid}/{n}")

    # ── Confusion Matrix ─────────────────────────────────────────
    _plot_confusion_matrix(
        conf_matrix, classes,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"Matriz de Confusão — Top-1 (acurácia: {top1_acc:.2%})",
    )

    # ── Métricas por classe (CSV) ────────────────────────────────
    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["especie", "total", "corretas", "acuracia"])
        writer.writeheader()
        for cls in classes:
            d = per_class_top1[cls]
            acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
            writer.writerow({
                "especie":   cls,
                "total":     d["total"],
                "corretas":  d["correct"],
                "acuracia":  f"{acc:.4f}",
            })
    print(f"[Eval] Métricas por classe → '{metrics_path}'")

    # ── Relatório texto ──────────────────────────────────────────
    report_lines = [
        "=== Relatório de Avaliação — Otolith CV ===\n",
        f"Data:          {__import__('datetime').datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"Backbone:      {backbone}",
        f"Segmentação:   {segment}",
        f"Contraste:     {contrast}",
        f"Imagens:       {n_valid}/{n} válidas",
        f"Espécies:      {len(classes)}",
        f"Top-1 acurácia: {top1_acc:.4f} ({top1_correct}/{n_valid})",
        f"Top-{top_k} acurácia: {topk_acc:.4f} ({topk_correct}/{n_valid})",
        "\n--- Por espécie (Top-1) ---",
    ]
    for cls in classes:
        d = per_class_top1[cls]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        report_lines.append(f"  {cls:<40} {d['correct']:>3}/{d['total']:<3}  {acc:.2%}")

    report_text = "\n".join(report_lines)
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[Eval] Relatório texto → '{report_path}'")
    print(f"\n{report_text}")


# ════════════════════════════════════════════════════════════════════
# Plot confusion matrix
# ════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str,
    title: str = "Matriz de Confusão",
):
    n = len(class_names)
    fig_size = max(6, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Normaliza por linha (recall por classe)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, where=row_sums != 0).astype(float)

    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c[:20] for c in class_names]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predito", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)

    # Anota células com valor absoluto
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Matriz de confusão → '{save_path}'")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser(description="Avaliação do pipeline de otólitos")
    parser.add_argument("--data_dir",    default="data/referencias/imagens")
    parser.add_argument("--output",      default="outputs/evaluation")
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--backbone",    default="resnet50")
    parser.add_argument("--no_segment",  action="store_true")
    parser.add_argument("--no_contrast", action="store_true")
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        output_dir=args.output,
        top_k=args.top_k,
        backbone=args.backbone,
        segment=not args.no_segment,
        contrast=not args.no_contrast,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
