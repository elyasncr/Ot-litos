"""
scripts/batch_identify.py
=========================
Processa uma pasta inteira de amostras e exporta os resultados
em CSV + relatório HTML com miniaturas das imagens.

Uso:
  python -m scripts.batch_identify \
      --samples_dir data/amostras \
      --db_path     models/reference_db.json \
      --output_dir  outputs \
      --top_k       5

Saída em outputs/:
  results.csv        → uma linha por amostra, top-K colunas
  report.html        → relatório visual com imagens lado a lado
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipeline.extractor import FeatureExtractor
from pipeline.identifier import load_database, identify_from_pil
from pipeline.preprocessor import preprocess_pil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════════
# CSV
# ════════════════════════════════════════════════════════════════════

def results_to_csv(all_results: list[dict], output_path: str):
    """
    Salva os resultados em CSV.

    Colunas:
      sample, top1_label, top1_score, top1_path,
              top2_label, top2_score, top2_path, ...
    """
    if not all_results:
        return

    top_k = max(len(r["matches"]) for r in all_results)
    fieldnames = ["sample", "processed_at"]
    for k in range(1, top_k + 1):
        fieldnames += [f"top{k}_label", f"top{k}_score", f"top{k}_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in all_results:
            row = {
                "sample":       entry["sample"],
                "processed_at": entry["processed_at"],
            }
            for i, match in enumerate(entry["matches"], 1):
                row[f"top{i}_label"] = match.get("label", "")
                row[f"top{i}_score"] = f"{match['score']:.6f}"
                row[f"top{i}_path"]  = match["path"]
            writer.writerow(row)

    print(f"[CSV] Salvo → '{output_path}'")


# ════════════════════════════════════════════════════════════════════
# Relatório HTML
# ════════════════════════════════════════════════════════════════════

def _img_to_b64(path: str, max_size: int = 200) -> str:
    """Converte imagem para base64 para embedar no HTML."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def results_to_html(all_results: list[dict], output_path: str):
    """
    Gera relatório HTML com:
     - miniatura da amostra
     - top-K referências mais similares lado a lado
     - barra de score colorida por confiança
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    cards_html = ""
    for entry in all_results:
        sample_b64 = _img_to_b64(entry["sample_path"])
        sample_img = (f'<img src="data:image/jpeg;base64,{sample_b64}" '
                      f'style="max-width:180px;max-height:180px;object-fit:contain">'
                      if sample_b64 else "<span style='color:#aaa'>sem preview</span>")

        match_cols = ""
        for m in entry["matches"]:
            score = m["score"]
            pct   = f"{score * 100:.1f}%"
            color = ("#1D9E75" if score > 0.80
                     else "#EF9F27" if score > 0.55
                     else "#E24B4A")
            ref_b64 = _img_to_b64(m["path"])
            ref_img = (f'<img src="data:image/jpeg;base64,{ref_b64}" '
                       f'style="max-width:140px;max-height:140px;object-fit:contain">'
                       if ref_b64 else "")
            label = m.get("label") or ""
            match_cols += f"""
            <div style="text-align:center;min-width:160px;max-width:160px;padding:6px">
              {ref_img}
              <div style="font-size:11px;color:#555;margin-top:4px;word-break:break-all">
                {Path(m['path']).name}
              </div>
              {f'<div style="font-size:11px;font-weight:500;color:#333">{label}</div>' if label else ''}
              <div style="background:#eee;border-radius:4px;height:8px;margin:4px 0">
                <div style="background:{color};width:{pct};height:8px;border-radius:4px"></div>
              </div>
              <div style="font-size:12px;color:{color};font-weight:600">{pct}</div>
            </div>"""

        cards_html += f"""
        <div style="border:1px solid #ddd;border-radius:10px;padding:14px;
                    margin-bottom:18px;display:flex;gap:16px;align-items:flex-start;
                    flex-wrap:wrap;background:#fafafa">
          <div style="text-align:center;min-width:190px">
            {sample_img}
            <div style="font-size:12px;font-weight:600;margin-top:6px;color:#333;word-break:break-all">
              {Path(entry['sample']).name}
            </div>
            <div style="font-size:10px;color:#999">{entry['processed_at']}</div>
          </div>
          <div style="flex:1;display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start">
            {match_cols}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Relatório Otólito CV — {ts}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1200px;
            margin: 0 auto; padding: 24px; color: #222; }}
    h1   {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
    .meta {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>🐟 Relatório — Identificação de Otólitos</h1>
  <p class="meta">Gerado em {ts} &nbsp;|&nbsp; {len(all_results)} amostras processadas</p>
  {cards_html}
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[HTML] Relatório → '{output_path}'")


# ════════════════════════════════════════════════════════════════════
# Pipeline de batch
# ════════════════════════════════════════════════════════════════════

def batch_identify(
    samples_dir: str,
    db_path: str = "models/reference_db.json",
    output_dir: str = "outputs",
    top_k: int = 5,
    backbone: str = "resnet50",
    segment: bool = True,
    contrast: bool = True,
    device: str = "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coleta amostras
    samples = [
        p for p in Path(samples_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not samples:
        print(f"[Batch] Nenhuma imagem encontrada em '{samples_dir}'")
        return

    print(f"[Batch] {len(samples)} amostras encontradas")

    db        = load_database(db_path)
    extractor = FeatureExtractor(backbone=backbone, device=device)

    all_results = []

    for sample_path in tqdm(samples, desc="Processando amostras"):
        try:
            pil_img = Image.open(sample_path).convert("RGB")
            matches = identify_from_pil(
                pil_img, db, extractor,
                top_k=top_k, segment=segment, contrast=contrast,
            )
            all_results.append({
                "sample":       str(sample_path),
                "sample_path":  str(sample_path),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "matches":      matches,
            })
        except Exception as exc:
            print(f"  [WARN] Pulando '{sample_path.name}': {exc}")

    # Exporta CSV
    results_to_csv(all_results, str(output_dir / "results.csv"))

    # Exporta HTML
    results_to_html(all_results, str(output_dir / "report.html"))

    # JSON bruto (útil para ingestão em outros sistemas)
    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[JSON] Salvo → '{json_path}'")
    print(f"\n✅ Concluído — {len(all_results)} amostras | outputs em '{output_dir}'")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser(description="Batch identification de otólitos")
    parser.add_argument("--samples_dir", default="data/amostras")
    parser.add_argument("--db_path",     default="models/reference_db.json")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--backbone",    default="resnet50")
    parser.add_argument("--no_segment",  action="store_true")
    parser.add_argument("--no_contrast", action="store_true")
    args = parser.parse_args()

    batch_identify(
        samples_dir=args.samples_dir,
        db_path=args.db_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        backbone=args.backbone,
        segment=not args.no_segment,
        contrast=not args.no_contrast,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
