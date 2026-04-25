"""
pipeline/identifier.py
======================
Orquestra o pipeline completo:
  PDF → imagens → pré-processamento → features → banco → resultados
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.database import ReferenceDatabase
from pipeline.extractor import FeatureExtractor
from pipeline.pdf_extractor import extract_all_pdfs
from pipeline.preprocessor import preprocess, preprocess_pil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════════
# Construção do banco de referências
# ════════════════════════════════════════════════════════════════════

def build_reference_database(
    image_dirs: list[str] | None = None,
    pdf_dirs: list[str] | None = None,
    db_output: str = "models/reference_db.json",
    segment: bool = True,
    contrast: bool = True,
    backbone: str = "resnet50",
    device: str = "cpu",
) -> ReferenceDatabase:
    """
    Constrói o banco de referências a partir de imagens e/ou PDFs.

    Fluxo:
      1. Extrai imagens de PDFs (se fornecidos)
      2. Coleta imagens de pastas (se fornecidas)
      3. Pré-processa cada imagem
      4. Extrai embeddings com CNN
      5. Cria índice FAISS/sklearn
      6. Salva em db_output

    Parâmetros opcionais de rotulagem:
      Se a pasta de imagens seguir a convenção nome_da_especie/imagem.jpg,
      o label é inferido automaticamente do nome da pasta.
    """
    extractor = FeatureExtractor(backbone=backbone, device=device)
    all_image_paths: list[str] = []
    all_labels: list[str] = []

    # ── Extrai imagens de PDFs ─────────────────────────────────────
    if pdf_dirs:
        for pdf_dir in pdf_dirs:
            imgs = extract_all_pdfs(pdf_dir, output_base_dir="_pdf_images")
            all_image_paths.extend(imgs)
            all_labels.extend([""] * len(imgs))  # sem label para PDFs

    # ── Coleta imagens de pastas ───────────────────────────────────
    if image_dirs:
        for folder in image_dirs:
            folder = Path(folder)
            for p in folder.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    all_image_paths.append(str(p))
                    # infere label do nome da pasta pai
                    label = p.parent.name if p.parent != folder else ""
                    all_labels.append(label)

    if not all_image_paths:
        raise RuntimeError(
            "Nenhuma imagem encontrada. Verifique image_dirs e pdf_dirs."
        )

    print(f"\n[Pipeline] {len(all_image_paths)} imagens para indexar")

    valid_paths, embeddings = extractor.extract_batch(
        all_image_paths, segment=segment, contrast=contrast
    )

    # Mantém apenas labels das imagens que foram processadas com sucesso
    path_set = set(valid_paths)
    valid_labels = [
        lbl for path, lbl in zip(all_image_paths, all_labels)
        if path in path_set
    ]

    db = ReferenceDatabase()
    db.build(valid_paths, embeddings, labels=valid_labels)
    Path(db_output).parent.mkdir(parents=True, exist_ok=True)
    db.save(db_output)
    return db


def load_database(db_path: str = "models/reference_db.json") -> ReferenceDatabase:
    db = ReferenceDatabase()
    db.load(db_path)
    return db


# ════════════════════════════════════════════════════════════════════
# Identificação de amostra
# ════════════════════════════════════════════════════════════════════

def identify_from_path(
    sample_path: str,
    db: ReferenceDatabase,
    extractor: FeatureExtractor,
    top_k: int = 5,
    segment: bool = True,
    contrast: bool = True,
) -> list[dict]:
    """Identifica espécie a partir de um caminho de arquivo."""
    img = preprocess(sample_path, segment=segment, contrast=contrast)
    vec = extractor.extract(img)
    results = db.search(vec, top_k=top_k)
    _print_results(sample_path, results)
    return results


def identify_from_pil(
    pil_image: Image.Image,
    db: ReferenceDatabase,
    extractor: FeatureExtractor,
    top_k: int = 5,
    segment: bool = True,
    contrast: bool = True,
) -> list[dict]:
    """Identifica espécie a partir de uma imagem PIL (usada pela UI)."""
    img = preprocess_pil(pil_image, segment=segment, contrast=contrast)
    vec = extractor.extract(img)
    results = db.search(vec, top_k=top_k)
    return results


# ── Helpers ──────────────────────────────────────────────────────

def _print_results(source: str, results: list[dict]):
    print(f"\n── Resultados para: {source} ──")
    for r in results:
        label = f"  [{r['label']}]" if r["label"] else ""
        print(f"  {r['rank']}. Score: {r['score']:.4f}{label}  →  {r['path']}")
