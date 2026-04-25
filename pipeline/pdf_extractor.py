"""
pipeline/pdf_extractor.py
=========================
Extrai imagens embutidas em PDFs de artigos e livros científicos.
Usa PyMuPDF (fitz) — 100% local, sem API, sem custo.
"""

from __future__ import annotations

from pathlib import Path

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str,
    min_width: int = 100,
    min_height: int = 100,
) -> list[str]:
    """
    Extrai todas as imagens de um PDF e salva em output_dir.

    Parâmetros
    ----------
    pdf_path   : caminho para o arquivo PDF
    output_dir : pasta onde as imagens serão salvas
    min_width  : ignora imagens menores que este valor (px) — evita ícones
    min_height : idem para altura

    Retorna lista com os caminhos das imagens salvas.
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF não instalado. Execute: pip install pymupdf"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    saved: list[str] = []

    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc):
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue

            w, h = base.get("width", 0), base.get("height", 0)
            if w < min_width or h < min_height:
                continue  # descarta miniaturas e ícones

            ext = base.get("ext", "png")
            filename = f"{pdf_stem}_p{page_num:03d}_i{img_idx:02d}.{ext}"
            out_path = output_dir / filename
            out_path.write_bytes(base["image"])
            saved.append(str(out_path))

    doc.close()
    print(f"[PDF] '{Path(pdf_path).name}' → {len(saved)} imagens extraídas")
    return saved


def extract_all_pdfs(
    pdf_dir: str,
    output_base_dir: str = "_pdf_images",
    min_width: int = 100,
    min_height: int = 100,
) -> list[str]:
    """
    Processa todos os PDFs dentro de pdf_dir.
    Salva imagens de cada PDF em uma subpasta separada.
    Retorna lista consolidada de todos os caminhos extraídos.
    """
    pdf_dir = Path(pdf_dir)
    all_images: list[str] = []

    pdfs = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    if not pdfs:
        print(f"[PDF] Nenhum PDF encontrado em '{pdf_dir}'")
        return all_images

    for pdf_path in pdfs:
        out = Path(output_base_dir) / pdf_path.stem
        imgs = extract_images_from_pdf(
            str(pdf_path), str(out),
            min_width=min_width, min_height=min_height
        )
        all_images.extend(imgs)

    print(f"[PDF] Total: {len(all_images)} imagens extraídas de {len(pdfs)} PDFs")
    return all_images
