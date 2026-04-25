"""
pipeline/preprocessor.py
========================
Pré-processamento e segmentação do otólito na imagem.

Etapas:
  1. Leitura e normalização de tamanho
  2. Remoção de ruído (Gaussian Blur)
  3. Segmentação: isola o otólito do fundo usando GrabCut (OpenCV)
     — sem necessidade de modelo externo, sem custo de API
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ── Configurações ──────────────────────────────────────────────────
TARGET_SIZE = (224, 224)   # tamanho de entrada da CNN
BLUR_KERNEL = (3, 3)


# ══════════════════════════════════════════════════════════════════
# Funções públicas
# ══════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> np.ndarray:
    """Lê a imagem em BGR (formato padrão do OpenCV)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    return img


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Remove ruído leve com filtro Gaussiano."""
    return cv2.GaussianBlur(img_bgr, BLUR_KERNEL, 0)


def segment_otolith(img_bgr: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """
    Segmenta o otólito usando GrabCut.

    O GrabCut é um algoritmo iterativo clássico do OpenCV que separa
    o objeto do fundo sem necessitar de nenhuma API ou modelo pago.

    Estratégia: assume que o otólito ocupa a região central da foto
    (retângulo com margem configurável). Funciona bem para fotos de
    laboratório com fundo uniforme. Para fundos complexos, recomenda-se
    fornecer uma máscara manual ou usar SAM (veja comentário no final).

    Retorna a imagem original com o fundo zerado (pixels fora do
    otólito → preto), facilitando a extração de features.
    """
    h, w = img_bgr.shape[:2]
    mw = int(w * margin)
    mh = int(h * margin)
    rect = (mw, mh, w - 2 * mw, h - 2 * mh)  # (x, y, largura, altura)

    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bg_model, fg_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Pixels 2 (possível BG) e 3 (possível FG) → tratamos como FG
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

    # Aplica máscara: fundo → preto
    segmented = img_bgr * fg_mask[:, :, np.newaxis]
    return segmented


def enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Melhora o contraste via CLAHE no canal L do espaço LAB.
    Útil para otólitos fotografados com iluminação irregular.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess(
    image_path: str,
    segment: bool = True,
    contrast: bool = True,
) -> Image.Image:
    """
    Pipeline completo de pré-processamento.

    Parâmetros
    ----------
    image_path : caminho para a imagem (jpg/png/tif)
    segment    : ativa a segmentação GrabCut para isolar o otólito
    contrast   : ativa o realce de contraste CLAHE

    Retorna objeto PIL pronto para a CNN.
    """
    img = load_image(image_path)
    img = denoise(img)

    if contrast:
        img = enhance_contrast(img)

    if segment:
        img = segment_otolith(img)

    # BGR → RGB → PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def preprocess_pil(
    pil_image: Image.Image,
    segment: bool = True,
    contrast: bool = True,
) -> Image.Image:
    """
    Mesma lógica, mas aceita objeto PIL como entrada.
    Útil para a UI (Gradio entrega PIL diretamente).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_bgr = denoise(img_bgr)

    if contrast:
        img_bgr = enhance_contrast(img_bgr)

    if segment:
        img_bgr = segment_otolith(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ── Nota sobre SAM (Segment Anything Model — Meta) ─────────────────
#
# Para fotos com fundo complexo (água, sedimento, múltiplos otólitos),
# o SAM oferece segmentação muito mais precisa. É gratuito e open-source.
#
# Instalação:
#   pip install segment-anything
#   # Download do checkpoint (~375 MB):
#   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#
# Uso básico:
#   from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
#   sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
#   generator = SamAutomaticMaskGenerator(sam)
#   masks = generator.generate(np.array(pil_image))
#   # Selecionar a máscara central (assumindo otólito no centro)
#   best_mask = max(masks, key=lambda m: m["area"] if _is_central(m) else 0)
