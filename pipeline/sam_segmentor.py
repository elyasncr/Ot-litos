"""
pipeline/sam_segmentor.py
=========================
Segmentação avançada com SAM — Segment Anything Model (Meta AI).

Grátis e open-source. Precisão muito superior ao GrabCut para fotos
com fundo complexo (sedimento, múltiplos otólitos, água, escamas).

════════════════════════════════════════════════════════════════════
INSTALAÇÃO (rode no terminal dentro do container):
════════════════════════════════════════════════════════════════════

  pip install segment-anything

  # Checkpoint vit_b (~375 MB) — mais leve, boa para CPU:
  wget -O models/sam_vit_b.pth \\
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

  # Checkpoint vit_h (~2.5 GB) — máxima precisão, recomenda GPU:
  wget -O models/sam_vit_h.pth \\
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

════════════════════════════════════════════════════════════════════
USO:
════════════════════════════════════════════════════════════════════

  from pipeline.sam_segmentor import SAMSegmentor
  seg = SAMSegmentor("models/sam_vit_b.pth", model_type="vit_b")

  pil_segmented = seg.segment(pil_image)
  # → retorna a imagem com fundo removido (preto)

  # Ou via pipeline de pré-processamento:
  from pipeline.preprocessor import preprocess_pil
  result = preprocess_pil(pil_image, segment=True, sam_segmentor=seg)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMSegmentor:
    """
    Wrapper sobre o Segment Anything Model da Meta.

    Estratégia de seleção da máscara:
      1. Gera todas as máscaras automáticas da imagem
      2. Filtra máscaras que tocam as bordas (provavelmente fundo)
      3. Seleciona a máscara com maior área que esteja próxima
         ao centro da imagem (assume otólito centralizado)
      4. Aplica a máscara: fundo → preto
    """

    def __init__(
        self,
        checkpoint: str = "models/sam_vit_b.pth",
        model_type: str = "vit_b",     # vit_b | vit_l | vit_h
        device: str = "cpu",
        points_per_side: int = 32,     # reduz para ~16 se CPU for lenta
    ):
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM não instalado.\n"
                "Execute: pip install segment-anything\n"
                "Depois baixe o checkpoint (veja docstring do módulo)."
            )

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: '{checkpoint}'\n"
                "Baixe com: wget -O models/sam_vit_b.pth "
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)

        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,   # ignora regiões minúsculas
        )
        print(f"[SAM] Modelo '{model_type}' carregado de '{checkpoint}'")

    def segment(self, pil_image: Image.Image) -> Image.Image:
        """
        Segmenta o otólito principal e retorna a imagem com fundo preto.
        Se nenhuma máscara adequada for encontrada, retorna a original.
        """
        img_rgb = np.array(pil_image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        masks = self.generator.generate(img_rgb)

        if not masks:
            return pil_image

        best_mask = self._select_best_mask(masks, h, w)
        if best_mask is None:
            return pil_image

        # Aplica máscara
        m = best_mask["segmentation"].astype(np.uint8)

        # Suaviza bordas da máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        segmented = img_rgb * m[:, :, np.newaxis]
        return Image.fromarray(segmented)

    def _select_best_mask(
        self, masks: list[dict], h: int, w: int
    ) -> dict | None:
        """
        Seleciona a melhor máscara para o otólito:
          - descarta máscaras que tocam as 4 bordas (provavelmente fundo)
          - prefere máscaras próximas ao centro
          - entre as candidatas centrais, escolhe a maior área
        """
        cx, cy = w / 2, h / 2
        border_margin = 0.05   # 5% das bordas

        candidates = []
        for m in masks:
            seg = m["segmentation"]
            # Descarta se toca todas as 4 bordas (fundo envolvente)
            touches_all = (
                seg[0, :].any() and seg[-1, :].any() and
                seg[:, 0].any() and seg[:, -1].any()
            )
            if touches_all:
                continue

            # Centróide da máscara
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            mask_cx, mask_cy = xs.mean(), ys.mean()

            # Distância ao centro da imagem (normalizada)
            dist = np.sqrt(((mask_cx - cx) / w) ** 2 + ((mask_cy - cy) / h) ** 2)

            candidates.append({
                "mask":  m,
                "area":  m["area"],
                "dist":  dist,
                "score": m.get("predicted_iou", 0),
            })

        if not candidates:
            return None

        # Ordena: prioriza candidatos centrais (dist < 0.35) por área,
        # depois os mais externos por score IoU
        central = [c for c in candidates if c["dist"] < 0.35]
        if central:
            return max(central, key=lambda c: c["area"])["mask"]

        return max(candidates, key=lambda c: c["score"])["mask"]


# ════════════════════════════════════════════════════════════════════
# Atualização do preprocessor para suportar SAM
# ════════════════════════════════════════════════════════════════════
#
# Para usar SAM em vez de GrabCut, inicialize o SAMSegmentor e passe
# no pré-processamento:
#
#   from pipeline.sam_segmentor import SAMSegmentor
#   from pipeline.preprocessor import denoise, enhance_contrast
#
#   sam = SAMSegmentor("models/sam_vit_b.pth")
#
#   def preprocess_with_sam(pil_image):
#       import numpy as np, cv2
#       img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
#       img_bgr = denoise(img_bgr)
#       img_bgr = enhance_contrast(img_bgr)
#       img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#       pil_denoised = Image.fromarray(img_rgb)
#       return sam.segment(pil_denoised)   # SAM recebe PIL, devolve PIL
