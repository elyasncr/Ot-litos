"""
pipeline/extractor.py
=====================
Extração de embeddings visuais usando CNN pré-treinada.

Modelos disponíveis (todos gratuitos, baixados automaticamente):
  - resnet50    → 2048 dims, boa acurácia geral
  - efficientnet_b0 → 1280 dims, mais leve
  - efficientnet_b4 → 1792 dims, mais preciso

Troca o BACKBONE_NAME abaixo ou passe como parâmetro.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from tqdm import tqdm

from pipeline.preprocessor import preprocess, preprocess_pil


BACKBONE_NAME = "resnet50"   # ou "efficientnet_b0", "efficientnet_b4"


def _build_backbone(name: str) -> tuple[nn.Module, int]:
    """Carrega o backbone sem cabeça classificadora e retorna (modelo, dims)."""

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_dim = m.fc.in_features          # 2048
        m.fc = nn.Identity()
        return m, feature_dim

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feature_dim = m.classifier[1].in_features   # 1280
        m.classifier = nn.Identity()
        return m, feature_dim

    if name == "efficientnet_b4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        feature_dim = m.classifier[1].in_features   # 1792
        m.classifier = nn.Identity()
        return m, feature_dim

    raise ValueError(f"Backbone desconhecido: '{name}'. "
                     "Opções: resnet50, efficientnet_b0, efficientnet_b4")


class FeatureExtractor:
    """
    Extrai um vetor de embedding para cada imagem usando uma CNN
    pré-treinada no ImageNet (sem fine-tuning).

    Para usar pesos de fine-tuning, chame load_finetuned_weights().
    """

    def __init__(self, backbone: str = BACKBONE_NAME, device: str = "cpu"):
        self.device = torch.device(device)
        self.backbone_name = backbone
        self.model, self.feature_dim = _build_backbone(backbone)
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def load_finetuned_weights(self, weights_path: str):
        """Carrega pesos salvos após fine-tuning (ver training/finetune.py)."""
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        print(f"[Extractor] Pesos fine-tuned carregados de '{weights_path}'")

    @torch.no_grad()
    def extract(self, pil_image: Image.Image) -> np.ndarray:
        """Retorna vetor numpy (feature_dim,) para uma imagem PIL."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        vec = self.model(tensor)
        return vec.squeeze().cpu().numpy()

    @torch.no_grad()
    def extract_from_path(self, image_path: str,
                          segment: bool = True,
                          contrast: bool = True) -> np.ndarray:
        """Pré-processa e extrai features a partir de um caminho de arquivo."""
        img = preprocess(image_path, segment=segment, contrast=contrast)
        return self.extract(img)

    def extract_batch(
        self,
        image_paths: list[str],
        segment: bool = True,
        contrast: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        """
        Processa uma lista de imagens em lote.
        Retorna (caminhos_válidos, matriz_embeddings [N, feature_dim]).
        Imagens com erro são ignoradas com aviso.
        """
        valid_paths, embeddings = [], []

        for path in tqdm(image_paths, desc="Extraindo features"):
            try:
                vec = self.extract_from_path(path, segment=segment,
                                             contrast=contrast)
                valid_paths.append(path)
                embeddings.append(vec)
            except Exception as exc:
                print(f"  [WARN] Pulando '{path}': {exc}")

        if not embeddings:
            raise RuntimeError("Nenhuma imagem pôde ser processada.")

        return valid_paths, np.array(embeddings, dtype="float32")
