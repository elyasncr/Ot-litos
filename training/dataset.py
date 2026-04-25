"""
training/dataset.py
===================
Dataset PyTorch para fine-tuning.

Convenção de pastas esperada:
  data/referencias/imagens/
    ├── Micropogonias_furnieri/
    │   ├── foto_001.jpg
    │   └── foto_002.jpg
    ├── Mugil_liza/
    │   └── foto_001.jpg
    └── ...

Cada subpasta é uma classe (espécie). O nome da pasta vira o label.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from pipeline.preprocessor import preprocess_pil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class OtolithDataset(Dataset):
    """
    Carrega imagens organizadas por espécie em subpastas.

    Parâmetros
    ----------
    root_dir  : pasta raiz com subpastas por espécie
    transform : transformações adicionais (augmentation)
    segment   : aplica segmentação GrabCut
    contrast  : aplica realce CLAHE
    """

    def __init__(
        self,
        root_dir: str,
        transform: T.Compose | None = None,
        segment: bool = True,
        contrast: bool = True,
    ):
        self.root = Path(root_dir)
        self.segment = segment
        self.contrast = contrast

        # Descobre classes e amostras
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(
                f"Nenhuma imagem encontrada em '{root_dir}'. "
                "Organize as imagens em subpastas por espécie."
            )

        # Augmentation padrão para treino
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"[Dataset] {len(self.classes)} espécies, {len(self.samples)} imagens")
        for cls in self.classes:
            n = sum(1 for _, idx in self.samples if idx == self.class_to_idx[cls])
            print(f"  {cls}: {n} imagens")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        pil_img = Image.open(path).convert("RGB")

        # Pré-processamento especializado (segmentação + contraste)
        pil_img = preprocess_pil(pil_img,
                                 segment=self.segment,
                                 contrast=self.contrast)
        tensor = self.transform(pil_img)
        return tensor, label


def get_val_transform() -> T.Compose:
    """Transformação de validação (sem augmentation)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
