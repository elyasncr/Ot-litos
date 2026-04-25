"""
training/finetune.py
====================
Fine-tuning da CNN pré-treinada em imagens de otólitos rotuladas.

Estratégia: Transfer Learning em 2 fases
  Fase 1 → Congela o backbone, treina só a nova cabeça (5 epochs)
  Fase 2 → Descongela as últimas camadas, treina tudo com LR menor (10 epochs)

Uso:
  python -m training.finetune \
    --data_dir data/referencias/imagens \
    --backbone resnet50 \
    --epochs_head 5 \
    --epochs_full 10 \
    --output models/finetuned_resnet50.pth

Requer: imagens organizadas em subpastas por espécie (ver dataset.py)
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from training.dataset import OtolithDataset, get_val_transform


def build_classifier(backbone_name: str, num_classes: int) -> nn.Module:
    """Constrói modelo com cabeça de classificação para fine-tuning."""

    if backbone_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return m

    if backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return m

    if backbone_name == "efficientnet_b4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return m

    raise ValueError(f"Backbone desconhecido: {backbone_name}")


def freeze_backbone(model: nn.Module, backbone_name: str):
    """Congela todos os parâmetros exceto a cabeça classificadora."""
    head_name = "fc" if "resnet" in backbone_name else "classifier"
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_name)


def unfreeze_last_layers(model: nn.Module, backbone_name: str, n_blocks: int = 2):
    """Descongela os últimos n_blocks do backbone para fine-tuning completo."""
    # Coleta todos os parâmetros em ordem
    all_params = list(model.named_parameters())
    # Descongela os últimos n blocos (aprox. últimos 20% dos parâmetros)
    threshold = int(len(all_params) * 0.8)
    for i, (name, param) in enumerate(all_params):
        param.requires_grad = i >= threshold


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Treina por 1 epoch. Retorna (loss_médio, acurácia)."""
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Avalia o modelo. Retorna (loss_médio, acurácia)."""
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def finetune(
    data_dir: str,
    output_path: str,
    backbone: str = "resnet50",
    epochs_head: int = 5,
    epochs_full: int = 10,
    batch_size: int = 16,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    val_split: float = 0.2,
    device_str: str = "cpu",
    segment: bool = True,
    contrast: bool = True,
):
    device = torch.device(device_str)
    print(f"\n[Fine-tune] Device: {device} | Backbone: {backbone}")

    # ── Dataset ──────────────────────────────────────────────────
    full_dataset = OtolithDataset(
        data_dir, segment=segment, contrast=contrast
    )
    num_classes = len(full_dataset.classes)

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Validação sem augmentation
    val_ds.dataset = copy.deepcopy(val_ds.dataset)
    val_ds.dataset.transform = get_val_transform()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"[Fine-tune] Treino: {n_train} | Validação: {n_val} | Classes: {num_classes}")

    # ── Modelo ────────────────────────────────────────────────────
    model = build_classifier(backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    best_state = None

    # ══ FASE 1: treina só a cabeça ════════════════════════════════
    print(f"\n── Fase 1: cabeça classificadora ({epochs_head} epochs) ──")
    freeze_backbone(model, backbone)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_head)

    for epoch in range(1, epochs_head + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{epochs_head} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"{time.time()-t0:.1f}s")

    # ══ FASE 2: descongelamento parcial ══════════════════════════
    print(f"\n── Fase 2: backbone parcial ({epochs_full} epochs) ──")
    unfreeze_last_layers(model, backbone)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_full
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_full)

    for epoch in range(1, epochs_full + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{epochs_full} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"{time.time()-t0:.1f}s")

    # ── Salva melhor modelo ───────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"\n[Fine-tune] Melhor val_acc={best_acc:.3f} → salvo em '{output_path}'")

    # Salva mapeamento de classes
    classes_path = Path(output_path).with_suffix(".classes.json")
    import json
    classes_path.write_text(json.dumps(full_dataset.classes, ensure_ascii=False))
    print(f"[Fine-tune] Classes salvas → '{classes_path}'")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning para otólitos")
    parser.add_argument("--data_dir",     default="data/referencias/imagens")
    parser.add_argument("--output",       default="models/finetuned_resnet50.pth")
    parser.add_argument("--backbone",     default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "efficientnet_b4"])
    parser.add_argument("--epochs_head",  type=int, default=5)
    parser.add_argument("--epochs_full",  type=int, default=10)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--no_segment",   action="store_true")
    parser.add_argument("--no_contrast",  action="store_true")
    args = parser.parse_args()

    finetune(
        data_dir=args.data_dir,
        output_path=args.output,
        backbone=args.backbone,
        epochs_head=args.epochs_head,
        epochs_full=args.epochs_full,
        batch_size=args.batch_size,
        device_str=args.device,
        segment=not args.no_segment,
        contrast=not args.no_contrast,
    )
