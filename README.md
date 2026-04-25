# 🐟 Otolith CV — Identificação de Espécies por Otólito

Sistema de visão computacional para comparar fotos de otólitos com imagens
de referência extraídas de artigos científicos e livros.

**100% local e gratuito — não usa nenhuma API paga.**

---

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- Sem GPU necessária (funciona na CPU; GPU acelera o fine-tuning)

---

## Início rápido

```bash
# 1. Clone / descompacte o projeto
cd otolith_cv

# 2. Coloque seus dados nas pastas corretas (ver abaixo)

# 3. Suba o container
docker compose up --build

# 4. Abra no navegador
#    http://localhost:7860
```

---

## Estrutura de dados

```
data/
├── referencias/
│   ├── imagens/
│   │   ├── Micropogonias_furnieri/   ← subpasta = nome da espécie
│   │   │   ├── ref_001.jpg
│   │   │   └── ref_002.jpg
│   │   └── Mugil_liza/
│   │       └── ref_001.jpg
│   └── pdfs/
│       ├── artigo_otolitos_2022.pdf  ← imagens extraídas automaticamente
│       └── livro_peixes_estuarinos.pdf
└── amostras/
    └── amostra_01.jpg                ← fotos novas para identificar
```

> **Dica:** Se não tiver imagens rotuladas por espécie ainda, coloque tudo
> em `data/referencias/imagens/` (sem subpastas). O banco funciona normalmente;
> apenas o fine-tuning e os labels nos resultados não estarão disponíveis.

---

## Fluxo de uso

### 1. Construir o banco de referências
Na interface web, vá à aba **"Banco de referências"** e clique em **"Construir banco"**.

Ou via terminal dentro do container:
```bash
docker compose exec otolith python -c "
from pipeline.identifier import build_reference_database
build_reference_database(
    image_dirs=['data/referencias/imagens'],
    pdf_dirs=['data/referencias/pdfs'],
)
"
```

### 2. Identificar amostras
Na aba **"Identificar amostra"**, faça upload da foto e clique em **"Identificar"**.

### 3. Fine-tuning (opcional, melhora a acurácia)
Requer imagens organizadas por espécie em subpastas.

Via interface: aba **"Fine-tuning"** → ajuste epochs e batch → **"Iniciar fine-tuning"**

Via terminal:
```bash
docker compose exec otolith python -m training.finetune \
    --data_dir data/referencias/imagens \
    --backbone resnet50 \
    --epochs_head 5 \
    --epochs_full 10
```

---

## Backbones disponíveis

| Backbone         | Dims  | Velocidade | Acurácia |
|------------------|-------|------------|----------|
| `resnet50`       | 2048  | ★★★        | ★★★      |
| `efficientnet_b0`| 1280  | ★★★★       | ★★★      |
| `efficientnet_b4`| 1792  | ★★         | ★★★★     |

Todos baixados automaticamente do torchvision na primeira execução (~100 MB).

---

## GPU (opcional)

Para usar GPU NVIDIA, edite o `docker-compose.yml` e descomente o bloco `deploy`,
e troque a imagem base no `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

---

## APIs e custos

| Componente        | Ferramenta        | Custo |
|-------------------|-------------------|-------|
| CNN (embeddings)  | PyTorch/torchvision | Grátis |
| Pré-processamento | OpenCV            | Grátis |
| Busca de similaridade | FAISS / sklearn | Grátis |
| Extração de PDFs  | PyMuPDF           | Grátis |
| Interface web     | Gradio            | Grátis |
| Segmentação avançada | SAM (opcional) | Grátis |

**Custo total: R$ 0,00.** Todo o processamento roda localmente no seu computador.

---

## Próximos passos sugeridos

- [ ] Adicionar mais imagens de referência por espécie
- [ ] Executar fine-tuning com imagens rotuladas
- [ ] Testar `efficientnet_b4` para maior acurácia
- [ ] Integrar SAM para segmentação mais precisa (ver comentários em `preprocessor.py`)
- [ ] Exportar resultados para CSV para análise estatística

---

## Processamento em lote (batch)

Processa toda a pasta `data/amostras/` de uma vez:

```bash
docker compose exec otolith python -m scripts.batch_identify \
    --samples_dir data/amostras \
    --top_k 5
# Saída em outputs/: results.csv, report.html, results.json
```

Ou pela aba **"Processamento em lote"** na interface web.

## Avaliação de acurácia

```bash
docker compose exec otolith python -m scripts.evaluate \
    --data_dir data/referencias/imagens \
    --top_k 5
# Saída em outputs/evaluation/: confusion_matrix.png, metrics.csv, report.txt
```

## Segmentação avançada com SAM (opcional)

```bash
pip install segment-anything
wget -O models/sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Ver `pipeline/sam_segmentor.py` para uso completo.
