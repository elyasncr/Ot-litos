# Plano de implementação — `Gabarito/`, `Amostras/` e suporte a HEIC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir os defaults de caminho do pipeline (de `data/...` para `Gabarito/`/`Amostras/`/`Referencias/` na raiz) e adicionar suporte transparente a HEIC.

**Architecture:** Decodificação HEIC via `pillow-heif` registrado uma única vez em `pipeline/__init__.py`. Constante `IMAGE_EXTENSIONS` consolidada em `pipeline/preprocessor.py` e importada pelos quatro consumidores. Defaults de caminho reescritos diretamente no código (sem env vars).

**Tech Stack:** Python 3.11, Pillow, OpenCV, pillow-heif (novo), Docker, Gradio.

**Notas para o executor:**
- O projeto **não é um repositório git** — pule os passos `git commit`. Após cada tarefa, faça os checkpoints de verificação especificados.
- O projeto **não tem suite de testes automatizados** — a verificação é feita rodando comandos manuais e olhando saídas, conforme cada passo.
- Ambiente: Windows 11 + bash. Use forward slashes em paths e `python` (não `python3`). Working directory: `D:\AI Solution\Otólitos`.
- O Docker NÃO é estritamente necessário para validar a maioria das tarefas — apenas a verificação final usa `docker compose`. As validações intermediárias podem ser feitas com Python local se houver venv ativo.

**Spec relacionada:** `docs/superpowers/specs/2026-04-25-gabarito-amostras-heic-design.md`

---

## Mapa de arquivos

**Modificar:**
- `requirements.txt`
- `pipeline/__init__.py`
- `pipeline/preprocessor.py:21-34` (adicionar `IMAGE_EXTENSIONS`, adaptar `load_image`)
- `pipeline/identifier.py:21` (remover duplicata, importar)
- `scripts/batch_identify.py:36, 244` (remover duplicata, importar; CLI default)
- `scripts/evaluate.py:46, 264` (remover duplicata, importar; CLI default)
- `training/dataset.py:30` (remover duplicata, importar)
- `training/finetune.py:246` (CLI default)
- `app.py:41-43` (constantes de caminho)
- `Dockerfile:20-21` (mkdir das pastas)
- `docker-compose.yml:8` (volumes)
- `README.md` (bloco "Estrutura de dados", exemplos de comando)

**Criar:**
- `Referencias/.gitkeep` (pasta vazia para imagens rotuladas futuras)

**Remover:**
- `data/` (estrutura inteira, atualmente vazia)

---

## Task 1 — Adicionar `pillow-heif` e registrar opener

**Files:**
- Modify: `requirements.txt`
- Modify: `pipeline/__init__.py:1`

- [ ] **Step 1: Adicionar dependência ao `requirements.txt`**

Adicionar a linha `pillow-heif==0.16.0` ao arquivo. O conteúdo final fica:

```
torch==2.2.2
torchvision==0.17.2
Pillow==10.3.0
pillow-heif==0.16.0
opencv-python-headless==4.9.0.80
faiss-cpu==1.8.0
scikit-learn==1.4.2
pymupdf==1.24.1
numpy==1.26.4
tqdm==4.66.2
gradio==4.31.5
matplotlib==3.8.4
pandas==2.2.2
```

- [ ] **Step 2: Instalar a nova dependência localmente**

Run: `pip install pillow-heif==0.16.0`
Expected: `Successfully installed pillow-heif-0.16.0` (ou similar). Se já estiver instalado, `Requirement already satisfied`.

- [ ] **Step 3: Registrar o opener HEIF em `pipeline/__init__.py`**

No topo do arquivo (antes de qualquer outro import de `pipeline.*`), adicionar:

```python
from pillow_heif import register_heif_opener

register_heif_opener()
```

O arquivo final fica:

```python
from pillow_heif import register_heif_opener

register_heif_opener()

from pipeline.preprocessor import preprocess, preprocess_pil
from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.pdf_extractor import extract_images_from_pdf, extract_all_pdfs
from pipeline.identifier import (
    build_reference_database,
    load_database,
    identify_from_path,
    identify_from_pil,
)

__all__ = [
    "preprocess", "preprocess_pil",
    "FeatureExtractor",
    "ReferenceDatabase",
    "extract_images_from_pdf", "extract_all_pdfs",
    "build_reference_database", "load_database",
    "identify_from_path", "identify_from_pil",
]
```

- [ ] **Step 4: Verificar que o opener foi registrado**

Run: `python -c "import pipeline; from PIL import Image; img = Image.open('Amostras/IMG_6199.HEIC'); print(img.size, img.mode)"`
Expected: imprime algo como `(4032, 3024) RGB` (dimensões podem variar). **Não deve** lançar `UnidentifiedImageError`. Se faltar `Amostras/IMG_6199.HEIC`, troque por outro nome de arquivo HEIC presente na pasta.

---

## Task 2 — Adaptar `load_image` para HEIC e expor `IMAGE_EXTENSIONS`

**Files:**
- Modify: `pipeline/preprocessor.py:21, 29-34`

- [ ] **Step 1: Adicionar `IMAGE_EXTENSIONS` e adaptar `load_image`**

Em `pipeline/preprocessor.py`, depois das constantes de tamanho (após linha 22) e substituindo o corpo atual de `load_image` (linhas 29-34), o trecho fica:

```python
# ── Configurações ──────────────────────────────────────────────────
TARGET_SIZE = (224, 224)   # tamanho de entrada da CNN
BLUR_KERNEL = (3, 3)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".webp", ".heic", ".heif",
}
HEIC_EXTENSIONS = {".heic", ".heif"}


# ══════════════════════════════════════════════════════════════════
# Funções públicas
# ══════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> np.ndarray:
    """Lê a imagem em BGR (formato padrão do OpenCV). Suporta HEIC via Pillow."""
    path = Path(image_path)
    if path.suffix.lower() in HEIC_EXTENSIONS:
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    return img
```

Garantir que `from pathlib import Path` está nos imports do arquivo. Se não estiver, adicionar.

- [ ] **Step 2: Verificar imports do arquivo**

Run: `python -c "import pathlib; from pipeline.preprocessor import load_image, IMAGE_EXTENSIONS, HEIC_EXTENSIONS; print(sorted(IMAGE_EXTENSIONS))"`
Expected: imprime `['.bmp', '.heic', '.heif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp']` sem erros.

- [ ] **Step 3: Verificar que `load_image` lê HEIC**

Run: `python -c "from pipeline.preprocessor import load_image; img = load_image('Amostras/IMG_6199.HEIC'); print('shape=', img.shape, 'dtype=', img.dtype)"`
Expected: imprime `shape= (altura, largura, 3) dtype= uint8` — algo como `shape= (3024, 4032, 3) dtype= uint8`.

- [ ] **Step 4: Verificar que `load_image` ainda lê JPG**

Crie uma JPG de teste a partir de uma HEIC para confirmar regressão zero:
Run:
```bash
python -c "
from PIL import Image
import pipeline  # registra HEIF
Image.open('Amostras/IMG_6199.HEIC').convert('RGB').save('/tmp/test.jpg')
from pipeline.preprocessor import load_image
img = load_image('/tmp/test.jpg')
print('jpg ok, shape=', img.shape)
"
```
Expected: `jpg ok, shape= (...)`. (No Windows, troque `/tmp/test.jpg` por um caminho válido como `outputs/_test.jpg`.)

---

## Task 3 — Consolidar `IMAGE_EXTENSIONS` nos 4 consumidores

**Files:**
- Modify: `pipeline/identifier.py:18-21`
- Modify: `scripts/batch_identify.py:32-36`
- Modify: `scripts/evaluate.py:42-46`
- Modify: `training/dataset.py:27-30`

- [ ] **Step 1: `pipeline/identifier.py`**

Substituir as linhas 18-21:

```python
from pipeline.database import ReferenceDatabase
from pipeline.extractor import FeatureExtractor
from pipeline.pdf_extractor import extract_all_pdfs
from pipeline.preprocessor import preprocess, preprocess_pil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
```

por:

```python
from pipeline.database import ReferenceDatabase
from pipeline.extractor import FeatureExtractor
from pipeline.pdf_extractor import extract_all_pdfs
from pipeline.preprocessor import preprocess, preprocess_pil, IMAGE_EXTENSIONS
```

(Remove a definição local; importa do `preprocessor`.)

- [ ] **Step 2: `scripts/batch_identify.py`**

Substituir linhas 32-36:

```python
from pipeline.extractor import FeatureExtractor
from pipeline.identifier import load_database, identify_from_pil
from pipeline.preprocessor import preprocess_pil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
```

por:

```python
from pipeline.extractor import FeatureExtractor
from pipeline.identifier import load_database, identify_from_pil
from pipeline.preprocessor import preprocess_pil, IMAGE_EXTENSIONS
```

- [ ] **Step 3: `scripts/evaluate.py`**

Substituir linhas 42-46:

```python
from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.preprocessor import preprocess_pil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
```

por:

```python
from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.preprocessor import preprocess_pil, IMAGE_EXTENSIONS
```

- [ ] **Step 4: `training/dataset.py`**

Substituir linhas 27-30:

```python
from pipeline.preprocessor import preprocess_pil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
```

por:

```python
from pipeline.preprocessor import preprocess_pil, IMAGE_EXTENSIONS
```

- [ ] **Step 5: Verificar todos os imports**

Run:
```bash
python -c "
from pipeline.identifier import IMAGE_EXTENSIONS as a
from scripts.batch_identify import IMAGE_EXTENSIONS as b
from scripts.evaluate import IMAGE_EXTENSIONS as c
from training.dataset import IMAGE_EXTENSIONS as d
print('all four equal:', a == b == c == d, '|', sorted(a))
"
```
Expected: `all four equal: True | ['.bmp', '.heic', '.heif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp']`

---

## Task 4 — Atualizar constantes de caminho em `app.py`

**Files:**
- Modify: `app.py:38-43`

- [ ] **Step 1: Substituir o bloco de paths padrão**

Substituir as linhas 38-43:

```python
# ── Paths padrão ─────────────────────────────────────────────────
DB_PATH         = "models/reference_db.json"
FINETUNED_PATH  = "models/finetuned_resnet50.pth"
IMAGES_DIR      = "data/referencias/imagens"
PDFS_DIR        = "data/referencias/pdfs"
SAMPLES_DIR     = "data/amostras"
```

por:

```python
# ── Paths padrão ─────────────────────────────────────────────────
DB_PATH         = "models/reference_db.json"
FINETUNED_PATH  = "models/finetuned_resnet50.pth"
IMAGES_DIR      = "Referencias"
PDFS_DIR        = "Gabarito"
SAMPLES_DIR     = "Amostras"
```

- [ ] **Step 2: Verificar que o módulo carrega**

Run: `python -c "import app; print(app.IMAGES_DIR, app.PDFS_DIR, app.SAMPLES_DIR)"`
Expected: `Referencias Gabarito Amostras` sem erros de import.

---

## Task 5 — Atualizar defaults de CLI nos scripts

**Files:**
- Modify: `scripts/batch_identify.py:244`
- Modify: `scripts/evaluate.py:264`
- Modify: `training/finetune.py:246`

- [ ] **Step 1: `scripts/batch_identify.py`**

Substituir linha 244:

```python
    parser.add_argument("--samples_dir", default="data/amostras")
```

por:

```python
    parser.add_argument("--samples_dir", default="Amostras")
```

- [ ] **Step 2: `scripts/evaluate.py`**

Substituir linha 264:

```python
    parser.add_argument("--data_dir",    default="data/referencias/imagens")
```

por:

```python
    parser.add_argument("--data_dir",    default="Referencias")
```

- [ ] **Step 3: `training/finetune.py`**

Substituir linha 246:

```python
    parser.add_argument("--data_dir",     default="data/referencias/imagens")
```

por:

```python
    parser.add_argument("--data_dir",     default="Referencias")
```

- [ ] **Step 4: Verificar `--help` dos três comandos**

Run:
```bash
python -m scripts.batch_identify --help | grep samples_dir
python -m scripts.evaluate --help | grep data_dir
python -m training.finetune --help | grep data_dir
```
Expected: cada linha mostra o novo default (`Amostras` ou `Referencias`) na descrição da flag.

---

## Task 6 — Atualizar `Dockerfile`

**Files:**
- Modify: `Dockerfile:19-21`

- [ ] **Step 1: Substituir o bloco de `mkdir`**

Substituir linhas 19-21:

```dockerfile
# Cria pastas de dados caso não existam ao montar volumes
RUN mkdir -p data/referencias/imagens data/referencias/pdfs \
             data/amostras models outputs
```

por:

```dockerfile
# Cria pastas de dados caso não existam ao montar volumes
RUN mkdir -p Gabarito Amostras Referencias models outputs
```

- [ ] **Step 2: (Opcional, requer Docker) Validar build**

Run: `docker compose build`
Expected: build conclui sem erros. **Pular se Docker não estiver disponível** — a verificação final do plano cobre isso.

---

## Task 7 — Atualizar `docker-compose.yml`

**Files:**
- Modify: `docker-compose.yml:6-10`

- [ ] **Step 1: Substituir o bloco de volumes**

Substituir linhas 6-10:

```yaml
    volumes:
      # Mapeie suas pastas locais aqui — os dados nunca ficam dentro da imagem
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
```

por:

```yaml
    volumes:
      # Mapeie suas pastas locais aqui — os dados nunca ficam dentro da imagem
      - ./Gabarito:/app/Gabarito
      - ./Amostras:/app/Amostras
      - ./Referencias:/app/Referencias
      - ./models:/app/models
      - ./outputs:/app/outputs
```

- [ ] **Step 2: Validar sintaxe YAML**

Run: `python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"`
Expected: sem erros (o módulo `yaml` faz parte do PyYAML — pode estar disponível via gradio/pymupdf; se não, instalar com `pip install pyyaml` apenas para esse check).

Alternativa: `docker compose config` (se Docker disponível) → imprime a configuração resolvida.

---

## Task 8 — Atualizar `README.md`

**Files:**
- Modify: `README.md` (linhas 36-50, ~64-72, ~83-89, ~144-149, ~155-161)

- [ ] **Step 1: Atualizar o bloco "Estrutura de dados"**

Substituir as linhas 34-54 (do título "## Estrutura de dados" até a "Dica" inclusive):

````markdown
## Estrutura de dados

```
Gabarito/                            ← PDFs de referência (atlases, artigos)
├── Atlas_otolitos_2016.pdf
└── ...

Referencias/                         ← imagens rotuladas por espécie (opcional)
├── Micropogonias_furnieri/          ← subpasta = nome da espécie
│   ├── ref_001.jpg
│   └── ref_002.jpg
└── Mugil_liza/
    └── ref_001.jpg

Amostras/                            ← fotos novas para identificar
├── foto_01.jpg
└── foto_02.HEIC                     ← HEIC suportado (iPhone)
```

> **Dica:** Se ainda não tiver imagens rotuladas por espécie, deixe `Referencias/` vazia.
> O banco funciona com apenas os PDFs em `Gabarito/`; só o fine-tuning e os labels nos
> resultados ficam indisponíveis até você popular `Referencias/`.
````

- [ ] **Step 2: Atualizar exemplos de comando dentro do README**

Buscar e substituir, no arquivo inteiro:

| De                                       | Para               |
|------------------------------------------|--------------------|
| `data/referencias/imagens`               | `Referencias`      |
| `data/referencias/pdfs`                  | `Gabarito`         |
| `data/amostras`                          | `Amostras`         |
| `image_dirs=['data/referencias/imagens']` | `image_dirs=['Referencias']` |
| `pdf_dirs=['data/referencias/pdfs']`     | `pdf_dirs=['Gabarito']` |

Sugestão: usar Edit com `replace_all` para cada par. Verificar visualmente que blocos como `--samples_dir data/amostras` viraram `--samples_dir Amostras`.

- [ ] **Step 3: Verificar que não sobrou nenhuma referência ao caminho antigo**

Run: `grep -n "data/" README.md`
Expected: idealmente nenhuma linha. Se sobrar alguma menção a `data/` que não seja parte do caminho antigo (ex: "dados" em PT), confirmar contexto e ajustar manualmente.

---

## Task 9 — Mudanças de filesystem

**Files:**
- Create: `Referencias/.gitkeep`
- Delete: `data/` (recursivo, vazio)

- [ ] **Step 1: Criar a pasta `Referencias/` com `.gitkeep`**

Run:
```bash
mkdir -p "Referencias"
touch "Referencias/.gitkeep"
```
Expected: pasta criada, arquivo `.gitkeep` vazio dentro.

- [ ] **Step 2: Verificar que `data/` está vazia antes de deletar**

Run: `find data -type f 2>/dev/null | head -20`
Expected: nenhum arquivo listado (a árvore deve ter só pastas vazias). **Se aparecer qualquer arquivo, NÃO delete — pause e reporte ao usuário.**

- [ ] **Step 3: Remover `data/`**

Run: `rm -rf data`
Expected: comando completa silenciosamente. Confirmar com `ls | grep -i data` que não retorna nada.

- [ ] **Step 4: Confirmar estrutura final na raiz**

Run: `ls -d Gabarito Amostras Referencias 2>/dev/null && echo OK || echo FAIL`
Expected: imprime as três pastas seguidas de `OK`.

---

## Task 10 — Verificação end-to-end

**Files:** (nenhum modificado nesta tarefa — só execução)

- [ ] **Step 1: Smoke import do pipeline**

Run: `python -c "import pipeline; from PIL import Image; print(Image.open('Amostras/IMG_6199.HEIC').size)"`
Expected: imprime as dimensões da imagem (ex: `(4032, 3024)`). Sem erros.

- [ ] **Step 2: Construir banco de referências a partir de `Gabarito/`**

Run:
```bash
python -c "
from pipeline.identifier import build_reference_database
db = build_reference_database(pdf_dirs=['Gabarito'], db_output='models/reference_db.json')
print(f'Banco construído com {len(db)} referências')
"
```
Expected: extrai imagens dos PDFs, processa, e imprime `Banco construído com N referências` (N depende do conteúdo dos PDFs, deve ser > 0). Pode demorar alguns minutos. Se quebrar em algum PDF específico, anotar o nome e o erro mas seguir.

- [ ] **Step 3: Rodar batch em subset de 5 amostras**

Criar uma pasta temporária com 5 HEICs e rodar o batch nela:

```bash
mkdir -p _smoke_test
cp Amostras/IMG_6199.HEIC Amostras/IMG_6201.HEIC Amostras/IMG_6202.HEIC Amostras/IMG_6203.HEIC Amostras/IMG_6204.HEIC _smoke_test/
python -m scripts.batch_identify --samples_dir _smoke_test --top_k 3
```

Expected:
- Não deve haver erros de leitura HEIC.
- `outputs/results.csv` deve existir e ter 6 linhas (1 header + 5 amostras).
- `outputs/report.html` deve existir.
- Verificar com: `wc -l outputs/results.csv` (espera 6) e `ls -la outputs/report.html` (existe e > 0 bytes).

Limpar: `rm -rf _smoke_test`

- [ ] **Step 4: (Opcional) Regressão da UI Gradio**

Se Docker disponível:
```bash
docker compose up --build
```
Abrir `http://localhost:7860`. Na aba "Identificar amostra", fazer upload de qualquer HEIC de `Amostras/`. Confirmar que aparece a imagem pré-processada e o ranking de similaridade. Se não houver Docker, esse passo pode ser pulado — a verificação dos passos 1-3 já cobre o pipeline.

- [ ] **Step 5: Reportar resultados ao usuário**

Resumir: quais passos passaram, quantas referências foram extraídas dos PDFs, quantas amostras foram identificadas no smoke test, e qualquer warning notável (ex: PDFs que falharam, HEICs corrompidos).

---

## Notas finais

- Se aparecer erro `cannot import name 'register_heif_opener'`, conferir versão do `pillow-heif` (≥ 0.10).
- Se a UI do Gradio reclamar de imagem inválida ao subir HEIC pelo navegador, o problema é separado: o navegador converte HEIC para JPEG antes do upload na maioria dos casos, então a UI não enxerga HEIC diretamente. O pipeline de batch (`scripts/batch_identify.py`) é onde HEIC importa.
- Nenhuma das mudanças quebra retrocompatibilidade com JPG/PNG — todos os formatos antigos continuam suportados pelo `IMAGE_EXTENSIONS` consolidado.
