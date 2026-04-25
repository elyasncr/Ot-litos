# Adaptar pipeline para `Gabarito/`, `Amostras/` e suporte a HEIC

**Data:** 2026-04-25
**Escopo:** Trocar a estrutura de dados padrão do projeto (de `data/referencias/...` e `data/amostras/` para `Gabarito/`, `Referencias/`, `Amostras/` na raiz) e adicionar suporte ao formato HEIC nas leituras de imagem.

---

## Motivação

O usuário colocou na raiz do projeto:

- `Gabarito/` — 5 PDFs científicos de referência (atlases de otólitos).
- `Amostras/` — ~270 fotos `.HEIC` (iPhone) de otólitos a identificar.

A estrutura esperada pelo `README.md` original (`data/referencias/pdfs`, `data/referencias/imagens`, `data/amostras`) está vazia. Adaptar o pipeline a essa nova organização e ao formato HEIC permite usar os dados sem mover arquivos nem converter as fotos manualmente.

---

## Decisões de design

### Caminhos: substituição completa dos defaults

A pasta `data/` antiga é removida. As constantes de caminho passam a apontar para a nova estrutura na raiz:

| Antigo                          | Novo            |
|---------------------------------|-----------------|
| `data/referencias/pdfs`         | `Gabarito`      |
| `data/referencias/imagens`      | `Referencias`   |
| `data/amostras`                 | `Amostras`      |

`Referencias/` é criada vazia (com `.gitkeep`) para uso futuro com imagens rotuladas por espécie — habilita fine-tuning e labels nos resultados sem precisar mexer no código de novo.

### HEIC: decodificação transparente via `pillow-heif`

Adiciona-se `pillow-heif==0.16.0` ao `requirements.txt`. As wheels já incluem `libheif`, então não é preciso instalar pacote do sistema no `Dockerfile`.

O opener HEIF é registrado uma única vez em `pipeline/__init__.py`, de modo que qualquer `Image.open(...)` no projeto passe a aceitar `.heic`/`.HEIC` transparentemente.

A única função que usa `cv2.imread` é `pipeline/preprocessor.py:load_image`. Ela é adaptada para detectar HEIC pela extensão e usar `Image.open(...).convert("RGB")` → `np.array` → conversão para BGR antes de devolver o ndarray esperado pelo restante do pipeline.

### Consolidação de `IMAGE_EXTENSIONS`

Hoje a constante está duplicada em quatro arquivos: `pipeline/identifier.py:21`, `scripts/batch_identify.py:36`, `scripts/evaluate.py:46` e `training/dataset.py:30` — com pequenas variações entre eles (alguns incluem `.webp`, outros não). Ela passa a viver junto de `load_image` em `pipeline/preprocessor.py`, e os quatro consumidores importam de lá. A nova lista inclui `.heic` e `.heif`. A varredura já usa `.suffix.lower()`, então variações de caixa (`.HEIC`/`.heic`) são pegas automaticamente.

---

## Mudanças por arquivo

| Arquivo | Mudança |
|---|---|
| `requirements.txt` | + `pillow-heif==0.16.0` |
| `pipeline/__init__.py` | Registra opener HEIF (`register_heif_opener()`) |
| `pipeline/preprocessor.py` | `load_image` lê HEIC via Pillow; expõe `IMAGE_EXTENSIONS` (com `.heic`, `.heif`) |
| `pipeline/identifier.py` | Importa `IMAGE_EXTENSIONS` de `preprocessor` (remove duplicata local) |
| `scripts/batch_identify.py` | Importa `IMAGE_EXTENSIONS` de `preprocessor`; default CLI `--samples_dir Amostras` |
| `scripts/evaluate.py` | Importa `IMAGE_EXTENSIONS` de `preprocessor` (remove duplicata local); default CLI `--data_dir Referencias` |
| `training/dataset.py` | Importa `IMAGE_EXTENSIONS` de `preprocessor` (remove duplicata local) |
| `training/finetune.py` | Default CLI `--data_dir Referencias` |
| `app.py` | Constantes: `IMAGES_DIR="Referencias"`, `PDFS_DIR="Gabarito"`, `SAMPLES_DIR="Amostras"` |
| `Dockerfile` | `mkdir -p Gabarito Amostras Referencias models outputs` (remove `data/...`) |
| `docker-compose.yml` | Volumes: `./Gabarito`, `./Amostras`, `./Referencias` (remove `./data`) |
| `README.md` | Atualizar bloco "Estrutura de dados" e exemplos de comando |
| `data/` | Removida (estava vazia) |
| `Referencias/.gitkeep` | Criado |

---

## Snippet de referência — `load_image` adaptado

```python
def load_image(image_path: str) -> np.ndarray:
    """Lê imagem em BGR. Suporta HEIC via Pillow."""
    path = Path(image_path)
    if path.suffix.lower() in {".heic", ".heif"}:
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    return img
```

---

## Verificação manual

O projeto não tem suite de testes automatizados. Validação após implementação:

1. **Smoke import** — `python -c "import pipeline; from PIL import Image; print(Image.open('Amostras/IMG_6199.HEIC').size)"` confirma que o opener HEIF foi registrado.
2. **`load_image` com HEIC** — script curto chamando `pipeline.preprocessor.load_image('Amostras/IMG_6199.HEIC')` e imprimindo `shape`; espera-se `(altura, largura, 3)` em BGR.
3. **End-to-end em subset** — construir o banco a partir de `Gabarito/` (`build_reference_database(pdf_dirs=['Gabarito'])`) e rodar `python -m scripts.batch_identify --top_k 3` em ~5 amostras. Confirmar que `outputs/results.csv` e `outputs/report.html` saem populados.
4. **Regressão UI** — `docker compose up --build`; abrir `http://localhost:7860`, fazer upload de uma HEIC e confirmar que a identificação retorna resultados.

---

## Fora de escopo

- Conversão prévia de HEIC para JPG em cache (descartada — ganho marginal para 270 imagens, custo de gerenciamento de cache não compensa).
- Configuração de caminhos via variáveis de ambiente (descartada — projeto pessoal, defaults explícitos no código bastam).
- Adicionar imagens rotuladas em `Referencias/` (a pasta é criada vazia para uso futuro).
- Suite de testes automatizados.
