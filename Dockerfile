FROM python:3.11-slim

# Dependências de sistema para OpenCV e processamento de imagem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch com CUDA 12.8 (suporte Blackwell — RTX 50-series)
# Wheels oficiais já trazem o runtime CUDA embutido; só precisa do driver NVIDIA no host.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cria pastas de dados caso não existam ao montar volumes
RUN mkdir -p data/referencias/imagens data/referencias/pdfs \
             data/amostras models outputs

EXPOSE 7860

CMD ["python", "app.py"]
