"""
pipeline/database.py
====================
Banco de vetores de referência e busca por similaridade.

- Com FAISS  → busca muito rápida, boa para bases grandes (>1000 imagens)
- Sem FAISS  → fallback para cosine_similarity do scikit-learn
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ReferenceDatabase:
    """
    Armazena embeddings das imagens de referência e realiza
    buscas por similaridade contra novos otólitos.
    """

    def __init__(self):
        self.paths: list[str] = []
        self.labels: list[str] = []        # nome da espécie (opcional)
        self.embeddings: np.ndarray | None = None
        self._faiss_index = None

    # ── Construção ─────────────────────────────────────────────────

    def build(
        self,
        paths: list[str],
        embeddings: np.ndarray,
        labels: list[str] | None = None,
    ):
        """
        Constrói o índice.

        paths      : caminhos das imagens de referência
        embeddings : matriz float32 [N, D]
        labels     : nome da espécie de cada imagem (opcional)
        """
        self.paths = list(paths)
        self.labels = labels if labels else [""] * len(paths)
        self.embeddings = embeddings.astype("float32")

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            emb_copy = self.embeddings.copy()
            faiss.normalize_L2(emb_copy)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(emb_copy)
            print(f"[DB] Índice FAISS criado — {len(paths)} referências, dim={dim}")
        else:
            print(f"[DB] Índice sklearn criado — {len(paths)} referências")

    # ── Persistência ────────────────────────────────────────────────

    def save(self, db_path: str):
        data = {
            "paths": self.paths,
            "labels": self.labels,
            "embeddings": self.embeddings.tolist(),
        }
        Path(db_path).write_text(json.dumps(data, ensure_ascii=False))
        print(f"[DB] Banco salvo → '{db_path}'")

    def load(self, db_path: str):
        data = json.loads(Path(db_path).read_text())
        embs = np.array(data["embeddings"], dtype="float32")
        self.build(data["paths"], embs, data.get("labels"))
        print(f"[DB] Banco carregado ← '{db_path}' ({len(self.paths)} refs)")

    def __len__(self) -> int:
        return len(self.paths)

    # ── Busca ───────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retorna os top_k resultados mais similares.

        Cada resultado é um dict:
          {
            "rank":  1,
            "score": 0.923,
            "path":  "data/referencias/...",
            "label": "Micropogonias furnieri",   # "" se não informado
          }
        """
        if self.embeddings is None or len(self.paths) == 0:
            raise RuntimeError("Banco vazio. Chame build() ou load() primeiro.")

        top_k = min(top_k, len(self.paths))
        query = query_embedding.astype("float32").reshape(1, -1)

        if FAISS_AVAILABLE and self._faiss_index is not None:
            q = query.copy()
            faiss.normalize_L2(q)
            scores, indices = self._faiss_index.search(q, top_k)
            scores, indices = scores[0].tolist(), indices[0].tolist()
        else:
            sim = cosine_similarity(query, self.embeddings)[0]
            indices = np.argsort(sim)[::-1][:top_k].tolist()
            scores = sim[indices].tolist()

        return [
            {
                "rank":  rank + 1,
                "score": float(score),
                "path":  self.paths[idx],
                "label": self.labels[idx],
            }
            for rank, (idx, score) in enumerate(zip(indices, scores))
        ]

    # ── Utilitários ────────────────────────────────────────────────

    def add(self, path: str, embedding: np.ndarray, label: str = ""):
        """Adiciona uma única referência ao banco já existente."""
        self.paths.append(path)
        self.labels.append(label)
        vec = embedding.astype("float32").reshape(1, -1)
        self.embeddings = (
            vec if self.embeddings is None
            else np.vstack([self.embeddings, vec])
        )
        if FAISS_AVAILABLE and self._faiss_index is not None:
            v = vec.copy()
            faiss.normalize_L2(v)
            self._faiss_index.add(v)
