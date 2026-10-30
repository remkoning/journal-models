"""Local sentence embeddings (free, reproducible). Model: BAAI/bge-base-en-v1.5, L2-normalized, so cosine = dot."""
from __future__ import annotations

import os

import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"
_model = None


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _model
    if _model is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=_device())
    return _model


def model_info() -> dict:
    import sentence_transformers
    import torch
    m = get_model()
    rev = None
    try:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(MODEL_NAME).sha
    except Exception:
        pass
    return {"model": MODEL_NAME, "hf_revision": rev, "dim": int(getattr(m, "get_embedding_dimension", m.get_sentence_embedding_dimension)()),
            "max_seq_length": int(m.max_seq_length), "normalized": True, "query_instruction": None,
            "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__}


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    m = get_model()
    v = m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500,
                 convert_to_numpy=True)
    return v.astype(np.float32)
