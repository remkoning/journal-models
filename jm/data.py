"""Data loader for downstream agents (EVALUATE / DEVELOP / CREDIT).

Text lives only in data/local/ (gitignored; rebuild with `.venv/bin/python -m data.build_all`).
IDs are short OpenAlex ids ("W2963...").

    from jm.data import load_works, text, load_embeddings, emb, load_pairs
    df = load_works(core_only=True)                 # one row per article
    text("W2963...")                                # abstract_norm (default) for corpus or develop-future works
    text("W2963...", field="abstract_clean")
    ids, X = load_embeddings()                       # X[i] is L2-normalized bge-base embedding of ids[i]
    pairs = load_pairs("cross", "test")              # list[dict] from evals/pairs/cross_test.jsonl
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "local"
MANIFEST = ROOT / "data" / "manifest"
EVALS = ROOT / "evals"
DEVELOP = ROOT / "develop"
CREDIT = ROOT / "credit"
JOURNALS = ("OrgSci", "SMJ", "MgmtSci")
JOURNAL_NAMES = {"OrgSci": "Organization Science", "SMJ": "Strategic Management Journal", "MgmtSci": "Management Science"}


def _sid(x: str) -> str:
    return str(x).rsplit("/", 1)[-1]


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Rebuild the local store: .venv/bin/python -m data.build_all")
    return p


@lru_cache(maxsize=1)
def _works() -> pd.DataFrame:
    return pd.read_parquet(_require(LOCAL / "works.parquet"))


def load_works(core_only: bool = False, journals: tuple[str, ...] | None = None, years: tuple[int, int] | None = None,
               split: str | None = None) -> pd.DataFrame:
    """All cleaned articles (OrgSci, SMJ, MgmtSci; 2016-2026). Returns a copy."""
    df = _works()
    m = pd.Series(True, index=df.index)
    if core_only:
        m &= df["in_core"]
    if journals:
        m &= df["journal"].isin(journals)
    if years:
        m &= df["year"].between(*years)
    if split:
        m &= df["split"] == split
    return df[m].copy()


@lru_cache(maxsize=1)
def _text_index() -> dict:
    idx = {}
    df = _works()
    for r in df[["openalex_id", "abstract_norm", "abstract_clean", "abstract_raw", "title"]].itertuples(index=False):
        idx[r.openalex_id] = {"abstract_norm": r.abstract_norm, "abstract_clean": r.abstract_clean,
                              "abstract_raw": r.abstract_raw, "title": r.title}
    fp = LOCAL / "develop_future.parquet"
    if fp.exists():
        fut = pd.read_parquet(fp).drop_duplicates("future_id")
        for r in fut.itertuples(index=False):
            if r.future_id not in idx:
                idx[r.future_id] = {"abstract_norm": r.abstract_norm, "abstract_clean": r.abstract_clean,
                                    "abstract_raw": r.abstract_raw, "title": r.title}
    return idx


def text(openalex_id: str, field: str = "abstract_norm") -> str:
    """Abstract text for a corpus work or a DEVELOP future work. field: abstract_norm | abstract_clean | abstract_raw | title."""
    return _text_index()[_sid(openalex_id)][field]


def title(openalex_id: str) -> str:
    return text(openalex_id, "title")


@lru_cache(maxsize=1)
def load_embeddings() -> tuple[list[str], np.ndarray]:
    ids = json.loads(_require(LOCAL / "embedding_ids.json").read_text())
    X = np.load(_require(LOCAL / "embeddings.npy"))
    return ids, X


@lru_cache(maxsize=1)
def _emb_index() -> dict:
    ids, _ = load_embeddings()
    return {k: i for i, k in enumerate(ids)}


def emb(openalex_ids) -> np.ndarray:
    """Embedding row(s) for corpus works (bge-base-en-v1.5 on abstract_norm, L2-normalized)."""
    _, X = load_embeddings()
    if isinstance(openalex_ids, str):
        return X[_emb_index()[_sid(openalex_ids)]]
    return X[[_emb_index()[_sid(i)] for i in openalex_ids]]


def load_future_embeddings() -> tuple[list[str], np.ndarray]:
    """Embeddings of DEVELOP citing works that are outside the corpus (same model)."""
    ids = json.loads(_require(LOCAL / "develop_future_embedding_ids.json").read_text())
    return ids, np.load(_require(LOCAL / "develop_future_embeddings.npy"))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in _require(p).read_text().splitlines() if line.strip()]


def load_pairs(kind: str, split: str) -> list[dict]:
    """kind: 'cross' | 'cite' | 'cross_fullms' ; split: train | val | test."""
    return _jsonl(EVALS / "pairs" / f"{kind}_{split}.jsonl")


def load_seeds() -> list[dict]:
    return _jsonl(DEVELOP / "seeds.jsonl")


def load_develop_future() -> pd.DataFrame:
    """One row per (seed, future work): kind in {citing_top, semantic_neighbor}, with text columns."""
    return pd.read_parquet(_require(LOCAL / "develop_future.parquet"))


def load_credit() -> dict:
    return {"players": json.loads((CREDIT / "players.json").read_text()),
            "future": json.loads((CREDIT / "future_T.json").read_text()),
            "edges": _jsonl(CREDIT / "citation_edges.jsonl")}
