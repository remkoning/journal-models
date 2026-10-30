"""Step 3: embed abstract_norm for every work -> data/local/embeddings.npy + embedding_ids.json (+ embedding_meta.json)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import embed as E  # noqa: E402
from jm.openalex import LOCAL  # noqa: E402


def text_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df.openalex_id + "\t" + df.abstract_norm).encode()).hexdigest()


def main():
    """Skips re-embedding when ids and texts are unchanged (pass --force to re-embed)."""
    df = pd.read_parquet(LOCAL / "works.parquet", columns=["openalex_id", "abstract_norm"])
    h = text_hash(df)
    ids_p, emb_p, meta_p = LOCAL / "embedding_ids.json", LOCAL / "embeddings.npy", LOCAL / "embedding_meta.json"
    if "--force" not in sys.argv and ids_p.exists() and emb_p.exists() and meta_p.exists():
        old = json.loads(meta_p.read_text())
        if json.loads(ids_p.read_text()) == df.openalex_id.tolist() and old.get("text_sha256", h) == h:
            print("embeddings up to date; not recomputed")
            return
    t0 = time.time()
    X = E.embed(df["abstract_norm"].tolist())
    tmp = LOCAL / "embeddings.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, emb_p)
    ids_p.write_text(json.dumps(df["openalex_id"].tolist()))
    meta = E.model_info() | {"text_field": "abstract_norm", "n": int(len(df)), "seconds": round(time.time() - t0, 1),
                             "built": time.strftime("%Y-%m-%d"), "text_sha256": h}
    (LOCAL / "embedding_meta.json").write_text(json.dumps(meta, indent=1))
    (ROOT / "data" / "manifest" / "embedding_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, X.shape)


if __name__ == "__main__":
    main()
