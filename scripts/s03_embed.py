"""Step 3: embed abstract_norm for every work -> data/local/embeddings.npy + embedding_ids.json (+ embedding_meta.json)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import embed as E  # noqa: E402
from jm.openalex import LOCAL  # noqa: E402


def main():
    df = pd.read_parquet(LOCAL / "works.parquet", columns=["openalex_id", "abstract_norm"])
    t0 = time.time()
    X = E.embed(df["abstract_norm"].tolist())
    np.save(LOCAL / "embeddings.npy", X)
    (LOCAL / "embedding_ids.json").write_text(json.dumps(df["openalex_id"].tolist()))
    meta = E.model_info() | {"text_field": "abstract_norm", "n": int(len(df)), "seconds": round(time.time() - t0, 1),
                             "built": time.strftime("%Y-%m-%d")}
    (LOCAL / "embedding_meta.json").write_text(json.dumps(meta, indent=1))
    (ROOT / "data" / "manifest" / "embedding_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, X.shape)


if __name__ == "__main__":
    main()
