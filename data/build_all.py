"""Regenerate the whole local data store with one command:

    .venv/bin/python -m data.build_all                  # restore data/local for the committed IDs/pairs/seeds
    .venv/bin/python -m data.build_all --rebuild-evals  # re-derive everything (overwrites committed manifests/pairs)

Steps: s01 fetch OpenAlex corpus (cached; ~52 credits cold) -> s02 clean/build works.parquet -> s03 embeddings
-> s04 EVALUATE pairs -> s05 DEVELOP seeds/futures -> s06 CREDIT sets -> s07 Gong external check -> s08 data card.
Without --rebuild-evals, committed artifacts (data/manifest, evals/, develop/, credit/) are kept and only the
gitignored text/embedding store is rebuilt (DEVELOP futures are re-fetched by ID: ~20 credits cold).
OpenAlex budget without a key is 1,000 credits/day; a cold full rebuild uses ~200.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script: str, *args: str):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([PY, str(ROOT / "scripts" / script), *args], check=True, cwd=ROOT)


def main():
    rebuild = "--rebuild-evals" in sys.argv
    committed = (ROOT / "data" / "manifest" / "works_manifest.csv").exists()
    run("s01_fetch_corpus.py")
    run("s02_build_works.py", *([] if rebuild or not committed else ["--pin"]))
    run("s03_embed.py")
    if rebuild or not (ROOT / "evals" / "pairs" / "cross_test.jsonl").exists():
        run("s04_pairs.py")
    if rebuild or not (ROOT / "develop" / "seeds.jsonl").exists():
        run("s05_develop.py")
    else:
        run("s05_develop.py", "--from-manifest")
    if rebuild or not (ROOT / "credit" / "players.json").exists():
        run("s06_credit.py")
    run("s07_external_gong.py")
    if rebuild or not (ROOT / "data" / "README.md").exists():
        run("s08_datacard.py")
    print("\nDone. Local store in data/local/; load with jm.data.load_works() / jm.data.text(id).")


if __name__ == "__main__":
    main()
