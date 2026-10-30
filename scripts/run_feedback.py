"""FEEDBACK: same paper, three journals. Where would each journal push the research?

1. Take recent held-out abstracts (test split, 2025-26), n per journal.
2. Each journal-model gives fit_score, what_we_value, three next-step suggestions, and a revised abstract.
3. A neutral classifier, blind to which journal wrote each suggestion, labels every suggestion with one category.
4. Embedding check: does each journal's revision move the paper toward that journal's body of work?

    .venv/bin/python -m scripts.run_feedback --mode contrast --n 10
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from jm.data import JOURNALS, JOURNAL_NAMES, emb, load_works, text
from jm.embed import embed
from jm.journal_model import JournalModel
from jm.llm import call, map_parallel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feedback"

CATEGORIES = {
    "theory": "develop or extend theory: new constructs, propositions, theoretical mechanisms or framing",
    "mechanism": "identify or test the causal mechanism or process behind the effect",
    "methods": "research design and identification: causal inference, experiments, robustness, estimation",
    "data": "new data, samples or measurement of constructs",
    "scope": "boundary conditions, context, heterogeneity, generalizability to other settings",
    "practice": "managerial, policy or practical implications and relevance",
    "positioning": "framing, positioning against a literature or conversation, clarity of contribution",
}

CLASS_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "category": {"type": "string", "enum": list(CATEGORIES)}},
        "required": ["item", "category"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}


def pick_papers(n: int, seed: int = 0) -> list[str]:
    df = load_works(core_only=True, split="test", years=(2025, 2026))
    rng = random.Random(f"feedback-{seed}")
    out = []
    for j in JOURNALS:
        out += rng.sample(sorted(df[df.journal == j].openalex_id), n)
    return out


def classify(suggestions: list[dict], batch: int = 30) -> dict[int, str]:
    """Blind classification: shuffled, no journal labels."""
    order = list(range(len(suggestions)))
    random.Random(1).shuffle(order)
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    system = "You are an expert in management research methods. Classify research suggestions precisely."

    def run(chunk):
        body = "\n".join(f"{k + 1}. {suggestions[i]['direction']}: {suggestions[i]['why']}" for k, i in enumerate(chunk))
        r = call(f"Assign each suggestion to exactly one category, choosing the best fit.\n\nCategories:\n{cats}\n\n"
                 f"Suggestions:\n{body}", system=system, model="opus", effort="low", schema=CLASS_SCHEMA,
                 tag="feedback/classify")
        got = {a["item"]: a["category"] for a in (r["json"] or {}).get("answers", [])}
        return {i: got.get(k + 1) for k, i in enumerate(chunk)}

    out = {}
    for res in map_parallel(run, [order[i:i + batch] for i in range(0, len(order), batch)], workers=3):
        if not isinstance(res, Exception):
            out.update(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="contrast")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    papers = pick_papers(args.n)
    meta = load_works().set_index("openalex_id")
    models = {j: JournalModel(j, mode=args.mode) for j in JOURNALS}
    jobs = [(pid, j) for pid in papers for j in JOURNALS]
    res = map_parallel(lambda job: models[job[1]].feedback(job[0], tag=f"feedback/{args.mode}"), jobs, workers=4)
    rows = []
    for (pid, j), r in zip(jobs, res):
        if isinstance(r, Exception) or not r:
            print(f"[warn] {pid} {j}: {r}")
            continue
        rows.append({"paper": pid, "paper_journal": meta.loc[pid, "journal"], "model": j, **r})

    # 1. Blind classification of suggestions
    sugg = [{"row": k, **s} for k, r in enumerate(rows) for s in r["suggestions"]]
    labels = classify(sugg)
    for i, s in enumerate(sugg):
        rows[s["row"]].setdefault("categories", []).append(labels.get(i))
    mix = {j: Counter(c for r in rows if r["model"] == j for c in r.get("categories", []) if c) for j in JOURNALS}

    # 2. Where do revisions move the paper? Centroids are each journal's train-split core papers.
    tr = load_works(core_only=True, split="train")
    cent = {j: emb(list(tr[tr.journal == j].openalex_id)).mean(0) for j in JOURNALS}
    cent = {j: c / np.linalg.norm(c) for j, c in cent.items()}
    rev = embed([r["revised_abstract"] for r in rows])
    move = []
    for r, e in zip(rows, rev):
        e0 = emb(r["paper"])
        d = {j: float(e @ cent[j] - e0 @ cent[j]) for j in JOURNALS}
        r["delta_toward"] = d
        move.append({"model": r["model"], "toward_own": d[r["model"]],
                     "toward_other": float(np.mean([d[k] for k in JOURNALS if k != r["model"]])),
                     "own_is_max": max(d, key=d.get) == r["model"]})

    summary = {
        "mode": args.mode, "n_papers": len(papers), "n_reviews": len(rows),
        "category_mix": {j: dict(mix[j]) for j in JOURNALS},
        "category_share": {j: {c: round(mix[j][c] / max(1, sum(mix[j].values())), 3) for c in CATEGORIES} for j in JOURNALS},
        "mean_fit_score_by_model_and_paper_journal": {
            j: {k: float(np.mean([r["fit_score"] for r in rows if r["model"] == j and r["paper_journal"] == k]))
                for k in JOURNALS} for j in JOURNALS},
        "movement": {j: {"mean_toward_own": float(np.mean([m["toward_own"] for m in move if m["model"] == j])),
                         "mean_toward_other": float(np.mean([m["toward_other"] for m in move if m["model"] == j])),
                         "share_own_is_max": float(np.mean([m["own_is_max"] for m in move if m["model"] == j]))}
                     for j in JOURNALS},
    }
    (OUT / f"feedback_{args.mode}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (OUT / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=2))

    # 3. Side-by-side examples (titles only; abstracts stay local)
    md = [f"# Same paper, three journals ({args.mode})\n"]
    for pid in papers[:: max(1, len(papers) // 6)][:6]:
        md.append(f"## {meta.loc[pid, 'title']} ({JOURNAL_NAMES[meta.loc[pid, 'journal']]}, {meta.loc[pid, 'year']})\n")
        for r in [r for r in rows if r["paper"] == pid]:
            md.append(f"**{JOURNAL_NAMES[r['model']]}**: fit {r['fit_score']}/100. {r['what_we_value']}")
            for s, c in zip(r["suggestions"], r.get("categories", [None] * 3)):
                md.append(f"- *[{c}]* **{s['direction']}**: {s['why']}")
            md.append("")
    (OUT / f"examples_{args.mode}.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
