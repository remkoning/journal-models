"""Robustness checks for the feedback categories (exhibit C).

1. Inter-classifier agreement: a second blind classifier (sonnet) re-labels all suggestions with
   the same taxonomy. Reports Cohen's kappa against the original (opus) labels, and the mix
   under the second labeler.
2. Exploratory refinement, chosen AFTER seeing the data: split "theory" into verbal theory and
   "formal" (a mathematical or formal model), then re-classify (opus).

    .venv/bin/python -m scripts.feedback_robustness
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

from jm.data import JOURNALS
from jm.llm import call, map_parallel
from scripts.run_feedback import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feedback"

CATS_FORMAL = {**{k: v for k, v in CATEGORIES.items() if k != "theory"},
               "theory": "develop or extend verbal theory: new constructs, propositions, theoretical framing",
               "formal": "build a formal or mathematical model (game theory, analytical model, comparative statics)"}


def classify(suggestions, cats, model, tag, batch=30):
    schema = {"type": "object", "properties": {"answers": {"type": "array", "items": {
        "type": "object", "properties": {"item": {"type": "integer"}, "category": {"type": "string", "enum": list(cats)}},
        "required": ["item", "category"], "additionalProperties": False}}},
        "required": ["answers"], "additionalProperties": False}
    order = list(range(len(suggestions)))
    random.Random(7).shuffle(order)
    desc = "\n".join(f"- {k}: {v}" for k, v in cats.items())

    def run(chunk):
        body = "\n".join(f"{k + 1}. {suggestions[i]['direction']}: {suggestions[i]['why']}" for k, i in enumerate(chunk))
        r = call(f"Assign each suggestion to exactly one category, choosing the best fit.\n\nCategories:\n{desc}\n\n"
                 f"Suggestions:\n{body}", system="You are an expert in management research methods. "
                 "Classify research suggestions precisely.", model=model, effort="low", schema=schema, tag=tag)
        got = {a["item"]: a["category"] for a in (r["json"] or {}).get("answers", [])}
        return {i: got.get(k + 1) for k, i in enumerate(chunk)}

    out = {}
    for res in map_parallel(run, [order[i:i + batch] for i in range(0, len(order), batch)], workers=3):
        if not isinstance(res, Exception):
            out.update(res)
    return [out.get(i) for i in range(len(suggestions))]


def mix(rows, labels, cats):
    by = {j: Counter() for j in JOURNALS}
    for (j, _), c in zip(rows, labels):
        if c:
            by[j][c] += 1
    share = {j: {c: round(by[j][c] / max(1, sum(by[j].values())), 3) for c in cats} for j in JOURNALS}
    chi = chi2_contingency([[by[j][c] for c in cats] for j in JOURNALS])
    return share, float(chi.statistic), float(chi.pvalue)


def main():
    rows = [json.loads(l) for l in (OUT / "feedback_contrast.jsonl").open()]
    sugg, meta, orig = [], [], []
    for r in rows:
        for s, c in zip(r["suggestions"], r.get("categories", [None] * 3)):
            sugg.append(s); meta.append((r["model"], r["paper"])); orig.append(c)
    alt = classify(sugg, CATEGORIES, "sonnet", "feedback/robust/sonnet")
    ok = [i for i in range(len(sugg)) if orig[i] and alt[i]]
    kappa = cohen_kappa_score([orig[i] for i in ok], [alt[i] for i in ok])
    agree = sum(orig[i] == alt[i] for i in ok) / len(ok)
    formal = classify(sugg, CATS_FORMAL, "opus", "feedback/robust/formal")
    res = {
        "n_suggestions": len(sugg),
        "second_classifier": {"model": "sonnet", "cohen_kappa_vs_opus": round(kappa, 3),
                              "raw_agreement": round(agree, 3)},
        "mix_second_classifier": dict(zip(["share", "chi2", "p"], mix(meta, alt, CATEGORIES))),
        "exploratory_formal_split": dict(zip(["share", "chi2", "p"], mix(meta, formal, CATS_FORMAL))),
    }
    (OUT / "robustness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
