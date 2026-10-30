"""Evolve a short, human-readable rulebook of one journal's taste.

    .venv/bin/python -m harness.run --journal OrgSci      # or --journal all (three journals, shared slots)

Task: in a topic-matched cross-journal pair, which abstract did journal J publish?
Loop, per iteration:
  1. Judge a pool of about 40 train pairs with the current rulebook (half against each other journal), batched.
  2. Hard-example mining: one reflection call sees the 9 pairs with the lowest P(truth) (wrong or least
     confident) plus 3 confident-correct anchors, and returns a revised rulebook and a changelog.
  3. Score the revision on a fixed, stratified val set of 30 pairs, ONE PAIR PER CALL; keep it if
     val >= best - 1 point.
At the end, the final and the empty (v0) rulebooks are each scored once on all of J's test pairs, one pair per call.

Why one pair per call for val and test: a batch of pairs that all involve J leaks the answer, because the model
can tell which style the items share. Batching is kept only for the train pool, which merely picks the examples
the reflector sees. Each pair is judged in both orders, as separate calls.

Every call goes through jm.llm (Claude Code CLI, disk-cached), so a rerun is free and resumes where a previous
run stopped. At most 3 calls run at once across all journals in the process. Outputs: results/harness/{J}/.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import numpy as np

from jm.data import JOURNAL_NAMES, JOURNALS, load_pairs, text
from jm.llm import call, map_parallel

OUT = Path(__file__).resolve().parents[1] / "results" / "harness"
BLOCK, POOL_BLOCKS, N_VAL, N_HARD, N_ANCHOR = 10, 2, 30, 9, 3   # pool = 2 blocks x 2 rivals = ~40 pairs
MAX_VALUES, MAX_FLAGS, SLACK = 12, 8, 0.01
EMPTY = {"values": [], "red_flags": []}
SLOTS = threading.Semaphore(3)   # max concurrent CLI calls, shared by all journals in this process

# Same as jm.journal_model.PAIR_SCHEMA; pinned here so edits elsewhere do not change cache keys mid-run.
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "p_A": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["item", "p_A"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "values": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "changelog": {"type": "array", "items": {
            "type": "object",
            "properties": {"change": {"type": "string", "enum": ["added", "dropped", "edited"]},
                           "before": {"type": "string"}, "after": {"type": "string"},
                           "reason": {"type": "string"}},
            "required": ["change", "before", "after", "reason"], "additionalProperties": False}},
    },
    "required": ["values", "red_flags", "changelog"], "additionalProperties": False,
}


# ---------------------------------------------------------------- rulebook

def render(j: str, rb: dict) -> str:
    """The rulebook as numbered markdown; empty string for the empty rulebook."""
    if not rb["values"] and not rb["red_flags"]:
        return ""
    name = JOURNAL_NAMES[j]
    lines = [f"## What {name} values"] + [f"{i}. {r}" for i, r in enumerate(rb["values"], 1)]
    lines += ["", f"## What {name} does not value / red flags"]
    lines += [f"{i}. {r}" for i, r in enumerate(rb["red_flags"], 1)]
    return "\n".join(lines)


def system(j: str, rb: dict) -> str:
    """The judge's exact system prompt."""
    md = render(j, rb)
    s = f"You are the editorial board of {JOURNAL_NAMES[j]}."
    return s + (f" Apply these rules:\n\n{md}" if md else "")


# ---------------------------------------------------------------- data

def ours(p: dict, j: str) -> str:
    return p["a_id"] if p["a_journal"] == j else p["b_id"]


def rival(p: dict, j: str) -> str:
    return p["b_journal"] if p["a_journal"] == j else p["a_journal"]


def rivals(j: str) -> list[str]:
    return [k for k in JOURNALS if k != j]


def involving(j: str, split: str) -> list[dict]:
    return sorted((p for p in load_pairs("cross", split) if j in (p["a_journal"], p["b_journal"])),
                  key=lambda p: p["pair_id"])


def blocks_of(pairs: list[dict], seed: str) -> list[list[dict]]:
    """Shuffle deterministically and cut into blocks of <= BLOCK pairs."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    k = -(-len(pairs) // BLOCK)
    return [pairs[i * len(pairs) // k:(i + 1) * len(pairs) // k] for i in range(k)]


def train_blocks(j: str) -> dict[str, list[list[dict]]]:
    """Fixed train blocks of <= 10 pairs, separately for each rival journal."""
    tr = involving(j, "train")
    return {k: blocks_of([p for p in tr if rival(p, j) == k], f"train-{j}-{k}") for k in rivals(j)}


def pool_blocks(tb: dict[str, list[list[dict]]], it: int) -> list[list[dict]]:
    """Iteration it's pool: the next POOL_BLOCKS blocks against each rival, cycling through the train set."""
    return [bs[(POOL_BLOCKS * (it - 1) + i) % len(bs)] for bs in tb.values() for i in range(POOL_BLOCKS)]


def val_set(j: str) -> list[dict]:
    """N_VAL val pairs, stratified across the two rivals (all of a rival's pairs if it has fewer than half)."""
    va = involving(j, "val")
    by = {k: [p for p in va if rival(p, j) == k] for k in rivals(j)}
    out, need = [], N_VAL
    for i, k in enumerate(sorted(by, key=lambda k: len(by[k]))):
        n = min(len(by[k]), need // (2 - i))
        out += random.Random(f"val-{j}-{k}").sample(by[k], n)
        need -= n
    return out


# ---------------------------------------------------------------- judge

def judge(j: str, rb: dict, blocks: list[list[dict]], tag: str, workers: int = 3) -> dict[str, float]:
    """P(J's own article is the one J published) per pair, averaged over both presentation orders.
    Each block is one call in A-B order and one call in B-A order (items reversed). Blocks of one pair
    give one pair per call (used for val and test)."""
    name = JOURNAL_NAMES[j]
    batches = [[(p, p["a_id"], p["b_id"]) for p in b] for b in blocks]
    batches += [[(p, p["b_id"], p["a_id"]) for p in reversed(b)] for b in blocks]
    sys_prompt = system(j, rb)

    def run(b):
        if len(b) == 1:
            _, x, y = b[0]
            prompt = (f"Below is a pair of abstracts on similar topics. One article was published in {name} and the "
                      f"other in a different leading management journal. Give p_A (as item 1): the probability "
                      f"(0-100) that Abstract A is the one {name} published.\n\n"
                      f"### Item 1\nAbstract A: {text(x)}\n\nAbstract B: {text(y)}")
        else:
            body = "\n\n".join(f"### Item {i + 1}\nAbstract A: {text(x)}\n\nAbstract B: {text(y)}"
                               for i, (_, x, y) in enumerate(b))
            prompt = (f"Each item below is a pair of abstracts on similar topics. In every pair, one article was "
                      f"published in {name} and the other in a different leading management journal. "
                      f"For each item, give p_A: the probability (0-100) that Abstract A is the one {name} "
                      f"published.\n\n{body}")
        with SLOTS:
            r = call(prompt, system=sys_prompt, model="opus", effort="low", schema=PAIR_SCHEMA,
                     tag=f"harness/{tag}/{j}")
        got = {a["item"]: a["p_A"] for a in (r["json"] or {}).get("answers", [])}
        return [(p["pair_id"], x == ours(p, j), got.get(i + 1)) for i, (p, x, _) in enumerate(b)]

    res = map_parallel(run, batches, workers=workers)
    errs = [r for r in res if isinstance(r, Exception)]
    if errs:  # fail loudly; a rerun resumes from the cache
        raise RuntimeError(f"{len(errs)} judge batch(es) failed: {errs[0]}")
    out: dict[str, list[float]] = {}
    for pid, a_is_ours, pa in (row for r in res for row in r):
        if pa is not None:
            out.setdefault(pid, []).append(pa / 100 if a_is_ours else 1 - pa / 100)
    n = sum(len(b) for b in blocks)
    if len(out) < n:
        print(f"[warn] {tag}/{j}: no answer for {n - len(out)} pair(s)", flush=True)
    return {pid: float(np.mean(v)) for pid, v in out.items()}


def correct(p: float) -> float:
    return 1.0 if p > 0.5 else 0.5 if p == 0.5 else 0.0


def acc(probs: dict[str, float], ids=None) -> float:
    ids = list(probs) if ids is None else [i for i in ids if i in probs]
    return float(np.mean([correct(probs[i]) for i in ids])) if ids else float("nan")


def mean_p(probs: dict[str, float]) -> float:
    """Mean probability on the truth: a finer signal than accuracy near the ceiling."""
    return float(np.mean(list(probs.values())))


def by_rival(probs: dict[str, float], pairs: list[dict], j: str) -> dict[str, float]:
    return {k: acc(probs, [p["pair_id"] for p in pairs if rival(p, j) == k]) for k in rivals(j)}


def bootstrap(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- reflect

def minibatch(pairs: list[dict], probs: dict[str, float], seed: str):
    """Hard examples (lowest P(truth): wrong first, then least confident) plus confident-correct anchors."""
    judged = sorted((p for p in pairs if p["pair_id"] in probs), key=lambda p: probs[p["pair_id"]])
    hard, rest = judged[:N_HARD], judged[N_HARD:]
    sure = [p for p in rest if probs[p["pair_id"]] >= 0.8] or rest
    anchors = random.Random(seed).sample(sure, min(N_ANCHOR, len(sure)))
    wrong = [p for p in hard if probs[p["pair_id"]] <= 0.5]
    shaky = [p for p in hard if probs[p["pair_id"]] > 0.5]
    return wrong, shaky, anchors


def reflect(j: str, rb: dict, pairs: list[dict], probs: dict[str, float], seed: str) -> dict:
    """One reflection call on the mined minibatch -> revised rulebook plus changelog."""
    name = JOURNAL_NAMES[j]
    wrong, shaky, anchors = minibatch(pairs, probs, seed)

    def block(ps):
        out = []
        for i, p in enumerate(ps, 1):
            a_ours = p["a_id"] == ours(p, j)
            pa = round(100 * (probs[p["pair_id"]] if a_ours else 1 - probs[p["pair_id"]]))
            truth = (f"A was published in {name}; B in another top journal on the same topic." if a_ours else
                     f"B was published in {name}; A in another top journal on the same topic.")
            out.append(f"### Pair {i}\nAbstract A: {text(p['a_id'])}\n\nAbstract B: {text(p['b_id'])}\n\n"
                       f"Board's answer: P(A is the {name} article) = {pa}%.\nTruth: {truth}")
        return "\n\n".join(out) or "(none)"

    current = render(j, rb) or ("(empty: the board has no rules yet, only the instruction "
                                f"'You are the editorial board of {name}.')")
    sys_prompt = (f"You maintain the rulebook of {name}'s editorial board. A judge model reads the rulebook "
                  f"and decides, for two abstracts on a similar topic, which one {name} published (the other "
                  f"appeared in another leading management journal). Your rulebook should capture what is "
                  f"distinctive about {name}'s taste, in rules a scholar can read and check.")
    prompt = f"""CURRENT RULEBOOK:
{current}

The board judged {len(probs)} train pairs with this rulebook. Below are the hardest ones: the {len(wrong)} it got WRONG and the {len(shaky)} it got right with the least confidence. Then {len(anchors)} it got right confidently, as anchors that a revision must not break. "Board's answer" is its probability that A is the {name} article, averaged over both presentation orders.

=== MISJUDGED ===

{block(wrong)}

=== RIGHT, BUT WITH LOW CONFIDENCE ===

{block(shaky)}

=== RIGHT AND CONFIDENT (anchors) ===

{block(anchors)}

=== TASK ===
Revise the rulebook so the board would judge pairs like these correctly and confidently.
- Diagnose what separates the {name} articles from their same-topic counterparts: the kind of question, theory, evidence, framing, scope or contribution.
- If the rulebook is empty, write a first draft. Otherwise make targeted changes: keep rules that still look right, and edit, drop or add rules only where these pairs give a reason. Do not overfit to a single pair.
- Two sections: `values` (what {name} values; at most {MAX_VALUES} rules) and `red_flags` (what {name} does not value, or what marks an abstract as belonging elsewhere; at most {MAX_FLAGS} rules). Fewer, sharper rules beat more rules; you need not fill every slot.
- Each rule is one plain line of at most 25 words that can be checked from an abstract alone. Do not number the rules.
- Rules are general principles that must transfer to new pairs on other topics. Never name, quote or paraphrase a specific paper, title, finding, setting or dataset from these examples; any example inside a rule must be generic, not drawn from one pair.
- `changelog`: one entry per rule you added, dropped or edited (unchanged rules are not listed). Use `before` for the old text (empty if added) and `after` for the new text (empty if dropped). `reason` is one line on the pattern that motivated the change."""
    with SLOTS:
        r = call(prompt, system=sys_prompt, model="opus", effort="high", schema=REFLECT_SCHEMA,
                 tag=f"harness/reflect/{j}")
    js = r["json"]
    if not js:
        raise RuntimeError(f"reflection returned no JSON: {r['text'][:300]}")
    clean = lambda xs, n: [s.strip().lstrip("0123456789.-) ").strip() for s in xs if s.strip()][:n]
    return {"values": clean(js["values"], MAX_VALUES), "red_flags": clean(js["red_flags"], MAX_FLAGS),
            "changelog": js["changelog"], "minibatch": {"wrong": [p["pair_id"] for p in wrong],
                                                        "shaky": [p["pair_id"] for p in shaky],
                                                        "anchors": [p["pair_id"] for p in anchors]}}


# ---------------------------------------------------------------- outputs

def write_rulebook(d: Path, j: str, v: int, rb: dict, status: str) -> None:
    name = JOURNAL_NAMES[j]
    body = render(j, rb) or f"*(empty: only the sentence \"You are the editorial board of {name}.\")*"
    (d / f"rulebook_v{v}.md").write_text(f"# {name} rulebook v{v}\n\n{status}\n\n{body}\n")


def write_changelog(d: Path, v: int, parent: int, status: str, note: str, changes: list[dict]) -> None:
    lines = [f"# Changelog v{v} (revision of v{parent})", "", status, "", note, ""]
    for c in changes:
        if c["change"] == "added":
            lines.append(f"- **ADDED**: {c['after']}\n  - *why*: {c['reason']}")
        elif c["change"] == "dropped":
            lines.append(f"- **DROPPED**: {c['before']}\n  - *why*: {c['reason']}")
        else:
            lines.append(f"- **EDITED**: {c['before']}\n  - *now*: {c['after']}\n  - *why*: {c['reason']}")
    (d / f"changelog_v{v}.md").write_text("\n".join(lines) + "\n")


def fmt_rivals(x: dict[str, float]) -> str:
    return ", ".join(f"vs {k} {v:.1%}" for k, v in x.items())


def main(j: str, iters: int) -> None:
    d = OUT / j
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tb = train_blocks(j)
    val = val_set(j)
    val_blocks = [[p] for p in val]          # one pair per call: no in-batch leakage
    test = involving(j, "test")

    log = []
    vp0 = judge(j, EMPTY, val_blocks, "val1")
    cur, cur_v, best = dict(EMPTY), 0, acc(vp0)
    vr0 = by_rival(vp0, val, j)
    print(f"[{j}] v0 val {best:.3f} ({fmt_rivals(vr0)})", flush=True)
    write_rulebook(d, j, 0, EMPTY, f"Seed rulebook. Val accuracy {best:.1%} ({fmt_rivals(vr0)}).")
    write_changelog(d, 0, 0, "Seed: the empty rulebook.", "", [])
    log.append({"iteration": 0, "version": 0, "parent": None, "rulebook": EMPTY, "val_acc": best,
                "val_by_rival": vr0, "val_mean_p": mean_p(vp0), "val_probs": vp0,
                "accepted": True, "best_before": None})

    for it in range(1, iters + 1):
        pb = pool_blocks(tb, it)
        pool = [p for b in pb for p in b]
        tr = judge(j, cur, pb, "train")
        new = reflect(j, cur, pool, tr, seed=f"{j}-{it}")
        rb = {"values": new["values"], "red_flags": new["red_flags"]}
        vp = judge(j, rb, val_blocks, "val1")
        va, vr = acc(vp), by_rival(vp, val, j)
        accepted = va + 1e-9 >= best - SLACK
        status = (f"**{'ACCEPTED' if accepted else 'REJECTED'}** at iteration {it}: val accuracy {va:.1%} "
                  f"({fmt_rivals(vr)}); best so far {best:.1%}, threshold {best - SLACK:.1%}. Revision of v{cur_v}.")
        mb = new["minibatch"]
        note = (f"Train pool: v{cur_v} judged {len(tr)} train pairs, accuracy {acc(tr):.1%} "
                f"({fmt_rivals(by_rival(tr, pool, j))}). Reflection saw {len(mb['wrong'])} misjudged, "
                f"{len(mb['shaky'])} low-confidence and {len(mb['anchors'])} anchor pairs.")
        write_rulebook(d, j, it, rb, status)
        write_changelog(d, it, cur_v, status, note, new["changelog"])
        log.append({"iteration": it, "version": it, "parent": cur_v, "rulebook": rb, "changelog": new["changelog"],
                    "minibatch": mb, "train_ids": [p["pair_id"] for p in pool], "train_probs": tr,
                    "train_acc": acc(tr), "val_acc": va, "val_by_rival": vr, "val_mean_p": mean_p(vp),
                    "val_probs": vp, "accepted": accepted, "best_before": best})
        print(f"[{j}] iter {it}: pool(v{cur_v}) {acc(tr):.3f} [{len(mb['wrong'])} wrong]  val(v{it}) {va:.3f} "
              f"({fmt_rivals(vr)})  best {best:.3f}  {'ACCEPT' if accepted else 'reject'}  "
              f"({len(rb['values'])}+{len(rb['red_flags'])} rules)", flush=True)
        if accepted:
            cur, cur_v, best = rb, it, max(best, va)
            (d / "final.md").write_text(system(j, cur) + "\n")   # drop-in stays valid if a run is interrupted
        (d / "log.json").write_text(json.dumps(log, indent=1))

    # drop-in: final.md is exactly the judge's system prompt (read it with harness.load.rulebook_system)
    (d / "final.md").write_text(system(j, cur) + "\n")
    final_val = next(r for r in log if r["version"] == cur_v)

    # test: touched once, at the very end
    test_blocks = [[p] for p in test]        # one pair per call
    res = {label: judge(j, r, test_blocks, "test1") for label, r in (("v0", EMPTY), ("final", cur))}

    def compare(ps):
        ids = sorted(set(res["v0"]) & set(res["final"]) & {p["pair_id"] for p in ps})
        x0 = np.array([correct(res["v0"][i]) for i in ids])
        x1 = np.array([correct(res["final"][i]) for i in ids])
        return {"n": len(ids), "v0": float(x0.mean()), "v0_ci": bootstrap(x0),
                "final": float(x1.mean()), "final_ci": bootstrap(x1),
                "diff": float((x1 - x0).mean()), "diff_ci": bootstrap(x1 - x0)}

    summary = {"journal": j, "final_version": cur_v, "n_val": len(val),
               "v0_val": log[0]["val_acc"], "v0_val_by_rival": log[0]["val_by_rival"],
               "final_val": final_val["val_acc"], "final_val_by_rival": final_val["val_by_rival"],
               "all": compare(test), "by_rival": {k: compare([p for p in test if rival(p, j) == k]) for k in rivals(j)},
               "seconds": round(time.time() - t0, 1)}
    (d / "test.json").write_text(json.dumps({"summary": summary, "probs": res}, indent=1))
    a = summary["all"]
    print(f"[{j}] TEST v0 {a['v0']:.3f}  final(v{cur_v}) {a['final']:.3f}  diff {a['diff']:+.3f} {a['diff_ci']}  "
          + "  ".join(f"vs {k}: {x['v0']:.3f}->{x['final']:.3f}" for k, x in summary["by_rival"].items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True, choices=[*JOURNALS, "all"])
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    js = JOURNALS if a.journal == "all" else [a.journal]
    threads = [threading.Thread(target=main, args=(j, a.iters)) for j in js]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
