"""Build trajectory.csv, RULEBOOKS.md and REPORT.md from results/harness/{J}/.

    .venv/bin/python -m harness.report

Works on a finished or an interrupted run. Makes one (cached) model call: a model-written summary of
cross-journal differences and of what each board learned not to value. If that call fails (e.g. a usage
limit), the summary is marked pending. Everything else is copied verbatim from the run logs.
"""
from __future__ import annotations

import csv
import json
import re

from harness.run import OUT
from jm.data import JOURNAL_NAMES, JOURNALS
from jm.llm import LEDGER, call

OLD = OUT / "superseded_batched"
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "differences": {"type": "array", "items": {"type": "string"}},
        "learned_not_to_value": {"type": "object", "properties": {j: {"type": "array", "items": {"type": "string"}}
                                                                  for j in JOURNALS},
                                 "required": list(JOURNALS), "additionalProperties": False},
    },
    "required": ["differences", "learned_not_to_value"], "additionalProperties": False,
}
QUOTE = re.compile(r"(?<!\w)['\"‘“][^'\"’”]{3,}['\"’”](?!\w)")  # a quoted phrase, not an apostrophe
PAIR_ORDER = [("OrgSci", "MgmtSci"), ("OrgSci", "SMJ"), ("SMJ", "MgmtSci")]
RESUME = ".venv/bin/python -m harness.run --journal all && .venv/bin/python -m harness.report"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(c) -> str:
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def cell(s: str) -> str:
    return s.replace("|", "\\|")


def correct(p: float) -> float:
    return 1.0 if p > 0.5 else 0.5 if p == 0.5 else 0.0


def load(j: str) -> dict | None:
    d = OUT / j
    if not (d / "log.json").exists():
        return None
    log = json.loads((d / "log.json").read_text())
    test = json.loads((d / "test.json").read_text())["summary"] if (d / "test.json").exists() else None
    final_v = test["final_version"] if test else [r for r in log if r["accepted"]][-1]["version"]
    fin = next(r for r in log if r["version"] == final_v)
    out = {"log": log, "test": test, "final": fin["rulebook"], "final_v": final_v, "final_val": fin["val_acc"],
           "shown_v": final_v, "rejected": False}
    if not fin["rulebook"]["values"] and len(log) > 1:  # nothing beat the empty board: show the best proposal
        best = max((r for r in log if r["iteration"] > 0), key=lambda r: (r["val_acc"], r["iteration"]))
        out.update(final=best["rulebook"], shown_v=best["version"], rejected=True, shown_val=best["val_acc"])
    return out


def revisions(log: list[dict]) -> list[tuple[int, dict]]:
    """Dropped and edited rules from accepted revisions only (rejected revisions were reverted)."""
    return [(r["version"], c) for r in log if r["iteration"] > 0 and r["accepted"]
            for c in r["changelog"] if c["change"] in ("dropped", "edited")]


def fmt_revision(v: int, c: dict) -> str:
    if c["change"] == "dropped":
        return f"- v{v} **dropped**: {c['before']}\n  - *why*: {c['reason']}"
    return f"- v{v} **edited**: {c['before']}\n  - *became*: {c['after']}\n  - *why*: {c['reason']}"


def model_summary(runs: dict) -> dict | None:
    parts = []
    for j, r in runs.items():
        note = ("(NOTE: no revision beat the empty rulebook on val; these are the best-scoring REJECTED proposal's "
                "rules)\n" if r["rejected"] else "")
        rules = note + "\n".join(f"- {x}" for x in r["final"]["values"])
        flags = "\n".join(f"- {x}" for x in r["final"]["red_flags"])
        revs = "\n".join(fmt_revision(v, c) for v, c in revisions(r["log"])) or "(none)"
        parts.append(f"=== {JOURNAL_NAMES[j]} ({j}) ===\nVALUES:\n{rules}\n\nRED FLAGS:\n{flags}\n\n"
                     f"RULES DROPPED OR EDITED ALONG THE WAY (accepted revisions only):\n{revs}")
    prompt = ("Below are three rulebooks. Each was evolved by a model, from an empty start, to tell which of two "
              "same-topic abstracts a given journal published (the other appeared in one of the other two journals). "
              "For each journal you also see the rules it dropped or edited along the way, with the model's reasons.\n\n"
              + "\n\n".join(parts) +
              "\n\nWrite two things, based ONLY on the text above (no outside knowledge about these journals):\n"
              "1. `differences`: 4-6 bullets, each at most 30 words, that a reader can scan in a minute to see how "
              "the three journals' tastes differ as these rulebooks describe them. Contrast journals by name "
              "(OrgSci, SMJ, MgmtSci). One bullet may note what all three share.\n"
              "2. `learned_not_to_value`: for each journal key, 0-4 bullets (each at most 25 words) on what the "
              "board first treated as mattering and later dropped, weakened or reversed, judged from the changelog "
              "entries. Skip purely cosmetic edits and merges. Return an empty list if there is nothing substantive.")
    try:
        r = call(prompt, system="You are a careful, plain-spoken analyst of academic journals' editorial taste.",
                 model="opus", effort="medium", schema=SUMMARY_SCHEMA, tag="harness/report", retries=0)
        return r["json"]
    except Exception as e:  # e.g. the plan's usage limit; the summary stays pending
        print(f"[warn] model-written summary unavailable: {str(e)[:200]}")
        return None


def usage() -> dict:
    recs = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("tag", "").startswith("harness/") and not r.get("cached")]
    kinds = {}
    for r in recs:
        k = r["tag"].split("/")[1]
        kinds[k] = kinds.get(k, 0) + 1
    ts = [r["ts"] for r in recs]
    return {"calls": len(recs), "by_kind": kinds, "notional_usd": sum(r.get("cost_usd", 0) for r in recs),
            "wall_minutes": (max(ts) - min(ts)) / 60 if ts else 0}


def side_by_side(runs: dict, key: str) -> list[str]:
    js = list(runs)
    head = [f"{JOURNAL_NAMES[j]} (v{runs[j]['shown_v']}{', rejected †' if runs[j]['rejected'] else ''})" for j in js]
    out = ["| # | " + " | ".join(head) + " |", "|---|" + "---|" * len(js)]
    for i in range(max(len(runs[j]["final"][key]) for j in js)):
        row = [cell(runs[j]["final"][key][i]) if i < len(runs[j]["final"][key]) else "" for j in js]
        out.append(f"| {i + 1} | " + " | ".join(row) + " |")
    return out


def test_row(x: dict) -> str:
    return (f"{pct(x['v0'])} {ci(x['v0_ci'])} | {pct(x['final'])} {ci(x['final_ci'])} | "
            f"{100 * x['diff']:+.1f} {ci(x['diff_ci'])}")


def status(runs: dict) -> list[str]:
    if all(r["test"] for r in runs.values()):
        return []
    done = ", ".join(f"{j} {len(r['log']) - 1}" for j, r in runs.items())
    return [f"> **Status: interrupted.** The Claude plan hit its 5-hour usage limit at about 13:20 ET on Oct 30 "
            f"(the CLI returned HTTP 429, \"resets 3:30pm\"). Iterations completed: {done} of 6. The rulebooks below "
            f"are each journal's best *accepted* rulebook so far, chosen on clean one-pair-per-call val. **Test "
            f"scoring has not run.** To finish (cached calls replay free): `{RESUME}`", ""]


def superseded(runs: dict) -> list[str]:
    """Earlier batched scoring (about 10 pairs of J per call), with the size of the in-batch leak."""
    R = ["", "## Superseded: batched scoring (in-batch leakage)", "",
         "An earlier run (12 iterations, 40 val pairs) scored val and test about 10 pairs per call. Every pair in such a "
         "batch holds one J article, so the model can infer which style is J's from what the items share, even with no "
         "rules (in an anonymized \"Journal X\" check of the main pilot, a board with no information scored up to 82% "
         "instead of 50%). **All batched val and test numbers, and that run's accept/reject decisions, are "
         "superseded.** Its files are in `superseded_batched/` for the record.", "",
         "Leak check on the same pairs: the empty (v0) board on the 30 current val pairs, scored both ways.", "",
         "| Board | Pairs | v0 val, batched | v0 val, one pair per call | OrgSci–MgmtSci pairs: batched → one per call |",
         "|---|---|---|---|---|"]
    for j, r in runs.items():
        f = OLD / j / "log.json"
        if not f.exists():
            continue
        old, new = json.loads(f.read_text())[0]["val_probs"], r["log"][0]["val_probs"]
        ids = sorted(set(old) & set(new))
        om = [i for i in ids if "OrgSci-MgmtSci" in i]
        a = lambda d, I: pct(sum(correct(d[i]) for i in I) / len(I))
        R.append(f"| {j} | {len(ids)} | {a(old, ids)} | {a(new, ids)} | "
                 f"{f'{a(old, om)} → {a(new, om)} (n = {len(om)})' if om else '—'} |")
    R += ["", "For a board told the journal's name, batching shows no consistent inflation on these pairs: batched v0 "
          "is higher for SMJ but lower for OrgSci and MgmtSci, gaps of 1-2 pairs, within noise. The named board "
          "presumably already knows the house style, so the leak matters most for anonymized boards. One pair per "
          "call is the clean protocol either way.", "",
          "Superseded batched test results (accuracy %, batched, for the record only):", "",
          "| Board | Test pairs | v0 | Batched run's final rulebook |", "|---|---|---|---|"]
    for j in JOURNALS:
        f = OLD / j / "test.json"
        if f.exists():
            s = json.loads(f.read_text())["summary"]["all"]
            fv = json.loads(f.read_text())["summary"]["final_version"]
            R.append(f"| {j} | {s['n']} | {pct(s['v0'])} | {pct(s['final'])} (v{fv}) |")
        else:
            R.append(f"| {j} | | not run (stopped before test) | |")
    return R


def main() -> None:
    runs = {j: r for j in JOURNALS if (r := load(j))}
    n_iter = max(len(r["log"]) for r in runs.values()) - 1
    have_test = all(r["test"] for r in runs.values())

    # trajectory.csv
    with open(OUT / "trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["journal", "iteration", "val_acc", "accepted", "val_mean_p"] + [f"val_acc_vs_{k}" for k in JOURNALS])
        for j, r in runs.items():
            for x in r["log"]:
                w.writerow([j, x["iteration"], round(x["val_acc"], 4), x["accepted"], round(x["val_mean_p"], 4)]
                           + [round(x["val_by_rival"][k], 4) if k in x["val_by_rival"] else "" for k in JOURNALS])

    summ = model_summary(runs) if len(runs) == len(JOURNALS) else None
    pending = (f"*(Model-written summary pending: the summary call could not run, because the CLI is at its usage "
               f"limit. Rerun `.venv/bin/python -m harness.report` after the limit resets.)*")

    # RULEBOOKS.md
    L = ["# What each journal's board learned to value", ""] + status(runs) + [
        "Each rulebook was evolved from an empty start by a judge-reflect loop with hard-example mining on "
        "topic-matched abstract pairs, and kept only if it did at least as well as the best so far on held-out "
        "validation pairs judged one pair per call (design in [REPORT.md](REPORT.md)). **Rules and changelog "
        "entries are verbatim model output.** Sections marked *model-written summary* are a separate Claude pass over "
        "those texts. Paste-ready system prompts: `{J}/final.md`, or `harness.load.rulebook_system(J)`.", "",
        "| Journal | Rulebook | Val accuracy, empty → rulebook | Val, OrgSci–MgmtSci pairs | Test, empty → rulebook |",
        "|---|---|---|---|---|"]
    for j, r in runs.items():
        om = {"OrgSci": "MgmtSci", "MgmtSci": "OrgSci"}.get(j)
        fin = next(x for x in r["log"] if x["version"] == r["final_v"])
        omtxt = f"{pct(r['log'][0]['val_by_rival'][om])} → {pct(fin['val_by_rival'][om])}" if om else "—"
        t = r["test"]
        ttxt = f"{pct(t['all']['v0'])} → {pct(t['all']['final'])} (n = {t['all']['n']})" if t else "pending"
        L.append(f"| {j} | v{r['final_v']} | {pct(r['log'][0]['val_acc'])} → {pct(r['final_val'])} | {omtxt} | {ttxt} |")
    L += ["", "## What each journal values", ""] + side_by_side(runs, "values")
    L += ["", "## What each journal does not value / red flags", ""] + side_by_side(runs, "red_flags")
    for j, r in runs.items():
        if r["rejected"]:
            L += ["", f"† No revision beat the empty rulebook on val for {j} (empty board: {pct(r['log'][0]['val_acc'])}% "
                  f"val), so its drop-in `{j}/final.md` is the empty board. The column shows the best-scoring proposal, "
                  f"v{r['shown_v']} (val {pct(r['shown_val'])}%), which was rejected: read it as what the loop proposed, "
                  f"not as a validated rulebook."]
    quoted = [(j, x) for j, r in runs.items() for key in ("values", "red_flags") for x in r["final"][key] if QUOTE.search(x)]
    if quoted:
        L += ["", "*Automatic lint: these rules contain a quoted phrase. Most are generic stock phrases, but check that "
              "none echoes the wording of one specific training abstract:* " + "; ".join(f"{j}: \"{x}\"" for j, x in quoted)]
    L += ["", "## Differences across journals (model-written summary)", ""]
    L += [f"- {b}" for b in summ["differences"]] if summ else [pending]
    L += ["", "## What each board learned to drop or revise", ""]
    for j, r in runs.items():
        L += [f"### {JOURNAL_NAMES[j]}", ""]
        bullets = (summ or {}).get("learned_not_to_value", {}).get(j) or []
        if bullets:
            L += ["*Model-written summary:*", ""] + [f"- {b}" for b in bullets] + [""]
        n_rej = sum(1 for x in r["log"] if x["iteration"] > 0 and not x["accepted"])
        L += [f"*Verbatim changelog entries from accepted revisions* ({n_rej} of {len(r['log']) - 1} proposed "
              f"revisions were rejected on val and reverted; their changes are not listed here but are in "
              f"`{j}/changelog_v*.md`):", ""]
        L += [fmt_revision(v, c) for v, c in revisions(r["log"])] or ["- (no rule was dropped or edited in an accepted revision)"]
        L += [""]
    (OUT / "RULEBOOKS.md").write_text("\n".join(L))

    # REPORT.md
    u = usage()
    R = ["# Rulebook harness: report", ""] + status(runs) + [
         "## Design", "",
         "- **Task.** Topic-matched cross-journal pairs (`evals/pairs/cross_*`). For journal J, every pair holding one "
         "J article asks: which of these two did J publish? MgmtSci = its strategy, organizations and entrepreneurship departments.",
         "- **Rulebook.** At most 12 \"What J values\" rules and at most 8 \"does not value / red flags\" rules, one line each, "
         "general principles only. v0 is empty: only \"You are the editorial board of J.\"",
         "- **Judge.** Claude Opus 5.5 (effort low, Claude Code CLI). System prompt: \"You are the editorial board of J. "
         "Apply these rules: ...\". It gives p_A for a pair. Each pair is judged in both orders, as separate calls, and the "
         "two answers are averaged. No journal names, titles or years are shown for the abstracts.",
         "- **One pair per call for val and test.** A batch of pairs that all involve J can leak the answer (see Superseded below).",
         "- **Loop (6 iterations per journal).** (1) The current rulebook judges a pool of about 40 train pairs, half "
         "against each rival, cycling through the train set; this is batched at 10 pairs per call, because it only picks "
         "examples. (2) *Hard-example mining*: one reflection call (Opus, effort high) sees the rulebook, the 9 pool pairs "
         "with the lowest P(truth) (wrong, then least confident) and 3 confident-correct anchors, all with the truth. "
         "It returns a revised rulebook plus a changelog. (3) The revision is scored on a fixed val set of 30 pairs, 15 "
         "against each rival. (4) It is kept if val accuracy is at least the best so far minus 1 point; otherwise it is reverted.",
         "- **Accuracy** is the share of pairs where the order-averaged P(J's article) is above 0.5 (ties score 1/2).",
         "- **Test.** The final and v0 rulebooks are each scored once on every test pair involving J, in both orders, "
         "one pair per call. 95% CIs come from 5,000 pair-bootstrap resamples, paired for the difference.", "",
         "## Test accuracy: empty rulebook (v0) vs final rulebook", ""]
    if have_test:
        R += ["Accuracy in %, with 95% bootstrap CIs. One pair per call.", "",
              "| Board | Test pairs | v0 | Final | Final − v0 | Final version | Val v0 → final |",
              "|---|---|---|---|---|---|---|"]
        for j, r in runs.items():
            t = r["test"]
            R.append(f"| {j} | {t['all']['n']} | {test_row(t['all'])} | v{t['final_version']} | "
                     f"{pct(t['v0_val'])} → {pct(t['final_val'])} |")
        R += ["", "### By journal pair", "",
              "Each journal pair is judged twice, once by each journal's board (each board asks \"which one is ours?\").", "",
              "| Journal pair | Board | Test pairs | v0 | Final | Final − v0 |", "|---|---|---|---|---|---|"]
        for a, b in PAIR_ORDER:
            for j, k in ((a, b), (b, a)):
                if j in runs:
                    x = runs[j]["test"]["by_rival"][k]
                    R.append(f"| {a}–{b} | {j} | {x['n']} | {test_row(x)} |")
    else:
        R += ["**Pending.** One-pair-per-call test scoring did not start before the plan's usage limit stopped the run. "
              f"It needs about 1,440 calls (about 50 minutes at 3 concurrent calls). To finish: `{RESUME}`.", "",
              "Clean val results so far (one pair per call), best accepted rulebook vs the empty board:", "",
              "| Board | Val pairs | v0 | Current rulebook | v0 → rulebook, OrgSci–MgmtSci val pairs |",
              "|---|---|---|---|---|"]
        for j, r in runs.items():
            om = {"OrgSci": "MgmtSci", "MgmtSci": "OrgSci"}.get(j)
            fin = next(x for x in r["log"] if x["version"] == r["final_v"])
            omtxt = f"{pct(r['log'][0]['val_by_rival'][om])} → {pct(fin['val_by_rival'][om])} (n = 15)" if om else "—"
            R.append(f"| {j} | 30 | {pct(r['log'][0]['val_acc'])} | {pct(r['final_val'])} (v{r['final_v']}) | {omtxt} |")
        R += ["", "These val numbers also drove the accept/reject decisions, so they are optimistic for the kept rulebooks."]
    R += ["", "## Validation trajectory", "",
          "Val accuracy (%) of each proposed rulebook on the same 30 pairs, one pair per call. **kept** = accepted, "
          "*rev* = rejected and reverted. *OM* = accuracy on the 15 OrgSci–MgmtSci val pairs only. Mean P(truth) and "
          "accuracy against each rival are in `trajectory.csv`.", "",
          "| Iteration | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for i in range(n_iter + 1):
        cells = []
        for j, r in runs.items():
            x = r["log"][i] if i < len(r["log"]) else None
            if x is None:
                cells.append("")
                continue
            mark = "v0" if i == 0 else ("**kept**" if x["accepted"] else "*rev*")
            om = {"OrgSci": "MgmtSci", "MgmtSci": "OrgSci"}.get(j)
            omtxt = f" · OM {pct(x['val_by_rival'][om])}" if om else ""
            cells.append(f"{pct(x['val_acc'])}{omtxt} · {mark}")
        R.append(f"| {i} | " + " | ".join(cells) + " |")
    R += superseded(runs)
    R += ["", "## Drop-in use", "",
          "`results/harness/{J}/final.md` is exactly the judge's system prompt with the current final rulebook (rewritten "
          "after every accepted revision). In code:", "",
          "```python", "from harness.load import rulebook_system",
          "system = rulebook_system(\"OrgSci\")  # \"You are the editorial board of Organization Science. Apply these rules: ...\"",
          "```", "",
          "## Caveats", "",
          "- **Ceiling.** Outside the OrgSci–MgmtSci pairs, the empty rulebook is already right on nearly every pair. "
          "Each reflection sees only a few true mistakes, and the rules partly *articulate* the model's existing priors "
          "about these journals rather than discover new ones.",
          "- **Small data.** On 30 val pairs, one pair is worth 3.3 points, so many accept/reject decisions are noise. The "
          "1-point slack is below that grain (1.7 points with half-credit ties), so in practice the rule is \"keep if val ≥ "
          "best\". OM val is 15 pairs; OM test is 34 pairs. Kept rulebooks' val scores are selected, hence optimistic.",
          "- **Train pool is batched.** The pool that picks the reflector's examples is judged 10 pairs per call. That only "
          "changes which examples the reflector sees, not any reported number.",
          "- **Style and topic cues.** Pairs are matched on topic by embeddings but are not identical. House style can "
          "drive rules as much as taste: cleaned SMJ abstracts keep only the Research Summary and are shorter (median "
          "about 123 words vs about 160 for the others), and MgmtSci abstracts quantify more. Rules about wording and "
          "quantification may partly encode these cues.",
          f"- **One short run.** At most 6 iterations (cut from 10 to fit the time box, because one-pair-per-call scoring "
          f"costs about 6 seconds a call), one seed, one train order, and the same model family judges and reflects. "
          f"The rules are one draw from a stochastic process; a rerun would phrase them differently.",
          "- The reflector is told to write general principles only. That is an instruction, not a guarantee; read "
          "rules that look oddly specific with care (RULEBOOKS.md lists rules containing quoted phrases).", "",
          "## Cost and time", "",
          f"- {u['calls']} uncached CLI calls in all, including the superseded batched runs "
          f"({', '.join(f'{k} {v}' for k, v in sorted(u['by_kind'].items()))}; `val1`/`test1` = one pair per call, "
          f"`val`/`test` = superseded batched scoring). All ran on the Claude plan via `claude -p` (notional list price "
          f"${u['notional_usd']:.2f}, not billed).",
          f"- Wall time from first to last call: about {u['wall_minutes']:.0f} minutes, with at most 3 concurrent calls.",
          "", "## Reproduce", "", "```bash",
          ".venv/bin/python -m harness.run --journal all     # or --journal OrgSci | SMJ | MgmtSci; 6 iterations by default",
          ".venv/bin/python -m harness.report", "```", "",
          "Every call is disk-cached (`.cache/llm`), so a rerun is free up to where it stopped, then continues. One process "
          "never runs more than 3 calls at once, across all journals.",
          "Per-iteration files: `{J}/rulebook_v{n}.md` and `{J}/changelog_v{n}.md` (rejected proposals included, marked). "
          "Full logs with every judged probability: `{J}/log.json` (and `{J}/test.json` once test has run).", ""]
    (OUT / "REPORT.md").write_text("\n".join(R))
    print(f"wrote {OUT}/trajectory.csv, RULEBOOKS.md, REPORT.md  ({u['calls']} calls, {u['wall_minutes']:.0f} min)")


if __name__ == "__main__":
    main()
