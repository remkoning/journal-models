# Rulebook harness: report

## Design

- **Task.** Topic-matched cross-journal pairs (`evals/pairs/cross_*`). For journal J, every pair holding one J article asks: which of these two did J publish? MgmtSci = its strategy, organizations and entrepreneurship departments.
- **Rulebook.** At most 12 "What J values" rules and at most 8 "does not value / red flags" rules, one line each, general principles only. v0 is empty: only "You are the editorial board of J."
- **Judge.** Claude Opus 5.5 (effort low, Claude Code CLI). System prompt: "You are the editorial board of J. Apply these rules: ...". It gives p_A for a pair. Each pair is judged in both orders, as separate calls, and the two answers are averaged. No journal names, titles or years are shown for the abstracts.
- **One pair per call for val and test.** A batch of pairs that all involve J can leak the answer (see Superseded below).
- **Loop (6 iterations per journal).** (1) The current rulebook judges a pool of about 40 train pairs, half against each rival, cycling through the train set; this is batched at 10 pairs per call, because it only picks examples. (2) *Hard-example mining*: one reflection call (Opus, effort high) sees the rulebook, the 9 pool pairs with the lowest P(truth) (wrong, then least confident) and 3 confident-correct anchors, all with the truth. It returns a revised rulebook plus a changelog. (3) The revision is scored on a fixed val set of 30 pairs, 15 against each rival. (4) It is kept if val accuracy is at least the best so far minus 1 point; otherwise it is reverted.
- **Accuracy** is the share of pairs where the order-averaged P(J's article) is above 0.5 (ties score 1/2).
- **Test.** The final and v0 rulebooks are each scored once on every test pair involving J, in both orders, one pair per call. 95% CIs come from 5,000 pair-bootstrap resamples, paired for the difference.

## Test accuracy: empty rulebook (v0) vs final rulebook

Accuracy in %, with 95% bootstrap CIs. One pair per call.

| Board | Test pairs | v0 | Final | Final − v0 | Final version | Val v0 → final |
|---|---|---|---|---|---|---|
| OrgSci | 147 | 91.8 [87.1, 95.9] | 89.1 [83.7, 93.9] | -2.7 [-6.1, 0.0] | v3 | 80.0 → 88.3 |
| SMJ | 146 | 92.5 [87.7, 96.6] | 85.6 [80.1, 91.1] | -6.8 [-12.3, -1.4] | v5 | 91.7 → 93.3 |
| MgmtSci | 67 | 98.5 [95.5, 100.0] | 95.5 [89.6, 100.0] | -3.0 [-7.5, 0.0] | v4 | 96.7 → 96.7 |

### By journal pair

Each journal pair is judged twice, once by each journal's board (each board asks "which one is ours?").

| Journal pair | Board | Test pairs | v0 | Final | Final − v0 |
|---|---|---|---|---|---|
| OrgSci–MgmtSci | OrgSci | 34 | 79.4 [64.7, 91.2] | 79.4 [64.7, 91.2] | +0.0 [-8.8, 8.8] |
| OrgSci–MgmtSci | MgmtSci | 34 | 97.1 [91.2, 100.0] | 91.2 [79.4, 100.0] | -5.9 [-14.7, 0.0] |
| OrgSci–SMJ | OrgSci | 113 | 95.6 [91.2, 99.1] | 92.0 [86.7, 96.5] | -3.5 [-7.1, -0.9] |
| OrgSci–SMJ | SMJ | 113 | 91.2 [85.8, 96.5] | 85.0 [77.9, 91.2] | -6.2 [-13.3, 0.0] |
| SMJ–MgmtSci | SMJ | 33 | 97.0 [90.9, 100.0] | 87.9 [75.8, 97.0] | -9.1 [-21.2, 0.0] |
| SMJ–MgmtSci | MgmtSci | 33 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.0 [0.0, 0.0] |

## Validation trajectory

Val accuracy (%) of each proposed rulebook on the same 30 pairs, one pair per call. **kept** = accepted, *rev* = rejected and reverted. *OM* = accuracy on the 15 OrgSci–MgmtSci val pairs only. Mean P(truth) and accuracy against each rival are in `trajectory.csv`.

| Iteration | OrgSci | SMJ | MgmtSci |
|---|---|---|---|
| 0 | 80.0 · OM 60.0 · v0 | 91.7 · v0 | 96.7 · OM 93.3 · v0 |
| 1 | 88.3 · OM 76.7 · **kept** | 93.3 · **kept** | 93.3 · OM 93.3 · *rev* |
| 2 | 83.3 · OM 66.7 · *rev* | 88.3 · *rev* | 91.7 · OM 90.0 · *rev* |
| 3 | 88.3 · OM 76.7 · **kept** | 90.0 · *rev* | 96.7 · OM 93.3 · **kept** |
| 4 | 86.7 · OM 73.3 · *rev* | 93.3 · **kept** | 96.7 · OM 93.3 · **kept** |
| 5 | 83.3 · OM 66.7 · *rev* | 93.3 · **kept** | 90.0 · OM 86.7 · *rev* |
| 6 | 80.0 · OM 63.3 · *rev* | 90.0 · *rev* | 93.3 · OM 86.7 · *rev* |

## Superseded: batched scoring (in-batch leakage)

An earlier run (12 iterations, 40 val pairs) scored val and test about 10 pairs per call. Every pair in such a batch holds one J article, so the model can infer which style is J's from what the items share, even with no rules (in an anonymized "Journal X" check of the main pilot, a board with no information scored up to 82% instead of 50%). **All batched val and test numbers, and that run's accept/reject decisions, are superseded.** Its files are in `superseded_batched/` for the record.

Leak check on the same pairs: the empty (v0) board on the 30 current val pairs, scored both ways.

| Board | Pairs | v0 val, batched | v0 val, one pair per call | OrgSci–MgmtSci pairs: batched → one per call |
|---|---|---|---|---|
| OrgSci | 30 | 76.7 | 80.0 | 53.3 → 60.0 (n = 15) |
| SMJ | 30 | 96.7 | 91.7 | — |
| MgmtSci | 30 | 90.0 | 96.7 | 80.0 → 93.3 (n = 15) |

For a board told the journal's name, batching shows no consistent inflation on these pairs: batched v0 is higher for SMJ but lower for OrgSci and MgmtSci, gaps of 1-2 pairs, within noise. The named board presumably already knows the house style, so the leak matters most for anonymized boards. One pair per call is the clean protocol either way.

Superseded batched test results (accuracy %, batched, for the record only):

| Board | Test pairs | v0 | Batched run's final rulebook |
|---|---|---|---|
| OrgSci | | not run (stopped before test) | |
| SMJ | 146 | 97.6 | 94.2 (v12) |
| MgmtSci | 67 | 94.0 | 94.0 (v12) |

## Drop-in use

`results/harness/{J}/final.md` is exactly the judge's system prompt with the current final rulebook (rewritten after every accepted revision). In code:

```python
from harness.load import rulebook_system
system = rulebook_system("OrgSci")  # "You are the editorial board of Organization Science. Apply these rules: ..."
```

## Caveats

- **Ceiling.** Outside the OrgSci–MgmtSci pairs, the empty rulebook is already right on nearly every pair. Each reflection sees only a few true mistakes, and the rules partly *articulate* the model's existing priors about these journals rather than discover new ones.
- **Small data.** On 30 val pairs, one pair is worth 3.3 points, so many accept/reject decisions are noise. The 1-point slack is below that grain (1.7 points with half-credit ties), so in practice the rule is "keep if val ≥ best". OM val is 15 pairs; OM test is 34 pairs. Kept rulebooks' val scores are selected, hence optimistic.
- **Train pool is batched.** The pool that picks the reflector's examples is judged 10 pairs per call. That only changes which examples the reflector sees, not any reported number.
- **Style and topic cues.** Pairs are matched on topic by embeddings but are not identical. House style can drive rules as much as taste: cleaned SMJ abstracts keep only the Research Summary and are shorter (median about 123 words vs about 160 for the others), and MgmtSci abstracts quantify more. Rules about wording and quantification may partly encode these cues.
- **One short run.** At most 6 iterations (cut from 10 to fit the time box, because one-pair-per-call scoring costs about 6 seconds a call), one seed, one train order, and the same model family judges and reflects. The rules are one draw from a stochastic process; a rerun would phrase them differently.
- The reflector is told to write general principles only. That is an instruction, not a guarantee; read rules that look oddly specific with care (RULEBOOKS.md lists rules containing quoted phrases).

## Cost and time

- 3549 uncached CLI calls in all, including the superseded batched runs (reflect 50, report 1, test 88, test1 1440, train 356, train1 12, train2 4, val 338, val1 1260; `val1`/`test1` = one pair per call, `val`/`test` = superseded batched scoring). All ran on the Claude plan via `claude -p` (notional list price $109.05, not billed).
- Wall time from first to last call: about 271 minutes, with at most 3 concurrent calls.

## Reproduce

```bash
.venv/bin/python -m harness.run --journal all     # or --journal OrgSci | SMJ | MgmtSci; 6 iterations by default
.venv/bin/python -m harness.report
```

Every call is disk-cached (`.cache/llm`), so a rerun is free up to where it stopped, then continues. One process never runs more than 3 calls at once, across all journals.
Per-iteration files: `{J}/rulebook_v{n}.md` and `{J}/changelog_v{n}.md` (rejected proposals included, marked). Full logs with every judged probability: `{J}/log.json` (and `{J}/test.json` once test has run).
