"""In-context journal-models: Claude conditioned on one journal's own abstracts.

A journal-model is a base model plus a "taste set": 100 abstracts the journal published.
These are placed in the system prompt. Three modes:

    plain       no abstracts, only "you are the editorial board of J" (the before condition)
    incontext   J's 100 abstracts (the v0 journal-model)
    anonymous=True renames the journal "Journal X", so its taste can only come from the abstracts.

    contrast    J's 100 abstracts plus 100 abstracts from the other two journals, labelled
                "published elsewhere" (the v0 journal-model, chosen on the validation pilot)
    rulebook    the readable rules evolved by the self-improving harness (harness/), no abstracts

Usage:
    jmod = JournalModel("OrgSci", mode="incontext")
    jmod.which_is_ours(pairs)          # fit test: P(paper a is J's) for topic-matched pairs
"""
from __future__ import annotations

import random

from jm.data import JOURNALS, JOURNAL_NAMES, load_works, text
from jm.llm import call, map_parallel

TASTE_N = 100


def taste_ids(journal: str, n: int = TASTE_N, seed: int = 0, exclude: set | None = None) -> list[str]:
    """n abstracts from J's train split, stratified across years, deterministic given seed."""
    df = load_works(core_only=True, journals=(journal,), split="train")
    if exclude:
        df = df[~df.openalex_id.isin(exclude)]
    df = df.sort_values("openalex_id")
    rng = random.Random(f"{journal}-{seed}")
    by_year = {y: list(g.openalex_id) for y, g in df.groupby("year")}
    for ids in by_year.values():
        rng.shuffle(ids)
    out, years = [], sorted(by_year)
    while len(out) < min(n, len(df)):           # round-robin over years
        for y in years:
            if by_year[y] and len(out) < n:
                out.append(by_year[y].pop())
    return out


def _block(ids: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {text(x)}" for i, x in enumerate(ids))


def system_prompt(journal: str, ids: list[str] | None, contrast_ids: list[str] | None = None,
                  anonymous: bool = False) -> str:
    name = "Journal X" if anonymous else JOURNAL_NAMES[journal]
    s = (f"You are the editorial board of {name}. You judge research the way this journal's editors "
         f"and reviewers do: which questions it values, what theory and evidence it expects, and what "
         f"counts as a contribution here.")
    if ids:
        s += (f"\n\nBelow are {len(ids)} abstracts of articles that {name} published. Together they "
              f"define this journal's taste. Study them closely.\n\n=== PUBLISHED IN {name.upper()} ===\n\n"
              + _block(ids))
    if contrast_ids:
        s += (f"\n\n=== PUBLISHED ELSEWHERE (other leading management journals, NOT {name}) ===\n\n"
              + _block(contrast_ids)
              + f"\n\nUse the contrast between the two sets to understand what is distinctive about {name}.")
    return s


SCORE_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"},
                       "score": {"type": "integer", "minimum": 0, "maximum": 100},
                       "why": {"type": "string"}},
        "required": ["item", "score", "why"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "fit_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "what_we_value": {"type": "string"},
        "suggestions": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
            "type": "object",
            "properties": {"direction": {"type": "string"}, "why": {"type": "string"}},
            "required": ["direction", "why"], "additionalProperties": False}},
        "revised_abstract": {"type": "string"},
    },
    "required": ["fit_score", "what_we_value", "suggestions", "revised_abstract"],
    "additionalProperties": False,
}

def rulebook_system(journal: str) -> str:
    """System prompt of the self-improving rulebook harness (results/harness/{J}/final.md)."""
    try:
        from harness.load import rulebook_system as _rs
        return _rs(journal)
    except ImportError:
        from pathlib import Path
        rules = (Path(__file__).resolve().parents[1] / "results" / "harness" / journal / "final.md").read_text()
        return f"You are the editorial board of {JOURNAL_NAMES[journal]}. Apply these rules:\n\n{rules}"


PAIR_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "p_A": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["item", "p_A"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}


class JournalModel:
    def __init__(self, journal: str, mode: str = "incontext", n: int = TASTE_N, seed: int = 0,
                 model: str = "opus", effort: str | None = "low", ids: list[str] | None = None,
                 contrast_n: int | None = None, anonymous: bool = False):
        assert journal in JOURNALS and mode in ("plain", "incontext", "contrast", "rulebook")
        self.journal, self.mode, self.model, self.effort = journal, mode, model, effort
        self.anonymous = anonymous   # "Journal X": all taste must come from the abstracts, none from the name
        self.ids = [] if mode in ("plain", "rulebook") else (ids if ids is not None else taste_ids(journal, n, seed))
        self.contrast_ids = []
        if mode == "contrast":
            others = [j for j in JOURNALS if j != journal]
            per = (contrast_n or len(self.ids)) // len(others)   # deterministic, so fixed across runs
            self.contrast_ids = [x for j in others for x in taste_ids(j, per, seed)]
        self.system = (rulebook_system(journal) if mode == "rulebook"
                       else system_prompt(journal, self.ids, self.contrast_ids, anonymous))

    @property
    def name(self) -> str:
        return "Journal X" if self.anonymous else JOURNAL_NAMES[self.journal]

    def which_is_ours(self, pairs: list[dict], batch: int = 10, workers: int = 4, seed: int = 0,
                      tag: str = "fit") -> dict[str, float]:
        """For each pair, P(this journal published paper a), averaged over both presentation orders."""
        items = []
        for p in pairs:
            items.append((p["pair_id"], p["a_id"], p["b_id"], "ab"))
            items.append((p["pair_id"], p["b_id"], p["a_id"], "ba"))
        random.Random(seed).shuffle(items)
        batches = [items[i:i + batch] for i in range(0, len(items), batch)]

        def run(b):
            body = "\n\n".join(f"### Item {k + 1}\nAbstract A: {text(x)}\n\nAbstract B: {text(y)}"
                               for k, (_, x, y, _) in enumerate(b))
            prompt = (f"Each item below is a pair of abstracts on similar topics. In every pair, one article was "
                      f"published in {self.name} and the other in a different leading management journal. "
                      f"For each item, give p_A: the probability (0-100) that Abstract A is the one {self.name} "
                      f"published.\n\n{body}")
            r = call(prompt, system=self.system, model=self.model, effort=self.effort,
                     schema=PAIR_SCHEMA, tag=f"{tag}/{self.journal}/{self.mode}{'/anon' if self.anonymous else ''}")
            got = {a["item"]: a["p_A"] for a in (r["json"] or {}).get("answers", [])}
            return [(pid, order, got.get(k + 1)) for k, (pid, _, _, order) in enumerate(b)]

        out: dict[str, list[float]] = {}
        for res in map_parallel(run, batches, workers=workers):
            if isinstance(res, Exception):
                print(f"[warn] batch failed: {res}")
                continue
            for pid, order, pa in res:
                if pa is None:
                    continue
                out.setdefault(pid, []).append(pa / 100 if order == "ab" else 1 - pa / 100)
        return {pid: sum(v) / len(v) for pid, v in out.items()}

    def score_prompt(self, ids: list[str]) -> str:
        body = "\n\n".join(f"### Item {k + 1}\n{text(x)}" for k, x in enumerate(ids))
        return (f"For each abstract below, how likely is it that {self.name} would publish this work? "
                f"Give a score from 0 to 100 (50 = borderline for this journal), and in `why` a short phrase "
                f"(at most 15 words) naming what drives your judgment.\n\n{body}")

    def score(self, ids: list[str], batch: int = 15, workers: int = 4, seed: int = 0,
              tag: str = "score") -> dict[str, dict]:
        """Fit score 0-100 plus a short reason for each abstract. Order within batches is shuffled."""
        ids = list(ids)
        random.Random(seed).shuffle(ids)
        batches = [ids[i:i + batch] for i in range(0, len(ids), batch)]

        def run(b):
            r = call(self.score_prompt(b), system=self.system, model=self.model, effort=self.effort,
                     schema=SCORE_SCHEMA, tag=f"{tag}/{self.journal}/{self.mode}")
            got = {a["item"]: a for a in (r["json"] or {}).get("answers", [])}
            return [(x, got.get(k + 1)) for k, x in enumerate(b)]

        out = {}
        for res in map_parallel(run, batches, workers=workers):
            if isinstance(res, Exception):
                print(f"[warn] batch failed: {res}")
                continue
            for x, a in res:
                if a is not None:
                    out[x] = {"score": a["score"], "why": a["why"]}
        return out

    def feedback(self, openalex_id: str, tag: str = "feedback") -> dict:
        """Journal-specific review of one abstract: fit, what we value, 3 next steps, revised abstract."""
        prompt = (f"An author is considering {self.name} for the work described below.\n\n"
                  f"ABSTRACT:\n{text(openalex_id)}\n\n"
                  f"As {self.name}'s editorial board, give: (1) fit_score, 0-100, for how likely {self.name} "
                  f"would publish this work as it stands; (2) what_we_value, one sentence on what this journal "
                  f"would value or question about it; (3) exactly three suggestions for where to take the research "
                  f"next so it makes a bigger contribution to this journal's community (each a concrete "
                  f"direction plus one-sentence why); (4) revised_abstract, the abstract rewritten (at most 180 "
                  f"words) as the paper would read after following your suggestions. Do not invent results; "
                  f"describe planned analyses as such.")
        r = call(prompt, system=self.system, model=self.model, effort=self.effort,
                 schema=FEEDBACK_SCHEMA, max_tokens=3000, tag=f"{tag}/{self.journal}/{self.mode}")
        return r["json"]
