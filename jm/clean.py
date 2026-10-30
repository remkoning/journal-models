"""Abstract cleaning: remove journal giveaways (INFORMS boilerplate, MgmtSci 'accepted by' lines, SMJ summary
headers, copyright/publisher text) so that EVALUATE cannot identify the journal from formatting.

Main entry points: clean_abstract(raw, journal) -> (clean, ms_department) ; normalize_length(clean, 180).
AUDIT_PATTERNS lists the giveaway regexes whose residual rate the data card reports.
"""
from __future__ import annotations

import html
import re

# ---------------------------------------------------------------- giveaway patterns (audited)
AUDIT_PATTERNS = {
    "accepted_by": r"accepted by",
    "funding_label": r"\bFunding\s*:",
    "supplemental": r"Supplemental Material|online appendix|internet appendix|e-companion|supplemental material|"
                    r"data files (?:are|is) available|replication (?:files|package)",
    "history_label": r"\bHistory\s*:",
    "copyright": r"©|Copyright\s*(?:©|\(c\)|\d{4})|John Wiley|All rights reserved|Creative Commons|"
                 r"published by INFORMS|on behalf of (?:the )?Strategic Management Society",
    "smj_headers": r"R\s?esearch\s+summary|Managerial\s+(?:summary|abstract)",
    "abstract_prefix": r"^\s*Abstract\b",
    "journal_name": r"Management Science|Organization Science|Strategic Management Journal|\bSMJ\b",
    "publisher_doi": r"doi\.org/10\.(?:1287|1002)/|10\.1287/(?:mnsc|orsc)",
}
AUDIT_RE = {k: re.compile(v, re.I if k not in ("journal_name",) else 0) for k, v in AUDIT_PATTERNS.items()}

# Labelled INFORMS tail sections; everything from the first label onward is boilerplate.
TAIL_LABELS = re.compile(
    r"\b(?:Funding|Supplemental Material|History|Open Access Statement|Conflict of Interest|"
    r"Conflicts? of Interest Statement|Data Ethics (?:&|and) Reproducibility Note|Data Availability(?: Statement)?|"
    r"Disclaimer|Acknowledgments?)\s*:", re.I)
# MgmtSci department line, e.g. "This paper was accepted by Toby Stuart, entrepreneurship and innovation."
ACCEPTED_RE = re.compile(
    r"(?:This|The) (?:paper|article|manuscript|work|research|study) (?:was|has been|is|had been) "
    r"(?:accepted|handled)(?: for publication)? by\s+(?P<who>(?:(?![a-z]{2}\.\s)(?!\s(?:for|in) the\s)[^,]){2,80}?),"
    r"\s*(?P<dept>[^.]{2,160}?)\s*(?:\.|$)", re.I)
ACCEPTED_SPECIAL_RE = re.compile(
    r"(?:This|The) (?:paper|article|manuscript|work|research|study) (?:was|has been|is|had been) "
    r"(?:accepted|handled)(?: for publication)? (?:by\s+(?P<who>.{2,80}?)\s+)?(?:for|in|as part of) "
    r"(?:the|a)\s+(?P<dept>[^.]*?(?:special|virtual)\s+(?:issue|section)[^.]*?)\s*\.", re.I)
ACCEPTED_ANY_RE = re.compile(r"(?:This|The) (?:paper|article|manuscript|work|research|study) (?:was|has been|is|had been) "
                             r"(?:accepted|handled)[^.]*(?:\.\s*[A-Z]\.)*[^.]*\.?", re.I)
SUPP_SENT_RE = re.compile(
    r"[^.]*\b(?:online appendix|internet appendix|e-companion|supplemental material|supplementary material|"
    r"data files|data and (?:the )?online appendix|replication (?:files|package|materials?))\b[^.]*?"
    r"\b(?:is|are) available\b[^.]*?(?:https?://\S+|doi\.org/\S+)?\s*\.?(?=\s|$)", re.I)
COPYRIGHT_TAIL_RE = re.compile(
    r"\s*(?:Copyright\s*)?(?:©|\(c\)\s*(?=\d{4}))\s*\d{4}.*$|\s*Copyright\s+\d{4}.*$|"
    r"\s*©.*$|\s*Published by INFORMS.*$|\s*This (?:article|work) is licensed under a Creative Commons.*$", re.I | re.S)
PUB_DOI_RE = re.compile(r"(?:\b(?:available\s+)?(?:at|from)\s+)?(?:https?://)?(?:dx\.)?doi\.org/10\.(?:1287|1002)/\S+|"
                        r"\b10\.1287/(?:mnsc|orsc)\S*", re.I)
JOURNAL_NAME_RE = re.compile(r"\b(?:Management Science|Organization Science|Strategic Management Journal|SMJ)\b")
_SUPP_WORDS = re.compile(r"\b(?:online appendix|internet appendix|e-companion|electronic companion|supplemental material|"
                         r"supplementary material|data files|replication (?:files|package|materials?)|online supplement)\b", re.I)


def _is_boiler_sentence(x: str) -> bool:
    if re.search(r"(?:dx\.)?doi\.org/10\.(?:1287|1002)/|\b10\.1287/(?:mnsc|orsc)", x, re.I):
        return True
    return bool(_SUPP_WORDS.search(x) and re.search(r"\bavailable\b", x, re.I) and len(x.split()) < 40)


SMJ_RS_RE = re.compile(r"R\s?esearch\s+summary\s*[:.]?\s*", re.I)
SMJ_MS_RE = re.compile(r"Managerial\s+(?:summary|abstract)\s*[:.]?\s*", re.I)


def _ws(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)                 # stray JATS/HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.,;:?!)\]])", r"\1", s)      # "news ." -> "news."
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return s


DEPT_FIX = {  # spelling variants only; substantive names are kept as written
    "organization": "organizations", "information system": "information systems",
    "information systems division": "information systems", "finance department": "finance",
    "behavioural economics and decision analysis": "behavioral economics and decision analysis",
    "behavioral economics and decisions analysis": "behavioral economics and decision analysis",
    "behavioral economics and decision analysis–fast track": "behavioral economics and decision analysis",
    "stochastic models & simulation": "stochastic models and simulation", "big data and analytics": "big data analytics",
    "revenue management and pricing department": "revenue management and pricing",
}


def parse_department(raw: str) -> tuple[str | None, str | None]:
    """Return (department, editor) from a MgmtSci 'accepted by' sentence, before stripping."""
    m = ACCEPTED_RE.search(raw)
    if m:
        dept = m.group("dept").strip()
        m2 = re.match(r"^[A-Z][\w\-’'.]+,\s*(.+)$", dept)   # "accepted by Adams, Renee, finance"
        if m2:
            dept = m2.group(1)
        dept = DEPT_FIX.get(dept.lower(), dept.lower())
        # 'accepted by X for the special issue on Y' can also contain a comma; prefer special-issue parse
        if re.search(r"special (?:issue|section)", dept):
            return "special issue/section: " + re.sub(r"^(?:the|a)\s+", "", dept), m.group("who").strip()
        return dept, m.group("who").strip()
    m = ACCEPTED_SPECIAL_RE.search(raw)
    if m:
        return "special issue/section: " + m.group("dept").strip().lower(), (m.group("who") or "").strip() or None
    return None, None


def _strip_accepted(s: str) -> str:
    for rx in (ACCEPTED_RE, ACCEPTED_SPECIAL_RE):
        s = rx.sub(" ", s)
    # any residual "accepted by ..." sentence (names with initials) through end-of-sentence
    m = ACCEPTED_ANY_RE.search(s)
    if m:
        s = s[:m.start()] + " " + s[m.end():]
    return s


def smj_research_summary(raw: str) -> str:
    """Keep only the Research Summary of an SMJ abstract (drops 'Abstract', headers and the Managerial Summary)."""
    s = raw
    rs = SMJ_RS_RE.search(s)
    if rs:
        s = s[rs.end():]
    ms = SMJ_MS_RE.search(s)
    if ms:
        s = s[:ms.start()]
    return s


def clean_abstract(raw: str, journal: str) -> tuple[str, str | None]:
    s = _ws(raw or "")
    dept = None
    if journal == "MgmtSci":
        dept, _ = parse_department(s)
    if journal == "SMJ" or (SMJ_RS_RE.search(s) and SMJ_MS_RE.search(s)):   # also Wiley GSJ/SEJ in DEVELOP futures
        s = smj_research_summary(s)
    s = re.sub(r"^\s*Abstract\s*[:.]?\s*", "", s)
    s = _strip_accepted(s)
    m = TAIL_LABELS.search(s)
    if m and m.start() > 0.5 * len(s):     # labelled boilerplate lives in the tail; don't cut real text
        s = s[:m.start()]
    elif m:
        s = TAIL_LABELS.sub(" ", s)
    s = " ".join(x for x in split_sentences(s) if not _is_boiler_sentence(x))
    s = SUPP_SENT_RE.sub(" ", s)
    s = PUB_DOI_RE.sub(" ", s)
    s = COPYRIGHT_TAIL_RE.sub("", s)
    s = JOURNAL_NAME_RE.sub("[JOURNAL]", s)
    s = _ws(s)
    if s and s[-1] not in ".?!\"”')":
        s = s.rstrip(" ,;:") + "."
    return s, dept


_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))(?<!\b[A-Z]\.)(?<!\bal\.)(?<!\bvs\.)(?<!e\.g\.)(?<!i\.e\.)(?<!\bcf\.)\s+(?=[A-Z0-9“\"(‘'])")


def split_sentences(s: str) -> list[str]:
    return [x for x in _SENT_SPLIT.split(s) if x.strip()]


def normalize_length(s: str, max_words: int = 180) -> str:
    """Truncate at a sentence boundary to <= max_words words (hard cut only if the first sentence is longer)."""
    if len(s.split()) <= max_words:
        return s
    out, n = [], 0
    for sent in split_sentences(s):
        k = len(sent.split())
        if n + k > max_words:
            break
        out.append(sent)
        n += k
    if not out:
        return " ".join(s.split()[:max_words])
    return " ".join(out)


# ---------------------------------------------------------------- non-research filters
NONRESEARCH_TITLE = re.compile(
    r"^\s*(?:erratum|errata|corrigend|correction|retraction|retracted|expression of concern|editorial|"
    r"editor[’']?s?\b|from the editors?|introduction to (?:the )?(?:special|virtual)|introduction:|call for papers|"
    r"comments?\b|a comment|reply\b|a reply|response\b|rejoinder|commentary|in memoriam|obituary|thank you|"
    r"reviewers?\b|table of contents|issue information|front matter|back matter|masthead|announcement|"
    r"award|best paper|editorial board)"
    r"|\b(?:comment on|reply to|response to|rejoinder to|virtual special issue|special issue introduction|"
    r"introduction to the special issue|a note from the editors?)\b|[—–:-]\s*(?:comment|reply|a reply|response)\s*$",
    re.I)
NONRESEARCH_ABSTRACT = re.compile(
    r"original article to which this erratum refers|in the originally published article|this article has been retracted|"
    r"were recognized as (?:a )?(?:runner|winner)|was recognized as (?:a )?(?:runner|winner)|seeks to publish|"
    r"is committed to publishing|welcomes high.quality|in this virtual special issue|this virtual special issue|"
    r"the strategic management journal seeks|this (?:commentary|discussion|comment|editorial|essay|rejoinder|reply) "
    r"(?:was|has been) accepted|acknowledge all (?:of )?our \d{4} reviewers|is a scholarly journal that publishes|"
    r"forthcoming in management science|^\s*we regret|an error (?:was|has been) (?:found|identified|introduced)",
    re.I)
DOI_PREFIX = {"OrgSci": "10.1287/orsc", "SMJ": "10.1002/smj", "MgmtSci": "10.1287/mnsc"}
KEEP_TYPES = {"article", "paratext", "conference-abstract"}   # paratext kept only if it passes content checks
