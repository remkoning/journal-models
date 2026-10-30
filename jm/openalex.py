"""Minimal OpenAlex client: polite pool (mailto), on-disk response cache, usage log, credit guard.

OpenAlex meters usage in credits (list+filter page = 1 credit; search/semantic = 10; single-entity GET = free).
Without an API key the daily budget is 1,000 credits ($0.10). Every live request is appended to
data/local/openalex_usage.jsonl with the rate-limit headers, and we refuse to go below MIN_REMAINING.
Responses are cached by URL under data/local/raw/openalex/, so rebuilding is free once fetched.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "local"
CACHE = LOCAL / "raw" / "openalex"
USAGE_LOG = LOCAL / "openalex_usage.jsonl"
MAILTO = "remkoning@gmail.com"
BASE = "https://api.openalex.org"
MIN_REMAINING = int(os.environ.get("JM_OA_MIN_REMAINING", 25))  # keep a small reserve of daily credits

SOURCES = {"OrgSci": "S206124708", "SMJ": "S102949365", "MgmtSci": "S33323087"}


class OpenAlexBudget(RuntimeError):
    pass


_last_remaining: int | None = None


def _url(path: str, params: dict) -> str:
    params = dict(params)
    params["mailto"] = MAILTO
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return f"{BASE}/{path}?{urllib.parse.urlencode(params, safe=':,|<>!')}"


def get(path: str, params: dict | None = None, *, use_cache: bool = True, retries: int = 4) -> dict:
    global _last_remaining
    url = _url(path, params or {})
    key_url = url.replace(f"&api_key={os.environ.get('OPENALEX_API_KEY', '')}", "")
    h = hashlib.sha256(key_url.encode()).hexdigest()[:24]
    cpath = CACHE / f"{h}.json"
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text())["body"]
    if _last_remaining is not None and _last_remaining < MIN_REMAINING:
        raise OpenAlexBudget(f"OpenAlex credits remaining {_last_remaining} < reserve {MIN_REMAINING}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"journal-models (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=120) as r:
                hdr = {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
                body = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code == 429 and "credit" in msg.lower():
                raise OpenAlexBudget(f"OpenAlex daily budget exhausted: {msg}") from e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} on {key_url}: {msg}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    try:
        _last_remaining = int(hdr.get("x-ratelimit-remaining"))
    except (TypeError, ValueError):
        pass
    LOCAL.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": key_url,
                            "credits_used": hdr.get("x-ratelimit-credits-used"),
                            "remaining": hdr.get("x-ratelimit-remaining"),
                            "remaining_usd": hdr.get("x-ratelimit-remaining-usd")}) + "\n")
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"url": key_url, "fetched": time.strftime("%Y-%m-%d"), "body": body}))
    return body


def list_all(filter_str: str, select: str, *, per_page: int = 200, sort: str | None = None,
             max_pages: int = 200) -> list[dict]:
    """Page through a filtered list with basic paging (cursor paging needs an API key). 1 credit per page.
    Basic paging reaches the first 10,000 results, which covers any single journal here.
    A stable sort (by id) keeps pages disjoint."""
    out, seen = [], set()
    for page in range(1, max_pages + 1):
        params = {"filter": filter_str, "per-page": per_page, "select": select, "page": page,
                  "sort": sort or "publication_date:asc"}
        d = get("works", params)
        for w in d["results"]:
            if w["id"] not in seen:
                seen.add(w["id"])
                out.append(w)
        if not d["results"] or page * per_page >= d["meta"]["count"]:
            break
    if len(out) != d["meta"]["count"]:
        print(f"WARNING: fetched {len(out)} unique of {d['meta']['count']} for {filter_str}")
    return out


def abstract_from_inv(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    return " ".join(w for _, w in sorted(pos))


def short_id(oa_id: str) -> str:
    return oa_id.rsplit("/", 1)[-1]


def usage_summary() -> dict:
    if not USAGE_LOG.exists():
        return {"calls": 0, "credits": 0}
    rows = [json.loads(line) for line in USAGE_LOG.read_text().splitlines() if line.strip()]
    credits = sum(int(r.get("credits_used") or 0) for r in rows)
    return {"calls": len(rows), "credits": credits, "last_remaining": rows[-1].get("remaining") if rows else None}
