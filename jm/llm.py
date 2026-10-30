"""Unified LLM access for the journal-models prototype.

Backends
--------
cli         Claude Code headless (`claude -p`). Bills to the user's Claude plan.
            Default for Claude models.
api         Anthropic Messages API via the official SDK. Key from ANTHROPIC_API_KEY,
            else CLAUDE_KEY from ~/.zshrc. Billed and budget-capped.
openrouter  OpenRouter chat completions, for non-Claude cross-family judges.
            Billed and budget-capped.

Every call is cached on disk, keyed by the full request, so reruns are free and
reproducible. Every call is also appended to a shared JSONL ledger. Billed backends
check the ledger against a per-backend cap before each call. CLI calls record their
notional list-price cost but are not capped.

Usage
-----
    from jm.llm import call, map_parallel
    r = call("Which abstract ...?", system=playbook, model="opus", schema=SCHEMA, tag="evaluate/gepa")
    r["json"], r["text"], r["cost_usd"]
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "llm"
CLI_CWD = ROOT / ".cache" / "cli_cwd"   # neutral cwd so no project CLAUDE.md is picked up
LEDGER = ROOT / "results" / "ledger.jsonl"

MODELS = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
}
# $ per 1M tokens (input, output, cache read). Anthropic table cached 2026-10-06.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CAPS_USD = {
    "api": float(os.environ.get("JM_CAP_API_USD", 80)),
    "openrouter": float(os.environ.get("JM_CAP_OPENROUTER_USD", 20)),
}

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- ledger / cache

def _ledger_append(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def spent(backend: str | None = None, tag_prefix: str | None = None) -> float:
    """Billed (or, for cli, notional) USD recorded in the ledger."""
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    with open(LEDGER) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        for line in f:
            r = json.loads(line)
            if r.get("cached"):
                continue
            if backend and r["backend"] != backend:
                continue
            if tag_prefix and not r.get("tag", "").startswith(tag_prefix):
                continue
            total += r.get("cost_usd", 0.0)
        fcntl.flock(f, fcntl.LOCK_UN)
    return total


def _check_cap(backend: str, est: float) -> None:
    if backend not in CAPS_USD:
        return
    s = spent(backend)
    if s + est > CAPS_USD[backend]:
        raise BudgetExceeded(f"{backend}: spent ${s:.2f} + est ${est:.2f} > cap ${CAPS_USD[backend]:.2f}")


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _parse_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------- backends

def _call_cli(prompt, system, model, schema, effort, timeout):
    CLI_CWD.mkdir(parents=True, exist_ok=True)
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--disable-slash-commands", "--no-chrome", "--setting-sources", "",
           "--output-format", "json",
           "--system-prompt", system or "You are a careful scholarly editor."]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    if effort:
        cmd += ["--effort", effort]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=CLI_CWD, env=env, timeout=timeout)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise LLMError(f"cli: non-JSON output (rc={p.returncode}): {p.stderr[-500:] or p.stdout[-500:]}")
    if d.get("is_error"):
        raise LLMError(f"cli: {d.get('subtype')}: {str(d.get('result'))[:500]}")
    text = d.get("result") or ""
    js = d.get("structured_output") if schema else _parse_json(text)
    return {"text": text, "json": js, "usage": d.get("usage", {}), "cost_usd": float(d.get("total_cost_usd") or 0.0)}


_api_client = None


def _anthropic_client():
    global _api_client
    if _api_client is None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_KEY")
        if not key:
            m = re.search(r'export CLAUDE_KEY="?([^"\n]+)"?', (Path.home() / ".zshrc").read_text())
            key = m.group(1) if m else None
        if not key:
            raise LLMError("api: no ANTHROPIC_API_KEY / CLAUDE_KEY found")
        _api_client = anthropic.Anthropic(api_key=key, max_retries=4)
    return _api_client


def _call_api(prompt, system, model, schema, effort, max_tokens):
    pin, pout, _ = PRICES[model]
    _check_cap("api", (len(prompt) + len(system or "")) / 3.5 / 1e6 * pin + max_tokens / 1e6 * pout)
    output_config = {}
    if effort:
        output_config["effort"] = effort
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    if output_config:
        kw["output_config"] = output_config
    resp = _anthropic_client().messages.create(**kw)
    if resp.stop_reason == "refusal":
        raise LLMError(f"api: refusal ({getattr(resp.stop_details, 'category', None)})")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    pin, pout, pcache = PRICES[model]
    cost = (u.input_tokens * pin + u.output_tokens * pout + (u.cache_read_input_tokens or 0) * pcache
            + (u.cache_creation_input_tokens or 0) * pin * 1.25) / 1e6
    return {"text": text, "json": _parse_json(text), "usage": u.model_dump(), "cost_usd": cost}


def _call_openrouter(prompt, system, model, schema, max_tokens, timeout):
    import requests
    _check_cap("openrouter", 0.05)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "usage": {"include": True}}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json=body)
    if r.status_code != 200:
        raise LLMError(f"openrouter {r.status_code}: {r.text[:500]}")
    d = r.json()
    text = d["choices"][0]["message"].get("content") or ""
    u = d.get("usage", {})
    return {"text": text, "json": _parse_json(text), "usage": u, "cost_usd": float(u.get("cost") or 0.0)}


# ---------------------------------------------------------------- public API

def call(prompt: str, *, system: str | None = None, model: str = "opus", backend: str | None = None,
         schema: dict | None = None, effort: str | None = None, max_tokens: int = 4000,
         tag: str = "", use_cache: bool = True, retries: int = 4, timeout: int = 600) -> dict:
    """One LLM call. `model` is an alias (opus/sonnet/haiku), a Claude model ID, or an OpenRouter
    model slug such as "openai/gpt-..." (which implies backend="openrouter")."""
    model = MODELS.get(model, model)
    if backend is None:
        backend = "openrouter" if "/" in model else "cli"
    payload = {"prompt": prompt, "system": system, "model": model, "schema": schema,
               "effort": effort, "max_tokens": max_tokens}
    key = _key(payload)
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if use_cache and path.exists():
        out = json.loads(path.read_text())
        _ledger_append({"ts": time.time(), "tag": tag, "backend": backend, "model": model, "key": key,
                        "cached": True, "cost_usd": 0.0})
        return out
    last = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            if backend == "cli":
                out = _call_cli(prompt, system, model, schema, effort, timeout)
            elif backend == "api":
                out = _call_api(prompt, system, model, schema, effort, max_tokens)
            elif backend == "openrouter":
                out = _call_openrouter(prompt, system, model, schema, max_tokens, timeout)
            else:
                raise ValueError(backend)
            break
        except BudgetExceeded:
            raise
        except Exception as e:  # rate limits, timeouts, transient CLI failures
            last = e
            time.sleep(min(120, 5 * 2 ** attempt))
    else:
        raise LLMError(f"{backend}/{model} failed after {retries + 1} attempts: {last}")
    out.update({"model": model, "backend": backend, "seconds": round(time.time() - t0, 2)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    _ledger_append({"ts": time.time(), "tag": tag, "backend": backend, "model": model, "key": key,
                    "cached": False, "cost_usd": out["cost_usd"], "seconds": out["seconds"],
                    "usage": out.get("usage")})
    return out


def map_parallel(fn, items, workers: int = 6):
    """Apply fn to items concurrently; returns results in order (exceptions returned, not raised)."""
    def safe(x):
        try:
            return fn(x)
        except BudgetExceeded:
            raise
        except Exception as e:
            return e
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, items))


def spend_report() -> dict:
    return {b: round(spent(b), 4) for b in ("cli", "api", "openrouter")} | {"caps": CAPS_USD}
