"""Drop-in access to the evolved rulebooks.

    from harness.load import rulebook_system
    system = rulebook_system("OrgSci")   # "You are the editorial board of Organization Science. Apply these rules: ..."

Returns the exact system prompt the harness judge used with the final rulebook.
"""
from __future__ import annotations

from pathlib import Path

FINAL = Path(__file__).resolve().parents[1] / "results" / "harness"


def rulebook_system(journal: str) -> str:
    """journal: OrgSci | SMJ | MgmtSci. Reads results/harness/{journal}/final.md."""
    return (FINAL / journal / "final.md").read_text().rstrip("\n")
