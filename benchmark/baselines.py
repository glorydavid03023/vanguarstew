"""Reference baseline maintainers — the opponents a challenger is judged against.

The pairwise signal is only as meaningful as the opponent. Two baselines are provided:

- ``empty`` — proposes nothing concrete. The absolute floor (keeps the original behavior).
- ``heuristic`` — a non-LLM maintainer that extrapolates the *recent commit cadence* and
  works down the *open-issue backlog* knowable at T. It beats the empty floor whenever the
  repo simply kept doing what it was already doing, so a challenger has to actually add
  signal to win.

Each baseline has the same shape as the agent's ``solve`` output (``plan`` + ``philosophy``
+ ``rationale``), so the judge and the objective score treat it exactly like a challenger.
Baselines are validator-owned; miners don't edit them.
"""

from __future__ import annotations

import re

from agent.context import load_context

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
# Filler verbs/words that make a poor "theme" — skip to the first meaningful token.
_STOP = frozenset({
    "add", "fix", "the", "and", "for", "with", "update", "updates", "remove",
    "make", "use", "using", "into", "from", "that", "this", "wip", "merge",
})


def _theme(text: str) -> str:
    """First meaningful word in a subject, as a coarse theme tag (else "")."""
    for w in _WORD.findall(text or ""):
        if w.lower() not in _STOP:
            return w.lower()
    return ""


def empty_baseline(repo_path=None, request="", context=None, n=5, **_kw) -> dict:
    """A naive maintainer that proposes nothing concrete — the bar to beat."""
    return {
        "plan": [],
        "philosophy": {},
        "action": "plan",
        "version_bump": None,
        "rationale": "baseline",
    }


def heuristic_baseline(repo_path=None, request="", context=None, n=5, **_kw) -> dict:
    """Plan by extrapolating recent commit cadence + the open-issue backlog (no LLM).

    Deterministic and offline: it reads only the frozen, knowable-at-T context. Open issues
    are treated as planned work first (a maintainer works the backlog), then the plan is
    filled by continuing the themes of the most recent commits.
    """
    ctx = context if context is not None else load_context(repo_path)
    plan: list[dict] = []
    seen: set[str] = set()

    def _push(title: str, kind: str) -> None:
        title = (title or "").strip()
        key = title.lower()
        if not title or key in seen or len(plan) >= n:
            return
        seen.add(key)
        plan.append({"title": title, "theme": _theme(title), "kind": kind})

    # 1) The open-issue backlog is the most direct signal of what's coming next.
    for issue in ctx.get("open_issues") or []:
        title = issue.get("title") if isinstance(issue, dict) else str(issue)
        _push(title, "issue")

    # 2) Fill the rest by continuing recent commit cadence (newest first).
    for commit in ctx.get("recent_commits") or []:
        subject = commit.get("subject") if isinstance(commit, dict) else str(commit)
        if subject:
            _push(f"continue: {subject}", "followup")

    return {
        "plan": plan[:n],
        "philosophy": {"summary": "extrapolate recent cadence and open-issue backlog"},
        "action": "plan",
        "version_bump": None,
        "rationale": (
            "heuristic reference: work the open-issue backlog, then continue the themes of "
            "the most recent commits"
        ),
    }


# Registry of selectable opponents (name -> solve-like callable).
BASELINES = {
    "empty": empty_baseline,
    "heuristic": heuristic_baseline,
}

DEFAULT_BASELINE = "empty"


def get_baseline(name: str | None):
    """Look up a baseline by name; raise a clear error listing the valid choices."""
    key = name or DEFAULT_BASELINE
    try:
        return BASELINES[key]
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r}; choose one of {sorted(BASELINES)}"
        ) from None
