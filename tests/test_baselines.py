"""Tests for the selectable reference baselines."""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["VANGUARSTEW_OFFLINE"] = "1"

from benchmark.baselines import (  # noqa: E402
    BASELINES,
    empty_baseline,
    get_baseline,
    heuristic_baseline,
)
from benchmark.runner import run_replay  # noqa: E402

CONTEXT = {
    "open_issues": [
        {"title": "Add plugin loader"},
        {"title": "Fix flaky release job"},
    ],
    "recent_commits": [
        {"subject": "refactor core engine"},
        {"subject": "add plugin loader"},   # duplicate theme of an open issue
        {"subject": "tidy docs"},
    ],
}


def test_empty_baseline_proposes_nothing():
    out = empty_baseline(context=CONTEXT, n=5)
    assert out["plan"] == []
    assert out["action"] == "plan"


def test_heuristic_prioritizes_issues_then_recent_commits():
    out = heuristic_baseline(context=CONTEXT, n=5)
    titles = [p["title"] for p in out["plan"]]
    # open issues come first, in order...
    assert titles[0] == "Add plugin loader"
    assert titles[1] == "Fix flaky release job"
    # ...then recent commits are continued.
    assert any(t.startswith("continue: refactor core engine") for t in titles)
    # every plan item carries the shape the scorer/judge expect.
    for item in out["plan"]:
        assert set(item) >= {"title", "theme", "kind"}


def test_heuristic_respects_n_and_dedupes():
    out = heuristic_baseline(context=CONTEXT, n=2)
    assert len(out["plan"]) == 2                       # capped at n
    lowered = [p["title"].lower() for p in out["plan"]]
    assert len(lowered) == len(set(lowered))           # no duplicate titles


def test_heuristic_theme_skips_filler_words():
    out = heuristic_baseline(
        context={"open_issues": [{"title": "Add plugin loader"}], "recent_commits": []},
        n=1,
    )
    assert out["plan"][0]["theme"] == "plugin"          # "add" is skipped as filler


def test_heuristic_empty_context_is_safe():
    out = heuristic_baseline(context={}, n=5)
    assert out["plan"] == []


def test_registry_and_selector():
    assert set(BASELINES) == {"empty", "heuristic"}
    assert get_baseline(None) is empty_baseline          # default
    assert get_baseline("heuristic") is heuristic_baseline
    with pytest.raises(ValueError):
        get_baseline("nope")


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_replay_runs_and_tallies_with_heuristic_baseline():
    d = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "init", "-q", d], check=True)
        subprocess.run(["git", "-C", d, "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", d, "config", "user.name", "t"], check=True)
        for i in range(20):
            with open(os.path.join(d, f"f{i}.py"), "w", encoding="utf-8") as f:
                f.write(f"x = {i}\n")
            subprocess.run(["git", "-C", d, "add", "-A"], check=True)
            subprocess.run(["git", "-C", d, "commit", "-q", "-m", f"commit {i}"], check=True)
        res = run_replay(d, agent_file=os.path.join(ROOT, "agent.py"),
                         n_tasks=2, horizon=3, baseline="heuristic")
        assert res.get("tasks", 0) >= 1
        assert "tally" in res and "decisive_margin" in res
        assert res["baseline"] == "heuristic"
    finally:
        shutil.rmtree(d, ignore_errors=True)
