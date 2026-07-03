"""Tests for the pairwise judge (offline, deterministic).

Covers the M2 addition: the judge weighs the decision process (philosophy + reasoning),
not just plan direction — so when plans are equal, sounder reasoning breaks the tie.
"""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["VANGUARSTEW_OFFLINE"] = "1"

from agent.llm import LLM  # noqa: E402
from benchmark.judge import _parse_winner, pairwise_judge  # noqa: E402


class _FakeLLM:
    """Online judge stand-in whose verdict is driven by a chosen bias, for testing."""

    def __init__(self, mode):
        self.offline = False
        self.mode = mode
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        if self.mode == "position_first":
            return '{"winner": "A"}'          # always picks whichever is shown FIRST
        if self.mode == "position_second":
            return '{"winner": "B"}'          # always picks whichever is shown SECOND
        if self.mode == "content":
            one = user.split("SUBMISSION ONE:")[1].split("SUBMISSION TWO:")[0]
            return '{"winner": "A"}' if "GOOD" in one else '{"winner": "B"}'
        return '{"winner": "tie"}'


_GOOD = {"philosophy": {"summary": "GOOD"}, "plan": [{"title": "real"}], "rationale": "GOOD"}
_BAD = {"philosophy": {}, "plan": [], "rationale": "meh"}


def test_parse_winner_tolerant():
    assert _parse_winner('{"winner": "A", "why": "clear"}') == "A"
    assert _parse_winner('{"winner":"B"}') == "B"
    # truncated JSON with smart quotes (the real failure that live-testing surfaced)
    assert _parse_winner('{"winner":"A","why":"aligns with the repo’s focus and its pla') == "A"
    assert _parse_winner("winner = tie") == "tie"
    assert _parse_winner("no verdict here") == "tie"
    assert _parse_winner("") == "tie"


def _sub(plan_items=0, philosophy=True, rationale=True):
    return {
        "philosophy": {"summary": "conservative, refactor-first"} if philosophy else {},
        "plan": [{"title": f"action {i}"} for i in range(plan_items)],
        "rationale": "weighed risk vs. priority" if rationale else "",
    }


def test_offline_prefers_richer_submission():
    llm = LLM(api_key="offline")
    strong, weak = _sub(3, True, True), _sub(0, False, False)
    assert pairwise_judge({}, strong, weak, [], llm, random.Random(0)) == "A"
    # position must not change the outcome
    assert pairwise_judge({}, weak, strong, [], llm, random.Random(0)) == "B"


def test_offline_tie_on_equal_submissions():
    llm = LLM(api_key="offline")
    a, b = _sub(2, True, True), _sub(2, True, True)
    assert pairwise_judge({}, a, b, [], llm) == "tie"


def test_decision_process_breaks_tie_when_plans_equal():
    # same plan length, but only one carries philosophy + reasoning -> it wins on process
    llm = LLM(api_key="offline")
    with_process, without = _sub(1, True, True), _sub(1, False, False)
    assert pairwise_judge({}, with_process, without, [], llm) == "A"
    assert pairwise_judge({}, without, with_process, [], llm) == "B"


def test_verbose_fluff_plan_does_not_beat_concise_substance():
    # A long plan padded with empty-of-substance items must NOT beat a shorter plan
    # of real maintainer actions. Guards the length-over-substance failure (#54);
    # ranking on raw len(plan) would have let the fluff win 6 > 2.
    llm = LLM(api_key="offline")
    fluff = {
        "philosophy": {},
        "plan": [{"title": "   "} for _ in range(6)] + [{"note": "we will consider things"}],
        "rationale": "we will think carefully and consider many aspects going forward",
    }
    substance = {
        "philosophy": {"direction": "stabilize toward v1.0", "values": ["conservative"]},
        "plan": [
            {"title": "fix release false-positive", "kind": "bugfix"},
            {"title": "cut patch release", "kind": "release"},
        ],
        "rationale": "cleared the release blocker before new work",
    }
    assert pairwise_judge({}, substance, fluff, [], llm) == "A"
    assert pairwise_judge({}, fluff, substance, [], llm) == "B"


def test_dual_order_keeps_consistent_winner():
    # A judge that genuinely prefers the stronger submission agrees across both orders.
    llm = _FakeLLM("content")
    assert pairwise_judge({}, _GOOD, _BAD, [], llm) == "A"
    assert llm.calls == 2  # both presentation orders were asked
    # winner tracks the content regardless of which argument position it's in
    assert pairwise_judge({}, _BAD, _GOOD, [], _FakeLLM("content")) == "B"


def test_dual_order_ties_a_position_biased_judge():
    # "always pick the first-shown" and "always pick the second-shown" are pure position
    # bias — dual-order must refuse to award either a spurious win.
    assert pairwise_judge({}, _GOOD, _BAD, [], _FakeLLM("position_first")) == "tie"
    assert pairwise_judge({}, _GOOD, _BAD, [], _FakeLLM("position_second")) == "tie"


def test_single_order_mode_makes_one_call_and_can_be_swayed():
    # With dual_order disabled, only one call is made and a position-biased judge decides it.
    llm = _FakeLLM("position_first")
    # rng.random() >= 0.5 -> no swap, submission_a shown first -> biased judge picks A.
    result = pairwise_judge({}, _GOOD, _BAD, [], llm, random.Random(1), dual_order=False)
    assert llm.calls == 1
    assert result in ("A", "B")  # a (biased) decision, not forced to tie
