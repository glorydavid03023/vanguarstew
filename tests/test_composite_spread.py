"""Tests for composite spread summary and CLI (deterministic, offline)."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.composite_spread import (  # noqa: E402
    composite_spread_headline,
    summarize_composite_spread,
)
from scripts import composite_spread as cli  # noqa: E402


def _single(judge, objective):
    return {
        "composite_mean": 0.6,
        "composite_parts": {"judge_mean": judge, "objective_mean": objective},
    }


def test_spread_is_judge_minus_objective():
    out = summarize_composite_spread(_single(0.7, 0.5))
    assert out["spread"] == 0.2
    assert out["kind"] == "single"


def test_generalization_reads_tuned_partition():
    art = {
        "tuned": _single(0.8, 0.4),
        "held_out": _single(0.5, 0.5),
        "generalization_gap": 0.1,
    }
    out = summarize_composite_spread(art)
    assert out["kind"] == "generalization"
    assert out["spread"] == 0.4


def test_missing_parts_yield_none_spread():
    out = summarize_composite_spread({"composite_mean": 0.5})
    assert out["spread"] is None


# --- the scored_repos guard: a run that scored nothing has no spread ------------------------------

def _unscored_multi():
    # run_multi_replay: every repo too small for the horizon -> scored_repos 0 and composite_parts
    # that are averages over empty lists (placeholder 0.0), not real means.
    return {
        "repos": 2, "scored_repos": 0, "skipped": 2, "composite_mean": 0.0,
        "composite_parts": {"judge_mean": 0.0, "objective_mean": 0.0},
        "per_repo": [
            {"repo": "o/a", "tasks": 0, "composite_mean": 0.0},
            {"repo": "o/b", "tasks": 0, "composite_mean": 0.0},
        ],
    }


def test_unscored_multi_run_has_no_means_or_spread():
    # Reading the placeholder as a real score reports a confident, perfectly balanced
    # "judge 0.0 vs objective 0.0 (delta +0.000)" for a run that measured nothing.
    out = summarize_composite_spread(_unscored_multi())
    assert out["kind"] == "multi"
    assert out["judge_mean"] is None
    assert out["objective_mean"] is None
    assert out["spread"] is None
    assert "n/a" in composite_spread_headline(out)


def test_scored_multi_run_keeps_its_means():
    # The guard fires only on scored_repos == 0; a run that scored is untouched.
    art = _unscored_multi()
    art.update(scored_repos=2, skipped=0,
               composite_parts={"judge_mean": 0.7, "objective_mean": 0.5})
    out = summarize_composite_spread(art)
    assert (out["judge_mean"], out["objective_mean"], out["spread"]) == (0.7, 0.5, 0.2)


def test_single_repo_zero_means_are_real_and_kept():
    # A single-repo run carries no scored_repos key, so a genuine 0.0 from a run that actually
    # scored must survive -- the guard must not swallow it.
    out = summarize_composite_spread({"tasks": 5, "composite_parts":
                                      {"judge_mean": 0.0, "objective_mean": 0.0}})
    assert (out["judge_mean"], out["objective_mean"], out["spread"]) == (0.0, 0.0, 0.0)


def test_generalization_unscored_tuned_partition_has_no_spread():
    # The headline partition is `tuned`; when it scored nothing its placeholder means must not
    # become a spread, even though held_out scored fine.
    out = summarize_composite_spread({
        "generalization_gap": 0.0,
        "tuned": {"scored_repos": 0, "per_repo": [],
                  "composite_parts": {"judge_mean": 0.0, "objective_mean": 0.0}},
        "held_out": {"scored_repos": 1, "per_repo": [{"repo": "o/b", "tasks": 4}],
                     "composite_parts": {"judge_mean": 0.6, "objective_mean": 0.4}},
    })
    assert out["kind"] == "generalization"
    assert out["spread"] is None


def test_boolean_scored_repos_is_not_coerced_to_zero():
    # `False` is not a numeric 0 (mirrors _is_number rejecting bool), so a malformed scored_repos
    # does not silently blank out real means.
    out = summarize_composite_spread({
        "scored_repos": False,
        "composite_parts": {"judge_mean": 0.3, "objective_mean": 0.2},
    })
    assert out["spread"] == 0.1


def test_malformed_parts_yield_none_spread():
    out = summarize_composite_spread({"composite_mean": 0.5, "composite_parts": 42})
    assert out["spread"] is None


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_non_finite_mean_yields_none_spread(bad):
    # json round-trips NaN/Infinity verbatim; a non-finite mean must degrade to None/n/a rather
    # than poisoning the spread (mirrors component_mix / trend), not pass through as +inf/+nan.
    out = summarize_composite_spread(_single(bad, 0.5))
    assert out["judge_mean"] is None
    assert out["spread"] is None
    assert "n/a" in composite_spread_headline(out)


def test_oversized_int_mean_is_not_numeric():
    out = summarize_composite_spread(_single(10**400, 0.5))
    assert out["judge_mean"] is None
    assert out["spread"] is None


def test_headline():
    out = summarize_composite_spread(_single(0.6, 0.4))
    assert "delta +0.200" in composite_spread_headline(out)


@pytest.fixture
def tmp_artifact(tmp_path):
    def write(payload):
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)
    return write


def test_cli(tmp_artifact, capsys):
    path = tmp_artifact(_single(0.55, 0.45))
    assert cli.run([path]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["spread"] == 0.1


def test_cli_missing_file(tmp_path, capsys):
    assert cli.run([str(tmp_path / "missing.json")]) == 2
    assert "cannot read artifact" in capsys.readouterr().err


def test_cli_invalid_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert cli.run([str(path)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_cli_non_object_artifact(tmp_path, capsys):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert cli.run([str(path)]) == 2
    assert "must be a JSON object" in capsys.readouterr().err


def test_cli_unreadable_path_is_handled(tmp_path, capsys):
    assert cli.run([str(tmp_path)]) == 2
    assert "cannot read artifact" in capsys.readouterr().err
