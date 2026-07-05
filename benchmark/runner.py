"""Orchestrate the time-travel replay: freeze -> run agents -> pairwise judge -> tally.

The agent entrypoint is loaded by file path (as ninja's validator loads `agent.py`), so the
top-level `agent.py` module and the `agent/` package don't collide. For MVP the challenger is
compared against a naive baseline maintainer; in M2+ this becomes challenger-vs-king.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

from agent.context import CONTEXT_FILE
from agent.llm import LLM
from benchmark.baselines import DEFAULT_BASELINE, get_baseline
from benchmark.freeze import write_frozen
from benchmark.github_context import enrich_context
from benchmark.judge import build_judge_report, judge_verbose, summarize_judge_orders
from benchmark.leakage import scrub_context
from benchmark.repo_set import RepoSetError, load_repo_set
from benchmark.score import (
    base_from_releases,
    composite_score,
    objective_component,
    objective_score,
    trajectory_overlap,
)
from benchmark.taskgen import generate_tasks

# Challenger-perspective judge outcome per row (mirrors score._JUDGE_OUTCOME, keyed by the
# runner's decoded winner label): a win is 1.0, a tie 0.5, a loss 0.0.
_JUDGE_COMPONENT = {"challenger": 1.0, "tie": 0.5, "baseline": 0.0}


def load_solve(agent_file: str = "agent.py"):
    root = os.path.dirname(os.path.abspath(agent_file))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("vanguarstew_entry", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def _submission(out: dict) -> dict:
    """The judged view of an agent's output: philosophy + plan + reasoning."""
    return {
        "philosophy": out.get("philosophy"),
        "plan": out.get("plan"),
        "rationale": out.get("rationale"),
    }


def _is_placeholder_source(source: str) -> bool:
    return "OWNER/" in source


def _materialize_repo_source(source: str, checkout_root: str | None) -> tuple[str, bool]:
    """Return a local repo path for a repo-set source plus whether it should be cleaned up."""
    if _is_placeholder_source(source):
        raise RepoSetError(
            f"repo-set source {source!r} is a placeholder (OWNER/...); "
            "copy the example config and replace placeholder sources with vetted repos"
        )
    if os.path.isdir(source):
        return source, False
    if checkout_root is None:
        raise RepoSetError(f"repo-set source not found locally: {source}")
    dest = os.path.join(checkout_root, f"repo_{len(os.listdir(checkout_root))}")
    try:
        subprocess.run(["git", "clone", "-q", source, dest], check=True, capture_output=True,
                       text=True)
    except subprocess.CalledProcessError as exc:
        raise RepoSetError(f"failed to clone repo-set source {source!r}: {exc.stderr.strip()}") from exc
    return dest, True


def run_replay(repo_path, agent_file="agent.py", n_tasks=3, horizon=5,
               model=None, api_base=None, api_key=None, work_dir=None, seed=0,
               enrich_github=False, github_token=None,
               recent_bias=False, rotation_seed=None,
               baseline=DEFAULT_BASELINE) -> dict:
    solve = load_solve(agent_file)
    opponent = get_baseline(baseline)
    llm = LLM(model=model, api_base=api_base, api_key=api_key)
    tasks = generate_tasks(
        repo_path, n_tasks, horizon, min_history=min_history,
        recent_bias=recent_bias, rotation_seed=rotation_seed, after=after, before=before)
    if not tasks:
        return {"error": "no usable tasks (repo too small for horizon/min_history)", "tasks": 0}

    rng = random.Random(seed)
    tally = {"challenger": 0, "baseline": 0, "tie": 0}
    rows = []
    base = work_dir or tempfile.mkdtemp(prefix="vanguarstew_work_")
    try:
        for k, task in enumerate(tasks):
            dest = os.path.join(base, f"task_{k}")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            ctx = write_frozen(repo_path, task["freeze_commit"], dest)
            if enrich_github:
                ctx = scrub_context(enrich_context(ctx, repo_path, token=github_token))
                with open(os.path.join(dest, CONTEXT_FILE), "w", encoding="utf-8") as f:
                    json.dump(ctx, f, indent=1)
            request = f"plan the next {horizon} maintainer actions"
            challenger = solve(
                repo_path=dest, request=request,
                model=model or "validator-managed-model",
                api_base=api_base or "", api_key=api_key or "offline", n=horizon,
            )
            baseline_out = opponent(repo_path=dest, request=request, context=ctx, n=horizon)
            winner = pairwise_judge(ctx, _submission(challenger), _submission(baseline_out),
                                    task["revealed"], llm, rng)
            who = {"A": "challenger", "B": "baseline", "tie": "tie"}[winner]
            tally[who] += 1
            obj = objective_score(
                challenger.get("plan"), task["revealed"],
                version_bump=challenger.get("version_bump"),
                base_version=base_from_releases(ctx.get("releases")),
                open_issues=ctx.get("open_issues"),
            )
            rows.append({
                "task": k,
                "freeze": task["freeze_commit"][:10],
                "winner": who,
                "judge_order": judge_order,
                "overlap": trajectory_overlap(challenger.get("plan"), task["revealed"]),
                "objective": obj,
                "composite": composite_score(winner, obj, w_judge, w_objective),
            })
    finally:
        if not work_dir:
            shutil.rmtree(base, ignore_errors=True)

    # The single-repo composite output contract: the mean blended score, plus the two
    # component means it blends (judge outcome + objective anchor) so the number is
    # inspectable and the multi-repo aggregate has explicit parts to average.
    composites = [r["composite"] for r in rows]
    judge_parts = [_JUDGE_COMPONENT[r["winner"]] for r in rows]
    objective_parts = [objective_component(r["objective"]) for r in rows]
    judge_order_stats = summarize_judge_orders(r.get("judge_order") for r in rows)
    return {
        "tasks": len(tasks),
        "baseline": baseline,
        "tally": tally,
        "decisive_margin": tally["challenger"] - tally["baseline"],
        "composite_mean": round(sum(composites) / len(composites), 3) if composites else 0.0,
        "composite_parts": {
            "judge_mean": round(sum(judge_parts) / len(judge_parts), 3) if judge_parts else 0.0,
            "objective_mean": (
                round(sum(objective_parts) / len(objective_parts), 3) if objective_parts else 0.0
            ),
        },
        "weights": {"judge": w_judge, "objective": w_objective},
        "rows": rows,
        "judge_order_stats": judge_order_stats,
        "judge_report": build_judge_report(tally, judge_order_stats),
        "offline": llm.offline,
        "github_enriched": enrich_github,
        "baseline": baseline,
    }
    if repo_set_meta is not None:
        result["repo_set"] = repo_set_meta
    return result
