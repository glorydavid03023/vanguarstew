"""Orchestrate the time-travel replay: freeze -> run agents -> pairwise judge -> tally.

The agent entrypoint is loaded by file path (as ninja's validator loads `agent.py`), so the
top-level `agent.py` module and the `agent/` package don't collide. For MVP the challenger is
compared against a naive baseline maintainer; in M2+ this becomes challenger-vs-king.
"""

from __future__ import annotations

import importlib.util
import os
import random
import shutil
import sys
import tempfile

from agent.llm import LLM
from benchmark.freeze import write_frozen
from benchmark.judge import pairwise_judge
from benchmark.score import trajectory_overlap
from benchmark.taskgen import generate_tasks


def load_solve(agent_file: str = "agent.py"):
    root = os.path.dirname(os.path.abspath(agent_file))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("vanguarstew_entry", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def baseline_solve(repo_path, request, **_kw) -> dict:
    """A naive maintainer that proposes nothing concrete — the bar to beat."""
    return {"plan": [], "philosophy": {}, "action": "plan", "rationale": "baseline"}


def run_replay(repo_path, agent_file="agent.py", n_tasks=3, horizon=5,
               model=None, api_base=None, api_key=None, work_dir=None, seed=0) -> dict:
    solve = load_solve(agent_file)
    llm = LLM(model=model, api_base=api_base, api_key=api_key)
    tasks = generate_tasks(repo_path, n_tasks, horizon)
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
            request = f"plan the next {horizon} maintainer actions"
            challenger = solve(
                repo_path=dest, request=request,
                model=model or "validator-managed-model",
                api_base=api_base or "", api_key=api_key or "offline", n=horizon,
            )
            baseline = baseline_solve(dest, request)
            winner = pairwise_judge(ctx, challenger.get("plan"), baseline.get("plan"),
                                    task["revealed"], llm, rng)
            who = {"A": "challenger", "B": "baseline", "tie": "tie"}[winner]
            tally[who] += 1
            rows.append({
                "task": k,
                "freeze": task["freeze_commit"][:10],
                "winner": who,
                "overlap": trajectory_overlap(challenger.get("plan"), task["revealed"]),
            })
    finally:
        if not work_dir:
            shutil.rmtree(base, ignore_errors=True)

    return {
        "tasks": len(tasks),
        "tally": tally,
        "decisive_margin": tally["challenger"] - tally["baseline"],
        "rows": rows,
        "offline": llm.offline,
    }
