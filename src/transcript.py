from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rollout import Rollout


def write_rollout(path: Path, rollout: Rollout) -> None:
    passed = rollout.counts.get("passed", 0)
    total = sum(rollout.counts.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as transcript:
        transcript.write(f"{rollout.task}: {passed}/{total} tests passed\n")


def write_run_summary(path: Path, rollouts: Iterable[Rollout]) -> None:
    task_results: dict[str, bool] = {}
    for rollout in rollouts:
        task_results[rollout.task] = (
            task_results.get(rollout.task, False) or rollout.official == 1
        )
    passed = sum(task_results.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as transcript:
        transcript.write(f"Run: {passed}/{len(task_results)} tasks passed\n")
