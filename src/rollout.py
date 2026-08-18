import re
import subprocess
from dataclasses import dataclass, field

from src.agents.agents import build_agent
from src.config import CONFIG, Config
from src.environment import REPO, TaskEnv

FAILURE_LINE = re.compile(
    r"FAIL:|FAILED |ERROR |^[ \t]*E[ \t]+|AssertionError|Compilation Error|"
    r"Parser Error|Runtime Error|Catalog Error|Binder Error|Test results:|FAILURE:"
)


@dataclass
class Rollout:
    task: str
    official: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    verifier_feedback: str | None = None

    @property
    def dense(self) -> float:
        total = sum(self.counts.values())
        return self.counts.get("passed", 0) / total if total else 0.0


def summarize_verifier_log(verifier_log: str) -> str | None:
    lines = [
        line.strip()
        for line in verifier_log.splitlines()
        if FAILURE_LINE.search(line)
    ]
    if not lines:
        return None
    return "\n".join(lines[:40])[:6_000]


def parse_test_counts(verifier_log: str) -> dict[str, int]:
    passed_components = len(
        re.findall(r"^[ \t]*PASS:[ \t]+\S", verifier_log, re.MULTILINE)
    )
    failed_components = len(
        re.findall(r"^[ \t]*FAIL:[ \t]+\S", verifier_log, re.MULTILINE)
    )
    if passed_components or failed_components:
        return {
            "passed": passed_components,
            "skipped": 0,
            "failed": failed_components,
        }

    match = re.search(
        r"Test results:\s*(\d+) passed,\s*(\d+) skipped,\s*(\d+) failed",
        verifier_log,
    )
    if not match:
        errors = len(re.findall(r"^ERROR\s+\S+::", verifier_log, re.MULTILINE))
        return {"passed": 0, "skipped": 0, "failed": errors}
    passed, skipped, failed = map(int, match.groups())
    if passed == skipped == failed == 0:
        failed = len(re.findall(r"^ERROR\s+\S+::", verifier_log, re.MULTILINE))
    return {"passed": passed, "skipped": skipped, "failed": failed}


def run_verifier(env: TaskEnv, task: str) -> tuple[int, dict[str, int], str]:
    env.enable_verifier_network()
    subprocess.run(
        ["docker", "cp", str(REPO / "tasks" / task / "tests"), f"{env.container}:/tests"],
        check=True,
        capture_output=True,
        text=True,
    )
    verifier = subprocess.run(
        ["docker", "exec", env.container, "bash", "/tests/test.sh"],
        capture_output=True,
        text=True,
        timeout=env.meta["verifier"]["timeout_sec"],
    )
    verifier_log = (verifier.stdout or "") + (verifier.stderr or "")
    reward = subprocess.run(
        ["docker", "exec", env.container, "cat", "/logs/verifier/reward.txt"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if reward not in {"0", "1"}:
        raise RuntimeError("verifier did not write a valid reward")
    return int(reward), parse_test_counts(verifier_log), verifier_log


def run_rollout(
    task: str,
    lessons: str | None = None,
    config: Config = CONFIG,
) -> Rollout:
    rollout = Rollout(task=task)
    with TaskEnv(task=task) as env:
        try:
            agent = build_agent(env, lessons=lessons, config=config)
            agent(env.instruction)
        except Exception as exc:
            rollout.error = f"agent: {type(exc).__name__}: {exc}"

        try:
            rollout.official, rollout.counts, verifier_log = run_verifier(env, task)
            rollout.verifier_feedback = summarize_verifier_log(verifier_log)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            rollout.error = f"verifier: {type(exc).__name__}: {exc}"

    return rollout
