import re
import subprocess
from dataclasses import dataclass, field

from strands import Agent
from strands.types.exceptions import EventLoopException, MaxTokensReachedException

from src.agents.agents import build_agent
from src.agents.prompts import (
    SERVICE_UNAVAILABLE_CONTINUATION,
    TOKEN_LIMIT_CONTINUATION,
)
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
    token_continuations: int = 0

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
        [
            "docker",
            "cp",
            str(REPO / "tasks" / task / "tests"),
            f"{env.container}:/tests",
        ],
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


def run_agent(
    env: TaskEnv,
    rollout: Rollout,
    lessons: str | None,
    config: Config,
) -> Agent | None:
    try:
        agent = build_agent(env, lessons=lessons, config=config)
        message = env.instruction
        service_retries = 0
        while True:
            try:
                agent(message)
                return agent
            except EventLoopException as exc:
                response = getattr(exc.original_exception, "response", {})
                error_code = response.get("Error", {}).get("Code")
                if error_code != "ServiceUnavailableException":
                    raise
                max_retries = max(0, config.model.request_attempts - 1)
                if service_retries >= max_retries:
                    rollout.error = (
                        "agent: ServiceUnavailableException after "
                        f"{service_retries} retries: {exc}"
                    )
                    return agent
                env.check_deadline()
                service_retries += 1
                print(
                    f"{rollout.task}: Bedrock service unavailable; retrying "
                    f"{service_retries}/{max_retries}"
                )
                message = SERVICE_UNAVAILABLE_CONTINUATION
            except MaxTokensReachedException as exc:
                if (
                    rollout.token_continuations
                    >= config.run.max_token_continuations
                ):
                    rollout.error = (
                        "agent: MaxTokensReachedException after "
                        f"{rollout.token_continuations} continuations: {exc}"
                    )
                    return agent
                env.check_deadline()
                rollout.token_continuations += 1
                print(
                    f"{rollout.task}: output token limit; continuing "
                    f"{rollout.token_continuations}/"
                    f"{config.run.max_token_continuations}"
                )
                message = TOKEN_LIMIT_CONTINUATION
    except Exception as exc:
        rollout.error = f"agent: {type(exc).__name__}: {exc}"
        return None


def execute_rollout(
    task: str,
    lessons: str | None = None,
    config: Config = CONFIG,
) -> tuple[Rollout, list[dict]]:
    rollout = Rollout(task=task)
    with TaskEnv(task=task) as env:
        agent = run_agent(env, rollout, lessons, config)
        trajectory = agent.messages if agent is not None else []

        try:
            rollout.official, rollout.counts, verifier_log = run_verifier(env, task)
            rollout.verifier_feedback = summarize_verifier_log(verifier_log)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            verifier_error = f"verifier: {type(exc).__name__}: {exc}"
            rollout.error = (
                f"{rollout.error}; {verifier_error}"
                if rollout.error
                else verifier_error
            )

    return rollout, trajectory


def run_rollout(
    task: str,
    lessons: str | None = None,
    config: Config = CONFIG,
) -> Rollout:
    rollout, _ = execute_rollout(task, lessons, config)

    return rollout
