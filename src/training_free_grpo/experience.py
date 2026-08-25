import json
import re

from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.models import BedrockModel

from src.config import Config
from src.training_free_grpo.notebook import (
    LESSON_WORD_LIMIT,
    normalize_lesson_text,
)
from src.training_free_grpo.prompts import (
    BATCH_UPDATE_PROMPT,
    BATCH_UPDATE_SYSTEM_PROMPT,
    CANDIDATE_PROMPT,
    CANDIDATE_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REPAIR_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

EVIDENCE_TYPES = {
    "rollout_contrast",
    "successful_repair",
    "reference_difference",
}
PROCESS_TRIGGER = "A changed dbt model has no successful targeted execution"
PROCESS_ACTION = (
    "Run the changed model, check the final status, fix the first blocking "
    "error, and run it again before finishing"
)


def experience_mode(rollouts: list[dict]) -> str | None:
    usable = [rollout for rollout in rollouts if not rollout.get("error")]
    scores = {rollout["dense"] for rollout in usable}
    if len(scores) > 1:
        return "contrast"
    if scores == {0.0}:
        return "all_zero"
    return None


def extract_candidates(
    task: str,
    instruction: str,
    rollouts: list[dict],
    lessons: list[dict],
    reference: dict,
    config: Config,
) -> dict:
    """Extract supported candidates from one task group."""
    mode = experience_mode(rollouts)
    if mode is None:
        return {
            "mode": None,
            "summaries": [],
            "candidates": [],
            "rejected_candidates": [],
            "skipped": (
                "The rollout group has no score contrast and is not all zero."
            ),
        }

    training = _training(config)
    model = _model(config)
    usable = [rollout for rollout in rollouts if not rollout.get("error")]
    summaries = []
    for rollout in usable:
        prompt = SUMMARY_PROMPT.format(
            instruction=instruction,
            rollout_number=rollout["rollout"],
            official=rollout["official"],
            dense=rollout["dense"],
            counts=json.dumps(rollout["counts"]),
            error=rollout["error"] or "None",
            trajectory=json.dumps(
                rollout["trajectory"], ensure_ascii=False, default=str
            ),
            verifier_feedback=(
                rollout.get("verifier_feedback")
                or "No verifier failure was recorded."
            ),
        )
        summary, summary_error, used_fallback = _summarize_rollout(
            model,
            prompt,
            rollout,
        )
        summaries.append(
            {
                "rollout": rollout["rollout"],
                "official": rollout["official"],
                "dense": rollout["dense"],
                "summary_error": summary_error,
                "summary_fallback": used_fallback,
                **summary,
            }
        )

    response = _ask(
        model,
        CANDIDATE_SYSTEM_PROMPT.format(
            max_lessons=training.max_lessons_per_task,
            word_limit=LESSON_WORD_LIMIT,
        ),
        CANDIDATE_PROMPT.format(
            mode=mode,
            instruction=instruction,
            lessons=_format_lessons(lessons),
            summaries=json.dumps(summaries, indent=2, ensure_ascii=False),
            reference=json.dumps(reference, indent=2, ensure_ascii=False),
        ),
    )

    candidates = []
    if (
        any(not _has_successful_targeted_run(rollout) for rollout in usable)
        and not _has_process_lesson(lessons)
    ):
        candidates.append(_process_candidate(task, usable))
    rejected = []
    for candidate in _parse_array(response):
        reason = _candidate_error(candidate, task, mode)
        if reason:
            rejected.append({"candidate": candidate, "reason": reason})
            continue
        normalized = {
            "trigger": normalize_lesson_text(candidate["trigger"]),
            "action": normalize_lesson_text(candidate["action"]),
            "evidence": candidate["evidence"],
            "source_task": task,
        }
        if any(
            item["trigger"] == normalized["trigger"]
            and item["action"] == normalized["action"]
            for item in candidates
        ):
            rejected.append(
                {"candidate": candidate, "reason": "candidate already exists"}
            )
            continue
        candidates.append(normalized)

    if len(candidates) > training.max_lessons_per_task:
        rejected.extend(
            {"candidate": item, "reason": "candidate limit reached"}
            for item in candidates[training.max_lessons_per_task :]
        )
        candidates = candidates[: training.max_lessons_per_task]

    return {
        "mode": mode,
        "summaries": summaries,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "raw_candidates": response,
        "skipped": None,
    }


def update_notebook(
    candidates: list[dict],
    lessons: list[dict],
    config: Config,
) -> dict:
    """Create one notebook update for one task batch."""
    if not candidates:
        return {
            "operations": [],
            "raw_update": "",
            "skipped": "The batch has no supported candidates.",
        }

    model = _model(config)
    response = _ask(
        model,
        BATCH_UPDATE_SYSTEM_PROMPT.format(
            max_operations=len(candidates),
            word_limit=LESSON_WORD_LIMIT,
        ),
        BATCH_UPDATE_PROMPT.format(
            lessons=_format_lessons(lessons),
            candidates=json.dumps(candidates, indent=2, ensure_ascii=False),
        ),
    )
    return {
        "operations": _parse_array(response),
        "raw_update": response,
        "skipped": None,
    }


def _training(config: Config):
    if config.training is None:
        raise ValueError("the selected configuration has no training section")
    return config.training


def _process_candidate(task: str, rollouts: list[dict]) -> dict:
    rollout = next(
        rollout
        for rollout in rollouts
        if not _has_successful_targeted_run(rollout)
    )
    return {
        "trigger": PROCESS_TRIGGER,
        "action": PROCESS_ACTION,
        "evidence": {
            "type": "process_failure",
            "detail": (
                f"Rollout {rollout['rollout']} has no successful targeted dbt "
                "run after its last project file change."
            ),
        },
        "source_task": task,
    }


def _summarize_rollout(
    model: BedrockModel,
    prompt: str,
    rollout: dict,
) -> tuple[dict, str | None, bool]:
    error = None
    original_prompt = prompt
    for _ in range(2):
        response = _ask(model, SUMMARY_SYSTEM_PROMPT, prompt)
        try:
            summary = _parse_object(response)
            for field in (
                "successful_repairs",
                "remaining_failures",
                "infrastructure_errors",
            ):
                summary.setdefault(field, [])
                if not isinstance(summary[field], list):
                    raise ValueError(f"{field} must be a list")
            return summary, error, False
        except ValueError as exc:
            error = f"{type(exc).__name__}: {exc}"
            prompt = SUMMARY_REPAIR_PROMPT.format(
                original_prompt=original_prompt,
                response=response,
            )

    return _fallback_summary(rollout), error, True


def _fallback_summary(rollout: dict) -> dict:
    feedback = rollout.get("verifier_feedback") or ""
    failures = []
    if feedback:
        failures.append(
            {
                "failure": feedback.splitlines()[0],
                "cause": "Unknown",
                "evidence": feedback,
            }
        )
    errors = [rollout["error"]] if rollout.get("error") else []
    return {
        "successful_repairs": [],
        "remaining_failures": failures,
        "infrastructure_errors": errors,
    }


def _model(config: Config) -> BedrockModel:
    training = _training(config)
    model = config.model
    return BedrockModel(
        model_id=model.id,
        region_name=model.bedrock_region,
        boto_client_config=BotocoreConfig(
            read_timeout=model.read_timeout,
            retries={
                "mode": "standard",
                "total_max_attempts": model.request_attempts,
            },
        ),
        max_tokens=model.max_tokens,
        temperature=training.experience_temperature,
        additional_request_fields={"reasoning_effort": model.reasoning_effort},
        streaming=False,
        service_tier="flex",
    )


def _ask(model: BedrockModel, system_prompt: str, prompt: str) -> str:
    agent = Agent(
        model=model,
        tools=[],
        system_prompt=system_prompt,
        callback_handler=None,
    )
    return str(agent(prompt)).strip()


def _format_lessons(lessons: list[dict]) -> str:
    if not lessons:
        return "None"
    return "\n".join(
        f"[{lesson['id']}] Trigger: {lesson['trigger']} Action: {lesson['action']}"
        for lesson in lessons
    )


def _parse_array(response: str) -> list[dict]:
    text = response.split("```json")[-1].split("```")[0].strip()
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            raise ValueError("experience response did not contain a JSON array")
        text = text[start : end + 1]
    values = json.loads(text)
    if not isinstance(values, list) or not all(
        isinstance(value, dict) for value in values
    ):
        raise ValueError("experience response must be a JSON array of objects")
    return values


def _parse_object(response: str) -> dict:
    text = response.split("```json")[-1].split("```")[0].strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("rollout summary did not contain a JSON object")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("rollout summary must be a JSON object")
    return value


def _candidate_error(candidate: dict, task: str, mode: str) -> str | None:
    trigger = candidate.get("trigger")
    action = candidate.get("action")
    evidence = candidate.get("evidence")
    if not isinstance(trigger, str) or not isinstance(action, str):
        return "trigger and action must be text"
    if not trigger.strip() or not action.strip():
        return "trigger and action cannot be empty"
    if not isinstance(evidence, dict) or evidence.get("type") not in EVIDENCE_TYPES:
        return "candidate has no valid evidence type"
    if mode == "all_zero" and evidence.get("type") == "rollout_contrast":
        return "an all-zero group has no higher-score rollout"
    if not str(evidence.get("detail", "")).strip():
        return "candidate has no evidence detail"

    lesson = f"{trigger} {action}".lower()
    forbidden = (
        task.lower(),
        task.lower().replace("-", " "),
        "test_",
        "component_",
        "::",
    )
    if any(value and value in lesson for value in forbidden):
        return "trigger or action contains a task-specific name"
    if any(f"{root}/" in lesson for root in ("analyses", "macros", "models", "seeds")):
        return "trigger or action contains an exact project path"
    return None


def _has_process_lesson(lessons: list[dict]) -> bool:
    return any(
        normalize_lesson_text(str(lesson.get("trigger", ""))) == PROCESS_TRIGGER
        and normalize_lesson_text(str(lesson.get("action", ""))) == PROCESS_ACTION
        for lesson in lessons
    )


def _has_successful_targeted_run(rollout: dict) -> bool:
    calls = {}
    last_change = -1
    successful_runs = []

    for index, message in enumerate(rollout.get("trajectory") or []):
        for block in message.get("content") or []:
            tool_use = block.get("toolUse")
            if tool_use:
                tool_id = tool_use.get("toolUseId")
                calls[tool_id] = (index, tool_use)
                if _changes_project(tool_use):
                    last_change = index

            tool_result = block.get("toolResult")
            if not tool_result:
                continue
            call = calls.get(tool_result.get("toolUseId"))
            if not call or not _is_targeted_dbt_run(call[1]):
                continue
            output = _tool_result_text(tool_result)
            if (
                tool_result.get("status") == "success"
                and "Completed successfully" in output
                and "[exit code" not in output
            ):
                successful_runs.append(call[0])

    return any(index > last_change for index in successful_runs)


def _changes_project(tool_use: dict) -> bool:
    name = tool_use.get("name")
    inputs = tool_use.get("input") or {}
    if name in {"write_file", "edit_file"}:
        return True
    if name != "bash":
        return False
    command = str(inputs.get("command", ""))
    return bool(
        re.search(r"(?:^|[;&|]\s*)(?:cp|mv|rm|touch)\s", command)
        or re.search(r"\bsed\s+[^\n]*\s-i(?:\s|$)", command)
        or re.search(r"(?:^|[^>])>>?\s*[^&]", command)
    )


def _is_targeted_dbt_run(tool_use: dict) -> bool:
    name = tool_use.get("name")
    inputs = tool_use.get("input") or {}
    if name == "dbt":
        return (
            inputs.get("command") in {"run", "build"}
            and bool(str(inputs.get("select", "")).strip())
        )
    if name != "bash":
        return False
    command = str(inputs.get("command", ""))
    return bool(
        re.search(r"\bdbt\s+(?:run|build)\b", command)
        and re.search(r"(?:--select|-s)(?:\s|=)\S+", command)
    )


def _tool_result_text(tool_result: dict) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for item in tool_result.get("content") or []
        if isinstance(item, dict)
    )
