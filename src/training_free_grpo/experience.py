import json

from src.agents.agents import build_experience_agent
from src.config import Config
from src.training_free_grpo.helpers import parse_json_response
from src.training_free_grpo.notebook import (
    LESSON_TYPES,
    LESSON_WORD_LIMIT,
    RETRIEVAL_TEXT_WORD_LIMIT,
    has_hidden_trigger,
    normalize_lesson_text,
    normalize_lesson_trigger,
    retrieval_text_error,
)
from src.training_free_grpo.prompts import (
    CANDIDATE_PROMPT,
    CANDIDATE_REPAIR_PROMPT,
    CANDIDATE_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REPAIR_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

EVIDENCE_SOURCES = {"instruction", "trajectory", "verifier", "reference"}


def learning_modes(rollouts: list[dict]) -> list[str]:
    scores = {
        rollout["dense"] for rollout in rollouts if not rollout.get("error")
    }
    if not scores:
        return []
    modes = []
    if len(scores) > 1:
        modes.append("relative")
    if max(scores) < 1.0:
        modes.append("completion")
    return modes


def extract_candidates(
    task: str,
    instruction: str,
    rollouts: list[dict],
    lessons: list[dict],
    reference: dict | None,
    config: Config,
) -> dict:
    modes = learning_modes(rollouts)
    if not modes:
        return {
            "modes": [],
            "summaries": [],
            "candidates": [],
            "rejected_candidates": [],
            "skipped": "The rollout group has no learning signal.",
        }

    usable = [rollout for rollout in rollouts if not rollout.get("error")]
    summaries = [
        _summarize_rollout(instruction, rollout, config)
        for rollout in usable
    ]
    candidate_agent = build_experience_agent(
        CANDIDATE_SYSTEM_PROMPT.format(
            max_lessons=config.training.max_lessons_per_task,
            word_limit=LESSON_WORD_LIMIT,
            retrieval_word_limit=RETRIEVAL_TEXT_WORD_LIMIT,
        ),
        config,
    )
    prompt = CANDIDATE_PROMPT.format(
        modes=", ".join(modes),
        instruction=instruction,
        lessons=_format_lessons(lessons),
        summaries=json.dumps(summaries, indent=2, ensure_ascii=False),
        reference=json.dumps(
            reference or {"status": "not available"},
            indent=2,
            ensure_ascii=False,
        ),
    )
    for attempt in range(2):
        response = str(
            candidate_agent(
                prompt if attempt == 0 else CANDIDATE_REPAIR_PROMPT
            )
        ).strip()
        try:
            parsed_candidates = parse_json_response(response, list)
            break
        except ValueError:
            if attempt == 1:
                raise

    candidates = []
    rejected = []
    for number, candidate in enumerate(parsed_candidates, start=1):
        reason = _candidate_error(candidate, modes, usable, task, reference)
        if reason:
            rejected.append({"candidate": candidate, "reason": reason})
            continue
        if len(candidates) >= config.training.max_lessons_per_task:
            rejected.append(
                {"candidate": candidate, "reason": "candidate limit reached"}
            )
            continue

        normalized = {
            "candidate_id": f"{task}:C{number}",
            "kind": candidate["kind"],
            "trigger": normalize_lesson_trigger(candidate["trigger"]),
            "action": normalize_lesson_text(candidate["action"]),
            "retrieval_text": normalize_lesson_text(
                candidate["retrieval_text"]
            ),
            "lesson_type": candidate["lesson_type"],
            "higher_rollouts": candidate.get("higher_rollouts", []),
            "lower_rollouts": candidate.get("lower_rollouts", []),
            "evidence_sources": list(
                dict.fromkeys(candidate.get("evidence_sources", []))
            ),
            "evidence": normalize_lesson_text(candidate["evidence"]),
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

    return {
        "modes": modes,
        "summaries": summaries,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "skipped": None,
    }


def _summarize_rollout(
    instruction: str,
    rollout: dict,
    config: Config,
) -> dict:
    agent = build_experience_agent(SUMMARY_SYSTEM_PROMPT, config)
    trajectory = [
        {
            "role": message.get("role"),
            "content": message.get("content") or [],
        }
        for message in rollout.get("trajectory") or []
    ]
    prompt = SUMMARY_PROMPT.format(
        instruction=instruction,
        rollout_number=rollout["rollout"],
        official=rollout["official"],
        dense=rollout["dense"],
        counts=json.dumps(rollout["counts"]),
        error=rollout.get("error") or "None",
        trajectory=json.dumps(trajectory, ensure_ascii=False, default=str),
        verifier_feedback=(
            rollout.get("verifier_feedback") or "No verifier failure was recorded."
        ),
    )
    response = ""
    error = None
    for attempt in range(2):
        message = (
            prompt
            if attempt == 0
            else SUMMARY_REPAIR_PROMPT
        )
        response = str(agent(message)).strip()
        try:
            summary = parse_json_response(response, dict)
            if not all(
                isinstance(summary.get(field), str)
                for field in ("strategy", "validation")
            ) or not all(
                isinstance(summary.get(field), list)
                and all(isinstance(item, str) for item in summary[field])
                for field in ("important_decisions", "remaining_failures")
            ):
                raise ValueError("summary has an invalid structure")
            return {
                "rollout": rollout["rollout"],
                "dense": rollout["dense"],
                "summary_error": error,
                "summary_fallback": False,
                **summary,
            }
        except ValueError as exc:
            error = f"{type(exc).__name__}: {exc}"

    return {
        "rollout": rollout["rollout"],
        "dense": rollout["dense"],
        "summary_error": error,
        "summary_fallback": True,
        "strategy": "The rollout strategy could not be summarized.",
        "important_decisions": [],
        "validation": "No structured validation summary is available.",
        "remaining_failures": [rollout.get("verifier_feedback") or "Unknown failure"],
    }


def _candidate_error(
    candidate: dict,
    modes: list[str],
    rollouts: list[dict],
    task: str,
    reference: dict | None,
) -> str | None:
    kind = candidate.get("kind")
    trigger = candidate.get("trigger")
    action = candidate.get("action")
    evidence = candidate.get("evidence")
    if kind not in modes or kind not in {"relative", "completion"}:
        return "candidate has no active learning mode"
    if not all(
        isinstance(value, str) and value.strip()
        for value in (trigger, action, evidence)
    ):
        return "candidate text is missing"
    if candidate.get("lesson_type") not in LESSON_TYPES:
        return "candidate has no valid lesson type"
    if has_hidden_trigger(trigger):
        return "candidate trigger depends on hidden evaluation"
    if len(f"{trigger} {action}".split()) > LESSON_WORD_LIMIT:
        return f"candidate is over {LESSON_WORD_LIMIT} words"
    retrieval_error = retrieval_text_error(candidate.get("retrieval_text"), task)
    if retrieval_error:
        return retrieval_error
    general_text = f"{trigger} {action}".lower()
    if any(
        value in general_text
        for value in (task.lower(), task.lower().replace("-", " "), "test_", "::")
    ):
        return "candidate contains a task-specific name"

    sources = candidate.get("evidence_sources")
    if not isinstance(sources, list) or not sources:
        return "candidate has no evidence source"
    if any(
        not isinstance(source, str) or source not in EVIDENCE_SOURCES
        for source in sources
    ):
        return "candidate has an unknown evidence source"
    available = {"instruction"}
    if any(rollout.get("trajectory") for rollout in rollouts):
        available.add("trajectory")
    if any(rollout.get("verifier_feedback") for rollout in rollouts):
        available.add("verifier")
    if reference and reference.get("changes"):
        available.add("reference")
    if any(source not in available for source in sources):
        return "candidate cites unavailable evidence"

    if kind == "relative":
        higher = candidate.get("higher_rollouts")
        lower = candidate.get("lower_rollouts")
        if not all(
            isinstance(values, list)
            and values
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            )
            for values in (higher, lower)
        ):
            return "relative candidate has no higher and lower rollout support"
        scores = {rollout["rollout"]: rollout["dense"] for rollout in rollouts}
        if any(value not in scores for value in higher + lower):
            return "relative candidate uses an unknown rollout"
        if min(scores[value] for value in higher) <= max(
            scores[value] for value in lower
        ):
            return "higher rollout does not have a higher score"
        return None

    if len(set(sources)) < 2:
        return "completion candidate needs two evidence sources"
    return None


def _format_lessons(lessons: list[dict]) -> str:
    if not lessons:
        return "None"
    return "\n".join(
        f"[{lesson['id']}] Trigger: {lesson['trigger']} Action: {lesson['action']}"
        for lesson in lessons
    )
