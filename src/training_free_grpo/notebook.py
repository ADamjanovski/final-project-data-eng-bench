import json
from pathlib import Path

from src.agents.agents import build_experience_agent
from src.config import Config
from src.training_free_grpo.helpers import parse_json_response
from src.training_free_grpo.prompts import (
    BATCH_UPDATE_PROMPT,
    BATCH_UPDATE_REPAIR_PROMPT,
    BATCH_UPDATE_SYSTEM_PROMPT,
)

LESSON_WORD_LIMIT = 64
RETRIEVAL_TEXT_WORD_LIMIT = 100
LESSON_TYPES = {"planning", "tool_use", "domain_reasoning"}
HIDDEN_TRIGGER_PHRASES = ("verifier", "test expects", "tests expect")
RETRIEVAL_TEXT_LABELS = (
    "domain:",
    "applies when:",
    "signals:",
    "strategy:",
    "backend:",
)


def normalize_lesson_text(value: str) -> str:
    return " ".join(value.split())


def normalize_lesson_trigger(value: str) -> str:
    trigger = normalize_lesson_text(value)
    return trigger[5:] if trigger.lower().startswith("when ") else trigger


def has_hidden_trigger(value: str) -> bool:
    trigger = value.lower()
    return any(phrase in trigger for phrase in HIDDEN_TRIGGER_PHRASES)


def retrieval_text_error(value: object, task: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "retrieval text is missing"
    text = normalize_lesson_text(value)
    words = len(text.split())
    if words < 40 or words > RETRIEVAL_TEXT_WORD_LIMIT:
        return "retrieval text must contain 40 to 100 words"
    lowered = text.lower()
    if any(label not in lowered for label in RETRIEVAL_TEXT_LABELS):
        return "retrieval text has an invalid structure"
    task_names = {task.lower(), task.lower().replace("-", " ")}
    if any(name and name in lowered for name in task_names):
        return "retrieval text contains a task-specific name"
    if any(value in lowered for value in ("verifier", "test_", "::", "/app/")):
        return "retrieval text contains hidden or task-specific details"
    return None


def load_notebook(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lessons = json.loads(path.read_text())
    if not isinstance(lessons, list):
        raise ValueError("notebook must contain a JSON list")
    return lessons


def save_notebook(path: Path, lessons: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lessons, indent=2, ensure_ascii=False) + "\n")


def format_notebook(lessons: list[dict]) -> str | None:
    if not lessons:
        return None
    return "\n".join(
        f"[{lesson['id']}] When {lesson['trigger']}, {lesson['action']}"
        for lesson in lessons
    )


def update_notebook(
    candidates: list[dict],
    lessons: list[dict],
    config: Config,
) -> dict:
    if not candidates:
        return {
            "operations": [],
            "rejected_operations": [],
            "skipped": "The batch has no supported candidates.",
        }

    agent = build_experience_agent(
        BATCH_UPDATE_SYSTEM_PROMPT.format(
            max_operations=len(candidates),
            word_limit=LESSON_WORD_LIMIT,
            retrieval_word_limit=RETRIEVAL_TEXT_WORD_LIMIT,
        ),
        config,
    )
    prompt = BATCH_UPDATE_PROMPT.format(
        lessons=json.dumps(lessons, indent=2, ensure_ascii=False),
        candidates=json.dumps(candidates, indent=2, ensure_ascii=False),
    )
    for attempt in range(2):
        response = str(
            agent(prompt if attempt == 0 else BATCH_UPDATE_REPAIR_PROMPT)
        ).strip()
        try:
            parsed_operations = parse_json_response(response, list)
            break
        except ValueError:
            if attempt == 1:
                raise

    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    operations = []
    rejected = []
    for operation in parsed_operations:
        candidate_ids = operation.get("candidate_ids", [])
        selected = (
            [
                by_id[value]
                for value in candidate_ids
                if isinstance(value, str) and value in by_id
            ]
            if isinstance(candidate_ids, list)
            else []
        )
        if (
            str(operation.get("operation", "")).upper()
            in {"ADD", "UPDATE", "MERGE", "DELETE"}
            and not selected
        ):
            rejected.append(
                {"operation": operation, "reason": "operation has no candidate"}
            )
            continue
        operation["source_tasks"] = sorted(
            {candidate["source_task"] for candidate in selected}
        )
        operations.append(operation)
    return {
        "operations": operations,
        "rejected_operations": rejected,
        "skipped": None,
    }


def apply_operations(
    lessons: list[dict],
    operations: list[dict],
    *,
    task: str,
    limit: int,
) -> tuple[list[dict], dict]:
    updated = [dict(lesson) for lesson in lessons]
    accepted = []
    rejected = []
    changes = 0
    next_number = max(
        (
            int(lesson["id"][1:])
            for lesson in updated
            if str(lesson.get("id", "")).startswith("L")
            and str(lesson["id"])[1:].isdigit()
        ),
        default=0,
    ) + 1

    for operation in operations:
        action = str(operation.get("operation", "")).upper()
        if action == "NONE":
            continue
        if changes >= limit:
            rejected.append({"operation": operation, "reason": "lesson limit reached"})
            continue
        if action not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            rejected.append({"operation": operation, "reason": "unknown operation"})
            continue

        ids = operation.get("ids") or []
        existing = {lesson["id"]: lesson for lesson in updated}
        reason = _id_error(action, ids, existing)
        if reason:
            rejected.append({"operation": operation, "reason": reason})
            continue

        trigger = normalize_lesson_trigger(str(operation.get("trigger", "")))
        lesson_action = normalize_lesson_text(str(operation.get("action", "")))
        retrieval_text = normalize_lesson_text(
            str(operation.get("retrieval_text", ""))
        )
        lesson_type = operation.get("lesson_type")
        if action != "DELETE":
            retrieval_error = retrieval_text_error(retrieval_text, task)
            if (
                not trigger
                or not lesson_action
                or has_hidden_trigger(trigger)
                or lesson_type not in LESSON_TYPES
                or len(f"{trigger} {lesson_action}".split()) > LESSON_WORD_LIMIT
                or retrieval_error
            ):
                rejected.append(
                    {
                        "operation": operation,
                        "reason": retrieval_error or "lesson content is invalid",
                    }
                )
                continue
        if action == "ADD" and _has_lesson(updated, trigger, lesson_action):
            rejected.append(
                {"operation": operation, "reason": "lesson already exists"}
            )
            continue

        sources = _source_tasks(operation.get("source_tasks"), task)
        if action == "DELETE":
            updated = [lesson for lesson in updated if lesson["id"] != ids[0]]
        elif action == "UPDATE":
            lesson = existing[ids[0]]
            lesson.update(
                trigger=trigger,
                action=lesson_action,
                retrieval_text=retrieval_text,
                lesson_type=lesson_type,
                source_tasks=sorted(
                    set(lesson.get("source_tasks", [])) | set(sources)
                ),
            )
        else:
            if action == "MERGE":
                for lesson_id in ids:
                    sources.extend(existing[lesson_id].get("source_tasks", []))
                updated = [lesson for lesson in updated if lesson["id"] not in ids]
            updated.append(
                {
                    "id": f"L{next_number}",
                    "trigger": trigger,
                    "action": lesson_action,
                    "retrieval_text": retrieval_text,
                    "lesson_type": lesson_type,
                    "source_tasks": sorted(set(sources)),
                }
            )
            next_number += 1

        accepted.append(operation)
        changes += 1

    return updated, {"accepted": accepted, "rejected": rejected}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _id_error(action: str, ids: object, existing: dict) -> str | None:
    if not isinstance(ids, list):
        return "ids must be a list"
    if action == "ADD" and ids:
        return "ADD cannot use ids"
    if action in {"UPDATE", "DELETE"} and (
        len(ids) != 1 or ids[0] not in existing
    ):
        return "invalid lesson id"
    if action == "MERGE" and (
        len(ids) < 2 or any(lesson_id not in existing for lesson_id in ids)
    ):
        return "invalid merge ids"
    return None


def _has_lesson(lessons: list[dict], trigger: str, action: str) -> bool:
    return any(
        normalize_lesson_text(str(lesson.get("trigger", ""))) == trigger
        and normalize_lesson_text(str(lesson.get("action", ""))) == action
        for lesson in lessons
    )


def _source_tasks(value: object, fallback: str) -> list[str]:
    if not isinstance(value, list):
        return [fallback]
    sources = [item for item in value if isinstance(item, str) and item]
    return sources or [fallback]
