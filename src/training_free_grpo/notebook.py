import json
from pathlib import Path

LESSON_WORD_LIMIT = 64


def normalize_lesson_text(value: str) -> str:
    return " ".join(value.split())


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


def apply_operations(
    lessons: list[dict],
    operations: list[dict],
    *,
    task: str,
    category: str,
    step: int,
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
        ids = operation.get("ids") or []
        trigger = normalize_lesson_text(str(operation.get("trigger", "")))
        lesson_action = normalize_lesson_text(str(operation.get("action", "")))
        source_tasks = operation.get("source_tasks")
        if not isinstance(source_tasks, list) or not source_tasks or not all(
            isinstance(value, str) and value for value in source_tasks
        ):
            source_tasks = [task]
        source_categories = operation.get("categories")
        if (
            not isinstance(source_categories, list)
            or not source_categories
            or not all(
                isinstance(value, str) and value for value in source_categories
            )
        ):
            source_categories = [category]

        if action == "NONE":
            continue
        if changes >= limit:
            rejected.append({"operation": operation, "reason": "lesson limit reached"})
            continue
        if action not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            rejected.append({"operation": operation, "reason": "unknown operation"})
            continue
        if not isinstance(ids, list):
            rejected.append({"operation": operation, "reason": "ids must be a list"})
            continue

        existing = {lesson["id"]: lesson for lesson in updated}
        if action == "ADD" and ids:
            rejected.append({"operation": operation, "reason": "ADD cannot use ids"})
            continue
        if action in {"UPDATE", "DELETE"} and (
            len(ids) != 1 or ids[0] not in existing
        ):
            rejected.append({"operation": operation, "reason": "invalid lesson id"})
            continue
        if action == "MERGE" and (
            len(ids) < 2 or any(lesson_id not in existing for lesson_id in ids)
        ):
            rejected.append({"operation": operation, "reason": "invalid merge ids"})
            continue
        if action != "DELETE" and (
            not trigger
            or not lesson_action
            or len(f"{trigger} {lesson_action}".split()) > LESSON_WORD_LIMIT
        ):
            rejected.append(
                {
                    "operation": operation,
                    "reason": (
                        "lesson is empty or over "
                        f"{LESSON_WORD_LIMIT} words"
                    ),
                }
            )
            continue

        if action == "ADD" and any(
            normalize_lesson_text(str(lesson.get("trigger", ""))) == trigger
            and normalize_lesson_text(str(lesson.get("action", "")))
            == lesson_action
            for lesson in updated
        ):
            rejected.append(
                {"operation": operation, "reason": "lesson already exists"}
            )
            continue

        if action == "DELETE":
            updated = [lesson for lesson in updated if lesson["id"] != ids[0]]
        elif action == "UPDATE":
            lesson = existing[ids[0]]
            lesson["trigger"] = trigger
            lesson["action"] = lesson_action
            lesson["source_tasks"] = sorted(
                set(lesson.get("source_tasks", [])) | set(source_tasks)
            )
            lesson["categories"] = sorted(
                set(lesson.get("categories", [])) | set(source_categories)
            )
            lesson["updated_step"] = step
        else:
            sources = set(source_tasks)
            categories = set(source_categories)
            if action == "MERGE":
                for lesson_id in ids:
                    sources.update(existing[lesson_id].get("source_tasks", []))
                    categories.update(existing[lesson_id].get("categories", []))
                updated = [lesson for lesson in updated if lesson["id"] not in ids]
            updated.append(
                {
                    "id": f"L{next_number}",
                    "trigger": trigger,
                    "action": lesson_action,
                    "source_tasks": sorted(sources),
                    "categories": sorted(categories),
                    "created_step": step,
                    "updated_step": step,
                }
            )
            next_number += 1

        accepted.append(operation)
        changes += 1

    return updated, {
        "accepted": accepted,
        "rejected": rejected,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
