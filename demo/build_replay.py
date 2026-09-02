"""Build small, browser-friendly replay files from Langfuse trace exports.

Run from the repository root, for example:

    python -m demo.build_replay \
      --trace demo/traces/c2-web-session-analytics.json \
      --arm c2 \
      --notebook runs/rag-training-gpt-oss-120b/\
rag-training-gpt-oss-120b-20260830T070928Z-dc79e41a/notebook.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LESSON_ID_RE = re.compile(r"^\[(L\d+)\]", re.MULTILINE)

BEATS: dict[str, dict[int, str]] = {
    "c0": {
        15: "Source model chosen",
        26: "First dbt run",
        71: "Schema bug becomes visible",
        81: "Last dbt run",
        82: "Output checked against the spec",
    },
    "c2": {
        13: "Source model chosen",
        19: "First dbt run",
        23: "Last dbt run",
        25: "Output checked against the spec",
    },
}

GROUPS: dict[str, list[dict[str, Any]]] = {
    "c0": [
        {
            "from": 33,
            "to": 68,
            "label": "timestamp casting",
            "path": "models/fct_session_quality.sql",
        }
    ],
    "c2": [],
}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value, strict=False)
    except json.JSONDecodeError:
        return value


def parse_tool_input(value: Any) -> tuple[dict[str, Any], str | None]:
    envelope = decode_json(value)
    if not isinstance(envelope, list) or not envelope:
        return {}, None
    item = envelope[0]
    if not isinstance(item, dict):
        return {}, None
    args = decode_json(item.get("content", "{}"))
    return (args if isinstance(args, dict) else {}), item.get("id")


def parse_tool_output(value: Any) -> str:
    outer = decode_json(value)
    if isinstance(outer, dict):
        message = decode_json(outer.get("message", ""))
        if isinstance(message, list):
            text = "\n".join(
                str(block.get("text", ""))
                for block in message
                if isinstance(block, dict) and block.get("text") is not None
            )
        else:
            text = str(message)
    elif outer is None:
        text = ""
    else:
        text = str(outer)
    return ANSI_RE.sub("", text).strip()


def parse_generation(value: Any) -> tuple[str, list[str]]:
    outer = decode_json(value)
    message = decode_json(outer.get("message", "[]")) if isinstance(outer, dict) else []
    reasoning = ""
    tool_ids: list[str] = []
    if not isinstance(message, list):
        return reasoning, tool_ids
    for block in message:
        if not isinstance(block, dict):
            continue
        rc = block.get("reasoningContent")
        if isinstance(rc, dict):
            rt = rc.get("reasoningText")
            if isinstance(rt, dict) and isinstance(rt.get("text"), str):
                reasoning = rt["text"].strip()
        tu = block.get("toolUse")
        if isinstance(tu, dict) and tu.get("toolUseId"):
            tool_ids.append(str(tu["toolUseId"]))
    return reasoning, tool_ids


def is_ok(result: str) -> bool:
    lowered = result.lower()
    hard_failures = (
        "catalog error",
        "compilation error",
        "runtime error",
        "traceback (most recent call last)",
        "command failed",
        "error: table with name",
        "error: use a path",
        "[exit code ",
        "no edit made:",
    )
    return not any(term in lowered for term in hard_failures)


def replay_files(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current: dict[str, str] = {}
    first: dict[str, str] = {}
    edits: Counter[str] = Counter()

    for event in events:
        args = event["input"]
        path = args.get("path")
        if not isinstance(path, str):
            continue
        if event["tool"] == "write_file":
            content = str(args.get("content", ""))
            first.setdefault(path, current.get(path, ""))
            current[path] = content
            edits[path] += 1
        elif event["tool"] == "edit_file":
            old = str(args.get("old_string", ""))
            new = str(args.get("new_string", ""))
            first.setdefault(path, current.get(path, ""))
            if path in current and old in current[path]:
                current[path] = current[path].replace(old, new, 1)
            elif path not in current:
                # This trace pair writes its shared model before editing it. Keep a
                # useful fallback for future traces where the original is absent.
                current[path] = new
            edits[path] += 1

    return [
        {
            "path": path,
            "before": first.get(path, ""),
            "after": content,
            "edits": edits[path],
        }
        for path, content in sorted(current.items())
    ]


def lesson_items(rendered: str, notebook_path: Path | None) -> list[dict[str, Any]]:
    ids = LESSON_ID_RE.findall(rendered or "")
    if not ids:
        return []
    if notebook_path is None:
        raise ValueError("This trace contains lessons; --notebook is required")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    lookup = {item["id"]: item for item in notebook}
    missing = [lesson_id for lesson_id in ids if lesson_id not in lookup]
    if missing:
        raise ValueError(f"Lesson IDs missing from notebook: {', '.join(missing)}")
    return [{key: lookup[lesson_id][key] for key in (
        "id", "trigger", "action", "lesson_type", "source_tasks", "retrieval_text"
    )} for lesson_id in ids]


def build(trace_path: Path, arm: str, notebook_path: Path | None) -> dict[str, Any]:
    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    observations = raw["observations"]
    root = next(item for item in observations if item.get("type") == "AGENT")
    root_input = decode_json(root.get("input"))
    root_output = decode_json(root.get("output"))
    if not isinstance(root_input, dict) or not isinstance(root_output, dict):
        raise ValueError("Trace root input/output is missing or invalid")

    root_start = parse_time(root["startTime"])
    tool_obs = sorted(
        (item for item in observations if item.get("type") == "TOOL"),
        key=lambda item: item["startTime"],
    )
    generations = sorted(
        (item for item in observations if item.get("type") == "GENERATION"),
        key=lambda item: item["startTime"],
    )

    generation_by_tool: dict[str, dict[str, Any]] = {}
    for generation in generations:
        reasoning, tool_ids = parse_generation(generation.get("output"))
        usage = generation.get("usageDetails") or {}
        parsed = {
            "reasoning": reasoning,
            "input_tokens": int(usage.get("input") or 0),
            "output_tokens": int(usage.get("output") or 0),
            "total_tokens": int(usage.get("total") or 0),
        }
        for tool_id in tool_ids:
            generation_by_tool[tool_id] = parsed

    events: list[dict[str, Any]] = []
    running_tokens = 0
    for index, observation in enumerate(tool_obs):
        args, tool_id = parse_tool_input(observation.get("input"))
        generation = generation_by_tool.get(tool_id or "", {})
        running_tokens += int(generation.get("total_tokens", 0))
        result = parse_tool_output(observation.get("output"))
        events.append(
            {
                "index": index,
                "t": round((parse_time(observation["startTime"]) - root_start).total_seconds(), 3),
                "tool": observation.get("name", "tool"),
                "input": args,
                "result": result,
                "ok": is_ok(result),
                "latency": round(float(observation.get("latency") or 0), 3),
                "reasoning": generation.get("reasoning", ""),
                "tokens": running_tokens,
                "beat": BEATS.get(arm, {}).get(index),
            }
        )

    usage_input = sum(int((g.get("usageDetails") or {}).get("input") or 0) for g in generations)
    usage_output = sum(int((g.get("usageDetails") or {}).get("output") or 0) for g in generations)
    counts = root_output.get("counts") or {}
    passed = int(counts.get("passed") or 0)
    failed = int(counts.get("failed") or 0)
    skipped = int(counts.get("skipped") or 0)
    rendered_lessons = str(root_input.get("lessons") or "")

    return {
        "task": root_input.get("task", "dbt-web-session-analytics"),
        "arm": arm,
        "config": {
            "model": generations[0].get("model", "openai.gpt-oss-120b-1:0"),
            "temperature": 0.2,
            "reasoning_effort": "medium",
        },
        "instruction": root_input.get("instruction", ""),
        "duration_s": round(float(root.get("latency") or 0), 3),
        "lessons": lesson_items(rendered_lessons, notebook_path),
        "events": events,
        "groups": GROUPS.get(arm, []),
        "tokens": {
            "prompt_first": int((generations[0].get("usageDetails") or {}).get("input") or 0),
            "input": usage_input,
            "output": usage_output,
            "total": usage_input + usage_output,
        },
        "files_changed": replay_files(events),
        "verifier": {
            "official": int(root_output.get("official") or 0),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped,
            "feedback": root_output.get("verifier_feedback", ""),
        },
        "totals": {"tool_calls": len(tool_obs), "model_calls": len(generations)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--arm", choices=("c0", "c2"), required=True)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    replay = build(args.trace, args.arm, args.notebook)
    output = args.output or Path("demo/data") / f"{replay['task']}__{args.arm}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"{args.arm}: {replay['totals']['tool_calls']} tools · "
        f"{replay['duration_s']:.1f}s · {replay['tokens']['total']:,} tokens · "
        f"{replay['verifier']['passed']}/{replay['verifier']['total']} tests → {output}"
    )


if __name__ == "__main__":
    main()
