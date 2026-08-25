import difflib
import json
from pathlib import Path

from src.environment import TaskEnv

MAX_FILE_CHARS = 30_000
MAX_REFERENCE_CHARS = 60_000
REFERENCE_HELPER = Path(__file__).parents[1] / "container/reference_files.py"


def get_reference(task: str, cache_dir: Path) -> dict:
    """Run the official solution once and cache its source changes."""
    cache_path = cache_dir / f"{task}.json"
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _create_reference(task)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        cache_path.write_text(
            json.dumps({"task": task, "error": error}, indent=2) + "\n"
        )
        raise RuntimeError(error) from exc
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def _create_reference(task: str) -> dict:
    environment = TaskEnv(task)
    solution_path = environment.task_dir / "solution" / "solve.sh"
    if not solution_path.is_file():
        raise RuntimeError(f"official solution is missing for {task}")

    with environment as env:
        backup = "/tmp/reference-before"
        env.write_text("/tmp/reference_files.py", REFERENCE_HELPER.read_text())
        before = _fingerprints(env, backup)
        env.write_text(
            "/tmp/official-solution.sh",
            _without_dbt_deps(solution_path.read_text()),
        )
        output = env.exec(
            ["bash", "/tmp/official-solution.sh"],
            timeout=900,
            output_limit=4_000,
        )
        if "[exit code" in output:
            raise RuntimeError(f"official solution failed:\n{output}")
        after = _fingerprints(env)

        changes = []
        reference_chars = 0
        for path in changed_paths(before, after):
            old = _read_source(env, backup, path) if path in before else ""
            new = _read_source(env, env.project, path) if path in after else ""
            difference = source_diff(path, old, new)
            remaining = MAX_REFERENCE_CHARS - reference_chars
            if remaining <= 0:
                break
            if len(difference) > remaining:
                difference = difference[:remaining] + "\n... [REFERENCE CUT]"
            changes.append({"path": path, "diff": difference})
            reference_chars += len(difference)

    if not changes:
        raise RuntimeError("official solution did not change a dbt source file")
    return {"task": task, "changes": changes}


def _fingerprints(env: TaskEnv, backup: str | None = None) -> dict[str, str]:
    args = ["python3", "/tmp/reference_files.py", env.project]
    if backup:
        args.append(backup)
    output = env.exec(
        args,
        output_limit=None,
    )
    return json.loads(output)


def _read_source(env: TaskEnv, root: str, path: str) -> str:
    content = env.exec(
        ["cat", f"{root}/{path}"],
        output_limit=None,
    )
    if len(content) > MAX_FILE_CHARS:
        return content[:MAX_FILE_CHARS] + "\n... [REFERENCE FILE CUT]"
    return content


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


def source_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"before/{path}",
            tofile=f"after/{path}",
        )
    )


def _without_dbt_deps(script: str) -> str:
    lines = []
    for line in script.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("dbt deps"):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}true  # dependencies are in the image")
        else:
            lines.append(line.replace("&& dbt deps &&", "&&"))
    return "\n".join(lines) + "\n"
