import json
import re
import subprocess
from typing import Callable

from strands import tool

from src.environment import TaskEnv, cut

MAX_READ_LINES = 500
MAX_SQL_ROWS = 50

WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|truncate|replace|"
    r"pragma|install|load|export|import|vacuum)\b",
    re.IGNORECASE,
)

BLOCKED_PATHS = re.compile(
    r"(?:\btests?\s*/|/tests?\b|\bsolution\b|\bsolve\.sh\b|\breward\.txt\b)",
    re.IGNORECASE,
)

DBT_DEPS = re.compile(r"\bdbt\s+deps\b", re.IGNORECASE)


def _tool_call(
    env: TaskEnv,
    operation: Callable[[], str],
) -> str:
    """Check the deadline and return tool errors to the agent."""
    try:
        env.check_deadline()
        result = operation()
    except subprocess.TimeoutExpired:
        result = "TIMEOUT: command exceeded its limit."
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        result = f"ERROR: {exc}"
    return cut(result)


def build_tools(env: TaskEnv) -> list:
    """Build the deliberately small tool set for one baseline rollout."""

    @tool
    def read_file(path: str, start: int = 1, end: int = 0) -> str:
        """Read up to 500 lines from a project-relative file.

        Args:
            path: Project-relative file path.
            start: First line, starting at 1.
            end: Last line. Zero reads up to 500 lines.
        """

        def read() -> str:
            if start < 1:
                raise ValueError("start must be at least 1")
            last = end or start + MAX_READ_LINES - 1
            if last < start or last - start + 1 > MAX_READ_LINES:
                raise ValueError(f"read at most {MAX_READ_LINES} lines")
            resolved = env.resolve(path)
            output = env.exec(["sed", "-n", f"{start},{last}p", resolved], timeout=60)
            return output or f"{path}: no file or no lines in that range."

        return _tool_call(
            env,
            read,
        )

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a new file or replace the complete file.

        Args:
            path: Project-relative destination.
            content: Complete file contents.
        """

        def write() -> str:
            resolved = env.resolve(path)
            env.write_text(resolved, content)
            return f"Wrote {path} ({len(content.splitlines())} lines)."

        return _tool_call(env, write)

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace one exact occurrence in a project-relative file.

        Args:
            path: Project-relative file path.
            old_string: Exact text expected once.
            new_string: Replacement text.
        """

        def edit() -> str:
            resolved = env.resolve(path)
            current = env.exec(["cat", resolved], timeout=60, output_limit=None)
            count = current.count(old_string)
            if count != 1:
                return f"No edit made: expected one match, found {count}."
            env.write_text(resolved, current.replace(old_string, new_string, 1))
            return f"Edited {path}."

        return _tool_call(
            env,
            edit,
        )

    @tool
    def run_sql(query: str, limit: int = MAX_SQL_ROWS) -> str:
        """Run one read-only DuckDB query and return at most 200 rows.

        Args:
            query: A SELECT or WITH query.
            limit: Maximum returned rows. Must be between 1 and 200. The default is 50.
        """

        def query_duckdb() -> str:
            statement = (query or "").strip()
            if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
                raise ValueError("query must start with SELECT or WITH")
            if WRITE_SQL.search(statement):
                raise ValueError('run_sql cannot create or change tables. Write the query as a dbt model with write_file, then call dbt with {"command": "run", "select": "model_name"}.')
            row_limit = max(1, min(int(limit), 200))
            output = env.exec(
                ["python3", "/tmp/duckdb_query.py", str(row_limit)],
                timeout=300,
                input_text=statement,
                output_limit=None,
            )
            data = json.loads(output)
            if "error" in data:
                return f"SQL error: {data['error']}"
            lines = [" | ".join(data["columns"])]
            lines.extend(" | ".join(row) for row in data["rows"])
            if data["truncated"]:
                lines.append(f"[cut at {row_limit} rows]")
            return "\n".join(lines)

        return _tool_call(env, query_duckdb)

    @tool
    def dbt(command: str, select: str = "", full_refresh: bool = False) -> str:
        """Run a supported dbt command in the project.

        Args:
            command: One of parse, compile, run, test, or snapshot.
            select: dbt selection expression. Required for compile, run, and test.
            full_refresh: Rebuild incremental models when running them.
        """

        def run_dbt() -> str:
            action = command.strip().lower()
            allowed = {"parse", "compile", "run", "test", "snapshot"}
            if action not in allowed:
                raise ValueError(f"command must be one of: {', '.join(sorted(allowed))}")
            selector = select.strip()
            if action in {"compile", "run", "test"} and not selector:
                raise ValueError(f"select is required for dbt {action}")
            args = ["dbt", action, "--profiles-dir", env.project, "--project-dir", env.project,]
            if selector:
                args.extend(["--select", selector])
            if full_refresh:
                if action != "run":
                    raise ValueError("full_refresh is only valid for dbt run")
                args.append("--full-refresh")
            return env.exec(args, cwd=env.project, timeout=600)

        return _tool_call(
            env,
            run_dbt,
        )

    @tool
    def bash(command: str, timeout: int = 120) -> str:
        """Run a shell command in the dbt project.
        Use this for project search and commands that have no dedicated tool.
        Use the dedicated tools for file changes, SQL queries, and dbt commands.

        Args:
            command: Shell command.
            timeout: Maximum seconds, capped at 600.
        """

        def run_bash() -> str:
            if BLOCKED_PATHS.search(command):
                raise ValueError(
                    "blocked path: tests, solution, solve.sh, and reward files are "
                    "hidden benchmark data"
                )
            if DBT_DEPS.search(command):
                raise ValueError(
                    "dbt dependencies are already installed. Do not run dbt deps. "
                    "Continue with dbt compile or dbt run."
                )
            return env.shell(command, timeout=max(1, min(timeout, 600))) or "(no output)"

        return _tool_call(env, run_bash)

    return [
        read_file,
        write_file,
        edit_file,
        run_sql,
        dbt,
        bash,
    ]
