import subprocess
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = "/app/dbt_models_duckdb"
NEW_PROJECT = "/app/dbt_project"
TRANSFORMS_PROJECT = "/app/dbt_transforms"
TRANSFORMS_PROJECT_MARKERS = (
    f"DuckDB dbt project: `{TRANSFORMS_PROJECT}",
    f"DuckDB project: `{TRANSFORMS_PROJECT}",
    f"dbt project: `{TRANSFORMS_PROJECT}",
    f"DuckDB: `{TRANSFORMS_PROJECT}",
)
DATABASE = "/app/database/retail.duckdb"
QUERY_HELPER = Path(__file__).with_name("container") / "duckdb_query.py"
MAX_OUTPUT_CHARS = 12_000

def cut(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Limit output and keep its start and end."""
    text = text or ""
    if len(text) <= limit:
        return text
    head_size = int(limit * 0.7)
    tail_size = int(limit * 0.25)
    removed = len(text) - head_size - tail_size
    return (
        f"{text[:head_size]}\n\n... [CUT: {removed} characters removed] ...\n\n"
        f"{text[-tail_size:]}"
    )


@dataclass
class TaskEnv:
    """Manage one task container and its run state."""

    task: str
    container: str | None = None
    deadline: float | None = None
    meta: dict = field(init=False, repr=False)
    project: str = field(init=False)

    def __post_init__(self) -> None:
        task_dir = REPO / "tasks" / self.task
        if not task_dir.is_dir():
            raise ValueError(f"unknown task: {self.task}")
        self.meta = tomllib.loads((task_dir / "task.toml").read_text())
        instruction = self.instruction
        if NEW_PROJECT in instruction:
            self.project = NEW_PROJECT
        elif any(marker in instruction for marker in TRANSFORMS_PROJECT_MARKERS):
            self.project = TRANSFORMS_PROJECT
        else:
            self.project = PROJECT

    @property
    def task_dir(self) -> Path:
        return REPO / "tasks" / self.task

    @property
    def instruction(self) -> str:
        return (self.task_dir / "instruction.md").read_text()

    def build_image(self) -> str:
        tag = f"deb-task:{self.task}"
        result = subprocess.run(
            ["docker", "build", "-t", tag, str(self.task_dir / "environment")],
            capture_output=True,
            text=True,
            timeout=self.meta["environment"]["build_timeout_sec"],
        )
        if result.returncode != 0:
            detail = cut((result.stdout or "") + (result.stderr or ""), 4_000)
            raise RuntimeError(f"image build failed for {self.task}:\n{detail}")
        return tag

    def start(self) -> None:
        if not QUERY_HELPER.is_file():
            raise RuntimeError(f"missing query helper: {QUERY_HELPER}")

        config = self.meta["environment"]
        self.container = f"deb-{self.task[:28]}-{uuid.uuid4().hex[:6]}"
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container,
                "--network",
                "none",
                "--cpus",
                str(config["cpus"]),
                "--memory",
                f"{config['memory_mb']}m",
                "-e",
                "DB_TYPE=duckdb",
                "-e",
                f"DUCKDB_PATH={DATABASE}",
                "--entrypoint",
                "sh",
                self.build_image(),
                "-c",
                "touch /tmp/entrypoint_ready && sleep infinity",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.container = None
            raise RuntimeError(f"container start failed: {result.stderr.strip()}")

        try:
            if self.project == TRANSFORMS_PROJECT:
                exists = subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.container,
                        "test",
                        "-e",
                        TRANSFORMS_PROJECT,
                    ],
                    capture_output=True,
                    text=True,
                )
                if exists.returncode != 0:
                    move = subprocess.run(
                        [
                            "docker",
                            "exec",
                            self.container,
                            "mv",
                            PROJECT,
                            TRANSFORMS_PROJECT,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if move.returncode != 0:
                        raise RuntimeError(
                            f"could not create project {TRANSFORMS_PROJECT}: "
                            f"{move.stderr.strip()}"
                        )
                    link = subprocess.run(
                        [
                            "docker",
                            "exec",
                            self.container,
                            "ln",
                            "-s",
                            TRANSFORMS_PROJECT,
                            PROJECT,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if link.returncode != 0:
                        raise RuntimeError(
                            f"could not create project alias {PROJECT}: "
                            f"{link.stderr.strip()}"
                        )

            if self.project == NEW_PROJECT:
                symlink = subprocess.run(
                    ["docker", "exec", self.container, "test", "-L", NEW_PROJECT],
                    capture_output=True,
                    text=True,
                )
                if symlink.returncode == 0:
                    raise RuntimeError(f"new project path is a symlink: {NEW_PROJECT}")
                mkdir = subprocess.run(
                    ["docker", "exec", self.container, "mkdir", "-p", NEW_PROJECT],
                    capture_output=True,
                    text=True,
                )
                if mkdir.returncode != 0:
                    raise RuntimeError(
                        f"could not create project {NEW_PROJECT}: {mkdir.stderr.strip()}"
                    )

            aliases = (
                []
                if self.project == TRANSFORMS_PROJECT
                else [TRANSFORMS_PROJECT]
            )
            if self.project == PROJECT:
                aliases.append(NEW_PROJECT)

            for alias in aliases:
                exists = subprocess.run(
                    ["docker", "exec", self.container, "test", "-e", alias],
                    capture_output=True,
                    text=True,
                )
                if exists.returncode == 0:
                    continue
                link = subprocess.run(
                    ["docker", "exec", self.container, "ln", "-s", PROJECT, alias],
                    capture_output=True,
                    text=True,
                )
                if link.returncode != 0:
                    raise RuntimeError(
                        f"could not create project alias {alias}: {link.stderr.strip()}"
                    )

            subprocess.run(
                ["docker", "cp", str(QUERY_HELPER), f"{self.container}:/tmp/duckdb_query.py"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            self.stop()
            raise

        self.deadline = time.monotonic() + float(self.meta["agent"]["timeout_sec"])

    def stop(self) -> None:
        if not self.container:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container],
            capture_output=True,
            text=True,
        )
        self.container = None

    def enable_verifier_network(self) -> None:
        """Give network access to the verifier after the agent stops."""
        if not self.container:
            raise RuntimeError("container is not running")
        for action, network in (("disconnect", "none"), ("connect", "bridge")):
            result = subprocess.run(
                ["docker", "network", action, network, self.container],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"could not {action} verifier network {network}: "
                    f"{result.stderr.strip()}"
                )

    def __enter__(self) -> "TaskEnv":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def exec(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        timeout: int = 300,
        input_text: str | None = None,
        output_limit: int | None = MAX_OUTPUT_CHARS,
    ) -> str:
        """Run a command in the container."""
        if not self.container:
            raise RuntimeError("container is not running")
        command = ["docker", "exec"]
        if input_text is not None:
            command.append("-i")
        if cwd:
            command.extend(["-w", cwd])
        command.append(self.container)
        command.extend(args)

        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output if output_limit is None else cut(output, output_limit)

    def shell(self, command: str, *, timeout: int = 120) -> str:
        return self.exec(
            ["bash", "-lc", command],
            cwd=self.project,
            timeout=timeout,
        )

    def resolve(self, path: str) -> str:
        """Resolve a path inside the dbt project."""
        value = (path or "").strip()
        if not value:
            raise ValueError("path is empty")
        if value.startswith("/"):
            raise ValueError("use a path relative to the dbt project")
        if any(part == ".." for part in Path(value).parts):
            raise ValueError("path may not leave the dbt project")
        return f"{self.project}/{value.removeprefix('./')}"

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("deadline reached; stop working on the task")

    def write_text(self, path: str, content: str) -> None:
        """Copy text to a resolved path in the container."""
        if not self.container:
            raise RuntimeError("container is not running")
        parent = str(Path(path).parent)
        mkdir = self.exec(["mkdir", "-p", parent], timeout=30)
        if "[exit code" in mkdir:
            raise RuntimeError(mkdir)
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        try:
            result = subprocess.run(
                ["docker", "cp", str(temporary_path), f"{self.container}:{path}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
        finally:
            temporary_path.unlink(missing_ok=True)
