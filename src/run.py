import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.config import DEFAULT_CONFIG_FILE, PROJECT_ROOT, load_config
from src.agents.observation import (
    configure_langfuse,
    create_rollout_observation,
    new_session_id,
    record_rollout,
)
from src.rollout import Rollout, run_rollout
from src.training_free_grpo.notebook import format_notebook, load_notebook
from src.training_free_grpo.rag import build_clients, retrieve_lessons
from src.transcript import write_rollout, write_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DuckDB baseline.")
    parser.add_argument("tasks", nargs="*", help="Task names; default is the full test split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--lessons-file", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    run_config = config.run
    tasks = args.tasks or (PROJECT_ROOT / "splits/test.txt").read_text().split()
    lessons = None
    if args.lessons_file:
        try:
            lessons = format_notebook(load_notebook(args.lessons_file))
        except (OSError, TypeError, ValueError, KeyError) as exc:
            parser.error(f"invalid lessons file: {exc}")
        if not lessons:
            parser.error("lessons file contains no lessons")
    if lessons and config.rag:
        parser.error("--lessons-file cannot be used with RAG")

    rag_clients = build_clients(config.model.bedrock_region) if config.rag else None
    output_path = Path(run_config.output_file)
    transcript_path = Path(run_config.transcript_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        langfuse = configure_langfuse()
    except RuntimeError as exc:
        parser.error(str(exc))

    session_id = new_session_id(run_config.session_prefix)
    print(f"Langfuse session: {session_id}")

    results: list[Rollout] = []
    with output_path.open("a") as output:
        for task in tasks:
            print(f"Running {task}")
            try:
                instruction = (
                    PROJECT_ROOT / "tasks" / task / "instruction.md"
                ).read_text()
                task_lessons = lessons
                if config.rag and rag_clients:
                    bedrock, s3vectors = rag_clients
                    retrieved = retrieve_lessons(
                        bedrock,
                        s3vectors,
                        config.rag.bucket,
                        config.rag.index,
                        instruction,
                        config.rag,
                    )
                    task_lessons = format_notebook(retrieved)
                    lesson_ids = ", ".join(
                        lesson["id"] for lesson in retrieved
                    )
                    print(f"{task}: retrieved {lesson_ids or 'no lessons'}")
                session, observation = create_rollout_observation(
                    langfuse,
                    session_id=session_id,
                    task=task,
                    instruction=instruction,
                    lessons=task_lessons,
                    model=config.model.id,
                    temperature=config.model.temperature,
                )
                with session:
                    with observation as trace:
                        rollout = run_rollout(
                            task,
                            lessons=task_lessons,
                            config=config,
                        )
                        record_rollout(langfuse, trace, rollout)
            except Exception as exc:
                rollout = Rollout(
                    task=task,
                    error=f"infra: {type(exc).__name__}: {exc}",
                )
                print(rollout.error)
            write_rollout(transcript_path, rollout)
            results.append(rollout)
            output.write(json.dumps(asdict(rollout)) + "\n")
            output.flush()
        write_run_summary(transcript_path, results)

    langfuse.flush()

    for rollout in results:
        passed = rollout.counts.get("passed", 0)
        total = sum(rollout.counts.values())
        print(
            f"{rollout.task}: official={rollout.official} "
            f"tests={passed}/{total} error={rollout.error or '-'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
