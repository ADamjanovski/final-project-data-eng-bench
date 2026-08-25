import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

from src.agents.observation import (
    configure_langfuse,
    create_experience_observation,
    create_rollout_observation,
    new_session_id,
    record_rollout,
)
from src.config import PROJECT_ROOT, Config, load_config
from src.environment import TaskEnv
from src.rollout import Rollout, execute_rollout
from src.training_free_grpo.experience import (
    experience_mode,
    extract_candidates,
    update_notebook,
)
from src.training_free_grpo.notebook import (
    append_jsonl,
    apply_operations,
    format_notebook,
    save_notebook,
)
from src.training_free_grpo.reference import get_reference
from src.transcript import write_rollout, write_run_summary

DEFAULT_CONFIG = PROJECT_ROOT / "src/configs/config-gpt-oss-120b-training.yaml"


def run_traced_rollout(
    langfuse,
    session_id: str,
    task: str,
    instruction: str,
    lessons: str | None,
    config: Config,
    rollout_number: int,
) -> tuple[Rollout, list[dict]]:
    session, observation = create_rollout_observation(
        langfuse,
        session_id=session_id,
        task=task,
        instruction=instruction,
        lessons=lessons,
        model=config.model.id,
        temperature=config.model.temperature,
        trace_prefix="training",
        rollout_number=rollout_number,
    )
    with session:
        with observation as trace:
            rollout, trajectory = execute_rollout(task, lessons, config)
            record_rollout(langfuse, trace, rollout)
    return rollout, trajectory


def batches(tasks: list[str], size: int):
    for start in range(0, len(tasks), size):
        yield tasks[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run notebook Training-Free GRPO.")
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task names; default is the train split.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config.training
    if training is None:
        parser.error("the selected configuration has no training section")
    if training.rollouts_per_task < 1:
        parser.error("rollouts_per_task must be at least one")
    if training.tasks_per_batch < 1:
        parser.error("tasks_per_batch must be at least one")

    tasks = args.tasks or (PROJECT_ROOT / "splits/train.txt").read_text().split()
    try:
        langfuse = configure_langfuse()
    except RuntimeError as exc:
        parser.error(str(exc))

    session_id = new_session_id(config.run.session_prefix)
    configured_output = Path(config.run.output_file)
    run_dir = configured_output.parent / session_id
    rollout_path = run_dir / configured_output.name
    notebook_path = run_dir / "notebook.json"
    reference_dir = run_dir / "references"
    experience_path = run_dir / "experience.jsonl"
    transcript_path = Path(config.run.transcript_file).with_name(f"{session_id}.txt")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Langfuse session: {session_id}")
    print(f"Training output: {run_dir}")

    notebook: list[dict] = []
    all_results: list[Rollout] = []
    step = 0
    batch_number = 0

    try:
        for epoch in range(training.epochs):
            epoch_tasks = list(tasks)
            random.Random(training.seed + epoch).shuffle(epoch_tasks)
            print(f"Epoch {epoch + 1}/{training.epochs}")

            for batch_tasks in batches(epoch_tasks, training.tasks_per_batch):
                batch_number += 1
                lessons = format_notebook(notebook)
                groups = []
                print(f"Batch {batch_number}: {', '.join(batch_tasks)}")

                for task in batch_tasks:
                    step += 1
                    task_env = TaskEnv(task)
                    instruction = task_env.instruction
                    category = task_env.meta["metadata"]["category"]
                    group = []
                    print(f"  Step {step}: {task}")

                    for rollout_number in range(1, training.rollouts_per_task + 1):
                        rollout, trajectory = run_traced_rollout(
                            langfuse,
                            session_id,
                            task,
                            instruction,
                            lessons,
                            config,
                            rollout_number,
                        )
                        record = {
                            **asdict(rollout),
                            "dense": rollout.dense,
                            "epoch": epoch,
                            "batch": batch_number,
                            "step": step,
                            "rollout": rollout_number,
                            "trajectory": trajectory,
                        }
                        group.append(record)
                        all_results.append(rollout)
                        append_jsonl(rollout_path, record)
                        write_rollout(transcript_path, rollout)
                        print(
                            f"    rollout {rollout_number}: "
                            f"official={rollout.official} dense={rollout.dense:.3f}"
                        )

                    groups.append(
                        {
                            "task": task,
                            "instruction": instruction,
                            "category": category,
                            "step": step,
                            "rollouts": group,
                        }
                    )

                candidates = []
                candidate_keys = set()
                extractions = []
                for group_data in groups:
                    task = group_data["task"]
                    group = group_data["rollouts"]
                    mode = experience_mode(group)
                    extraction = {
                        "mode": mode,
                        "summaries": [],
                        "candidates": [],
                        "rejected_candidates": [],
                        "skipped": (
                            "The rollout group has no score contrast and is "
                            "not all zero."
                        ),
                    }
                    reference_error = None

                    if mode is not None:
                        try:
                            reference = get_reference(task, reference_dir)
                            session, observation = create_experience_observation(
                                langfuse,
                                session_id=session_id,
                                task=task,
                                epoch=epoch,
                                step=group_data["step"],
                            )
                            with session:
                                with observation as trace:
                                    extraction = extract_candidates(
                                        task,
                                        group_data["instruction"],
                                        group,
                                        notebook,
                                        reference,
                                        config,
                                    )
                                    trace.update(output=extraction)
                        except Exception as exc:
                            reference_error = f"{type(exc).__name__}: {exc}"
                            extraction = {
                                "mode": mode,
                                "summaries": [],
                                "candidates": [],
                                "rejected_candidates": [],
                                "skipped": "Reference or extraction failed.",
                                "error": reference_error,
                            }

                    for candidate in extraction["candidates"]:
                        key = (candidate["trigger"], candidate["action"])
                        if key not in candidate_keys:
                            candidate_keys.add(key)
                            candidates.append(candidate)
                    extractions.append(
                        {
                            "task": task,
                            "scores": [item["dense"] for item in group],
                            "reference_error": reference_error,
                            **extraction,
                        }
                    )

                session, observation = create_experience_observation(
                    langfuse,
                    session_id=session_id,
                    task=f"batch-{batch_number}",
                    epoch=epoch,
                    step=batch_number,
                )
                try:
                    with session:
                        with observation as trace:
                            update = update_notebook(candidates, notebook, config)
                            trace.update(output=update)
                except Exception as exc:
                    update = {
                        "operations": [],
                        "raw_update": "",
                        "skipped": "Batch notebook update failed.",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                categories = {
                    group["task"]: group["category"] for group in groups
                }
                operations = update["operations"]
                for operation in operations:
                    sources = [
                        task
                        for task in operation.get("source_tasks", [])
                        if task in categories
                    ]
                    if not sources:
                        sources = sorted({item["source_task"] for item in candidates})
                    operation["source_tasks"] = sources
                    operation["categories"] = sorted(
                        {categories[task] for task in sources}
                    )

                notebook, changes = apply_operations(
                    notebook,
                    operations,
                    task=f"batch-{batch_number}",
                    category="training",
                    step=step,
                    limit=training.max_lessons_per_task * len(batch_tasks),
                )
                append_jsonl(
                    experience_path,
                    {
                        "epoch": epoch,
                        "batch": batch_number,
                        "tasks": [group["task"] for group in groups],
                        "extractions": extractions,
                        "candidates": candidates,
                        **update,
                        "accepted": changes["accepted"],
                        "rejected": changes["rejected"],
                    },
                )
                save_notebook(notebook_path, notebook)
                save_notebook(
                    run_dir / f"notebook-batch-{batch_number:03d}.json",
                    notebook,
                )
                print(f"  candidates: {len(candidates)}")
                print(f"  notebook lessons: {len(notebook)}")
    finally:
        write_run_summary(transcript_path, all_results)
        langfuse.flush()

    passed = sum(result.official for result in all_results)
    print(f"Training rollouts passed: {passed}/{len(all_results)}")
    print(f"Final notebook lessons: {len(notebook)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
