import argparse
import json
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
    extract_candidates,
    learning_modes,
)
from src.training_free_grpo.notebook import (
    append_jsonl,
    apply_operations,
    format_notebook,
    load_notebook,
    save_notebook,
    update_notebook,
)
from src.training_free_grpo.reference import get_reference
from src.training_free_grpo.rag import (
    build_clients,
    ensure_store,
    retrieve_lessons,
    sync_lessons,
)
from src.transcript import write_rollout, write_run_summary

DEFAULT_CONFIG = PROJECT_ROOT / "src/configs/config-gpt-oss-120b-training.yaml"
CREDENTIAL_ERRORS = (
    "credentials were refreshed, but the refreshed credentials are still expired",
    "expiredtoken",
    "security token included in the request is expired",
    "sso session associated with this profile has expired",
    "token has expired and refresh failed",
)


class CredentialsExpired(RuntimeError):
    pass


def credentials_expired(value: object) -> bool:
    message = str(value).lower()
    return any(error in message for error in CREDENTIAL_ERRORS)


def load_results(path: Path) -> list[Rollout]:
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        results.append(
            Rollout(
                task=record["task"],
                official=record.get("official", 0),
                counts=record.get("counts", {}),
                error=record.get("error"),
                verifier_feedback=record.get("verifier_feedback"),
                token_continuations=record.get("token_continuations", 0),
            )
        )
    return results


def run_traced_rollout(
    langfuse,
    session_id: str,
    task: str,
    instruction: str,
    lessons: str | None,
    config: Config,
    rollout_number: int,
) -> tuple[Rollout, list[dict]]:
    rollout = None
    trajectory = []
    try:
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
    except Exception as exc:
        print(f"{task}: tracing unavailable: {type(exc).__name__}: {exc}")
        if rollout is None:
            return execute_rollout(task, lessons, config)
    return rollout, trajectory


def batches(tasks: list[str], size: int):
    for start in range(0, len(tasks), size):
        yield tasks[start : start + size]


def rag_index_name(prefix: str, session_id: str) -> str:
    suffix = session_id.rsplit("-", 1)[-1].lower()
    return f"{prefix[:54].rstrip('-')}-{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Training-Free GRPO.")
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task names; default is the train split.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a training run directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training = config.training
    if training is None:
        parser.error("the selected configuration has no training section")
    if training.rollouts_per_task < 1:
        parser.error("rollouts_per_task must be at least one")
    if training.tasks_per_batch < 1:
        parser.error("tasks_per_batch must be at least one")

    if args.resume and args.tasks:
        parser.error("do not provide tasks when resuming a run")

    tasks = args.tasks or (PROJECT_ROOT / "splits/train.txt").read_text().split()
    try:
        langfuse = configure_langfuse()
    except RuntimeError as exc:
        parser.error(str(exc))

    configured_output = Path(config.run.output_file)
    session_id = (
        args.resume.name
        if args.resume
        else new_session_id(config.run.session_prefix)
    )
    run_dir = (
        args.resume.resolve()
        if args.resume
        else configured_output.parent / session_id
    )
    rollout_path = run_dir / configured_output.name
    notebook_path = run_dir / "notebook.json"
    reference_dir = run_dir / "references"
    experience_path = run_dir / "experience.jsonl"
    checkpoint_path = run_dir / "checkpoint.json"
    transcript_path = Path(config.run.transcript_file).with_name(f"{session_id}.txt")
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    completed_batches = 0
    step = 0
    batch_number = 0
    if args.resume:
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint not found: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text())
        tasks = checkpoint["tasks"]
        expected = {
            "epochs": training.epochs,
            "tasks_per_batch": training.tasks_per_batch,
            "seed": training.seed,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            parser.error("the training configuration does not match the checkpoint")
        completed_batches = checkpoint["completed_batches"]
        step = checkpoint["step"]

    rag_index = None
    if config.rag:
        rag_index = (
            checkpoint.get("rag_index")
            if checkpoint
            else rag_index_name(config.rag.index, session_id)
        )
        if not rag_index:
            parser.error("the RAG checkpoint has no vector index")
    elif checkpoint and checkpoint.get("rag_index"):
        parser.error("the checkpoint requires a RAG configuration")

    print(f"Langfuse session: {session_id}")
    print(f"Training output: {run_dir}")
    if args.resume:
        print(f"Resuming after batch {completed_batches}")
    if rag_index:
        print(f"RAG index: {rag_index}")

    if args.resume and completed_batches:
        notebook = load_notebook(
            run_dir / f"notebook-batch-{completed_batches:03d}.json"
        )
    else:
        notebook = []
    all_results = load_results(rollout_path) if args.resume else []
    if not args.resume:
        checkpoint_path.write_text(
            json.dumps(
                {
                    "tasks": tasks,
                    "epochs": training.epochs,
                    "tasks_per_batch": training.tasks_per_batch,
                    "seed": training.seed,
                    "completed_batches": 0,
                    "step": 0,
                    "rag_index": rag_index,
                },
                indent=2,
            )
            + "\n"
        )

    rag_clients = build_clients(config.model.bedrock_region) if config.rag else None

    try:
        if config.rag and rag_clients:
            bedrock, s3vectors = rag_clients
            ensure_store(
                s3vectors,
                config.rag.bucket,
                rag_index,
                config.rag,
            )
            if args.resume:
                sync_lessons(
                    bedrock,
                    s3vectors,
                    config.rag.bucket,
                    rag_index,
                    notebook,
                    config.rag,
                    update_step=completed_batches,
                )

        for epoch in range(training.epochs):
            epoch_tasks = list(tasks)
            random.Random(training.seed + epoch).shuffle(epoch_tasks)
            print(f"Epoch {epoch + 1}/{training.epochs}")

            for batch_tasks in batches(epoch_tasks, training.tasks_per_batch):
                batch_number += 1
                if batch_number <= completed_batches:
                    continue
                groups = []
                batch_records = []
                batch_results = []
                print(f"Batch {batch_number}: {', '.join(batch_tasks)}")

                for task in batch_tasks:
                    step += 1
                    task_env = TaskEnv(task)
                    instruction = task_env.instruction
                    print(f"  Step {step}: {task}")
                    retrieved = []
                    if config.rag and rag_clients:
                        bedrock, s3vectors = rag_clients
                        retrieved = retrieve_lessons(
                            bedrock,
                            s3vectors,
                            config.rag.bucket,
                            rag_index,
                            instruction,
                            config.rag,
                        )
                        lesson_ids = ", ".join(
                            lesson["id"] for lesson in retrieved
                        )
                        print(f"    retrieved: {lesson_ids or 'no lessons'}")
                        lessons = format_notebook(retrieved)
                    else:
                        lessons = format_notebook(notebook)
                    group = []

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
                        if credentials_expired(rollout.error):
                            raise CredentialsExpired(rollout.error)
                        record = {
                            **asdict(rollout),
                            "dense": rollout.dense,
                            "epoch": epoch,
                            "batch": batch_number,
                            "step": step,
                            "rollout": rollout_number,
                            "trajectory": trajectory,
                            "retrieved_lessons": [
                                lesson["id"] for lesson in retrieved
                            ],
                        }
                        group.append(record)
                        batch_records.append(record)
                        batch_results.append(rollout)
                        print(
                            f"    rollout {rollout_number}: "
                            f"official={rollout.official} dense={rollout.dense:.3f}"
                        )

                    groups.append(
                        {
                            "task": task,
                            "instruction": instruction,
                            "step": step,
                            "rollouts": group,
                            "retrieved_lessons": [
                                lesson["id"] for lesson in retrieved
                            ],
                        }
                    )

                candidates = []
                candidate_keys = set()
                extractions = []
                for group_data in groups:
                    task = group_data["task"]
                    group = group_data["rollouts"]
                    modes = learning_modes(group)
                    extraction = {
                        "modes": modes,
                        "summaries": [],
                        "candidates": [],
                        "rejected_candidates": [],
                        "skipped": "The rollout group has no learning signal.",
                    }
                    extraction_error = None
                    reference_error = None

                    if modes:
                        try:
                            reference = None
                            if "completion" in modes:
                                try:
                                    reference = get_reference(task, reference_dir)
                                except RuntimeError as exc:
                                    reference_error = str(exc)
                                    reference = {"error": reference_error}
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
                            if credentials_expired(exc):
                                raise CredentialsExpired(str(exc)) from exc
                            extraction_error = f"{type(exc).__name__}: {exc}"
                            extraction = {
                                "modes": modes,
                                "summaries": [],
                                "candidates": [],
                                "rejected_candidates": [],
                                "skipped": "Experience extraction failed.",
                                "error": extraction_error,
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
                            "extraction_error": extraction_error,
                            "reference_error": reference_error,
                            **extraction,
                        }
                    )

                try:
                    session, observation = create_experience_observation(
                        langfuse,
                        session_id=session_id,
                        task=f"batch-{batch_number}",
                        epoch=epoch,
                        step=batch_number,
                    )
                    with session:
                        with observation as trace:
                            update = update_notebook(candidates, notebook, config)
                            trace.update(output=update)
                except Exception as exc:
                    if credentials_expired(exc):
                        raise CredentialsExpired(str(exc)) from exc
                    update = {
                        "operations": [],
                        "raw_update": "",
                        "skipped": "Batch notebook update failed.",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                operations = update["operations"]
                notebook, changes = apply_operations(
                    notebook,
                    operations,
                    task=f"batch-{batch_number}",
                    limit=(
                        training.max_lessons_per_task * len(batch_tasks)
                    ),
                )
                if config.rag and rag_clients:
                    bedrock, s3vectors = rag_clients
                    try:
                        sync_lessons(
                            bedrock,
                            s3vectors,
                            config.rag.bucket,
                            rag_index,
                            notebook,
                            config.rag,
                            update_step=batch_number,
                        )
                    except Exception as exc:
                        if credentials_expired(exc):
                            raise CredentialsExpired(str(exc)) from exc
                        raise
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
                for record, result in zip(batch_records, batch_results):
                    append_jsonl(rollout_path, record)
                    write_rollout(transcript_path, result)
                all_results.extend(batch_results)
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "tasks": tasks,
                            "epochs": training.epochs,
                            "tasks_per_batch": training.tasks_per_batch,
                            "seed": training.seed,
                            "completed_batches": batch_number,
                            "step": step,
                            "rag_index": rag_index,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"  candidates: {len(candidates)}")
                print(f"  notebook lessons: {len(notebook)}")
    except CredentialsExpired as exc:
        print(f"AWS credentials expired: {exc}")
        print(f"Resume directory: {run_dir}")
        return 2
    except Exception as exc:
        if not credentials_expired(exc):
            raise
        print(f"AWS credentials expired: {exc}")
        print(f"Resume directory: {run_dir}")
        return 2
    finally:
        langfuse.flush()

    write_run_summary(transcript_path, all_results)
    passed = sum(result.official for result in all_results)
    print(f"Training rollouts passed: {passed}/{len(all_results)}")
    print(f"Final notebook lessons: {len(notebook)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
