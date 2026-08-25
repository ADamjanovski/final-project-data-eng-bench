import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4
from langfuse import get_client, propagate_attributes
from src.rollout import Rollout


def configure_langfuse():
    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Langfuse needs these environment variables: " + ", ".join(missing)
        )

    client = get_client()
    if not client.auth_check():
        raise RuntimeError("Langfuse rejected the configured keys")
    return client


def new_session_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def create_rollout_observation(
    client,
    *,
    session_id: str,
    task: str,
    instruction: str,
    lessons: str | None,
    model: str,
    temperature: float | None,
    trace_prefix: str = "baseline",
    rollout_number: int | None = None,
) -> tuple[Any, Any]:
    """Return the session and trace contexts for one rollout."""
    session = propagate_attributes(session_id=session_id)
    trace_name = f"{trace_prefix}:{task}"
    if rollout_number is not None:
        trace_name += f":rollout-{rollout_number}"
    trace = client.start_as_current_observation(
        name=trace_name,
        as_type="agent",
        input={
            "task": task,
            "instruction": instruction,
            "lessons": lessons or "",
            "rollout": rollout_number,
        },
        metadata={
            "model": model,
            "temperature": temperature,
            "rollout": rollout_number,
        },
    )
    return session, trace


def create_experience_observation(
    client,
    *,
    session_id: str,
    task: str,
    epoch: int,
    step: int,
) -> tuple[Any, Any]:
    session = propagate_attributes(session_id=session_id)
    trace = client.start_as_current_observation(
        name=f"experience:{task}",
        as_type="agent",
        input={"task": task, "epoch": epoch, "step": step},
    )
    return session, trace


def record_rollout(client, trace, rollout: Rollout) -> None:
    trace.update(output=asdict(rollout))
    client.score_current_trace(name="official", value=rollout.official)
    client.score_current_trace(name="dense", value=rollout.dense)
