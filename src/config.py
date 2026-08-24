from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = (
    Path(__file__).with_name("configs")
    / "config-gpt-oss-120b-baseline-test.yaml"
)

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    bedrock_region: str
    temperature: float
    reasoning_effort: str
    max_tokens: int
    read_timeout: int
    request_attempts: int


@dataclass(frozen=True)
class RunConfig:
    session_prefix: str
    max_invalid_tool_calls: int
    max_token_continuations: int
    max_text_tool_continuations: int
    output_file: str
    transcript_file: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    rollouts_per_task: int
    seed: int
    experience_temperature: float
    small_notebook_threshold: int
    max_lessons_per_task: int


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    run: RunConfig
    training: TrainingConfig | None = None


def load_config(path: Path = DEFAULT_CONFIG_FILE) -> Config:
    values = yaml.safe_load(path.read_text())
    return Config(
        model=ModelConfig(**values["model"]),
        run=RunConfig(**values["run"]),
        training=(
            TrainingConfig(**values["training"])
            if "training" in values
            else None
        ),
    )


CONFIG = load_config()
