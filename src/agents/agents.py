from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.models import BedrockModel

from src.config import CONFIG, Config
from src.environment import PROJECT, TaskEnv
from src.agents.hooks import InvalidToolCallGuard
from src.agents.prompts import build_prompt
from src.agents.tools import build_tools


def build_agent(
    env: TaskEnv,
    lessons: str | None = None,
    config: Config = CONFIG,
) -> Agent:
    model_config = config.model
    model = BedrockModel(
        model_id=model_config.id,
        region_name=model_config.bedrock_region,
        boto_client_config=BotocoreConfig(
            read_timeout=model_config.read_timeout,
            retries={
                "mode": "standard",
                "total_max_attempts": model_config.request_attempts,
            },
        ),
        max_tokens=model_config.max_tokens,
        temperature=model_config.temperature,
        additional_request_fields={
            "reasoning_effort": model_config.reasoning_effort,
        },
        streaming=False,
    )

    prompt = build_prompt(
        project=PROJECT,
        timeout=int(env.meta["agent"]["timeout_sec"]),
        lessons=lessons,
    )
    return Agent(
        model=model,
        tools=build_tools(env),
        system_prompt=prompt,
        callback_handler=None,
        hooks=[InvalidToolCallGuard(config.run.max_invalid_tool_calls)],
    )
