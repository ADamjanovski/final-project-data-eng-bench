import json
import re
from collections.abc import Collection

from strands.hooks import (
    AfterToolsEvent,
    BeforeToolsEvent,
    HookRegistry,
)

from src.agents.prompts import TOOL_INPUT_EXAMPLES, invalid_tool_call_error

TEXT_TOOL_PATTERNS = (
    re.compile(
        r"</?(?:analysis|commentary)\s+to=(?:functions\.)?"
        r"([A-Za-z_][A-Za-z0-9_-]*)"
    ),
    re.compile(
        r"<\|channel\|>(?:analysis|commentary)\s+to=(?:functions\.)?"
        r"([A-Za-z_][A-Za-z0-9_-]*)"
    ),
    re.compile(r"tool_name=<([A-Za-z_][A-Za-z0-9_-]*)"),
)

TEXT_TOOL_FIELDS = {
    "edit_file": ("path", "old_string", "new_string"),
    "write_file": ("path", "content"),
    "run_sql": ("query",),
}


def find_text_tool(message: dict, registered_tools: Collection[str]) -> str | None:
    """Identify a tool call that the model returned as plain text."""
    content = message.get("content", [])
    if any("toolUse" in block for block in content):
        return None

    text_parts = []
    for block in content:
        if isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        reasoning = block.get("reasoningContent")
        if not isinstance(reasoning, dict):
            continue
        if isinstance(reasoning.get("text"), str):
            text_parts.append(reasoning["text"])
        reasoning_text = reasoning.get("reasoningText")
        if isinstance(reasoning_text, str):
            text_parts.append(reasoning_text)
        elif isinstance(reasoning_text, dict) and isinstance(
            reasoning_text.get("text"), str
        ):
            text_parts.append(reasoning_text["text"])

    text = "\n".join(text_parts)
    for pattern in TEXT_TOOL_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1) in registered_tools:
            return match.group(1)
    for tool_name, fields in TEXT_TOOL_FIELDS.items():
        if tool_name in registered_tools and all(
            f'"{field}"' in text for field in fields
        ):
            return tool_name
    return None


class InvalidToolCallGuard:
    """Explain invalid tool input and stop repeated invalid calls."""

    def __init__(self, max_attempts: int) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self._attempts = 0
        self._stop_after_tools = False

    def register_hooks(self, registry: HookRegistry, **_kwargs: object) -> None:
        registry.add_callback(BeforeToolsEvent, self.before_tools)
        registry.add_callback(AfterToolsEvent, self.after_tools)

    def before_tools(self, event: BeforeToolsEvent) -> None:
        invalid = self._first_invalid_call(event)
        if invalid is None:
            self._attempts = 0
            self._stop_after_tools = False
            return

        tool_name, problem = invalid
        self._attempts += 1
        self._stop_after_tools = self._attempts >= self.max_attempts
        example = json.dumps(TOOL_INPUT_EXAMPLES.get(tool_name, {}))
        event.cancel = invalid_tool_call_error(
            tool_name,
            problem,
            example,
            self._attempts,
            self.max_attempts,
        )

    def after_tools(self, event: AfterToolsEvent) -> None:
        if not self._stop_after_tools:
            return
        event.end_turn = (
            f"Stopped after {self.max_attempts} consecutive invalid tool calls. "
            "The verifier will check the work completed so far."
        )
        self._stop_after_tools = False

    @staticmethod
    def _first_invalid_call(
        event: BeforeToolsEvent,
    ) -> tuple[str, str] | None:
        registry = event.agent.tool_registry.registry
        for block in event.message.get("content", []):
            tool_use = block.get("toolUse")
            if not tool_use:
                continue
            tool_name = tool_use.get("name", "")
            tool = registry.get(tool_name)
            if tool is None:
                continue
            schema = tool.tool_spec["inputSchema"]["json"]
            tool_input = tool_use.get("input")
            if not isinstance(tool_input, dict):
                return tool_name, "input must be an object"
            values = tool_input
            missing = [
                field
                for field in schema.get("required", [])
                if field not in values or values[field] in (None, "")
            ]
            unexpected = sorted(set(values) - set(schema.get("properties", {})))
            if missing:
                return tool_name, f"missing required field(s): {', '.join(missing)}"
            if unexpected:
                return tool_name, f"unsupported field(s): {', '.join(unexpected)}"
        return None
