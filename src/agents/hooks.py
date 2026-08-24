import json
import re

from strands.hooks import (
    AfterInvocationEvent,
    AfterToolsEvent,
    BeforeToolsEvent,
    HookRegistry,
)

from src.agents.prompts import invalid_tool_call_error, text_tool_call_error


TOOL_INPUT_EXAMPLES = {
    "read_file": {"path": "models/example.sql"},
    "write_file": {"path": "models/example.sql", "content": "select 1"},
    "edit_file": {
        "path": "models/example.sql",
        "old_string": "old text",
        "new_string": "new text",
    },
    "run_sql": {"query": "SELECT * FROM model_name LIMIT 10"},
    "dbt": {"command": "run", "select": "model_name"},
    "bash": {"command": "echo $DB_TYPE"},
}

TEXT_TOOL_PATTERNS = (
    re.compile(
        r"</?(?:analysis|commentary)\s+to=(?:functions\.)?"
        r"([A-Za-z_][A-Za-z0-9_-]*)"
    ),
    re.compile(r"tool_name=<([A-Za-z_][A-Za-z0-9_-]*)"),
)

class TextToolCallGuard:
    """Resume when the model writes a tool call as text."""

    def __init__(self, max_continuations: int) -> None:
        if max_continuations < 0:
            raise ValueError("max_continuations may not be negative")
        self.max_continuations = max_continuations
        self._continuations = 0

    def register_hooks(self, registry: HookRegistry, **_kwargs: object) -> None:
        registry.add_callback(AfterInvocationEvent, self.after_invocation)

    def after_invocation(self, event: AfterInvocationEvent) -> None:
        tool_name = self._text_tool_name(event)
        if tool_name is None or self._continuations >= self.max_continuations:
            return

        self._continuations += 1
        print(
            f"tool-like text for {tool_name}; continuing "
            f"{self._continuations}/{self.max_continuations}"
        )
        event.resume = text_tool_call_error(tool_name)

    @staticmethod
    def _text_tool_name(event: AfterInvocationEvent) -> str | None:
        result = event.result
        if result is None or result.stop_reason != "end_turn":
            return None

        content = result.message.get("content", [])
        if any("toolUse" in block for block in content):
            return None

        text_parts = [block["text"] for block in content if "text" in block]
        for block in content:
            reasoning = block.get("reasoningContent", {})
            if "text" in reasoning:
                text_parts.append(reasoning["text"])
            reasoning_text = reasoning.get("reasoningText", {})
            if "text" in reasoning_text:
                text_parts.append(reasoning_text["text"])

        text = "\n".join(text_parts)
        registered_tools = event.agent.tool_registry.registry
        for pattern in TEXT_TOOL_PATTERNS:
            match = pattern.search(text)
            if match and match.group(1) in registered_tools:
                return match.group(1)
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
