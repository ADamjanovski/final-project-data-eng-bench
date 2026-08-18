import json
from strands.hooks import AfterToolsEvent, BeforeToolsEvent, HookRegistry
from src.agents.prompts import invalid_tool_call_error


TOOL_INPUT_EXAMPLES = {
    "search_project": {"pattern": "customer_id", "file_glob": "*.sql"},
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
    "time_left": {},
}


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
