import json


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


SYSTEM_PROMPT = """\
You are a senior analytics engineer working in a DuckDB dbt project inside a \
container. Complete only the requirements in the ticket.

The project directory is {project}. Tool paths are relative to that directory. \
The task limit is {timeout} seconds.

Work from evidence: use Bash to find the relevant models, read the ticket \
and existing SQL, inspect data with read-only queries, make the smallest \
appropriate edit, and run the selected dbt model. Use dbt compile when Jinja or \
refs need checking. \
Before finishing, query the built output to check its columns, row count, and \
important values against the ticket.
Stop searching after you find the required files.
Do not investigate the other backend.

The dbt packages are already installed. Do not run dbt deps. Do not end a dbt model with a semicolon.

Use this completion rule:

1. Read the full ticket and identify each required result.
2. Make the smallest change that satisfies all requirements.
3. Run the most relevant targeted dbt command.
5. Check only edge cases that the ticket requires or that follow directly from the changed logic. Combine related checks into one query. Do not invent unrelated edge cases.

After the first successful targeted dbt run:

1. Plan no more than three validation queries.
2. Use one query for the output schema, one query for output-wide checks, and one query that compares source data with the output when needed.
3. Combine several checks in each query.
5. Do not repeat a query or a check that already passed.
6. If the checks pass, stop immediately. Continue only when a result shows that the model is wrong.

Never read tests, test_outputs.py, solution, solve.sh, verifier logs, or reward \
files. They are hidden benchmark data. Do not run the full dbt project. Use \
targeted commands only.

When done, briefly state what changed and what you ran.
"""

INVALID_TOOL_CALL_ERROR = """\
INVALID TOOL CALL: {tool_name} has invalid input: {problem}.
Example input: {example}
Do not repeat the invalid call. Supply the required input, use a different tool, \
or finish if the task is complete. Invalid attempt {attempt} of {max_attempts}.
"""

TEXT_TOOL_CALL_ERROR = """\
REPAIR REQUIRED: your last response put a {tool_name} request in reasoning or \
normal text. No tool ran.

Repeat the same intended request now as one native structured tool call to the \
registered {tool_name} tool. Reuse the argument values from your last response. \
The input object must have this shape:

{example}

The call itself must be a toolUse block. Do not print the arguments in \
reasoning, normal text, Markdown, XML, a code block, `tool_name=...`, or \
`<analysis to=...>`. Do not explain the repair. Make the tool call now.
"""

TOKEN_LIMIT_CONTINUATION = """\
Your last response reached the output token limit. Continue from the current \
state. Do not repeat completed work. Use the tools to finish the remaining work.
"""

SERVICE_UNAVAILABLE_CONTINUATION = """\
The model service was temporarily unavailable. Continue from the current state. \
Do not repeat completed work.
"""

LESSON_BLOCK = """\

LEARNED EXPERIENCE
These lessons are checks from earlier rollouts. Use only lessons relevant to \
this ticket.

For each relevant lesson:
1. Check whether its trigger is present.
2. Apply its action when the trigger is present.
3. Verify the result before you finish.

The ticket has priority when a lesson conflicts with the ticket. Do not ignore \
a relevant lesson without evidence.

{lessons}
"""


def build_prompt(project: str, timeout: int, lessons: str | None = None) -> str:
    prompt = SYSTEM_PROMPT.format(project=project, timeout=timeout)
    if lessons and lessons.strip():
        prompt += LESSON_BLOCK.format(lessons=lessons.strip())
    return prompt


def invalid_tool_call_error(
    tool_name: str,
    problem: str,
    example: str,
    attempt: int,
    max_attempts: int,
) -> str:
    return INVALID_TOOL_CALL_ERROR.format(
        tool_name=tool_name,
        problem=problem,
        example=example,
        attempt=attempt,
        max_attempts=max_attempts,
    ).strip()


def text_tool_call_error(tool_name: str) -> str:
    example = json.dumps(TOOL_INPUT_EXAMPLES.get(tool_name, {}))
    return TEXT_TOOL_CALL_ERROR.format(
        tool_name=tool_name,
        example=example,
    ).strip()
