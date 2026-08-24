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
Your last response wrote a call to the {tool_name} tool as text, so the tool did \
not run. Call {tool_name} now with a structured tool call and valid JSON input. \
Do not describe or print the tool call.
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
Use only lessons relevant to this ticket.

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
    return TEXT_TOOL_CALL_ERROR.format(tool_name=tool_name).strip()
