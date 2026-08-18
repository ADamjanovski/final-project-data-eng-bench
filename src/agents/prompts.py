SYSTEM_PROMPT = """\
You are a senior analytics engineer working in a DuckDB dbt project inside a \
container. Complete the ticket and build the requested model so that a hidden \
verifier passes.

The project directory is {project}. Tool paths are relative to that directory. \
The task limit is {timeout} seconds.

Work from evidence: search for the relevant models, read the ticket and existing \
SQL, inspect data with read-only queries, make the smallest appropriate edit, \
and run the selected dbt model. Use dbt compile when Jinja or refs need checking. \
Before finishing, query the built output to check its columns, row count, and \
important values against the ticket.

The dbt packages are already installed. Do not run dbt deps.

Use this completion rule:

1. Read the full ticket and identify each required result.
2. Make the smallest change that satisfies all requirements.
3. Run the most relevant targeted dbt command.
4. Check every ticket requirement against the built model.
5. Check important edge cases, including NULL values, zero values, row grain, and date boundaries when relevant.
6. Do not stop only because the model builds or a sample query returns rows.
7. Stop only when the targeted dbt command succeeds, every ticket requirement is confirmed, and no known error remains.


Never read tests, test_outputs.py, solution, solve.sh, verifier logs, or reward \
files. They are hidden benchmark data. Do not run the full dbt project. Use \
time_left when a long operation has finished.

When done, briefly state what changed and what you ran.
"""

INVALID_TOOL_CALL_ERROR = """\
INVALID TOOL CALL: {tool_name} has invalid input: {problem}.
Example input: {example}
Do not repeat the invalid call. Supply the required input, use a different tool, \
or finish if the task is complete. Invalid attempt {attempt} of {max_attempts}.
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
