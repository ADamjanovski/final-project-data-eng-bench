SUMMARY_SYSTEM_PROMPT = """\
You analyze one completed dbt-agent rollout. The trajectory is evidence. Do not \
continue it. Do not request files or tools. Do not invent a cause.

Return only one JSON object with this form:

{
  "successful_repairs": [
    {"failure": "...", "action": "...", "evidence": "..."}
  ],
  "remaining_failures": [
    {"failure": "...", "cause": "...", "evidence": "..."}
  ],
  "infrastructure_errors": ["..."]
}

Include only evidence that can help a later rollout. A successful repair needs \
tool output that shows the specific failure disappeared after the action. Treat \
downstream failures as consequences of the first blocking failure.
"""

SUMMARY_PROMPT = """\
TASK
{instruction}

ROLLOUT {rollout_number}
Official score: {official}
Dense score: {dense:.4f}
Test counts: {counts}
Run error: {error}

TRAJECTORY
{trajectory}

VERIFIER FEEDBACK
{verifier_feedback}

Analyze only the supplied record and return the JSON object.
"""

SUMMARY_REPAIR_PROMPT = """\
Your previous response did not use the required JSON structure.

Return only one JSON object with these three list fields:
- successful_repairs
- remaining_failures
- infrastructure_errors

ORIGINAL ROLLOUT RECORD
{original_prompt}

PREVIOUS RESPONSE
{response}
"""

CANDIDATE_SYSTEM_PROMPT = """\
You extract reusable lessons from dbt-agent rollouts and a verified reference \
solution. The reference is one valid solution. Use its important method, but do \
not copy its task-specific code or names.

A candidate needs evidence from at least one source:
1. A lower-score rollout has a failure that a higher-score rollout does not \
have, and their actions explain the difference.
2. One rollout first causes a failure and then repairs that failure.
3. The verified reference contains a specific change that resolves an \
unresolved rollout failure.

The extraction mode controls the evidence:
- contrast: rollout_contrast, successful_repair, or reference_difference is \
allowed.
- all_zero: there is no winner. Do not use rollout_contrast. Use only a \
successful repair inside one rollout or a specific reference difference that \
resolves a rollout failure.

Do not create process_failure candidates. The runner checks tool results and \
creates the one global process lesson when it is needed.

Do not claim a fix from a shared unresolved error unless the reference supports \
the fix. Treat compiler failures as blocking causes. Group downstream failures \
under their first blocking cause. Ignore infrastructure, setup, profile, data, \
and formatting differences.

A successful repair is valid only when the evidence shows that the specific \
error disappeared after the action. Use exact SQL behavior. Do not present two \
operations as equivalent when they have different behavior. For example, \
COALESCE alone does not protect a division from a zero denominator.

A failed rollout proves that an error exists. It does not prove that a proposed \
repair is correct. Reject a repair that conflicts with the higher-score rollout \
or the verified reference. A malformed or unexecuted tool call cannot support a \
SQL, Jinja, join, or formula lesson.

BAD LESSON
Trigger: The ABC velocity test fails.
Action: Change avg_monthly_revenue.
This is task-specific, has no general condition, and has no proven repair.

GOOD LESSON
Trigger: A classification compares each row with one overall benchmark.
Action: Calculate the benchmark at the required grain, use one value for all \
rows, and check values near every classification boundary.

BAD PROCESS LESSON
Action: Always run dbt.

GOOD PROCESS LESSON
Trigger: A changed model has no successful targeted execution.
Action: Run the changed model, check the final status, fix the first blocking \
error, and run it again before finishing.

These examples show the required quality. Do not copy them unless the rollout \
evidence supports them.

Return zero to {max_lessons} candidates. A task can produce several candidates \
only when independent causes support them. Each trigger and action together must \
contain no more than {word_limit} words.

The trigger and action must be general. They must not contain a task name, test \
name, pytest node ID, exact file path, or task-specific model, table, or column \
name. Do not copy an identifier from the task, trajectory, verifier, or \
reference, even as an example. Exact names are permitted only in evidence.

Return only one JSON array. Each object must use this form:

{{
  "trigger": "general condition",
  "action": "general action",
  "evidence": {{
    "type": "one allowed evidence type",
    "detail": "specific supporting evidence"
  }}
}}

The allowed evidence types are rollout_contrast, successful_repair, and \
reference_difference.
"""

CANDIDATE_PROMPT = """\
EXTRACTION MODE
{mode}

TASK
{instruction}

CURRENT NOTEBOOK
{lessons}

ROLLOUT SUMMARIES
{summaries}

VERIFIED REFERENCE SOURCE CHANGES
{reference}

Extract only supported and reusable candidate lessons.
"""

BATCH_UPDATE_SYSTEM_PROMPT = """\
You maintain a compact notebook for a dbt coding agent. Reconcile all candidate \
lessons from one task batch with the existing notebook.

Each final lesson has a trigger and an action. Together they must contain no \
more than {word_limit} words. Keep lessons concrete and reusable. Reject \
unsupported, infrastructure-specific, and task-specific lessons.

Do not copy an identifier from candidate evidence into a trigger or action, \
even as an example. Replace it with its general role. Check that every SQL \
recommendation is technically correct and preserves the behavior supported by \
the evidence. COALESCE alone does not protect a division from a zero \
denominator.

Keep the trigger and action of a process_failure candidate unchanged. Select \
ADD when it is new, UPDATE when an existing process lesson needs its source, or \
NONE when the same lesson already exists.

Compare every candidate with all existing lessons. Prefer UPDATE when an \
existing lesson covers the same failure pattern. Use MERGE when two or more \
lessons overlap and one rule can replace them. Use DELETE when a lesson is \
contradicted, unsupported, task-specific, or fully covered by another lesson. \
Use ADD only when no existing lesson covers the candidate. Do not keep two \
lessons that prescribe the same action for the same condition.

Return only one JSON array. Each object must use this form:

{{
  "operation": "ADD | UPDATE | MERGE | DELETE | NONE",
  "ids": ["existing lesson IDs"],
  "trigger": "condition in which the lesson applies",
  "action": "action to take",
  "source_tasks": ["candidate source task names"]
}}

Use no IDs for ADD or NONE, one ID for UPDATE or DELETE, and two or more IDs for \
MERGE. Use only provided lesson IDs and candidate source task names. Return at \
most {max_operations} operations that change the notebook.
"""

BATCH_UPDATE_PROMPT = """\
EXISTING NOTEBOOK
{lessons}

CANDIDATES FROM THIS BATCH
{candidates}

Return the notebook operations for this batch.
"""
