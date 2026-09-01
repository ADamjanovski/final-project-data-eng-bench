SUMMARY_SYSTEM_PROMPT = """\
You summarize one completed dbt-agent rollout. Describe what the agent did and \
what remained wrong. The trajectory and verifier feedback are evidence. Do not \
continue the task. Do not create lessons. Do not invent results.

All rollouts use DuckDB. Ignore instruction sections that apply only to another \
backend.

Return only one JSON object:

{
  "strategy": "short description of the approach",
  "important_decisions": ["important decision and its result"],
  "validation": "what validation ran and its final result",
  "remaining_failures": ["unresolved failure and known cause"]
}

Keep the first blocking failure. Do not replace it with its downstream effects.
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

Summarize only the supplied rollout.
"""

SUMMARY_REPAIR_PROMPT = """\
Your previous response did not use the required JSON structure.

Return only one JSON object with these fields:
- strategy
- important_decisions
- validation
- remaining_failures

Use the original request and evidence already in this conversation. Do not \
repeat the evidence.
"""

CANDIDATE_SYSTEM_PROMPT = """\
You extract reusable strategy lessons from a group of dbt-agent rollouts.

There are two independent learning modes:

RELATIVE LEARNING
Compare higher-score and lower-score rollouts. Extract a lesson only when a \
decision by a higher-score rollout helps to explain the score difference. The \
evidence must say what the higher rollout did, what the lower rollout did or did \
not do, and how that difference relates to the score difference. Describe only \
observed actions. Do not replace a missing action with a different cause.

COMPLETION LEARNING
The best rollout is still incomplete. Find an important cause that it missed. \
Use at least two of these evidence sources: instruction, trajectory, verifier, \
and reference. Use these exact source names. The reference is one valid \
solution. Use its method, but do not copy task-specific code or names.

All rollouts use DuckDB. Apply only DuckDB instruction sections. If sources \
conflict, use the verifier result for the evaluated output. Return no completion \
candidate unless two sources support the same cause.

A group can produce both types of lessons. Group related test failures by their \
common cause. Do not create one lesson for each test. Keep lessons strategic and \
actionable. Good subjects include planning, tool order, validation, data grain, \
backend selection, SQL reasoning, and error recovery.

The trigger and action must apply to other tasks. Do not include a task name, \
test name, exact path, or task-specific model, table, column, value, or formula. \
Do not write vague advice such as "read carefully" or "validate the work."
Specific names can appear in evidence, but not in the trigger or action.

The trigger must be observable while the agent works. It can use the ticket, \
project files, data, or tool results. Do not make a verifier expectation or a \
hidden test expectation the trigger.

Return zero to {max_lessons} candidates. Each trigger and action together must \
contain no more than {word_limit} words. Classify each lesson as planning, \
tool_use, or domain_reasoning.

Each candidate must also contain retrieval_text. It is only a search description \
of the trigger and action. It is not a second lesson and must not add a new \
cause, method, check, or repair. It must contain {retrieval_word_limit} words or \
fewer and use this exact labeled form:

Domain: a broad business or data domain and the relevant technology. Applies \
when: a general paraphrase of the trigger that an agent can observe from the \
ticket, project, data, or tool output. Signals: visible symptoms described by \
their role, not by exact task identifiers. Strategy: a general paraphrase of \
the action with no added solution. Backend: DuckDB, Snowflake, or cross-backend.

Keep retrieval_text between 40 and {retrieval_word_limit} words. Use the source \
instruction only to understand the broad domain and visible signals. Ignore \
setup and backend boilerplate unless the lesson depends on it. Do not include \
tests, verifier results, scores, expected values, exact thresholds, task names, \
paths, code, Jinja, or task-specific model, table, column, field, and category \
names. Describe identifiers by their semantic role. Use cross-backend unless the \
action changes for DuckDB or Snowflake.

GOOD RELATIVE LESSON
Trigger: An aggregate combines sources with different grains.
Action: Confirm the required output grain and join cardinality before adding \
aggregates; then check row uniqueness.
Retrieval text: Domain: dimensional analytics with dbt SQL. Applies when: a task \
combines measures from sources at different grains. Signals: duplicate entity \
rows, inflated totals, or unstable counts after joins. Strategy: define the \
required output grain, inspect join cardinality, pre-aggregate inputs, and check \
uniqueness. Backend: cross-backend.

BAD RETRIEVAL TEXT
Signals: a test expects an exact named field to be above a numeric threshold. \
Strategy: copy the task-specific SQL expression. Backend: DuckDB.
This leaks evaluation details, exact values, and code, and it can add a strategy \
that is not present in the lesson action.

BAD RELATIVE LESSON
The lower rollout never wrote the required files, but the lesson says dbt did \
not recognize files that were created. This invents a cause.

GOOD RELATIVE CAUSE
Trigger: A rollout finishes exploration before it creates all required outputs.
Action: Create every required output before starting final validation.

GOOD COMPLETION LESSON
Trigger: Instructions contain rules for more than one database backend.
Action: Check the active backend first and apply only the schema, syntax, and \
configuration rules for that backend.

BAD GENERALIZATION
Trigger: A campaign_tier column has incorrect values.
Action: Fix campaign_tier with the required campaign rules.

BAD HIDDEN TRIGGER
Trigger: A verifier test expects a specific staging model name.
Action: Create the model with that name.

GOOD OBSERVABLE TRIGGER
Trigger: A ticket requires an exact model name for a new output.
Action: Create the model with that exact name and use it in downstream refs.

GOOD GENERALIZATION
Trigger: Several classification rules form a priority waterfall.
Action: Put the rules in one ordered decision expression and verify each boundary.

Return only one JSON array. Each object must use this form:

{{
  "kind": "relative | completion",
  "trigger": "general condition",
  "action": "general strategic action",
  "retrieval_text": "Domain: ... Applies when: ... Signals: ... Strategy: ... Backend: ...",
  "lesson_type": "planning | tool_use | domain_reasoning",
  "higher_rollouts": [1],
  "lower_rollouts": [2],
  "evidence_sources": ["instruction", "trajectory"],
  "evidence": "specific support for this lesson"
}}

For a relative candidate, provide higher_rollouts, lower_rollouts, and at least \
one evidence source. For a completion candidate, leave both rollout lists empty \
and provide at least two evidence sources. Use only these exact values: \
instruction, trajectory, verifier, reference.
"""

CANDIDATE_PROMPT = """\
ACTIVE LEARNING MODES
{modes}

TASK
{instruction}

CURRENT NOTEBOOK
{lessons}

ROLLOUT SUMMARIES
{summaries}

REFERENCE CHANGES
{reference}

Extract only general lessons with clear evidence. Return an empty array when no \
supported lesson exists.
"""

CANDIDATE_REPAIR_PROMPT = """\
Your previous response was not a valid JSON array.

Return only the corrected JSON array. Use the original evidence already in this \
conversation. Do not repeat the evidence outside the array.
"""

BATCH_UPDATE_SYSTEM_PROMPT = """\
You maintain a compact strategy notebook for a dbt coding agent. Reconcile all \
candidates from one task batch with the existing notebook.

Each lesson has a trigger, an action, retrieval_text, and a lesson type. The \
trigger and action together must contain no more than {word_limit} words. Keep \
lessons general, concrete, and useful on future tasks. Keep each trigger \
observable from the ticket, project, data, or tool output. Do not use hidden \
tests or the verifier as a trigger.

retrieval_text must be between 40 and {retrieval_word_limit} words and use this \
exact labeled form: Domain: ... Applies when: ... Signals: ... Strategy: ... \
Backend: ... . It is only a search description of the final trigger and action. \
Applies when must paraphrase the trigger. Strategy must paraphrase the action and \
must not add a cause, method, check, or repair. Do not concatenate old retrieval \
text. Do not use tests, verifier results, scores, expected values, exact \
thresholds, task names, paths, code, Jinja, or task-specific model, table, \
column, field, and category names. Describe identifiers by their semantic role. \
Use cross-backend unless the action changes for DuckDB or Snowflake.

Use these operations:
- ADD when no existing lesson has the same meaning.
- UPDATE when a candidate improves an existing lesson.
- MERGE when two or more existing lessons have the same meaning.
- DELETE when an existing lesson is contradicted or fully covered.
- NONE when no change is necessary.

Do not copy names or exact code from candidate evidence. Do not keep two lessons \
that give the same action for the same condition.

Return only one JSON array:

{{
  "operation": "ADD | UPDATE | MERGE | DELETE | NONE",
  "ids": ["existing lesson IDs"],
  "candidate_ids": ["supporting candidate IDs"],
  "trigger": "general condition",
  "action": "general strategic action",
  "retrieval_text": "Domain: ... Applies when: ... Signals: ... Strategy: ... Backend: ...",
  "lesson_type": "planning | tool_use | domain_reasoning"
}}

Use no existing IDs for ADD or NONE, one ID for UPDATE or DELETE, and two or \
more IDs for MERGE. Every ADD, UPDATE, MERGE, or DELETE must cite at least one \
supporting candidate ID. Use only the provided IDs. Return at most \
{max_operations} operations that change the notebook.
"""

BATCH_UPDATE_PROMPT = """\
EXISTING NOTEBOOK
{lessons}

CANDIDATES FROM THIS BATCH
{candidates}

Return the notebook operations for this batch.
"""

BATCH_UPDATE_REPAIR_PROMPT = """\
Your previous response was not a valid JSON array.

Return only the corrected notebook-operation array. Use the notebook and \
candidates already in this conversation.
"""
