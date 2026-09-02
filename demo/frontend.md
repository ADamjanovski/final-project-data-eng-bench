# Front end for the C0 vs C2 replay demo

This file specifies the demo front end. It assumes the plan in `demo.md` and
changes it in the places where the two downloaded traces turned out to be
better than that plan expected.

The front end replays two recorded runs of one task side by side. It does not
call the agent, Bedrock, or AWS during the presentation.

## Verdict on the idea

The idea is right and the data supports it better than expected. Keep all three
parts: the shared instruction as an opening message, the side-by-side tool
stream, and a click that opens the retrieved lessons.

Four changes are worth making.

1. **Align the two panes on a real shared clock, not on beats.** `demo.md`
   asked for beat alignment so the panes would not drift. In this pair the
   drift is the argument. C2 finishes in 102.8 seconds and C0 runs for 330.0
   seconds. Let C2 finish first and sit idle while C0 keeps editing. Keep the
   beats as bookmarks the scrubber can jump to.
2. **Add a third screen after the race.** The instruction message and the tool
   stream do not by themselves say *why* C2 passed. A closing screen with four
   rows of concrete difference does. That screen is the takeaway.
3. **Show the lesson block from the first frame, and put provenance behind the
   click.** The lessons are visible in the C2 pane before the first tool call,
   because that is when the agent received them. The click opens source tasks,
   lesson type, and retrieval text. This keeps `demo.md`'s rule that the always
   visible pane reads like advice rather than like a retrieval report.
4. **Show tokens, not just tool calls.** C2 paid 542 extra prompt tokens for
   the lesson block and spent 260,931 tokens in total. C0 spent 1,773,881. That
   ratio is the budget claim in the project title, and it is free to display.

Lead with the result, not with caveats. The screens below state what happened
and let the numbers carry it. See `## Framing` for the two lines of sourcing
that stay on screen and for the questions to have answers ready for.

## What this pair actually shows

Every number below was read out of the two trace files. Nothing is estimated.

| Measure | C0, no lessons | C2, 10 retrieved lessons |
|---|---:|---:|
| Official result | 0 of 22 | 22 of 22 |
| Tool calls | 85 | 28 |
| Model calls | 86 | 29 |
| Wall clock | 330.0 s | 102.8 s |
| Total tokens | 1,773,881 | 260,931 |
| First prompt tokens | 2,099 | 2,641 |
| `edit_file` calls | 27 | 2 |
| `edit_file` calls that changed nothing | 7 | 0 |
| `dbt` calls | 22 | 5 |

The two runs failed and passed for reasons that are visible in the files they
wrote, not only in the counts.

| Difference | C0 | C2 |
|---|---|---|
| Source model | `ref('stg_ga__events')`, raw page-event grain | `ref('stg_web_sessions')`, session grain |
| Schema config | `schema='web_analytics'` in the model config **and** in `profiles.yml` | `schema: web_analytics` in `profiles.yml` only |
| Resulting schema | `web_analytics_web_analytics` | `web_analytics` |
| Sessions produced | 974, expected 5029 | 5029 |
| Bounce sessions | 0, expected 2530 | 2530 |

C0 found its own schema bug at tool call 72, when
`SELECT ... FROM web_analytics.fct_session_quality` failed and the same query
against `web_analytics_web_analytics` returned 974 rows. It never fixed it. The
verifier's first error is that exact catalog error.

C0 also spent tool calls 33 to 68 on timestamp casting: `CAST(... AS
TIMESTAMP)`, then `TO_TIMESTAMP(..., 'MM/DD/YYYY')`, then `DATEDIFF('second',
...)`, with a `dbt run` between most of them. That window holds 18 `edit_file`
calls, all on `models/fct_session_quality.sql`, and 11 `dbt run` calls, and it
lasts 2 minutes 10 seconds of the run's 5 minutes 30.

## The three lessons that matter

C2 received ten lessons. Three of them describe C0's failures directly.

| Lesson | Trigger, shortened | C0 behaviour | C2 behaviour |
|---|---|---|---|
| L8 | Models materialize in a prefixed schema because the project does not define the target schema | Ends in `web_analytics_web_analytics`, verifier cannot find the tables | Sets schema in one place, lands in `web_analytics` |
| L16 | Interval or timestamp syntax unsupported on the target backend | 18 edits and 11 `dbt run` calls on one timestamp expression | Inspects one sample row, then uses a source whose columns are already typed |
| L50 | Model outputs do not match the specification for known sample sessions | Measures 974 sessions and 0 bounces, then stops | Measures 5029 and 2530 against the stated expectation, then stops |

L8 and L50 are the two defensible links. L16 is weaker and should be presented
as suggestive, not causal: C2 avoided the timestamp problem mainly because it
chose `stg_web_sessions`, whose `session_start` and `duration_seconds` columns
are already typed, so no casting was needed. C0 chose `stg_ga__events`, whose
`event_timestamp` is `VARCHAR`. If only two links are shown, drop L16.

Provenance, for the questions this will attract:

- L8 came from `dbt-customer-cltv-forecasting`,
  `dbt-customer-order-analytics`, `dbt-payment-fraud-analysis`, and
  `dbt-rfm-customer-segmentation-2`. None are web analytics. This is the
  strongest transfer claim in the demo.
- L16 came from `dbt-retail-task-07-cart-recovery` and
  `marketing-campaigns-harbor`. Also unrelated domains.
- L50 came from `web-session-quality-scoring`, which is a **train** task in the
  same domain. It is a different task: different project location, different
  profile name, different source tables, different required models. It is still
  the same domain. The provenance click already prints the source task, so the
  screen carries this without a caption. Have the distinction ready in speech.

Only 5 of the 70 lessons in the index have more than one source task. L8 is one
of them.

## Screens

Three screens, one linear path. A single spacebar or one button moves forward.

### Screen 1, the brief

```text
┌──────────────────────────────────────────────────────────────────────┐
│  recorded run, replayed at 8x                       [task dropdown]  │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ USER                                                          │  │
│  │ # Web Session Analytics Platform                              │  │
│  │ Build models that score sessions, analyze visitor behavior...  │  │
│  │ Required models: fct_session_quality, rpt_session_summary,     │  │
│  │ rpt_visitor_segments                                          │  │
│  │ Total sessions: 5029   Bounce sessions: 2530                   │  │
│  │                                        [read full instruction] │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Same model. Same task. Same instruction. One difference:            │
│                                                                      │
│  ┌─ C0 ───────────────────────┐  ┌─ C2 ────────────────────────────┐ │
│  │ no lessons                 │  │ 10 lessons retrieved            │ │
│  │ prompt: 2,099 tokens       │  │ prompt: 2,641 tokens  (+542)    │ │
│  │                            │  │ [ show the 10 lessons ]         │ │
│  └────────────────────────────┘  └─────────────────────────────────┘ │
│                                                    [ start the run ] │
└──────────────────────────────────────────────────────────────────────┘
```

Rules for this screen.

- The instruction renders as one chat message from the user. Collapse it to
  about eight lines. `read full instruction` opens the whole 5,032 characters in
  an overlay.
- Show the expected `5029` and `2530` on the collapsed card. The audience needs
  them before the race, or the closing screen means nothing.
- The two condition cards are the only asymmetry on this screen. Do not put
  results here.
- `show the 10 lessons` opens the lesson panel. Presenting from this screen is
  the intended path: read the instruction, open the lessons, start the run.

### Screen 2, the race

```text
┌──────────────────────────────────────────────────────────────────────┐
│  recorded run, replayed at 8x    elapsed 00:41 / 05:30    [4x 8x 20x]│
│  ◀◀ ▶ ⏸ ▶▶      ●───────●─────────────●──────────────●───────────    │
├───────────────────────────────────┬──────────────────────────────────┤
│ C0   no lessons                   │ C2   10 lessons retrieved        │
│ 85 tools · 1.77M tokens           │ 28 tools · 261K tokens           │
│                                   │ ┌─ LESSONS ────────────[10]───┐  │
│                                   │ │ · schema is not defined ... │  │
│                                   │ │ ◆ interval or timestamp ... │  │
│                                   │ │ · outputs do not match ...  │  │
│                                   │ │                    [expand] │  │
│                                   │ └─────────────────────────────┘  │
├───────────────────────────────────┼──────────────────────────────────┤
│ ● bash    echo $DB_TYPE           │ ● bash    echo $DB_TYPE          │
│ ● bash    ls -R /app/dbt_project  │ ● bash    ls -R /app/dbt_project │
│ ● bash    grep -R "session" ...   │ ● run_sql table_name ILIKE       │
│ ● read    fct_sessions_increm...  │           '%session%'            │
│ ● write   fct_session_quality.sql │ ● run_sql SELECT * FROM          │
│           from stg_ga__events     │           stg_web_sessions       │
│ ● dbt     compile  ✓              │ ● write   profiles.yml           │
│ ▾ ×18 edit_file · 11 dbt run      │ ● write   fct_session_quality.sql│
│        timestamp casting          │           from stg_web_sessions  │
│ ● run_sql web_analytics.fct_...   │ ● dbt     run  ✓                 │
│           ✗ Table does not exist  │ ● run_sql 5029 | 2530  ✓         │
│ ● run_sql web_analytics_web_ana...│                                  │
│           974                     │  ┌──────────────────────────┐    │
│                                   │  │  22 / 22   PASS          │    │
│                                   │  │  102.8 s                 │    │
│                                   │  └──────────────────────────┘    │
│                                   │  finished. waiting 3m 47s for C0 │
└───────────────────────────────────┴──────────────────────────────────┘
```

Rules for this screen.

- One clock drives both panes. Each pane's timeline is normalised so its own
  root observation starts at zero. The absolute start times differ, 13:31:19Z
  for C2 and 13:35:26Z for C0, and must not be used.
- C2 stops at 102.8 s and shows its verifier card. C0 keeps going to 330.0 s.
  The `waiting` line counts up. That idle pane is the point of the screen.
- Keep the real gaps between events, because they are model thinking time, but
  cap any single rendered gap at 1.5 s of screen time so nothing stalls.
- Auto-scroll each pane to its newest event. Never scroll one pane because the
  other advanced.
- Default speed 8x. C0 then runs about 41 s and C2 about 13 s, which is roughly
  one event every half second in the C0 pane. Offer 4x, 8x, and 20x.
- The header token and tool counters count up as the replay advances. They do
  not show the final total until the pane finishes.
- The verifier card appears only when a pane reaches its end.

### Screen 3, the verdict

```text
┌──────────────────────────────────────────────────────────────────────┐
│                          why C2 passed                               │
├───────────────────────────────────┬──────────────────────────────────┤
│       0 / 22   FAIL               │       22 / 22   PASS             │
│  85 tools · 330 s · 1.77M tok     │  28 tools · 103 s · 261K tok     │
├───────────────────────────────────┴──────────────────────────────────┤
│                                                                      │
│                          C0                    C2                    │
│  source model     stg_ga__events        stg_web_sessions             │
│  schema           web_analytics_        web_analytics                │
│                   web_analytics                                      │
│  sessions         974                   5029      expected 5029     │
│  bounces          0                     2530      expected 2530     │
│                                                                      │
│  C0's own words, tool call 72:                                       │
│    SELECT ... FROM web_analytics.fct_session_quality                 │
│    → Catalog Error: Table with name fct_session_quality does not     │
│      exist! Did you mean "web_analytics_web_analytics..."            │
│                                                                      │
│  C2 had this lesson in context:                                      │
│    When models materialize in the default or prefixed schema         │
│    because the project does not define the target analytics schema   │
│    → configure the explicit analytics schema before running dbt      │
│    learned from: dbt-payment-fraud-analysis and 3 other tasks        │
│                                                                      │
│  lesson → edit link: our annotation          [ diffs ]  [ sources ]  │
└──────────────────────────────────────────────────────────────────────┘
```

Rules for this screen.

- The four-row table is the strongest artifact in the demo. Give it the space.
- Quote C0's catalog error verbatim and truncate it. A paraphrase invites the
  question of what the real text said.
- The attribution line is one small caption, not a paragraph. It reads as
  sourcing, which is what it is.
- `diffs` opens the file view. Default to `models/fct_session_quality.sql`,
  which is the only file both arms wrote and the only one worth reading.

## Timeline and beats

Beats are no longer the alignment mechanism. They are scrubber bookmarks and
chapter markers.

| Beat | C0 event | C0 at | C2 event | C2 at |
|---|---:|---:|---:|---:|
| Source model chosen | 15 | 36.2 s | 13 | 29.8 s |
| First `dbt run` | 26 | 70.8 s | 19 | 47.3 s |
| Schema bug becomes visible | 71 | 253.8 s | — | — |
| Last `dbt run` | 81 | 299.0 s | 23 | 67.2 s |
| Output checked against the spec | 82 | 307.4 s | 25 | 75.7 s |

A beat with no counterpart in one pane still marks the timeline. Render it as a
tick with no event.

## The loop collapse

C0's pane has a long stretch of `edit_file` followed by `dbt run` on one file.
Rendering 36 rows of it is unreadable and misrepresents the run as noise rather
than as a specific stuck loop.

There is exactly one such group in this pair. Events 33 to 68 hold 18
`edit_file` calls, all on `models/fct_session_quality.sql`, 11 `dbt run` calls,
and 7 re-orientation calls (`read_file`, `bash grep`, `run_sql`) on the same
file. It spans 130 seconds.

Because the group is interrupted by those re-orientation calls, a rule of the
form "three or more consecutive `edit_file` or `dbt` events" splits it into
five fragments. Do not use that rule. Specify the range by hand in the build
output, alongside its label:

```json
"groups": [
  { "from": 33, "to": 68, "label": "timestamp casting",
    "path": "models/fct_session_quality.sql" }
]
```

Render it as one row:

```text
▾ ×18 edit_file · 11 dbt run   models/fct_session_quality.sql
      timestamp casting                                       2m 10s
```

The counters tick up in real replay time, so the row grows while the clock
runs. Clicking it expands the individual events.

Do not collapse in the C2 pane. It has no such stretch.

## The lesson panel

Three states.

**Collapsed**, always visible in the C2 pane header. `LESSONS 10`, plus the
first line of each trigger, up to three lines with a count for the rest.

**Expanded**, the click the demo needs. Ten cards. Each card shows the trigger
and the action, formatted as advice:

```text
When a model ranks rows within a category using window functions
→ use DENSE_RANK partitioned by the category and check the minimum rank is 1
```

No ID, no distance, no lesson type on the card face. This follows `demo.md`.

**Provenance**, a second click on one card. Shows `learned from` with the
source task names, the lesson type, and the retrieval text. Label it
`what the index stored`.

Highlight rule during the race: when the replay reaches an event listed in the
curated links, mark that lesson card with `◆` and hold the mark for the rest of
the run. Marks accumulate. Two or three of ten cards end marked. Marking all
ten would be the obvious move and it backfires — a reviewer who checks one
unmarked lesson against the trace loses confidence in the marked ones too.
Sparse marks read as selective, which is the claim.

Do not animate a lesson card into the event stream. It implies the agent cited
the lesson. It did not.

## Curated lesson links

Three links for this pair. Store them in `demo/data/links.json`.

```json
{
  "dbt-web-session-analytics": {
    "c2": [
      { "event": 10, "lesson": "L8",  "note": "schema set in one place only" },
      { "event": 8,  "lesson": "L16", "note": "reads one sample row before casting" },
      { "event": 25, "lesson": "L50", "note": "checks 5029 and 2530 against the spec" }
    ]
  }
}
```

These are our annotations, written after reading both runs. The reasoning text
in the export does not mention the lessons, so the caption on screen 3 credits
them to us rather than to the agent.

## Data pipeline

The two downloaded files are raw Langfuse per-trace exports, 15.6 MB and
2.5 MB. Do not load them in the browser.

Move them and add a build step.

```text
demo/
  frontend.md                 this file
  build_replay.py             raw trace → replay file
  traces/                     raw exports, git-ignored
    c0-web-session-analytics.json
    c2-web-session-analytics.json
  data/
    dbt-web-session-analytics__c0.json
    dbt-web-session-analytics__c2.json
    links.json
    manifest.json
  index.html
```

`traces/` already matches the `traces/` rule in the repository `.gitignore`, so
no change to that file is needed. The built files in `data/` are small and are
committed.

`build_replay.py` reads one raw trace and writes one replay file.

```bash
uv run python -m demo.build_replay \
  --trace demo/traces/c2-web-session-analytics.json \
  --arm c2 \
  --notebook runs/rag-training-gpt-oss-120b/rag-training-gpt-oss-120b-20260830T070928Z-dc79e41a/notebook.json
```

Steps.

1. Read the root `AGENT` observation. Its `input` holds `task`, `instruction`,
   and `lessons`. Its `output` holds the verifier result. Both are present in
   these two exports, so no results file and no AWS call is needed.
2. Keep the `TOOL` observations, sorted by `startTime`. Build one event each:
   name, parsed input, output text, `latency`, and offset in seconds from the
   root `startTime`.
3. Attach the model's reasoning to each event. Sort the `GENERATION`
   observations by `startTime` and take `reasoningContent.reasoningText.text`
   from the output message. The generation that precedes a tool call is the one
   that requested it. C2 has 29 generations and 28 tool calls; C0 has 86 and 85.
   The extra one is the final turn with no tool use.
4. Sum `usageDetails.input` and `usageDetails.output` over the generations, both
   as a running series and as a total. Read `modelParameters` from any
   generation for the config block.
5. Parse the rendered `lessons` string into items. The format is
   `[L8] When TRIGGER, ACTION`, one per line. Look each ID up in `--notebook`
   to attach `retrieval_text`, `lesson_type`, and `source_tasks`. Fail loudly if
   an ID is missing, because that means the wrong notebook was passed.
6. Replay `write_file` and `edit_file` inputs in order to build
   `files_changed`. `before` is the first content seen for a path, `after` the
   last.
7. Copy the hand-specified `groups` ranges and labels through to the output.
8. Mark the beats from the table above.

Only step 5 needs the notebook. Nothing in the pipeline needs credentials.

### Replay file schema

```json
{
  "task": "dbt-web-session-analytics",
  "arm": "c2",
  "config": { "model": "openai.gpt-oss-120b-1:0", "temperature": 0.2, "reasoning_effort": "medium" },
  "instruction": "...",
  "duration_s": 102.811,
  "lessons": [
    {
      "id": "L8",
      "trigger": "...",
      "action": "...",
      "lesson_type": "planning",
      "source_tasks": ["dbt-payment-fraud-analysis", "..."],
      "retrieval_text": "..."
    }
  ],
  "events": [
    {
      "index": 0,
      "t": 2.926,
      "tool": "bash",
      "input": { "command": "echo $DB_TYPE" },
      "result": "duckdb\n",
      "ok": true,
      "latency": 0.099,
      "reasoning": "We need to create dbt project with models ...",
      "beat": null
    }
  ],
  "groups": [
    { "from": 33, "to": 68, "label": "timestamp casting",
      "path": "models/fct_session_quality.sql" }
  ],
  "tokens": { "prompt_first": 2641, "input": 256978, "output": 3953, "total": 260931 },
  "files_changed": [
    { "path": "models/fct_session_quality.sql", "before": "...", "after": "...", "edits": 1 }
  ],
  "verifier": { "official": 1, "passed": 22, "failed": 0, "skipped": 0, "total": 22 },
  "totals": { "tool_calls": 28, "model_calls": 29 }
}
```

## Gotchas found in the real exports

These cost time if they are discovered while building the front end.

- **Tool input is double-encoded.** A `TOOL` observation's `input` is a JSON
  string holding a one-element list; that element's `content` is another JSON
  string holding the arguments. Its `output` is a JSON string whose `message`
  is a JSON string holding a list of `{"text": ...}` blocks.
- **The JSON contains raw control characters.** `json.loads` fails on the inner
  strings unless called with `strict=False`. This is the first thing that will
  break the build script.
- **`verifier_feedback` and `counts` disagree.** For C0 the text reads
  `0 passed, 0 skipped, 3 failed` while `counts.failed` is 22. The pytest
  listing shows 3 `FAILED` and 19 `ERROR`. `counts` includes the errors. Show
  `counts`. It gives `0 / 22` against `22 / 22`, which is the comparable pair.
  `demo.md` warned about a `0 / 3` denominator mismatch. This trace does not
  have that problem.
- **`dbt` output carries ANSI escape codes.** Every line starts with `\x1b[0m`.
  Strip them in the build script, not in the browser.
- **Lesson IDs are not stable across training runs.** L50 in the
  `dc79e41a` notebook is not L50 in `retreived.txt`, which came from a
  different index. Always pass the notebook that matches the index named in
  `src/configs/config-gpt-oss-120b-c2-retrieval-text-test.yaml`, which is
  `rag-training-gpt-oss-120b-dc79e41a`.
- **Distances are not in the export.** The root observation stores the rendered
  lesson block, not the retrieval response. Recovering distances needs a live
  `retrieve_lessons` call. The panel does not show distances, so skip it.
- **The root observation exists.** `demo.md` planned for it to be missing and
  built fallbacks. It is present in both files. Drop the fallbacks.
- **C2's first tool call is not its first action.** Both runs open with
  `echo $DB_TYPE`, which the instruction demands. Do not present that as a
  difference.
- **The reasoning text does not mention lessons.** Checked all 29 C2
  generations. This is why the links are labelled as annotations.

## Framing

Two lines of sourcing stay on screen. Both are short captions, not disclaimers,
and both exist because they are the first things a reviewer will test.

| Where | String |
|---|---|
| Header, all screens | `recorded run, replayed at 8x` |
| Screen 3, under the links | `lesson → edit link: our annotation` |

The header line is unavoidable: the panes advance faster than the wall clock
they display, and a viewer who notices that without being told stops trusting
the clock. The annotation line costs six words and buys the answer to "did the
agent say it used the lesson?" — which is coming, because the C2 reasoning text
does not mention the lessons and `tests.md` is in the repository the reviewers
have.

Everything else goes in speech, not on the screen:

- **Scope.** This is one task. Keep the 32-task aggregate on the adjacent deck
  slide so the demo is read as an example of the mechanism, and the slide as
  the measurement. The aggregate is a good number on its own: C2 passed 10 of
  32 against C1's 5 and C0's 6, at roughly half C1's cost.
- **L50's source task.** `web-session-quality-scoring` is a train task in the
  same domain. The provenance click prints it, so the screen is already
  sourced. If it comes up, the answer is that L8 and L16 came from payment
  fraud, RFM segmentation, and cart recovery, and L8 is the one that matched
  C0's actual failure.
- **Run-to-run variance.** The two C2 evaluation runs passed 10 and 8 of 32,
  and 5 tasks passed in both. If asked whether this task is cherry-picked: it
  passed C2 in both runs, which is why it was chosen over
  `dbt-gl-reconciliation`, whose swing is larger and unstable.

## Tech choices

- One `index.html`. Vanilla JavaScript. No build tooling, no framework, no CDN.
  The room may have no network.
- No diff library. A minimal LCS line diff is about 40 lines and only one file
  pair needs it.
- Tool colours, one per tool: `bash`, `read_file`, `write_file`, `edit_file`,
  `run_sql`, `dbt`. Six colours, plus one error state.
- SQL and YAML render in a monospace block with no syntax highlighting. Nothing
  in the demo depends on it, and a highlighter is a library.
- The task dropdown reads `manifest.json`. With one pair it holds one entry.
  Build it anyway so adding `dbt-weekly-sales-growth` needs no code change.
- Keyboard: space plays and pauses, arrows step one event, `1` `2` `3` jump to
  screens, `[` and `]` jump between beats. A presenter should not need a mouse.

## Build order

1. `build_replay.py`, both arms. Check the printed totals against the tables in
   this file. Every number here came from those traces, so a mismatch means the
   script is wrong.
2. Screen 2 with the event stream and the shared clock. No lessons, no
   collapse, no verifier. Confirm C2 finishes at 102.8 s and C0 at 330.0 s.
3. The loop collapse and the group labels.
4. Screen 1 and the instruction message.
5. The lesson panel, all three states.
6. `links.json` and the `◆` highlight.
7. Screen 3.
8. The diff view.
9. `manifest.json`, then convert `dbt-weekly-sales-growth` if its two traces
   still exist in Langfuse.
10. Record the fallback film, per `demo.md`.

Steps 1 and 2 are the demo. Everything after step 3 improves it. If time runs
out, a working screen 2 plus a spoken summary is still a good demo.

## Acceptance checks

- The C0 pane ends on `0 / 22` and the C2 pane on `22 / 22`.
- The C2 pane is idle for the last 227 seconds of replay time.
- The token counters end on 1,773,881 and 260,931.
- The collapsed group in the C0 pane covers events 33 to 68 and reads
  `×18 edit_file · 11 dbt run`.
- Two or three lesson cards are marked at the end of the C2 run, not ten.
- The whole demo runs with the network interface disabled.
- A cold start to the end of screen 3 takes under 90 seconds at 8x.

## Open decisions

- **Speed.** 8x is the recommendation. Watch it once at 4x before deciding; the
  C0 grind may read better slower.
- **Whether to keep the reasoning text.** It is in the build output and it makes
  each event legible, but it adds a line per row and the C0 pane is already
  long. Suggestion: show it on hover only.
- **Whether to add the C1 pair.** A three-way race is harder to read and the C1
  story is weaker. Suggestion: keep C1 in the deck as a table, not in the demo.
