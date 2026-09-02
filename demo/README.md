# C0 vs C2 replay demo

This offline demo replays two captured Langfuse traces on a shared clock. It
does not call the agent, AWS, or Bedrock.

## Run the demo

From the repository root:

```bash
python3 -m http.server 8080 --directory demo
```

Open <http://localhost:8080>. Space advances or pauses, arrows step events,
number keys jump between screens, and `[` / `]` jump between chapter marks.

## Rebuild the replay data

```bash
python3 -m demo.build_replay \
  --trace demo/traces/c0-web-session-analytics.json \
  --arm c0

python3 -m demo.build_replay \
  --trace demo/traces/c2-web-session-analytics.json \
  --arm c2 \
  --notebook runs/rag-training-gpt-oss-120b/\
rag-training-gpt-oss-120b-20260830T070928Z-dc79e41a/notebook.json
```

The raw trace exports stay under `demo/traces/` and are ignored by Git. The
small generated replay JSON files under `demo/data/` are the browser assets.
