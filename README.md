# Needle Tool-Calling API

An **OpenAI-shaped** tools / extraction / embeddings micro-model API powered by
[Needle 3](https://github.com/cactus-compute/needle) — Cactus Compute's 35.3 MB
"Automation Foundation Model" (Apache-2.0). The **anti-Ollama**: sub-512 MB,
free-plan friendly, grammar-guaranteed JSON, zero credentials.

> **This is not a chat LLM.** Needle 3 returns tool calls, structured records
> and embeddings. Unsupported/off-topic requests come back as empty or
> low-confidence tool calls — never as invented prose.

Sibling template to [whistle-stt-api](https://github.com/lNamelessl/whistle-stt-api)
(same engine family, speech-to-text).

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/new?github_url=https://github.com/lNamelessl/needle-tools-api)

## What you get

| Endpoint | Shape | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI `chat.completion` | tools-only subset; `tool_calls[].function.arguments` is a **JSON string** per spec; `finish_reason` `"tool_calls"` / `"stop"` |
| `POST /v1/extract` | `{text, schema}` → typed JSON | grammar-guaranteed to parse (`null` when nothing matches); `"strict":true` → 422 on ungrounded values |
| `POST /v1/embeddings` | OpenAI `list` | 3072-dim unit-norm vectors |
| `GET /v1/models` | OpenAI `list` | advertises `needle3` + deployed tools |
| `GET /health` | JSON | `{"status":"ok","model_loaded":true,"rss_mb":…}` |

Non-OpenAI extras ride in a top-level `needle` object: `confidence`,
`reasoning`, `suppressed_calls`, `prefill_tps`/`decode_tps`, `peak_ram_mb`.

## Quick start

```bash
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "needle3",
  "messages": [{"role": "user", "content": "what'"'"'s the weather in Tokyo right now?"}]
}'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0_…",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Execute the tool on your client, then send the result back as a `role:"tool"`
message to get `finish_reason:"stop"`:

```python
from openai import OpenAI
client = OpenAI(base_url="https://your-service.up.railway.app/v1", api_key="unused")

r = client.chat.completions.create(model="needle3", messages=[
    {"role": "user", "content": "what's the weather in Tokyo right now?"}])
call = r.choices[0].message.tool_calls[0]          # execute get_weather yourself
result = {"city": "Tokyo", "temp_c": 18, "sky": "rain"}

r2 = client.chat.completions.create(model="needle3", messages=[
    {"role": "user", "content": "what's the weather in Tokyo right now?"},
    {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
])
assert r2.choices[0].finish_reason == "stop"
```

## OpenAI compatibility — honest subset

- **Fixed toolset per deployment.** Tools come from `TOOLSET_PATH` (demo:
  `get_weather`, `get_time`, `search_notes`). Request-level `tools` are accepted
  but ignored. Needle binds one toolset per agent; swap the JSON and redeploy.
  More than 5 declared tools switch on Needle's retrieval head (top-5 per turn).
- **No streaming.** `stream:true` → `400` (`streaming_unsupported`). Needle
  returns complete, grammar-constrained dicts — there are no token streams.
- **No free text.** After tool results the model returns `content: null` with
  `finish_reason:"stop"`; the answer is your tool results. Off-topic input
  returns empty or low-confidence calls, never invented prose.
- **Confidence floor.** Needle is early-stage: off-topic requests can produce a
  plausible-but-wrong nearest-tool call. Calls below `CONFIDENCE_FLOOR`
  (default `0.30`) are moved to `needle.suppressed_calls` and the turn finishes
  with `"stop"`. Gate on the `needle.confidence` extra for product logic.
- **Sampling flags ignored.** Grammar-constrained decoding has no
  temperature/top_p; `max_tokens` is honored (1–512).
- **Context window 8,192 tokens.** Requests whose replayed conversation
  (prefix + messages + `max_tokens`) would overflow get a clear `400`.
- **Usage numbers are estimates** (byte-level heuristic), marked in `needle.note`.
- **English responses** (`locale: en-US` system facts by default).

## Multi-turn semantics

Each request is stateless. The server replays the OpenAI tool loop natively:
last user message → `complete()`; each `role:"tool"` result →
`complete(json.dumps(result))`, exactly how Needle was trained. Earlier
user/assistant turns are folded into a short labeled context preamble.
Needle's own one-active-agent-per-process design means inference is serialized
(1 uvicorn worker + lock) — right for demo tier, not for high concurrency.

## Configuration (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `TOOLSET_PATH` | `/app/tools.json` | your toolset (flat, OpenAI-wrapped, or `{"tools":[…]}`) |
| `CONFIDENCE_FLOOR` | `0.30` | calls below this confidence are suppressed → `"stop"` |
| `NEEDLE_SYSTEM_FACTS` | `locale: en-US; device: railway-vps` | environment **facts**, not instructions (keys: `date`, `locale`, `device`, `user`, …; `date:` auto-added) |
| `PORT` | `8000` | Railway injects it at runtime |

Zero required variables, zero credentials, no volumes. Telemetry is disabled at
the image level (`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1` — the Needle binary
honors both) and `HF_HUB_OFFLINE=1` because the weights are vendored.

## Bring your own toolset

```json
{
  "tools": [{
    "name": "book_table",
    "description": "Book a restaurant table.",
    "parameters": {
      "type": "object",
      "properties": {"people": {"type": "integer"}, "when": {"type": "string", "format": "date-time"}},
      "required": ["people", "when"]
    }
  }]
}
```

Design tips from Cactus: one tool per action, names users would say, formats in
descriptions, enums/bounds in the grammar, five tools or fewer for best results.
Point `TOOLSET_PATH` at your file (mount or bake it) and redeploy.

## Vendored model & upgrades

- `vendor/needle3.cact` — 35,335,380 B, pinned.
- `vendor/libneedle3.so` — engine **3.1.0**, manylinux2014_x86_64, pinned to
  match `cactus-needle==3.1.1` (`ENGINE_VERSIONS[3]`).
- The Docker build **fails** if the pair doesn't load and answer a warm-up
  `complete("ping")` + `embed("ping")`.
- Upgrade: bump pins in `requirements.txt`/`Dockerfile`, run
  `python scripts/fetch_models.py --engine-version <new>`, commit, rebuild.

## Security posture

Tool execution happens **client-side** — the service never fetches URLs, so
there is no SSRF surface. No credentials exist to leak. Telemetry (client and
binary) is disabled. Non-root runtime user. RAM stays around 100–150 MB
(`peak_ram_mb` in every response, `rss_mb` in `/health`).

## License

Apache-2.0 (same as Needle 3).
