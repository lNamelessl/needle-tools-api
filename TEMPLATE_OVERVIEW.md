# Deploy and Host

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/needle-tools-api)

Needle Tool-Calling API serves Cactus Compute's **Needle 3** — a 35.3 MB "Automation Foundation Model" (Apache-2.0, 115k+ downloads in its first three weeks) — as an **OpenAI-shaped HTTP API** for tool calling, structured extraction, and text embeddings. It is **not a chat LLM**: it returns grammar-guaranteed JSON tool calls and typed records, and it fits in roughly 150 MB of RAM on a free-plan-friendly 1 vCPU / 512 MB deployment.

This template ships with **zero required variables and zero credentials**. The 35 MB model and its linux-x86_64 engine are vendored inside the image, so deploys never download anything at runtime and never call Hugging Face.

## About Hosting

Hosting this template runs a single FastAPI service (one uvicorn worker) on `python:3.12-slim`. Needle's engine keeps one active agent per process, so inference is serialized behind a lock — a deliberate demo-tier design for sub-512 MB instances. The container listens on Railway's injected `PORT`, exposes `/health` (used as the deploy healthcheck), and reports resident memory so you can watch it stay far below 512 MB.

Endpoints:

- `POST /v1/chat/completions` — OpenAI tool-calls loop: send `messages`, get `tool_calls` with **stringified JSON arguments** and `finish_reason:"tool_calls"`; execute tools on your client and send `role:"tool"` results back for `finish_reason:"stop"`. `stream:true` is rejected with 400 (Needle returns complete, grammar-constrained dicts — no token streaming).
- `POST /v1/extract` — `{text, schema}` in, typed JSON out; the decode grammar guarantees the output always parses (returns `null` when nothing matches). Opt-in `"strict":true` turns ungrounded values into a 422 instead.
- `POST /v1/embeddings` — OpenAI-shaped 3072-dimension sentence vectors (unit-norm). Useful for relative similarity and routing, not contrastively trained — cosine spread is narrow.
- `GET /v1/models`, `GET /health`.

Honest disclosures you should know before deploying: the deployed **toolset is fixed per deployment** (`TOOLSET_PATH`, demo: `get_weather`, `get_time`, `search_notes` — request-level `tools` are accepted but ignored); the model is **early-stage** and can answer off-topic requests with its nearest tool at a low confidence score — the response carries `confidence`, `reasoning` and `suppressed_calls` extras, and a server-side confidence floor (0.30 default, `CONFIDENCE_FLOOR` variable) suppresses weak calls into `finish_reason:"stop"`; the context window is **8,192 tokens** with a 400 on overflow; versions are pinned (`cactus-needle==3.1.1`, engine 3.1.0) — upgrades are a re-vendor + rebuild; **telemetry is disabled** (`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1`); tool execution stays client-side, so the service has **zero SSRF surface**; responses are in English.

## Why Deploy

- **The anti-Ollama lane:** sub-512 MB, free-plan-deployable tool-calling API — most tool-calling templates on Railway drag multi-GB LLM runtimes behind them.
- **Zero configuration:** no API keys, no databases, no required variables. Deploy, get a domain, curl it.
- **Grammar-guaranteed JSON:** arguments are constrained by a byte-level grammar compiled from your schemas, so `arguments` always parse — no "sorry, the JSON was malformed" retries.
- **Deterministic deploys:** model weights and engine are pinned and vendored in the image; builds are reproducible and air-gap-friendly.
- **Same engine family as our Whistle STT API template**, so you can pair speech-to-text with tool calling using one vendor's runtime.

## Common Use Cases

- Add OpenAI-compatible function calling to an existing app without hosting a chat LLM.
- Messy-text-to-JSON extraction: invoices, bookings, notifications, forms (names, dates, amounts) with a schema you control.
- Intent routing and command parsing for agents, bots, and smart-home style UIs.
- Embedding-based search, matching and routing on edge-adjacent budgets (3072-dim vectors from a 35 MB model).
- A private, telemetry-off fallback for compliance-sensitive prototypes.

## Dependencies for

### Deployment Dependencies

None. The template has **zero required environment variables and zero external services**:

- Model weights (`needle3.cact`) and the manylinux engine are vendored in the repo/image (~37 MB).
- Runtime env baked into the image: `PORT` (Railway-injected), `NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1`, `TOOLSET_PATH=/app/tools.json`, `CONFIDENCE_FLOOR=0.30`.
- Optional variables: `CONFIDENCE_FLOOR` (refusal threshold, 0–1), `NEEDLE_SYSTEM_FACTS` (environment-facts string like `locale: en-US; user: Alex` — facts, not instructions), `TOOLSET_PATH` (mount your own toolset JSON).

Needle 3 is Apache-2.0 (Cactus Compute); this API wrapper is also Apache-2.0.
