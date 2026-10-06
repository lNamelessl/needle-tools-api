"""FastAPI application: OpenAI-shaped tools/extraction/embeddings over Needle 3.

Not a chat LLM: Needle 3 is a 35 MB automation foundation model that returns
tool calls, structured extractions and embeddings. Unsupported requests come
back as empty tool calls (or confidence-gated suppressed calls), never as
invented prose.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from . import openai_map as om
from .engine import CONTEXT_TOKENS, NeedleEngine, rss_mb
from .toolset import ToolsetError, load_toolset

log = logging.getLogger("needle-api")

SYSTEM_FACTS = os.environ.get("NEEDLE_SYSTEM_FACTS", "locale: en-US; device: railway-vps")
CONFIDENCE_FLOOR = float(os.environ.get("CONFIDENCE_FLOOR", "0.30"))
TOOLSET_PATH = os.environ.get("TOOLSET_PATH", "/app/tools.json")

STATE = {"ready": False, "error": None, "engine": None, "tools": [], "started": time.time()}
MODEL_CARD = {
    "id": om.MODEL_ID,
    "object": "model",
    "created": int(time.time()),
    "owned_by": "cactus-compute",
    "needle": {
        "generation": 3,
        "package": "cactus-needle==3.1.1",
        "engine": "3.1.0",
        "weights": "needle3.cact (35,335,380 B, Apache-2.0)",
        "capabilities": ["tool_calls", "extraction", "embeddings"],
        "context_tokens": CONTEXT_TOKENS,
        "embed_dimensions": None,       # filled after boot warm-up
        "confidence_floor": CONFIDENCE_FLOOR,
        "chat_model": False,            # NOT a chat LLM; no free text
        "streaming": False,
    },
}


@contextlib.asynccontextmanager
async def lifespan(app):
    try:
        tools = load_toolset(TOOLSET_PATH)
        log.info("toolset loaded from %s: %s", TOOLSET_PATH, [t["name"] for t in tools])
        engine = NeedleEngine(tools, SYSTEM_FACTS, confidence_floor=CONFIDENCE_FLOOR)
        MODEL_CARD["needle"]["embed_dimensions"] = engine.embed_dim
        STATE.update(ready=True, engine=engine, tools=[t["name"] for t in tools], error=None)
        log.info("engine ready: embed_dim=%s prefix_tokens=%s",
                 engine.embed_dim, engine._prefix_tokens)
    except Exception as exc:  # pragma: no cover - boot failure surfaces in /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("boot failed")
    yield
    STATE["engine"] = None


app = FastAPI(title="Needle Tool-Calling API", version=__version__, lifespan=lifespan)


@app.exception_handler(om.ApiError)
async def api_error_handler(request, exc: om.ApiError):
    return JSONResponse(status_code=exc.status, content=exc.body())


async def _payload(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        raise om.ApiError(400, "request body must be valid JSON")


@app.get("/")
async def root():
    return {"service": "Needle Tool-Calling API", "version": __version__,
            "model": om.MODEL_ID, "endpoints": ["/v1/chat/completions", "/v1/extract",
                                                "/v1/embeddings", "/v1/models", "/health"]}


@app.get("/health")
async def health():
    body = {
        "status": "ok" if STATE["ready"] else "unavailable",
        "model_loaded": bool(STATE["engine"] is not None),
        "rss_mb": rss_mb(),
        "uptime_s": round(time.time() - STATE["started"], 1),
        "telemetry": "disabled",
        "env": {
            "NEEDLE_TELEMETRY": os.environ.get("NEEDLE_TELEMETRY"),
            "DO_NOT_TRACK": os.environ.get("DO_NOT_TRACK"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "CONFIDENCE_FLOOR": os.environ.get("CONFIDENCE_FLOOR"),
            "TOOLSET_PATH": TOOLSET_PATH,
        },
        "tools": STATE["tools"],
    }
    if STATE["engine"] is not None:
        body["embed_dim"] = STATE["engine"].embed_dim
    if not STATE["ready"]:
        body["boot_error"] = STATE["error"]
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/v1/models")
async def list_models():
    card = dict(MODEL_CARD)
    card["needle"] = dict(MODEL_CARD["needle"], tools=STATE["tools"])
    return om.models_response(card)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not STATE["ready"]:
        raise om.ApiError(503, "engine is still booting", err_type="service_unavailable")
    parsed = om.parse_chat_request(await _payload(request))
    engine = STATE["engine"]
    try:
        final = engine.chat(parsed["messages"], max_new_tokens=parsed["max_new_tokens"])
    except OverflowError as exc:
        raise om.ApiError(400, str(exc), code="context_overflow")
    except ValueError as exc:
        raise om.ApiError(400, str(exc))
    except RuntimeError as exc:
        raise om.ApiError(502, f"engine error: {exc}", err_type="engine_error")
    return om.chat_response(final, MODEL_CARD, parsed["messages"],
                            parsed["max_new_tokens"], engine._prefix_tokens)


@app.post("/v1/extract")
async def extract(request: Request):
    if not STATE["ready"]:
        raise om.ApiError(503, "engine is still booting", err_type="service_unavailable")
    parsed = om.parse_extract_request(await _payload(request))
    engine = STATE["engine"]
    try:
        data = engine.extract(parsed["text"], parsed["schema_tool"],
                              strict=parsed["strict"],
                              max_new_tokens=parsed["max_new_tokens"])
    except Exception as exc:
        # strict=True can raise ExtractionValidationError: surface it cleanly.
        name = type(exc).__name__
        if name == "ExtractionValidationError":
            raise om.ApiError(422, f"extraction failed grounding validation: {exc}",
                              err_type="extraction_validation_error", code="ungrounded")
        raise om.ApiError(502, f"engine error: {exc}", err_type="engine_error")
    return om.extract_response(data)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    if not STATE["ready"]:
        raise om.ApiError(503, "engine is still booting", err_type="service_unavailable")
    parsed = om.parse_embeddings_request(await _payload(request))
    engine = STATE["engine"]
    vectors = []
    for text in parsed["inputs"]:
        try:
            vectors.append(engine.embed(text))
        except RuntimeError as exc:
            raise om.ApiError(502, f"engine error: {exc}", err_type="engine_error")
    return om.embeddings_response(vectors, parsed["inputs"])
