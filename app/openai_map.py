"""OpenAI wire-format mapping: request validation and response envelopes.

Deliberate subset of the OpenAI surface, honest about it:
- `tools` in requests is accepted but ignored (the toolset is fixed per
  deployment via TOOLSET_PATH);
- `temperature`/`top_p` are accepted but ignored (grammar-constrained decoding
  leaves only max_new_tokens);
- `stream:true` is rejected with 400 (Needle returns complete dicts, no token
  streaming);
- `tool_calls[].function.arguments` is a JSON-encoded STRING per the OpenAI
  spec, exactly as Cactus's own platform formats them;
- finish_reason is "tool_calls" when the (floor-filtered) model emitted calls,
  "stop" otherwise — which covers both refusal (off-topic) and "done" (after
  tool results; Needle generates no free text, content stays null).
"""

from __future__ import annotations

import json
import time
import uuid

from . import __version__
from .engine import estimate_tokens

MODEL_ID = "needle3"


class ApiError(Exception):
    def __init__(self, status, message, err_type="invalid_request_error", code=None, param=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param

    def body(self):
        error = {"message": self.message, "type": self.err_type}
        if self.param:
            error["param"] = self.param
        if self.code:
            error["code"] = self.code
        return {"error": error}


def _new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def parse_chat_request(payload):
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")
    if payload.get("stream") is True:
        raise ApiError(400,
                       "streaming is not supported: Needle 3 returns complete, "
                       "grammar-constrained responses rather than token streams",
                       code="streaming_unsupported", param="stream")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "'messages' must be a non-empty array",
                       param="messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ApiError(400, f"messages[{index}] must be an object with a 'role'",
                           param=f"messages[{index}]")
    max_new_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", 512))
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 512:
        raise ApiError(400, "'max_tokens' must be an integer between 1 and 512 "
                            "(grammar-constrained decoding caps output at 512)",
                       param="max_tokens")
    # Accepted for compatibility, ignored: tools, temperature, top_p, model.
    return {"messages": messages, "max_new_tokens": max_new_tokens}


def parse_extract_request(payload):
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "'text' must be a non-empty string", param="text")
    schema = payload.get("schema")
    schema_tool = _normalize_schema(schema)
    max_new_tokens = payload.get("max_tokens", 512)
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 512:
        raise ApiError(400, "'max_tokens' must be an integer between 1 and 512",
                       param="max_tokens")
    return {"text": text, "schema_tool": schema_tool,
            "strict": bool(payload.get("strict", False)),
            "max_new_tokens": max_new_tokens}


def _normalize_schema(schema):
    """Accept a flat tool dict, an OpenAI-wrapped function, or a bare
    parameters object (wrapped as the single 'extract' tool)."""
    if not isinstance(schema, dict):
        raise ApiError(400, "'schema' must be a JSON object", param="schema")
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    if "parameters" not in schema and ("properties" in schema or schema.get("type") == "object"):
        schema = {"name": "extract", "description": "The record to extract",
                  "parameters": schema}
    if not isinstance(schema.get("name"), str) or not schema["name"]:
        raise ApiError(400, "'schema' needs a 'name' (or a bare parameters object)",
                       param="schema")
    if not isinstance(schema.get("parameters"), dict):
        raise ApiError(400, "'schema.parameters' must be a JSON object", param="schema")
    return schema


def parse_embeddings_request(payload):
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")
    raw = payload.get("input")
    if isinstance(raw, str):
        inputs = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
        inputs = raw
    else:
        raise ApiError(400, "'input' must be a string or a non-empty array of strings",
                       param="input")
    if len(inputs) > 32:
        raise ApiError(400, "at most 32 inputs per request", param="input")
    return {"inputs": inputs}


def _tool_call_envelope(index, call):
    return {
        "id": f"call_{index}_{uuid.uuid4().hex[:20]}",
        "type": "function",
        "function": {
            "name": str(call.get("name")),
            # OpenAI spec: arguments is a JSON-encoded STRING.
            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
        },
    }


def chat_response(final, model_card, messages, max_new_tokens, prefix_tokens):
    """final = engine.chat(...) result; model_card = /v1/models entry dict."""
    calls = final["calls"]
    message = {"role": "assistant", "content": None}
    finish_reason = "stop"
    if calls:
        message["tool_calls"] = [_tool_call_envelope(i, c) for i, c in enumerate(calls)]
        finish_reason = "tool_calls"

    prompt_tokens = prefix_tokens + sum(
        estimate_tokens(json.dumps(message, ensure_ascii=False)) for message in messages)
    output_chars = len(final.get("reasoning") or "")
    for call in calls:
        output_chars += len(json.dumps(call, ensure_ascii=False)) + 24
    completion_tokens = max(1, output_chars // 3)

    return {
        "id": _new_id("chatcmpl-needle"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "needle": {
            "note": "non-OpenAI extras; token counts are estimates",
            "engine_type": final["raw_type"],
            "confidence": final["confidence"],
            "reasoning": final["reasoning"],
            "suppressed_calls": final["suppressed_calls"],
            "confidence_floor": final["floor"],
            "success": final["success"],
            "error": final["error"],
            "prefill_tps": final["prefill_tps"],
            "decode_tps": final["decode_tps"],
            "peak_ram_mb": final["peak_ram_mb"],
        },
    }


def extract_response(data, confidence=None, reasoning=None):
    return {
        "id": _new_id("ext-needle"),
        "object": "extraction",
        "created": int(time.time()),
        "model": MODEL_ID,
        "data": data,
        "needle": {
            "confidence": confidence,
            "reasoning": reasoning,
        },
    }


def embeddings_response(vectors, inputs):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": MODEL_ID,
        "usage": {
            "prompt_tokens": sum(estimate_tokens(text) for text in inputs),
            "total_tokens": sum(estimate_tokens(text) for text in inputs),
        },
    }


def models_response(model_card):
    return {"object": "list", "data": [model_card]}
