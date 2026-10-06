"""Engine singleton: one Needle agent per process, serialized by a lock.

Needle's engine keeps one active agent slot per process and is not thread-safe
(the C state is global), so every inference call holds a process-wide lock and
uvicorn runs a single worker. Each HTTP request is stateless: `stateless=True`
resets the conversation before the first `complete()` of the request, then the
OpenAI tool loop is replayed natively — the user query, then each tool result
fed back as `complete(json.dumps(result))`, exactly as the model was trained.
"""

from __future__ import annotations

import json
import os
import threading

CONTEXT_TOKENS = 8192          # Needle 3 context window (per Cactus porting guide)
CONTEXT_SLACK = 96             # safety margin for control tokens
_TURN_OVERHEAD_TOKENS = 24     # im_start/im_end + tool-result wrappers per turn
_RESULT_CHAR_CAP = 2000
_LINE_CAP = 400


def estimate_tokens(text):
    """Conservative byte-level estimate (English averages ~3.5-4 chars/token)."""
    return (len(text or "") + 2) // 3


def rss_mb():
    """Resident set size in MB on Linux, else None."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


def _text_of(content):
    """User content: plain string, or a list of OpenAI content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _split_messages(messages):
    """-> (preamble, query, rounds).

    Rounds is a list of rounds; each round is a list of tool-result payload
    strings, in arrival order (parallel tool calls feed sequentially).
    """
    last_user = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    if last_user is None:
        raise ValueError("messages must include at least one message with role 'user'")

    preamble_lines = []
    for message in messages[:last_user]:
        role = message.get("role")
        if role == "user":
            preamble_lines.append(f"user: {_text_of(message.get('content'))[:_LINE_CAP]}")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                names = ", ".join(
                    (call.get("function") or {}).get("name", "?") for call in calls
                    if isinstance(call, dict))
                preamble_lines.append(f"assistant: called {names}")
            else:
                preamble_lines.append(f"assistant: {_text_of(message.get('content'))[:_LINE_CAP]}")
        elif role == "tool":
            preamble_lines.append(f"tool result: {str(message.get('content'))[:_LINE_CAP]}")
    preamble = "\n".join(line for line in preamble_lines if line.strip())

    query = _text_of(messages[last_user].get("content"))
    if not query.strip():
        raise ValueError("the last user message is empty")

    rounds, current = [], None
    for message in messages[last_user + 1:]:
        role = message.get("role")
        if role == "tool":
            if current is None:
                current = []
                rounds.append(current)
            current.append(str(message.get("content") or "")[:_RESULT_CHAR_CAP])
        elif role == "assistant" and (message.get("tool_calls") or []):
            current = None  # a new round of results follows
    return preamble, query, rounds


def _result_payload(content):
    """Feed tool results as JSON when they parse, else as raw text."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        pass
    return content


def _unwrap_request_tools(tools):
    """OpenAI request tools are accepted but ignored (toolset is fixed per deployment)."""
    return None


class NeedleEngine:
    def __init__(self, tools, system_facts, confidence_floor=0.30, buffer_size=65536):
        import needle

        self._needle = needle
        self._lock = threading.Lock()
        self._tools = tools
        self._system_facts = system_facts
        self._confidence_floor = max(0.0, float(confidence_floor))
        self._agent = needle.Needle(tools=tools, system=system_facts,
                                    stateless=True, buffer_size=buffer_size)
        self._prefix_tokens = self._measure_prefix()
        self.embed_dim = len(self._agent.embed("warm up"))
        warm = self._agent.complete("ping", max_new_tokens=64)
        if warm.get("success") is False:
            raise RuntimeError(f"engine warm-up failed: {warm.get('error')}")
        self.peak_ram_mb = warm.get("peak_ram_mb")

    def _measure_prefix(self):
        """The engine reports the system+tools prefix size as needle_init's return."""
        try:
            from needle import _lib

            lib = _lib(3)
            count = lib.needle_init(self._agent._system, self._agent._tools_json, None)
            if isinstance(count, int) and count > 0:
                return count
        except Exception:
            pass
        return estimate_tokens(self._system_facts) + estimate_tokens(
            json.dumps(self._tools, ensure_ascii=False))

    def _guard(self, preamble, query, rounds, max_new_tokens):
        """400 before the engine ever sees an overflowing conversation."""
        required = self._prefix_tokens + max_new_tokens + CONTEXT_SLACK
        required += estimate_tokens(preamble) + estimate_tokens(query)
        required += 2 * _TURN_OVERHEAD_TOKENS
        for round in rounds:
            for payload in round:
                required += estimate_tokens(payload) + _TURN_OVERHEAD_TOKENS
        budget = CONTEXT_TOKENS - CONTEXT_SLACK
        if required > budget:
            raise OverflowError(
                f"conversation exceeds the {CONTEXT_TOKENS}-token context window: "
                f"~{required} tokens required (prefix ~{self._prefix_tokens} + messages + "
                f"max_new_tokens {max_new_tokens}); shorten messages, fewer tool rounds, "
                f"or lower max_new_tokens")

    def _finalize(self, response, max_new_tokens):
        calls = list(response.get("function_calls") or [])
        suppressed = list(response.get("suppressed_calls") or [])
        confidence = response.get("confidence")
        floor_applied = False
        if (calls and self._confidence_floor > 0
                and isinstance(confidence, (int, float)) and confidence < self._confidence_floor):
            suppressed = suppressed + calls
            calls = []
            floor_applied = True
        return {
            "raw_type": response.get("type"),
            "calls": calls,
            "suppressed_calls": suppressed,
            "confidence": confidence,
            "reasoning": response.get("reasoning"),
            "success": response.get("success"),
            "error": response.get("error"),
            "prefill_tps": response.get("prefill_tps"),
            "decode_tps": response.get("decode_tps"),
            "peak_ram_mb": response.get("peak_ram_mb"),
            "floor": self._confidence_floor if floor_applied else None,
            "max_new_tokens": max_new_tokens,
        }

    def chat(self, messages, max_new_tokens=512):
        """Replay the OpenAI tool loop natively; returns the finalized final envelope."""
        preamble, query, rounds = _split_messages(messages)
        with self._lock:
            self._guard(preamble, query, rounds, max_new_tokens)
            text = f"{preamble}\n\n{query}" if preamble else query
            response = self._agent.complete(text, max_new_tokens=max_new_tokens)
            for round in rounds:
                for payload in round:
                    response = self._agent.complete(
                        _result_payload(payload), max_new_tokens=max_new_tokens)
            return self._finalize(response, max_new_tokens)

    def extract(self, text, schema_tool, strict=False, max_new_tokens=512):
        """One-shot extraction; grammar-guaranteed to parse. Returns dict or None."""
        with self._lock:
            return self._needle.extract(
                text, schema_tool, system=self._system_facts or None,
                max_new_tokens=max_new_tokens, strict=bool(strict), generation=3)

    def embed(self, text):
        with self._lock:
            return self._agent.embed(text)
