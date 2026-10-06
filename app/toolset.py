"""Load and validate the deployment-fixed toolset (TOOLSET_PATH).

The toolset is fixed per deployment: Needle binds one toolset per agent, and
the agent is bound once at boot. Request-level `tools` fields are accepted for
OpenAI compatibility but ignored.
"""

from __future__ import annotations

import json
import os


class ToolsetError(ValueError):
    """The configured toolset is missing or malformed."""


def _normalize(entry, index):
    if isinstance(entry, dict) and entry.get("type") == "function" and isinstance(entry.get("function"), dict):
        entry = entry["function"]  # OpenAI-wrapped form
    if not isinstance(entry, dict):
        raise ToolsetError(f"tool #{index} must be a JSON object")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolsetError(f"tool #{index} needs a non-empty string 'name'")
    params = entry.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(params, dict):
        raise ToolsetError(f"tool '{name}': 'parameters' must be a JSON object")
    tool = {"name": name.strip(), "parameters": params}
    if isinstance(entry.get("description"), str) and entry["description"].strip():
        tool["description"] = entry["description"].strip()
    if isinstance(entry.get("triggers"), list) and entry["triggers"]:
        tool["triggers"] = entry["triggers"]
    return tool


def load_toolset(path):
    """Read TOOLSET_PATH (a JSON array of tools, or {"tools": [...]}) into flat form."""
    if not path:
        raise ToolsetError("TOOLSET_PATH is not set")
    if not os.path.exists(path):
        raise ToolsetError(f"TOOLSET_PATH does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolsetError(f"TOOLSET_PATH is not valid JSON: {exc}") from exc
    entries = raw.get("tools") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ToolsetError("toolset must be a non-empty JSON array of tools")
    tools, seen = [], set()
    for index, entry in enumerate(entries):
        tool = _normalize(entry, index)
        if tool["name"] in seen:
            raise ToolsetError(f"duplicate tool name: {tool['name']}")
        seen.add(tool["name"])
        tools.append(tool)
    return tools
