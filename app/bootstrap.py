"""Shared boot path: build the engine singleton and prove it works.

Used by the Docker build gate (`RUN python -c "from app import bootstrap;
bootstrap.run()"`) and available for local smoke tests. The API server builds
the same NeedleEngine in its lifespan.
"""

from __future__ import annotations

import os

from .engine import NeedleEngine
from .toolset import load_toolset


def build_engine():
    tools = load_toolset(os.environ.get("TOOLSET_PATH", "/app/tools.json"))
    return NeedleEngine(
        tools,
        os.environ.get("NEEDLE_SYSTEM_FACTS", "locale: en-US; device: railway-vps"),
        confidence_floor=float(os.environ.get("CONFIDENCE_FLOOR", "0.30")),
    )


def run():
    engine = build_engine()
    probe = engine._agent.complete("ping", max_new_tokens=64)
    if probe.get("success") is False:
        raise RuntimeError(f"warm-up complete() failed: {probe.get('error')}")
    vector = engine._agent.embed("ping")
    if not vector or len(vector) == 0:
        raise RuntimeError("warm-up embed() returned an empty vector")
    print(f"[bootstrap] engine OK: embed_dim={len(vector)} "
          f"prefix_tokens={engine._prefix_tokens} peak_ram_mb={probe.get('peak_ram_mb')}")


if __name__ == "__main__":
    run()
