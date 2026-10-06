#!/usr/bin/env python
"""Re-vendor the model artifacts for a version upgrade.

Downloads the pinned needle3.cact weights and the pinned linux-x86_64 engine
wheel from Hugging Face into vendor/. After upgrading pins in
requirements.txt / Dockerfile (cache folder carries the engine version), run:

    python scripts/fetch_models.py [--engine-version 3.1.0]

Then rebuild — the Docker warm-up gate proves the pair loads.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

WEIGHTS_URL = "https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact"
ENGINE_WHEEL = ("https://huggingface.co/Cactus-Compute/needle3/resolve/main/"
                "python/cactus_needle-{version}-py3-none-manylinux2014_x86_64.whl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-version", default="3.1.0")
    args = parser.parse_args()

    vendor = Path(__file__).resolve().parent.parent / "vendor"
    vendor.mkdir(exist_ok=True)

    weights = vendor / "needle3.cact"
    print(f"fetching {WEIGHTS_URL}")
    weights.write_bytes(_download(WEIGHTS_URL))
    print(f"  -> {weights} ({weights.stat().st_size} bytes)")

    wheel_url = ENGINE_WHEEL.format(version=args.engine_version)
    print(f"fetching {wheel_url}")
    blob = _download(wheel_url)
    member = "needle/libneedle3.so"
    with zipfile.ZipFile(_io(blob)) as archive:
        data = archive.read(member)
    engine = vendor / "libneedle3.so"
    engine.write_bytes(data)
    print(f"  -> {engine} ({len(data)} bytes, member {member})")


def _download(url):
    import urllib.request

    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _io(blob):
    import io

    return io.BytesIO(blob)


if __name__ == "__main__":
    main()
