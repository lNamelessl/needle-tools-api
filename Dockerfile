# Needle Tool-Calling API — OpenAI-shaped tools/extraction/embeddings over Needle 3.
# The model weights (needle3.cact, 35,335,380 B) and the linux-x86_64 engine
# (libneedle3.so, engine 3.1.0) are vendored in vendor/ and copied to fixed paths,
# so builds never touch the network for the model and runtime never calls Hugging
# Face (HF_HUB_OFFLINE=1).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOME=/home/needle \
    # Telemetry: the Python client AND the binary both honor these (binary needs both).
    NEEDLE_TELEMETRY=0 \
    DO_NOT_TRACK=1 \
    # Runtime never fetches: weights are pre-seeded, engine path is pinned.
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOOLSET_PATH=/app/tools.json \
    NEEDLE3_LIB_PATH=/app/engine/libneedle.so \
    CONFIDENCE_FLOOR=0.30 \
    NEEDLE_SYSTEM_FACTS="locale: en-US; device: railway-vps"

RUN useradd --create-home --uid 10001 needle

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vendored model artifacts -> fixed paths the engine resolves deterministically:
#   weights: $HOME/.cache/cactus-needle/v3/3.1.0/needle3.cact  (fetch.cache_dir(3))
#   engine:  NEEDLE3_LIB_PATH override
COPY vendor/needle3.cact /home/needle/.cache/cactus-needle/v3/3.1.0/needle3.cact
COPY vendor/libneedle3.so /app/engine/libneedle.so
RUN chown -R needle:needle /home/needle/.cache /app/engine \
    && chmod 0644 /home/needle/.cache/cactus-needle/v3/3.1.0/needle3.cact \
                  /app/engine/libneedle.so

COPY app/ ./app/
COPY tools.json .

# Build gate: proves the manylinux2014 engine loads on linux/x86_64 and the
# weights parse. A broken or mismatched vendor/ fails the build here.
RUN python -c "from app import bootstrap; bootstrap.run()"

USER needle
EXPOSE 8000

# Single worker: the engine holds one active agent per process (not thread-safe);
# app-level lock serializes inference.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
