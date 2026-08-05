FROM python:3.12-slim

# ffmpeg is needed for merging, audio extraction, and subtitle embedding
# (Phase 6's mux later reuses the same binary).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Phase 5: demucs (vocal separation) pulls in torch, ~2GB -- behind a build
# flag so the base image stays small when you don't want it. Rebuild with
# --build-arg WITH_DEMUCS=true (see docker-compose.yml) to enable
# strip-vocals jobs; without it those jobs fail fast with a clear error
# instead of hanging (see worker.run_separation).
ARG WITH_DEMUCS=false
RUN if [ "$WITH_DEMUCS" = "true" ]; then pip install --no-cache-dir demucs; fi

COPY app ./app

ENV DOWNLOADS_DIR=/downloads \
    DB_PATH=/data/jobs.db \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# `pip install -U yt-dlp` runs before every start so a `docker compose
# restart` is always the first fix to try when YouTube breaks an extractor.
# Binds 0.0.0.0, not 127.0.0.1 -- a loopback bind is only reachable from the
# host the container runs on, which makes this unusable on a headless NAS.
ENTRYPOINT ["sh", "-c", "pip install -q -U yt-dlp && exec uvicorn app.main:app --host 0.0.0.0 --port 8080"]
