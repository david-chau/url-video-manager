FROM python:3.12-slim

# ffmpeg is needed for merging, audio extraction, and subtitle embedding
# (Phase 6's mux later reuses the same binary). curl+unzip are only needed
# to install Deno below, not at runtime, but removing them after would save
# a few MB for a lot of extra Dockerfile complexity -- not worth it here.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip \
    && rm -rf /var/lib/apt/lists/*

# YouTube requires solving a JS "n challenge" to unlock real (non-image-only)
# formats -- yt-dlp offloads this to an external JS runtime rather than
# bundling one. Deno is the default/supported runtime. Without this, every
# download fails with "Requested format is not available" even though
# cookies and everything else are fine. Installed to /usr/local/bin (not
# /root/.deno/bin, Deno's default) so it's still on PATH if the container is
# ever run as a non-root user via compose's `user:`.
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -y \
    && mv /root/.deno/bin/deno /usr/local/bin/deno \
    && chmod +x /usr/local/bin/deno \
    && rm -rf /root/.deno

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

# `pip install -U` runs before every start so a `docker compose restart` is
# always the first fix to try when YouTube breaks an extractor. [default]
# pulls in yt-dlp-ejs (the n-challenge solver scripts) -- keep it on the
# update too, or a restart silently drops back to bare yt-dlp.
# Binds 0.0.0.0, not 127.0.0.1 -- a loopback bind is only reachable from the
# host the container runs on, which makes this unusable on a headless NAS.
ENTRYPOINT ["sh", "-c", "pip install -q -U 'yt-dlp[default]' && exec uvicorn app.main:app --host 0.0.0.0 --port 8080"]
