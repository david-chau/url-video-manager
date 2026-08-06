FROM python:3.12-slim

# ffmpeg is needed for merging, audio extraction, and subtitle embedding
# (Phase 6's mux later reuses the same binary). curl+unzip are only needed
# to install Deno below, not at runtime, but removing them after would save
# a few MB for a lot of extra Dockerfile complexity -- not worth it here.
#
# tesseract-ocr + chi-sim/chi-tra language packs (Phase 7's OCR hardsub
# pipeline) are installed unconditionally here, unlike WITH_DEMUCS/
# WITH_TRANSCRIBE below -- pure C, no AVX/torch crash risk on the target
# Synology's non-AVX Celeron, and small (~15-40MB for these two packs).
# PaddleOCR would carry the same AVX risk as torch/demucs and isn't used
# here; Tesseract's Chinese accuracy is noticeably weaker than PaddleOCR's.
# # ponytail: one OCR engine, not a second engine behind its own flag --
# # swap in PaddleOCR (its own build ARG, same shape as WITH_DEMUCS) if
# # real-world accuracy turns out to matter enough to justify it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl unzip tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
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

# Phase 7: Whisper (speech-to-text ASR) transcription. faster-whisper's
# ctranslate2 backend, not the PyTorch reference implementation -- better
# CPU perf/memory, and ctranslate2 has a better track record than raw torch
# on older/non-AVX CPUs (see the plan's Synology section: torch is
# documented to hard-crash with "Illegal instruction" on a non-AVX CPU;
# verify ctranslate2 doesn't before relying on this on the actual NAS, don't
# just assume). Same shape as WITH_DEMUCS above: fails fast with a clear
# error at job time instead of being installed unconditionally.
ARG WITH_TRANSCRIBE=false
RUN if [ "$WITH_TRANSCRIBE" = "true" ]; then pip install --no-cache-dir faster-whisper; fi

# Phase 8: argos-translate (offline NMT subtitle translation) -- chosen over
# a cloud translation API specifically to keep this app self-hosted, no API
# key, no call ever leaving the box. Pulls in ctranslate2 (same backend as
# faster-whisper above) plus, as of argostranslate 1.11, a transitive
# stanza/torch dependency for sentence-boundary detection -- heavier than
# WITH_TRANSCRIBE alone (verified: ~1.3GB installed, torch included even
# though this app only ever calls the plain per-cue translate() API, not
# stanza's sentence splitter). Still gated the same way, same reasoning.
# Language-pair models are NOT baked in here; they're downloaded into
# TRANSLATE_MODEL_DIR (under /data, the persistent volume) the first time a
# given pair is used -- see worker.get_argos_translator.
ARG WITH_TRANSLATE=false
RUN if [ "$WITH_TRANSLATE" = "true" ]; then pip install --no-cache-dir argostranslate; fi

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
