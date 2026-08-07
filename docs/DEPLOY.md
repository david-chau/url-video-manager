# Deployment

## What ships in the image

OCR (Tesseract), Whisper transcription, and offline translation are all in `:latest`. Only vocal stripping is behind a build flag, because it pulls ~2GB of PyTorch:

```bash
docker compose build --build-arg WITH_DEMUCS=true
```

Models are not baked in. Whisper (~500MB for `small`) and argos language pairs download on first use into `/data`, so they survive container recreation and cost that download only once.

There is a single tag. Whisper and translation used to live behind manual-only `:transcribe` / `:translate` tags, which meant a NAS pointed at one of them ran a hand-built image that pushes to `main` never refreshed, and whichever feature wasn't in that tag was not merely unused but uninstallable. Everything except Demucs is now in the image CI builds.

## CPU requirements

On CPUs without AVX — common on low-power NAS hardware, e.g. Synology's Celeron models — PyTorch can crash with `Illegal instruction`. That is why Demucs stays optional.

- **Whisper is unaffected.** It runs on ctranslate2, not PyTorch. Verified on a DS920+ (Celeron J4125, no AVX): a 4:24 clip transcribed in 5m22s with `WHISPER_MODEL=small`, roughly 1.2× realtime.
- **Translation is unverified there.** argos-translate pulls stanza/PyTorch transitively. This app only calls the plain per-cue `translate()` API and never stanza's sentence splitter, so PyTorch may never execute — but that hasn't been tested on non-AVX hardware.
- **OCR has no restriction.** Plain C, no GPU or AVX dependency.

Pre-flight check before enabling Demucs:

```bash
docker run --rm python:3.12-slim sh -c \
  "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
   python -c 'import torch; print(torch.rand(4,4).sum())'"
```

Prints a number → the CPU can run it. Crashes → it can't; build and run that variant elsewhere.

## Building from source

```bash
docker compose build
docker compose up -d
```

## CI

`.github/workflows/publish.yml` runs `test_worker.py`, then builds and pushes `linux/amd64` on every push to `main`:

- `:latest` and `:<git-sha>` — the normal image
- `:demucs` — manual only (Actions → Publish image → Run workflow → `with_demucs`)

The build stamps `GIT_SHA` into the image, and every job's log opens with `build=<sha> engines=<list>`. That line answers "is the container actually running the code I just pushed?" from the UI, without shell access — a stale image and a broken feature look identical otherwise.

## NAS setup (e.g. Synology)

- Bind mount `./data` must be **local storage**, not an SMB/NFS share — SQLite's WAL mode doesn't tolerate network filesystems.
- Uncomment `user: "uid:gid"` in `docker-compose.yml` (find yours with `id <you>`) so downloads aren't root-owned in the shared folder.
- The NAS is pull-only: `docker compose pull && docker compose up -d`.

### The `user:` / `HOME` interaction

Running as a non-root user leaves `HOME=/`, which that user cannot write. Every library resolving `~` or an XDG path then fails mid-job, each on a different directory — `/.cache` for faster-whisper's model download, `/.local` for argos-translate's data dir.

The app detects an unwritable `HOME` at startup and repoints it (plus `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`) at `APP_HOME` on the persistent volume, logging:

```
[startup] HOME was not writable, repointed to /data/home
```

A writable `HOME` is left alone, and if `/data` isn't usable it declines to move `HOME` rather than pointing it somewhere equally broken.

## Expectations on NAS hardware

Whisper is CPU-bound: budget roughly realtime per video on a Celeron, serialized by `TRANSCRIBE_SLOTS`. A 34-episode season is an overnight job. Batch actions queue rather than running in parallel, so starting all of them at once is safe — every row shows `transcribing` immediately, but only `TRANSCRIBE_SLOTS` of them are actually running.

## Logs

Three places, same content:

- **The UI's Log button** — per job, including engine settings, cue counts, and the build SHA.
- **`docker logs`** — discarded whenever the container is recreated, which is exactly when history is wanted.
- **`APP_LOG_PATH`** (`/data/app.log`, rotated at 8MB) — survives recreation and is readable over a file share without shell access.

Long transcribe/OCR runs print a liveness line every `HEARTBEAT_SECONDS`, so a slow job and a hung one are distinguishable without `docker stats`.
