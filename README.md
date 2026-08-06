# URL Video Manager

A self-hosted queue for downloading videos/audio from YouTube and other sites (via [yt-dlp](https://github.com/yt-dlp/yt-dlp)), with subtitle handling, optional vocal stripping, and hardsub subtitle generation. One FastAPI app, one SQLite file, one Docker image — built to run unattended on a NAS.

## Features

- **Queue-based downloads** — paste URLs (one per line) or a playlist link, watch live progress over SSE, survives page refresh and container restart.
- **Quality / format control** — resolution cap, and mp4/mkv/webm output with a source-stream preference that avoids landing on webm just because that's what YouTube happened to serve as "best."
- **Format probing** — see every format YouTube actually offers for a URL and pick an exact one.
- **Subtitles from the video** — download real subtitle tracks, embed them, or merge two languages into one bilingual track (top/bottom per cue).
- **Vocal stripping** — remove vocals from an audio-only download via [Demucs](https://github.com/facebookresearch/demucs) (karaoke-style instrumental). Heavy dependency, off by default.
- **Hardsub subtitle generation** — for reuploads with only burned-in subtitles and no real track:
  - **Whisper** (ASR) transcribes the spoken audio.
  - **OCR** (Tesseract) reads the actual burned-in pixels off sampled frames.
  - Either or both; results feed into the same embedding pipeline as downloaded subtitles.
- **Subtitle translation** — translate a generated track into a second language via offline [argos-translate](https://github.com/argosopentech/argos-translate) (no cloud, no API key), keeping both tracks.
- **In-browser player** — a "Play…" button opens a real `<video>` player with a native subtitle-track picker (every `.srt` sidecar converted to WebVTT on the fly), not just a bare file link.
- **Cookies from the UI** — paste or upload a `cookies.txt` (needed for age-gated/bot-checked videos) without SSH.
- **Regenerate subtitles without re-downloading** — fix a bad OCR/Whisper/translate result and re-run just that stage against the file already on disk.
- **Per-job file downloads** — every artifact (main output + every subtitle variant) individually downloadable.
- **Per-job logs** — full yt-dlp/ffmpeg/tesseract output, not just a truncated error string; every log line is timestamped.

## Quick start

Pull and run the published image — no build required:

```bash
git clone https://github.com/david-chau/url-video-manager.git
cd url-video-manager
mkdir -p downloads data
docker compose up -d
```

Open `http://localhost:1208` (or your host's IP, if running on a NAS). Paste a URL, pick a quality, hit **Add**.

To update later:

```bash
docker compose pull && docker compose up -d --force-recreate
```

## Building from source

```bash
git clone https://github.com/david-chau/url-video-manager.git
cd url-video-manager
docker compose build
docker compose up -d
```

`.github/workflows/publish.yml` builds and pushes `:latest` (linux/amd64) automatically on every push to `main` — the compose file above just pulls that image.

### Optional heavy features

Vocal stripping, Whisper transcription, and offline translation each pull in a large dependency (PyTorch or similar) and are **off by default**. Enable them with a build arg:

```bash
docker compose build --build-arg WITH_DEMUCS=true       # vocal stripping
docker compose build --build-arg WITH_TRANSCRIBE=true   # Whisper ASR
docker compose build --build-arg WITH_TRANSLATE=true    # argos-translate
```

**Check your CPU has AVX before enabling these.** PyTorch (and libraries built on it) can crash with `Illegal instruction` on CPUs without AVX — common on low-power NAS hardware (e.g. Synology's Celeron-based models). Pre-flight check:

```bash
docker run --rm python:3.12-slim sh -c \
  "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
   python -c 'import torch; print(torch.rand(4,4).sum())'"
```

Prints a number → your CPU can run these features. Crashes → it can't; run that build on different hardware (e.g. build on a Mac/PC, point it at the NAS's `downloads` folder over the network) instead.

OCR (Tesseract) has no such restriction and is always available — it's plain C, no GPU/AVX dependency.

## Configuration

All settings are environment variables in `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `MAX_CONCURRENT` | `2` | Simultaneous downloads |
| `SEPARATION_SLOTS` | `1` | Simultaneous vocal-stripping jobs (independent of `MAX_CONCURRENT`) |
| `TRANSCRIBE_SLOTS` | `1` | Simultaneous Whisper/OCR jobs (independent of the above) |
| `DEMUCS_MODEL` | `htdemucs` | Demucs model; `mdx_extra_q` is faster/lower quality |
| `KEEP_VOCALS` | unset | Set `1` to keep the isolated vocal track alongside the instrumental |
| `AUDIO_FORMAT` | unset | Unset = passthrough (no re-encode); set e.g. `mp3` to force one |
| `WHISPER_MODEL` | `small` | Whisper model size; `base` is faster/lower quality |
| `OCR_SAMPLE_FPS` | `2` | Frames per second sampled for OCR |
| `OCR_CROP_BOTTOM_PCT` | `0.22` | Bottom fraction of the frame OCR'd (where hardsubs usually sit) |
| `OCR_LANG` | `chi_sim+chi_tra+eng` | Tesseract language packs, used when a job has no language hint |
| `TRANSLATE_MODEL_DIR` | `/data/argos-models` | Where translation models are cached (persists across restarts) |
| `YTDLP_COOKIES` | `/data/cookies.txt` | Cookie file path (set via the UI, not by hand) |
| `APP_PASSWORD` | unset | Set to require HTTP Basic auth on every route |

## Deploying on a NAS (e.g. Synology)

- Bind mount `./data` must be **local storage**, not an SMB/NFS share — SQLite's WAL mode doesn't tolerate network filesystems.
- Uncomment `user: "uid:gid"` in `docker-compose.yml` (find yours with `id <you>`) so downloaded files aren't root-owned in the shared folder.
- The image is `linux/amd64` — build on a machine with that architecture (or cross-build: `docker buildx build --platform linux/amd64 ...`), then transfer or push to a registry the NAS can pull from.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_worker.py
```

Plain `assert`-based checks, no test framework — see `test_worker.py`. No persistent virtualenv is committed; create and discard one as needed.

## Architecture

- `app/main.py` — FastAPI routes, SSE progress stream, startup recovery
- `app/worker.py` — the queue loop, yt-dlp calls, all post-processing stages (separation, transcription, translation, muxing)
- `app/db.py` — SQLite schema, append-only migrations, thread-safe connections
- `app/merge_srt.py` — vendored bilingual subtitle merger (kept byte-for-byte identical to its source)
- `app/static/index.html` — the entire frontend, vanilla JS, no build step

Job status flows through `queued → running → [separating] → [transcribing] → [muxing] → done`, with `error`/`canceled` as terminal states reachable from any point. Every stage writes progress to SQLite (not just an in-memory socket), so a page refresh or container restart never loses state — interrupted jobs resume from whichever stage they were in, without re-downloading.
