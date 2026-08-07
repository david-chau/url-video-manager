# URL Video Manager

A self-hosted queue for downloading videos and audio from YouTube and other sites (via [yt-dlp](https://github.com/yt-dlp/yt-dlp)), with subtitle handling, subtitle generation, and offline translation. One FastAPI app, one SQLite file, one Docker image — built to run unattended on a NAS.

## Features

- **Queue-based downloads** — paste URLs or a playlist link, live progress over SSE, survives page refresh and container restart.
- **Quality / format control** — resolution cap, mp4/mkv/webm output, and a format probe to pick an exact stream.
- **Subtitles from the video** — download real tracks, embed them, or merge two languages into one bilingual track.
- **Subtitle generation** — Whisper (ASR, transcribes the audio) and Tesseract OCR (reads burned-in hardsubs off frames), for reuploads with no real subtitle track.
- **Offline translation** — [argos-translate](https://github.com/argosopentech/argos-translate), no cloud or API key. Inline after generation, or applied later to any `.srt` already on disk.
- **Batch and playlist actions** — generate or translate subtitles across selected rows or a whole playlist, settings picked once.
- **In-browser player** — real `<video>` player with a subtitle-track picker, every `.srt` converted to WebVTT on the fly.
- **Re-run without re-downloading** — fix a bad OCR/Whisper/translate result against the file already on disk.
- **Vocal stripping** — optional, via [Demucs](https://github.com/facebookresearch/demucs). See [docs/DEPLOY.md](docs/DEPLOY.md).
- **Cookies from the UI**, per-job file downloads, search/filter, and per-job logs.

## Quick start

```bash
git clone https://github.com/david-chau/url-video-manager.git
cd url-video-manager
mkdir -p downloads data
docker compose up -d
```

Open `http://localhost:1208` (or the host's IP on a NAS). Paste a URL, pick a quality, hit **Add**.

Update later:

```bash
docker compose pull && docker compose up -d --force-recreate
```

Whisper and translation models download on first use into `/data` (~500MB for Whisper `small`), not baked into the image — so the first transcription of a fresh install is slow, and later ones aren't.

## Options

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
| `WHISPER_MODEL_DIR` | `/data/whisper-models` | Where the Whisper model is downloaded (persists across restarts) |
| `OCR_SAMPLE_FPS` | `2` | Frames per second sampled for OCR |
| `OCR_CROP_BOTTOM_PCT` | `0.22` | Bottom fraction of the frame OCR'd (where hardsubs usually sit) |
| `OCR_LANG` | `chi_sim+chi_tra+eng` | Tesseract packs, when a job has no hint or override. A lone CJK script paired with `eng` garbles CJK — keep both CJK packs together |
| `OCR_BINARIZE` | `1` | Per-frame Otsu thresholding before OCR; `0` disables |
| `TRANSLATE_MODEL_DIR` | `/data/argos-models` | Where translation models are cached (persists across restarts) |
| `HEARTBEAT_SECONDS` | `30` | How often a long transcribe/OCR run prints a liveness line |
| `APP_LOG_PATH` | `/data/app.log` | Persistent copy of everything the app logs |
| `APP_LOG_MAX_BYTES` | `8388608` | Rotate the above at this size, keeping one previous file |
| `APP_HOME` | `/data/home` | Where `HOME` is repointed when the container's own isn't writable |
| `YTDLP_COOKIES` | `/data/cookies.txt` | Cookie file path (set via the UI, not by hand) |
| `APP_PASSWORD` | unset | Set to require HTTP Basic auth on every route |

## Docs

- [docs/DEPLOY.md](docs/DEPLOY.md) — building from source, what ships in the image, CPU requirements, NAS setup, CI
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module layout, job lifecycle, design rules, testing
