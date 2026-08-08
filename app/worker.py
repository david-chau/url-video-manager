"""asyncio queue loop, yt-dlp calls, playlist expansion.

yt-dlp is used as a library (not a subprocess) so progress, format probing,
and playlist enumeration go through a real API instead of stdout regex.
"""
import asyncio
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pysrt
import yt_dlp
import yt_dlp.utils
from yt_dlp.utils import DownloadCancelled

from . import db
from .merge_srt import merge_subtitles

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
AUDIO_FORMAT = os.environ.get("AUDIO_FORMAT")  # unset = passthrough, no re-encode
DOWNLOADS_DIR = os.environ.get("DOWNLOADS_DIR", "/downloads")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES")

# Where '~' should point when the real HOME isn't writable.
APP_HOME = os.environ.get("APP_HOME", "/data/home")


def ensure_writable_home(app_home: str = APP_HOME) -> str | None:
    """Repoints HOME (and the XDG dirs derived from it) at the persistent
    volume when the inherited HOME can't be written to. Returns the new
    home, or None if the existing one was already fine.

    compose's `user:` (needed so downloads aren't root-owned in a Synology
    shared folder) leaves HOME='/', which no non-root user can write. Every
    library that resolves '~' or an XDG path then dies mid-job, and each
    one picks a different directory to die on -- faster-whisper's HF cache
    at '/.cache', argos-translate's data dir at '/.local'. Both were fixed
    one at a time by pointing that specific tool somewhere else, which
    fixes exactly the tools already known to break and leaves the next one
    to fail in production. This fixes the cause instead.

    Must run before any library that reads these is imported; all of them
    are imported lazily inside their own functions, so module scope here is
    early enough (the same reasoning ARGOS_PACKAGES_DIR relies on)."""
    home = os.environ.get("HOME") or ""
    if home and os.path.isdir(home) and os.access(home, os.W_OK):
        return None
    try:
        os.makedirs(app_home, exist_ok=True)
    except OSError:
        # /data not mounted or read-only -- leave HOME alone rather than
        # pointing it somewhere equally unwritable. The engines will still
        # fail, but with their own clear message, not a confusing one.
        return None
    os.environ["HOME"] = app_home
    os.environ.setdefault("XDG_DATA_HOME", os.path.join(app_home, ".local", "share"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(app_home, ".cache"))
    os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(app_home, ".config"))
    return app_home


_REPOINTED_HOME = ensure_writable_home()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


APP_LOG_PATH = os.environ.get("APP_LOG_PATH", "/data/app.log")
# Rotate at 8MB, keeping one previous file. Small on purpose: this exists so
# a NAS-side question ("what did it do overnight?") is answerable over SMB
# without docker, not as a full log-management system.
APP_LOG_MAX_BYTES = int(os.environ.get("APP_LOG_MAX_BYTES", str(8 * 1024 * 1024)))
_app_log_lock = threading.Lock()


def log_line(msg: str) -> None:
    """Every stdout line the app prints -- job errors, startup recovery --
    goes through this so they're all consistently timestamped. `docker
    logs` timestamps are opt-in (--timestamps) and mark receipt time, not
    necessarily when the underlying event happened; this puts the
    timestamp in the line itself regardless of how logs get viewed.

    Also appended to APP_LOG_PATH on the persistent volume. `docker logs`
    is lost on container recreation, which is exactly when you most want
    the history (every `compose pull && up -d` discards it), and reading it
    at all needs shell access to the NAS -- /data is already mounted and
    browsable over SMB."""
    line = f"[{_timestamp()}] {msg}"
    print(line, flush=True)
    with _app_log_lock:  # log_line is called from worker threads, not just the loop
        try:
            if os.path.exists(APP_LOG_PATH) and os.path.getsize(APP_LOG_PATH) > APP_LOG_MAX_BYTES:
                os.replace(APP_LOG_PATH, APP_LOG_PATH + ".1")
            os.makedirs(os.path.dirname(APP_LOG_PATH) or ".", exist_ok=True)
            with open(APP_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            # A read-only or missing /data must never take the app down over
            # a log line -- stdout above already carried it.
            pass


def build_stamp() -> str:
    """Which build is running, and which optional engines it actually has.
    Stamped at the head of every job's log because "is the container even
    running the code I just pushed?" was the real answer to more than one
    bug here, and nothing in the UI could answer it -- a stale image looks
    exactly like a broken feature."""
    engines = []
    for name, mod in (("whisper", "faster_whisper"), ("translate", "argostranslate"), ("demucs", "demucs")):
        if importlib.util.find_spec(mod) is not None:
            engines.append(name)
    engines.append("ocr")  # tesseract is unconditional in the image
    return f"build={os.environ.get('GIT_SHA', 'unknown')[:12]} engines={'+'.join(sorted(engines))}"


def append_job_log(job_id: int, msg: str, echo: bool = True) -> None:
    """Timestamped append to a job's own `log` column -- what the UI's Log
    button shows -- mirrored to stdout so `docker logs` shows the same
    story. Both, by default, deliberately: when these lines went only to
    the DB, a whisper run that started, produced 129 cues and muxed
    cleanly was completely invisible to `docker logs`, which showed the
    engine starting and then nothing forever. Every diagnosis from the
    outside then chases a job that already finished.

    echo=False is for text already on stdout by another route (a
    JobLogger dump, or fail_job's own log_line), to avoid printing twice.

    Re-reads the row rather than taking a `prior_log` from the caller's
    job dict: a job dict is a snapshot from whenever the stage started, so
    appending onto it silently drops every line written since (which is
    exactly what the pipeline does -- each stage holds its own copy).
    Trimmed to the last 8KB; the tail is what matters.
    # ponytail: read-modify-write, no locking. Stages within one job run
    # strictly in sequence, so the only racer would be two stages of the
    # SAME job at once, which the pipeline never does. Revisit if that
    # ever stops being true.
    """
    if echo:
        log_line(f"[job {job_id}] {msg}")
    prior = (db.get_job(job_id) or {}).get("log") or ""
    db.update_job(job_id, log=(prior + f"\n[{_timestamp()}] {msg}")[-8192:])


# Phase 5: vocal separation gets its own semaphore, completely independent
# of the download semaphore -- a CPU-bound demucs run must never block, or
# be blocked by, I/O-bound downloads. Both are module-level so queue_loop
# and the separation pipeline share the same instances (and tests can
# inspect them directly).
SEPARATION_SLOTS = int(os.environ.get("SEPARATION_SLOTS", "1"))
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
KEEP_VOCALS = os.environ.get("KEEP_VOCALS", "") == "1"
DOWNLOAD_SEM = asyncio.Semaphore(MAX_CONCURRENT)
SEPARATION_SEM = asyncio.Semaphore(SEPARATION_SLOTS)

# Phase 7: hardsub subtitle generation (Whisper ASR / Tesseract OCR) gets its
# own semaphore too, same reasoning as SEPARATION_SEM -- this is CPU-heavy
# work that must not starve, or be starved by, the I/O-bound download queue.
# Independent of both DOWNLOAD_SEM and SEPARATION_SEM.
TRANSCRIBE_SLOTS = int(os.environ.get("TRANSCRIBE_SLOTS", "1"))
TRANSCRIBE_SEM = asyncio.Semaphore(TRANSCRIBE_SLOTS)
# small/fast fallback if 'small' proves too slow on the Synology's Celeron;
# see WITH_TRANSCRIBE in the Dockerfile for the gate that makes this usable
# at all.
# 'medium' rather than 'small': measured better on real material here, and
# accuracy is the point of transcribing at all. Costs roughly 3x the
# runtime per step up (~15min for a 4:24 clip on the target NAS, doubled
# again when translate_to is set, since that's a second decode). Drop to
# 'small' or 'base' per job in the UI, or via this env var, when throughput
# matters more than accuracy.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
# Models a job is allowed to request. A whitelist, not passthrough
# validation: faster-whisper treats an unrecognized name as a Hugging Face
# repo id and downloads it, so an unchecked string from the client is a
# "fetch and execute arbitrary model weights" primitive. Sizes here are the
# standard multilingual set -- '.en' variants are omitted because this app
# exists for non-English source material.
#
# Rough cost on the target NAS, per 4:24 clip, from the measured 'small'
# run (5m22s): tiny ~1.5min, base ~2.5min, small ~5min, medium ~15min,
# large-v3 ~45min+. Doubles again when translate_to is set, since that's a
# second decode of the same audio.
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def whisper_model_for(job: dict) -> str:
    """The job's requested model, or the container default. Anything not in
    WHISPER_MODELS falls back rather than raising: a stale row naming a
    model that's since been dropped from the list should still transcribe,
    just not with whatever it asked for."""
    requested = (job.get("whisper_model") or "").strip()
    return requested if requested in WHISPER_MODELS else WHISPER_MODEL
OCR_SAMPLE_FPS = float(os.environ.get("OCR_SAMPLE_FPS", "2"))
OCR_CROP_BOTTOM_PCT = float(os.environ.get("OCR_CROP_BOTTOM_PCT", "0.22"))
# Tesseract's '+'-joined multi-language syntax. Used when the job has no
# gen_subs_lang hint, or the hint doesn't match one of the small set of
# mappings in ocr_lang_for() below.
#
# chi_sim+eng (a single CJK script paired with eng) is a confirmed-bad
# combination, not a hypothetical one: Tesseract's language model gets
# confused by a lone CJK script mixed with Latin and produces pure noise --
# e.g. an actual on-screen "九阳神拳吧" OCR'd as "FLBA SIE" with chi_sim+eng,
# but correctly as "九陽神拳吧" with chi_sim+chi_tra(+eng). Two CJK scripts
# together anchor it; eng can safely ride along once both are present.
# Verified directly against a rendered test image, not assumed.
OCR_LANG = os.environ.get("OCR_LANG", "chi_sim+chi_tra+eng")
# Per-frame Otsu binarization before OCR (see binarize_for_ocr). On by
# default -- stylized hardsubs are unreadable to Tesseract without it. The
# escape hatch exists because the previous, cruder attempt at this made
# some real input strictly worse, so a way to switch it off on the NAS
# without waiting for a rebuild is worth one env var.
OCR_BINARIZE = os.environ.get("OCR_BINARIZE", "1") not in ("0", "false", "False", "")

# Seconds between stdout liveness lines inside the long transcribe loops.
# 30s is sparse enough not to bury `docker logs` on a multi-hour OCR run,
# frequent enough that "still working" is obvious without waiting.
HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", "30"))

# Phase 8: subtitle translation (argos-translate) runs inline in the
# 'transcribing' stage -- no own semaphore, it's cheap text-only work
# compared to Whisper/OCR/demucs, riding under TRANSCRIBE_SEM instead.
# argos-translate downloads a language-pair model (tens to a few hundred MB)
# on first use of that pair; ARGOS_PACKAGES_DIR points that download at
# /data (this app's one persistent volume, see YTDLP_COOKIES above) so it's
# fetched once across container restarts, not baked into every image build.
# Must be set before argostranslate.settings is first imported (done lazily,
# inside get_argos_translator below, same gating as faster_whisper) --
# setting it here at module import time is early enough.
TRANSLATE_MODEL_DIR = os.environ.get("TRANSLATE_MODEL_DIR", "/data/argos-models")
os.environ.setdefault("ARGOS_PACKAGES_DIR", TRANSLATE_MODEL_DIR)

# Same treatment for faster-whisper, and for the same two reasons. Its model
# (~500MB for 'small') downloads from the HF hub on first use, and without
# this it lands in huggingface_hub's default cache under $HOME -- which is
# '/' when the container runs as a non-root user via compose's `user:`, so
# the download dies with PermissionError: '/.cache' before the model is ever
# loaded. HF_HOME covers the hub's own bookkeeping files, download_root (see
# run_whisper_transcribe) covers the model itself; both are needed, and both
# belong on /data so the download survives a container restart.
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/data/whisper-models")
os.environ.setdefault("HF_HOME", WHISPER_MODEL_DIR)

# Job ids the user has asked to cancel. Checked on every progress_hook fire.
CANCELED: set[int] = set()

# job_id -> ident of the thread currently running its blocking yt-dlp call.
_job_thread: dict[int, int] = {}
# thread ident -> most recent subprocess.Popen spawned by yt-dlp on that
# thread (ffmpeg merge/embed/extract-audio). Lets cancel() kill a merge in
# progress instead of only being able to abort while bytes are moving.
_thread_proc: dict[int, subprocess.Popen] = {}

# ponytail: thread-id keyed, not job-id keyed, because yt-dlp's Popen wrapper
# doesn't know which job it's running for. Holds because asyncio.to_thread
# dedicates one thread pool thread per in-flight job for the duration of the
# call. If jobs ever shared a thread mid-call this'd need a real handle
# threaded through instead.
if not getattr(yt_dlp.utils.Popen, "_uvm_patched", False):
    _orig_popen_init = yt_dlp.utils.Popen.__init__

    def _tracking_popen_init(self, *a, **kw):
        _orig_popen_init(self, *a, **kw)
        _thread_proc[threading.get_ident()] = self

    yt_dlp.utils.Popen.__init__ = _tracking_popen_init
    yt_dlp.utils.Popen._uvm_patched = True


# ------------------------------------------------------------------ presets

_HEIGHT_CAP = {"1080": 1080, "720": 720, "480": 480}  # "best": uncapped

CONTAINER_CHOICES = ("mp4", "mkv", "webm")


def build_format(kind: str, quality: str, container: str = "mp4") -> str:
    """kind is the only source of truth for audio vs video.

    container steers *source stream* preference, not just the output file
    extension: on modern YouTube the highest-quality video-only streams are
    almost always VP9/AV1, so an unfiltered `bv*+ba/b` selector lands on
    one of those regardless of what the merge step is asked to produce.

    The first choice is H.264 by codec (`vcodec^=avc1`), not by container.
    Filtering on `[ext=mp4]` alone is not enough and was a real bug:
    YouTube serves AV1 *inside mp4*, so `[ext=mp4]` happily selected an AV1
    stream (itag 398), producing a file that plays in desktop Chrome via
    software decode but is refused outright by iOS Safari, which only
    decodes AV1 on the newest hardware. H.264 plays everywhere, remuxes
    into mp4 or mkv with no re-encode, and costs a little efficiency at the
    same resolution -- the right trade for a library meant to be watched on
    a phone.

    Falls through to any mp4, then to anything at all, so a video with no
    H.264 rendition still downloads. container='webm' skips the preference
    entirely and takes the highest-quality stream outright."""
    if kind == "audio":
        return "ba/b"
    if quality and quality.startswith("fmt:"):
        return quality.split(":", 1)[1]
    cap = _HEIGHT_CAP.get(quality)
    h = f"[height<={cap}]" if cap else ""
    if container == "webm":
        return f"bv*{h}+ba/b{h}"
    return (
        f"bv*{h}[vcodec^=avc1]+ba[ext=m4a]"
        f"/b{h}[vcodec^=avc1]"
        f"/bv*{h}[ext=mp4]+ba[ext=m4a]"
        f"/b{h}[ext=mp4]"
        f"/bv*{h}+ba/b{h}"
    )


# ------------------------------------------------------------- url classify

_PLAYLIST_PATH_RE = re.compile(r"(/playlist|/channel/|/c/|/@[^/]+/videos)")


def classify_url(url: str) -> str:
    """'playlist' or 'video'. A watch?v=X&list=Y URL stays a single video --
    that's the common accident this guards against."""
    try:
        p = urlparse(url)
    except ValueError:
        return "video"
    qs = parse_qs(p.query)
    if "list" in qs and "v" not in qs:
        return "playlist"
    if _PLAYLIST_PATH_RE.search(p.path or ""):
        return "playlist"
    return "video"


# --------------------------------------------------------------- bulk parse

def parse_bulk_urls(text: str, existing_urls: set[str]) -> tuple[list[str], list[str]]:
    """Split textarea input into (new_urls, duplicate_urls). Drops blanks and
    '#' comments, dedupes within the paste and against URLs already known."""
    seen: set[str] = set()
    new_urls: list[str] = []
    dupes: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen or line in existing_urls:
            dupes.append(line)
            continue
        seen.add(line)
        new_urls.append(line)
    return new_urls, dupes


# ------------------------------------------------------------ format helper

def _fmt_speed(bps) -> str | None:
    if not bps:
        return None
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if bps < 1024:
            return f"{bps:.1f}{unit}"
        bps /= 1024
    return f"{bps:.1f}TB/s"


def _fmt_eta(seconds) -> str | None:
    if seconds is None:
        return None
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def stage_n_guess(fmt: str) -> int:
    """bv*+ba selects two separate streams -> two files. A literal
    format_id or audio-only 'ba/b' is one file. This is a guess made from
    our own format string, not from yt-dlp internals: info_dict at
    progress_hook time carries neither 'requested_downloads' (filled in only
    after the whole download finishes) nor a populated 'requested_formats'
    (empty on the per-file dict the hook actually receives, verified against
    yt-dlp 2025-current) -- so the plan's literal field name doesn't exist at
    hook-time. stage_i below is tracked from distinct filenames instead, and
    stage_n is bumped up if more files show up than guessed, so the bar is
    still monotonic even if the guess is wrong."""
    return 2 if "+" in fmt else 1


class JobLogger:
    """Ring-buffered per-job log (~8KB) passed to yt-dlp as `logger`, plus
    warnings/errors also go to the container's own stdout (`docker logs`)
    as they happen -- the DB copy is only visible through the UI's log
    viewer, and `docker logs` is often the first thing reached for on a
    NAS when something's failing."""

    def __init__(self, job_id: int | None = None, cap: int = 8192):
        self.job_id = job_id
        self.cap = cap
        self.lines: list[str] = []

    def _prefix(self) -> str:
        return f"[job {self.job_id}] " if self.job_id is not None else ""

    def _append(self, msg: str) -> str:
        # Every line stored (not just what's printed to stdout) gets a
        # timestamp -- log_line only ever timestamped stdout, so the UI's
        # Log modal (which reads this stored text, not stdout) showed none
        # at all. Same format as log_line's stdout lines, for the same
        # reason: direct correlation between the two if you're looking at
        # both.
        line = f"[{_timestamp()}] {msg}"
        self.lines.append(line)
        return line

    def debug(self, msg):
        self._append(msg)

    def info(self, msg):
        self._append(msg)

    def warning(self, msg):
        self._append(f"WARNING: {msg}")
        log_line(f"{self._prefix()}WARNING: {msg}")

    def error(self, msg):
        self._append(f"ERROR: {msg}")
        log_line(f"{self._prefix()}ERROR: {msg}")

    def dump(self) -> str:
        text = "\n".join(self.lines)
        return text[-self.cap:]


def make_progress_hook(job_id: int, stage_n_hint: int, flush_interval: float = 1.0):
    """Progress hook factory. Flushes to SQLite at most once/sec per job
    (fires dozens of times/sec unthrottled) and scales progress across
    multi-file downloads so the bar never rewinds when bv*+ba downloads
    video then audio as two separate files."""
    state = {"files": [], "last_flush": 0.0}

    def hook(d):
        if job_id in CANCELED:
            raise DownloadCancelled(f"job {job_id} canceled")

        status = d.get("status")
        info = d.get("info_dict") or {}
        filename = d.get("filename") or info.get("filename")
        if filename not in state["files"]:
            state["files"].append(filename)
        stage_i = state["files"].index(filename) + 1 if filename else 1
        stage_n = max(stage_n_hint, stage_i)

        if status == "finished":
            pct = 100.0
        else:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            pct = min(100.0, downloaded / total * 100.0) if total else 0.0

        overall = ((stage_i - 1) + pct / 100.0) / stage_n * 100.0

        vcodec = info.get("vcodec")
        acodec = info.get("acodec")
        if vcodec and vcodec != "none":
            label = "video"
        elif acodec and acodec != "none":
            label = "audio"
        else:
            label = "file"
        stage_label = f"{label} {stage_i} of {stage_n}" if stage_n > 1 else label

        now = time.monotonic()
        if status == "downloading" and now - state["last_flush"] < flush_interval:
            return
        state["last_flush"] = now

        db.update_job(
            job_id,
            progress=round(overall, 1),
            stage=stage_label,
            stage_i=stage_i,
            stage_n=stage_n,
            speed=_fmt_speed(d.get("speed")),
            eta=_fmt_eta(d.get("eta")),
        )

    return hook


# ------------------------------------------------------------- ydl options

def build_ydl_opts(job: dict, progress_hook=None, logger=None) -> dict:
    kind = job["kind"]
    container = job.get("container") or "mp4"
    fmt = build_format(kind, job["quality"], container)

    if job.get("parent_id"):
        # playlist_index only exists when yt-dlp is walking a playlist.
        # These children are downloaded individually with noplaylist=True,
        # so a bare %(playlist_index)03d rendered the literal "NA" into
        # every filename in the season ("NA - Ep1", "NA - Ep2", ...).
        # yt-dlp's &REPLACEMENT|DEFAULT syntax emits the padded prefix when
        # the field is genuinely there and nothing at all when it isn't.
        # playlist_title needs the same treatment for the same reason --
        # without it these files land in a directory literally named "NA".
        # An empty path component collapses harmlessly, so the file just
        # sits in the downloads root when there's no playlist title.
        outtmpl = os.path.join(
            DOWNLOADS_DIR,
            "%(playlist_title|).100B",
            "%(playlist_index&{:03d} - |)s%(title).150B [%(id)s].%(ext)s",
        )
    else:
        outtmpl = os.path.join(DOWNLOADS_DIR, "%(title).150B [%(id)s].%(ext)s")

    opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        # every single-video job: a watch?v=X&list=Y URL must not silently
        # expand into the whole playlist. Only the Phase 4 expander (which
        # runs extract_flat, not this function) sets this False.
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,  # a stalled connection raises, not hangs forever
        "continuedl": True,
        "postprocessors": [],
    }
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]
    if logger:
        opts["logger"] = logger
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        opts["cookiefile"] = YTDLP_COOKIES

    subs = job.get("subs")
    if subs:
        opts.update(build_subtitle_opts(subs))
        if job.get("embed_subs") and kind != "audio":
            opts["postprocessors"].append({"key": "FFmpegEmbedSubtitle"})
            # webm+subs falls back to mkv rather than webm: ffmpeg's subtitle
            # embedder is unreliable targeting a raw .webm container, and mkv
            # holds VP9/Opus (webm's own codecs) natively, so nothing is lost.
            opts["merge_output_format"] = container if container in ("mp4", "mkv") else "mkv"

    # Applies even without subs: a merge (bv*+ba) otherwise lands on
    # whatever extension the video-only stream happened to be, usually
    # webm. container='webm' leaves this unset deliberately -- letting
    # yt-dlp/ffmpeg pick naturally is the "explicitly asked for webm" case.
    if kind != "audio" and "merge_output_format" not in opts and container in ("mp4", "mkv"):
        opts["merge_output_format"] = container

    if kind == "audio" and AUDIO_FORMAT:
        opts["postprocessors"].append({
            "key": "FFmpegExtractAudio",
            "preferredcodec": AUDIO_FORMAT,
        })
        # else: passthrough -- keep the source m4a/opus, no re-encode.

    return opts


def build_subtitle_opts(subs: str) -> dict:
    langs = [s.strip() for s in subs.split(",") if s.strip()]
    return {
        "writesubtitles": True,
        "writeautomaticsub": True,  # auto-generated fallback when no human track
        "subtitleslangs": langs,
        "subtitlesformat": "srt/best",  # sidecar .srt always kept, Phase 6 needs it
    }


# ------------------------------------------------------------------- probe

def probe_formats(url: str) -> list[dict]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        opts["cookiefile"] = YTDLP_COOKIES
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    out = []
    for f in info.get("formats") or []:
        out.append({
            "format_id": f.get("format_id"),
            "ext": f.get("ext"),
            "height": f.get("height"),
            "fps": f.get("fps"),
            "filesize": f.get("filesize") or f.get("filesize_approx"),
            "acodec": f.get("acodec"),
            "vcodec": f.get("vcodec"),
            "note": f.get("format_note"),
        })
    return out


# --------------------------------------------------------------- download

def run_download(job: dict) -> dict:
    """Blocking. Runs under asyncio.to_thread. Returns the DB fields to
    write back for this job."""
    job_id = job["id"]
    logger = JobLogger(job_id)
    hook = make_progress_hook(job_id, stage_n_guess(build_format(job["kind"], job["quality"])))
    opts = build_ydl_opts(job, progress_hook=hook, logger=logger)

    _job_thread[job_id] = threading.get_ident()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(job["url"], download=True)
        # NOT d['filename'] from the 'finished' progress-hook event -- that's
        # pre-postprocessing and has the wrong extension after a merge or
        # audio extraction.
        downloads = info.get("requested_downloads") or []
        filepath = downloads[0]["filepath"] if downloads else (info.get("filepath") or info.get("_filename"))
        return {
            "status": "done",
            "progress": 100.0,
            "filepath": filepath,
            "title": info.get("title") or job["url"],
            "error": None,
            "log": logger.dump(),
            # not a DB column -- carries 'subtitles' (human) vs
            # 'automatic_captions' (auto) provenance for Phase 6a's rollup
            # dedup + language matching. Stripped by the caller before the
            # result dict is handed to db.update_job.
            "_info": info,
        }
    except DownloadCancelled:
        return {"status": "canceled", "error": "canceled by user", "log": logger.dump()}
    except Exception as e:  # yt-dlp raises a lot of different types
        # yt-dlp's own logger callback already prints WARNING/ERROR lines as
        # they happen; this covers exceptions that never went through it
        # (e.g. raised before the logger was wired up, or by a postprocessor).
        log_line(f"[job {job_id}] ERROR: {type(e).__name__}: {e}")
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "log": logger.dump()}
    finally:
        _job_thread.pop(job_id, None)
        _thread_proc.pop(threading.get_ident(), None)
        CANCELED.discard(job_id)


def expand_playlist(job: dict) -> None:
    """Blocking. extract_flat enumeration, insert children in one
    transaction, then mark the parent done. Off the request thread --
    hundreds of entries would otherwise stall it."""
    job_id = job["id"]
    logger = JobLogger(job_id)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "socket_timeout": 30,
        "logger": logger,
    }
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        opts["cookiefile"] = YTDLP_COOKIES
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(job["url"], download=False)
        entries = info.get("entries") or []
        if job_id in CANCELED:
            db.update_job(job_id, status="canceled", error="canceled by user")
            CANCELED.discard(job_id)
            return
        conn = db.get_conn()
        with conn:
            for entry in entries:
                if not entry:
                    continue
                url = entry.get("url") or entry.get("webpage_url")
                if not url:
                    continue
                conn.execute(
                    "INSERT INTO jobs (url, title, kind, quality, container, subs, embed_subs, parent_id, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued')",
                    (
                        url,
                        entry.get("title"),
                        job.get("child_kind") or "video",
                        job["quality"],
                        job.get("container") or "mp4",
                        job["subs"],
                        job["embed_subs"],
                        job_id,
                    ),
                )
        db.update_job(job_id, status="done", title=info.get("title") or job["url"], progress=100.0, log=logger.dump())
    except Exception as e:
        log_line(f"[job {job_id}] ERROR: {type(e).__name__}: {e}")
        db.update_job(job_id, status="error", error=f"{type(e).__name__}: {e}", log=logger.dump())


# ------------------------------------------------------------------ cancel

def request_cancel(job: dict) -> None:
    job_id = job["id"]
    if job["status"] == "queued":
        db.update_job(job_id, status="canceled", error="canceled by user")
        return
    CANCELED.add(job_id)
    thread_id = _job_thread.get(job_id)
    proc = _thread_proc.get(thread_id) if thread_id is not None else None
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


# ------------------------------------------------------------------ delete

def safe_delete_file(
    filepath: str, downloads_dir: str = DOWNLOADS_DIR, keep_sidecars: bool = False,
) -> list[str]:
    """os.path.realpath the stored path and assert it's under downloads_dir
    before unlinking anything -- the path comes out of the DB rather than
    the request, so this is cheap insurance against a corrupted row, not a
    live hole. Also removes .part/.srt/.vtt siblings.

    keep_sidecars=True removes only the media file and its .part, leaving
    subtitles alone. That's the re-download case: the point is to replace
    the video (e.g. an AV1 file that won't play on iOS) while keeping
    transcripts that took an hour of Whisper to produce and are still
    perfectly valid -- they're named against the same stem the re-download
    will land on."""
    if not filepath:
        return []
    real = os.path.realpath(filepath)
    root = os.path.realpath(downloads_dir)
    if real != root and not real.startswith(root + os.sep):
        raise ValueError(f"refusing to delete outside downloads dir: {real}")

    removed = []
    base, _ext = os.path.splitext(real)
    if keep_sidecars:
        candidates = {real, real + ".part"}
    else:
        candidates = {real, real + ".part", base + ".srt"}
        d, prefix = os.path.dirname(real), os.path.basename(base)
        try:
            for name in os.listdir(d):
                if name.startswith(prefix) and (name.endswith(".srt") or name.endswith(".part") or name.endswith(".vtt")):
                    candidates.add(os.path.join(d, name))
        except FileNotFoundError:
            pass
    for c in candidates:
        if os.path.exists(c):
            os.remove(c)
            removed.append(c)
    return removed


# ------------------------------------------------------- separation (P5)

_DEMUCS_PCT_RE = re.compile(r"(\d+)%")


class DemucsProgressParser:
    """Demucs writes its tqdm bar to stderr with '\\r' redraws and emits no
    newline until the whole separation finishes. readline()/line-iteration
    over that blocks until the run is over and the bar sits frozen at 0% the
    entire time. Feed raw chunks (os.read(fd, n)) here instead; each feed()
    call returns the latest percent found, checking the newest (possibly
    still-unterminated) segment first."""

    def __init__(self):
        self.buf = ""

    def feed(self, chunk: str) -> int | None:
        self.buf += chunk
        segments = re.split(r"[\r\n]", self.buf)
        self.buf = segments[-1]  # last segment may still be mid-write, keep it
        # newest first: the in-progress segment, then completed ones in
        # reverse order -- tqdm's most recent redraw carries the real state.
        for seg in [self.buf] + segments[:-1][::-1]:
            m = _DEMUCS_PCT_RE.search(seg)
            if m:
                return int(m.group(1))
        return None


def run_separation(job: dict) -> dict:
    """Blocking. job['filepath'] is the completed download. Runs demucs,
    encodes no_vocals.wav to the job's audio format, and returns the DB
    fields to write back. Caller (worker._run_separation_stage) is
    responsible for the 'separating' status transition and for running this
    under SEPARATION_SEM."""
    job_id = job["id"]
    src = job["filepath"]

    if shutil.which("demucs") is None:
        msg = "vocal separation not enabled -- rebuild with WITH_DEMUCS=true"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}

    _job_thread[job_id] = threading.get_ident()
    tmpdir = tempfile.mkdtemp(prefix=f"demucs-{job_id}-")
    try:
        proc = subprocess.Popen(
            ["demucs", "-n", DEMUCS_MODEL, "--two-stems=vocals", "-o", tmpdir, src],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        _thread_proc[threading.get_ident()] = proc
        parser = DemucsProgressParser()
        last_flush = 0.0
        fd = proc.stderr.fileno()
        while True:
            if job_id in CANCELED:
                try:
                    proc.kill()
                except OSError:
                    pass
                break
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            pct = parser.feed(chunk.decode(errors="replace"))
            if pct is not None:
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    db.update_job(job_id, progress=float(pct), stage="separating vocals")
                    last_flush = now
        ret = proc.wait()

        if job_id in CANCELED:
            return {"status": "canceled", "error": "canceled by user"}
        if ret != 0:
            msg = f"demucs exited with status {ret}"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}

        # Demucs writes nested at {tmpdir}/{model}/{track_stem}/no_vocals.wav,
        # not flat in tmpdir.
        stem = os.path.splitext(os.path.basename(src))[0]
        novocals_wav = os.path.join(tmpdir, DEMUCS_MODEL, stem, "no_vocals.wav")
        if not os.path.exists(novocals_wav):
            msg = f"demucs output not found: {novocals_wav}"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}

        ext = AUDIO_FORMAT or os.path.splitext(src)[1].lstrip(".") or "m4a"
        out_path = os.path.join(os.path.dirname(src), f"{stem}_novocals.{ext}")
        subprocess.run(["ffmpeg", "-y", "-i", novocals_wav, out_path], check=True, capture_output=True)

        if KEEP_VOCALS:
            vocals_wav = os.path.join(tmpdir, DEMUCS_MODEL, stem, "vocals.wav")
            if os.path.exists(vocals_wav):
                vocals_out = os.path.join(os.path.dirname(src), f"{stem}_vocals.{ext}")
                subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, vocals_out], check=True, capture_output=True)

        return {"status": "done", "progress": 100.0, "filepath": out_path}
    except subprocess.CalledProcessError as e:
        log_line(f"[job {job_id}] ERROR: ffmpeg encode failed: {e}")
        return {"status": "error", "error": f"ffmpeg encode failed: {e}"}
    finally:
        _job_thread.pop(job_id, None)
        _thread_proc.pop(threading.get_ident(), None)
        shutil.rmtree(tmpdir, ignore_errors=True)


# -------------------------------------------------------- subtitles (P6)

# Small, deliberately partial ISO 639-1 -> 639-2 table covering realistic
# usage (ffmpeg's language= metadata wants a real 3-letter code, not a
# display string like "zh+en"). Falls back to the 2-letter code for
# anything not listed -- not spec-correct but harmless as a player hint.
# ponytail: hardcoded table, not the pycountry dependency, for ~15 languages.
_ISO_639_2 = {
    "en": "eng", "zh": "zho", "es": "spa", "fr": "fra", "de": "deu",
    "ja": "jpn", "ko": "kor", "pt": "por", "ru": "rus", "it": "ita",
    "ar": "ara", "hi": "hin", "vi": "vie", "th": "tha", "id": "ind",
    "nl": "nld", "pl": "pol", "tr": "tur",
}


def iso639_2(lang_code: str) -> str:
    base = (lang_code or "").split("-")[0].lower()
    return _ISO_639_2.get(base, base)


def find_subtitle_files(base: str) -> dict[str, str]:
    """base is a filepath without extension, e.g. '/downloads/Title [id]'.
    Returns {lang_code: srt_path} for every '<base>.<lang>.srt' sidecar
    yt-dlp wrote alongside the output."""
    out = {}
    d = os.path.dirname(base) or "."
    prefix = os.path.basename(base)
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return out
    for name in names:
        if name.startswith(prefix + ".") and name.endswith(".srt"):
            lang = name[len(prefix) + 1: -len(".srt")]
            out[lang] = os.path.join(d, name)
    return out


def match_lang(requested: str, available: dict[str, str], human: set[str] | None = None) -> str | None:
    """Tolerant prefix match: a request for 'zh' finds 'zh-Hans', 'zh-Hant',
    'zh-CN', 'zh-TW', etc. Exact key wins; among prefix matches, a
    human-authored track (in `human`) is preferred over an auto-generated
    one, then alphabetical for determinism."""
    if requested in available:
        return requested
    want = requested.split("-")[0].lower()
    candidates = sorted(
        (lang for lang in available if lang.split("-")[0].lower() == want),
        key=lambda lang: (lang not in (human or set()), lang),
    )
    return candidates[0] if candidates else None


def _overlap_words(prev_words: list[str], cur_words: list[str]) -> int:
    """Longest suffix of prev_words that's also a prefix of cur_words."""
    n = min(len(prev_words), len(cur_words))
    for k in range(n, 0, -1):
        if prev_words[-k:] == cur_words[:k]:
            return k
    return 0


def dedup_rollup_captions(items: list) -> list:
    """YouTube auto-generated (writeautomaticsub) tracks are rollup
    captions: each cue repeats the tail words of the previous cue, an
    artifact of vtt->srt conversion. Trims the repeated leading words off
    each cue against the raw text of its predecessor; drops a cue outright
    if nothing new remains. Mutates and returns the surviving items."""
    out = []
    prev_raw = ""
    for item in items:
        raw = item.text.strip()
        words = raw.split()
        n = _overlap_words(prev_raw.split(), words)
        trimmed = " ".join(words[n:]) if n else raw
        prev_raw = raw
        if trimmed:
            item.text = trimmed
            out.append(item)
    return out


def dedup_srt_file(src_path: str, dst_path: str) -> None:
    subs = pysrt.open(src_path)
    kept = dedup_rollup_captions(list(subs))
    out = pysrt.SubRipFile()
    for i, item in enumerate(kept, start=1):
        item.index = i
        out.append(item)
    out.save(dst_path, encoding="utf-8")


def merge_bilingual_subs(
    job: dict, orig_base: str, info: dict | None, logger: "JobLogger",
    auto_sync: bool = False, manual_offset_sec: float | None = None,
) -> str | None:
    """Runs vendored merge_subtitles for job['sub_primary']/['sub_secondary']
    against the sidecar .srt files next to the download. Degrades, never
    raises: a missing language or a zero-cue merge result falls back to
    keeping the original per-language tracks (caller just skips the merged
    track then). auto_sync defaults off -- 6a's two tracks share a timebase
    (offset is 0 by construction); 6c's externally uploaded track passes
    auto_sync=True since the offset there is genuinely unknown."""
    primary, secondary = job.get("sub_primary"), job.get("sub_secondary")
    if not primary or not secondary:
        return None

    available = find_subtitle_files(orig_base)
    know_provenance = info is not None
    human = set((info or {}).get("subtitles") or {}) if know_provenance else set()

    p_lang = match_lang(primary, available, human)
    s_lang = match_lang(secondary, available, human)
    if not p_lang or not s_lang:
        missing = primary if not p_lang else secondary
        logger.warning(f"merge_subs: '{missing}' track not available, keeping single track")
        return None

    tmpdir = tempfile.mkdtemp(prefix="submerge-")
    try:
        p_path, s_path = available[p_lang], available[s_lang]
        # Auto-generated tracks need the rollup-caption scrub before
        # merging. If we don't know provenance (e.g. resumed after a
        # restart lost the original info dict), leave tracks as downloaded
        # rather than guessing.
        if know_provenance and p_lang not in human:
            p_path = os.path.join(tmpdir, "primary.srt")
            dedup_srt_file(available[p_lang], p_path)
        if know_provenance and s_lang not in human:
            s_path = os.path.join(tmpdir, "secondary.srt")
            dedup_srt_file(available[s_lang], s_path)

        out_path = f"{orig_base}.{primary}-{secondary}.srt"
        merge_subtitles(
            p_path, s_path, out_path,
            auto_sync=auto_sync, manual_offset_sec=manual_offset_sec,
        )

        if not os.path.exists(out_path) or len(pysrt.open(out_path)) == 0:
            logger.warning("merge_subs: merge produced zero cues, keeping original tracks")
            if os.path.exists(out_path):
                os.remove(out_path)
            return None
        return out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def collect_sub_tracks(job: dict, orig_base: str, merged_srt: str | None) -> list[tuple[str, str, str]]:
    """(path, iso639-2 code, display title) for every subtitle track that
    should be embedded: each requested language plus the merged bilingual
    track, if one was produced."""
    available = find_subtitle_files(orig_base)
    tracks: list[tuple[str, str, str]] = []
    seen = set()
    for requested in (job.get("subs") or "").split(","):
        requested = requested.strip()
        if not requested:
            continue
        lang = match_lang(requested, available)
        if lang and available[lang] not in seen:
            tracks.append((available[lang], iso639_2(lang), lang))
            seen.add(available[lang])
    if merged_srt:
        tracks.append((merged_srt, iso639_2(job["sub_primary"]), f"{job['sub_primary']}+{job['sub_secondary']}"))
    return tracks


def build_mux_cmd(src_path: str, sub_tracks: list[tuple[str, str, str]], out_path: str, copy_all: bool) -> list[str]:
    """Stream-copy ffmpeg argv: one -i/-map pair per subtitle track.
    copy_all=True maps every stream already in src_path (video jobs, adding
    the merged track alongside subs yt-dlp already embedded); False maps
    only the audio stream (audio-only outputs, Phase 6b).

    The subtitle codec follows the CONTAINER, and getting this wrong is how
    an .mp4 ends up not being an mp4: SRT is a Matroska subtitle codec and
    MP4 can't hold it, so an mp4 output must transcode subtitles to
    mov_text. (This is why the in-place mux used to write a '.tmp.mkv' and
    rename it over the .mp4 -- producing a Matroska file with an .mp4
    name, which Chrome plays and iOS Safari refuses outright.)

    +faststart for mp4 moves the moov atom to the front. Without it a
    progressive HTTP player has to fetch the end of the file before it can
    start, which desktop browsers paper over with range requests and iOS
    often simply fails."""
    ext = os.path.splitext(out_path)[1].lower()
    is_mp4 = ext in (".mp4", ".m4v")
    cmd = ["ffmpeg", "-y", "-i", src_path]
    for path, _lang, _title in sub_tracks:
        cmd += ["-i", path]
    cmd += ["-map", "0"] if copy_all else ["-map", "0:a"]
    for i in range(1, len(sub_tracks) + 1):
        cmd += ["-map", str(i)]
    cmd += ["-c", "copy"] if copy_all else ["-c:a", "copy"]
    cmd += ["-c:s", "mov_text" if is_mp4 else "srt"]
    if is_mp4:
        cmd += ["-movflags", "+faststart"]
    for i, (_path, lang, title) in enumerate(sub_tracks):
        cmd += [f"-metadata:s:s:{i}", f"language={lang}"]
        if title:
            cmd += [f"-metadata:s:s:{i}", f"title={title}"]
    cmd.append(out_path)
    return cmd


def run_mux(src_path: str, sub_tracks: list[tuple[str, str, str]], out_path: str, copy_all: bool) -> None:
    """ffmpeg can't read and write the same file, so when out_path ==
    src_path (adding a track to an already-muxed video in place) this
    writes to a temp file first and atomically replaces the original."""
    same_file = os.path.abspath(out_path) == os.path.abspath(src_path)
    # The temp file MUST keep the real extension. ffmpeg picks its muxer
    # from the output filename, so a '.tmp.mkv' scratch file produced
    # Matroska and os.replace then gave it an .mp4 name -- a container lie
    # that desktop players tolerate and iOS does not.
    base, ext = os.path.splitext(out_path)
    tmp_out = f"{base}.tmp{ext}" if same_file else out_path
    cmd = build_mux_cmd(src_path, sub_tracks, tmp_out, copy_all)
    subprocess.run(cmd, check=True, capture_output=True)
    if same_file:
        os.replace(tmp_out, out_path)


def probe_media(path: str) -> dict:
    """ffprobe summary of what a file ACTUALLY is: container, and the codec
    of each stream. Cheap (reads headers, not the file) and the only honest
    answer to "why won't this play on my phone" -- the extension is a
    filename, not a fact, and this project has already shipped a Matroska
    file called .mp4 once."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=format_name,duration:stream=index,codec_type,codec_name,profile",
        path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, timeout=30).stdout
        return json.loads(out or b"{}")
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def describe_media(path: str) -> str:
    """One log-friendly line from probe_media, e.g.
    'container=mov,mp4,m4a video=h264(High) audio=aac subs=mov_text'."""
    info = probe_media(path)
    if "error" in info:
        return f"probe failed: {info['error']}"
    parts = [f"container={info.get('format', {}).get('format_name', '?')}"]
    for st in info.get("streams", []):
        kind = {"video": "video", "audio": "audio", "subtitle": "subs"}.get(st.get("codec_type"), st.get("codec_type"))
        name = st.get("codec_name", "?")
        profile = st.get("profile")
        parts.append(f"{kind}={name}({profile})" if profile else f"{kind}={name}")
    return " ".join(parts)


def resolve_orig_base(job: dict) -> str:
    """Recovers the pre-separation/pre-mux filename stem from the job's
    current filepath, without needing a DB column for it: our own
    transformations are the only things that change the stem (demucs
    appends '_novocals'; muxing only ever changes the extension or, for
    video, replaces the file in place at the same path)."""
    stem = os.path.splitext(job["filepath"])[0]
    suffix = "_novocals"
    if job.get("strip_vocals") and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def list_job_files(job: dict) -> list[str]:
    """Every file on disk sharing this job's pre-pipeline filename stem: the
    current output (original download, post-separation '_novocals', or
    post-mux -- whichever stage last touched it) plus every sidecar .srt
    (downloaded, bilingual-merged, Whisper/OCR-generated, translated, or
    uploaded via 6c) -- they're all named against the same
    resolve_orig_base() stem no matter which pipeline stages ran, since
    every srt-writing stage receives that same pre-pipeline base. Used by
    the UI's per-job file list so every artifact is individually
    downloadable, not just the main muxed output via 'Open'.

    Deliberately a plain string-prefix match, not requiring a '.' boundary
    the way find_subtitle_files/find_audio_source do -- that's needed here
    specifically to also catch demucs's '_novocals' suffix, which has no
    dot before it."""
    if not job.get("filepath"):
        return []
    base = resolve_orig_base(job)
    d = os.path.dirname(base) or "."
    prefix = os.path.basename(base)
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return []
    return [os.path.join(d, n) for n in sorted(names) if n.startswith(prefix)]


def find_audio_source(job: dict) -> str | None:
    """Locates the pre-mux audio file next to an already-muxed .mkv (Phase
    6c re-upload path) -- our mux step always writes a *new* .mkv and never
    touches the original audio sidecar, so it's still on disk."""
    stem = resolve_orig_base(job)
    if job.get("strip_vocals"):
        stem += "_novocals"
    d = os.path.dirname(stem) or "."
    base = os.path.basename(stem)
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return None
    for name in names:
        if name.startswith(base + ".") and not name.endswith(".srt") and not name.endswith(".mkv"):
            return os.path.join(d, name)
    return None


def decode_srt_bytes(raw: bytes) -> str:
    """Uploaded .srt files (Phase 6c) are frequently not UTF-8 -- Chinese
    subtitle sites commonly use GBK, GB18030, or BIG5, and pysrt.open()
    assumes UTF-8 and raises UnicodeDecodeError on the first cue. yt-dlp's
    own subtitle downloads are always UTF-8 already, so this only runs for
    the upload path."""
    import charset_normalizer
    best = charset_normalizer.from_bytes(raw).best()
    if best is None:
        return raw.decode("utf-8", errors="replace")
    return str(best)


# --------------------------------------------------------- transcribe (P7)
#
# Hardsub subtitle generation for reuploads that only have burned-in
# subtitles: Whisper (ASR, transcribes spoken audio) and/or Tesseract OCR
# (reads the actual burned-in pixels off sampled frames). Both produce a
# plain .srt that feeds into the exact same collect_sub_tracks/run_mux
# machinery as a downloaded or bilingual-merged track above -- this section
# only needs to get from "job + a source file" to "(path, iso-lang, title)
# tuples", the muxing itself is unchanged.

# chi_sim+chi_tra+eng, not chi_sim+eng -- see the OCR_LANG comment above,
# a lone CJK script combined with eng is confirmed to produce garbage.
_OCR_LANG_HINTS = {"zh": "chi_sim+chi_tra+eng", "en": "eng"}


OCR_REGIONS = ("bottom", "full")


def ocr_lang_for(gen_subs_lang: str | None, ocr_lang: str | None = None) -> str:
    """Resolves the Tesseract '+'-joined lang string for a job.

    An explicit per-job ocr_lang wins outright and is used verbatim -- the
    gen_subs_lang mapping below can't express the distinction that matters
    most in practice: 'zh' says nothing about simplified vs. traditional,
    and including chi_tra makes tesseract render simplified source text in
    *traditional* forms (九阳 -> 九陽). Only the user knows which their
    video is, so when they say, we don't second-guess it.

    Falling back: the gen_subs_lang hint through _OCR_LANG_HINTS, then the
    OCR_LANG env default."""
    if ocr_lang and ocr_lang.strip():
        return ocr_lang.strip()
    if not gen_subs_lang:
        return OCR_LANG
    return _OCR_LANG_HINTS.get(gen_subs_lang.split("-")[0].lower(), OCR_LANG)


def otsu_threshold(hist: list[int]) -> int:
    """Otsu's method: the 0-255 grey level that maximizes between-class
    variance, i.e. the split this particular histogram itself argues for.
    Pure stdlib arithmetic over PIL's 256-bin histogram -- numpy/opencv
    would each be a new dependency for one textbook loop.

    Degenerate input (a uniform image, where every pixel lands in one bin)
    has no meaningful split; 127 is returned and the caller's binarize step
    turns the frame into a single flat colour, which reads as "no text
    here" -- the same thing a blank frame already means downstream."""
    total = sum(hist)
    if total == 0:
        return 127
    sum_all = sum(i * h for i, h in enumerate(hist))
    w_b = 0
    sum_b = 0
    best_var = -1.0
    best_t = 127
    for t, h in enumerate(hist):
        w_b += h
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * h
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def binarize_for_ocr(img):
    """Grayscale -> per-frame Otsu -> guaranteed dark-text-on-light output.
    Takes and returns a PIL Image (imported by the caller, not here --
    Pillow rides in as pytesseract's dependency and stays out of this
    module's import list).

    This is the deferred half of the note above run_ocr_transcribe's frame
    loop. A *fixed* threshold was tried and rejected there, and correctly:
    it has to assume a polarity, and hardsub styling doesn't hold one still
    -- the constant that rescued white-on-video text blanked blue-on-dark
    text completely. Otsu removes the constant (each frame's own histogram
    picks the split), and the polarity guess is removed separately below,
    which is what makes this safe to run unconditionally where the fixed
    version wasn't.

    Polarity: subtitle glyphs are always the minority of pixels in a
    subtitle-sized crop -- a line of text simply doesn't cover half its own
    bounding box, whatever colour it is. So whichever class has fewer
    pixels is the text, and the image is inverted if needed to land on the
    dark-on-light that Tesseract is trained for. That's a property of
    subtitles, not of a colour, which is exactly what the fixed threshold
    lacked.
    """
    grey = img.convert("L")
    t = otsu_threshold(grey.histogram())
    bw = grey.point(lambda p: 255 if p > t else 0)
    hist = bw.histogram()
    # Minority class is the text; make it the dark one.
    if hist[0] > hist[255]:
        bw = bw.point(lambda p: 255 - p)
    return bw


def write_srt(cues: list[tuple[float, float, str]], path: str) -> None:
    """cues: (start_seconds, end_seconds, text). Shared by both the Whisper
    and OCR pipelines so SRT timestamp formatting (HH:MM:SS,mmm) and index
    numbering aren't written twice."""
    out = pysrt.SubRipFile()
    for i, (start, end, text) in enumerate(cues, start=1):
        out.append(pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(seconds=start),
            end=pysrt.SubRipTime(seconds=end),
            text=text,
        ))
    out.save(path, encoding="utf-8")


def srt_to_vtt(srt_path: str) -> str:
    """Converts an .srt sidecar to WebVTT text, in memory -- the browser's
    native <track kind="subtitles"> element only accepts WebVTT, not SRT,
    which is otherwise what every subtitle track this app produces or
    downloads is stored as. Structurally the formats are nearly identical:
    a 'WEBVTT' header line, and '.' instead of ',' as the millisecond
    separator, are the only real differences. All of this app's own .srt
    files are UTF-8 (yt-dlp writes UTF-8, write_srt writes UTF-8, the 6c
    upload path decodes to UTF-8 before saving) -- pysrt's default read
    encoding is correct here, unlike the upload path's own raw bytes,
    which go through decode_srt_bytes first for exactly that reason."""
    subs = pysrt.open(srt_path)
    lines = ["WEBVTT", ""]
    for item in subs:
        start, end = item.start, item.end
        lines.append(
            f"{int(start.hours):02d}:{int(start.minutes):02d}:{int(start.seconds):02d}.{int(start.milliseconds):03d}"
            " --> "
            f"{int(end.hours):02d}:{int(end.minutes):02d}:{int(end.seconds):02d}.{int(end.milliseconds):03d}"
        )
        lines.append(item.text)
        lines.append("")
    return "\n".join(lines)


def ocr_frames_to_cues(
    frame_texts: list[tuple[float, str]], frame_interval: float, similarity: float = 0.6,
) -> list[tuple[float, float, str]]:
    """Pure merge logic, unit-tested with synthetic input -- no real
    tesseract/ffmpeg involved. Normalization is strip + collapse internal
    whitespace only (not punctuation).

    Grouping is *fuzzy* (difflib.SequenceMatcher, stdlib, no new
    dependency), not an exact string match -- verified directly against a
    real run: OCR on a completely static line of on-screen text produced a
    different misread on nearly every 0.5s-apart frame (one wrong
    character each time). Exact matching treated every one of those as a
    text change, fragmenting a single real 3-second cue into six
    wrong-duration ones -- this is the actual mechanism behind subtitles
    coming out visibly mistimed, not just occasionally misspelled. Each
    frame is compared against its group's current majority-vote reading
    (not the first frame, which self-corrects as more frames arrive), and
    a finished cue's text is that group's most common normalized reading,
    not whichever frame happened to be sampled first.

    frame_texts: (timestamp_seconds, raw_ocr_text) for every sampled frame,
    in timestamp order. Blank (whitespace-only) frames close and skip any
    open cue -- that's "no subtitle showing".

    A single dissimilar frame doesn't split a cue by itself -- it's held
    as `pending` for one more frame first. If the *next* frame matches the
    cue's majority again, the held frame was just a one-off OCR blip and
    gets folded back in silently (this is exactly what happens with the
    real captured noise above: frame 4 of 6 misreads badly, frame 5 reads
    correctly again). Only two *consecutive* dissimilar frames confirm a
    genuine cue change, opening the new cue at the first of the two. A
    trailing pending frame with no next frame to confirm or deny it is
    kept as its own short cue rather than silently dropped -- we have no
    evidence either way, and dropping what might be real content is worse
    than occasionally keeping a spurious one-frame one."""
    from collections import Counter
    from difflib import SequenceMatcher

    def norm(s: str) -> str:
        return " ".join(s.split())

    def is_similar(a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= similarity

    cues: list[tuple[float, float, str]] = []
    group: list[str] = []
    group_start = group_end = 0.0
    pending: tuple[float, str] | None = None

    def majority() -> str:
        return Counter(group).most_common(1)[0][0]

    def flush():
        if group:
            cues.append((group_start, group_end + frame_interval, majority()))

    for ts, raw in frame_texts:
        t = norm(raw)
        if not t:
            pending = None
            flush()
            group = []
            continue
        if not group:
            group, group_start, group_end, pending = [t], ts, ts, None
            continue
        if is_similar(t, majority()):
            pending = None
            group.append(t)
            group_end = ts
            continue
        if pending is None:
            pending = (ts, t)
            continue
        # second consecutive miss confirms a real cue change, not noise
        flush()
        group = [pending[1], t]
        group_start, group_end = pending[0], ts
        pending = None
    if pending is not None:
        flush()
        cues.append((pending[0], pending[0] + frame_interval, pending[1]))
    else:
        flush()
    return cues


def extract_audio_wav(job_id: int, src: str, out_wav: str) -> None:
    """ffmpeg mono 16kHz wav -- Whisper's expected input. Popen (not
    subprocess.run) and registered in _thread_proc so request_cancel can
    kill it mid-extraction, same tracking run_separation uses for demucs.
    Raises subprocess.CalledProcessError on a genuine failure; caller checks
    CANCELED separately to tell a kill apart from a real error."""
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-vn", out_wav]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _thread_proc[threading.get_ident()] = proc
    _, stderr = proc.communicate()
    if job_id in CANCELED:
        return
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


# Approximate on-disk size of each int8 model, for a percentage during the
# first-use download. Approximate on purpose: the exact figure depends on
# the ctranslate2 conversion and the tokenizer files that ride along, and
# being 10% out on a progress readout costs nothing, whereas pinning exact
# byte counts would need updating every time upstream repacks a model.
_WHISPER_MODEL_BYTES = {
    "tiny": 75_000_000,
    "base": 145_000_000,
    "small": 500_000_000,
    "medium": 1_530_000_000,
    "large-v3": 3_090_000_000,
}


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass  # a file vanishing mid-walk is normal during a download
    return total


@contextlib.contextmanager
def _watch_model_download(job_id: int, model_name: str):
    """Reports first-use model download progress by measuring the download
    directory, since faster-whisper exposes no progress callback.

    Stays quiet when the model is already cached: nothing is downloaded, the
    directory doesn't grow, and no line is emitted. That's why this measures
    growth rather than trying to detect 'is it cached?' up front -- the
    latter needs the huggingface_hub cache layout, an implementation detail
    that would silently start lying if it changed."""
    start_size = _dir_size(WHISPER_MODEL_DIR)
    expected = _WHISPER_MODEL_BYTES.get(model_name)
    stop = threading.Event()

    def watch():
        announced = False
        while not stop.wait(HEARTBEAT_SECONDS):
            grown = _dir_size(WHISPER_MODEL_DIR) - start_size
            if grown <= 0:
                continue  # already cached, or nothing has landed yet
            if not announced:
                announced = True
                append_job_log(job_id, f"whisper: downloading model {model_name!r} (first use of this size)")
            mb = grown / 1_000_000
            pct = f", ~{min(99, int(grown / expected * 100))}%" if expected else ""
            db.update_job(job_id, stage=f"downloading whisper model ({mb:.0f}MB{pct})")
            log_line(f"[job {job_id}] whisper: model download {mb:.0f}MB{pct}")

    db.update_job(job_id, stage="loading whisper model")
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)
        grown = _dir_size(WHISPER_MODEL_DIR) - start_size
        if grown > 0:
            append_job_log(job_id, f"whisper: model {model_name!r} downloaded, {grown / 1_000_000:.0f}MB")


# Credit lines Whisper memorized from the subtitle corpus it was trained on
# and emits verbatim over non-speech audio. VAD removes most of the audio
# that triggers this; these are the ones that still get through, matched on
# the whole cue so a sentence merely containing the words survives.
_HALLUCINATED_CUES = re.compile(
    r"""^\W*(
        (sub(title)?s?|caption(ing)?)\s*(are|were)?\s*(brought\s+to\s+you\s+)?
            (by|from|courtesy\s+of|provided\s+by)\b.*
      | .*\b(amara\.org|opensubtitles|subscene|addic7ed|cdramabase)\b.*
      | thanks?\s+(you\s+)?for\s+watching.*
      | (please\s+)?(don't\s+forget\s+to\s+)?(like|subscribe|follow)(\s+and\s+\w+)*[\s!.]*
      | 字幕(由|组|制作|翻译).*
      | 请?(不吝)?(点赞|订阅|关注).*
    )\W*$""",
    re.IGNORECASE | re.VERBOSE,
)


def drop_hallucinated_cues(
    cues: list[tuple[float, float, str]],
) -> tuple[list[tuple[float, float, str]], list[str]]:
    """Removes memorized-credit cues. Returns (kept, dropped_texts) so the
    caller can report what went, rather than silently editing a transcript.

    Deliberately conservative: whole-cue matches only. A cue that merely
    contains "subscribe" inside a real sentence is real dialogue, and
    dropping actual content is worse than leaving one artifact in."""
    kept, dropped = [], []
    for start, end, text in cues:
        if _HALLUCINATED_CUES.match(text.strip()):
            dropped.append(text.strip())
        else:
            kept.append((start, end, text))
    return kept, dropped


def run_whisper_transcribe(job: dict, current_filepath: str, out_srt: str, task: str = "transcribe") -> dict:
    """Blocking. kind='video' jobs get their audio extracted to a temp mono
    16kHz wav first; kind='audio' jobs transcribe current_filepath directly
    (whatever file is current post-separation). Caller
    (worker._run_transcribe_stage) runs this under TRANSCRIBE_SEM.

    task='translate' makes Whisper emit ENGLISH directly from the audio
    instead of the spoken language -- a different decoding task, not a
    post-process, so it needs its own pass over the audio. It is a much
    better English track than transcribing and then running the text
    through argos: the model translates from the audio itself, with all the
    context that carries, rather than from a transcript that has already
    thrown that away. Whisper's translate task only ever targets English;
    any other target still goes through argos."""
    job_id = job["id"]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        msg = "speech-to-text not enabled -- rebuild with WITH_TRANSCRIBE=true"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}

    tmpdir = None
    audio_src = current_filepath
    try:
        if job["kind"] != "audio":
            tmpdir = tempfile.mkdtemp(prefix=f"whisper-{job_id}-")
            audio_src = os.path.join(tmpdir, "audio.wav")
            extract_audio_wav(job_id, current_filepath, audio_src)
            if job_id in CANCELED:
                return {"status": "canceled", "error": "canceled by user"}

        # download_root: see WHISPER_MODEL_DIR above. Created here rather
        # than assumed -- /data is a mounted volume, and the first whisper
        # job on a fresh install is exactly when it won't exist yet.
        os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
        model_name = whisper_model_for(job)
        # WhisperModel() blocks while it downloads, with no callback to hook
        # -- and on a first run that's up to ~3GB of total silence, which is
        # indistinguishable from a hang (this app has already burned an
        # afternoon on exactly that confusion once). Watch the download
        # directory grow instead of asking the library.
        with _watch_model_download(job_id, model_name):
            model = WhisperModel(
                model_name, device="cpu", compute_type="int8", download_root=WHISPER_MODEL_DIR,
            )
        # vad_filter and condition_on_previous_text both exist to suppress
        # hallucination, which is a different failure from mistranscription
        # and is not fixed by a bigger model:
        #
        # Whisper was trained on scraped subtitle files, so on audio with no
        # speech in it (music, room tone, silence) it emits the credit lines
        # it memorized from that corpus -- "Subtitles brought to you by
        # <site>", "Thanks for watching". Observed here as an English credit
        # string sitting over Chinese dialogue. A larger model produces
        # those strings MORE fluently, not less, which is why large-v3 made
        # no difference. VAD cuts the non-speech spans before they ever
        # reach the decoder, removing the prompt for it.
        #
        # condition_on_previous_text=False stops the other half: with it on,
        # each window is conditioned on the previous window's text, so one
        # hallucinated line becomes the context for the next and the model
        # locks into repeating it for the rest of the file. Off, every
        # window is judged on its own audio. The cost is slightly less
        # coherence across sentence boundaries -- worth it.
        segments, info = model.transcribe(
            audio_src,
            language=job.get("gen_subs_lang") or None,
            task=task,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        total = info.duration or 0
        # Same reasoning as run_ocr_transcribe's log line -- which model and
        # which language (hint vs. auto-detected) actually ran is the first
        # thing worth knowing when a transcript looks wrong, visible from
        # the UI's Log button. getattr, not info.language directly -- this
        # is a diagnostic line, not worth risking an unhandled AttributeError
        # over if a future faster-whisper version renames the field.
        detected_lang = getattr(info, "language", "?")
        append_job_log(job_id, f"whisper: task={task!r} model={model_name!r} lang_hint={job.get('gen_subs_lang')!r} detected_lang={detected_lang!r}")

        cues = []
        last_flush = 0.0
        last_beat = time.monotonic()
        for seg in segments:
            if job_id in CANCELED:
                return {"status": "canceled", "error": "canceled by user"}
            text = seg.text.strip()
            if text:
                cues.append((seg.start, seg.end, text))
            now = time.monotonic()
            if now - last_flush >= 1.0:
                pct = min(100.0, seg.end / total * 100.0) if total else 0.0
                db.update_job(job_id, progress=round(pct, 1), stage="transcribing (whisper)")
                last_flush = now
            # Heartbeat to stdout. The DB progress above already drives the
            # UI bar, but from `docker logs` a multi-minute transcribe was
            # indistinguishable from a hang -- which is exactly the wrong
            # guess to invite, since the fix for one is "wait" and for the
            # other "restart". Sparse on purpose: this is a liveness
            # signal, not a progress bar.
            if now - last_beat >= HEARTBEAT_SECONDS:
                log_line(f"[job {job_id}] whisper[{task}]: {seg.end:.0f}s/{total:.0f}s, {len(cues)} cues so far")
                last_beat = now

        cues, dropped = drop_hallucinated_cues(cues)
        if dropped:
            sample = "; ".join(dropped[:3])
            append_job_log(job_id, f"whisper[{task}]: dropped {len(dropped)} hallucinated cue(s): {sample}")
        if not cues:
            msg = "whisper produced no cues (silent audio, or an unsupported/mismatched language hint)"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}
        write_srt(cues, out_srt)
        append_job_log(job_id, f"whisper[{task}]: {len(cues)} cues -> {os.path.basename(out_srt)}")
        return {"status": "done", "path": out_srt, "cues": len(cues), "dropped": len(dropped)}
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg audio extraction failed: {e}"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}
    except Exception as e:
        # Model load/download is the realistic failure here and it raises
        # anything but CalledProcessError -- PermissionError when the HF
        # cache isn't writable, OSError/network errors on the ~500MB first
        # fetch, RuntimeError from ctranslate2 on an unsupported CPU. This
        # function's contract is an error dict, so honour it for all of
        # them: an escaping exception becomes a job that sits in
        # 'transcribing' forever with nothing shown to the user.
        msg = f"whisper failed: {type(e).__name__}: {e}"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def run_ocr_transcribe(job: dict, video_src: str, out_srt: str) -> dict:
    """Blocking. Samples frames from video_src via ffmpeg (cropped to the
    bottom OCR_CROP_BOTTOM_PCT of the frame, at OCR_SAMPLE_FPS), OCRs each
    with pytesseract, and merges consecutive matching-text frames into cues
    via ocr_frames_to_cues. Caller (worker._run_transcribe_stage) runs this
    under TRANSCRIBE_SEM. Only ever called for kind='video' jobs -- there
    are no frames on an audio-only job, and main.py's _create_job clamps
    gen_subs='ocr'/'both' down to 'whisper' before a job like that is ever
    created."""
    job_id = job["id"]
    if shutil.which("tesseract") is None:
        # Shouldn't happen -- tesseract-ocr is installed unconditionally in
        # the Dockerfile, not gated behind a build flag like demucs/whisper
        # -- but defend anyway rather than let pytesseract raise an opaque
        # FileNotFoundError partway through a frame loop.
        msg = "tesseract binary not found on PATH"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}

    import pytesseract
    from PIL import Image  # Pillow: pytesseract's own dependency, not ours

    fps = OCR_SAMPLE_FPS
    lang = ocr_lang_for(job.get("gen_subs_lang"), job.get("ocr_lang"))
    region = job.get("ocr_region") or "bottom"
    # Recorded both to stdout and the job's own DB log -- the single most
    # useful line for diagnosing a bad OCR result after the fact (which
    # lang string and which region actually ran), visible from the UI's Log
    # button without needing docker logs at all.
    append_job_log(job_id, f"ocr: lang={lang!r} region={region!r} sample_fps={fps} crop_bottom_pct={OCR_CROP_BOTTOM_PCT} binarize={OCR_BINARIZE}")
    tmpdir = tempfile.mkdtemp(prefix=f"ocr-{job_id}-")
    try:
        pct = OCR_CROP_BOTTOM_PCT
        # 'full' OCRs the whole frame -- slower (more pixels per frame, and
        # --psm 7's single-line assumption is a worse fit), but the bottom
        # crop misses text placed anywhere else, e.g. corner labels or
        # captions positioned mid-frame, which read as "hardsubs silently
        # not decoded" rather than as an error.
        vf = f"fps={fps}" if region == "full" else f"fps={fps},crop=iw:ih*{pct}:0:ih*(1-{pct})"
        psm_config = "--psm 11" if region == "full" else "--psm 7"
        cmd = ["ffmpeg", "-y", "-i", video_src, "-vf", vf, "-q:v", "2", os.path.join(tmpdir, "%06d.png")]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _thread_proc[threading.get_ident()] = proc
        _, stderr = proc.communicate()
        if job_id in CANCELED:
            return {"status": "canceled", "error": "canceled by user"}
        if proc.returncode != 0:
            msg = f"ffmpeg frame extraction failed: {stderr.decode(errors='replace')[-500:]}"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}

        frames = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        if not frames:
            msg = "ocr: ffmpeg extracted no frames"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}

        frame_interval = 1.0 / fps
        frame_texts = []
        last_flush = 0.0
        last_beat = time.monotonic()
        for i, name in enumerate(frames):
            if job_id in CANCELED:
                return {"status": "canceled", "error": "canceled by user"}
            # --psm 7: treat the frame as a single line of text, not a full
            # page. Skips Tesseract's page-layout analysis entirely, which
            # is not just wasted work here but actively confusing it --
            # measured faster AND more accurate on real subtitle-style
            # crops (verified directly: 0.50s/frame default vs 0.41s/frame
            # with --psm 7, on the same misread-prone case).
            #
            # Frames are binarized before OCR (binarize_for_ocr): --psm 7
            # alone was not enough in practice -- stylized hardsubs
            # (gradient fill, contrasting outline, decorative face) over
            # moving video read as noise even with the right lang packs.
            # The earlier *fixed* threshold that got rejected here is not
            # what's running: see binarize_for_ocr for why per-frame Otsu
            # plus a derived polarity is a different proposition.
            # OCR_BINARIZE=0 turns it off without a rebuild if some
            # subtitle style turns out to do worse with it.
            # --psm 7 trades away the default mode's tolerance for a
            # text-free frame: forcing "there is exactly one line here"
            # onto a crop with no real text can make Tesseract's internal
            # line-finder latch onto a tiny noise fragment and raise
            # TesseractError ("Image too small to scale") instead of just
            # returning "" the way the default full-page mode would have.
            # Confirmed as a real, input-dependent failure (not one this
            # synthetic-image testing could force on demand) -- semantically
            # this frame has no recognizable text, exactly what a blank
            # result already means to ocr_frames_to_cues, so treat it the
            # same rather than letting it kill the whole job.
            #
            # region='full' overrides all of the above to --psm 11 (sparse
            # text, no layout assumption): "exactly one line" is simply
            # wrong for a whole frame, which is the case where text can be
            # in several places at once -- that's the entire reason to pick
            # full-frame in the first place.
            try:
                frame_img = Image.open(os.path.join(tmpdir, name))
                if OCR_BINARIZE:
                    frame_img = binarize_for_ocr(frame_img)
                text = pytesseract.image_to_string(frame_img, lang=lang, config=psm_config)
            except pytesseract.TesseractError:
                text = ""
            frame_texts.append((i * frame_interval, text))
            now = time.monotonic()
            if now - last_flush >= 1.0:
                db.update_job(job_id, progress=round((i + 1) / len(frames) * 100.0, 1), stage="transcribing (ocr)")
                last_flush = now
            # Liveness on stdout, same reasoning as the whisper loop above.
            if now - last_beat >= HEARTBEAT_SECONDS:
                blank_so_far = sum(1 for _t, txt in frame_texts if not txt.strip())
                log_line(f"[job {job_id}] ocr: frame {i + 1}/{len(frames)}, {blank_so_far} blank so far")
                last_beat = now

        cues = ocr_frames_to_cues(frame_texts, frame_interval)
        if not cues:
            msg = "ocr produced no cues (no burned-in subtitle text detected)"
            log_line(f"[job {job_id}] ERROR: {msg}")
            return {"status": "error", "error": msg}
        write_srt(cues, out_srt)
        # Frame count alongside cue count: a high blank-frame ratio is the
        # signature of OCR reading nothing at all, which otherwise looks
        # identical to a video that simply has few subtitles.
        blank = sum(1 for _t, txt in frame_texts if not txt.strip())
        append_job_log(
            job_id,
            f"ocr: {len(frames)} frames, {blank} blank, {len(cues)} cues -> {os.path.basename(out_srt)}",
        )
        return {"status": "done", "path": out_srt}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------- translate (P8)
#
# Translates an already-generated (Whisper/OCR) .srt into a second,
# additional soft-sub track via offline argos-translate -- no cloud API, no
# API key, an explicit user choice to keep this app self-hosted. Runs
# inline in run_transcribe below, after whichever engine(s) produced the
# original-language track(s): the original track is always kept, the
# translation is one more (path, iso-lang, title) tuple fed to the same
# collect_sub_tracks/run_mux machinery.

def get_argos_translator(source_lang: str, target_lang: str):
    """Ensures the source->target argos-translate package is installed
    (downloading it into TRANSLATE_MODEL_DIR on first use -- see the
    ARGOS_PACKAGES_DIR module comment above) and returns a plain str->str
    callable for that language pair. Raises RuntimeError with a clear,
    specific message -- never an opaque traceback -- if argostranslate
    isn't installed (WITH_TRANSLATE=false build) or the pair doesn't exist
    in argos's package index at all."""
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        raise RuntimeError("translation not enabled -- rebuild with WITH_TRANSLATE=true")

    from_code = (source_lang or "").split("-")[0].lower()
    to_code = (target_lang or "").split("-")[0].lower()

    installed = argostranslate.package.get_installed_packages()
    have = any(p.from_code == from_code and p.to_code == to_code for p in installed)
    if not have:
        argostranslate.package.update_package_index()
        ok = argostranslate.package.install_package_for_language_pair(from_code, to_code)
        if not ok:
            raise RuntimeError(
                f"no argos-translate package available for '{from_code}' -> '{to_code}'"
            )

    def _translate(text: str) -> str:
        return argostranslate.translate.translate(text, from_code, to_code)

    return _translate


def _translate_cues_with(
    cues: list[tuple[float, float, str]], translate_fn,
) -> list[tuple[float, float, str]]:
    """Pure mapping, unit-tested with a fake translate_fn -- no real
    argos-translate model involved. One call per cue (not batched: subtitle
    cue counts are small enough that per-cue call overhead doesn't matter,
    and it keeps the 1:1 timing/count guarantee trivially true rather than
    depending on a join/split round-trip through a batch translator)."""
    return [(start, end, translate_fn(text)) for start, end, text in cues]


def translate_cues(
    cues: list[tuple[float, float, str]], source_lang: str, target_lang: str,
) -> list[tuple[float, float, str]]:
    """Resolves the argos-translate model for source_lang->target_lang (see
    get_argos_translator) and translates every cue's text, keeping timings
    and cue count exactly 1:1 with the input."""
    translate_fn = get_argos_translator(source_lang, target_lang)
    return _translate_cues_with(cues, translate_fn)


def run_translate(job: dict, src_srt: str, source_lang: str, target_lang: str, out_srt: str) -> dict:
    """Blocking. Reads src_srt (an already-generated whisper/ocr track) back
    with pysrt, translates every cue via translate_cues, and writes out_srt
    with the same timings via write_srt (the same helper the whisper/ocr
    pipelines use, so timestamp formatting isn't written a third time)."""
    job_id = job["id"]
    try:
        subs = pysrt.open(src_srt)
        cues = [(item.start.ordinal / 1000.0, item.end.ordinal / 1000.0, item.text) for item in subs]
        translated = translate_cues(cues, source_lang, target_lang)
        write_srt(translated, out_srt)
        return {"status": "done", "path": out_srt}
    except RuntimeError as e:
        # get_argos_translator's own clear-message failures (not enabled /
        # language pair not available) -- pass the message straight through.
        log_line(f"[job {job_id}] ERROR: {e}")
        return {"status": "error", "error": str(e)}
    except Exception as e:
        msg = f"translation failed: {type(e).__name__}: {e}"
        log_line(f"[job {job_id}] ERROR: {msg}")
        return {"status": "error", "error": msg}


def _add_translated_track(
    job: dict, orig_base: str, src_srt: str, engine: str,
    source_lang: str, target_lang: str, tracks: list[tuple[str, str, str]],
) -> dict | None:
    """Runs run_translate for one already-generated track and appends
    (path, iso-lang, title) to tracks in place on success. Returns an error/
    canceled result dict on failure (already logged by run_translate), or
    None on success -- mirrors run_transcribe's own early-return-on-failure
    shape so callers can `if err: return err` the same way they already do
    for run_whisper_transcribe/run_ocr_transcribe."""
    out_srt = f"{orig_base}.{engine}.{target_lang}.srt"
    result = run_translate(job, src_srt, source_lang, target_lang, out_srt)
    if result["status"] != "done":
        return result
    tracks.append((out_srt, iso639_2(target_lang), f"{engine} -> {target_lang}"))
    append_job_log(job["id"], f"translate: {engine} {source_lang} -> {target_lang}, {os.path.basename(out_srt)}")
    return None


def is_whisper_translatable(target_lang: str | None) -> bool:
    """Whether Whisper's own translate task can produce this target.

    It only ever emits English -- the task is 'translate to English', not
    'translate to X' -- so every other target still goes through argos.
    Accepts the regional forms too ('en-US', 'en-GB'): they differ from
    'en' in spelling conventions Whisper doesn't promise anyway, and
    refusing them would silently route an English request to the weaker
    engine."""
    return (target_lang or "").split("-")[0].strip().lower() == "en"


def _warn_translate_failed(job_id: int, engine: str, err: dict) -> None:
    """Records a failed translation without failing the job.

    Translation is an optional extra bolted onto the end of a transcript
    that already succeeded. Failing the whole stage threw away five minutes
    of correct whisper output -- transcript written to disk, then discarded
    unmuxed with the job marked 'error' -- because an add-on couldn't reach
    its model directory. The transcript is the thing that was asked for;
    ship it, and say clearly what didn't happen on top of it.

    'Translate subs...' on the finished job retries just the translation
    once the cause is fixed, with no re-transcription."""
    msg = f"WARNING: translation skipped, transcript kept -- {err.get('error')}"
    append_job_log(job_id, msg)


def write_track_meta(srt_path: str, meta: dict) -> None:
    """Writes '<srt>.json' beside a generated subtitle file recording how it
    was produced.

    The job row only ever holds the CURRENT settings, so a regen overwrites
    the record of what made the file before it -- which makes comparing two
    runs impossible exactly when you're trying to decide between them. The
    sidecar travels with the .srt instead, and survives regeneration
    because the filename carries the engine and model too.

    Best-effort: a metadata write must never fail a job whose transcript
    already succeeded."""
    try:
        with open(srt_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"written": _timestamp(), **meta}, f, ensure_ascii=False, indent=2)
    except OSError as e:
        log_line(f"[track meta] could not write {os.path.basename(srt_path)}.json: {e}")


def whisper_track_slug(job: dict, task: str = "transcribe") -> str:
    """Filename/track segment identifying the engine AND model, e.g.
    'whisper-medium'. Including the model is what lets a small run and a
    medium run of the same video coexist -- with a bare '.whisper.srt' the
    second overwrites the first, so the two can never be compared, which is
    the entire reason for choosing a model per job."""
    return f"whisper-{whisper_model_for(job)}"


def run_transcribe(job: dict, current_filepath: str, orig_base: str) -> dict:
    """Blocking. Runs whichever of whisper/ocr job['gen_subs'] asks for
    ('whisper'|'ocr'|'both') and returns the (path, iso-lang, title) tuples
    to hand to collect_sub_tracks/run_mux alongside downloaded/merged
    tracks. Sets _job_thread so the ffmpeg/tesseract subprocesses spawned
    below are cancelable the same way run_separation's demucs call is.

    Phase 8: if job['translate_to'] is set (main.py's _create_job already
    clamps it to None when gen_subs_lang isn't -- argos-translate can't
    guess the source language), each generated track above is additionally
    translated and appended as one more track, original kept alongside it.
    Checked between tracks (job_id in CANCELED), not mid-translation --
    translating a short subtitle file's text is fast, unlike the Whisper
    segment loop's need for finer-grained cancellation."""
    job_id = job["id"]
    _job_thread[job_id] = threading.get_ident()
    lang_hint = job.get("gen_subs_lang")
    iso = iso639_2(lang_hint) if lang_hint else "und"
    want = job.get("gen_subs") or "off"
    target_lang = (job.get("translate_to") or "").strip() or None
    tracks: list[tuple[str, str, str]] = []
    try:
        if want in ("whisper", "both"):
            slug = whisper_track_slug(job)
            out_srt = f"{orig_base}.{slug}.srt"
            result = run_whisper_transcribe(job, current_filepath, out_srt)
            if result["status"] != "done":
                return result
            write_track_meta(out_srt, {
                "engine": "whisper", "task": "transcribe",
                "model": whisper_model_for(job), "lang_hint": lang_hint,
                "build": os.environ.get("GIT_SHA", "unknown")[:12],
                "cues": result.get("cues"), "dropped_hallucinations": result.get("dropped"),
            })
            tracks.append((out_srt, iso, slug if lang_hint else f"{slug} (auto)"))
            if target_lang and lang_hint:
                if job_id in CANCELED:
                    return {"status": "canceled", "error": "canceled by user"}
                if is_whisper_translatable(target_lang):
                    # Second decode of the same audio, not a post-process of
                    # the transcript above -- which is the whole point. See
                    # run_whisper_transcribe's docstring.
                    en_srt = f"{orig_base}.{slug}.{target_lang}.srt"
                    result = run_whisper_transcribe(job, current_filepath, en_srt, task="translate")
                    if result["status"] == "canceled":
                        return result
                    if result["status"] != "done":
                        _warn_translate_failed(job_id, "whisper", result)
                    else:
                        write_track_meta(en_srt, {
                            "engine": "whisper", "task": "translate",
                            "model": whisper_model_for(job), "lang_hint": lang_hint,
                            "target_lang": target_lang,
                            "build": os.environ.get("GIT_SHA", "unknown")[:12],
                            "cues": result.get("cues"), "dropped_hallucinations": result.get("dropped"),
                        })
                        tracks.append((en_srt, iso639_2(target_lang), f"{slug} -> {target_lang}"))
                else:
                    err = _add_translated_track(job, orig_base, out_srt, slug, lang_hint, target_lang, tracks)
                    if err is not None:
                        if err.get("status") == "canceled":
                            return err
                        _warn_translate_failed(job_id, "whisper", err)

        # kind='audio' has no frames -- main.py already clamps ocr/both down
        # to whisper at job-creation time, so this only guards a stale row
        # from before that clamp existed (or a hand-edited DB row).
        if want in ("ocr", "both") and job["kind"] == "video":
            out_srt = f"{orig_base}.ocr.srt"
            result = run_ocr_transcribe(job, current_filepath, out_srt)
            if result["status"] != "done":
                return result
            write_track_meta(out_srt, {
                "engine": "ocr", "lang": ocr_lang_for(lang_hint, job.get("ocr_lang")),
                "region": job.get("ocr_region") or "bottom",
                "sample_fps": OCR_SAMPLE_FPS, "binarize": OCR_BINARIZE,
                "crop_bottom_pct": OCR_CROP_BOTTOM_PCT,
                "build": os.environ.get("GIT_SHA", "unknown")[:12],
            })
            tracks.append((out_srt, iso, "ocr" if lang_hint else "ocr (auto)"))
            if target_lang and lang_hint:
                if job_id in CANCELED:
                    return {"status": "canceled", "error": "canceled by user"}
                err = _add_translated_track(job, orig_base, out_srt, "ocr", lang_hint, target_lang, tracks)
                if err is not None:
                    if err.get("status") == "canceled":
                        return err
                    _warn_translate_failed(job_id, "ocr", err)

        if not tracks:
            return {"status": "error", "error": "no subtitle tracks generated"}
        return {"status": "done", "tracks": tracks}
    finally:
        _job_thread.pop(job_id, None)
        _thread_proc.pop(threading.get_ident(), None)
        CANCELED.discard(job_id)


async def _run_transcribe_stage(job: dict, orig_base: str, current_filepath: str) -> list[tuple[str, str, str]] | None:
    """Runs whisper/OCR under TRANSCRIBE_SEM and writes status='transcribing'
    to the DB while it runs. Returns the (path, iso-lang, title) tuples
    generated, or None if the job ended in error/canceled (already written
    to the DB here, same shape as _run_separation_stage)."""
    job_id = job["id"]
    db.update_job(job_id, status="transcribing", progress=0)
    await asyncio.to_thread(
        append_job_log, job_id,
        f"stage: transcribing (engine={job.get('gen_subs')!r} lang_hint={job.get('gen_subs_lang')!r}"
        f" translate_to={job.get('translate_to')!r}) {build_stamp()}",
    )
    async with TRANSCRIBE_SEM:
        result = await asyncio.to_thread(run_transcribe, job, current_filepath, orig_base)
    if result.get("status") in ("error", "canceled"):
        db.update_job(job_id, status=result["status"], error=result.get("error"))
        await asyncio.to_thread(append_job_log, job_id, f"stage: transcribing {result['status']}: {result.get('error')}")
        CANCELED.discard(job_id)
        return None
    names = ", ".join(os.path.basename(p) for p, _lang, _title in result["tracks"])
    await asyncio.to_thread(append_job_log, job_id, f"stage: transcribing done -- {len(result['tracks'])} track(s): {names}")
    return result["tracks"]


async def _run_separation_stage(job: dict, orig_filepath: str) -> str | None:
    """Runs demucs under SEPARATION_SEM and writes the resulting status/
    filepath to the DB. Returns the novocals filepath on success, or None
    if the job ended in error/canceled (already written)."""
    job_id = job["id"]
    db.update_job(job_id, status="separating", progress=0)
    await asyncio.to_thread(append_job_log, job_id, f"stage: separating (model={DEMUCS_MODEL!r})")
    async with SEPARATION_SEM:
        sep_job = {**job, "filepath": orig_filepath}
        sep_result = await asyncio.to_thread(run_separation, sep_job)
    if sep_result.get("status") in ("error", "canceled"):
        db.update_job(job_id, **sep_result)
        await asyncio.to_thread(append_job_log, job_id, f"stage: separating {sep_result['status']}: {sep_result.get('error')}")
        CANCELED.discard(job_id)
        return None
    db.update_job(job_id, filepath=sep_result["filepath"], progress=100.0)
    await asyncio.to_thread(append_job_log, job_id, f"stage: separating done -> {os.path.basename(sep_result['filepath'])}")
    return sep_result["filepath"]


async def _run_subs_and_mux_stage(
    job: dict, orig_base: str, current_filepath: str, info: dict | None,
    generated_tracks: list[tuple[str, str, str]] | None = None,
) -> None:
    """Bilingual merge (6a) then muxing: audio jobs get subs muxed in
    manually since yt-dlp's embed postprocessor only works on video
    containers (6b); video jobs with a merged track get it added as one
    more subtitle stream alongside what yt-dlp already embedded (6a).
    generated_tracks (Phase 7 -- Whisper/OCR) is just more tuples in the
    same list, fed to the same collect_sub_tracks/run_mux call either way.

    orig_base is the pre-pipeline filename stem (no extension) -- callers
    compute it once: os.path.splitext(orig_filepath)[0] for a job running
    fresh, resolve_orig_base(job) when resuming after a restart (the job's
    filepath may already be post-separation by the time it's resumed)."""
    job_id = job["id"]
    generated_tracks = generated_tracks or []
    merged_srt = None
    if job.get("subs") and job.get("merge_subs"):
        logger = JobLogger(job_id)
        merged_srt = await asyncio.to_thread(merge_bilingual_subs, job, orig_base, info, logger)
        if logger.lines:
            await asyncio.to_thread(append_job_log, job_id, logger.dump(), False)

    if job["kind"] == "audio":
        sub_tracks = []
        if job.get("subs") and job.get("embed_subs"):
            sub_tracks = collect_sub_tracks(job, orig_base, merged_srt)
        sub_tracks = sub_tracks + generated_tracks
        if sub_tracks:
            db.update_job(job_id, status="muxing")
            stem = os.path.splitext(current_filepath)[0]
            out_path = stem + ".mkv"
            titles = ", ".join(t for _p, _l, t in sub_tracks)
            await asyncio.to_thread(append_job_log, job_id, f"stage: muxing {len(sub_tracks)} subtitle track(s) [{titles}] -> {os.path.basename(out_path)}")
            await asyncio.to_thread(run_mux, current_filepath, sub_tracks, out_path, False)
            db.update_job(job_id, status="done", filepath=out_path, progress=100.0)
            await asyncio.to_thread(
                lambda: append_job_log(job_id, f"output: {os.path.basename(out_path)} -- {describe_media(out_path)}"),
            )
            await asyncio.to_thread(append_job_log, job_id, "stage: done")
            return
    elif job["kind"] == "video":
        tracks = []
        if merged_srt:
            title = f"{job['sub_primary']}+{job['sub_secondary']}"
            tracks.append((merged_srt, iso639_2(job["sub_primary"]), title))
        tracks += generated_tracks
        if tracks:
            db.update_job(job_id, status="muxing")
            titles = ", ".join(t for _p, _l, t in tracks)
            await asyncio.to_thread(append_job_log, job_id, f"stage: muxing {len(tracks)} subtitle track(s) [{titles}] -> {os.path.basename(current_filepath)}")
            await asyncio.to_thread(run_mux, current_filepath, tracks, current_filepath, True)
            db.update_job(job_id, status="done", progress=100.0)
            await asyncio.to_thread(
                lambda: append_job_log(job_id, f"output: {os.path.basename(current_filepath)} -- {describe_media(current_filepath)}"),
            )
            await asyncio.to_thread(append_job_log, job_id, "stage: done")
            return

    db.update_job(job_id, status="done", progress=100.0)
    if current_filepath:
        await asyncio.to_thread(
            lambda: append_job_log(job_id, f"output: {os.path.basename(current_filepath)} -- {describe_media(current_filepath)}"),
        )
    await asyncio.to_thread(append_job_log, job_id, "stage: done (nothing to mux)")


def fail_job(job_id: int, e: BaseException) -> None:
    """Terminal failure path for an unexpected exception in a background
    job task: get it onto the row, and onto the job's own log, so it
    reaches the UI instead of only `docker logs`.

    These coroutines are launched with a bare asyncio.create_task and
    nobody awaits the result, so an escaping exception surfaces as
    "Task exception was never retrieved" on stderr and NOTHING else -- the
    row keeps whatever active status it had and the job appears frozen in
    the table forever. _dispatch has always done this for queued jobs; the
    resume paths are the gap this closes."""
    msg = f"{type(e).__name__}: {e}"
    log_line(f"[job {job_id}] ERROR: {msg}")
    try:
        db.update_job(job_id, status="error", error=msg)
        append_job_log(job_id, f"ERROR: {msg}", echo=False)
    except Exception:  # noqa: BLE001 -- a DB write failing here must not mask the original error
        log_line(f"[job {job_id}] ERROR: additionally failed to record the error above")


async def resume_separation(job: dict) -> None:
    """Startup recovery for a 'separating' row whose source file still
    exists on disk: demucs isn't resumable mid-run, so this restarts step 2
    from scratch and continues through the same merge/mux stages a fresh
    job would hit -- it does not re-download.
    # ponytail: the yt-dlp info dict (human vs auto-generated subtitle
    # provenance) doesn't survive a restart, so a merge_subs job resumed
    # here skips rollup-caption dedup rather than guessing at it. Delete
    # and retry from scratch if that matters for a given job.
    """
    try:
        orig_filepath = job["filepath"]
        orig_base = os.path.splitext(orig_filepath)[0]
        current_filepath = await _run_separation_stage(job, orig_filepath)
        if current_filepath is None:
            return
        generated_tracks = None
        if job.get("gen_subs") and job["gen_subs"] not in (None, "off"):
            generated_tracks = await _run_transcribe_stage(job, orig_base, current_filepath)
            if generated_tracks is None:
                return
        await _run_subs_and_mux_stage(job, orig_base, current_filepath, None, generated_tracks)
    except Exception as e:
        fail_job(job["id"], e)


async def resume_transcribe(job: dict, mux: bool = True) -> None:
    """Startup recovery for a 'transcribing' row whose source file still
    exists on disk: like demucs, Whisper/OCR aren't meaningfully resumable
    mid-run, so this restarts the stage from scratch and continues through
    the same mux stage a fresh job would hit -- it does not re-download or
    re-run separation (job['filepath'] is already whatever separation left
    behind, if strip_vocals ran). resolve_orig_base recovers the true
    pre-pipeline stem regardless of whether separation already ran, the
    same way resume_separation's caller in main.py resolves it for a job
    stuck earlier in the pipeline.
    mux=False writes the .srt sidecars and stops there, leaving the video
    file byte-for-byte untouched. Muxing rewrites the media in place, so
    without this there is no way to try a different model or language on a
    finished job without also rewriting a file that was already correct --
    and the sidecars are individually downloadable and picked up by the
    player either way, so embedding is a convenience, not a requirement.

    # ponytail: the yt-dlp info dict doesn't survive a restart here either,
    # same as resume_separation -- a resumed merge_subs job skips rollup
    # dedup, see that docstring for why.
    """
    try:
        orig_base = resolve_orig_base(job)
        current_filepath = job["filepath"]
        generated_tracks = await _run_transcribe_stage(job, orig_base, current_filepath)
        if generated_tracks is None:
            return
        if not mux:
            names = ", ".join(os.path.basename(p) for p, _l, _t in generated_tracks)
            db.update_job(job["id"], status="done", progress=100.0, stage=None)
            await asyncio.to_thread(
                append_job_log, job["id"],
                f"stage: done -- wrote {names} as sidecar(s), video left untouched",
            )
            return
        await _run_subs_and_mux_stage(job, orig_base, current_filepath, None, generated_tracks)
    except Exception as e:
        fail_job(job["id"], e)


async def translate_existing_subs(job: dict, src_srt: str, source_lang: str, target_lang: str) -> None:
    """Translates one .srt already on disk and muxes the result in as an
    extra track. Distinct from the translate_to option on generation, which
    can only translate what it just produced -- having a good zh transcript
    and wanting en out of it shouldn't mean re-running whisper for five
    minutes to reach the translate step attached to the end of it.

    Reuses the same mux stage the transcribe pipeline ends with, so the
    output naming, track titles and in-place video rewrite are identical to
    a translate produced the original way."""
    job_id = job["id"]
    try:
        db.update_job(job_id, status="transcribing", progress=0, stage="translating", error=None)
        append_job_log(job_id, f"stage: translating {os.path.basename(src_srt)} {source_lang} -> {target_lang} {build_stamp()}")
        orig_base = resolve_orig_base(job)
        # Name it off the source track so translating '<stem>.whisper.srt'
        # gives '<stem>.whisper.en.srt' -- the same shape _add_translated_track
        # produces inline, rather than a second naming scheme for the same
        # kind of file.
        stem = os.path.basename(src_srt)[len(os.path.basename(orig_base)):].lstrip(".")[: -len(".srt")]
        engine = stem or "sub"
        out_srt = f"{orig_base}.{engine}.{target_lang}.srt"
        result = await asyncio.to_thread(
            run_translate, job, src_srt, source_lang, target_lang, out_srt,
        )
        if result["status"] != "done":
            db.update_job(job_id, status="error", error=result.get("error"))
            append_job_log(job_id, f"stage: translating failed: {result.get('error')}", echo=False)
            return
        append_job_log(job_id, f"translate: {os.path.basename(out_srt)}")
        tracks = [(out_srt, iso639_2(target_lang), f"{engine} -> {target_lang}")]
        await _run_subs_and_mux_stage(job, orig_base, job["filepath"], None, tracks)
    except Exception as e:
        fail_job(job_id, e)


# -------------------------------------------------------------- queue loop

async def _dispatch(job: dict) -> None:
    try:
        if job["kind"] == "playlist":
            db.update_job(job["id"], status="expanding")
            await asyncio.to_thread(expand_playlist, job)
        else:
            await _process_job(job)
    except Exception as e:  # never let one bad job kill the loop
        fail_job(job["id"], e)


async def _process_job(job: dict) -> None:
    """Download, then chain whatever Phase 5/6 post-processing the job asked
    for: vocal separation, bilingual subtitle merge, subtitle muxing."""
    job_id = job["id"]
    result = await asyncio.to_thread(run_download, job)
    info = result.pop("_info", None)
    db.update_job(job_id, **result)
    if result["status"] != "done":
        return
    job = {**job, **result}
    await _continue_after_download(job, info)


async def _continue_after_download(job: dict, info: dict | None) -> None:
    job_id = job["id"]
    orig_filepath = job["filepath"]
    orig_base = os.path.splitext(orig_filepath)[0]
    current_filepath = orig_filepath
    try:
        if job["kind"] == "audio" and job.get("strip_vocals"):
            current_filepath = await _run_separation_stage(job, orig_filepath)
            if current_filepath is None:
                return  # error/canceled already written by _run_separation_stage
        generated_tracks = None
        if job.get("gen_subs") and job["gen_subs"] not in (None, "off"):
            generated_tracks = await _run_transcribe_stage(job, orig_base, current_filepath)
            if generated_tracks is None:
                return  # error/canceled already written by _run_transcribe_stage
        await _run_subs_and_mux_stage(job, orig_base, current_filepath, info, generated_tracks)
    except Exception as e:
        fail_job(job_id, e)


async def queue_loop(stop_event: asyncio.Event) -> None:
    """One asyncio loop task started in FastAPI lifespan, dispatching under
    DOWNLOAD_SEM (MAX_CONCURRENT). The semaphore is acquired before the claim
    so at most MAX_CONCURRENT rows are ever marked 'running' at once. Note
    this only gates the download step -- Phase 5 separation runs under its
    own SEPARATION_SEM, acquired later and independently in _process_job."""
    sem = DOWNLOAD_SEM

    async def _run_one(job):
        try:
            await _dispatch(job)
        finally:
            sem.release()

    while not stop_event.is_set():
        await sem.acquire()
        job = await asyncio.to_thread(db.claim_next_job)
        if job is None:
            sem.release()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            continue
        asyncio.create_task(_run_one(job))
