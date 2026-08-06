"""asyncio queue loop, yt-dlp calls, playlist expansion.

yt-dlp is used as a library (not a subprocess) so progress, format probing,
and playlist enumeration go through a real API instead of stdout regex.
"""
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
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
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
OCR_SAMPLE_FPS = float(os.environ.get("OCR_SAMPLE_FPS", "2"))
OCR_CROP_BOTTOM_PCT = float(os.environ.get("OCR_CROP_BOTTOM_PCT", "0.22"))
# Tesseract's '+'-joined multi-language syntax. Used when the job has no
# gen_subs_lang hint, or the hint doesn't match one of the small set of
# mappings in ocr_lang_for() below.
OCR_LANG = os.environ.get("OCR_LANG", "chi_sim+eng")

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
    almost always VP9/AV1 in webm, so an unfiltered `bv*+ba/b` selector
    lands on webm by default regardless of what the merge step is asked to
    produce. mp4/mkv here both prefer mp4+m4a (h264/aac) source streams
    first -- those remux cleanly into either container with no re-encode --
    and only fall through to an unrestricted `bv*+ba/b` (which may well be
    webm) when nothing mp4-compatible exists at the requested quality.
    Choosing container='webm' explicitly skips that preference and takes
    the highest-quality stream outright, same as the old unconditional
    behavior."""
    if kind == "audio":
        return "ba/b"
    if quality and quality.startswith("fmt:"):
        return quality.split(":", 1)[1]
    cap = _HEIGHT_CAP.get(quality)
    h = f"[height<={cap}]" if cap else ""
    if container == "webm":
        return f"bv*{h}+ba/b{h}"
    return f"bv*{h}[ext=mp4]+ba[ext=m4a]/b{h}[ext=mp4]/bv*{h}+ba/b{h}"


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

    def debug(self, msg):
        self.lines.append(msg)

    def info(self, msg):
        self.lines.append(msg)

    def warning(self, msg):
        self.lines.append(f"WARNING: {msg}")
        print(f"{self._prefix()}WARNING: {msg}", flush=True)

    def error(self, msg):
        self.lines.append(f"ERROR: {msg}")
        print(f"{self._prefix()}ERROR: {msg}", flush=True)

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
        outtmpl = os.path.join(
            DOWNLOADS_DIR,
            "%(playlist_title).100B",
            "%(playlist_index)03d - %(title).150B [%(id)s].%(ext)s",
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
        print(f"[job {job_id}] ERROR: {type(e).__name__}: {e}", flush=True)
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
        print(f"[job {job_id}] ERROR: {type(e).__name__}: {e}", flush=True)
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

def safe_delete_file(filepath: str, downloads_dir: str = DOWNLOADS_DIR) -> list[str]:
    """os.path.realpath the stored path and assert it's under downloads_dir
    before unlinking anything -- the path comes out of the DB rather than
    the request, so this is cheap insurance against a corrupted row, not a
    live hole. Also removes .part/.srt/.vtt siblings."""
    if not filepath:
        return []
    real = os.path.realpath(filepath)
    root = os.path.realpath(downloads_dir)
    if real != root and not real.startswith(root + os.sep):
        raise ValueError(f"refusing to delete outside downloads dir: {real}")

    removed = []
    base, _ext = os.path.splitext(real)
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
        print(f"[job {job_id}] ERROR: {msg}", flush=True)
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
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
            return {"status": "error", "error": msg}

        # Demucs writes nested at {tmpdir}/{model}/{track_stem}/no_vocals.wav,
        # not flat in tmpdir.
        stem = os.path.splitext(os.path.basename(src))[0]
        novocals_wav = os.path.join(tmpdir, DEMUCS_MODEL, stem, "no_vocals.wav")
        if not os.path.exists(novocals_wav):
            msg = f"demucs output not found: {novocals_wav}"
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
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
        print(f"[job {job_id}] ERROR: ffmpeg encode failed: {e}", flush=True)
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
    only the audio stream (audio-only outputs, Phase 6b)."""
    cmd = ["ffmpeg", "-y", "-i", src_path]
    for path, _lang, _title in sub_tracks:
        cmd += ["-i", path]
    cmd += ["-map", "0"] if copy_all else ["-map", "0:a"]
    for i in range(1, len(sub_tracks) + 1):
        cmd += ["-map", str(i)]
    cmd += ["-c", "copy"] if copy_all else ["-c:a", "copy"]
    cmd += ["-c:s", "srt"]
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
    tmp_out = out_path + ".tmp.mkv" if same_file else out_path
    cmd = build_mux_cmd(src_path, sub_tracks, tmp_out, copy_all)
    subprocess.run(cmd, check=True, capture_output=True)
    if same_file:
        os.replace(tmp_out, out_path)


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

_OCR_LANG_HINTS = {"zh": "chi_sim+eng", "en": "eng"}


def ocr_lang_for(gen_subs_lang: str | None) -> str:
    """Small, deliberately partial mapping from a job's language hint to a
    Tesseract '+'-joined lang string -- not a general-purpose table for
    every possible language, just the realistic cases. Falls back to
    OCR_LANG for anything else."""
    if not gen_subs_lang:
        return OCR_LANG
    return _OCR_LANG_HINTS.get(gen_subs_lang.split("-")[0].lower(), OCR_LANG)


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


def ocr_frames_to_cues(frame_texts: list[tuple[float, str]], frame_interval: float) -> list[tuple[float, float, str]]:
    """Pure merge logic, unit-tested with synthetic input -- no real
    tesseract/ffmpeg involved. Normalization is strip + collapse internal
    whitespace only (not punctuation -- OCR punctuation noise between
    otherwise-identical frames is a known source of spurious cue splits;
    # ponytail: not normalized further until real usage shows it matters,
    # a fuzzy/edit-distance match is the upgrade path).

    frame_texts: (timestamp_seconds, raw_ocr_text) for every sampled frame,
    in timestamp order. Blank (whitespace-only) frames close and skip any
    open cue -- that's "no subtitle showing". Consecutive frames whose
    normalized text matches extend the current cue through to that frame's
    timestamp + one frame_interval (covering the gap to the next sample);
    a text change closes the current cue and opens a new one."""
    def norm(s: str) -> str:
        return " ".join(s.split())

    cues: list[tuple[float, float, str]] = []
    cur_text: str | None = None
    cur_start = cur_end = 0.0
    for ts, raw in frame_texts:
        t = norm(raw)
        if not t:
            if cur_text is not None:
                cues.append((cur_start, cur_end + frame_interval, cur_text))
                cur_text = None
            continue
        if t == cur_text:
            cur_end = ts
            continue
        if cur_text is not None:
            cues.append((cur_start, cur_end + frame_interval, cur_text))
        cur_text, cur_start, cur_end = t, ts, ts
    if cur_text is not None:
        cues.append((cur_start, cur_end + frame_interval, cur_text))
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


def run_whisper_transcribe(job: dict, current_filepath: str, out_srt: str) -> dict:
    """Blocking. kind='video' jobs get their audio extracted to a temp mono
    16kHz wav first; kind='audio' jobs transcribe current_filepath directly
    (whatever file is current post-separation). Caller
    (worker._run_transcribe_stage) runs this under TRANSCRIBE_SEM."""
    job_id = job["id"]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        msg = "speech-to-text not enabled -- rebuild with WITH_TRANSCRIBE=true"
        print(f"[job {job_id}] ERROR: {msg}", flush=True)
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

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_src, language=job.get("gen_subs_lang") or None)
        total = info.duration or 0

        cues = []
        last_flush = 0.0
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

        if not cues:
            msg = "whisper produced no cues (silent audio, or an unsupported/mismatched language hint)"
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
            return {"status": "error", "error": msg}
        write_srt(cues, out_srt)
        return {"status": "done", "path": out_srt}
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg audio extraction failed: {e}"
        print(f"[job {job_id}] ERROR: {msg}", flush=True)
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
        print(f"[job {job_id}] ERROR: {msg}", flush=True)
        return {"status": "error", "error": msg}

    import pytesseract
    from PIL import Image  # Pillow: pytesseract's own dependency, not ours

    fps = OCR_SAMPLE_FPS
    lang = ocr_lang_for(job.get("gen_subs_lang"))
    tmpdir = tempfile.mkdtemp(prefix=f"ocr-{job_id}-")
    try:
        pct = OCR_CROP_BOTTOM_PCT
        vf = f"fps={fps},crop=iw:ih*{pct}:0:ih*(1-{pct})"
        cmd = ["ffmpeg", "-y", "-i", video_src, "-vf", vf, "-q:v", "2", os.path.join(tmpdir, "%06d.png")]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _thread_proc[threading.get_ident()] = proc
        _, stderr = proc.communicate()
        if job_id in CANCELED:
            return {"status": "canceled", "error": "canceled by user"}
        if proc.returncode != 0:
            msg = f"ffmpeg frame extraction failed: {stderr.decode(errors='replace')[-500:]}"
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
            return {"status": "error", "error": msg}

        frames = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        if not frames:
            msg = "ocr: ffmpeg extracted no frames"
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
            return {"status": "error", "error": msg}

        frame_interval = 1.0 / fps
        frame_texts = []
        last_flush = 0.0
        for i, name in enumerate(frames):
            if job_id in CANCELED:
                return {"status": "canceled", "error": "canceled by user"}
            text = pytesseract.image_to_string(Image.open(os.path.join(tmpdir, name)), lang=lang)
            frame_texts.append((i * frame_interval, text))
            now = time.monotonic()
            if now - last_flush >= 1.0:
                db.update_job(job_id, progress=round((i + 1) / len(frames) * 100.0, 1), stage="transcribing (ocr)")
                last_flush = now

        cues = ocr_frames_to_cues(frame_texts, frame_interval)
        if not cues:
            msg = "ocr produced no cues (no burned-in subtitle text detected)"
            print(f"[job {job_id}] ERROR: {msg}", flush=True)
            return {"status": "error", "error": msg}
        write_srt(cues, out_srt)
        return {"status": "done", "path": out_srt}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_transcribe(job: dict, current_filepath: str, orig_base: str) -> dict:
    """Blocking. Runs whichever of whisper/ocr job['gen_subs'] asks for
    ('whisper'|'ocr'|'both') and returns the (path, iso-lang, title) tuples
    to hand to collect_sub_tracks/run_mux alongside downloaded/merged
    tracks. Sets _job_thread so the ffmpeg/tesseract subprocesses spawned
    below are cancelable the same way run_separation's demucs call is."""
    job_id = job["id"]
    _job_thread[job_id] = threading.get_ident()
    lang_hint = job.get("gen_subs_lang")
    iso = iso639_2(lang_hint) if lang_hint else "und"
    want = job.get("gen_subs") or "off"
    tracks: list[tuple[str, str, str]] = []
    try:
        if want in ("whisper", "both"):
            out_srt = f"{orig_base}.whisper.srt"
            result = run_whisper_transcribe(job, current_filepath, out_srt)
            if result["status"] != "done":
                return result
            tracks.append((out_srt, iso, "whisper" if lang_hint else "whisper (auto)"))

        # kind='audio' has no frames -- main.py already clamps ocr/both down
        # to whisper at job-creation time, so this only guards a stale row
        # from before that clamp existed (or a hand-edited DB row).
        if want in ("ocr", "both") and job["kind"] == "video":
            out_srt = f"{orig_base}.ocr.srt"
            result = run_ocr_transcribe(job, current_filepath, out_srt)
            if result["status"] != "done":
                return result
            tracks.append((out_srt, iso, "ocr" if lang_hint else "ocr (auto)"))

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
    async with TRANSCRIBE_SEM:
        result = await asyncio.to_thread(run_transcribe, job, current_filepath, orig_base)
    if result.get("status") in ("error", "canceled"):
        db.update_job(job_id, status=result["status"], error=result.get("error"))
        CANCELED.discard(job_id)
        return None
    return result["tracks"]


async def _run_separation_stage(job: dict, orig_filepath: str) -> str | None:
    """Runs demucs under SEPARATION_SEM and writes the resulting status/
    filepath to the DB. Returns the novocals filepath on success, or None
    if the job ended in error/canceled (already written)."""
    job_id = job["id"]
    db.update_job(job_id, status="separating", progress=0)
    async with SEPARATION_SEM:
        sep_job = {**job, "filepath": orig_filepath}
        sep_result = await asyncio.to_thread(run_separation, sep_job)
    if sep_result.get("status") in ("error", "canceled"):
        db.update_job(job_id, **sep_result)
        CANCELED.discard(job_id)
        return None
    db.update_job(job_id, filepath=sep_result["filepath"], progress=100.0)
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
            prior = job.get("log") or ""
            db.update_job(job_id, log=(prior + "\n" + logger.dump())[-8192:])

    if job["kind"] == "audio":
        sub_tracks = []
        if job.get("subs") and job.get("embed_subs"):
            sub_tracks = collect_sub_tracks(job, orig_base, merged_srt)
        sub_tracks = sub_tracks + generated_tracks
        if sub_tracks:
            db.update_job(job_id, status="muxing")
            stem = os.path.splitext(current_filepath)[0]
            out_path = stem + ".mkv"
            await asyncio.to_thread(run_mux, current_filepath, sub_tracks, out_path, False)
            db.update_job(job_id, status="done", filepath=out_path, progress=100.0)
            return
    elif job["kind"] == "video":
        tracks = []
        if merged_srt:
            title = f"{job['sub_primary']}+{job['sub_secondary']}"
            tracks.append((merged_srt, iso639_2(job["sub_primary"]), title))
        tracks += generated_tracks
        if tracks:
            db.update_job(job_id, status="muxing")
            await asyncio.to_thread(run_mux, current_filepath, tracks, current_filepath, True)
            db.update_job(job_id, status="done", progress=100.0)
            return

    db.update_job(job_id, status="done", progress=100.0)


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


async def resume_transcribe(job: dict) -> None:
    """Startup recovery for a 'transcribing' row whose source file still
    exists on disk: like demucs, Whisper/OCR aren't meaningfully resumable
    mid-run, so this restarts the stage from scratch and continues through
    the same mux stage a fresh job would hit -- it does not re-download or
    re-run separation (job['filepath'] is already whatever separation left
    behind, if strip_vocals ran). resolve_orig_base recovers the true
    pre-pipeline stem regardless of whether separation already ran, the
    same way resume_separation's caller in main.py resolves it for a job
    stuck earlier in the pipeline.
    # ponytail: the yt-dlp info dict doesn't survive a restart here either,
    # same as resume_separation -- a resumed merge_subs job skips rollup
    # dedup, see that docstring for why.
    """
    orig_base = resolve_orig_base(job)
    current_filepath = job["filepath"]
    generated_tracks = await _run_transcribe_stage(job, orig_base, current_filepath)
    if generated_tracks is None:
        return
    await _run_subs_and_mux_stage(job, orig_base, current_filepath, None, generated_tracks)


# -------------------------------------------------------------- queue loop

async def _dispatch(job: dict) -> None:
    try:
        if job["kind"] == "playlist":
            db.update_job(job["id"], status="expanding")
            await asyncio.to_thread(expand_playlist, job)
        else:
            await _process_job(job)
    except Exception as e:  # never let one bad job kill the loop
        print(f"[job {job['id']}] ERROR: {type(e).__name__}: {e}", flush=True)
        db.update_job(job["id"], status="error", error=f"{type(e).__name__}: {e}")


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
        print(f"[job {job_id}] ERROR: {type(e).__name__}: {e}", flush=True)
        db.update_job(job_id, status="error", error=f"{type(e).__name__}: {e}")


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
