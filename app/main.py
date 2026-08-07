"""FastAPI app: routes, SSE stream, startup recovery."""
import asyncio
import base64
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, worker

APP_PASSWORD = os.environ.get("APP_PASSWORD")
STATIC_DIR = Path(__file__).parent / "static"
TERMINAL_STATUSES = ("done", "error", "canceled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(worker.DOWNLOADS_DIR, exist_ok=True)
    worker.log_line(f"[startup] {worker.build_stamp()} log={worker.APP_LOG_PATH}")
    if worker._REPOINTED_HOME:
        worker.log_line(f"[startup] HOME was not writable, repointed to {worker._REPOINTED_HOME}")
    await asyncio.to_thread(db.init_db)
    n = await asyncio.to_thread(db.reset_stuck_jobs)
    if n:
        worker.log_line(f"[startup] requeued {n} interrupted job(s)")

    # 'separating' is deliberately excluded from reset_stuck_jobs (Phase 1):
    # a job interrupted there has a complete source download on disk, and
    # requeuing it from scratch throws that away. Resume in place if the
    # source is still there; only demote to queued if it's gone.
    sep_jobs = await asyncio.to_thread(db.get_jobs_by_status, "separating")
    for job in sep_jobs:
        if job.get("filepath") and os.path.exists(job["filepath"]):
            asyncio.create_task(worker.resume_separation(job))
        else:
            await asyncio.to_thread(db.update_job, job["id"], status="queued", filepath=None)
    if sep_jobs:
        worker.log_line(f"[startup] resumed/requeued {len(sep_jobs)} separating job(s)")

    # Phase 7: same resumable-in-place treatment for 'transcribing' -- a job
    # interrupted mid-Whisper/OCR has a complete source file on disk (post-
    # separation if strip_vocals ran), and neither engine is meaningfully
    # resumable mid-run anyway, so restart the stage from scratch rather
    # than requeue from the top.
    tr_jobs = await asyncio.to_thread(db.get_jobs_by_status, "transcribing")
    for job in tr_jobs:
        if job.get("filepath") and os.path.exists(job["filepath"]):
            asyncio.create_task(worker.resume_transcribe(job))
        else:
            await asyncio.to_thread(db.update_job, job["id"], status="queued", filepath=None)
    if tr_jobs:
        worker.log_line(f"[startup] resumed/requeued {len(tr_jobs)} transcribing job(s)")

    stop_event = asyncio.Event()
    task = asyncio.create_task(worker.queue_loop(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await task


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- auth
# Optional HTTP Basic auth, gated on APP_PASSWORD being set (unset by
# default). Middleware rather than a route Depends so it also covers the
# /files StaticFiles mount and the SSE stream, not just declared routes.
@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if APP_PASSWORD:
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})
    return await call_next(request)


# -------------------------------------------------------------- models

class AddJobRequest(BaseModel):
    url: str
    kind: str = "video"
    quality: str = "best"
    container: str = "mp4"           # mp4 | mkv | webm -- ignored for kind='audio'
    subs: str | None = None
    embed_subs: bool = True
    strip_vocals: bool = False       # Phase 5 -- no-op unless kind='audio'
    merge_subs: bool = False         # Phase 6a
    sub_primary: str | None = None   # e.g. 'zh' -- top line
    sub_secondary: str | None = None  # e.g. 'en' -- bottom line
    gen_subs: str = "off"            # Phase 7 -- off|whisper|ocr|both
    gen_subs_lang: str | None = None  # e.g. 'zh' -- hint for both engines
    translate_to: str | None = None  # Phase 8 -- e.g. 'en'; needs gen_subs_lang set
    ocr_lang: str | None = None      # verbatim tesseract lang string; overrides the gen_subs_lang mapping
    ocr_region: str = "bottom"       # bottom | full
    whisper_model: str | None = None # tiny|base|small|medium|large-v3; None = container default


class BulkJobRequest(BaseModel):
    text: str
    kind: str = "video"
    quality: str = "best"
    container: str = "mp4"
    subs: str | None = None
    embed_subs: bool = True
    strip_vocals: bool = False
    merge_subs: bool = False
    sub_primary: str | None = None
    sub_secondary: str | None = None
    gen_subs: str = "off"
    gen_subs_lang: str | None = None
    translate_to: str | None = None
    ocr_lang: str | None = None
    ocr_region: str = "bottom"
    whisper_model: str | None = None
    force: bool = False


class DeleteJobsRequest(BaseModel):
    ids: list[int]
    delete_files: bool = False


class ProbeRequest(BaseModel):
    url: str


class RegenSubsRequest(BaseModel):
    # None means "keep whatever's already stored on the job" -- lets the
    # button just re-run with the current settings (e.g. after the
    # OCR_LANG fix) without the UI needing to resend every field.
    gen_subs: str | None = None
    gen_subs_lang: str | None = None
    translate_to: str | None = None
    ocr_lang: str | None = None
    ocr_region: str | None = None
    whisper_model: str | None = None


class TranslateSubsRequest(BaseModel):
    srt_name: str                    # bare filename of one of this job's own .srt sidecars
    source_lang: str                 # e.g. 'zh' -- argos has no detection, so both are required
    target_lang: str                 # e.g. 'en'


# ------------------------------------------------------------- helpers

def _clamp_gen_subs(
    kind: str, gen_subs: str, gen_subs_lang: str | None, translate_to: str | None,
) -> tuple[str, str | None]:
    """Shared by _create_job and /regen-subs so the two never drift apart.
    gen_subs='ocr'/'both' needs video frames that don't exist for an
    audio-only job -- clamped down to 'whisper'. translate_to (Phase 8)
    needs a known source language: argos-translate has no language
    detection of its own, so it rides on gen_subs_lang as that hint. No
    gen_subs_lang, or nothing being generated in the first place
    (gen_subs='off'), just turns translation off rather than erroring --
    clamp-not-reject, same as container/strip_vocals below."""
    if gen_subs not in ("off", "whisper", "ocr", "both"):
        gen_subs = "off"
    if kind == "audio" and gen_subs in ("ocr", "both"):
        gen_subs = "whisper"
    translate_to = (translate_to or "").strip() or None
    if gen_subs == "off" or not gen_subs_lang:
        translate_to = None
    return gen_subs, translate_to


def _create_job(
    url: str, kind: str, quality: str, subs: str | None, embed_subs: bool,
    strip_vocals: bool = False, merge_subs: bool = False,
    sub_primary: str | None = None, sub_secondary: str | None = None,
    container: str = "mp4", gen_subs: str = "off", gen_subs_lang: str | None = None,
    translate_to: str | None = None, ocr_lang: str | None = None,
    ocr_region: str = "bottom", whisper_model: str | None = None,
) -> dict:
    """Playlist classification is explicit, done at insert time -- not
    guessed at download time. Phase 1's noplaylist=True stays on every
    single-video job; only a URL classified as a playlist up front becomes
    a kind='playlist' parent row that the worker expands.

    strip_vocals is a no-op on anything but kind='audio' -- clamp it here
    rather than trust the client, same reasoning as kind being the only
    source of truth for audio-vs-video. container gets the same treatment:
    an unrecognized value falls back to mp4 rather than reaching yt-dlp."""
    cls = worker.classify_url(url)
    strip_vocals = bool(strip_vocals) and kind == "audio"
    if container not in worker.CONTAINER_CHOICES:
        container = "mp4"
    gen_subs, translate_to = _clamp_gen_subs(kind, gen_subs, gen_subs_lang, translate_to)
    if ocr_region not in worker.OCR_REGIONS:
        ocr_region = "bottom"
    ocr_lang = (ocr_lang or "").strip() or None
    # Unknown model names are dropped to None (= container default) rather
    # than passed through: faster-whisper would treat an unrecognized string
    # as a Hugging Face repo id and download it.
    if whisper_model not in worker.WHISPER_MODELS:
        whisper_model = None
    common = dict(
        subs=subs, embed_subs=int(embed_subs), strip_vocals=int(strip_vocals),
        merge_subs=int(bool(merge_subs)), sub_primary=sub_primary, sub_secondary=sub_secondary,
        container=container, gen_subs=gen_subs, gen_subs_lang=gen_subs_lang,
        translate_to=translate_to, ocr_lang=ocr_lang, ocr_region=ocr_region,
        whisper_model=whisper_model,
    )
    if cls == "playlist":
        job_id = db.insert_job(
            url=url, kind="playlist", quality=quality, child_kind=kind, status="queued", **common,
        )
    else:
        job_id = db.insert_job(url=url, kind=kind, quality=quality, status="queued", **common)
    return db.get_job(job_id)


# -------------------------------------------------------------- routes

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/jobs")
async def api_list_jobs():
    return await asyncio.to_thread(db.get_jobs)


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: int):
    """Single-job fetch with the full row, `log` included -- the SSE tick
    deliberately strips `log` to keep the once-per-second payload small
    (see /api/events), so this is what the UI's Log modal polls while a
    job is still actively producing log output, not just after it's done."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}/files")
async def api_job_files(job_id: int):
    """Every artifact on disk for this job -- the main output plus every
    sidecar .srt -- individually linked. 'Open'/fileUrl() only ever
    surfaced the main muxed file; subtitle sidecars had no UI path to them
    at all before this."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    paths = await asyncio.to_thread(worker.list_job_files, job)
    files = []
    for p in paths:
        rel = os.path.relpath(p, worker.DOWNLOADS_DIR)
        files.append({"name": os.path.basename(p), "url": "/files/" + rel.replace(os.sep, "/")})
    return {"files": files}


@app.get("/api/jobs/{job_id}/vtt/{filename}")
async def api_job_subtitle_vtt(job_id: int, filename: str):
    """SRT->WebVTT conversion for the in-browser player's <track> elements
    -- browsers only accept WebVTT there, not SRT, which is what every
    subtitle file this app has is stored as. filename is matched against
    this job's own list_job_files() output (the same safe boundary
    /api/jobs/{id}/files already uses), not resolved as an arbitrary path,
    so this can't be used to read anything outside what that job already
    exposes."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    paths = await asyncio.to_thread(worker.list_job_files, job)
    match = next((p for p in paths if os.path.basename(p) == filename), None)
    if not match or not match.endswith(".srt"):
        raise HTTPException(404, "subtitle file not found for this job")
    vtt_text = await asyncio.to_thread(worker.srt_to_vtt, match)
    return Response(content=vtt_text, media_type="text/vtt")


@app.post("/api/jobs/{job_id}/regen-subs")
async def api_regen_subs(job_id: int, req: RegenSubsRequest):
    """Re-runs Whisper/OCR/translate against the file already on disk --
    no re-download, no re-running the actual video/audio fetch. This is
    worker.resume_transcribe (originally built for restart recovery) fired
    manually instead of automatically; the source file being untouched is
    exactly the same precondition either way.

    Clean for kind='audio' every time -- that mux step only ever maps the
    audio stream (-map 0:a), so old subtitle tracks are dropped and fresh
    ones added, no accumulation. For kind='video' the mux step copies
    every existing stream (-map 0) and adds new tracks alongside them, so
    repeated regenerations *do* accumulate extra subtitle streams rather
    than replacing -- same tradeoff already accepted for the 6c re-upload
    path (see write_and_process's docstring in api_upload_subtitle).
    Fine for players that let you pick a track; delete + retry for a
    guaranteed single-track file."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] not in ("done", "error", "canceled"):
        raise HTTPException(400, "job is currently active, cancel it first")
    if not job.get("filepath") or not os.path.exists(job["filepath"]):
        raise HTTPException(400, "no downloaded file on disk for this job -- use Retry for a full re-download")

    gen_subs = req.gen_subs if req.gen_subs is not None else (job.get("gen_subs") or "off")
    gen_subs_lang = req.gen_subs_lang if req.gen_subs_lang is not None else job.get("gen_subs_lang")
    translate_to = req.translate_to if req.translate_to is not None else job.get("translate_to")
    ocr_lang = req.ocr_lang if req.ocr_lang is not None else job.get("ocr_lang")
    ocr_region = req.ocr_region if req.ocr_region is not None else job.get("ocr_region")
    whisper_model = req.whisper_model if req.whisper_model is not None else job.get("whisper_model")
    if ocr_region not in worker.OCR_REGIONS:
        ocr_region = "bottom"
    if whisper_model not in worker.WHISPER_MODELS:
        whisper_model = None
    ocr_lang = (ocr_lang or "").strip() or None
    gen_subs, translate_to = _clamp_gen_subs(job["kind"], gen_subs, gen_subs_lang, translate_to)
    if gen_subs == "off":
        raise HTTPException(400, "nothing to generate -- set gen_subs to whisper/ocr/both first")

    await asyncio.to_thread(
        db.update_job, job_id, gen_subs=gen_subs, gen_subs_lang=gen_subs_lang, translate_to=translate_to,
        ocr_lang=ocr_lang, ocr_region=ocr_region, whisper_model=whisper_model,
        status="transcribing", progress=0, error=None, stage=None,
    )
    updated_job = await asyncio.to_thread(db.get_job, job_id)
    asyncio.create_task(worker.resume_transcribe(updated_job))
    return updated_job


@app.post("/api/jobs/{job_id}/translate-subs")
async def api_translate_subs(job_id: int, req: TranslateSubsRequest):
    """Translates one .srt already on disk into another language and muxes
    it in, without re-running whisper/OCR. The translate_to option on
    generation can only translate what that same run produced, so getting
    en out of an existing good zh transcript otherwise meant transcribing
    the whole file again just to reach the translate step on the end."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] not in ("done", "error", "canceled"):
        raise HTTPException(400, "job is currently active, cancel it first")
    if not job.get("filepath") or not os.path.exists(job["filepath"]):
        raise HTTPException(400, "no downloaded file on disk for this job")

    source_lang = (req.source_lang or "").strip()
    target_lang = (req.target_lang or "").strip()
    if not source_lang or not target_lang:
        raise HTTPException(400, "source_lang and target_lang are both required -- argos has no language detection")
    if source_lang == target_lang:
        raise HTTPException(400, "source and target language are the same")

    # Resolve the requested name against this job's own sidecars rather
    # than joining it onto a directory: a raw name from the client would
    # otherwise be a path-traversal write primitive ('../../etc/x.srt').
    srts = {os.path.basename(p): p for p in await asyncio.to_thread(worker.list_job_files, job) if p.endswith(".srt")}
    src_srt = srts.get(os.path.basename(req.srt_name or ""))
    if not src_srt:
        raise HTTPException(400, f"no such subtitle file for this job: {req.srt_name!r}")

    asyncio.create_task(worker.translate_existing_subs(job, src_srt, source_lang, target_lang))
    return await asyncio.to_thread(db.get_job, job_id)


@app.get("/api/jobs/{job_id}/subs")
async def api_list_subs(job_id: int):
    """The job's .srt sidecars by bare filename -- what the translate modal
    offers as source tracks."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    files = await asyncio.to_thread(worker.list_job_files, job)
    return {"subs": [os.path.basename(p) for p in files if p.endswith(".srt")]}


@app.post("/api/jobs")
async def api_add_job(req: AddJobRequest):
    if not req.url.strip():
        raise HTTPException(400, "url required")
    return await asyncio.to_thread(
        _create_job, req.url.strip(), req.kind, req.quality, req.subs, req.embed_subs,
        req.strip_vocals, req.merge_subs, req.sub_primary, req.sub_secondary, req.container,
        req.gen_subs, req.gen_subs_lang, req.translate_to, req.ocr_lang, req.ocr_region,
        req.whisper_model,
    )


@app.post("/api/jobs/bulk")
async def api_bulk_add(req: BulkJobRequest):
    existing = {j["url"] for j in await asyncio.to_thread(db.get_jobs)}
    new_urls, dupes = worker.parse_bulk_urls(req.text, set() if req.force else existing)
    added = []
    for url in new_urls:
        job = await asyncio.to_thread(
            _create_job, url, req.kind, req.quality, req.subs, req.embed_subs,
            req.strip_vocals, req.merge_subs, req.sub_primary, req.sub_secondary, req.container,
            req.gen_subs, req.gen_subs_lang, req.translate_to, req.ocr_lang, req.ocr_region,
            req.whisper_model,
        )
        added.append(job)
    return {"added": added, "duplicates": dupes}


@app.post("/api/probe")
async def api_probe(req: ProbeRequest):
    try:
        formats = await asyncio.to_thread(worker.probe_formats, req.url)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"formats": formats}


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel(job_id: int):
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    await asyncio.to_thread(worker.request_cancel, job)
    return await asyncio.to_thread(db.get_job, job_id)


@app.post("/api/jobs/{job_id}/retry")
async def api_retry(job_id: int):
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] not in ("error", "canceled"):
        raise HTTPException(400, "only error/canceled jobs can be retried")
    await asyncio.to_thread(
        db.update_job, job_id, status="queued", error=None, progress=0,
        log=None, stage=None, stage_i=None, stage_n=None, speed=None, eta=None,
    )
    return await asyncio.to_thread(db.get_job, job_id)


@app.delete("/api/jobs/{job_id}")
async def api_delete_one(job_id: int, file: int = 0):
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] not in TERMINAL_STATUSES:
        await asyncio.to_thread(worker.request_cancel, job)
    if file:
        await asyncio.to_thread(worker.safe_delete_file, job.get("filepath"))
    await asyncio.to_thread(db.delete_job, job_id)
    return {"ok": True}


@app.post("/api/jobs/delete")
async def api_delete_many(req: DeleteJobsRequest):
    for jid in req.ids:
        job = await asyncio.to_thread(db.get_job, jid)
        if not job:
            continue
        if job["status"] not in TERMINAL_STATUSES:
            await asyncio.to_thread(worker.request_cancel, job)
        if req.delete_files:
            await asyncio.to_thread(worker.safe_delete_file, job.get("filepath"))
        await asyncio.to_thread(db.delete_job, jid)
    return {"ok": True}


@app.post("/api/jobs/clear")
async def api_clear_completed():
    jobs = await asyncio.to_thread(db.get_jobs)
    ids = [j["id"] for j in jobs if j["status"] in TERMINAL_STATUSES]
    for jid in ids:
        await asyncio.to_thread(db.delete_job, jid)
    return {"ok": True, "cleared": len(ids)}


@app.post("/api/jobs/{job_id}/expand")
async def api_expand(job_id: int):
    """Phase 4 gap: a watch?v=X&list=Y URL is classified as a single video
    by design (the common accident) -- this lets the user override that
    call after the fact. Reclassifies the row as kind='playlist' and puts
    it back on the queue so the *existing* claim/dispatch path expands it
    exactly like a playlist submitted fresh -- no separate expansion
    trigger to maintain. (The plan sketches setting status='expanding'
    directly, but nothing polls for that status outside of _dispatch
    reacting to a just-claimed 'queued' row, so that would sit inert until
    a restart; 'queued' is the one status that actually gets picked up.)"""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["kind"] == "playlist":
        raise HTTPException(400, "already a playlist")
    if job["status"] not in ("queued", "done", "error", "canceled"):
        raise HTTPException(400, "job is currently active, cancel it first")
    await asyncio.to_thread(
        db.update_job, job_id, kind="playlist", child_kind=job["kind"],
        status="queued", progress=0, error=None,
    )
    return await asyncio.to_thread(db.get_job, job_id)


@app.post("/api/jobs/{job_id}/subtitle")
async def api_upload_subtitle(
    job_id: int, file: UploadFile = File(...), lang: str = Form(...),
    offset_sec: float | None = Form(None),
):
    """Phase 6c: external subtitle upload, for when better subs come from a
    subtitle site rather than the video itself. Unlike 6a (same yt-dlp
    download, offset is 0 by construction), the offset here is genuinely
    unknown -- auto_sync runs unless offset_sec pins it."""
    job = await asyncio.to_thread(db.get_job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if not job.get("filepath"):
        raise HTTPException(400, "job has no output yet")
    if not lang.strip():
        raise HTTPException(400, "lang required")

    raw = await file.read()
    text = await asyncio.to_thread(worker.decode_srt_bytes, raw)
    orig_base = await asyncio.to_thread(worker.resolve_orig_base, job)
    dest = f"{orig_base}.{lang.strip()}.srt"

    def write_and_process():
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)

        logger = worker.JobLogger(job_id)
        merged_srt = None
        if job.get("merge_subs") and job.get("sub_primary") and job.get("sub_secondary"):
            merged_srt = worker.merge_bilingual_subs(
                job, orig_base, None, logger,
                auto_sync=(offset_sec is None), manual_offset_sec=offset_sec,
            )

        # Re-mux is only attempted for audio outputs: the pre-mux audio
        # sidecar is always still on disk (our mux step only ever writes a
        # *new* .mkv), so it's safe to rebuild from scratch. A video job's
        # .mkv was already muxed in place (Phase 6a) -- re-running against
        # it would stack a second "X+Y" stream alongside the first rather
        # than replacing it, so that case just refreshes the .srt on disk
        # and leaves the container as-is; delete + retry for a clean remux.
        out_filepath = None
        if job["kind"] == "audio" and job.get("embed_subs"):
            audio_src = worker.find_audio_source(job)
            if audio_src:
                sub_tracks = worker.collect_sub_tracks(job, orig_base, merged_srt)
                if sub_tracks:
                    stem = os.path.splitext(audio_src)[0]
                    out_path = stem + ".mkv"
                    worker.run_mux(audio_src, sub_tracks, out_path, False)
                    out_filepath = out_path
        return merged_srt, out_filepath, logger.dump()

    merged_srt, out_filepath, log_text = await asyncio.to_thread(write_and_process)
    update = {}
    if out_filepath:
        update["filepath"] = out_filepath
    if log_text:
        prior = job.get("log") or ""
        update["log"] = (prior + "\n" + log_text)[-8192:]
    if update:
        await asyncio.to_thread(db.update_job, job_id, **update)
    return await asyncio.to_thread(db.get_job, job_id)


@app.get("/api/cookies")
async def api_cookies_status():
    """No server-side login flow exists for this -- Google killed the
    device-code OAuth path yt-dlp used to offer, so a cookies.txt exported
    from a real logged-in browser session is the only way in. This endpoint
    just removes the SSH/scp step: paste or upload that file here instead
    of onto the NAS by hand, from any device including a phone."""
    path = worker.YTDLP_COOKIES
    if not path:
        return {"configured": False}
    exists = os.path.exists(path)
    return {
        "configured": True,
        "path": path,
        "exists": exists,
        "updated_at": os.path.getmtime(path) if exists else None,
    }


@app.post("/api/cookies")
async def api_cookies_upload(
    file: UploadFile | None = File(None), text: str | None = Form(None),
):
    path = worker.YTDLP_COOKIES
    if not path:
        raise HTTPException(400, "YTDLP_COOKIES is not set for this container")

    if file is not None:
        raw = await file.read()
        content = raw.decode("utf-8", errors="replace")
    elif text is not None and text.strip():
        content = text
    else:
        raise HTTPException(400, "provide a file or pasted text")

    # Loose sanity check, not a strict parse -- catches the "pasted the
    # wrong thing" case (an HTML page, a DevTools cookie-header string)
    # without being fussy about which exporter produced the file.
    head = content.lstrip()[:4096]
    if "\tTRUE\t" not in head and "\tFALSE\t" not in head and "Netscape" not in head:
        raise HTTPException(
            400,
            "doesn't look like a Netscape-format cookies.txt "
            "(export with a 'cookies.txt' browser extension, not copy-pasted headers)",
        )

    def write():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(write)
    return await api_cookies_status()


@app.get("/api/events")
async def api_events():
    async def gen():
        while True:
            jobs = await asyncio.to_thread(db.get_jobs)
            active = [j for j in jobs if j["status"] not in TERMINAL_STATUSES]
            for j in active:
                j.pop("log", None)  # keep the tick small; log is fetched via /api/jobs
            yield f"data: {json.dumps(active)}\n\n"
            await asyncio.sleep(1)

    # X-Accel-Buffering: no -- so a reverse proxy in front of this doesn't
    # buffer the stream and freeze progress while downloads keep running.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# StaticFiles requires the directory to exist at mount (import) time, before
# the lifespan startup hook has a chance to create it.
os.makedirs(worker.DOWNLOADS_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=worker.DOWNLOADS_DIR), name="files")
