# Architecture

## Modules

- `app/main.py` — FastAPI routes, SSE progress stream, startup recovery
- `app/worker.py` — the queue loop, yt-dlp calls, and every post-processing stage (separation, transcription, translation, muxing)
- `app/db.py` — SQLite schema, append-only migrations, thread-safe connections
- `app/merge_srt.py` — vendored bilingual subtitle merger, kept byte-for-byte identical to its source
- `app/static/index.html` — the entire frontend: vanilla JS, no build step, no dependencies

## Job lifecycle

```
queued → running → [separating] → [transcribing] → [muxing] → done
```

`error` and `canceled` are terminal states reachable from any point. Bracketed stages run only if the job asked for them.

Every stage writes progress to SQLite rather than only to an in-memory socket, so a page refresh or container restart never loses state. Interrupted jobs resume from whichever stage they were in, without re-downloading — `resume_separation` and `resume_transcribe` restart that stage from scratch (neither Demucs nor Whisper is meaningfully resumable mid-run) and continue through the same mux stage a fresh job would hit.

## Design rules

These are properties the code currently has, each one the result of a bug that violated it. Changes should preserve them.

**Failures are visible.** A background task launched with a bare `create_task` and never awaited surfaces an exception as `Task exception was never retrieved` on stderr and nothing else — the row keeps its active status and the job appears frozen forever. Every such coroutine routes exceptions through `fail_job()`, which writes the status, the error column, and a job-log line.

**Failures are non-destructive.** An optional step failing keeps whatever already succeeded. Translation failing after a five-minute Whisper pass warns and keeps the transcript, rather than discarding correct output because an add-on couldn't reach its model directory.

**Terminal ≠ useless.** A job that ends in `error` still exposes every file-based action, because a late failure leaves real output on disk. The API accepts `done`/`error`/`canceled` for those routes; the UI must not be stricter than the API.

**Logs go everywhere at once.** `append_job_log` writes to the job's DB column *and* stdout *and* the persistent app log. When stage lines went only to the DB, a successful run was invisible to `docker logs`, and diagnosing from outside the container meant chasing a job that had already finished.

**Fix the cause, not the caller.** Three separate bugs came from a non-writable `HOME` in a non-root container, each fixed by pointing one library elsewhere, each leaving the next library to fail in production. See [DEPLOY.md](DEPLOY.md#the-user--home-interaction).

**Client input is resolved, not joined.** `translate-subs` takes an `srt_name` from the client and reads that file. It is resolved against the job's own sidecars rather than joined onto a directory, so `../../etc/passwd` is unreachable.

## Subtitle files

Sidecars are named against the job's *pre-pipeline* stem, whatever stages ran:

```
<title> [<id>].whisper.srt        generated
<title> [<id>].ocr.srt            generated
<title> [<id>].whisper.en.srt     translated from the whisper track
<title> [<id>].zh-en.srt          bilingual merge
```

`resolve_orig_base()` recovers that stem from the current filepath without a DB column, since only this app's own transformations change it (Demucs appends `_novocals`; muxing changes the extension or replaces the file in place). The frontend mirrors this in `srtSuffix()` to address a track kind across a whole playlist, where no single filename applies.

## Testing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_worker.py
```

Plain `assert`-based checks in `test_worker.py`, no framework, run directly by CI. No virtualenv is committed.

Tests use nothing outside `requirements.txt` — notably not `fastapi.testclient`, which needs `httpx`. Endpoint tests call the route coroutines directly and assert on `HTTPException` instead, rather than adding a test-only dependency to the production image. (A test that passed locally on an incidentally-installed `httpx` and failed in CI is why this is written down.)
