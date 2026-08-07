"""sqlite3 schema + migrations + thread-local connections.

Two rules that are load-bearing (see plan): every thread gets its own
connection (yt-dlp's progress hook writes from a worker thread spawned by
asyncio.to_thread), and job claims are a single atomic UPDATE ... RETURNING,
never SELECT-then-UPDATE.
"""
import os
import sqlite3
import threading

DB_PATH = os.environ.get("DB_PATH", "/data/jobs.db")

_local = threading.local()

# Full schema up front, including columns Phase 5/6 will use later
# (strip_vocals, merge_subs, sub_primary, sub_secondary) so there's no
# migration churn once those phases land. Only phase 1-4 columns are wired
# up by app code today.
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL,
  title TEXT,
  kind TEXT NOT NULL DEFAULT 'video',      -- video | audio | playlist
  quality TEXT NOT NULL DEFAULT 'best',    -- best|1080|720|480, or fmt:<format_id>
  subs TEXT,                               -- comma langs, "en,pt"; NULL = none
  embed_subs INTEGER DEFAULT 1,
  strip_vocals INTEGER DEFAULT 0,
  merge_subs INTEGER DEFAULT 0,
  sub_primary TEXT,                        -- top line,    e.g. 'zh'  (phase 6)
  sub_secondary TEXT,                      -- bottom line, e.g. 'en'  (phase 6)
  parent_id INTEGER,                       -- playlist child -> parent row
  child_kind TEXT,                         -- kind='playlist' rows only: video|audio to give children
  status TEXT NOT NULL DEFAULT 'queued',
  -- queued|expanding|running|separating|transcribing|muxing|done|error|canceled
  progress REAL DEFAULT 0,
  stage TEXT, stage_i INTEGER, stage_n INTEGER,   -- "video 1 of 2"
  speed TEXT, eta TEXT,
  filepath TEXT, error TEXT, log TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_parent ON jobs(parent_id);
"""

# Video source streams are almost always VP9/AV1-in-webm on modern
# YouTube; without a container preference the merge naturally lands there
# regardless of what the output was supposed to be. mp4 default; mkv or
# webm selectable per job. Meaningless for kind='audio', left at its
# default there.
_ADD_CONTAINER = "ALTER TABLE jobs ADD COLUMN container TEXT NOT NULL DEFAULT 'mp4';"

# Phase 7: hardsub subtitle generation (Whisper ASR / Tesseract OCR) for
# reuploads that only have burned-in subtitles. gen_subs: NULL/'off'
# (default, no generation), 'whisper', 'ocr', or 'both'. gen_subs_lang is an
# optional hint (e.g. 'zh') fed to both Whisper's language= param and OCR's
# tesseract lang-pack selection.
_ADD_GEN_SUBS = (
    "ALTER TABLE jobs ADD COLUMN gen_subs TEXT;"
    "ALTER TABLE jobs ADD COLUMN gen_subs_lang TEXT;"
)

# Phase 8: translate a generated (Whisper/OCR) track into a second,
# additional soft-sub track via offline argos-translate. NULL/empty = off.
# Needs gen_subs_lang set too (argos-translate has no language detection of
# its own) -- clamped in main.py's _create_job the same way gen_subs='ocr'
# is clamped for kind='audio'.
_ADD_TRANSLATE = "ALTER TABLE jobs ADD COLUMN translate_to TEXT;"

# Per-job OCR tuning, both NULL = "use the container's env defaults".
#
# ocr_lang is a *verbatim* tesseract lang string ('chi_sim', 'chi_sim+eng',
# ...), not a language hint to be mapped. gen_subs_lang alone turned out to
# be too coarse for OCR: 'zh' can't distinguish simplified from traditional,
# and the choice materially changes output -- on simplified-Chinese source,
# including chi_tra makes tesseract emit *traditional* character forms
# (verified: 九阳神拳吧 read as 九陽神拳吧 with chi_sim+chi_tra+eng, but
# correctly as 九阳... with chi_sim alone). No mapping can guess that from
# 'zh', so it's a direct user choice.
#
# ocr_region: 'bottom' (default, OCR_CROP_BOTTOM_PCT of the frame -- where
# dialogue hardsubs almost always sit, and much faster) or 'full' (whole
# frame, for videos with captions/labels elsewhere on screen, e.g. corner
# annotations that the bottom crop misses entirely).
_ADD_OCR_OPTS = (
    "ALTER TABLE jobs ADD COLUMN ocr_lang TEXT;"
    "ALTER TABLE jobs ADD COLUMN ocr_region TEXT;"
)

# Per-job Whisper model size. NULL = use the WHISPER_MODEL env default.
# Worth choosing per job rather than per container: model size is the main
# accuracy/time tradeoff (roughly 3x the runtime per step up), and it's
# usually worth paying on one difficult video without slowing down every
# other job on the box.
_ADD_WHISPER_MODEL = "ALTER TABLE jobs ADD COLUMN whisper_model TEXT;"

# Append-only: each entry is applied once, in order, tracked via
# PRAGMA user_version. Future phases append here, never edit past entries.
MIGRATIONS = [SCHEMA, _ADD_CONTAINER, _ADD_GEN_SUBS, _ADD_TRANSLATE, _ADD_OCR_OPTS, _ADD_WHISPER_MODEL]

# Columns writers are allowed to touch via update_job(); keeps callers from
# typo-ing a column name into a silent no-op.
_JOB_COLUMNS = {
    "url", "title", "kind", "quality", "container", "subs", "embed_subs", "strip_vocals",
    "merge_subs", "sub_primary", "sub_secondary", "gen_subs", "gen_subs_lang", "translate_to",
    "ocr_lang", "ocr_region", "whisper_model",
    "parent_id", "child_kind",
    "status", "progress", "stage", "stage_i", "stage_n", "speed", "eta",
    "filepath", "error", "log",
}


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i in range(version, len(MIGRATIONS)):
        conn.executescript(MIGRATIONS[i])
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def claim_next_job() -> dict | None:
    """Atomically grab the oldest queued job. Never SELECT-then-UPDATE:
    with MAX_CONCURRENT>1, two workers racing the same SELECT would both
    grab the same row."""
    conn = get_conn()
    cur = conn.execute(
        """
        UPDATE jobs SET status='running', updated_at=datetime('now')
        WHERE id = (SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1)
        RETURNING *
        """
    )
    row = cur.fetchone()
    conn.commit()
    return row_to_dict(row)


def insert_job(**fields) -> int:
    conn = get_conn()
    cols = list(fields.keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, [fields[c] for c in cols])
    conn.commit()
    return cur.lastrowid


def get_job(job_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row_to_dict(row)


def get_job_by_url_status(url: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM jobs WHERE url=? ORDER BY id DESC LIMIT 1", (url,)
    ).fetchone()
    return row_to_dict(row)


def get_jobs() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def get_jobs_by_status(status: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM jobs WHERE status=? ORDER BY id ASC", (status,)).fetchall()
    return [dict(r) for r in rows]


def get_children(parent_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE parent_id=? ORDER BY id ASC", (parent_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def update_job(job_id: int, **fields) -> None:
    if not fields:
        return
    bad = set(fields) - _JOB_COLUMNS
    if bad:
        raise ValueError(f"not a job column: {bad}")
    conn = get_conn()
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c}=?" for c in cols) + ", updated_at=datetime('now')"
    conn.execute(
        f"UPDATE jobs SET {set_clause} WHERE id=?",
        [fields[c] for c in cols] + [job_id],
    )
    conn.commit()


def delete_job(job_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()


def reset_stuck_jobs() -> int:
    """Startup recovery. running/expanding/muxing get requeued -- yt-dlp's
    continuedl resumes the .part file. 'separating' and 'transcribing' are
    deliberately excluded: those statuses don't exist until Phase 5/7, and
    resetting them here would throw away a completed download (and, for
    transcribing, a possibly-already-finished separation step too)."""
    conn = get_conn()
    cur = conn.execute(
        "UPDATE jobs SET status='queued', updated_at=datetime('now') "
        "WHERE status IN ('running','expanding','muxing')"
    )
    conn.commit()
    return cur.rowcount
