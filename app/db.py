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
  -- queued|expanding|running|separating|muxing|done|error|canceled
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

# Append-only: each entry is applied once, in order, tracked via
# PRAGMA user_version. Future phases append here, never edit past entries.
MIGRATIONS = [SCHEMA]

# Columns writers are allowed to touch via update_job(); keeps callers from
# typo-ing a column name into a silent no-op.
_JOB_COLUMNS = {
    "url", "title", "kind", "quality", "subs", "embed_subs", "strip_vocals",
    "merge_subs", "sub_primary", "sub_secondary", "parent_id", "child_kind",
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
    continuedl resumes the .part file. 'separating' is deliberately excluded:
    that status doesn't exist until Phase 5, and resetting it there would
    throw away a completed download."""
    conn = get_conn()
    cur = conn.execute(
        "UPDATE jobs SET status='queued', updated_at=datetime('now') "
        "WHERE status IN ('running','expanding','muxing')"
    )
    conn.commit()
    return cur.rowcount
