"""Plain assert-based checks, no pytest. Grows one block per phase.

Run: python3 test_worker.py
"""
import asyncio
import os
import tempfile
import threading

# DB_PATH / DOWNLOADS_DIR are read as module-level constants at import time,
# so point them at scratch locations before importing app.db / app.worker.
_tmpdir = tempfile.mkdtemp(prefix="uvm-test-")
os.environ["DB_PATH"] = os.path.join(_tmpdir, "jobs.db")
os.environ["DOWNLOADS_DIR"] = os.path.join(_tmpdir, "downloads")
os.makedirs(os.environ["DOWNLOADS_DIR"], exist_ok=True)

import pysrt  # noqa: E402
from app import db, worker  # noqa: E402
from yt_dlp.utils import DownloadCancelled  # noqa: E402


def fresh_db():
    db.init_db()
    db.get_conn().execute("DELETE FROM jobs")
    db.get_conn().commit()


# --------------------------------------------------------------------- P1

def test_p1_format_presets():
    assert worker.build_format("video", "best") == "bv*+ba/b"
    assert worker.build_format("video", "1080") == "bv*[height<=1080]+ba/b[height<=1080]"
    assert worker.build_format("video", "720") == "bv*[height<=720]+ba/b[height<=720]"
    assert worker.build_format("video", "480") == "bv*[height<=480]+ba/b[height<=480]"
    assert worker.build_format("video", "fmt:137") == "137"
    # kind is the only source of truth for audio vs video -- quality is ignored
    assert worker.build_format("audio", "1080") == "ba/b"
    assert worker.build_format("audio", "best") == "ba/b"
    print("ok: format presets")


def test_p1_atomic_claim():
    fresh_db()
    n = 20
    for i in range(n):
        db.insert_job(url=f"https://example.com/{i}", kind="video", status="queued")

    claimed = []
    lock = threading.Lock()

    def worker_thread():
        while True:
            job = db.claim_next_job()  # each thread gets its own connection
            if job is None:
                return
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker_thread) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == n, f"expected {n} claims, got {len(claimed)}"
    assert len(set(claimed)) == n, "a row was claimed by more than one caller"
    print("ok: atomic claim under concurrency (no duplicate claims)")


def test_p1_outtmpl_uses_id():
    job = {"id": 1, "kind": "video", "quality": "best", "subs": None, "embed_subs": 0, "parent_id": None}
    opts = worker.build_ydl_opts(job)
    assert "%(id)s" in opts["outtmpl"], "outtmpl must embed the video id so same-titled videos don't collide"

    child = {"id": 2, "kind": "video", "quality": "best", "subs": None, "embed_subs": 0, "parent_id": 1}
    opts_child = worker.build_ydl_opts(child)
    assert "%(id)s" in opts_child["outtmpl"]
    assert "playlist_index" in opts_child["outtmpl"]
    assert opts_child["outtmpl"] != opts["outtmpl"]
    print("ok: outtmpl distinguishes same-titled videos via %(id)s")


def test_p1_startup_recovery():
    fresh_db()
    ids = {}
    for status in ("running", "expanding", "muxing", "separating", "queued", "done"):
        ids[status] = db.insert_job(url=f"https://example.com/{status}", kind="video", status=status)

    n = db.reset_stuck_jobs()
    assert n == 3, f"expected 3 rows reset (running/expanding/muxing), got {n}"

    for status in ("running", "expanding", "muxing"):
        assert db.get_job(ids[status])["status"] == "queued"
    # 'separating' doesn't exist until Phase 5 -- resetting it here would
    # throw away a completed download, so it must NOT be touched.
    assert db.get_job(ids["separating"])["status"] == "separating"
    assert db.get_job(ids["queued"])["status"] == "queued"
    assert db.get_job(ids["done"])["status"] == "done"
    print("ok: startup recovery requeues running/expanding/muxing, leaves separating alone")


# --------------------------------------------------------------------- P2

def test_p2_progress_throttle():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/x", kind="video", status="running")
    hook = worker.make_progress_hook(job_id, stage_n_hint=1)  # default 1s throttle
    for _ in range(25):
        hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100, "info_dict": {}})
    # 25 fires in well under a second -> at most the first write should land
    row = db.get_job(job_id)
    assert row["progress"] == 50.0, "at least the first write should have landed"

    writes = []
    orig = db.update_job
    db.update_job = lambda jid, **kw: (writes.append(kw), orig(jid, **kw))[1]
    try:
        for _ in range(25):
            hook({"status": "downloading", "downloaded_bytes": 60, "total_bytes": 100, "info_dict": {}})
    finally:
        db.update_job = orig
    assert len(writes) <= 1, f"expected <=1 DB write/sec, got {len(writes)}"
    print("ok: progress hook throttles to <=1 write/sec")


def test_p2_multifile_progress_monotonic():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/x", kind="video", status="running")
    # flush_interval=0 so every hook fire is observable
    hook = worker.make_progress_hook(job_id, stage_n_hint=2, flush_interval=0)
    observed = []

    def record(jid, **kw):
        if "progress" in kw:
            observed.append(kw["progress"])

    orig = db.update_job
    db.update_job = record
    try:
        video_info = {"vcodec": "avc1", "acodec": "none"}
        audio_info = {"vcodec": "none", "acodec": "opus"}
        # file 1 (video): 0 -> 100%
        for pct in (0, 25, 50, 75, 100):
            hook({"status": "downloading", "filename": "a.f137.mp4", "downloaded_bytes": pct,
                  "total_bytes": 100, "info_dict": video_info})
        # file 2 (audio): 0 -> 100%, must NOT rewind below file 1's ending value
        for pct in (0, 25, 50, 75, 100):
            hook({"status": "downloading", "filename": "a.f251.webm", "downloaded_bytes": pct,
                  "total_bytes": 100, "info_dict": audio_info})
    finally:
        db.update_job = orig

    assert observed == sorted(observed), f"progress rewound: {observed}"
    assert observed[0] < 50 and abs(observed[4] - 50.0) < 0.01, observed
    assert observed[-1] == 100.0
    print("ok: multi-file progress is monotonic across a 2-file bv*+ba download (no rewind)")


def test_p2_cancel_set_aborts():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/x", kind="video", status="running")
    hook = worker.make_progress_hook(job_id, stage_n_hint=1)
    worker.CANCELED.add(job_id)
    try:
        raised = False
        try:
            hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 100, "info_dict": {}})
        except DownloadCancelled:
            raised = True
        assert raised, "hook must raise DownloadCancelled once job_id is in CANCELED"
    finally:
        worker.CANCELED.discard(job_id)
    print("ok: cancel set aborts the download via the progress hook")


def test_p2_realpath_guard():
    downloads = os.environ["DOWNLOADS_DIR"]
    inside = os.path.join(downloads, "video [abc123].mp4")
    with open(inside, "w") as f:
        f.write("x")
    removed = worker.safe_delete_file(inside, downloads_dir=downloads)
    assert os.path.realpath(inside) in removed
    assert not os.path.exists(inside)

    outside = "/etc/hosts"
    raised = False
    try:
        worker.safe_delete_file(outside, downloads_dir=downloads)
    except ValueError:
        raised = True
    assert raised, "must refuse to delete a path outside the downloads dir"
    assert os.path.exists(outside), "must not have touched the file outside downloads_dir"
    print("ok: realpath guard rejects a filepath outside downloads dir")


# --------------------------------------------------------------------- P3

def test_p3_subtitle_opts():
    opts = worker.build_subtitle_opts("en,pt")
    assert opts["writesubtitles"] is True
    assert opts["writeautomaticsub"] is True
    assert opts["subtitleslangs"] == ["en", "pt"]
    assert opts["subtitlesformat"] == "srt/best"

    job = {"id": 1, "kind": "video", "quality": "best", "subs": "en,pt", "embed_subs": 1, "parent_id": None}
    ydl_opts = worker.build_ydl_opts(job)
    assert ydl_opts["subtitleslangs"] == ["en", "pt"]
    assert {"key": "FFmpegEmbedSubtitle"} in ydl_opts["postprocessors"]
    assert ydl_opts["merge_output_format"] == "mkv"

    # embedding into a bare audio container isn't handled by yt-dlp's
    # FFmpegEmbedSubtitle -- Phase 6b's manual ffmpeg mux does that later.
    audio_job = dict(job, kind="audio")
    audio_opts = worker.build_ydl_opts(audio_job)
    assert {"key": "FFmpegEmbedSubtitle"} not in audio_opts["postprocessors"]
    print("ok: subtitle option assembly for multi-lang + embed")


# --------------------------------------------------------------------- P4

def test_p4_bulk_parse():
    text = "\n".join([
        "https://a.com/1",
        "# a comment",
        "",
        "https://a.com/2",
        "https://a.com/1",   # dupe within paste
        "   ",
        "https://a.com/3",
    ])
    new, dupes = worker.parse_bulk_urls(text, existing_urls={"https://a.com/3"})
    assert new == ["https://a.com/1", "https://a.com/2"]
    assert dupes == ["https://a.com/1", "https://a.com/3"]
    print("ok: bulk URL parser strips blanks/comments and dedupes (in-paste + existing)")


def test_p4_playlist_classifier():
    cases = {
        "https://youtube.com/watch?v=aaa": "video",
        "https://youtube.com/watch?v=aaa&list=PLxxx": "video",  # the common accident
        "https://youtube.com/playlist?list=PLxxx": "playlist",
        "https://youtube.com/channel/UCxxxx": "playlist",
        "https://youtube.com/channel/UCxxxx/videos": "playlist",
        "https://youtube.com/@somehandle/videos": "playlist",
        "https://youtube.com/c/somechannel": "playlist",
        "https://youtube.com/watch?v=aaa": "video",
    }
    for url, expected in cases.items():
        got = worker.classify_url(url)
        assert got == expected, f"{url}: expected {expected}, got {got}"
    print("ok: playlist classifier (watch+list stays a single video)")


# --------------------------------------------------------------------- P5

def test_p5_demucs_progress_parser_no_trailing_newline():
    """The bug this guards against: demucs' tqdm bar redraws stderr with
    '\\r' and never emits a newline until the whole separation is done.
    readline()/line-iteration over that blocks for the entire run and the
    bar sits frozen at 0%. Feed the parser a single chunk containing
    multiple \\r-delimited redraws with NO trailing newline -- exactly what
    a synchronous read of a still-running demucs' stderr looks like -- and
    it must still extract the latest (last) percent, not the first."""
    parser = worker.DemucsProgressParser()
    chunk = (
        "Separating track song.mp3\r"
        "  0%|          | 0/889 [00:00<?, ?it/s]\r"
        " 45%|####      | 400/889 [00:10<00:12, 39.61it/s]\r"
        "100%|##########| 889/889 [00:22<00:00, 39.85it/s]"  # no trailing \r or \n
    )
    pct = parser.feed(chunk)
    assert pct == 100, f"expected the latest (last) percent, got {pct}"
    print("ok: demucs \\r-delimited progress with no trailing newline parses (the hang bug)")


def test_p5_demucs_progress_parser_streams_incrementally():
    """The whole point of chunked reads over readline() is that progress is
    visible *during* the run, not just at the end -- feed the bar in pieces
    (as os.read would deliver it) and confirm an early, non-final percent is
    observable partway through, proving it doesn't wait for full completion."""
    parser = worker.DemucsProgressParser()
    full = "  0%|x\r 10%|x\r 20%|x\r 30%|x\r 40%|x\r 50%|x"
    seen = []
    # feed in small arbitrary-sized pieces, not aligned to the \r boundaries
    for i in range(0, len(full), 5):
        pct = parser.feed(full[i:i + 5])
        if pct is not None:
            seen.append(pct)
    assert seen, "expected at least one percent observed mid-stream"
    assert seen[-1] == 50, f"expected the final observed percent to be 50, got {seen[-1]}"
    assert any(p < 50 for p in seen), f"expected an intermediate percent before completion, saw only {seen}"
    print("ok: demucs progress is observable incrementally, not just at the end")


def test_p5_independent_semaphores():
    """Separation is CPU-bound and must never block, or be blocked by,
    I/O-bound downloads -- confirm SEPARATION_SLOTS and MAX_CONCURRENT are
    genuinely separate asyncio.Semaphore instances, not the same one."""
    assert worker.DOWNLOAD_SEM is not worker.SEPARATION_SEM

    async def check():
        acquired = 0
        for _ in range(worker.MAX_CONCURRENT):
            await worker.DOWNLOAD_SEM.acquire()
            acquired += 1
        assert worker.DOWNLOAD_SEM.locked(), "download semaphore should be fully drained"
        try:
            # separation must still be free even though the download
            # semaphore is completely exhausted
            got = await asyncio.wait_for(worker.SEPARATION_SEM.acquire(), timeout=0.5)
            assert got
            worker.SEPARATION_SEM.release()
        finally:
            for _ in range(acquired):
                worker.DOWNLOAD_SEM.release()

    asyncio.run(check())
    print("ok: SEPARATION_SLOTS and MAX_CONCURRENT are independent semaphores")


def test_p5_missing_demucs_binary_fails_fast():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/song", kind="audio", status="separating")
    src = os.path.join(os.environ["DOWNLOADS_DIR"], "song.m4a")
    with open(src, "w") as f:
        f.write("x")
    job = db.get_job(job_id)
    job["filepath"] = src
    # demucs is (almost certainly) not on PATH in this test environment --
    # confirm the clear, specific error rather than a hang or a traceback.
    import shutil as _shutil
    if _shutil.which("demucs") is not None:
        print("skip: demucs is actually installed here, can't exercise the not-on-PATH path")
        return
    result = worker.run_separation(job)
    assert result["status"] == "error"
    assert "WITH_DEMUCS=true" in result["error"]
    print("ok: missing demucs binary fails fast with a clear error")


# --------------------------------------------------------------------- P6

def test_p6_lang_prefix_matching():
    available = {"zh-Hans": "/x/v.zh-Hans.srt", "en": "/x/v.en.srt"}
    assert worker.match_lang("zh", available) == "zh-Hans"
    assert worker.match_lang("en", available) == "en"
    assert worker.match_lang("pt", available) is None

    available2 = {"zh-Hant": "/x/v.zh-Hant.srt", "zh-TW": "/x/v.zh-TW.srt"}
    got = worker.match_lang("zh", available2)
    assert got in available2, "zh must match some zh-* variant"

    # human-authored preferred over auto-generated when both prefix-match
    available3 = {"zh-Hans": "/x/a.srt", "zh-Hant": "/x/b.srt"}
    got = worker.match_lang("zh", available3, human={"zh-Hant"})
    assert got == "zh-Hant", "human-authored variant should win the tie-break"
    print("ok: tolerant language-prefix matching (zh matches zh-Hans/zh-Hant/zh-TW)")


def test_p6_rollup_caption_dedup():
    """Synthetic sample of YouTube's auto-caption rollup pattern: each cue
    repeats the tail words of the previous one."""
    class FakeCue:
        def __init__(self, text):
            self.text = text

    cues = [
        FakeCue("the quick brown"),
        FakeCue("quick brown fox jumps"),
        FakeCue("brown fox jumps over the"),
        FakeCue("fox jumps over the lazy dog"),
    ]
    kept = worker.dedup_rollup_captions(cues)
    joined = " ".join(c.text for c in kept)
    assert joined == "the quick brown fox jumps over the lazy dog", joined
    # no cue's text should reappear verbatim in the next cue (the bug)
    for a, b in zip(kept, kept[1:]):
        assert a.text != b.text
        assert not (a.text and b.text.startswith(a.text))
    print("ok: rollup-caption dedup strips repeated tail words between neighbours")


def _write_srt(path, entries):
    """entries: list of (start_ms, end_ms, text)."""
    lines = []
    for i, (start, end, text) in enumerate(entries, start=1):
        def ts(ms):
            h, ms = divmod(ms, 3600000)
            m, ms = divmod(ms, 60000)
            s, ms = divmod(ms, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_p6_merge_subtitles_zh_over_en():
    """Two tiny hand-written .srt files with deliberately mismatched cue
    boundaries: one zh cue spans two en cues. Timings should follow the zh
    (primary/'chi') track and the merged text should carry zh first, en
    lines after."""
    d = tempfile.mkdtemp(prefix="uvm-merge-test-")
    zh_path = os.path.join(d, "v.zh.srt")
    en_path = os.path.join(d, "v.en.srt")
    out_path = os.path.join(d, "v.zh-en.srt")
    _write_srt(zh_path, [(0, 4000, "你好世界")])  # "hello world" (one wide cue)
    _write_srt(en_path, [(0, 2000, "Hello"), (2000, 4000, "world")])  # two narrower cues

    from app.merge_srt import merge_subtitles
    merge_subtitles(zh_path, en_path, out_path, auto_sync=False)

    merged = pysrt.open(out_path)
    assert len(merged) == 1
    assert merged[0].text == "你好世界\nHello\nworld", merged[0].text
    assert merged[0].start.ordinal == 0
    assert merged[0].end.ordinal == 4000
    print("ok: merge_subtitles produces zh-over-en output, timings follow the primary track")


def test_p6_gbk_srt_decodes_without_raising():
    # Realistic subtitle-length content -- charset_normalizer's statistical
    # detector needs enough bytes to disambiguate GBK from other CJK
    # encodings (a 4-character sample is genuinely ambiguous with Korean
    # CP949); a couple of real subtitle lines is representative of what the
    # upload path actually receives.
    content = (
        "1\n00:00:00,000 --> 00:00:02,000\n"
        "你好，世界，这是一个用于验证编码检测的字幕示例。\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n"
        "第二行字幕内容，包含更多中文文字以提高检测准确性。\n"
    )
    raw = content.encode("gbk")
    # pysrt.open() (utf-8 only) would raise UnicodeDecodeError on this --
    # that's exactly the bug decode_srt_bytes exists to avoid.
    try:
        raw.decode("utf-8")
        assert False, "fixture should not happen to be valid utf-8"
    except UnicodeDecodeError:
        pass
    text = worker.decode_srt_bytes(raw)
    assert "你好" in text and "世界" in text, f"GBK bytes did not decode correctly: {text!r}"
    print("ok: GBK-encoded .srt decodes without raising UnicodeDecodeError")


def test_p6_mux_argv_construction():
    tracks = [
        ("/downloads/v.zh.srt", "zho", "zh"),
        ("/downloads/v.en.srt", "eng", "en"),
        ("/downloads/v.zh-en.srt", "zho", "zh+en"),
    ]
    cmd = worker.build_mux_cmd("/downloads/v.m4a", tracks, "/downloads/v.mkv", copy_all=False)
    assert cmd[:4] == ["ffmpeg", "-y", "-i", "/downloads/v.m4a"]
    for path, _lang, _title in tracks:
        assert "-i" in cmd and path in cmd
    assert "0:a" in cmd  # audio-only source: map only the audio stream, not '0'
    for i in range(1, len(tracks) + 1):
        assert str(i) in cmd
    assert "-c:a" in cmd and "copy" in cmd
    assert "-c:s" in cmd and "srt" in cmd
    assert "language=zho" in cmd
    assert "language=eng" in cmd
    assert "title=zh+en" in cmd
    assert cmd[-1] == "/downloads/v.mkv"

    # video (copy_all=True) maps every stream from the source, not just audio
    cmd_video = worker.build_mux_cmd("/downloads/v.mkv", tracks[:1], "/downloads/v.mkv", copy_all=True)
    assert "0" in cmd_video and "0:a" not in cmd_video
    print("ok: ffmpeg mux argv construction for N subtitle languages")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} test blocks passed.")
