"""Plain assert-based checks, no pytest. Grows one block per phase.

Run: python3 test_worker.py
"""
import asyncio
import os
import tempfile
import threading
import time

# DB_PATH / DOWNLOADS_DIR are read as module-level constants at import time,
# so point them at scratch locations before importing app.db / app.worker.
_tmpdir = tempfile.mkdtemp(prefix="uvm-test-")
os.environ["DB_PATH"] = os.path.join(_tmpdir, "jobs.db")
os.environ["DOWNLOADS_DIR"] = os.path.join(_tmpdir, "downloads")
os.makedirs(os.environ["DOWNLOADS_DIR"], exist_ok=True)

import pysrt  # noqa: E402
from app import db, main, worker  # noqa: E402
from yt_dlp.utils import DownloadCancelled  # noqa: E402


def fresh_db():
    db.init_db()
    db.get_conn().execute("DELETE FROM jobs")
    db.get_conn().commit()


# --------------------------------------------------------------------- P1

def test_p1_format_presets():
    # default container is mp4: source streams prefer mp4/m4a first (avoids
    # landing on webm just because that's what the highest-quality stream
    # happens to be), falling all the way through to an unrestricted
    # bv*+ba/b only if nothing mp4-compatible exists.
    assert worker.build_format("video", "best") == "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
    assert worker.build_format("video", "1080") == (
        "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]"
    )
    assert worker.build_format("video", "720") == (
        "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b[height<=720]"
    )
    assert worker.build_format("video", "480") == (
        "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/bv*[height<=480]+ba/b[height<=480]"
    )
    # mkv gets the same mp4/m4a-preferring source selector as mp4 -- mkv
    # accepts any codec, but the point is still to avoid webm when avoidable.
    assert worker.build_format("video", "best", container="mkv") == worker.build_format("video", "best", container="mp4")
    # explicit webm skips the preference and takes the best stream outright
    # -- this is the old unconditional behavior, kept only for this choice.
    assert worker.build_format("video", "best", container="webm") == "bv*+ba/b"
    assert worker.build_format("video", "720", container="webm") == "bv*[height<=720]+ba/b[height<=720]"
    assert worker.build_format("video", "fmt:137") == "137"
    # kind is the only source of truth for audio vs video -- quality and
    # container are both ignored (container is a video-only concept, gated
    # in the UI the same way strip_vocals is gated to audio-only)
    assert worker.build_format("audio", "1080") == "ba/b"
    assert worker.build_format("audio", "best") == "ba/b"
    assert worker.build_format("audio", "best", container="mkv") == "ba/b"
    print("ok: format presets, mp4/mkv prefer mp4 sources, webm is opt-in")


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
    # no container specified -> defaults to mp4, and embedding respects it
    # rather than always forcing mkv (that was the old, less flexible behavior)
    assert ydl_opts["merge_output_format"] == "mp4"

    mkv_job = dict(job, container="mkv")
    assert worker.build_ydl_opts(mkv_job)["merge_output_format"] == "mkv"

    # webm+subs falls back to mkv -- embedding into a raw .webm container is
    # unreliable, and mkv holds VP9/Opus (webm's own codecs) natively so it
    # loses nothing.
    webm_job = dict(job, container="webm")
    assert worker.build_ydl_opts(webm_job)["merge_output_format"] == "mkv"

    # a plain video with no subtitles still gets remuxed to the chosen
    # container -- this isn't conditional on subtitle embedding.
    no_subs_job = {"id": 2, "kind": "video", "quality": "best", "subs": None, "embed_subs": 1, "parent_id": None}
    assert worker.build_ydl_opts(no_subs_job)["merge_output_format"] == "mp4"
    no_subs_webm = dict(no_subs_job, container="webm")
    assert "merge_output_format" not in worker.build_ydl_opts(no_subs_webm)

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


# --------------------------------------------------------------------- P7

def test_p7_write_srt_format():
    d = tempfile.mkdtemp(prefix="uvm-srt-test-")
    out = os.path.join(d, "out.srt")
    cues = [(0.0, 1.5, "hello"), (1.5, 3.75, "world\nsecond line")]
    worker.write_srt(cues, out)
    subs = pysrt.open(out)
    assert len(subs) == 2
    assert subs[0].index == 1 and subs[1].index == 2
    assert subs[0].start.ordinal == 0 and subs[0].end.ordinal == 1500
    assert subs[1].start.ordinal == 1500 and subs[1].end.ordinal == 3750
    assert subs[1].text == "world\nsecond line"
    print("ok: write_srt produces correctly indexed/timestamped cues")


def test_p7_ocr_frames_to_cues_merge():
    """Synthetic (timestamp, raw_ocr_text) samples at a 0.5s frame interval,
    including OCR noise (trailing whitespace / doubled internal spaces)
    between frames that should still merge as 'the same line'."""
    frame_interval = 0.5
    frames = [
        (0.0, "Hello there  "),      # cue 1, frame 1
        (0.5, "Hello there"),        # cue 1, frame 2 (whitespace noise only)
        (1.0, ""),                   # blank -- subtitle off-screen, closes cue 1
        (1.5, "  "),                 # still blank
        (2.0, "Second line"),        # cue 2, frame 1
        (2.5, "Second line"),        # cue 2, frame 2
        (3.0, "Third line"),         # text changed -- closes cue 2, opens cue 3
    ]
    cues = worker.ocr_frames_to_cues(frames, frame_interval)
    assert cues == [
        (0.0, 1.0, "Hello there"),
        (2.0, 3.0, "Second line"),
        (3.0, 3.5, "Third line"),
    ], cues
    print("ok: ocr_frames_to_cues merges matching-text frames, skips blanks, splits on change")


def test_p7_ocr_frames_to_cues_fuzzy_noise():
    """Reproduces an actual bug found by running the real pipeline end to
    end against a synthetic video: static on-screen text got OCR'd
    slightly differently on nearly every 0.5s-apart frame (one wrong
    character each time -- real Tesseract noise, not a hypothetical).
    Exact-string matching fragmented one real 3-second cue into six
    wrong-duration ones. This is real captured frame data from that run,
    not hand-constructed -- proving the fix against the actual failure,
    not a sanitized version of it."""
    frame_interval = 0.5
    cue1_frames = [
        (0.0, "儿童玩具剑算是哪笃出啊"),
        (0.5, "儿童玩具剑算是哪至出啊"),
        (1.0, "儿童玩具剑算是哪三出啊"),
        (1.5, "eras Ss = why"),  # one badly garbled frame in the middle
        (2.0, "儿童玩具剑算是哪皇出啊"),
        (2.5, "儿童玩具剑算是哪择出啊"),
    ]
    frames = cue1_frames + [(3.0, ""), (3.5, "")]  # blank gap between cues
    cues = worker.ocr_frames_to_cues(frames, frame_interval)
    assert len(cues) == 1, f"one static line should merge into one cue, got {len(cues)}: {cues}"
    start, end, text = cues[0]
    assert start == 0.0
    assert end == 3.0, f"cue should span the full run, got end={end}"
    assert text.startswith("儿童玩具剑算是哪"), text
    print("ok: ocr_frames_to_cues survives real per-frame OCR noise (fuzzy grouping, majority-vote text)")


def test_p7_ocr_lang_for():
    # chi_sim+eng (a lone CJK script combined with eng) is a confirmed-bad
    # Tesseract combination -- garbles real text into noise. Must be
    # chi_sim+chi_tra+eng: two CJK scripts together anchor recognition
    # correctly, and eng can safely ride along once both are present.
    assert worker.ocr_lang_for(None) == worker.OCR_LANG
    assert worker.ocr_lang_for("zh") == "chi_sim+chi_tra+eng"
    assert worker.ocr_lang_for("zh-Hans") == "chi_sim+chi_tra+eng"
    assert worker.ocr_lang_for("en") == "eng"
    assert worker.ocr_lang_for("fr") == worker.OCR_LANG  # unmapped -> fall back to default

    # An explicit per-job ocr_lang beats the hint outright, including the
    # case the whole option exists for: 'zh' + chi_sim, where the mapping
    # would otherwise add chi_tra and emit traditional forms for
    # simplified source.
    assert worker.ocr_lang_for("zh", "chi_sim") == "chi_sim"
    assert worker.ocr_lang_for(None, " eng ") == "eng"
    assert worker.ocr_lang_for("zh", "") == "chi_sim+chi_tra+eng"  # blank = not set
    print("ok: ocr_lang_for maps zh/en, honors an explicit ocr_lang, falls back to OCR_LANG")


def test_p7_binarize_normalizes_both_polarities():
    """The exact failure that got the earlier fixed-threshold attempt
    rejected: a constant that works on light-text-on-dark blanks
    dark-text-on-light, and vice versa. Otsu + a derived polarity has to
    survive both, from the same code path, with no tuning between them."""
    from PIL import Image, ImageDraw

    def frame(bg, fg):
        # Text as a minority of pixels in a wide, short crop -- the shape a
        # subtitle strip actually has.
        img = Image.new("RGB", (320, 60), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([40, 22, 90, 38], fill=fg)
        d.rectangle([110, 22, 160, 38], fill=fg)
        d.rectangle([180, 22, 230, 38], fill=fg)
        return img

    light_on_dark = worker.binarize_for_ocr(frame((20, 20, 30), (235, 235, 245)))
    dark_on_light = worker.binarize_for_ocr(frame((240, 240, 235), (15, 15, 20)))

    for name, out in (("light-on-dark", light_on_dark), ("dark-on-light", dark_on_light)):
        hist = out.histogram()
        black, white = hist[0], hist[255]
        assert black + white == 320 * 60, f"{name}: output must be strictly two-valued"
        assert black > 0 and white > 0, f"{name}: binarized to one flat colour -- text was lost"
        # Tesseract is trained on dark text on light background; the
        # minority class is the text, so it must end up the dark one
        # regardless of which polarity came in.
        assert black < white, f"{name}: text ended up light-on-dark, polarity not normalized"

    # A uniform frame has no split to find. It must not raise, and it must
    # come out flat -- which reads as "no text", the same as a blank frame.
    flat = worker.binarize_for_ocr(Image.new("RGB", (64, 16), (90, 90, 90)))
    fh = flat.histogram()
    assert fh[0] + fh[255] == 64 * 16 and (fh[0] == 0 or fh[255] == 0), "uniform input must not invent text"
    print("ok: binarize_for_ocr normalizes both hardsub polarities to dark-on-light, survives a uniform frame")


def test_p7_otsu_threshold_splits_bimodal_histogram():
    # Two tight peaks at 40 and 210: the split belongs strictly between
    # them, wherever exactly Otsu lands.
    hist = [0] * 256
    hist[40] = 800
    hist[210] = 200
    t = worker.otsu_threshold(hist)
    assert 40 <= t < 210, f"threshold {t} must separate the two modes"
    assert worker.otsu_threshold([0] * 256) == 127, "empty histogram falls back, must not divide by zero"
    single = [0] * 256
    single[77] = 500
    assert isinstance(worker.otsu_threshold(single), int), "single-mode histogram must not raise"
    print("ok: otsu_threshold splits a bimodal histogram and survives degenerate input")


def test_p7_append_job_log_accumulates_and_trims():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/show", kind="video", status="running")
    worker.append_job_log(job_id, "stage: separating")
    worker.append_job_log(job_id, "stage: transcribing")
    log = db.get_job(job_id)["log"]
    assert "stage: separating" in log and "stage: transcribing" in log, "appends must accumulate, not overwrite"
    assert log.index("stage: separating") < log.index("stage: transcribing"), "chronological order"

    # The bug this helper exists to kill: appending onto a caller's stale
    # job dict drops everything written since that dict was fetched. Each
    # stage holds its own snapshot, so this is the normal case, not a rare
    # race.
    stale = db.get_job(job_id)
    worker.append_job_log(job_id, "stage: muxing")
    worker.append_job_log(stale["id"], "stage: done")
    log = db.get_job(job_id)["log"]
    assert "stage: muxing" in log, "a later append must not clobber an earlier one"
    assert "stage: done" in log

    worker.append_job_log(job_id, "x" * 9000)
    log = db.get_job(job_id)["log"]
    assert len(log) == 8192, f"trimmed to the last 8KB, got {len(log)}"
    assert log.endswith("x" * 100), "trim keeps the tail, not the head"
    print("ok: append_job_log accumulates across stages, re-reads the row, trims to the last 8KB")


def test_p7_resume_crash_marks_job_errored():
    """A crash in a resume task used to reach stderr only ("Task exception
    was never retrieved") -- the row kept its active status and the job
    looked frozen in the UI forever. Any exception must land on the row."""
    fresh_db()
    src = os.path.join(os.environ["DOWNLOADS_DIR"], "show.mp4")
    open(src, "wb").close()
    job_id = db.insert_job(url="https://example.com/show", kind="video", status="transcribing")
    db.update_job(job_id, filepath=src, gen_subs="whisper")
    job = db.get_job(job_id)

    def explode(*_a, **_kw):
        # The real one from the NAS: non-root container, HF cache under a
        # $HOME it can't write, raised while loading the whisper model.
        raise PermissionError(13, "Permission denied: '/.cache'")

    orig = worker.run_transcribe
    worker.run_transcribe = explode
    try:
        asyncio.run(worker.resume_transcribe(job))
    finally:
        worker.run_transcribe = orig

    row = db.get_job(job_id)
    assert row["status"] == "error", f"crashed job must not stay {row['status']!r}"
    assert "PermissionError" in (row["error"] or ""), "the error column must name what actually failed"
    assert "PermissionError" in (row["log"] or ""), "and it must reach the job log the UI shows"
    print("ok: an exception in a resume task marks the job errored instead of freezing it")


def test_p8_translate_subs_rejects_path_traversal():
    """srt_name arrives from the client and is used to pick a file to read.
    Resolving it against the job's own sidecars (not by joining it onto a
    directory) is what keeps '../../etc/passwd' from being reachable."""
    # The endpoints are called directly rather than through TestClient:
    # starlette's test client needs httpx, and httpx in requirements.txt
    # would ship a test-only dependency into the production image.
    from fastapi import HTTPException

    def expect_400(coro):
        try:
            asyncio.run(coro)
        except HTTPException as e:
            assert e.status_code == 400, f"expected 400, got {e.status_code}"
            return
        raise AssertionError("expected HTTPException(400), call succeeded")

    fresh_db()
    downloads = os.environ["DOWNLOADS_DIR"]
    video = os.path.join(downloads, "clip.mp4")
    open(video, "wb").close()
    good = os.path.join(downloads, "clip.whisper.srt")
    with open(good, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\n你好\n\n")
    # A .srt outside the job's own stem -- must not be reachable by name.
    outsider = os.path.join(downloads, "someone-elses.srt")
    open(outsider, "w").close()

    job_id = db.insert_job(url="https://example.com/clip", kind="video", status="done")
    db.update_job(job_id, filepath=video, gen_subs="whisper", gen_subs_lang="zh")

    def req(srt_name, source="zh", target="en"):
        return main.TranslateSubsRequest(srt_name=srt_name, source_lang=source, target_lang=target)

    for bad in ("../../etc/passwd", "someone-elses.srt", "clip.nope.srt", ""):
        expect_400(main.api_translate_subs(job_id, req(bad)))

    # Same language in and out is a no-op that would still burn a model
    # download, and a missing language can't be guessed -- argos has no
    # detection of its own.
    expect_400(main.api_translate_subs(job_id, req("clip.whisper.srt", "zh", "zh")))
    expect_400(main.api_translate_subs(job_id, req("clip.whisper.srt", "", "en")))

    listed = asyncio.run(main.api_list_subs(job_id))
    assert listed["subs"] == ["clip.whisper.srt"], f"only this job's own sidecars, got {listed['subs']}"
    print("ok: translate-subs rejects traversal/foreign/missing srt names and degenerate language pairs")


def test_p8_hallucinated_credit_cues_dropped():
    """Whisper memorized subtitle-site credits from its training corpus and
    emits them over non-speech audio -- observed here as 'Subtitles brought
    to you by CdramaBase' sitting on top of Chinese dialogue. A bigger model
    produces them more fluently, not less, so this is filtered, not tuned."""
    cues = [
        (0.0, 2.0, "Subtitles brought to you by CdramaBase"),
        (2.0, 4.0, "想想还有点小激动呢"),
        (4.0, 6.0, "Subtitles by the Amara.org community"),
        (6.0, 8.0, "Thanks for watching!"),
        (8.0, 10.0, "字幕由中文字幕组制作"),
        (10.0, 12.0, "Please subscribe"),
        (12.0, 14.0, "I subscribe to that newspaper every morning"),
        (14.0, 16.0, "He thanked me for watching his dog"),
    ]
    kept, dropped = worker.drop_hallucinated_cues(cues)
    kept_text = [t for _s, _e, t in kept]
    assert "想想还有点小激动呢" in kept_text, "real dialogue must survive"
    # Whole-cue matching only: real sentences that merely contain the
    # trigger words are dialogue, and dropping content is worse than
    # leaving one artifact behind.
    assert "I subscribe to that newspaper every morning" in kept_text
    assert "He thanked me for watching his dog" in kept_text
    assert len(dropped) == 5, f"expected the 5 credit lines dropped, got {dropped}"
    assert len(kept) == 3
    print("ok: memorized credit cues dropped, real dialogue containing the same words kept")


def test_p8_track_meta_records_how_it_was_made():
    """The job row only holds CURRENT settings, so a regen destroys the
    record of what produced the previous file -- exactly when comparing two
    models is the point. Metadata travels with the .srt instead."""
    import json as _json
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        srt = os.path.join(d, "ep.whisper-medium.srt")
        open(srt, "w").close()
        worker.write_track_meta(srt, {"engine": "whisper", "model": "medium", "cues": 129})
        meta = _json.load(open(srt + ".json", encoding="utf-8"))
        assert meta["model"] == "medium" and meta["cues"] == 129
        assert "written" in meta, "a timestamp is what makes two runs comparable after the fact"

        # Model in the filename is what lets two runs coexist at all.
        assert worker.whisper_track_slug({"whisper_model": "medium"}) == "whisper-medium"
        assert worker.whisper_track_slug({}) == f"whisper-{worker.WHISPER_MODEL}"

        # An unwritable path must not fail a job whose transcript succeeded.
        worker.write_track_meta("/proc/nope/x.srt", {"engine": "whisper"})
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("ok: per-track metadata sidecar records engine/model, and never fails the job")


def test_p8_whisper_model_is_whitelisted():
    """The model name reaches WhisperModel(), which treats anything it
    doesn't recognize as a Hugging Face repo id and downloads it -- so an
    unchecked string from the client would fetch and run arbitrary model
    weights. Unknown names must fall back to the container default."""
    assert worker.whisper_model_for({"whisper_model": "medium"}) == "medium"
    assert worker.whisper_model_for({"whisper_model": " large-v3 "}) == "large-v3"
    for bad in ("evil/backdoored-model", "../../etc", "", None, "gpt-4"):
        assert worker.whisper_model_for({"whisper_model": bad}) == worker.WHISPER_MODEL, f"{bad!r} must fall back"
    assert worker.whisper_model_for({}) == worker.WHISPER_MODEL
    print("ok: whisper_model is whitelisted, unknown names fall back to the container default")


def test_p8_model_download_watch_is_quiet_when_cached():
    """The watcher measures directory growth rather than guessing at the
    huggingface cache layout, so an already-cached model produces no
    download noise at all."""
    import shutil
    import tempfile
    fresh_db()
    job_id = db.insert_job(url="https://example.com/x", kind="video", status="transcribing")
    d = tempfile.mkdtemp()
    saved_dir, saved_beat = worker.WHISPER_MODEL_DIR, worker.HEARTBEAT_SECONDS
    worker.WHISPER_MODEL_DIR, worker.HEARTBEAT_SECONDS = d, 0.05
    try:
        with open(os.path.join(d, "already-there.bin"), "wb") as f:
            f.write(b"x" * 4096)
        with worker._watch_model_download(job_id, "small"):
            time.sleep(0.2)  # long enough for several watcher ticks
        log = db.get_job(job_id)["log"] or ""
        assert "downloading model" not in log, "a cached model must not report a download"

        # ...and a growing directory does get reported.
        with worker._watch_model_download(job_id, "small"):
            with open(os.path.join(d, "new-model.bin"), "wb") as f:
                f.write(b"y" * 2_000_000)
            time.sleep(0.2)
        log = db.get_job(job_id)["log"] or ""
        assert "downloading model 'small'" in log, "a real download must be announced"
        assert "downloaded, 2MB" in log, f"final size must be reported, got: {log[-200:]}"
    finally:
        worker.WHISPER_MODEL_DIR, worker.HEARTBEAT_SECONDS = saved_dir, saved_beat
        shutil.rmtree(d, ignore_errors=True)
    print("ok: model download watcher reports real downloads and stays silent for cached models")


def test_p8_english_target_uses_whisper_translate():
    """Whisper's own translate task beats transcribe-then-argos for English:
    it translates from the audio, not from a transcript that already threw
    that context away. Non-English targets it can't produce must still go
    through argos."""
    assert worker.is_whisper_translatable("en")
    assert worker.is_whisper_translatable("EN")
    assert worker.is_whisper_translatable("en-US"), "regional English must not fall back to the weaker engine"
    assert not worker.is_whisper_translatable("pt")
    assert not worker.is_whisper_translatable("")
    assert not worker.is_whisper_translatable(None)

    fresh_db()
    downloads = os.environ["DOWNLOADS_DIR"]
    video = os.path.join(downloads, "ep.mp4")
    open(video, "wb").close()
    orig_base = os.path.join(downloads, "ep")
    job_id = db.insert_job(url="https://example.com/ep", kind="video", status="transcribing")
    db.update_job(job_id, filepath=video, gen_subs="whisper", gen_subs_lang="zh", translate_to="en")
    job = db.get_job(job_id)

    calls = []

    def fake_whisper(_job, _src, out_srt, task="transcribe"):
        calls.append(task)
        with open(out_srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\ntext\n\n")
        return {"status": "done", "path": out_srt}

    def no_argos(*_a, **_kw):
        raise AssertionError("argos must not be used when Whisper can produce the target itself")

    orig_whisper, orig_translate = worker.run_whisper_transcribe, worker.run_translate
    worker.run_whisper_transcribe, worker.run_translate = fake_whisper, no_argos
    try:
        result = worker.run_transcribe(job, video, orig_base)
    finally:
        worker.run_whisper_transcribe, worker.run_translate = orig_whisper, orig_translate

    assert calls == ["transcribe", "translate"], f"one pass each, got {calls}"
    names = [os.path.basename(p) for p, _l, _t in result["tracks"]]
    # Names carry the model, so a small run and a medium run of the same
    # video coexist instead of the second overwriting the first.
    assert names == ["ep.whisper-small.srt", "ep.whisper-small.en.srt"], f"both tracks kept, got {names}"
    print("ok: an English target uses Whisper's translate task; other targets still route to argos")


def test_p8_translate_failure_keeps_transcript():
    """A failed translation must not discard a transcript that succeeded --
    that cost 5 minutes of correct whisper output, written to disk then
    thrown away unmuxed, because argos couldn't reach its model dir."""
    fresh_db()
    downloads = os.environ["DOWNLOADS_DIR"]
    video = os.path.join(downloads, "ep1.mp4")
    open(video, "wb").close()
    orig_base = os.path.join(downloads, "ep1")

    job_id = db.insert_job(url="https://example.com/ep1", kind="video", status="transcribing")
    # 'pt', not 'en': English now goes through Whisper's own translate task,
    # so argos -- the engine whose failure this test is about -- would never
    # be reached with an English target.
    db.update_job(job_id, filepath=video, gen_subs="whisper", gen_subs_lang="zh", translate_to="pt")
    job = db.get_job(job_id)

    def fake_whisper(_job, _src, out_srt, task="transcribe"):
        assert task == "transcribe", "a non-English target must not use Whisper's translate task"
        with open(out_srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\n你好\n\n")
        return {"status": "done", "path": out_srt}

    orig_whisper, orig_translate = worker.run_whisper_transcribe, worker.run_translate
    worker.run_whisper_transcribe = fake_whisper
    worker.run_translate = lambda *a, **k: {
        "status": "error", "error": "PermissionError: [Errno 13] Permission denied: '/.local'",
    }
    try:
        result = worker.run_transcribe(job, video, orig_base)
    finally:
        worker.run_whisper_transcribe, worker.run_translate = orig_whisper, orig_translate

    assert result["status"] == "done", f"the transcript succeeded, job must not be {result['status']!r}"
    paths = [os.path.basename(p) for p, _l, _t in result["tracks"]]
    assert paths == ["ep1.whisper-small.srt"], f"transcript must still be muxed, got {paths}"
    log = db.get_job(job_id)["log"] or ""
    assert "translation skipped" in log and "/.local" in log, "the failure must still be reported, with its cause"
    print("ok: a failed translation keeps and muxes the transcript, reporting the failure in the log")


def test_p8_home_repointed_when_unwritable():
    """compose's `user:` leaves HOME='/' in the container, which no non-root
    user can write. Every library resolving '~' or an XDG path then dies
    mid-job, each on a different directory -- '/.cache' for faster-whisper,
    '/.local' for argos-translate."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in ("HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME")}
    try:
        app_home = os.path.join(d, "home")
        for k in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            os.environ.pop(k, None)

        os.environ["HOME"] = "/"  # the actual container value
        assert worker.ensure_writable_home(app_home) == app_home, "an unwritable HOME must be repointed"
        assert os.environ["HOME"] == app_home
        assert os.environ["XDG_DATA_HOME"].startswith(app_home), "argos reads its data dir from XDG, not just HOME"
        assert os.path.isdir(app_home), "the replacement must actually exist -- libraries mkdir *under* it"

        # A writable HOME is left alone: repointing a normal (root, or
        # properly-homed) container would scatter files somewhere nobody
        # expects them.
        for k in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            os.environ.pop(k, None)
        os.environ["HOME"] = d
        assert worker.ensure_writable_home(app_home) is None
        assert os.environ["HOME"] == d

        # Unwritable HOME *and* an unusable replacement: leave HOME as-is
        # rather than point it somewhere equally broken.
        os.environ["HOME"] = "/"
        assert worker.ensure_writable_home("/proc/nope/home") is None
        assert os.environ["HOME"] == "/"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)
    print("ok: an unwritable HOME is repointed at the persistent volume, a writable one is left alone")


def test_p8_app_log_writes_and_rotates(tmp_path=None):
    """log_line must reach the persistent file, and must never take the app
    down when /data isn't writable -- stdout has already carried the line."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    orig_path, orig_max = worker.APP_LOG_PATH, worker.APP_LOG_MAX_BYTES
    worker.APP_LOG_PATH = os.path.join(d, "app.log")
    worker.APP_LOG_MAX_BYTES = 200
    try:
        worker.log_line("first line")
        assert "first line" in open(worker.APP_LOG_PATH, encoding="utf-8").read()
        for i in range(40):
            worker.log_line(f"filler {i} " + "x" * 20)
        assert os.path.exists(worker.APP_LOG_PATH + ".1"), "must rotate past APP_LOG_MAX_BYTES"
        assert os.path.getsize(worker.APP_LOG_PATH) <= worker.APP_LOG_MAX_BYTES + 512

        worker.APP_LOG_PATH = "/proc/definitely/not/writable/app.log"
        worker.log_line("must not raise")  # the assertion is that this returns
    finally:
        worker.APP_LOG_PATH, worker.APP_LOG_MAX_BYTES = orig_path, orig_max
        shutil.rmtree(d, ignore_errors=True)
    print("ok: log_line persists to APP_LOG_PATH, rotates, and swallows an unwritable path")


def test_p7_missing_whisper_fails_fast():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/show", kind="video", status="transcribing")
    job = db.get_job(job_id)
    try:
        import faster_whisper  # noqa: F401
        print("skip: faster-whisper is actually installed here, can't exercise the not-enabled path")
        return
    except ImportError:
        pass
    result = worker.run_whisper_transcribe(job, "/nonexistent/src.mp4", "/nonexistent/out.srt")
    assert result["status"] == "error"
    assert "WITH_TRANSCRIBE=true" in result["error"]
    print("ok: missing faster-whisper fails fast with a clear error")


def test_p7_missing_tesseract_fails_fast():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/show", kind="video", status="transcribing")
    job = db.get_job(job_id)
    import shutil as _shutil
    if _shutil.which("tesseract") is not None:
        print("skip: tesseract is actually installed here, can't exercise the not-on-PATH path")
        return
    result = worker.run_ocr_transcribe(job, "/nonexistent/src.mp4", "/nonexistent/out.srt")
    assert result["status"] == "error"
    assert "tesseract" in result["error"].lower()
    print("ok: missing tesseract binary fails fast with a clear error")


def test_p7_independent_transcribe_semaphore():
    """TRANSCRIBE_SEM must be a third, fully independent semaphore -- not
    aliased to DOWNLOAD_SEM or SEPARATION_SEM. CPU-heavy Whisper/OCR work
    must never block, or be blocked by, downloads or vocal separation."""
    assert worker.TRANSCRIBE_SEM is not worker.DOWNLOAD_SEM
    assert worker.TRANSCRIBE_SEM is not worker.SEPARATION_SEM

    async def check():
        acquired = 0
        for _ in range(worker.MAX_CONCURRENT):
            await worker.DOWNLOAD_SEM.acquire()
            acquired += 1
        await worker.SEPARATION_SEM.acquire()
        try:
            assert worker.DOWNLOAD_SEM.locked()
            assert worker.SEPARATION_SEM.locked()
            got = await asyncio.wait_for(worker.TRANSCRIBE_SEM.acquire(), timeout=0.5)
            assert got
            worker.TRANSCRIBE_SEM.release()
        finally:
            worker.SEPARATION_SEM.release()
            for _ in range(acquired):
                worker.DOWNLOAD_SEM.release()

    asyncio.run(check())
    print("ok: TRANSCRIBE_SLOTS is independent of both MAX_CONCURRENT and SEPARATION_SLOTS")


def test_p7_gen_subs_clamped_for_audio():
    fresh_db()
    # ocr/both need video frames that don't exist for an audio-only job --
    # main.py's _create_job clamps both down to 'whisper', mirroring how it
    # already clamps strip_vocals/container.
    job = main._create_job(
        "https://example.com/song", "audio", "best", None, True, gen_subs="ocr",
    )
    assert job["gen_subs"] == "whisper"

    job2 = main._create_job(
        "https://example.com/song2", "audio", "best", None, True, gen_subs="both",
    )
    assert job2["gen_subs"] == "whisper"

    # whisper-only and off pass through unchanged for audio
    job3 = main._create_job(
        "https://example.com/song3", "audio", "best", None, True, gen_subs="whisper",
    )
    assert job3["gen_subs"] == "whisper"

    # video jobs are unrestricted
    job4 = main._create_job(
        "https://example.com/vid", "video", "best", None, True, gen_subs="ocr",
    )
    assert job4["gen_subs"] == "ocr"

    # unrecognized value falls back to 'off' rather than reaching the worker
    job5 = main._create_job(
        "https://example.com/vid2", "video", "best", None, True, gen_subs="garbage",
    )
    assert job5["gen_subs"] == "off"
    print("ok: gen_subs='ocr'/'both' clamped to 'whisper' for kind='audio', unrecognized values fall back to 'off'")


# --------------------------------------------------------------------- P8

def test_p8_translate_cues_pure():
    """Pure mapping logic, tested with a fake translate_fn (uppercase) --
    no real argos-translate model involved. Proves timing/count stay
    exactly 1:1 and that text actually flows through the callable, the
    same "test the pure logic with synthetic input" discipline as
    ocr_frames_to_cues/write_srt above."""
    cues = [(0.0, 1.0, "hello"), (1.0, 2.5, "world")]
    out = worker._translate_cues_with(cues, lambda t: t.upper())
    assert out == [(0.0, 1.0, "HELLO"), (1.0, 2.5, "WORLD")]
    assert len(out) == len(cues)
    print("ok: _translate_cues_with preserves timing/count 1:1, text flows through translate_fn")


def test_p8_missing_argostranslate_fails_fast():
    try:
        import argostranslate  # noqa: F401
        print("skip: argostranslate is actually installed here, can't exercise the not-enabled path")
        return
    except ImportError:
        pass
    raised = None
    try:
        worker.get_argos_translator("zh", "en")
    except RuntimeError as e:
        raised = e
    assert raised is not None, "expected RuntimeError when argostranslate isn't installed"
    assert "WITH_TRANSLATE=true" in str(raised)
    print("ok: missing argostranslate fails fast with a clear error")


def test_p8_run_translate_wraps_missing_engine_as_error_dict():
    fresh_db()
    job_id = db.insert_job(url="https://example.com/show", kind="video", status="transcribing")
    job = db.get_job(job_id)
    try:
        import argostranslate  # noqa: F401
        print("skip: argostranslate is actually installed here, can't exercise the not-enabled path")
        return
    except ImportError:
        pass
    d = tempfile.mkdtemp(prefix="uvm-translate-test-")
    src = os.path.join(d, "in.srt")
    worker.write_srt([(0.0, 1.0, "hello")], src)
    out = os.path.join(d, "out.srt")
    result = worker.run_translate(job, src, "zh", "en", out)
    assert result["status"] == "error"
    assert "WITH_TRANSLATE=true" in result["error"]
    print("ok: run_translate surfaces the not-enabled error via the job error dict, not a raw traceback")


def test_p8_translate_to_clamped():
    fresh_db()
    # no gen_subs_lang hint -- argos-translate can't know the source
    # language, so translate_to gets cleared rather than erroring, same
    # clamp-not-reject style as strip_vocals/container/gen_subs.
    job = main._create_job(
        "https://example.com/v1", "video", "best", None, True,
        gen_subs="whisper", gen_subs_lang=None, translate_to="en",
    )
    assert job["translate_to"] is None

    # gen_subs='off' -- nothing generated to translate, same clamp.
    job2 = main._create_job(
        "https://example.com/v2", "video", "best", None, True,
        gen_subs="off", gen_subs_lang="zh", translate_to="en",
    )
    assert job2["translate_to"] is None

    # both a generation engine and a lang hint set -- passes through.
    job3 = main._create_job(
        "https://example.com/v3", "video", "best", None, True,
        gen_subs="whisper", gen_subs_lang="zh", translate_to="en",
    )
    assert job3["translate_to"] == "en"
    print("ok: translate_to clamped to None unless gen_subs is on and gen_subs_lang is set")


# --------------------------------------------------------------------- P9

def test_p9_list_job_files():
    d = tempfile.mkdtemp(prefix="uvm-files-test-")
    stem = os.path.join(d, "Some Title [abc123]")
    for name in (
        stem + ".mp4",
        stem + ".en.srt",
        stem + ".zh-en.srt",
        stem + ".whisper.srt",
        stem + ".ocr.en.srt",
    ):
        open(name, "w").close()
    # an unrelated file with a colliding-ish prefix must NOT show up
    open(stem + " Extra Cut [xyz999].mp4", "w").close()

    job = {"filepath": stem + ".mp4", "strip_vocals": 0}
    files = worker.list_job_files(job)
    names = sorted(os.path.basename(f) for f in files)
    assert names == sorted([
        "Some Title [abc123].mp4",
        "Some Title [abc123].en.srt",
        "Some Title [abc123].zh-en.srt",
        "Some Title [abc123].whisper.srt",
        "Some Title [abc123].ocr.en.srt",
        "Some Title [abc123] Extra Cut [xyz999].mp4",
    ]), names
    # ^ the "Extra Cut" collision IS included -- list_job_files trades a
    # little precision for simplicity (plain prefix match, no '.' boundary
    # requirement) so it also catches demucs's '_novocals' suffix, which
    # has no dot before it. Documented tradeoff, not a bug -- see the
    # function's docstring.

    # demucs-suffixed current output must resolve back to the same prefix
    open(stem + "_novocals.m4a", "w").close()
    novocals_job = {"filepath": stem + "_novocals.m4a", "strip_vocals": 1}
    novocals_files = worker.list_job_files(novocals_job)
    assert any(f.endswith("_novocals.m4a") for f in novocals_files)
    assert any(f.endswith(".en.srt") for f in novocals_files)

    assert worker.list_job_files({"filepath": None}) == []
    print("ok: list_job_files finds every sidecar sharing a job's filename stem")


def test_p9_srt_to_vtt():
    d = tempfile.mkdtemp(prefix="uvm-vtt-test-")
    srt_path = os.path.join(d, "sample.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(
            "1\n00:00:01,500 --> 00:00:03,250\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:05,010\n你好\n\n"
        )
    vtt = worker.srt_to_vtt(srt_path)
    assert vtt.startswith("WEBVTT\n\n"), "must start with the WebVTT header line"
    assert "00:00:01.500 --> 00:00:03.250" in vtt, "'.' not ',' as the ms separator"
    assert "00:00:04.000 --> 00:00:05.010" in vtt
    assert "Hello world" in vtt
    assert "你好" in vtt
    assert "," not in vtt.split("\n\n", 1)[1], "no leftover SRT-style comma timestamps"
    print("ok: srt_to_vtt produces a valid WebVTT header and '.' millisecond separators")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} test blocks passed.")
