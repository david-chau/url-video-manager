import argparse
import pysrt


def overlap_ms(a, b):
    latest_start = max(a.start.ordinal, b.start.ordinal)
    earliest_end = min(a.end.ordinal, b.end.ordinal)
    return max(0, earliest_end - latest_start)


def overlap_ratio(a, b):
    overlap = overlap_ms(a, b)

    a_duration = a.end.ordinal - a.start.ordinal
    b_duration = b.end.ordinal - b.start.ordinal

    min_duration = min(a_duration, b_duration)
    return overlap / min_duration if min_duration > 0 else 0


def clean_text(text):
    t = text.strip()

    # Remove short bracket-only labels like [Name]
    if t.startswith('[') and t.endswith(']') and len(t) < 20:
        return ""

    return t


def shift_subs(subs, offset_ms):
    for s in subs:
        s.start.ordinal += offset_ms
        s.end.ordinal += offset_ms


# 🔥 AUTO OFFSET DETECTION
def detect_offset(chi, eng, search_range_sec=180, step_ms=500):
    """
    Try different offsets and pick the one with best overlap score
    """
    best_offset = 0
    best_score = 0

    chi_sample = chi[:200]   # limit for speed
    eng_sample = eng[:200]

    for offset_ms in range(-search_range_sec * 1000,
                           search_range_sec * 1000,
                           step_ms):

        score = 0

        for c in chi_sample:
            for e in eng_sample:
                # simulate shifted EN
                fake_start = e.start.ordinal + offset_ms
                fake_end = e.end.ordinal + offset_ms

                latest_start = max(c.start.ordinal, fake_start)
                earliest_end = min(c.end.ordinal, fake_end)
                overlap = max(0, earliest_end - latest_start)

                if overlap > 0:
                    score += overlap

        if score > best_score:
            best_score = score
            best_offset = offset_ms

    return best_offset


def merge_subtitles(chi_path, eng_path, output_path,
                    reverse=False,
                    threshold=0.1,
                    auto_sync=True,
                    manual_offset_sec=None):

    chi = pysrt.open(chi_path)
    eng = pysrt.open(eng_path)

    # 🔥 AUTO SYNC
    if manual_offset_sec is not None:
        offset_ms = int(manual_offset_sec * 1000)
        print(f"[INFO] Using manual offset: {manual_offset_sec}s")

    elif auto_sync:
        print("[INFO] Detecting offset...")
        offset_ms = detect_offset(chi, eng)
        print(f"[INFO] Detected offset: {offset_ms / 1000:.2f}s")

    else:
        offset_ms = 0

    # Apply offset
    if offset_ms != 0:
        shift_subs(eng, offset_ms)

    merged = pysrt.SubRipFile()
    used_eng = set()

    for c in chi:
        matches = []

        for i, e in enumerate(eng):
            if i in used_eng:
                continue

            overlap = overlap_ms(c, e)

            if overlap > 0:
                ratio = overlap_ratio(c, e)

                if ratio >= threshold or overlap > 100:
                    t = clean_text(e.text)
                    if t:
                        matches.append(t)
                        used_eng.add(i)

        # Deduplicate while preserving order
        seen = set()
        unique_matches = []
        for m in matches:
            if m not in seen:
                unique_matches.append(m)
                seen.add(m)

        chi_text = c.text.strip()
        eng_text = "\n".join(unique_matches)

        if reverse:
            text = (eng_text + "\n" + chi_text).strip()
        else:
            text = (chi_text + "\n" + eng_text).strip()

        merged.append(pysrt.SubRipItem(
            index=len(merged) + 1,
            start=c.start,
            end=c.end,
            text=text
        ))

    merged.save(output_path, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Auto-sync + merge bilingual subtitles"
    )

    parser.add_argument("sub1", help="Chinese subtitle file")
    parser.add_argument("sub2", help="English subtitle file")

    parser.add_argument("-o", "--output", default="merged.srt")

    parser.add_argument("--reverse", action="store_true",
                        help="English above Chinese")

    parser.add_argument("--threshold", type=float, default=0.1)

    parser.add_argument("--no-auto-sync", action="store_true",
                        help="Disable auto sync")

    parser.add_argument("--offset", type=float,
                        help="Manual offset in seconds (overrides auto)")

    args = parser.parse_args()

    merge_subtitles(
        args.sub1,
        args.sub2,
        args.output,
        args.reverse,
        args.threshold,
        auto_sync=not args.no_auto_sync,
        manual_offset_sec=args.offset
    )


if __name__ == "__main__":
    main()
