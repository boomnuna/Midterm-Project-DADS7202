"""
dedupe_and_clean.py

Run this AFTER scrape_images.py.

Search-engine crawlers commonly return:
  - the same image multiple times (from different result pages)
  - corrupted / unreadable files
  - near-duplicates (same product photo, tiny crop/resize difference)

This script:
  1. Moves files that fail to open as valid images into
     removed_dataset/<class_name>/corrupt/
  2. Moves exact duplicates (identical file hash) into
     removed_dataset/<class_name>/exact_duplicate/
  3. Flags near-duplicates (perceptually similar) for manual review —
     it does NOT move these automatically, since two genuinely different
     armor classes can look similar; you should eyeball the flagged pairs
     yourself and move/delete whichever copy you decide is wrong.

Nothing is permanently deleted. If a file gets moved out by mistake
(e.g. a corrupt-detection false positive), it's sitting in
removed_dataset/ and you can move it back into raw_dataset/ yourself.

LOGGING: every moved file and every flagged near-duplicate pair is written
to removed_dataset/dedupe_log.csv, so you have a full record to review —
not just whatever scrolled past in the console (which truncates long
lists). Columns: timestamp, class, action, file, matched_with, detail.

Usage:
    python dedupe_and_clean.py
"""

import os
import csv
import shutil
import hashlib
from datetime import datetime
from PIL import Image

try:
    import imagehash  # optional, for near-duplicate detection
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

DATASET_ROOT = "raw_dataset"
REMOVED_ROOT = "removed_dataset"   # moved-out files land here instead of being deleted
LOG_PATH = os.path.join(REMOVED_ROOT, "dedupe_log.csv")
NEAR_DUP_THRESHOLD = 5  # lower = stricter match (imagehash Hamming distance)

_log_rows = []  # collected in memory, written out once at the end


def log_event(class_name: str, action: str, file: str, matched_with: str = "", detail: str = ""):
    _log_rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "class": class_name,
        "action": action,
        "file": os.path.basename(file),
        "matched_with": os.path.basename(matched_with) if matched_with else "",
        "detail": detail,
    })


def write_log():
    os.makedirs(REMOVED_ROOT, exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "class", "action", "file", "matched_with", "detail"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(_log_rows)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def move_out(fpath: str, class_name: str, reason: str):
    """Move a file into removed_dataset/<class_name>/<reason>/ instead of deleting it."""
    dest_dir = os.path.join(REMOVED_ROOT, class_name, reason)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(fpath))

    # avoid overwriting if a file with the same name was already moved out
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(dest_path)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        dest_path = f"{base}_{i}{ext}"

    shutil.move(fpath, dest_path)


def clean_class_folder(class_dir: str, class_name: str):
    moved_corrupt = 0
    moved_exact_dup = 0
    seen_hashes = {}
    valid_files = []

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # 1) validity check
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as e:
            move_out(fpath, class_name, "corrupt")
            log_event(class_name, "moved_corrupt", fpath, detail=str(e))
            moved_corrupt += 1
            continue

        # 2) exact duplicate check
        h = file_hash(fpath)
        if h in seen_hashes:
            original = seen_hashes[h]
            move_out(fpath, class_name, "exact_duplicate")
            log_event(class_name, "moved_exact_duplicate", fpath, matched_with=original, detail=f"md5={h}")
            moved_exact_dup += 1
            continue
        seen_hashes[h] = fpath
        valid_files.append(fpath)

    print(f"  moved {moved_corrupt} corrupt files, "
          f"{moved_exact_dup} exact duplicates to '{REMOVED_ROOT}/{class_name}/' — "
          f"{len(valid_files)} valid files remain in place")

    # 3) near-duplicate flagging (optional, only if imagehash is installed)
    if HAS_IMAGEHASH and len(valid_files) > 1:
        hashes = {}
        for fpath in valid_files:
            try:
                h = imagehash.phash(Image.open(fpath))
                hashes[fpath] = h
            except Exception:
                continue

        flagged_pairs = []
        files = list(hashes.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                diff = hashes[files[i]] - hashes[files[j]]
                if diff <= NEAR_DUP_THRESHOLD:
                    flagged_pairs.append((files[i], files[j], diff))

        if flagged_pairs:
            print(f"  {len(flagged_pairs)} near-duplicate pairs flagged for manual review "
                  f"(NOT moved automatically — see {LOG_PATH} for the full list):")
            for a, b, diff in flagged_pairs[:20]:  # don't spam if there are hundreds
                print(f"    {os.path.basename(a)}  <->  {os.path.basename(b)}  (distance={diff})")
            if len(flagged_pairs) > 20:
                print(f"    ... and {len(flagged_pairs) - 20} more (full list in {LOG_PATH})")

            # log every flagged pair, not just the first 20 shown on screen
            for a, b, diff in flagged_pairs:
                log_event(class_name, "flagged_near_duplicate", a, matched_with=b, detail=f"phash_distance={diff}")


def main():
    if not os.path.isdir(DATASET_ROOT):
        print(f"'{DATASET_ROOT}' not found — run scrape_images.py first.")
        return

    if not HAS_IMAGEHASH:
        print("Tip: install 'imagehash' (pip install imagehash) to also "
              "detect near-duplicate images, not just exact ones.\n")

    for class_name in os.listdir(DATASET_ROOT):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"Cleaning class '{class_name}'...")
        clean_class_folder(class_dir, class_name)

    write_log()

    if os.path.isdir(REMOVED_ROOT):
        print(f"\nAll moved-out files are under '{REMOVED_ROOT}/' "
              f"(nothing was deleted) — spot-check it, then delete that "
              f"folder yourself once you're confident.")
        print(f"Full log of every moved file and every flagged near-duplicate "
              f"pair (with match + reason) is in: {LOG_PATH}")


if __name__ == "__main__":
    main()
