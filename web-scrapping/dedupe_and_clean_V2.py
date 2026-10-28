"""
dedupe_and_clean.py

Run this AFTER scrape_images.py.

Search-engine crawlers commonly return:
  - the same image multiple times (from different result pages)
  - corrupted / unreadable files
  - near-duplicates (same product photo, tiny crop/resize difference)

This script:
  1. Moves files that fail to actually LOAD as usable images into
     removed_dataset/<class_name>/corrupt/ (see note below on why this
     checks img.load(), not img.verify())
  2. Moves exact duplicates (identical file hash) into
     removed_dataset/<class_name>/exact_duplicate/
  3. Handles near-duplicates (perceptually similar) according to
     MOVE_NEAR_DUPLICATES below:
       - False (default): only FLAGS them for manual review — doesn't
         move/delete anything, since two genuinely different images
         (e.g. two different poses of the same character) can still
         look similar; you eyeball the flagged pairs yourself.
       - True: actually MOVES all-but-one of each near-duplicate GROUP
         into removed_dataset/<class_name>/near_duplicate/, keeping one
         representative per group automatically. Turn this on once
         you've reviewed a few flagged batches and trust the threshold
         for your dataset.

Nothing is ever permanently deleted. If a file gets moved out by mistake,
it's sitting in removed_dataset/ and you can move it back into
raw_dataset/ yourself.

LOGGING: every moved file and every flagged/grouped near-duplicate is
written to removed_dataset/dedupe_log.csv — not just whatever scrolled
past in the console (which truncates long lists). Columns: timestamp,
class, action, file, matched_with, detail.

WHY img.load() INSTEAD OF img.verify():
  verify() only checks the file's internal structure/headers — it does
  NOT guarantee the image can actually be decoded and used. Testing
  showed a truncated JPEG (simulating an incomplete download) PASSED
  verify() but failed load() with "image file is truncated" — and this
  project's actual training code (TransformFactory, via
  Image.open(path).convert("RGB")) uses the same underlying decode path
  as load(), never verify(). A file that passes verify() but fails
  load()/convert() would crash training mid-run instead of being caught
  here.

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

# Set to True to actually move all-but-one of each near-duplicate GROUP
# out of raw_dataset/ automatically. Default False = flag-only (current
# behavior) — recommended until you've reviewed a batch of flagged pairs
# for your specific dataset and trust NEAR_DUP_THRESHOLD isn't too loose.
MOVE_NEAR_DUPLICATES = True

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


def is_loadable_detailed(path):
    """
    Returns (True, "") if the image can actually be decoded end-to-end,
    or (False, error_message) if not — matching exactly what happens
    during training (Image.open(path).convert("RGB")), not just whether
    the file's headers look structurally valid (which is all
    img.verify() checks).
    """
    try:
        with Image.open(path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


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


class UnionFind:
    """
    Groups near-duplicates TRANSITIVELY: if A~B and B~C (each pair within
    NEAR_DUP_THRESHOLD of each other) but A and C weren't directly compared
    as similar enough, all three still belong in ONE group together —
    otherwise moving "all but one per pair" independently could keep two
    or more images from what is really a single near-duplicate cluster.
    """
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def groups(self):
        result = {}
        for item in self.parent:
            root = self.find(item)
            result.setdefault(root, []).append(item)
        return result


def clean_class_folder(class_dir: str, class_name: str):
    moved_corrupt = 0
    moved_exact_dup = 0
    seen_hashes = {}
    valid_files = []

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # 1) actually-loadable check (see module docstring for why this
        # is load(), not verify())
        ok, error_detail = is_loadable_detailed(fpath)
        if not ok:
            move_out(fpath, class_name, "corrupt")
            log_event(class_name, "moved_corrupt", fpath, detail=error_detail)
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

    print(f"  moved {moved_corrupt} unloadable files, "
          f"{moved_exact_dup} exact duplicates to '{REMOVED_ROOT}/{class_name}/' — "
          f"{len(valid_files)} valid files remain in place")

    # 3) near-duplicate detection (optional, only if imagehash is installed)
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

        if not flagged_pairs:
            return

        # always log every flagged pair, regardless of MOVE_NEAR_DUPLICATES —
        # this is your record of what was CONSIDERED similar, even for
        # groups you end up keeping
        for a, b, diff in flagged_pairs:
            log_event(class_name, "flagged_near_duplicate", a, matched_with=b, detail=f"phash_distance={diff}")

        if not MOVE_NEAR_DUPLICATES:
            print(f"  {len(flagged_pairs)} near-duplicate pairs flagged for manual review "
                  f"(NOT moved — MOVE_NEAR_DUPLICATES=False — see {LOG_PATH} for the full list):")
            for a, b, diff in flagged_pairs[:20]:
                print(f"    {os.path.basename(a)}  <->  {os.path.basename(b)}  (distance={diff})")
            if len(flagged_pairs) > 20:
                print(f"    ... and {len(flagged_pairs) - 20} more (full list in {LOG_PATH})")
            return

        # ---- MOVE_NEAR_DUPLICATES=True: group transitively, keep one per group ----
        uf = UnionFind(valid_files)
        for a, b, diff in flagged_pairs:
            uf.union(a, b)

        moved_near_dup = 0
        for group in uf.groups().values():
            if len(group) < 2:
                continue
            group_sorted = sorted(group)  # deterministic: keep alphabetically first
            keep = group_sorted[0]
            for extra in group_sorted[1:]:
                move_out(extra, class_name, "near_duplicate")
                log_event(class_name, "moved_near_duplicate", extra, matched_with=keep,
                          detail=f"group_size={len(group)}")
                moved_near_dup += 1

        print(f"  moved {moved_near_dup} near-duplicates to "
              f"'{REMOVED_ROOT}/{class_name}/near_duplicate/' "
              f"(kept 1 representative per group of {len(flagged_pairs)} flagged pair(s))")


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
        print(f"Full log of every moved/flagged file (with match + reason) "
              f"is in: {LOG_PATH}")


if __name__ == "__main__":
    main()
