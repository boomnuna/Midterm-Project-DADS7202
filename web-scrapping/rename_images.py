"""
Rename all image files in a folder to gear1_0001, gear1_0002, ...

Usage:
    python rename_images.py                     # renames images in the CURRENT folder
    python rename_images.py /path/to/folder      # renames images in a specific folder
    python rename_images.py --prefix gear1 --start 1 --pad 4
    python rename_images.py --dry-run            # preview only, no changes

Tip: if you don't want to type a long path every time, just copy this
script into whichever folder you want to rename, then run it from a
terminal opened in that folder (or double-click it, depending on your
OS setup) — no path argument needed.

By default this does a safe two-pass rename (temp names first) so it
never overwrites an existing file even if names collide mid-process.
"""

import argparse
import os
import sys

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".webp", ".heic", ".heif", ".jfif",
}


def get_image_files(folder):
    files = []
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(name)
    # Sort alphabetically/naturally so the renumbering is predictable
    files.sort()
    return files


def rename_images(folder, prefix="gear1", start=1, pad=4, dry_run=False):
    files = get_image_files(folder)

    if not files:
        print(f"No image files found in: {folder}")
        return

    print(f"Found {len(files)} image file(s) in: {folder}\n")

    # Build the final name list first
    new_names = []
    for i, old_name in enumerate(files, start=start):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"{prefix}_{i:0{pad}d}{ext}"
        new_names.append(new_name)

    if dry_run:
        for old_name, new_name in zip(files, new_names):
            print(f"[DRY RUN] {old_name}  ->  {new_name}")
        print("\nNo files were changed (dry run).")
        return

    # Pass 1: rename everything to a temporary name to avoid collisions
    temp_names = []
    for old_name in files:
        temp_name = f"__tmp__{old_name}"
        os.rename(os.path.join(folder, old_name), os.path.join(folder, temp_name))
        temp_names.append(temp_name)

    # Pass 2: rename temp names to the final sequential names
    for temp_name, new_name, old_name in zip(temp_names, new_names, files):
        os.rename(os.path.join(folder, temp_name), os.path.join(folder, new_name))
        print(f"{old_name}  ->  {new_name}")

    print(f"\nDone. Renamed {len(files)} file(s).")


def main():
    parser = argparse.ArgumentParser(description="Rename all images in a folder sequentially.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=os.getcwd(),
        help="Path to the folder containing images (default: current folder)",
    )
    parser.add_argument("--prefix", default="gear1", help="Filename prefix (default: gear1)")
    parser.add_argument("--start", type=int, default=1, help="Starting number (default: 1)")
    parser.add_argument("--pad", type=int, default=4, help="Zero-padding width for the number (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Preview new names without renaming")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)

    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    print(f"Working in: {folder}\n")
    rename_images(folder, prefix=args.prefix, start=args.start, pad=args.pad, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
