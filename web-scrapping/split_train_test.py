"""
scripts/split_train_test.py

ONE-TIME preprocessing script (not run every training session): takes a
flat dataset (one folder per class) and physically COPIES files into a
train/test folder structure on disk — turning:

    data/<class_name>/*.jpg

into:

    data/train/<class_name>/*.jpg
    data/test/<class_name>/*.jpg

This gives you the "Option C" layout DataModule already supports (see
src/data/dataset.py) — a manually-fixed test set, with validation carved
out of train/ automatically at load time on every run.

WHY A SEPARATE SCRIPT INSTEAD OF DOING THIS AT LOAD TIME:
Splitting once and writing real files means the test set is IDENTICAL
across every training run, every group member, every Colab session —
nobody can accidentally get a different test set just because file
ordering changed. Re-running this script is a deliberate, visible action,
not something that happens silently as a side effect of training.

Usage:
    python -m scripts.split_train_test --input data --output data --test-fraction 0.15

    # preview the split without copying anything:
    python -m scripts.split_train_test --input data --output data --test-fraction 0.15 --dry-run
"""

import sys
import argparse
import shutil
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# list all class name 
def find_class_folders(input_root: Path) -> list:
    """Flat layout = every subdirectory of input_root is a class folder."""
    return sorted([d for d in input_root.iterdir() if d.is_dir()])

# train/test split 
def split_class_files(image_paths: list, test_fraction: float, seed: int) -> tuple:
    """Returns (train_paths, test_paths) for one class — plain random split."""
    return train_test_split(image_paths, test_size=test_fraction, random_state=seed)


def main():
    # add argument 
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the flat dataset (class folders directly inside)")
    parser.add_argument("--output", required=True,
                         help="Where to write train/ and test/ (can be the same as --input — "
                              "class folders get replaced by train/<class> and test/<class>)")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                         help="Fraction of each class held out for test (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would happen without copying any files")
    args = parser.parse_args()

    # define path 
    input_root = Path(args.input)
    output_root = Path(args.output)
    class_folders = find_class_folders(input_root)

    if not class_folders:
        print(f"No class folders found under {input_root} — check --input path.")
        return

    print(f"Found {len(class_folders)} classes: {[c.name for c in class_folders]}")

    manifest_rows = []
    train_root = output_root / "train"
    test_root = output_root / "test"

    for class_dir in class_folders:
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS] # loop through each img 
        if not image_paths:
            print(f"  {class_dir.name}: no images found, skipping")
            continue

        train_paths, test_paths = split_class_files(image_paths, args.test_fraction, args.seed) # train/test splitting 

        print(f"  {class_dir.name:20s}: {len(image_paths):4d} total -> "
              f"{len(train_paths):4d} train / {len(test_paths):4d} test")

        if args.dry_run:
            continue

        # make directory
        class_train_dir = train_root / class_dir.name
        class_test_dir = test_root / class_dir.name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        # copy and move train dataset to destination path
        for src in train_paths:
            dst = class_train_dir / src.name
            shutil.copy2(src, dst)
            manifest_rows.append({"class": class_dir.name, "split": "train",
                                   "source_path": str(src), "dest_path": str(dst)})
            
        # copy and move test dataset to destination path
        for src in test_paths:
            dst = class_test_dir / src.name
            shutil.copy2(src, dst)
            manifest_rows.append({"class": class_dir.name, "split": "test",
                                   "source_path": str(src), "dest_path": str(dst)})

    if args.dry_run:
        print("\nDry run — no files were copied. Remove --dry-run to actually split.")
        return

    # log moving file 
    manifest_path = output_root / "split_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "split", "source_path", "dest_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Files copied to:\n  {train_root}\n  {test_root}")
    print(f"Manifest (which file went where) saved to: {manifest_path}")
    print("\nNext step: point config.data_root at the output folder, e.g.")
    print(f'  data_root: Path = Path("{output_root}")')
    print("DataModule will auto-detect the train/test-only layout and carve "
          "validation out of train/ automatically — see config.val_fraction.")


if __name__ == "__main__":
    main()
