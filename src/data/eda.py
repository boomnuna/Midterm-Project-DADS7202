"""
src/data/eda.py

Covers the assignment's required "Dataset description / EDA" section:
  - per-class image counts (+ imbalance flag)
  - image size distribution
  - color vs grayscale count
  - brightness distribution per class

Deliberately reads images directly from disk (not through the DataLoader)
since EDA should reflect the RAW collected images, before any resize/
normalize/augment is applied.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class EDAAnalyzer:
    def __init__(self, data_root: Path, class_names: list):
        self.data_root = Path(data_root)
        self.class_names = class_names

    def _iter_class_images(self, class_name: str):
        """
        Works with either folder layout (pre-split train/val/test, or
        flat per-class) by searching recursively for a subfolder with
        this class name.
        """
        for path in self.data_root.rglob(class_name):
            if path.is_dir():
                yield from (p for p in path.iterdir() if p.suffix.lower() in
                            {".jpg", ".jpeg", ".png", ".bmp"})

    def analyze(self) -> dict:
        report = {
            "counts": {},
            "sizes": defaultdict(list),      # class -> list of (w, h)
            "is_grayscale": defaultdict(list),  # class -> list of bool
            "brightness": defaultdict(list),  # class -> list of mean pixel brightness
        }

        for class_name in self.class_names:
            paths = list(self._iter_class_images(class_name))
            report["counts"][class_name] = len(paths)

            for p in paths:
                try:
                    with Image.open(p) as img:
                        report["sizes"][class_name].append(img.size)  # (width, height)
                        is_gray = img.mode in ("L", "1") or (
                            img.mode == "RGB" and np.array(img).std(axis=2).mean() < 2.0
                        )
                        report["is_grayscale"][class_name].append(is_gray)

                        gray_arr = np.array(img.convert("L"), dtype=np.float32)
                        report["brightness"][class_name].append(gray_arr.mean())
                except Exception:
                    continue  # corrupt file — dedupe_and_clean.py should have caught these already

        return report

    def print_summary(self, report: dict):
        print("=" * 60)
        print("DATASET EDA SUMMARY")
        print("=" * 60)

        total = sum(report["counts"].values())
        print(f"\nTotal images: {total}")
        print("\nPer-class counts:")
        for name, count in report["counts"].items():
            pct = 100 * count / total if total else 0
            print(f"  {name:20s}: {count:5d}  ({pct:.1f}%)")

        counts = list(report["counts"].values())
        if counts and max(counts) > 0:
            imbalance_ratio = max(counts) / max(min(counts), 1)
            print(f"\nImbalance ratio (max/min class count): {imbalance_ratio:.2f}")
            if imbalance_ratio > 1.5:
                print("  -> FLAGGED as imbalanced (ratio > 1.5). "
                      "Report your handling strategy (see config.imbalance_strategy).")
            else:
                print("  -> Roughly balanced.")

        print("\nGrayscale vs color:")
        for name in self.class_names:
            flags = report["is_grayscale"].get(name, [])
            n_gray = sum(flags)
            print(f"  {name:20s}: {n_gray} grayscale / {len(flags) - n_gray} color")

        print("\nImage size ranges (width x height):")
        for name in self.class_names:
            sizes = report["sizes"].get(name, [])
            if sizes:
                widths = [w for w, h in sizes]
                heights = [h for w, h in sizes]
                print(f"  {name:20s}: W[{min(widths)}-{max(widths)}] H[{min(heights)}-{max(heights)}]")

        print("\nBrightness (mean grayscale pixel value, 0-255):")
        for name in self.class_names:
            b = report["brightness"].get(name, [])
            if b:
                print(f"  {name:20s}: mean={np.mean(b):.1f}  std={np.std(b):.1f}")

    def plot_summary(self, report: dict, save_path: Path = None):
        """Produces the 3 charts you'll actually want on your slides."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # 1) class counts bar chart
        names = list(report["counts"].keys())
        counts = list(report["counts"].values())
        axes[0].bar(names, counts, color="steelblue")
        axes[0].set_title("Images per class")
        axes[0].tick_params(axis="x", rotation=30)

        # 2) brightness distribution per class (boxplot)
        brightness_data = [report["brightness"].get(name, []) for name in names]
        axes[1].boxplot(brightness_data, labels=names)
        axes[1].set_title("Brightness distribution per class")
        axes[1].tick_params(axis="x", rotation=30)

        # 3) image area distribution (width * height) per class
        area_data = [[w * h for w, h in report["sizes"].get(name, [])] for name in names]
        axes[2].boxplot(area_data, labels=names)
        axes[2].set_title("Image area (px²) distribution per class")
        axes[2].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved EDA plot to {save_path}")
        plt.close(fig)
