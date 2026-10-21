"""
scripts/gradcam_correct_wrong.py

GradCAM ของ "ทุกภาพใน test set" ของ median seed แต่ละ backbone แล้วแยกเป็น
  - correct/<true_class>/...      ภาพที่ทายถูก
  - wrong/<true>_as_<pred>/...    ภาพที่ทายผิด
พร้อม contact sheet สำหรับใส่รายงาน
ไม่เทรนใหม่ ไม่แก้ experiment_log.csv / experiment_results.pkl (อ่านอย่างเดียว)

วางไฟล์นี้ในโฟลเดอร์ scripts/ ข้างๆ run_all_experiments.py แล้วรันจาก root ของโปรเจกต์ (ทีละวิธีเทรน):
    python -m scripts.gradcam_correct_wrong --tag freeze_all
    python -m scripts.gradcam_correct_wrong --tag unfreeze_last_n
ต้องตั้ง config.output_root ให้ชี้โฟลเดอร์ผลของวิธีนั้น (ที่มี experiment_results.pkl, logs/, checkpoints/)

ตัวเลือก:
    --backbones resnet50 vgg16     เฉพาะบาง backbone
    --sheet-per-class 30           จำนวนภาพที่ถูกต้องต่อคลาสใน contact sheet (สุ่ม seed คงที่)
    --sheet-seed 42
    --out PATH                     ค่าเริ่มต้น: <output_root>/gradcam_correct_wrong/<tag>
"""
import sys
import json
import pickle
import random
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory
from src.models.classifier import CNNClassifier
from src.interpretability.gradcam import GradCAM
from src.utils.checkpoint import CheckpointManager

PRIMARY_METRIC = "f1_macro"


def median_run(runs):
    """เรียง macro-F1 จากน้อยไปมาก (เท่ากันเรียงตามเลข seed) แล้วเอาลำดับที่ 5 จาก 10"""
    ranked = sorted(runs, key=lambda r: (r[PRIMARY_METRIC], r["seed"]))
    return ranked[(len(ranked) - 1) // 2]


def test_samples(data_module):
    ds = data_module.test_dataset
    under = ds.dataset if hasattr(ds, "dataset") else ds
    if hasattr(ds, "indices"):
        return [under.samples[i] for i in ds.indices]
    return list(under.samples)


def contact_sheet(items, path, title, ncols=6):
    """items = [(overlay_np, caption)]"""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.45 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (img, cap) in zip(axs.ravel(), items):
        ax.imshow(img)
        ax.set_title(cap, fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="ชื่อวิธีเทรน เช่น freeze_all หรือ unfreeze_last_n")
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--sheet-per-class", type=int, default=30)
    ap.add_argument("--sheet-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = Config()
    if args.backbones:
        config.backbone_names = args.backbones
    out_root = Path(args.out) if args.out else config.output_root / "gradcam_correct_wrong" / args.tag

    data_module = DataModule(config)
    class_names = data_module.class_names
    samples = test_samples(data_module)

    # ---- ตรวจ test set ก่อน: ถ้าขาดคลาส ป้ายจะเลื่อนและทุกอย่างผิด ----
    counts = {n: sum(1 for _, l in samples if class_names[l] == n) for n in class_names}
    print("Test images per class:", counts, "| total:", len(samples))
    if any(v == 0 for v in counts.values()):
        sys.exit("STOP: test folder is missing a class -> labels would shift. Fix the dataset first.")

    # ---- อ่านผลเดิม (read-only) เพื่อหา median seed ----
    with open(config.output_root / "experiment_results.pkl", "rb") as f:
        results = pickle.load(f)

    raw_tf = TransformFactory(config.image_size).raw_transform()
    eval_tf = TransformFactory(config.image_size).eval_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.sheet_seed)

    for backbone in config.backbone_names:
        run = median_run(results[backbone])
        seed = run["seed"]
        run_id = f"{backbone}_seed{seed}"
        print(f"\n=== {backbone}: median seed {seed} ({PRIMARY_METRIC}={run[PRIMARY_METRIC]:.4f}, "
              f"accuracy={run['accuracy']:.4f}) ===")

        ckpt_mgr = CheckpointManager(config.output_root / "checkpoints", run_id)
        if not ckpt_mgr.has_checkpoint():
            print(f"  WARNING: no checkpoint for {run_id} -> skipped")
            continue
        try:
            with open(config.output_root / "logs" / f"{run_id}_config.json") as f:
                rc = json.load(f)
            hidden, dropout = rc.get("head_hidden_dim", config.head_hidden_dim), rc.get("head_dropout", config.head_dropout)
        except (OSError, json.JSONDecodeError):
            hidden, dropout = config.head_hidden_dim, config.head_dropout
            print("  WARNING: run config json unreadable, using config.py head size")

        model = CNNClassifier(backbone_name=backbone, num_classes=data_module.num_classes,
                              head_hidden_dim=hidden, head_dropout=dropout, pretrained=False)
        try:
            ckpt = ckpt_mgr.load_best(map_location="cpu") if ckpt_mgr.has_best_checkpoint() \
                else ckpt_mgr.load_latest(map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
        except (RuntimeError, KeyError, OSError) as e:
            print(f"  WARNING: could not load checkpoint ({e}) -> skipped")
            continue
        model.to(device).eval()
        cam = GradCAM(model)

        bdir = out_root / backbone
        correct_items = {n: [] for n in class_names}   # (overlay, caption)
        wrong_items = []
        n_correct = 0

        for path, label in samples:
            true_name = class_names[label]
            img = Image.open(path).convert("RGB")
            raw_np = raw_tf(img).permute(1, 2, 0).numpy()
            heatmap, pred = cam.generate(eval_tf(img).unsqueeze(0))      # ใช้คลาสที่โมเดลทายเอง
            overlay = GradCAM.overlay_heatmap(raw_np, heatmap)
            pred_name = class_names[pred]
            stem = Path(path).stem

            if pred == label:
                n_correct += 1
                d = bdir / "correct" / true_name
                correct_items[true_name].append((overlay, f"{true_name}\n{stem[:18]}"))
            else:
                d = bdir / "wrong" / f"{true_name}_as_{pred_name}"
                wrong_items.append((overlay, f"true: {true_name}\npred: {pred_name}"))
            d.mkdir(parents=True, exist_ok=True)
            plt.imsave(d / f"gradcam_{stem}.png", overlay)

        acc = n_correct / len(samples)
        print(f"  correct {n_correct} / wrong {len(samples) - n_correct}  (accuracy {acc:.4f})")
        if abs(acc - run["accuracy"]) > 0.005:
            print(f"  !! WARNING: accuracy here ({acc:.4f}) differs from the saved result "
                  f"({run['accuracy']:.4f}) -> checkpoint/test set may not match this run. Do not use these figures.")

        # ---- contact sheets ----
        contact_sheet(wrong_items, bdir / "sheet_wrong_all.png",
                      f"{backbone} (seed {seed}) – misclassified test images ({len(wrong_items)})")
        for n in class_names:
            items = correct_items[n]
            pick = rng.sample(items, min(args.sheet_per_class, len(items)))
            contact_sheet(pick, bdir / f"sheet_correct_{n}.png",
                          f"{backbone} (seed {seed}) – correct, true class {n} "
                          f"({len(pick)} of {len(items)} shown)")

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()