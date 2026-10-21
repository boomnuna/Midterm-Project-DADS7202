"""
src/interpretability/gradcam_report_figs.py

สร้างรูป GradCAM สำหรับรายงาน 
ใช้ median seed ของแต่ละ backbone, อ่านผลเดิมอย่างเดียว (ไม่เทรนใหม่)

ผลลัพธ์ (ต่อ 1 วิธีเทรน):
  fig_correct_summary.png
      แถวบน = Original (1 รูปต่อคลาส), แถวถัดไป = GradCAM ของแต่ละ backbone, คอลัมน์ = คลาส
      ถ้า backbone ใดทายรูปเหล่านี้ผิด จะแจ้งเตือนใน console และใต้ภาพเป็นสีแดง
  fig_wrong.png
      รูปที่ทายผิด "5 รูปแรก" ของแต่ละ backbone (แต่ละ backbone ผิดคนละรูปกันได้)
  selection.csv   รูปที่ใช้จริง + ผลทำนายของทุก backbone

วิธีรัน (ตั้ง config.output_root ให้ชี้โฟลเดอร์ผลของวิธีนั้นก่อน เหมือนตอนรัน gradcam_correct_wrong):
    python -m src.interpretability.gradcam_report_figs --tag selective_unfreeze
    python -m src.interpretability.gradcam_report_figs --tag freeze_all

ตัวเลือก:
    --n-wrong 5      จำนวนรูปที่ผิดต่อ backbone
    --out PATH       ค่าเริ่มต้น: <output_root>/gradcam_report/<tag>
"""
import sys
import csv
import json
import pickle
import argparse
from pathlib import Path
_here = Path(__file__).resolve()
for up in (_here.parent.parent, _here.parent.parent.parent):
    sys.path.append(str(up))

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

# รูปที่ต้องการ (ตัด prefix gradcam_ ออกแล้ว) — คลาสละ 5 รูป
# รูปที่ใช้ในรูป "ทายถูก": 1 รูปต่อคลาส (ตัด prefix gradcam_ ออกแล้ว)
SUMMARY = {"gomu_gomu": "gomu_gomu_0089", "mera_mera": "mera_mera_0136",
           "ope_ope": "ope_ope_0030", "yami_yami": "yami_yami_0074"}
NICE = {"resnet50": "ResNet-50", "vgg16": "VGG-16", "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v3_small": "MobileNetV3-Small"}
COL_OK, COL_BAD = "#1a7f1a", "#c0392b"


def median_run(runs):
    ranked = sorted(runs, key=lambda r: (r[PRIMARY_METRIC], r["seed"]))
    return ranked[(len(ranked) - 1) // 2]


def test_samples(data_module):
    ds = data_module.test_dataset
    under = ds.dataset if hasattr(ds, "dataset") else ds
    if hasattr(ds, "indices"):
        return [under.samples[i] for i in ds.indices]
    return list(under.samples)


def load_model(backbone, run_id, config, num_classes, device):
    ckpt_mgr = CheckpointManager(config.output_root / "checkpoints", run_id)
    if not ckpt_mgr.has_checkpoint():
        raise FileNotFoundError(f"no checkpoint for {run_id}")
    try:
        with open(config.output_root / "logs" / f"{run_id}_config.json") as f:
            rc = json.load(f)
        hidden = rc.get("head_hidden_dim", config.head_hidden_dim)
        dropout = rc.get("head_dropout", config.head_dropout)
    except (OSError, json.JSONDecodeError):
        hidden, dropout = config.head_hidden_dim, config.head_dropout
        print("  WARNING: run config json unreadable, using config.py head size")
    model = CNNClassifier(backbone_name=backbone, num_classes=num_classes,
                          head_hidden_dim=hidden, head_dropout=dropout, pretrained=False)
    ckpt = ckpt_mgr.load_best(map_location="cpu") if ckpt_mgr.has_best_checkpoint() \
        else ckpt_mgr.load_latest(map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def explain(model, cam, path, raw_tf, eval_tf, device):
    """-> (overlay, raw, pred, conf)  อธิบายคลาสที่โมเดลทายเอง"""
    img = Image.open(path).convert("RGB")
    raw_np = raw_tf(img).permute(1, 2, 0).numpy()
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    heat, pred = cam.generate(x)
    return GradCAM.overlay_heatmap(raw_np, heat), raw_np, int(pred), float(prob[pred])


def draw(cells, backbones, class_names, save_path, title, ncols=5, true_each_cell=False,
         with_original=False):
    """cells[backbone] = list ของ dict(raw, overlay, true, pred, conf, stem) ยาวไม่เกิน ncols
    ค่าเริ่มต้น: 1 แถวต่อ backbone (เฉพาะ GradCAM), true label อยู่บนรูปแถวบนสุด
    true_each_cell=True: ใส่ true label บนทุกรูป (ใช้กับรูปที่ผิด ซึ่งแต่ละ backbone ผิดคนละรูป)
    with_original=True : เพิ่มแถว Original ของแต่ละ backbone"""
    per = 2 if with_original else 1
    nrows = len(backbones) * per
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.15 * ncols + 1.0, (2.15 + (0.35 if true_each_cell else 0.2)) * nrows + 0.5),
                            squeeze=False,
                            gridspec_kw={"hspace": 0.5 if true_each_cell else 0.28, "wspace": 0.06,
                                         "left": 0.07, "right": 0.99, "top": 0.88 if true_each_cell else 0.90, "bottom": 0.04})
    for ax in axs.ravel():
        ax.axis("off")
    for bi, b in enumerate(backbones):
        r_cam = bi * per + (1 if with_original else 0)
        for ci, c in enumerate(cells[b][:ncols]):
            if with_original:
                axs[bi * per][ci].imshow(c["raw"])
            ax = axs[r_cam][ci]
            ax.imshow(c["overlay"])
            ok = c["true"] == c["pred"]
            ax.text(0.5, -0.03, f"Pred: {class_names[c['pred']]} ({c['conf']:.0%})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                    color=COL_OK if ok else COL_BAD)
            if true_each_cell or bi == 0:
                top = axs[bi * per][ci]
                top.text(0.5, 1.03, f"True: {class_names[c['true']]}\n{c['stem']}",
                         transform=top.transAxes, ha="center", va="bottom", fontsize=7)
        axs[bi * per][0].text(-0.08, 0.5 if not with_original else 0.0,
                              f"[{NICE.get(b, b)}]", transform=axs[bi * per][0].transAxes,
                              rotation=90, ha="right", va="center", fontsize=9, fontweight="bold",
                              color="#c0392b")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_summary(originals, cells, backbones, class_names, save_path, title):
    """แถวบนสุด = Original (true label อยู่บนรูป) / แถวถัดไป = GradCAM ของแต่ละ backbone / คอลัมน์ = คลาส"""
    ncols, nrows = len(originals), len(backbones) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 1.0, 2.3 * nrows + 0.6), squeeze=False,
                            gridspec_kw={"hspace": 0.3, "wspace": 0.06, "left": 0.1, "right": 0.99,
                                         "top": 0.88, "bottom": 0.04})
    for ax in axs.ravel():
        ax.axis("off")
    for ci, o in enumerate(originals):
        axs[0][ci].imshow(o["raw"])
        axs[0][ci].text(0.5, 1.03, f"True: {class_names[o['true']]}\n{o['stem']}", transform=axs[0][ci].transAxes,
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    axs[0][0].text(-0.08, 0.5, "Original", transform=axs[0][0].transAxes, rotation=90,
                   ha="right", va="center", fontsize=9, fontweight="bold", color="#2a7ab9")
    for bi, b in enumerate(backbones):
        for ci, c in enumerate(cells[b]):
            ax = axs[bi + 1][ci]
            ax.imshow(c["overlay"])
            ax.text(0.5, -0.03, f"Pred: {class_names[c['pred']]} ({c['conf']:.0%})", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color=COL_OK if c["true"] == c["pred"] else COL_BAD)
        axs[bi + 1][0].text(-0.08, 0.5, f"[{NICE.get(b, b)}]", transform=axs[bi + 1][0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=9, fontweight="bold", color="#c0392b")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-wrong", type=int, default=5)
    ap.add_argument("--with-original", action="store_true", help="เพิ่มแถว Original (ค่าเริ่มต้น: ไม่มี เพื่อให้รูปสั้น)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = Config()
    backbones = list(config.backbone_names)
    out_dir = Path(args.out) if args.out else config.output_root / "gradcam_report" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    data_module = DataModule(config)
    class_names = data_module.class_names
    samples = test_samples(data_module)
    counts = {n: sum(1 for _, l in samples if class_names[l] == n) for n in class_names}
    print("Test images per class:", counts, "| total:", len(samples))
    if any(v == 0 for v in counts.values()):
        sys.exit("STOP: test folder is missing a class -> labels would shift.")

    by_stem = {Path(p).stem: (p, l) for p, l in samples}

    with open(config.output_root / "experiment_results.pkl", "rb") as f:
        results = pickle.load(f)

    tf = TransformFactory(config.image_size)
    raw_tf, eval_tf = tf.raw_transform(), tf.eval_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- pass 1: ทำนายทุกรูปของทุก backbone (median seed) ----
    models, cams, preds, seeds = {}, {}, {}, {}
    for b in backbones:
        run = median_run(results[b])
        seeds[b] = run["seed"]
        try:
            m = load_model(b, f"{b}_seed{run['seed']}", config, data_module.num_classes, device)
        except (FileNotFoundError, RuntimeError, KeyError, OSError) as e:
            print(f"  skip {b}: {e}")
            continue
        models[b], cams[b] = m, GradCAM(m)
        preds[b] = {}
        with torch.no_grad():
            for p, l in samples:
                x = eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                preds[b][p] = int(m(x).argmax(1).item())
        n_wrong = sum(1 for p, l in samples if preds[b][p] != l)
        acc = 1 - n_wrong / len(samples)
        print(f"{b}: median seed {run['seed']}  wrong {n_wrong}/{len(samples)}  "
              f"(accuracy {acc:.4f}, saved {run['accuracy']:.4f})")
        if abs(acc - run["accuracy"]) > 0.005:
            sys.exit(f"STOP: accuracy for {b} differs from the saved result -> checkpoint/test set mismatch.")
    backbones = [b for b in backbones if b in models]

    def all_correct(path, label):
        return all(preds[b][path] == label for b in backbones)
    # ---- รูปที่ทายถูก: 1 รูปต่อคลาส, แถวบนสุด = Original, แถวถัดไป = GradCAM ของแต่ละ backbone ----
    sel_rows = []
    summary_stems = [SUMMARY[c] for c in class_names if c in SUMMARY]
    problems = []
    cells = {b: [] for b in backbones}
    originals = []
    for s in summary_stems:
        if s not in by_stem:
            sys.exit(f"STOP: {s} ไม่อยู่ใน test set")
        p, l = by_stem[s]
        originals.append(dict(raw=None, true=l, stem=s))
        for b in backbones:
            ov, raw, pr, cf = explain(models[b], cams[b], p, raw_tf, eval_tf, device)
            cells[b].append(dict(raw=raw, overlay=ov, true=l, pred=pr, conf=cf, stem=s))
            originals[-1]["raw"] = raw
            sel_rows.append(["summary", b, seeds[b], s, class_names[l], class_names[pr], f"{cf:.4f}"])
            if pr != l:
                problems.append((s, NICE.get(b, b), class_names[pr]))
    if problems:
        print("\n!! แจ้งเตือน: รูปที่ขอบาง backbone ทายผิด (ใต้ภาพเป็นสีแดง):")
        for s, b, pr in problems:
            print(f"   {s}: {b} ทายเป็น {pr}")
    else:
        print("\nรูปที่ขอทั้ง 4 รูป: ทุก backbone ทายถูกทั้งหมด")
    draw_summary(originals, cells, backbones, class_names, out_dir / "fig_correct_summary.png",
                 f"GradCAM — {args.tag} — correct predictions")

    wrong_cells = {b: [] for b in backbones}
    for b in backbones:
        wrong = [(p, l) for p, l in samples if preds[b][p] != l][:args.n_wrong]
        for p, l in wrong:
            ov, raw, pr, cf = explain(models[b], cams[b], p, raw_tf, eval_tf, device)
            s = Path(p).stem
            wrong_cells[b].append(dict(raw=raw, overlay=ov, true=l, pred=pr, conf=cf, stem=s))
            sel_rows.append(["wrong", b, seeds[b], s, class_names[l], class_names[pr], f"{cf:.4f}"])
        if not wrong:
            print(f"  {b}: ไม่มีรูปที่ทายผิดเลย")
    draw(wrong_cells, backbones, class_names, out_dir / "fig_wrong.png",
         f"GradCAM — {args.tag} — misclassified (first {args.n_wrong} per backbone)", ncols=args.n_wrong,
         true_each_cell=True, with_original=args.with_original)

    with open(out_dir / "selection.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["set", "backbone", "seed", "image", "true", "pred", "confidence"])
        w.writerows(sel_rows)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()