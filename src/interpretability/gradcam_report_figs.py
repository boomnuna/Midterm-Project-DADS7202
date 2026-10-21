"""
src/interpretability/gradcam_report_figs.py

สร้างรูป GradCAM สำหรับรายงาน (รูปแบบเดียวกับตัวอย่าง: ต่อ backbone มี 2 แถว = Original / GradCAM)
ใช้ median seed ของแต่ละ backbone, อ่านผลเดิมอย่างเดียว (ไม่เทรนใหม่)

ผลลัพธ์ (ต่อ 1 วิธีเทรน):
  fig_correct_gomu_gomu.png, fig_correct_mera_mera.png, fig_correct_ope_ope.png, fig_correct_yami_yami.png
      คลาสละ 5 รูป x 4 backbone  (รูปที่ทุก backbone ทายถูก)
  fig_wrong.png
      รูปที่ทายผิด "5 รูปแรก" ของแต่ละ backbone (แต่ละ backbone ผิดคนละรูปกันได้)
  selection.csv   รูปที่ใช้จริง + ผลทำนายของทุก backbone

วิธีรัน (ตั้ง config.output_root ให้ชี้โฟลเดอร์ผลของวิธีนั้นก่อน เหมือนตอนรัน gradcam_correct_wrong):
    python -m src.interpretability.gradcam_report_figs --tag selective_unfreeze
    python -m src.interpretability.gradcam_report_figs --tag freeze_all

ตัวเลือก:
    --no-replace     ไม่สลับรูปที่บาง backbone ทายผิด (ใช้รูปที่ขอตามเดิม แต่ชื่อใต้ภาพเป็นสีแดงถ้าทายผิด)
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
WANTED = {
    "gomu_gomu": ["gomu_gomu_0001", "gomu_gomu_0014", "gomu_gomu_0018", "gomu_gomu_0089", "gomu_gomu_0540"],
    "mera_mera": ["mera_mera_0068", "mera_mera_0136", "mera_mera_0181", "mera_mera_0248", "mera_mera_0294"],
    "ope_ope":   ["ope_ope_0030", "ope_ope_0107", "ope_ope_0147", "ope_ope_0286", "ope_ope_0373"],
    "yami_yami": ["yami_yami_0074", "yami_yami_0081", "yami_yami_0094", "yami_yami_0132", "yami_yami_0185"],
}
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


def draw(cells, backbones, class_names, save_path, title, ncols=5):
    """cells[backbone] = list ของ dict(raw, overlay, true, pred, conf, stem) ยาวไม่เกิน ncols"""
    nrows = len(backbones) * 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 1.4, 2.35 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for bi, b in enumerate(backbones):
        r0, r1 = bi * 2, bi * 2 + 1
        for ci, c in enumerate(cells[b][:ncols]):
            axs[r0][ci].imshow(c["raw"])
            axs[r0][ci].set_title(f"True: {class_names[c['true']]}\n{c['stem']}", fontsize=7)
            axs[r1][ci].imshow(c["overlay"])
            ok = c["true"] == c["pred"]
            axs[r1][ci].text(0.5, -0.04, f"Pred: {class_names[c['pred']]} ({c['conf']:.0%})",
                             transform=axs[r1][ci].transAxes, ha="center", va="top", fontsize=7.5,
                             color=COL_OK if ok else COL_BAD)
        for r, lab in ((r0, "Original"), (r1, "GradCAM")):
            axs[r][0].text(-0.08, 0.5, f"[{NICE.get(b, b)}]\n{lab}", transform=axs[r][0].transAxes,
                           rotation=90, ha="right", va="center", fontsize=9, fontweight="bold",
                           color="#2a7ab9" if r == r0 else "#c0392b")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0.03, 0, 1, 0.98))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-replace", action="store_true")
    ap.add_argument("--n-wrong", type=int, default=5)
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
    missing = [s for v in WANTED.values() for s in v if s not in by_stem]
    if missing:
        print("WARNING: ไม่พบรูปเหล่านี้ใน test set (ข้าม/สลับให้):", missing)

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

    # ---- เลือกรูปที่ทายถูก (สลับถ้ามี backbone ทายผิด) ----
    chosen, swaps, still_wrong = {}, [], []
    for ci, cname in enumerate(class_names):
        want = [s for s in WANTED.get(cname, []) if s in by_stem]
        pool = [Path(p).stem for p, l in samples if l == ci and Path(p).stem not in WANTED.get(cname, [])
                and all_correct(p, l)]
        final = []
        for s in WANTED.get(cname, []):
            if s not in by_stem:
                repl = pool.pop(0) if pool else None
                if repl:
                    final.append(repl); swaps.append((s, repl, "not in test set"))
                continue
            p, l = by_stem[s]
            if all_correct(p, l) or args.no_replace:
                final.append(s)
                if not all_correct(p, l):
                    still_wrong.append(s)
            else:
                bad = [NICE.get(b, b) for b in backbones if preds[b][p] != l]
                repl = pool.pop(0) if pool else None
                if repl:
                    final.append(repl); swaps.append((s, repl, "wrong in: " + ", ".join(bad)))
                else:
                    final.append(s); still_wrong.append(s)
        chosen[cname] = final

    if swaps:
        print("\nสลับรูป (ขอ -> ใช้จริง):")
        for a, b_, why in swaps:
            print(f"  {a} -> {b_}   [{why}]")
    if still_wrong:
        print("\nรูปที่ยังมี backbone ทายผิด (ชื่อใต้ภาพจะเป็นสีแดง):", still_wrong)

    # ---- pass 2: GradCAM + วาดรูป ----
    sel_rows = []
    for ci, cname in enumerate(class_names):
        cells = {b: [] for b in backbones}
        for s in chosen[cname]:
            p, l = by_stem[s]
            for b in backbones:
                ov, raw, pr, cf = explain(models[b], cams[b], p, raw_tf, eval_tf, device)
                cells[b].append(dict(raw=raw, overlay=ov, true=l, pred=pr, conf=cf, stem=s))
                sel_rows.append(["correct", b, seeds[b], s, class_names[l], class_names[pr], f"{cf:.4f}"])
        draw(cells, backbones, class_names, out_dir / f"fig_correct_{cname}.png",
             f"GradCAM — {args.tag} — correct predictions: {cname}")

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
         f"GradCAM — {args.tag} — misclassified (first {args.n_wrong} per backbone)", ncols=args.n_wrong)

    with open(out_dir / "selection.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["set", "backbone", "seed", "image", "true", "pred", "confidence"])
        w.writerows(sel_rows)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()