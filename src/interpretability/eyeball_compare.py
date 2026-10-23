"""
scripts/eyeball_compare.py   (Eyeball analysis, guideline ข้อ 7)

เทียบรูปเดียวกันระหว่าง 3 โมเดลต่อ backbone:
   [รูป]  |  Pre-trained (ImageNet, ยังไม่เทรน) top-1  |  Freeze all (median seed)  |  Selective Unfreeze (median seed)
และสร้าง hard_cases.png = รูปทั้งหมดที่ Selective Unfreeze (median seed) ทายผิดอย่างน้อย 1 backbone
ไว้ตรวจด้วยตาว่าคำบรรยายภาพในรายงาน (ผลถักไหมพรม โมเดลสามมิติ ภาพมืด ตัวละคร โลโก้) ตรงกับภาพจริงไหม
ไม่เทรนใหม่ ไม่แก้ผลเดิม (อ่านอย่างเดียว)

วางไฟล์ใน scripts/ ข้างๆ gradcam_correct_wrong.py แล้วรันจาก root ของโปรเจกต์ (ใส่ root อย่างน้อย 1 อัน):
    python -m scripts.eyeball_compare --freeze-root <path> --unfreeze-root <path>   # เทียบ 3 โมเดล
    python -m scripts.eyeball_compare --freeze-root <path>                          # Pre-trained + Freeze all เท่านั้น
    python -m scripts.eyeball_compare --unfreeze-root <path>                        # Pre-trained + Selective เท่านั้น
(แต่ละ root ต้องมี experiment_results.pkl, logs/, checkpoints/ เหมือนที่ใช้กับ gradcam_correct_wrong.py)

ตัวเลือก:
    --backbones vgg16 resnet50     ค่าเริ่มต้น: ทั้ง 4
    --images ope_ope_0093 mera_mera_0140     ระบุรูปเอง (ชื่อไฟล์ไม่ต้องมีนามสกุล) ไม่ระบุ = เลือกอัตโนมัติ
    --per-group 4                  จำนวนรูปต่อกลุ่มที่เลือกอัตโนมัติ
    --out PATH                     ค่าเริ่มต้น: <root ที่ใส่>/eyeball_compare (ถ้าใส่สองอัน ใช้ unfreeze-root)
กลุ่มที่เลือกอัตโนมัติต่อ backbone (ถ้าใส่ root เดียว: ใช้กลุ่ม W=ผิด และ C=ถูก คลาสละ 1 รูป):
    A) Freeze ผิด -> Selective ถูก   (หลักฐานว่า Selective Unfreeze ช่วยได้)
    B) ผิดทั้งสองวิธี                 (ภาพยากจริง)
    C) ถูกทั้งสองวิธี คลาสละ 1 รูป    (ไว้เทียบ)
"""
import sys, csv, json, pickle, random, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory
from src.models.classifier import CNNClassifier
from src.utils.checkpoint import CheckpointManager

PRIMARY = "f1_macro"
# ตรงกับ backbone_factory.py (IMAGENET1K_V1)
PRETRAINED = {
    "resnet50":           (models.resnet50,           models.ResNet50_Weights.IMAGENET1K_V1),
    "vgg16":              (models.vgg16,              models.VGG16_Weights.IMAGENET1K_V1),
    "efficientnet_b0":    (models.efficientnet_b0,    models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
}


def median_run(runs):
    ranked = sorted(runs, key=lambda r: (r[PRIMARY], r["seed"]))
    return ranked[(len(ranked) - 1) // 2]


def test_samples(dm):
    ds = dm.test_dataset
    under = ds.dataset if hasattr(ds, "dataset") else ds
    if hasattr(ds, "indices"):
        return [under.samples[i] for i in ds.indices]
    return list(under.samples)


def load_trained(root, backbone, dm, config, device):
    root = Path(root)
    with open(root / "experiment_results.pkl", "rb") as f:
        run = median_run(pickle.load(f)[backbone])
    seed = run["seed"]
    run_id = f"{backbone}_seed{seed}"
    mgr = CheckpointManager(root / "checkpoints", run_id)
    try:
        with open(root / "logs" / f"{run_id}_config.json") as f:
            rc = json.load(f)
        hidden, dropout = rc.get("head_hidden_dim", config.head_hidden_dim), rc.get("head_dropout", config.head_dropout)
    except (OSError, json.JSONDecodeError):
        hidden, dropout = config.head_hidden_dim, config.head_dropout
        print(f"  WARNING {run_id}: config json unreadable, using config.py head size")
    model = CNNClassifier(backbone_name=backbone, num_classes=dm.num_classes,
                          head_hidden_dim=hidden, head_dropout=dropout, pretrained=False)
    ckpt = mgr.load_best(map_location="cpu") if mgr.has_best_checkpoint() else mgr.load_latest(map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), seed


@torch.no_grad()
def predict(model, x):
    p = torch.softmax(model(x), dim=1)[0]
    c = int(p.argmax())
    return c, float(p[c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-root", default=None)
    ap.add_argument("--unfreeze-root", default=None)
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.freeze_root or args.unfreeze_root):
        sys.exit("ต้องใส่อย่างน้อย 1 อัน: --freeze-root และ/หรือ --unfreeze-root")

    methods = []   # (key, label, root)
    if args.freeze_root:
        methods.append(("freeze", "Freeze all", args.freeze_root))
    if args.unfreeze_root:
        methods.append(("selective", "Selective Unfreeze", args.unfreeze_root))
    keys = [m[0] for m in methods]

    config = Config()
    backbones = args.backbones or config.backbone_names
    out = Path(args.out) if args.out else Path(args.unfreeze_root or args.freeze_root) / "eyeball_compare"
    out.mkdir(parents=True, exist_ok=True)

    dm = DataModule(config)
    classes = dm.class_names
    samples = test_samples(dm)
    eval_tf = TransformFactory(config.image_size).eval_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(42)

    pred, seeds = {}, {}   # (backbone, method) -> {stem: (pred_idx, conf, true_idx, path)}
    for b in backbones:
        for key, label, root in methods:
            m, seed = load_trained(root, b, dm, config, device)
            seeds[(b, key)] = seed
            d = {}
            for path, lab in samples:
                x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                d[Path(path).stem] = (*predict(m, x), lab, path)
            pred[(b, key)] = d
            del m
        print(b + ": median seed " + ", ".join(f"{k}={seeds[(b,k)]}" for k in keys))

    wrong = lambda D, s: D[s][0] != D[s][2]
    rows_csv, hard = [], {}
    main_key = keys[-1]   # วิธีที่ใช้หา hard cases: selective ถ้ามี ไม่งั้น freeze

    for b in backbones:
        ctor, w = PRETRAINED[b]
        cats = w.meta["categories"]
        pre = ctor(weights=w).to(device).eval()
        D = {k: pred[(b, k)] for k in keys}
        ref = D[keys[0]]

        if args.images:
            chosen = [(s, "manual") for s in args.images if s in ref]
        elif len(keys) == 2:
            F, S = D["freeze"], D["selective"]
            fixed = [s for s in F if wrong(F, s) and not wrong(S, s)]
            both = [s for s in F if wrong(F, s) and wrong(S, s)]
            ok = [s for s in F if not wrong(F, s) and not wrong(S, s)]
            rng.shuffle(fixed); rng.shuffle(both)
            chosen = [(s, "A: Freeze wrong -> Selective correct") for s in fixed[:args.per_group]]
            chosen += [(s, "B: wrong in both") for s in both[:args.per_group]]
            for ci in range(len(classes)):
                pool = [s for s in ok if F[s][2] == ci]
                if pool:
                    chosen.append((rng.choice(pool), "C: correct in both"))
        else:
            R = D[keys[0]]
            bad = [s for s in R if wrong(R, s)]
            ok = [s for s in R if not wrong(R, s)]
            rng.shuffle(bad)
            chosen = [(s, "W: wrong") for s in bad[:2 * args.per_group]]
            for ci in range(len(classes)):
                pool = [s for s in ok if R[s][2] == ci]
                if pool:
                    chosen.append((rng.choice(pool), "C: correct"))
        if not chosen:
            print(f"{b}: ไม่มีรูปให้เทียบ"); continue

        ncol = 2 + len(keys)
        fig, axs = plt.subplots(len(chosen), ncol, figsize=(2.75 * ncol, 2.9 * len(chosen)), squeeze=False)
        for r, (stem, grp) in enumerate(chosen):
            path, true = ref[stem][3], ref[stem][2]
            img = Image.open(path).convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pp = torch.softmax(pre(x), dim=1)[0]
            pi = int(pp.argmax())
            axs[r][0].imshow(img); axs[r][0].set_title(f"{stem}\ntrue: {classes[true]}", fontsize=8)
            axs[r][1].text(0.5, 0.5, f"Pre-trained\n(ImageNet, no training)\n\n{cats[pi]}\n({float(pp[pi]):.2f})",
                           ha="center", va="center", fontsize=9)
            row = [b, stem, classes[true], grp, cats[pi], f"{float(pp[pi]):.4f}"]
            for ci_, (key, label, _) in enumerate(methods):
                pi_, cf = D[key][stem][0], D[key][stem][1]
                good = pi_ == true
                axs[r][2 + ci_].text(0.5, 0.5, f"{label}\n\n{classes[pi_]}\n({cf:.2f})\n{'CORRECT' if good else 'WRONG'}",
                                     ha="center", va="center", fontsize=9, color="green" if good else "red", weight="bold")
                row += [classes[pi_], f"{cf:.4f}"]
            for a in axs[r][1:]:
                a.set_xticks([]); a.set_yticks([])
            axs[r][0].axis("off")
            axs[r][1].set_title(grp, fontsize=7, loc="left")
            rows_csv.append(row)
        fig.suptitle(f"{b}  (median seed: " + ", ".join(f"{k}={seeds[(b,k)]}" for k in keys) + ")", fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f"eyeball_{b}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", out / f"eyeball_{b}.png")

        M = D[main_key]
        for s in M:
            if wrong(M, s):
                hard.setdefault(s, (M[s][3], M[s][2], []))[2].append((b, classes[M[s][0]]))

    with open(out / "eyeball_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.writer(f)
        w_.writerow(["backbone", "image", "true", "group", "pretrained_top1_imagenet", "pretrained_conf"]
                    + [c for k in keys for c in (f"{k}_pred", f"{k}_conf")])
        w_.writerows(rows_csv)

    if hard:
        label_main = dict((k, l) for k, l, _ in methods)[main_key]
        items = sorted(hard.items(), key=lambda kv: -len(kv[1][2]))
        ncols = 4; nrows = int(np.ceil(len(items) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows), squeeze=False)
        for a in axs.ravel(): a.axis("off")
        for a, (stem, (path, true, who)) in zip(axs.ravel(), items):
            a.imshow(Image.open(path).convert("RGB"))
            wr = ", ".join(f"{bb[:6]}->{pc}" for bb, pc in who)
            a.set_title(f"{stem}\ntrue: {classes[true]}\nwrong in {len(who)}/{len(backbones)}: {wr}", fontsize=8)
        fig.suptitle(f"Images wrong under {label_main} (median seed) - check descriptions by eye", fontsize=12)
        plt.tight_layout()
        plt.savefig(out / "hard_cases.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", out / "hard_cases.png", "| unique hard images:", len(items))
        for stem, (path, true, who) in items:
            print(f"  {stem:22s} true={classes[true]:10s} wrong in {len(who)}: {who}")


if __name__ == "__main__":
    main()
