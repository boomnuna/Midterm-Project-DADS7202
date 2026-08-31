# CNN Image Classifier — DADS Deep Learning Homework

PyTorch pipeline for the "train a CNN image classifier on a new,
self-collected dataset, compare 3-5 pretrained backbones with repeated
runs for mean±SD" assignment.

Currently configured with a **temporary demo dataset (chihuahua vs
cookie)** so the whole pipeline can be built and tested before the
group's real dataset topic/images are finalized. See "Switching to your
real dataset" below — it's a one-line change.

## Why OOP for this project

Three things about the assignment specifically call for it, rather than
it being OOP for its own sake:
1. **3-5 backbones must all be trained/evaluated the same way** —
   `CNNClassifier` gives every architecture (ResNet, VGG, EfficientNet,
   MobileNet...) an identical `.forward()`, `.freeze_backbone()`, and
   `.gradcam_target_layer()` interface, so `Trainer`/`Evaluator`/`GradCAM`
   don't need architecture-specific branches.
2. **Every architecture must be retrained 3-10 times with different
   seeds** — `ExperimentRunner` just loops `Trainer` objects; state
   (optimizer, scheduler, history) naturally resets each time because
   each run gets a fresh `Trainer` instance.
3. **Swapping the dataset topic later shouldn't touch training code** —
   `DataModule` is the only class that knows about file paths; everything
   downstream only sees `.train_loader()`, `.class_names`, etc.

## Project structure

```
.
├── config.py                        # ALL hyperparameters/paths — edit this, not the code
├── requirements.txt
├── configs/
│   └── wandb_sweep.yaml              # W&B Sweep definition (group hyperparameter search)
├── data/                             # (gitignored) your images go here
│   └── demo_chihuahua_vs_cookie/
├── outputs/                          # (gitignored) plots, logs, saved results land here
│   ├── logs/                         # per-run .log files + per-run metrics .json
│   └── experiment_log.csv            # one row per run, across ALL experiments ever
├── src/
│   ├── data/
│   │   ├── transforms.py             # TransformFactory — augmentation pipelines
│   │   ├── dataset.py                # DataModule — loading, splitting, class weights
│   │   └── eda.py                    # EDAAnalyzer — dataset description / EDA (section 1)
│   ├── models/
│   │   ├── backbone_factory.py       # BackboneFactory — registry of pretrained CNNs
│   │   └── classifier.py             # CNNClassifier, ClassifierHead (section 3)
│   ├── training/
│   │   ├── trainer.py                # Trainer — one full train/val run (section 4)
│   │   ├── repeated_runs.py          # ExperimentRunner — N repeats × all backbones
│   │   └── hyperparameter_tuning.py  # OptunaTuner — Optuna search, any backbone
│   ├── evaluation/
│   │   ├── metrics.py                # Evaluator — accuracy/precision/recall/F1/CM (section 5)
│   │   └── statistics.py             # StatisticalComparator — mean±SD, p-values (section 6)
│   ├── interpretability/
│   │   └── gradcam.py                # GradCAM — section 7 (mandatory)
│   └── utils/
│       ├── seed.py                   # reproducibility helper
│       ├── logger.py                 # ExperimentLogger — file + CSV logging
│       ├── checkpoint.py             # CheckpointManager — save/resume training state
│       └── colab_utils.py            # Google Drive mounting, Colab environment detection
└── scripts/                          # entry points — run these, don't import them elsewhere
    ├── run_eda.py                    # -> section 1 outputs
    ├── show_pretrained_baseline.py   # -> section 3's "why finetune" evidence
    ├── train_single.py               # quick 1-architecture debug run
    ├── run_all_experiments.py        # THE full pipeline -> sections 3, 5, 6, 7 outputs
    ├── tune_hyperparameters.py       # Optuna search entry point
    ├── wandb_sweep_train.py          # entry point called by `wandb agent` (see below)
    └── colab_notebook_setup.py       # copy-paste reference for a Colab notebook's setup cells
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset layout expected

Either of these works (`DataModule` auto-detects which one you have):

```
data/demo_chihuahua_vs_cookie/
├── train/
│   ├── chihuahua/*.jpg
│   └── cookie/*.jpg
├── val/
│   ├── chihuahua/*.jpg
│   └── cookie/*.jpg
└── test/
    ├── chihuahua/*.jpg
    └── cookie/*.jpg
```
or, simpler (let the pipeline split it for you):
```
data/demo_chihuahua_vs_cookie/
├── chihuahua/*.jpg
└── cookie/*.jpg
```

**Note:** chihuahua is an actual ImageNet class — fine for a throwaway
pipeline test, but don't forget to swap this out before your real
submission (see below), since it wouldn't satisfy the assignment's
"outside the 1000 ImageNet classes" condition.

## Switching to your real dataset

Edit **one line** in `config.py`:
```python
data_root: Path = Path("data/YOUR_REAL_DATASET_FOLDER")
```
Also update `backbone_names`, `num_repeats`, and other hyperparameters
in `config.py` as you tune them — nothing else in `src/` needs to change.

## Running the pipeline

```bash
# 1. EDA — class counts, image sizes, brightness, imbalance check
python -m scripts.run_eda

# 2. Prove the pretrained model doesn't know your classes (assignment section 3)
python -m scripts.show_pretrained_baseline

# 3. Quick single-architecture run to sanity-check hyperparameters
python -m scripts.train_single --backbone resnet50

# 4. Full experiment: all backbones × N repeats × full evaluation + GradCAM
python -m scripts.run_all_experiments
```

`run_all_experiments.py` is slow by design (it's doing exactly what the
assignment asks: N full training runs per architecture) — always sanity
check with `train_single.py` first so you're not debugging inside a long
run.

## Logging

Every training run (via `train_single.py`, `run_all_experiments.py`, or
the tuning scripts) automatically produces:

- **`outputs/logs/<run_id>.log`** — full timestamped text log for that
  one run (per-epoch metrics, warnings, early stopping messages). Open
  this when one specific run behaved oddly and you need the detail.
- **`outputs/experiment_log.csv`** — ONE ROW appended per completed run,
  across every experiment you've ever run in this project folder. Open
  this in Excel or `pandas.read_csv(...)` to compare everything at a
  glance — architecture, seed, accuracy, F1, final losses — without
  re-running anything. This is the file to hand your groupmates when
  someone asks "wait, which run had the best result again?"
- **`outputs/logs/<run_id>_metrics.json`** — full per-class metrics for
  that run (precision/recall/F1 per class, not just macro-averaged),
  for whenever the summary CSV isn't detailed enough.

This local file logging works with zero setup and zero external
dependencies — it's there even if you never touch Optuna or W&B at all.

## Hyperparameter tuning: Optuna vs W&B Sweep

Both are wired up; **honest trade-off** so you can pick deliberately
rather than by hype:

|  | Optuna (`scripts/tune_hyperparameters.py`) | W&B Sweep (`configs/wandb_sweep.yaml`) |
|---|---|---|
| Setup for solo tuning | Trivial — just run the script | Needs a free W&B account |
| **Group members sharing one search** | Needs shared *storage* (a database) — a SQLite file on a shared drive works but can hit write-lock issues with concurrent writers; a real shared DB (MySQL/Postgres) works well but is more setup | **This is what it's built for** — everyone runs `wandb agent SWEEP_ID` from their own machine, results land in one shared cloud project automatically |
| Live dashboard for the group to watch together | No (local only, unless you add `optuna-dashboard`) | Yes, built-in web UI |
| Search algorithm | TPE (Bayesian) + pruning | Bayesian / grid / random, your choice, + Hyperband early-termination |

**Recommendation matching what's in the assignment PDF:** since the
professor's note specifically highlights W&B Sweep for *"collecting the
whole group's experiment results into one project"* — use **W&B Sweep**
if your group wants to parallelize the search across members' laptops.
Use **Optuna** if one person is tuning alone on one machine, or if you'd
rather not depend on an external cloud service.

### Using Optuna
```bash
python -m scripts.tune_hyperparameters --backbone resnet50 --n-trials 20
```
Saves `outputs/best_params_resnet50.json`. Repeat per backbone. For a
group-shared study (same trials pool, multiple machines):
```bash
python -m scripts.tune_hyperparameters --backbone resnet50 \
    --storage "sqlite:///outputs/optuna_study.db" --study-name resnet50_group_search
```
(point `--storage` at a real shared DB URL instead of SQLite if multiple
people will run this at the same time from different machines)

### Using W&B Sweep
```bash
pip install wandb
wandb login                              # one-time, needs a free account

# ONE person runs this once, shares the printed SWEEP_ID with the group:
wandb sweep configs/wandb_sweep.yaml

# EVERY group member (including that person) then runs, from their own machine:
wandb agent <SWEEP_ID>
```
Watch progress together at the URL W&B prints when the sweep starts.
Stop agents (Ctrl+C) once you've spent your trial budget, then check the
W&B dashboard's "best run" for the winning hyperparameters per
architecture.

## Running on Google Colab (no local GPU/PC)

Colab has two failure modes this project is specifically built to
survive: **local disk is wiped on disconnect**, and **sessions
disconnect without warning** (idle timeout, hard max length, or the GPU
just gets reclaimed). The fix is two-layered:

1. **Everything persists to Google Drive**, not Colab's local disk
2. **Checkpointing** — every epoch is saved; a disconnect loses at most
   the current epoch's progress, not the whole run. Already-completed
   runs are auto-skipped when you resume.

See `scripts/colab_notebook_setup.py` for ready-to-paste notebook cells.
Short version:

```python
# Cell 1 — run once per session (safe to re-run after a disconnect)
!git clone https://github.com/YOUR_ORG/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt -q

import sys; sys.path.append('.')
from src.utils.colab_utils import mount_drive, colab_output_root, print_colab_session_tips
mount_drive()

from config import Config
config = Config()
config.output_root = colab_output_root()   # <- outputs/checkpoints/logs now live on Drive
print_colab_session_tips()
```

```bash
# Cell 2 — the actual training. Split by architecture across sessions/
# group members if you want to parallelize:
python -m scripts.run_all_experiments --backbones resnet50
```

**After ANY disconnect:** reconnect the runtime, re-run Cell 1 (just
remounts Drive, doesn't re-download models), then re-run Cell 2 with the
SAME `--backbones` argument. That's it — no manual bookkeeping about
which epoch you were on.

**What NOT to rely on:** there's no code trick that prevents a Colab
disconnect outright (keeping the tab open/focused helps somewhat, but
nothing guarantees survival through many unattended hours). The design
here deliberately doesn't try to prevent disconnects — it makes
recovering from them cheap instead, which is the more reliable strategy.

**Splitting work across group members:** since `--backbones` lets each
person run a different architecture from their own Colab account against
the SAME Drive-backed `outputs/` folder (share the Drive folder with
the group, or each sync their own and merge `experiment_log.csv`/
`outputs/logs/*.json` at the end), the group's `num_repeats ×
len(backbone_names)` total runs can happen in parallel instead of one
person running everything sequentially.

## Mapping to the assignment's presentation sections

| Assignment section | Where it comes from |
|---|---|
| 1. Dataset description / EDA | `scripts/run_eda.py` output + your own notes on data sources/photo settings |
| 2. Data preparation | `src/data/transforms.py` (augmentation), `src/data/dataset.py` (splitting) |
| 3. Model architecture | `src/models/` + `scripts/show_pretrained_baseline.py` |
| 4. Training method | `src/training/trainer.py` + `config.py` hyperparameters |
| 5. Evaluation metrics | `src/evaluation/metrics.py` |
| 6. Experimental results (mean±SD, p-values) | `src/evaluation/statistics.py`, run via `run_all_experiments.py` |
| 7. Discussion, GradCAM, eyeball analysis | `src/interpretability/gradcam.py` outputs + your own written analysis |

## What's NOT automated (still needs your judgment)

- Writing the actual "Discussion and conclusions" analysis text
- Deciding WHICH GradCAM/eyeball examples are most interesting to show
- The "why is our data collection unbiased" explanation for your slides
- Choosing good Optuna search ranges / W&B sweep parameter ranges for
  YOUR specific dataset — the defaults in this project are reasonable
  starting points, not guaranteed-optimal

