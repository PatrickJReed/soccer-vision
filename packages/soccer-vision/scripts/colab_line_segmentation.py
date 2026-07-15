# %% [markdown]
# # Line-segmentation trainer (Colab) — SegFormer-B0 on the line-mask dataset
#
# Trains `nvidia/mit-b0` (SegformerForSemanticSegmentation, 6 classes) on the
# auto-label dataset produced by `soccer_vision.line_dataset` and evaluates it on
# the three HONEST tiers (spec: docs/superpowers/specs/
# 2026-07-15-line-segmentation-colab-design.md):
#
#   tier 1 (val1): view-held-out within training games — per-epoch model selection
#   tier 2 (val2): game-held-out on training fields    — evaluated ONCE, at the end
#   tier 3 (test): field-held-out                      — the deployment claim, final-only
#
# Degraded v0 mode: when no view manifest exists (all view_id == -1) the tier-1
# split falls back to a per-game TIME-BLOCKED split with a loud smoke-test-only
# banner. NEVER a random frame split (adjacent frames are near-duplicates).
#
# SELF-CONTAINED: no soccer_vision import at runtime — Colab only has the dataset
# dir (Drive: manifest.parquet + images/ + masks/ + dataset_stats.json).
#
# Repo-side verification (no torch/transformers needed):
#   python scripts/colab_line_segmentation.py --dry-run --dataset <dataset_dir>

# %%
# ---- CONFIG ------------------------------------------------------------------
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
# The action cells below no-op when this file runs as a repo CLI (--dry-run).
# Set True to run the cells outside Colab (e.g. a local GPU box).
RUN_CELLS = IN_COLAB

DATASET_DIR = Path("/content/drive/MyDrive/soccer-vision/line_dataset_v0")
OUT_DIR = Path("/content/drive/MyDrive/soccer-vision/line_seg_v0")

# Honest-eval split config
HELDOUT_FIELDS: list[str] = []   # tier 3: field-held-out TEST (final-only)
HELDOUT_GAMES: list[str] = []    # tier 2: game-held-out val2 (final-only)
HELDOUT_VIEW_FRACTION = 0.25     # tier 1: per-game seeded view holdout -> val1
SEED = 0
TIME_BLOCK_TRAIN_FRAC = 0.85     # degraded v0 mode: first 85% of frames -> train

# Model / training
MODEL_NAME = "nvidia/mit-b0"
EPOCHS = 20
BATCH = 8
LR = 6e-5
LR_SCHEDULE = "cosine"           # "cosine" | "constant"
CROP = 512                       # training crop (full-res source frames)
LINE_CROP_BIAS = 0.5             # fraction of crops forced to contain >=1 line pixel
VAL_SIZE = 512                   # deterministic val resize (masks NEAREST)
NUM_WORKERS = 2
CLASS_WEIGHT_MAX_MASKS = 512     # cap on masks read for class-frequency weights

# Cell switches (Colab run-all order: smoke first, then the full run)
SMOKE_TEST = True                # tiny 2-epoch path validation; set False once green
SMOKE_DATASET_DIR = DATASET_DIR  # the 70-pair v0 dataset
RUN_FULL_TRAINING = True

# Class vocabulary — HARDCODED copy of soccer_vision/pitch/line_masks.py::LINE_CLASSES
# (plus background 0). KEEP IN SYNC with that file; this script must not import
# soccer_vision at runtime (Colab only has the dataset dir).
CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "touchline",
    2: "goal_line",
    3: "midline",
    4: "box_line",
    5: "center_circle",
}
NUM_CLASSES = len(CLASS_NAMES)
LINE_CLASS_IDS = [c for c in CLASS_NAMES if c != 0]  # background EXCLUDED from mean IoU
# RGB tints for the visual cells (line_masks.py _OVERLAY_BGR, converted BGR->RGB)
CLASS_RGB = {1: (255, 255, 0), 2: (0, 128, 255), 3: (255, 128, 0),
             4: (255, 0, 255), 5: (0, 255, 0)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # SegFormer pretraining normalization
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEGRADED_BANNER = "\n".join([
    "=" * 78,
    "!!  DEGRADED v0 SPLIT — SMOKE-TEST ONLY  !!",
    "No view manifest (every remaining row has view_id == -1), so tier 1 falls",
    f"back to a per-game TIME-BLOCKED split: first {TIME_BLOCK_TRAIN_FRAC:.0%} of frames"
    " -> train, rest -> val1.",
    "Adjacent-frame near-duplicates INFLATE val1 — do NOT read it as generalization.",
    "(Never a random frame split — the view-embedding tautology lesson.)",
    "=" * 78,
])

_TIER_DESC = {
    "train": "train",
    "val1": "tier 1 (view-held-out val1, model selection)",
    "val2": "tier 2 (game-held-out val2, final-only)",
    "test": "tier 3 (field-held-out TEST, final-only)",
}


# %%
# ---- Manifest + tiered splits (pure; shared by the Colab cells and --dry-run) --
def load_manifest(dataset_dir: str | Path) -> pd.DataFrame:
    """Manifest-first read (orphan-file safety): rows whose image or mask file is
    missing on disk are WARNED about and DROPPED, with counts — never silent."""
    dataset_dir = Path(dataset_dir)
    df = pd.read_parquet(dataset_dir / "manifest.parquet")
    img_ok = df["image"].map(lambda p: (dataset_dir / str(p)).exists())
    msk_ok = df["mask"].map(lambda p: (dataset_dir / str(p)).exists())
    bad = ~(img_ok & msk_ok)
    if int(bad.sum()):
        print(f"WARNING: {int(bad.sum())}/{len(df)} manifest rows missing files on disk "
              f"({int((~img_ok).sum())} images, {int((~msk_ok).sum())} masks) — dropped")
    else:
        print(f"manifest: {len(df)} rows, 0 missing files")
    return df.loc[~bad].reset_index(drop=True)


def assign_tiers(
    manifest: pd.DataFrame,
    *,
    heldout_fields: list[str],
    heldout_games: list[str],
    heldout_view_fraction: float = HELDOUT_VIEW_FRACTION,
    seed: int = SEED,
) -> tuple[pd.DataFrame, bool]:
    """The honesty core: adds a 'tier' column; returns (df, degraded_mode).

    test  : field_id in heldout_fields (tier 3) — never trained/selected on.
    val2  : game_id in heldout_games among remaining rows (tier 2) — final-only.
    val1  : per remaining game, a seeded heldout_view_fraction of its distinct
            view_ids (tier 1) — per-epoch model selection. Rows with view_id == -1
            inside a view-labeled game stay in train (they can't be view-held).
    degraded v0: if ALL remaining rows have view_id == -1, per-game TIME-BLOCKED
            split (first TIME_BLOCK_TRAIN_FRAC of frames train, rest val1) + loud
            banner. NEVER a random frame split.
    Deterministic under `seed`: str-seeded per-game RNG (stable across processes)
    over the sorted view list; groupby iterates games in sorted order.
    """
    df = manifest.copy()
    df["tier"] = "train"
    df.loc[df["field_id"].isin(list(heldout_fields)), "tier"] = "test"
    df.loc[(df["tier"] == "train") & df["game_id"].isin(list(heldout_games)),
           "tier"] = "val2"
    rem = df.index[df["tier"] == "train"]
    degraded = len(rem) > 0 and bool((df.loc[rem, "view_id"] == -1).all())
    if degraded:
        print(_DEGRADED_BANNER)
        for _gid, g in df.loc[rem].groupby("game_id"):
            order = g.sort_values("frame").index
            n_train = int(len(order) * TIME_BLOCK_TRAIN_FRAC)
            if len(order) > 1:  # keep both sides non-empty when possible
                n_train = max(1, min(len(order) - 1, n_train))
            df.loc[order[n_train:], "tier"] = "val1"
    else:
        for gid, g in df.loc[rem].groupby("game_id"):
            views = sorted(int(v) for v in g["view_id"].unique() if v != -1)
            if len(views) < 2:
                print(f"note: game {gid} has {len(views)} labeled view(s) — "
                      "no val1 holdout possible for it")
                continue
            n_hold = max(1, round(len(views) * heldout_view_fraction))
            n_hold = min(n_hold, len(views) - 1)  # never hold out ALL of a game's views
            held = set(random.Random(f"{seed}:{gid}").sample(views, n_hold))
            df.loc[g.index[g["view_id"].isin(held)], "tier"] = "val1"
    return df, degraded


def summarize_tiers(df: pd.DataFrame) -> str:
    """Per-tier row counts. Empty tiers are reported 'not evaluable', never silent."""
    lines = []
    for tier in ("train", "val1", "val2", "test"):
        sub = df[df["tier"] == tier]
        desc = _TIER_DESC[tier]
        if len(sub) == 0:
            lines.append(f"{desc:<46} not evaluable with this config (0 rows)")
        else:
            lines.append(f"{desc:<46} {len(sub):>5} rows  "
                         f"({sub['game_id'].nunique()} game(s), "
                         f"{sub['field_id'].nunique()} field(s))")
    if (df["tier"] == "train").sum() == 0:
        lines.append("WARNING: train tier is EMPTY — check the heldout config")
    return "\n".join(lines)


# %%
# ---- Class frequencies -> CE weights (pure; numpy/cv2 only) --------------------
def class_pixel_frequencies(
    df: pd.DataFrame, dataset_dir: str | Path, max_masks: int = CLASS_WEIGHT_MAX_MASKS
) -> np.ndarray:
    """Per-class pixel fractions counted from the rows' mask PNGs (up to max_masks,
    evenly spaced through the selection — dataset-scale guard)."""
    import cv2  # lazy: --dry-run only needs this on the no-stats-json fallback

    paths = df["mask"].tolist()
    if len(paths) > max_masks:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_masks).astype(int)]
    counts = np.zeros(NUM_CLASSES, np.int64)
    for rel in paths:
        m = cv2.imread(str(Path(dataset_dir) / rel), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"WARNING: unreadable mask {rel} — skipped in frequency count")
            continue
        counts += np.bincount(m.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts / max(int(counts.sum()), 1)


def stats_class_frequencies(dataset_dir: str | Path) -> np.ndarray | None:
    """Class fractions from dataset_stats.json (per-game fracs weighted by n_written);
    None when the sidecar is absent/empty. The json omits background -> derived."""
    p = Path(dataset_dir) / "dataset_stats.json"
    if not p.exists():
        return None
    games = json.loads(p.read_text()).get("games", {})
    name_to_id = {v: k for k, v in CLASS_NAMES.items()}
    freq, total = np.zeros(NUM_CLASSES), 0
    for g in games.values():
        n = int(g.get("n_written", 0))
        for name, frac in g.get("class_pixel_frac", {}).items():
            if name in name_to_id:
                freq[name_to_id[name]] += float(frac) * n
        total += n
    if total == 0:
        return None
    freq /= total
    freq[0] = max(0.0, 1.0 - float(freq[1:].sum()))
    return freq


def class_weights_from_freq(freq: np.ndarray) -> np.ndarray:
    """Inverse-sqrt pixel-frequency CE weights, normalized to mean 1. Background
    (~99.9% of pixels) gets a tiny weight — unweighted CE just learns 'grass'.
    Absent classes are clamped (no inf); they contribute no gradient anyway."""
    f = np.maximum(np.asarray(freq, np.float64), 1e-8)
    w = 1.0 / np.sqrt(f)
    return (w / w.mean()).astype(np.float32)


def format_class_table(freq: np.ndarray, weights: np.ndarray | None = None,
                       title: str = "class pixel frequencies") -> str:
    lines = [title, f"  {'id':>2} {'class':<14}{'pixel %':>12}"
             + ("" if weights is None else f"{'CE weight':>12}")]
    for cid in sorted(CLASS_NAMES):
        row = f"  {cid:>2} {CLASS_NAMES[cid]:<14}{freq[cid]:>11.4%}"
        if weights is not None:
            row += f"{weights[cid]:>12.4f}"
        lines.append(row)
    return "\n".join(lines)


# %%
# ---- Colab deps (torch/torchvision/cv2 are preinstalled; transformers may not be)
if RUN_CELLS:
    try:
        import transformers  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "transformers>=4.40"])

# %%
# ---- Mount Drive, load the manifest, assign tiers ------------------------------
if RUN_CELLS:
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
    manifest = load_manifest(DATASET_DIR)
    tiers_df, degraded_split = assign_tiers(
        manifest, heldout_fields=HELDOUT_FIELDS, heldout_games=HELDOUT_GAMES,
        heldout_view_fraction=HELDOUT_VIEW_FRACTION, seed=SEED)
    print(summarize_tiers(tiers_df))


# %%
# ---- Data pipeline (defined lazily: --dry-run must never import torch) ---------
def _read_pair(dataset_dir, row):
    """(RGB uint8 HxWx3 image, uint8 HxW class mask) for one manifest row."""
    import cv2

    img = cv2.imread(str(Path(dataset_dir) / row["image"]), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(Path(dataset_dir) / row["mask"]), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise FileNotFoundError(f"unreadable pair: {row['image']} / {row['mask']}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), msk


def make_datasets(tiers_df, dataset_dir, *, crop=CROP, val_size=VAL_SIZE,
                  line_bias=LINE_CROP_BIAS):
    """{tier: torch Dataset} for the non-empty tiers.

    Train: random crop x crop window, ~line_bias of draws recentered on a random
    line pixel (so batches aren't all grass) + hflip (classes are side-merged:
    safe) + mild color jitter. Crops/flips are pixel-exact on the mask — the mask
    is never interpolated. Val/test: deterministic full-frame resize to val_size
    (bilinear image, NEAREST mask), no augmentation; the 16:9 -> square squash is
    identical for every tier, so IoUs stay comparable.

    Augmentation randomness uses the GLOBAL `random` module: torch DataLoader
    reseeds it per worker per epoch from torch's (manually seeded) generator, so
    crops vary across epochs yet the run stays reproducible.
    """
    import cv2
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                    saturation=0.2, hue=0.02)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _train_crop(img, msk):
        h, w = msk.shape
        if h < crop or w < crop:  # defensive: pad small frames with background
            img = np.pad(img, ((0, max(0, crop - h)), (0, max(0, crop - w)), (0, 0)))
            msk = np.pad(msk, ((0, max(0, crop - h)), (0, max(0, crop - w))))
            h, w = msk.shape
        y0, x0 = random.randint(0, h - crop), random.randint(0, w - crop)
        if random.random() < line_bias:
            ys, xs = np.nonzero(msk)
            if len(xs):  # recenter on a random line pixel, jittered, clamped
                j = random.randrange(len(xs))
                cy = int(ys[j]) + random.randint(-crop // 4, crop // 4)
                cx = int(xs[j]) + random.randint(-crop // 4, crop // 4)
                y0 = min(max(cy - crop // 2, 0), h - crop)
                x0 = min(max(cx - crop // 2, 0), w - crop)
        return img[y0:y0 + crop, x0:x0 + crop], msk[y0:y0 + crop, x0:x0 + crop]

    class LineSegDataset(Dataset):
        def __init__(self, rows, train):
            self.rows = rows.reset_index(drop=True)
            self.train = train

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            img, msk = _read_pair(dataset_dir, self.rows.iloc[i])
            if self.train:
                img, msk = _train_crop(img, msk)
                if random.random() < 0.5:
                    img, msk = img[:, ::-1], msk[:, ::-1]
            else:
                img = cv2.resize(img, (val_size, val_size),
                                 interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (val_size, val_size),
                                 interpolation=cv2.INTER_NEAREST)
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            x = x.float() / 255.0
            if self.train:
                x = jitter(x)
            x = (x - mean) / std
            return x, torch.from_numpy(np.ascontiguousarray(msk)).long()

    return {tier: LineSegDataset(tiers_df[tiers_df["tier"] == tier], tier == "train")
            for tier in ("train", "val1", "val2", "test")
            if (tiers_df["tier"] == tier).any()}


# %%
# ---- Metrics: per-class IoU + mean line-IoU (background EXCLUDED) ---------------
def iou_from_confusion(conf: np.ndarray) -> tuple[dict[int, float], float]:
    """({class_id: IoU (nan when the class never appears)}, mean over LINE classes
    present in GT/pred — background excluded from the mean)."""
    diag = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - diag
    iou = {c: (float(diag[c] / union[c]) if union[c] > 0 else float("nan"))
           for c in CLASS_NAMES}
    line = [iou[c] for c in LINE_CLASS_IDS if not math.isnan(iou[c])]
    return iou, (float(np.mean(line)) if line else float("nan"))


def format_iou_table(iou: dict[int, float], mean_line: float, title: str) -> str:
    lines = [title]
    for cid in sorted(CLASS_NAMES):
        v = iou[cid]
        shown = "   n/a" if math.isnan(v) else f"{v:.4f}"
        excl = "  (excluded from mean)" if cid == 0 else ""
        lines.append(f"  {cid} {CLASS_NAMES[cid]:<14} IoU {shown}{excl}")
    lines.append(f"  mean line-IoU: "
                 f"{'n/a' if math.isnan(mean_line) else f'{mean_line:.4f}'}")
    return "\n".join(lines)


def evaluate(model, dataset, device, batch=BATCH):
    """Per-class IoU over a tier: accumulate a 6x6 confusion matrix at label
    resolution (logits bilinearly upsampled from the model's 1/4-res output)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    dl = DataLoader(dataset, batch_size=batch, shuffle=False,
                    num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, y in dl:
            logits = model(pixel_values=x.to(device)).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
            idx = y.numpy().astype(np.int64) * NUM_CLASSES + pred.astype(np.int64)
            conf += np.bincount(idx.ravel(),
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES,
                                                                    NUM_CLASSES)
    return iou_from_confusion(conf)


# %%
# ---- Model + training loop ------------------------------------------------------
def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    """SegFormer-B0, 6 classes, id2label from the hardcoded vocabulary."""
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES,
        id2label=dict(CLASS_NAMES),
        label2id={v: k for k, v in CLASS_NAMES.items()})


def train_model(tiers_df, dataset_dir, out_dir, *, epochs=EPOCHS, batch=BATCH,
                lr=LR, crop=CROP, val_size=VAL_SIZE, lr_schedule=LR_SCHEDULE,
                max_batches=None, seed=SEED, extra_config=None):
    """Train with class-weighted CE; select on val1 mean line-IoU per epoch.
    Best checkpoint -> out_dir/best_model + config-echo sidecar train_config.json.
    max_batches caps batches per epoch (smoke test). Returns (model, dsets, best,
    history) — model is the LAST-epoch state; reload best_model for final eval.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dsets = make_datasets(tiers_df, dataset_dir, crop=crop, val_size=val_size)
    if "train" not in dsets or "val1" not in dsets:
        raise ValueError("need non-empty train and val1 tiers to train")

    # Weighted CE: inverse-sqrt pixel frequency computed from the TRAINING masks.
    freq = class_pixel_frequencies(tiers_df[tiers_df["tier"] == "train"], dataset_dir)
    weights = class_weights_from_freq(freq)
    print(format_class_table(freq, weights,
                             title="train-tier class frequencies -> CE weights"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = build_model().to(device)
    w = torch.tensor(weights, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if lr_schedule == "cosine" else None)
    dl = DataLoader(dsets["train"], batch_size=batch, shuffle=True,
                    num_workers=NUM_WORKERS,
                    drop_last=len(dsets["train"]) > batch)

    config_echo = {
        "model_name": MODEL_NAME, "dataset_dir": str(dataset_dir),
        "epochs": epochs, "batch": batch, "lr": lr, "lr_schedule": lr_schedule,
        "crop": crop, "val_size": val_size, "line_crop_bias": LINE_CROP_BIAS,
        "max_batches": max_batches, "seed": seed,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_frequencies": [float(v) for v in freq],
        "class_weights": [float(v) for v in weights],
        "tier_counts": {t: int((tiers_df["tier"] == t).sum())
                        for t in ("train", "val1", "val2", "test")},
        **(extra_config or {}),
    }
    best = {"epoch": -1, "val1_mean_line_iou": float("-inf")}
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for bi, (x, y) in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(pixel_values=x).logits  # (B, 6, crop/4, crop/4)
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
            loss = F.cross_entropy(logits, y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if sched is not None:
            sched.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        iou, mean_line = evaluate(model, dsets["val1"], device, batch=batch)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val1_mean_line_iou": mean_line,
                        "val1_iou": {CLASS_NAMES[c]: iou[c] for c in CLASS_NAMES}})
        print(format_iou_table(
            iou, mean_line,
            title=f"epoch {epoch}: train loss {train_loss:.4f} — tier 1 (val1)"))
        if not math.isnan(mean_line) and mean_line > best["val1_mean_line_iou"]:
            best = {"epoch": epoch, "val1_mean_line_iou": mean_line,
                    "val1_iou": {CLASS_NAMES[c]: iou[c] for c in CLASS_NAMES}}
            model.save_pretrained(out_dir / "best_model")
            (out_dir / "train_config.json").write_text(json.dumps(
                {**config_echo, "best": best, "history": history}, indent=2))
            print(f"  ** new best -> {out_dir / 'best_model'}")
    if best["epoch"] < 0:
        print("WARNING: no epoch produced a finite val1 mean line-IoU — "
              "no checkpoint saved")
    else:
        (out_dir / "train_config.json").write_text(json.dumps(
            {**config_echo, "best": best, "history": history}, indent=2))
    return model, dsets, best, history


# %% [markdown]
# ## SMOKE TEST — validates the whole path in minutes (CPU/T4-capable)
# Points at the 70-pair v0 dataset. EXPECT the degraded time-blocked banner (v0 has
# no view manifest). Run this before the full training cell on any fresh setup;
# set `SMOKE_TEST = False` in the config once it's green.

# %%
if RUN_CELLS and SMOKE_TEST:
    smoke_manifest = load_manifest(SMOKE_DATASET_DIR)
    smoke_tiers, smoke_degraded = assign_tiers(
        smoke_manifest, heldout_fields=[], heldout_games=[], seed=SEED)
    print(summarize_tiers(smoke_tiers))
    train_model(smoke_tiers, SMOKE_DATASET_DIR, "/content/line_seg_smoke",
                epochs=2, batch=2, crop=256, max_batches=8, lr=LR,
                extra_config={"smoke": True, "degraded_split": smoke_degraded})
    print("SMOKE OK — full path (data -> model -> loss -> metrics -> checkpoint) runs")

# %% [markdown]
# ## Full training run — best-by-val1 checkpoint lands in OUT_DIR on Drive

# %%
if RUN_CELLS and RUN_FULL_TRAINING:
    model, dsets, best, history = train_model(
        tiers_df, DATASET_DIR, OUT_DIR,
        extra_config={"degraded_split": degraded_split,
                      "heldout_fields": HELDOUT_FIELDS,
                      "heldout_games": HELDOUT_GAMES,
                      "heldout_view_fraction": HELDOUT_VIEW_FRACTION})
    print(f"best epoch {best['epoch']}: "
          f"val1 mean line-IoU {best['val1_mean_line_iou']:.4f}")

# %% [markdown]
# ## FINAL-ONLY evaluation — tiers 2/3, run ONCE after model selection
# (Evaluating these per-epoch would leak them into model selection.)

# %%
if RUN_CELLS and RUN_FULL_TRAINING:
    import torch
    from transformers import SegformerForSemanticSegmentation

    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = SegformerForSemanticSegmentation.from_pretrained(
        OUT_DIR / "best_model").to(_dev)
    for _tier in ("val2", "test"):
        if _tier in dsets:
            _iou, _mean = evaluate(best_model, dsets[_tier], _dev)
            print(format_iou_table(_iou, _mean, title=f"FINAL — {_TIER_DESC[_tier]}"))
        else:
            print(f"FINAL — {_TIER_DESC[_tier]}: not evaluable with this config (0 rows)")


# %%
# ---- Visuals + single-image inference (the future per-frame-fit consumer seam) --
def tint_mask(img_rgb, mask, alpha=0.6):
    """Class-tinted overlay (mirrors line_masks.py mask_overlay, RGB)."""
    out = img_rgb.astype(np.float64).copy()
    for cid, rgb in CLASS_RGB.items():
        sel = mask == cid
        if sel.any():
            out[sel] = np.asarray(rgb, np.float64) * alpha + out[sel] * (1.0 - alpha)
    return out.astype(np.uint8)


def predict_mask(model, image, device=None, size=VAL_SIZE):
    """Single-image inference: path or RGB array -> (pred mask at the ORIGINAL
    resolution, tinted RGB overlay). The seam the per-frame-fit consumer calls."""
    import cv2
    import torch
    import torch.nn.functional as F

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"unreadable image: {image}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    device = device or next(model.parameters()).device
    h0, w0 = image.shape[:2]
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
    x = (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) \
        / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=x[None].to(device)).logits
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear",
                               align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return pred, tint_mask(image, pred)


def show_val_grid(model, tiers_df, dataset_dir, tier="val1", n=6, save_to=None):
    """N rows of (frame | GT tint | prediction tint), evenly spaced through the
    tier. Renders the layout only — Patrick assesses the images."""
    import matplotlib.pyplot as plt

    rows = tiers_df[tiers_df["tier"] == tier]
    if rows.empty:
        print(f"no rows in tier {tier} — nothing to show")
        return
    pick = rows.iloc[np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]
    fig, axes = plt.subplots(len(pick), 3, figsize=(15, 2.9 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(pick.iterrows()):
        img, gt = _read_pair(dataset_dir, row)
        _, pred_tint = predict_mask(model, img)
        panels = [(img, f"{row['game_id']} f{int(row['frame'])}"),
                  (tint_mask(img, gt), "GT"), (pred_tint, "prediction")]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
        print(f"saved {save_to}")
    plt.show()


# %%
if RUN_CELLS and RUN_FULL_TRAINING:
    show_val_grid(best_model, tiers_df, DATASET_DIR, tier="val1", n=6,
                  save_to="/content/val_grid.png")

# %%
if RUN_CELLS and RUN_FULL_TRAINING:
    # single-image inference demo on the first val1 frame
    import matplotlib.pyplot as plt

    _row = tiers_df[tiers_df["tier"] == "val1"].iloc[0]
    _pred, _tint = predict_mask(best_model, Path(DATASET_DIR) / _row["image"])
    print("pred mask:", _pred.shape, "classes present:",
          {int(c): CLASS_NAMES[int(c)] for c in np.unique(_pred)})
    plt.figure(figsize=(10, 6))
    plt.imshow(_tint)
    plt.axis("off")
    plt.show()


# %%
# ---- Repo-side CLI: --dry-run split/manifest verification (NO torch needed) -----
def dry_run(dataset, heldout_fields, heldout_games, heldout_view_fraction, seed):
    print(f"== dry-run: {dataset} ==")
    df = load_manifest(dataset)
    tiers, degraded = assign_tiers(
        df, heldout_fields=heldout_fields, heldout_games=heldout_games,
        heldout_view_fraction=heldout_view_fraction, seed=seed)
    print()
    print(summarize_tiers(tiers))
    print()
    freq = stats_class_frequencies(dataset)
    src = "dataset_stats.json"
    if freq is None:
        freq = class_pixel_frequencies(tiers, dataset)
        src = "sampled masks"
    print(format_class_table(freq, class_weights_from_freq(freq),
                             title=f"class pixel frequencies ({src}) -> CE weights"))
    print()
    print(f"degraded v0 mode: {degraded}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Line-segmentation Colab trainer — repo-side --dry-run "
                    "verification (training itself runs as Colab cells)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load manifest, apply splits, print tier counts + class "
                         "frequencies (no torch/transformers)")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="dataset dir (manifest.parquet + images/ + masks/)")
    ap.add_argument("--heldout-field", action="append", default=None,
                    help="tier-3 field_id (repeatable)")
    ap.add_argument("--heldout-game", action="append", default=None,
                    help="tier-2 game_id (repeatable)")
    ap.add_argument("--heldout-view-fraction", type=float,
                    default=HELDOUT_VIEW_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if not args.dry_run:
        ap.error("only --dry-run is supported from the CLI; "
                 "training runs as Colab cells")
    dry_run(args.dataset, args.heldout_field or [], args.heldout_game or [],
            args.heldout_view_fraction, args.seed)


if __name__ == "__main__" and not IN_COLAB:
    main()
