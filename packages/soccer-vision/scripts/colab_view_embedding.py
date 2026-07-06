# %% [markdown]
# # View-embedding trainer (Colab) — feeds on the view_dataset export
#
# Upload THREE files to the Colab session (e.g. drag into /content/):
#   1. view_dataset.parquet   (the per-frame view manifest)
#   2. view_dataset.json      (sidecar: content_hash, params, stats)
#   3. oceanside_clip.mp4      (the SAME clip the labeler used)
#
# This script is SELF-CONTAINED — only torch/torchvision/opencv/pandas/numpy
# (all preinstalled in Colab). It reimplements the essential ViewFrameReader
# decode logic so you do NOT need to pip-install the soccer_vision package.
#
# What the export gives you (per frame row): view_id (smoothed pseudo-label),
# view_key ("game:view_id" — the contrastive CLASS), confidence/weight (QC/loss
# weight), margin + ambiguous (pan-transition flag), view_second (runner-up view
# for soft labels), split (train/val), t_seconds. See stats in the .json.

# %%
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

EXPORT_DIR = Path("/content")                     # where you dropped the 2 export files
VIDEO      = Path("/content/oceanside_clip.mp4")  # the uploaded clip
IMG_SIZE   = 224                                  # backbone input
DOWNSCALE  = 0.5                                  # decode at half-res first (cheap), then resize

manifest = pd.read_parquet(EXPORT_DIR / "view_dataset.parquet")
meta = json.loads((EXPORT_DIR / "view_dataset.json").read_text())
print("rows:", len(manifest), "| views:", meta["digest"]["n_views"],
      "| train/val:", meta["stats"]["n_train"], "/", meta["stats"]["n_val"])
print("stats:", {k: meta["stats"][k] for k in ("switch_rate", "n_ambiguous",
                                                "confidence_median")})


# %%
def _content_hash(video_path, edge_bytes=1 << 20):
    """Path/mtime-independent fingerprint — must match meta['video']['content_hash']."""
    p = Path(video_path); size = p.stat().st_size
    h = hashlib.sha1(); h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(edge_bytes))
        if size > edge_bytes:
            f.seek(max(0, size - edge_bytes)); h.update(f.read(edge_bytes))
    return h.hexdigest()[:16]

# Sanity: confirm the uploaded clip is the one the manifest was built for.
got, want = _content_hash(VIDEO), meta["video"]["content_hash"]
assert got == want, f"wrong/re-encoded video: {got} != {want}"
print("content hash OK:", got)


# %%
class ViewFrameDataset(Dataset):
    """Decodes frames on demand from the mp4 using the manifest.

    Contrastive labels: `view_id` (int) / `view_key` (str). Rows with view_id == -1
    (unlabelable low-keypoint frames) are dropped. Each worker opens its OWN cv2
    capture (a VideoCapture is not picklable) via lazy init keyed on worker id.
    Returns (img[C,H,W] float in RGB, view_id:int, weight:float).
    """
    def __init__(self, manifest, video_path, split=None, img_size=224, downscale=0.5,
                 drop_unassigned=True):
        m = manifest
        if split is not None:
            m = m[m["split"] == split]
        if drop_unassigned:
            m = m[m["view_id"] != -1]
        self.rows = m.reset_index(drop=True)
        self.video_path = str(video_path)
        self.img_size = img_size
        self.downscale = downscale
        self._cap = None
        self._pos = 0
        # contiguous 0..K-1 class ids for a classification/contrastive head
        self.view_ids = sorted(self.rows["view_id"].unique().tolist())
        self.view_to_class = {v: i for i, v in enumerate(self.view_ids)}

    def __len__(self):
        return len(self.rows)

    def _cap_open(self):
        # lazy per-worker capture (survives DataLoader fork with num_workers>0)
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            self._pos = 0
        return self._cap

    def _decode(self, frame_idx):
        cap = self._cap_open()
        if frame_idx < self._pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); self._pos = frame_idx
        while self._pos < frame_idx:
            if not cap.grab(): break
            self._pos += 1
        ok, frame = cap.read(); self._pos += 1
        if not ok:
            raise IndexError(f"decode failed at frame {frame_idx}")
        return frame  # BGR

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        frame = self._decode(int(row["frame"]))
        if self.downscale != 1.0:
            frame = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                 # torch wants RGB
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return img, self.view_to_class[int(row["view_id"])], float(row["weight"])


def _worker_init(_):
    # ensure each worker gets a fresh capture (defensive; lazy init also handles it)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._cap = None


train_ds = ViewFrameDataset(manifest, VIDEO, split="train", img_size=IMG_SIZE, downscale=DOWNSCALE)
val_ds   = ViewFrameDataset(manifest, VIDEO, split="val",   img_size=IMG_SIZE, downscale=DOWNSCALE)
print("train:", len(train_ds), "val:", len(val_ds), "classes:", len(train_ds.view_ids))

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2,
                      worker_init_fn=_worker_init, drop_last=True)
val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2,
                      worker_init_fn=_worker_init)

# smoke test one batch
xb, yb, wb = next(iter(train_dl))
print("batch:", xb.shape, xb.dtype, "labels:", yb[:8].tolist(), "weights:", wb[:4].tolist())


# %% [markdown]
# ## Starter model — backbone → embedding, trained on the view pseudo-labels
#
# Simplest pretext that yields a clustering-ready embedding: classify the view_id
# (cross-entropy) on a pretrained backbone; take the penultimate features as the
# low-dim vector. Swap in a SupCon / triplet loss later for a metric embedding;
# swap in a masked-autoencoder once tracking boxes exist (then `n_boxes>0` rows
# carry a player keep_mask you can union into the MAE mask).

# %%
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 128
n_classes = len(train_ds.view_ids)

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()                       # -> 512-d features
model = nn.Sequential(
    backbone,
    nn.Linear(512, EMB_DIM), nn.ReLU(),           # the embedding you cluster later
).to(device)
head = nn.Linear(EMB_DIM, n_classes).to(device)   # view-id classification pretext

opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-4)
ce = nn.CrossEntropyLoss(reduction="none")

def run_epoch(dl, train=True):
    model.train(train); head.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            emb = model(xb); logits = head(emb)
            loss = (ce(logits, yb) * wb).mean()    # weight by assignment confidence
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * len(yb)
            correct += int((logits.argmax(1) == yb).sum()); tot += len(yb)
    return loss_sum / max(tot, 1), correct / max(tot, 1)

for epoch in range(5):
    tl, ta = run_epoch(train_dl, True)
    vl, va = run_epoch(val_dl, False)
    print(f"epoch {epoch}: train loss {tl:.3f} acc {ta:.2f} | val loss {vl:.3f} acc {va:.2f}")

# %% [markdown]
# `model(x)` now yields a 128-d embedding per frame. Next steps you own:
#  - dump embeddings for all frames, UMAP + cluster, compare to view_id (should recover them).
#  - upgrade the loss to SupCon/triplet for a true metric space, or MAE once masks exist.
#  - to scale cross-game, re-export each game and concatenate manifests (view_key is
#    game-scoped, so views never collide across clips).

# %% [markdown]
# ## Embedding quality — does the 128-d space recover the views on its own?
# Val accuracy says the backbone can classify views; this asks the deeper question:
# do the embeddings CLUSTER by view unsupervised? (Caveat: the head was trained with
# view cross-entropy, so high agreement is confirmatory, not independent — the honest
# generalization test is a time-gapped split / a second game.) Reports KMeans-vs-view
# ARI + purity (numbers) and a 2D map (to eyeball for a pan manifold).

# %%
import collections
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

full_ds = ViewFrameDataset(manifest, VIDEO, split=None, img_size=IMG_SIZE, downscale=DOWNSCALE)
full_dl = DataLoader(full_ds, batch_size=64, shuffle=False, num_workers=2,
                     worker_init_fn=_worker_init)
model.eval()
embs, views = [], []
with torch.no_grad():
    for xb, yb, _wb in full_dl:
        embs.append(model(xb.to(device)).cpu().numpy())
        views.append(yb.numpy())
E = np.concatenate(embs); V = np.concatenate(views)
print("embeddings:", E.shape)

k = len(full_ds.view_ids)
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E)
ari = adjusted_rand_score(V, km.labels_)
purity = sum(collections.Counter(V[km.labels_ == c]).most_common(1)[0][1]
             for c in set(km.labels_)) / len(V)
sil = silhouette_score(E, V)   # separation of the TRUE view labels in embedding space
print(f"KMeans({k}) vs view_id:  ARI={ari:.3f}  purity={purity:.3f}  |  "
      f"silhouette(view_id)={sil:.3f}")
print("  ARI/purity ~1.0 => embeddings recover the views; silhouette>0 => views are separated")

# 2D map (UMAP if available, else t-SNE) colored by view_id — eyeball for a pan manifold
try:
    import umap
    XY = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(E); proj = "UMAP"
except Exception:
    from sklearn.manifold import TSNE
    XY = TSNE(n_components=2, init="pca", random_state=0).fit_transform(E); proj = "t-SNE"
plt.figure(figsize=(7, 6))
sc = plt.scatter(XY[:, 0], XY[:, 1], c=V, cmap="tab20", s=14)
plt.title(f"{proj} of 128-d embeddings, colored by view_id"); plt.colorbar(sc, label="view")
plt.tight_layout(); plt.savefig("/content/embedding_map.png", dpi=120); plt.show()
print("saved /content/embedding_map.png")

# %% [markdown]
# ## Held-out-views probe — does the embedding GENERALIZE, or just memorize labels?
# Train a FRESH model that never sees 3 held-out views, then check whether those unseen
# views still form clean clusters in its embedding. High ARI/silhouette on the held-out
# views => the backbone learned general "field-view appearance" (real generalization, the
# result that justifies the learned path over classical per-clip ORB). Low => it only
# memorized the trained view labels.

# %%
import torch.nn as nn
from torchvision import models

HELDOUT = [4, 9, 11]                      # excluded from training entirely
seen = manifest[~manifest["view_id"].isin(HELDOUT)]
seen_train = ViewFrameDataset(seen, VIDEO, split="train", img_size=IMG_SIZE, downscale=DOWNSCALE)
tr_dl = DataLoader(seen_train, batch_size=32, shuffle=True, num_workers=2,
                   worker_init_fn=_worker_init, drop_last=True)

bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); bb.fc = nn.Identity()
m2 = nn.Sequential(bb, nn.Linear(512, EMB_DIM), nn.ReLU()).to(device)
h2 = nn.Linear(EMB_DIM, len(seen_train.view_ids)).to(device)
opt2 = torch.optim.Adam(list(m2.parameters()) + list(h2.parameters()), lr=1e-4)
ce2 = nn.CrossEntropyLoss()
for ep in range(5):
    m2.train(); h2.train()
    for xb, yb, _w in tr_dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = ce2(h2(m2(xb)), yb); opt2.zero_grad(); loss.backward(); opt2.step()
    print("epoch", ep, "done")

# embed ALL frames (incl held-out) with the held-out-BLIND model
full2 = ViewFrameDataset(manifest, VIDEO, split=None, img_size=IMG_SIZE, downscale=DOWNSCALE)
f2dl = DataLoader(full2, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=_worker_init)
m2.eval(); E2 = []
with torch.no_grad():
    for xb, _y, _w in f2dl:
        E2.append(m2(xb.to(device)).cpu().numpy())
E2 = np.concatenate(E2); Vall = full2.rows["view_id"].to_numpy()

mask = np.isin(Vall, HELDOUT); Eh, Vh = E2[mask], Vall[mask]
kmh = KMeans(n_clusters=len(HELDOUT), n_init=10, random_state=0).fit(Eh)
print(f"HELD-OUT {HELDOUT} (never trained): "
      f"ARI={adjusted_rand_score(Vh, kmh.labels_):.3f}  "
      f"silhouette={silhouette_score(Eh, Vh):.3f}  n={int(mask.sum())}")
print("  high => backbone generalizes to unseen views; low => memorized trained labels only")
