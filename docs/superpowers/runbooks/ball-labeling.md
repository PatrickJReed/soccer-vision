# Ball Labeling Runbook (Phase 2)

## Goal

Build a 400–500-frame, single-class (`ball`) dataset from Patrick's Trace
footage to fine-tune a ball-only YOLOv8 detector for soccer-vision.

## Steps

### 1. Extract frames

From the soccer-vision repo root:

```bash
uv run python scripts/extract_ball_frames.py \
    --inputs ~/Downloads/<Game1>.mov ~/Downloads/<Game2>.mov ~/Downloads/<Game3>.mov \
    --out data/labeled/ball_v1/raw \
    --per-video 100
```

Use 3–5 different games to cover lighting, field, and kit variation. Goal:
≥500 sampled frames in `data/labeled/ball_v1/raw/`.

### 2. Create the Roboflow project

1. Go to https://app.roboflow.com → New Project.
2. Project type: **Object Detection**.
3. Single annotation class: `ball`.
4. License: keep private.
5. Upload all frames from `data/labeled/ball_v1/raw/`.

### 3. Label

For each frame: draw a tight bounding box around the ball if visible. Skip
frames where the ball is fully out of frame or fully occluded. Aim for tight
boxes — small-object detection is sensitive to box quality.

Productivity tip: Roboflow's auto-suggest can pre-place boxes after the first
~50; review each before confirming.

### 4. Generate version + export

1. In Roboflow: **Versions → Generate New Version**.
2. Preprocessing: Auto-Orient, Resize 1280x1280 (Letterbox).
3. Augmentations: Mosaic, Mixup, HSV jitter (H ±15°, S ±25%, V ±25%),
   Motion Blur (1–5px), Scale ±50%, Brightness ±20%.
4. Train/Valid/Test split: 80 / 10 / 10.
5. Export as **YOLOv8** format; copy the API curl or `pip install roboflow`
   download snippet.

### 5. Hand off to training

The download snippet goes into `examples/finetune_ball.ipynb` (Task 3) at
the dataset-download cell. From there the Colab training notebook drives
the fine-tune.
