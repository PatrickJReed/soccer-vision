# soccer-vision

Team-level positional analytics for 9v9 youth soccer from Trace camera footage.

**Status:** Phase 0 complete (scaffolding). Phase 1 (structured bake-off) next.

## What this is

A notebook-driven Python toolkit that ingests Trace footage and emits
team-level positional analytics: shape, space and pitch ownership, gaps,
ball-relative positioning, dynamics, zonal occupation, and youth-specific
metrics like a swarm index. Output is parquet tables, static plots, a per-game
HTML report, video overlays, and season-level analyses.

Scope is deliberately narrow: positional (no passing yet), team-level (no
player-specific work), post-game (no realtime), notebook-driven (no web app).

## Quickstart

Local development:

```bash
git clone <repo-url>
cd soccer-vision
uv sync
uv run pytest
```

Colab usage (per notebook):

```python
!pip install -q git+https://github.com/<owner>/soccer-vision.git
import soccer_vision
```

## Layout

- `packages/soccer-vision/` — core Python package (uv workspace member)
- `examples/` — Colab notebooks (bake-off, pipeline demo, season analysis)
- `data/` — small reference artifacts only (gitignored: any `.mp4` clips, full-game footage, model outputs)
- `docs/superpowers/specs/` — design specifications
- `docs/superpowers/plans/` — implementation plans

### Tracking backend extras

The `RoboflowBackend` (chosen during Phase 1) requires upstream model code that
isn't pulled in by default — it's a heavyweight install (Ultralytics, supervision,
PyTorch, the sports package). Install with:

```bash
uv pip install -e "packages/soccer-vision[roboflow]"
```

Without the extras, the package imports fine and all other functionality works;
only `RoboflowBackend.process()` will fail at call time with a clear ImportError.

### Using a fine-tuned ball detector (Phase 2 output)

If you have a fine-tuned ball-only model (see
[`docs/superpowers/runbooks/ball-labeling.md`](docs/superpowers/runbooks/ball-labeling.md)),
point `RoboflowBackend` at it via the `ball_weights_path` constructor arg:

```python
from pathlib import Path
from soccer_vision.tracking.roboflow import RoboflowBackend

backend = RoboflowBackend(
    ball_weights_path=Path("data/labeled/ball_v1/best.pt"),
)
df = backend.process(Path("data/games/<game>.mp4"))
```

When `ball_weights_path=None` (default), the adapter downloads roboflow's
original ball detector to `~/.cache/soccer_vision/weights/`. Override
forces the local file — useful for Phase 2 evaluation and the production
pipeline once acceptance is met.

## Bake-off setup

The bake-off notebooks (`examples/bakeoff_*.ipynb`) need one Google Drive
folder before they can run in Colab: create `MyDrive/soccer-vision/` and
upload `bakeoff_clip.mp4` there. The notebooks mount Drive and read the
clip from this path. Footage stays out of git history because the file
exceeds GitHub's 100 MB limit (and there's no reason to cache it on
GitHub's CDN).

### Run order

Open each notebook via its Colab badge (Colab will prompt for GitHub auth
the first time on a private repo):

| Step | Notebook | Runtime |
|---|---|---|
| 1 | [bakeoff_roboflow.ipynb](examples/bakeoff_roboflow.ipynb) | Colab Pro + GPU |
| 2 | [bakeoff_atfa.ipynb](examples/bakeoff_atfa.ipynb) | Colab Pro + GPU |
| 3 | [bakeoff_tz.ipynb](examples/bakeoff_tz.ipynb) | Colab Pro + GPU |
| 4 | [colab_bakeoff.ipynb](examples/colab_bakeoff.ipynb) | Local (Jupyter) |

For steps 1–3: download each notebook's two output files (`trajectories.parquet`
and `annotated.mp4`) and place them at `data/bakeoff_outputs/{roboflow,atfa,tz}/`
locally. Step 4 reads those files and produces the side-by-side scoring view.

## Design

See [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](docs/superpowers/specs/2026-05-27-soccer-vision-design.md).
