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

## Bake-off setup

The bake-off notebooks (`examples/bakeoff_*.ipynb`) need two one-time setup
steps before they can run in Colab:

1. **Google Drive folder.** Create `MyDrive/soccer-vision/` and upload
   `bakeoff_clip.mp4` to it. The notebooks mount Drive and read the clip
   from this path. Footage stays out of git history (the file exceeds
   GitHub's 100 MB limit and contains a youth game).
2. **Colab Secret `GITHUB_PAT`.** In Colab, open the 🔑 (Secrets) sidebar
   and add a secret named `GITHUB_PAT` containing a GitHub Personal Access
   Token with `repo` read scope. The notebooks use it to `pip install` the
   private `soccer-vision` package.

## Design

See [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](docs/superpowers/specs/2026-05-27-soccer-vision-design.md).
