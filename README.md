# soccer-vision

Team-level positional analytics for 9v9 youth soccer from Trace camera footage.

**Status:** pre-implementation. Design spec approved 2026-05-27. Implementation
plan and Phase 0 scaffolding pending.

## What this is

A notebook-driven Python toolkit that ingests Trace footage and emits
team-level positional analytics: shape, space and pitch ownership, gaps,
ball-relative positioning, dynamics, zonal occupation, and youth-specific
metrics like a swarm index. Output is parquet tables, static plots, a per-game
HTML report, video overlays, and season-level analyses.

Scope is deliberately narrow: positional (no passing yet), team-level (no
player-specific work), post-game (no realtime), notebook-driven (no web app).

## Design

See [`docs/superpowers/specs/2026-05-27-soccer-vision-design.md`](docs/superpowers/specs/2026-05-27-soccer-vision-design.md)
for the full specification.
