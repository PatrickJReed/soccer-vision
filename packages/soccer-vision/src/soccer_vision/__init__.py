"""soccer-vision: team-level positional analytics for 9v9 youth soccer."""

from soccer_vision.pipeline import (
    PipelineResult,
    analyze_video,
    assemble_from_parquet,
    assemble_phases,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineResult",
    "__version__",
    "analyze_video",
    "assemble_from_parquet",
    "assemble_phases",
]
