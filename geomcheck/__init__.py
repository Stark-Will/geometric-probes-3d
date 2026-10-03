"""Deterministic, CPU-only geometric checks for (generated) triangle meshes."""
__version__ = "0.1.0"

from .probes import CONFIG, compute_all, decimation_available, load_mesh  # noqa: E402,F401

__all__ = ["CONFIG", "compute_all", "decimation_available", "load_mesh", "__version__"]
