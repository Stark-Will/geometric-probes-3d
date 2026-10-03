# geomcheck

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22995915.svg)](https://doi.org/10.5281/zenodo.22995915)

**Deterministic, CPU-only mesh probes for generated 3D assets.**

`geomcheck` computes a fixed set of geometric checks on a triangle mesh: open boundaries,
connected components and floaters, non-manifold and inconsistently wound edges,
self-intersections, hidden (never-visible) surface, thin walls, triangle quality and surface
roughness. It is pure Python on top of [Trimesh](https://trimesh.org),
[PyMeshLab](https://pymeshlab.readthedocs.io) and Embree (via `embreex`), needs no GPU, and takes
seconds per mesh.

Every probe reads the mesh after the same normalisation (bounding box centred at the origin,
diagonal scaled to 1, vertices welded), so all lengths are fractions of the bounding-box diagonal.
There are no learned parameters, and random sampling uses a fixed seed, so a given file and library
build produce the same numbers.

!!! warning "Validity, not quality"
    The probes tell you whether a mesh is geometrically *valid* — printable, riggable,
    exportable. They are not a measure of how good a model looks. See
    [Limitations](limitations.md) for what the accompanying study found.

## Where to go next

- [Quickstart](quickstart.md) — install and probe your first mesh.
- [Probe reference](probes.md) — every value `compute_all` returns and how it is computed.
- [Limitations](limitations.md) — what the probes can and cannot measure.
- [Citation](citation.md) — how to cite the paper and code.

## Paper and data

`geomcheck` is the probe implementation behind the paper
*Geometric validity is not perceptual quality: what deterministic probes can and cannot measure in
generated 3D assets* (Miles Carter, 2026, [doi:10.5281/zenodo.22995915](https://doi.org/10.5281/zenodo.22995915)).
The [GitHub repository](https://github.com/Stark-Will/geometric-probes-3d) holds the paper source,
pre-registrations, analysis scripts and per-asset statistics.

Built by the team behind [Modelfy](https://modelfy.art), an AI 3D model generator. No Modelfy
system, model or data was evaluated in the study.
