# geomcheck

Deterministic, CPU-only geometric probes for triangle meshes (GLB, OBJ, PLY, STL, … anything
[trimesh](https://trimesh.org) can load). `geomcheck` reports open boundaries, connected components
and floaters, non-manifold and inconsistently wound edges, self-intersections, hidden (never-visible)
surface, thin walls, triangle quality and surface roughness. All lengths are in units of the
bounding-box diagonal, there are no learned parameters, and sampling uses a fixed seed.

```bash
pip install geomcheck
```

```python
from geomcheck import compute_all

r = compute_all("model.glb", decimate_to=10000)
print(r["watertight"], r["n_small_components"], r["self_intersect_face_frac"],
      r["hidden_surface_frac"], r["thin_frac_0.005"])
```

On Linux, PyMeshLab's quadric-decimation plugin needs the system library `libOpenGL.so.0`
(`sudo apt-get install libopengl0` on Debian/Ubuntu). No GPU or display is required.

**Scope.** These probes measure geometric *validity* (is the file printable, riggable, exportable?),
not perceived quality. The accompanying paper — *Geometric validity is not perceptual quality: what
deterministic probes can and cannot measure in generated 3D assets* (Miles Carter, 2026,
[doi:10.5281/zenodo.22995915](https://doi.org/10.5281/zenodo.22995915)) — found that the probes
detect injected holes and floaters reliably but correlate only weakly with human or MLLM geometry
ratings. Use them as validity gates and confound checks, not as perceptual judges.

- Documentation: <https://geomcheck.readthedocs.io>
- Source, paper, per-asset statistics: <https://github.com/Stark-Will/geometric-probes-3d>
- Licence: MIT
