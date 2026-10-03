# Quickstart

## Install

```bash
pip install geomcheck
```

Python 3.10 or newer. The runtime dependencies are NumPy, SciPy, Trimesh, PyMeshLab and `embreex`;
all ship binary wheels, and nothing needs a GPU or a display.

To work on the code, or to reproduce the paper, install from source:

```bash
git clone https://github.com/Stark-Will/geometric-probes-3d
cd geometric-probes-3d
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

The paper's exact environment is pinned in `requirements.lock.txt` in the repository.

!!! note "Linux: install `libOpenGL.so.0` for decimation"
    PyMeshLab loads its filters from plugins. On Linux, the plugin that provides quadric
    edge-collapse decimation links against the system library `libOpenGL.so.0`. Minimal
    images (Docker, CI runners, servers) often lack it; PyMeshLab then prints
    `Unable to load the following plugins: … libfilter_meshing.so` on first use and the
    decimation filter is missing. Install it with

    ```bash
    sudo apt-get install libopengl0        # Debian / Ubuntu
    sudo dnf install libglvnd-opengl       # Fedora / RHEL
    ```

    Only the library is needed, not a GPU or display. You can check with
    `geomcheck.decimation_available()`. Without it, every probe still works on undecimated
    meshes, and `compute_all(..., decimate_to=N)` raises `PyMeshLabFilterUnavailable` with
    these instructions instead of an `AttributeError`.

## Probe a mesh

```python
from geomcheck import compute_all

r = compute_all("model.glb")          # any file trimesh can load: GLB, OBJ, PLY, STL, ...
print(r["watertight"], r["n_boundary_loops"], r["n_small_components"])
print(r["self_intersect_face_frac"], r["hidden_surface_frac"], r["thin_frac_0.005"])
```

`compute_all` accepts a path or a `trimesh.Trimesh` and returns a flat `dict` of plain Python
scalars, so it is easy to collect into a table:

```python
import trimesh
from geomcheck import compute_all

sphere = trimesh.creation.icosphere(subdivisions=3)
floater = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
floater.apply_translation((2.5, 0, 0))

r = compute_all(trimesh.util.concatenate([sphere, floater]))
assert r["n_components"] == 2 and r["n_small_components"] == 1
```

### Options

```python
compute_all(path_or_mesh, do_self_intersection=True, do_rays=True, decimate_to=None)
```

| Argument | Effect |
|---|---|
| `do_self_intersection` | Run the PyMeshLab self-intersection probe. Set `False` to skip it. |
| `do_rays` | Run the Embree ray probes (hidden surface, thin walls). Set `False` to skip them. |
| `decimate_to` | If set and the mesh has more faces, first decimate it to about this many faces (quadric edge collapse, topology/boundary/normal preserving). The paper used `10000` for every corpus except 3D-DefectBench, whose meshes were already near that size. |

Decimation can change thickness and intersection counts, so compare numbers only between meshes
probed with the same setting. `orig_n_faces` always records the face count before decimation.

## Many meshes

The repository includes a memory-safe batch runner (one mesh per subprocess, wall-clock timeout,
RSS watchdog, resumable). It writes one JSON per mesh and a combined CSV:

```bash
python scripts/run_probes.py --glob 'my_models/*.glb' --out my_results --workers 2
python scripts/run_probes.py --glob 'my_models/*.glb' --out my_results_dec --decimate 10000
```

The runner additionally needs `psutil` and `pandas`.

## Configuration

All thresholds live in one dictionary, `geomcheck.CONFIG`; see the
[probe reference](probes.md#configuration). The paper's results were produced with these values
unchanged.
