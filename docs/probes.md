# Probe reference

All descriptions below follow `geomcheck/probes.py`, the single source of truth for every
threshold. `compute_all` returns one flat dictionary with the keys listed here.

## Pipeline

1. **Load.** `load_mesh(path)` reads any Trimesh-readable file as a single triangle mesh, with scene
   transforms applied and materials skipped (`trimesh.load(..., force="mesh", process=False)`).
2. **Normalise and weld.** `normalise_and_weld` centres the bounding box at the origin and scales its
   diagonal to 1. Vertices are welded by rounding the normalised coordinates to 10⁻⁶
   (`merge_decimals=6`); each welded group is replaced by its mean position. Welding matters for GLB
   files, which often split vertices along UV seams — without it a closed sphere would look like a
   soup of open triangles.
3. **Optional decimation.** With `decimate_to=N` and more than `N` faces, the mesh is decimated by
   PyMeshLab quadric edge collapse with `preservetopology=True`, `preserveboundary=True`,
   `preservenormal=True`, `planarquadric=True`, then normalised and welded again. Topology
   preservation avoids creating holes or self-intersections that the probes would then measure.
4. **Probes.** Topology/boundary, triangle quality and roughness always run; self-intersection
   (`do_self_intersection`) and the ray probes (`do_rays`) can be switched off.

Lengths are in units of the bounding-box diagonal; "faces", "edges" and "samples" below refer to the
normalised (and, if requested, decimated) mesh.

## Size and bookkeeping

| Key | Meaning |
|---|---|
| `orig_bbox_diag` | Bounding-box diagonal of the input, in the file's own units (the only unnormalised value). |
| `orig_n_faces` | Face count after welding, before any decimation. |
| `n_vertices`, `n_faces`, `n_edges` | Counts on the probed mesh (edges are undirected and unique). |
| `degenerate_face_frac` | Fraction of faces with area below 10⁻¹². |
| `duplicate_face_frac` | Fraction of faces that repeat another face's vertex set (extra copies only). |

## Boundary and manifoldness

Each undirected edge is counted by how many faces use it: 1 = boundary, 2 = manifold,
more than 2 = non-manifold.

| Key | Meaning |
|---|---|
| `boundary_edge_frac` | Fraction of edges used by exactly one face (open boundary / hole rims). |
| `boundary_length` | Total length of boundary edges, in bbox-diagonal units. |
| `n_boundary_loops` | Connected components of the boundary-edge graph (≈ number of holes or open rims). |
| `nonmanifold_edge_frac` | Fraction of edges used by more than two faces. |
| `winding_inconsistent_edge_frac` | Among manifold edges, the fraction whose two half-edges run in the same direction, i.e. adjacent faces with inconsistent orientation. |
| `watertight` | `True` when there is no boundary edge and no non-manifold edge. |

## Topology: components and floaters

Faces are grouped into connected components through shared edges (manifold or not).

| Key | Meaning |
|---|---|
| `n_components` | Number of face-connected components. |
| `largest_component_area_frac` | Area of the largest component / total area. |
| `n_small_components` | Components smaller than 1 % of the total surface area (`small_comp_area_frac=0.01`) — floaters and debris. |
| `small_component_area_frac` | Their total area / total area. |
| `n_large_components` | Remaining components. |
| `euler_characteristic` | χ = V − E + F, with V the number of vertices referenced by a face. A closed sphere has χ = 2. |

The paper's classifier derives a genus proxy g = max(0, 2C − χ)/2 from `n_components` (C) and χ;
it is not returned by `compute_all`.

## Triangle quality

For a triangle with area A and edge lengths a, b, c the quality is
q = 4√3·A / (a² + b² + c²), which is 1 for an equilateral triangle and 0 for a degenerate one.

| Key | Meaning |
|---|---|
| `tri_quality_median`, `tri_quality_p05` | Median and 5th percentile of q. |
| `sliver_face_frac`, `sliver_area_frac` | Fraction of faces, and of area, with q < 0.1 (`sliver_q`). |
| `edge_length_cv` | Coefficient of variation (std / mean) of all triangle edge lengths. |

## Roughness

On every interior manifold edge the dihedral angle is the angle between the two incident face
normals (0° on a flat surface).

| Key | Meaning |
|---|---|
| `dihedral_mean_deg` | Edge-length-weighted mean of that angle, in degrees. |
| `dihedral_gt30_lenfrac`, `dihedral_gt60_lenfrac`, `dihedral_gt90_lenfrac` | Fraction of interior edge length whose angle exceeds 30°, 60°, 90°. |
| `angle_defect_mean`, `angle_defect_p95` | Mean and 95th percentile of the absolute angle defect (discrete Gaussian curvature) on vertices whose star is closed and manifold. `NaN` if it cannot be computed. |

## Self-intersection

PyMeshLab's `compute_selection_by_self_intersections_per_face` marks every face that intersects
another face of the same mesh — interpenetrating parts, crossing sheets.

| Key | Meaning |
|---|---|
| `self_intersect_face_frac` | Fraction of faces marked. |
| `self_intersect_area_frac` | Fraction of area marked (`NaN` if PyMeshLab returned a different face count). |

## Hidden surface and thin walls (ray probes)

20,000 points are sampled on the surface, area-weighted, with seed 0 (`n_surface_samples`, `seed`).
Rays are cast with Embree, starting 10⁻⁴ off the surface (`ray_eps`).

| Key | Meaning |
|---|---|
| `hidden_surface_frac` | Fraction of samples from which a ray fails to escape in **every** one of 64 Fibonacci-sphere directions (`n_visibility_dirs`): surface that can never be seen from outside, such as internal shells or enclosed parts. |
| `thickness_defined_frac` | Fraction of samples at which a thickness could be measured. |
| `thin_frac_0.005`, `thin_frac_0.01` | Fraction of samples thinner than 0.005 and 0.01 diagonals (`thin_tau`). |
| `thickness_median` | Median thickness over samples where it is defined (`NaN` if none). |

Thickness at a sample is the shorter of two rays, along the inward and the outward normal, that hit a
face whose normal points the same way as the ray — that is, the ray travelled through material and
exited the surface. Rays that escape or hit a front-facing surface do not define a thickness.

## Configuration

```python
CONFIG = dict(
    merge_decimals=6,          # vertex welding: round normalised coordinates to 1e-6
    small_comp_area_frac=0.01, # component counted as 'small/floater' if < 1% of total area
    n_surface_samples=20000,   # area-weighted surface samples for ray probes
    n_visibility_dirs=64,      # Fibonacci-sphere directions for hidden-surface probe
    ray_eps=1e-4,              # ray origin offset (units of bbox diagonal)
    thin_tau=(0.005, 0.01),    # thin-wall thresholds (units of bbox diagonal)
    sliver_q=0.1,              # triangle quality below this => sliver
    dihedral_thresholds_deg=(30.0, 60.0, 90.0),
    seed=0,
)
```

These values were frozen before the paper's expert-set scoring. Changing them changes the meaning of
the features; the frozen classifiers in the repository assume the defaults.

## Per-face flags and rendering

`geomcheck.visualize` (needs `pip install "geomcheck[viz]"` for matplotlib) exposes
`face_flags(V, F)`, which returns boolean per-face masks — `self_intersect`, `small_component`,
`thin` (< 0.005), `hidden`, `boundary` — using the same rules as the probes but evaluated at face
centroids, and `render(ax, V, F, flags)`, the dependency-light shaded renderer used for the gallery
figures. `prepare(path, decimate_to=10000)` loads, normalises and optionally decimates a file into
`(V, F)` for both.

## Determinism

Probe outputs are deterministic for a given input file and library versions (fixed seed, no
learned parameters); the test suite checks that two runs are identical. The paper does not claim
bit-identity across library versions it did not run.
