"""Deterministic CPU geometric probes for triangle meshes.

All probes operate on a mesh normalised to unit bounding-box diagonal and centred at the
bounding-box centre, so every length-based quantity is scale invariant. No learned parameters.
Random sampling uses a fixed seed, so results are bit-reproducible for a given input file and
library versions.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh

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


# ----------------------------------------------------------------------------- loading
def load_mesh(path: str) -> trimesh.Trimesh:
    """Load any trimesh-readable file as a single triangle mesh (scene transforms applied)."""
    m = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"not a triangle mesh: {type(m)}")
    return m


def normalise_and_weld(m: trimesh.Trimesh, decimals: int = CONFIG["merge_decimals"]):
    """Return (V, F) with bbox-centre at origin, bbox diagonal == 1, vertices welded by position."""
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    if len(V) == 0 or len(F) == 0:
        raise ValueError("empty mesh")
    lo, hi = V.min(0), V.max(0)
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag) or diag <= 0:
        raise ValueError("degenerate bounding box")
    V = (V - (lo + hi) / 2.0) / diag
    key = np.round(V, decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    # use the mean position of welded vertices for stability
    Vw = np.zeros_like(uniq)
    np.add.at(Vw, inv, V)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    Vw /= cnt[:, None]
    Fw = inv[F]
    return Vw, Fw, diag


# ----------------------------------------------------------------------------- helpers
def _tri_geometry(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    area = 0.5 * area2
    e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = cr / area2[:, None]
    return area, e, n


def _edge_table(F):
    """Undirected edges -> (unique_edges, face_index_per_halfedge, inverse, counts, directed)."""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    face_of_he = np.tile(np.arange(len(F)), 3)
    und = np.sort(he, axis=1)
    uniq, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    return uniq, face_of_he, inv.reshape(-1), cnt, he


def _components_from_edges(n_faces, face_of_he, inv, cnt):
    """Face connected components via shared manifold-or-not edges (union-find, vectorised by scipy)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    order = np.argsort(inv, kind="stable")
    inv_s, f_s = inv[order], face_of_he[order]
    # connect consecutive half-edges that share the same undirected edge
    same = inv_s[1:] == inv_s[:-1]
    r, c = f_s[1:][same], f_s[:-1][same]
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n_faces, n_faces))
    ncomp, labels = connected_components(g, directed=False)
    return ncomp, labels


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def _sample_surface(V, F, area, n, seed):
    rng = np.random.default_rng(seed)
    w = area / area.sum()
    fi = rng.choice(len(F), size=n, p=w)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = V[F[fi, 0]], V[F[fi, 1]], V[F[fi, 2]]
    p = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return p, fi


# ----------------------------------------------------------------------------- PyMeshLab feature detection
# PyMeshLab exposes filters only for the plugins it could load at import time. On Linux the
# plugin that provides quadric edge-collapse decimation (libfilter_meshing.so) links against the
# system OpenGL dispatch library libOpenGL.so.0; if that library is missing the plugin is skipped
# silently and the filter simply does not exist on MeshSet. Detect this instead of failing with
# an opaque AttributeError.
_DECIMATION_FILTER = "meshing_decimation_quadric_edge_collapse"
_SELF_INTERSECTION_FILTER = "compute_selection_by_self_intersections_per_face"


class PyMeshLabFilterUnavailable(RuntimeError):
    """A PyMeshLab filter needed by a probe is not available in this installation."""


def pymeshlab_has_filter(name: str) -> bool:
    """True if PyMeshLab is importable and exposes filter ``name`` (i.e. its plugin loaded)."""
    try:
        import pymeshlab
    except ImportError:
        return False
    return hasattr(pymeshlab.MeshSet(), name)


def decimation_available() -> bool:
    """True if :func:`decimate` (and therefore ``compute_all(..., decimate_to=N)``) can run."""
    return pymeshlab_has_filter(_DECIMATION_FILTER)


def _require_filter(ms, name: str) -> None:
    if not hasattr(ms, name):
        raise PyMeshLabFilterUnavailable(
            f"PyMeshLab filter '{name}' is not available: the PyMeshLab plugin that provides it "
            "failed to load (PyMeshLab prints 'Unable to load the following plugins' on import). "
            "On Linux this is usually the missing system library libOpenGL.so.0; install it, e.g. "
            "'sudo apt-get install libopengl0' (Debian/Ubuntu) or 'sudo dnf install "
            "libglvnd-opengl' (Fedora/RHEL), then restart Python. No GPU or display is needed.")


# ----------------------------------------------------------------------------- probes
def topology_probes(V, F, area, e):
    out = {}
    total_area = float(area.sum())
    out["n_vertices"] = int(len(V))
    out["n_faces"] = int(len(F))
    degenerate = area < 1e-12
    out["degenerate_face_frac"] = float(degenerate.mean())
    sf = np.sort(F, axis=1)
    _, dup_cnt = np.unique(sf, axis=0, return_counts=True)
    out["duplicate_face_frac"] = float((dup_cnt[dup_cnt > 1] - 1).sum() / len(F))

    uniq, face_of_he, inv, cnt, he = _edge_table(F)
    n_edges = len(uniq)
    out["n_edges"] = int(n_edges)
    boundary = cnt == 1
    nonman = cnt > 2
    elen = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
    out["boundary_edge_frac"] = float(boundary.mean())
    out["boundary_length"] = float(elen[boundary].sum())  # in bbox-diagonal units
    out["nonmanifold_edge_frac"] = float(nonman.mean())
    # boundary loops = connected components of the boundary-edge graph
    if boundary.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        be = uniq[boundary]
        verts, be_i = np.unique(be.ravel(), return_inverse=True)
        be_i = be_i.reshape(-1, 2)
        g = coo_matrix((np.ones(len(be_i)), (be_i[:, 0], be_i[:, 1])), shape=(len(verts), len(verts)))
        out["n_boundary_loops"] = int(connected_components(g, directed=False)[0])
    else:
        out["n_boundary_loops"] = 0
    # winding consistency: a manifold edge should be traversed once in each direction
    man2 = cnt == 2
    dir_sign = np.where(he[:, 0] < he[:, 1], 1, -1)
    s = np.zeros(n_edges, dtype=np.int64)
    np.add.at(s, inv, dir_sign)
    out["winding_inconsistent_edge_frac"] = float((man2 & (s != 0)).sum() / max(man2.sum(), 1))

    # components
    ncomp, labels = _components_from_edges(len(F), face_of_he, inv, cnt)
    comp_area = np.bincount(labels, weights=area, minlength=ncomp)
    small = comp_area < CONFIG["small_comp_area_frac"] * total_area
    out["n_components"] = int(ncomp)
    out["largest_component_area_frac"] = float(comp_area.max() / total_area)
    out["n_small_components"] = int(small.sum())
    out["small_component_area_frac"] = float(comp_area[small].sum() / total_area)
    out["n_large_components"] = int((~small).sum())
    out["watertight"] = bool((not boundary.any()) and (not nonman.any()))
    out["euler_characteristic"] = int(len(np.unique(F)) - n_edges + len(F))
    return out, dict(labels=labels, uniq=uniq, cnt=cnt, face_of_he=face_of_he, inv=inv)


def quality_probes(V, F, area, e):
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 4.0 * math.sqrt(3.0) * area / (e ** 2).sum(1)  # 1 = equilateral, 0 = degenerate
    q = np.nan_to_num(q, nan=0.0)
    out["tri_quality_median"] = float(np.median(q))
    out["tri_quality_p05"] = float(np.percentile(q, 5))
    out["sliver_face_frac"] = float((q < CONFIG["sliver_q"]).mean())
    out["sliver_area_frac"] = float(area[q < CONFIG["sliver_q"]].sum() / area.sum())
    el = e.ravel()
    out["edge_length_cv"] = float(el.std() / max(el.mean(), 1e-15))
    return out


def roughness_probes(V, F, area, n, aux):
    out = {}
    uniq, cnt, face_of_he, inv = aux["uniq"], aux["cnt"], aux["face_of_he"], aux["inv"]
    # interior manifold edges: pair the two incident faces
    order = np.argsort(inv, kind="stable")
    inv_s, f_s = inv[order], face_of_he[order]
    first = np.r_[True, inv_s[1:] != inv_s[:-1]]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, len(inv_s)])
    m2 = counts == 2
    fa, fb = f_s[starts[m2]], f_s[starts[m2] + 1]
    eid = inv_s[starts[m2]]
    valid = np.isfinite(n[fa]).all(1) & np.isfinite(n[fb]).all(1)
    fa, fb, eid = fa[valid], fb[valid], eid[valid]
    cosang = np.clip((n[fa] * n[fb]).sum(1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    elen = np.linalg.norm(V[uniq[eid, 0]] - V[uniq[eid, 1]], axis=1)
    w = elen / max(elen.sum(), 1e-15)
    out["dihedral_mean_deg"] = float((ang * w).sum()) if len(ang) else 0.0
    for t in CONFIG["dihedral_thresholds_deg"]:
        out[f"dihedral_gt{int(t)}_lenfrac"] = float(w[ang > t].sum()) if len(ang) else 0.0
    # discrete Gaussian curvature (angle defect) on vertices whose star is closed and manifold
    try:
        tm = trimesh.Trimesh(V, F, process=False)
        defects = np.abs(trimesh.curvature.vertex_defects(tm))
        bverts = np.unique(uniq[cnt != 2].ravel()) if (cnt != 2).any() else np.array([], dtype=int)
        mask = np.ones(len(V), bool)
        mask[bverts] = False
        used = np.zeros(len(V), bool)
        used[np.unique(F)] = True
        mask &= used
        d = defects[mask]
        out["angle_defect_mean"] = float(d.mean()) if len(d) else 0.0
        out["angle_defect_p95"] = float(np.percentile(d, 95)) if len(d) else 0.0
    except Exception:
        out["angle_defect_mean"] = float("nan")
        out["angle_defect_p95"] = float("nan")
    return out


def self_intersection_probe(V, F, area):
    """Fraction of faces (and area) that intersect other faces of the same mesh (PyMeshLab)."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    _require_filter(ms, _SELF_INTERSECTION_FILTER)
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V.astype(np.float64), face_matrix=F.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    sel = np.asarray(ms.current_mesh().face_selection_array(), dtype=bool)
    if len(sel) != len(F):  # pymeshlab may drop unreferenced data; be conservative
        return {"self_intersect_face_frac": float(sel.mean()) if len(sel) else float("nan"),
                "self_intersect_area_frac": float("nan")}
    return {"self_intersect_face_frac": float(sel.mean()),
            "self_intersect_area_frac": float(area[sel].sum() / area.sum())}


def _first_hits(intersector, origins, dirs):
    """Return (t, tri) of the first hit per ray; t=inf, tri=-1 when the ray escapes."""
    t = np.full(len(origins), np.inf)
    tri = np.full(len(origins), -1, dtype=np.int64)
    if len(origins) == 0:
        return t, tri
    it, ir, loc = intersector.intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    if len(ir):
        t[ir] = ((loc - origins[ir]) * dirs[ir]).sum(1)
        tri[ir] = it
    return t, tri


def ray_probes(V, F, area, n):
    """Hidden-surface fraction and thin-wall fractions via Embree ray casting (embreex)."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    cfg = CONFIG
    ok = area > 1e-12
    Fs, As, Ns = F[ok], area[ok], n[ok]
    tm = trimesh.Trimesh(V, Fs, process=False)
    rmi = RayMeshIntersector(tm, scale_to_box=False)
    p, fi = _sample_surface(V, Fs, As, cfg["n_surface_samples"], cfg["seed"])
    nrm = Ns[fi]
    out = {}
    # (1) hidden surface: a sample is externally visible if a ray from it escapes in any direction
    dirs = fibonacci_sphere(cfg["n_visibility_dirs"])
    visible = np.zeros(len(p), bool)
    for d in dirs:
        idx = np.flatnonzero(~visible)
        if len(idx) == 0:
            break
        dd = np.broadcast_to(d, (len(idx), 3)).copy()
        o = p[idx] + cfg["ray_eps"] * dd
        t, _ = _first_hits(rmi, o, dd)
        visible[idx[~np.isfinite(t)]] = True
    out["hidden_surface_frac"] = float(1.0 - visible.mean())
    # (2) thin walls: cast along +n and -n; accept a hit only if it exits a surface
    #     (ray direction . hit-face normal > 0), i.e. the ray travelled through material.
    thick = np.full(len(p), np.inf)
    for sgn in (-1.0, 1.0):
        d = sgn * nrm
        o = p + cfg["ray_eps"] * d
        t, tri = _first_hits(rmi, o, d)
        hit = tri >= 0
        exits = np.zeros(len(p), bool)
        exits[hit] = (Ns[tri[hit]] * d[hit]).sum(1) > 0
        cand = np.where(hit & exits, t + cfg["ray_eps"], np.inf)
        thick = np.minimum(thick, cand)
    finite = np.isfinite(thick)
    out["thickness_defined_frac"] = float(finite.mean())
    for tau in cfg["thin_tau"]:
        out[f"thin_frac_{tau:g}"] = float((thick < tau).mean())
    out["thickness_median"] = float(np.median(thick[finite])) if finite.any() else float("nan")
    return out


def decimate(V, F, target_faces: int):
    """Quadric edge-collapse decimation (PyMeshLab, deterministic) to ~target_faces faces.
    Frozen parameters: preservetopology=True, preserveboundary=True, preservenormal=True,
    planarquadric=True, qualitythr=0.3 (default). Topology preservation avoids creating
    decimation artefacts (holes/self-intersections) that the probes would otherwise measure."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    _require_filter(ms, _DECIMATION_FILTER)
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V.astype(np.float64), face_matrix=F.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces), preservetopology=True,
                                                preserveboundary=True, preservenormal=True,
                                                planarquadric=True)
    cm = ms.current_mesh()
    return np.asarray(cm.vertex_matrix(), dtype=np.float64), np.asarray(cm.face_matrix(), dtype=np.int64)


def compute_all(path_or_mesh, do_self_intersection=True, do_rays=True, decimate_to=None) -> dict:
    m = load_mesh(path_or_mesh) if isinstance(path_or_mesh, str) else path_or_mesh
    V, F, diag = normalise_and_weld(m)
    res = {"orig_bbox_diag": diag, "orig_n_faces": int(len(F))}
    if decimate_to is not None and len(F) > decimate_to:
        V, F = decimate(V, F, decimate_to)
        V, F, _ = normalise_and_weld(trimesh.Trimesh(V, F, process=False))
    area, e, n = _tri_geometry(V, F)
    topo, aux = topology_probes(V, F, area, e)
    res.update(topo)
    res.update(quality_probes(V, F, area, e))
    res.update(roughness_probes(V, F, area, n, aux))
    if do_self_intersection:
        res.update(self_intersection_probe(V, F, area))
    if do_rays:
        res.update(ray_probes(V, F, area, n))
    return res
