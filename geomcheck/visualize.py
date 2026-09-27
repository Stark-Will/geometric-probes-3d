"""Per-face defect flags and a dependency-light shaded renderer (matplotlib) for paper figures."""
import numpy as np, trimesh
from .probes import load_mesh, normalise_and_weld, decimate, _tri_geometry, _edge_table, _components_from_edges, fibonacci_sphere, CONFIG


def prepare(path, decimate_to=10000):
    V, F, _ = normalise_and_weld(load_mesh(path))
    if decimate_to and len(F) > decimate_to:
        V, F = decimate(V, F, decimate_to)
        V, F, _ = normalise_and_weld(trimesh.Trimesh(V, F, process=False))
    return V, F


def face_flags(V, F):
    """Boolean per-face flags: self_intersect, small_component, thin, hidden (same rules as probes.py)."""
    import pymeshlab
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    area, e, n = _tri_geometry(V, F)
    uniq, face_of_he, inv, cnt, he = _edge_table(F)
    ncomp, labels = _components_from_edges(len(F), face_of_he, inv, cnt)
    comp_area = np.bincount(labels, weights=area, minlength=ncomp)
    small = (comp_area < CONFIG["small_comp_area_frac"] * area.sum())[labels]
    ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    si = np.asarray(ms.current_mesh().face_selection_array(), bool)
    if len(si) != len(F): si = np.zeros(len(F), bool)
    ok = np.isfinite(n).all(1) & (area > 1e-12)
    nn = np.where(ok[:, None], n, 0.0)
    c = V[F].mean(1)
    rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False), scale_to_box=False)
    eps = CONFIG["ray_eps"]
    def first(o, d):
        t = np.full(len(o), np.inf); tri = np.full(len(o), -1)
        it, ir, loc = rmi.intersects_id(o, d, multiple_hits=False, return_locations=True)
        if len(ir): t[ir] = ((loc - o[ir]) * d[ir]).sum(1); tri[ir] = it
        return t, tri
    thick = np.full(len(F), np.inf)
    for s in (-1.0, 1.0):
        d = s * nn; t, tri = first(c + eps * d, d)
        hit = tri >= 0; ex = np.zeros(len(F), bool)
        ex[hit] = (nn[tri[hit]] * d[hit]).sum(1) > 0
        thick = np.minimum(thick, np.where(hit & ex, t + eps, np.inf))
    thin = ok & (thick < 0.005)
    vis = np.zeros(len(F), bool)
    for d in fibonacci_sphere(CONFIG["n_visibility_dirs"]):
        idx = np.flatnonzero(~vis)
        if not len(idx): break
        dd = np.broadcast_to(d, (len(idx), 3)).copy()
        t, _ = first(c[idx] + eps * dd, dd); vis[idx[~np.isfinite(t)]] = True
    hidden = ok & ~vis
    bnd = np.zeros(len(F), bool); bnd[face_of_he[(cnt == 1)[inv]]] = True  # faces touching a boundary (hole) edge
    return dict(self_intersect=si, small_component=small, thin=thin, hidden=hidden, boundary=bnd)


COLORS = {"self_intersect": (0.85, 0.1, 0.1), "small_component": (1.0, 0.55, 0.0),
          "thin": (0.15, 0.35, 0.95), "hidden": (0.6, 0.2, 0.75), "boundary": (0.1, 0.7, 0.2)}


def render(ax, V, F, flags=None, elev=20, azim=35, base=(0.82, 0.82, 0.82), order=("hidden", "thin", "boundary", "small_component", "self_intersect"), alpha_unflagged=1.0, zoom=1.35):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    area, e, n = _tri_geometry(V, F)
    n = np.nan_to_num(n)
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    light = view + np.array([0.3, -0.2, 0.5]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    col = np.tile(np.array(base), (len(F), 1))
    if flags:
        for k in order:
            if k in flags: col[flags[k]] = COLORS[k]
    col = np.clip(col * shade[:, None], 0, 1)
    flagged = np.zeros(len(F), bool)
    if flags:
        for k in order:
            if k in flags: flagged |= flags[k]
    col = np.concatenate([col, np.where(flagged, 1.0, alpha_unflagged)[:, None]], 1)
    # display with y-up meshes (GLB) -> swap to z-up for matplotlib
    P = V[:, [0, 2, 1]] * np.array([1, -1, 1])
    pc = Poly3DCollection(P[F], facecolors=col, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    lo, hi = P.min(0), P.max(0); ctr = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    try: ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except Exception: pass
