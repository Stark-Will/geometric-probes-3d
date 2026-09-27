"""GSO synthetic defect injection (pre-registered in analysis/PREREGISTRATION_M2.md §2).
Usage: python scripts/gso_inject.py <object_name>   (one object per process; memory-safe)
Writes results/gso_meshes/<name>/{clean,<defect>_S<k>}.ply and meta.json.
"""
import hashlib, json, os, sys, zipfile, io
import numpy as np, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from geomcheck.probes import normalise_and_weld, decimate, _tri_geometry

SEV = {
    "holes": [0.002, 0.01, 0.04],          # total removed area fraction (3 patches)
    "floaters": [1, 3, 10],                # number of r=0.01 icospheres
    "interpenetrating": [0.05, 0.10, 0.20],# ellipsoid major radius
    "thin_fin": [0.008, 0.004, 0.002],     # plate thickness (S1 thickest)
    "noise": [0.001, 0.003, 0.01],         # sigma of normal displacement
    "crossing_sheet": [0.005, 0.02, 0.05], # duplicated patch area fraction
    "hidden_shell": [0.25, 0.5, 0.9],      # radius as fraction of max inscribed distance
}


def surf_point(V, F, area, n, rng):
    fi = rng.choice(len(F), p=area / area.sum())
    u, v = rng.random(2)
    if u + v > 1: u, v = 1 - u, 1 - v
    a, b, c = V[F[fi]]
    return a + u * (b - a) + v * (c - a), n[fi]


def patch(V, F, area, center, frac):
    cen = V[F].mean(1)
    order = np.argsort(np.linalg.norm(cen - center, axis=1))
    cum = np.cumsum(area[order])
    k = int(np.searchsorted(cum, frac * area.sum())) + 1
    return order[:k]


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    return trimesh.transformations.quaternion_matrix(q)[:3, :3]


def tangent(nrm):
    a = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
    t = np.cross(nrm, a); return t / np.linalg.norm(t)


def add(V, F, V2, F2):
    return np.vstack([V, V2]), np.vstack([F, F2 + len(V)])


def main(name):
    out = f"{ROOT}/results/gso_meshes/{name}"; os.makedirs(out, exist_ok=True)
    seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2**32
    with zipfile.ZipFile(f"{ROOT}/data/gso/zips/{name}.zip") as z:
        raw = z.read("meshes/model.obj")
    m = trimesh.load(io.BytesIO(raw), file_type="obj", force="mesh", process=False, skip_materials=True)
    V, F, _ = normalise_and_weld(m)
    n_raw = len(F)
    if len(F) > 10000:
        V, F = decimate(V, F, 10000)
    V, F, _ = normalise_and_weld(trimesh.Trimesh(V, F, process=False))
    area, e, n = _tri_geometry(V, F)
    ok = area > 1e-12
    meta = dict(name=name, seed=seed, raw_faces=n_raw, clean_faces=int(len(F)), skipped=[])
    save = lambda tag, VV, FF: trimesh.Trimesh(VV, FF, process=False).export(f"{out}/{tag}.ply")
    save("clean", V, F)
    base = trimesh.Trimesh(V, F, process=False)
    for defect, levels in SEV.items():
        for k, s in enumerate(levels, 1):
            rng = np.random.default_rng([seed, list(SEV).index(defect), k])
            tag = f"{defect}_S{k}"
            if defect == "holes":
                rm = np.zeros(len(F), bool)
                for _ in range(3):
                    c, _n = surf_point(V, F[ok], area[ok], n[ok], rng)
                    rm[patch(V, F, area, c, s / 3)] = True
                save(tag, V, F[~rm])
            elif defect == "floaters":
                VV, FF = V, F
                for _ in range(s):
                    c, nr = surf_point(V, F[ok], area[ok], n[ok], rng)
                    sp = trimesh.creation.icosphere(1, radius=0.01); sp.apply_translation(c + 0.05 * nr)
                    VV, FF = add(VV, FF, sp.vertices, sp.faces)
                save(tag, VV, FF)
            elif defect == "interpenetrating":
                c, nr = surf_point(V, F[ok], area[ok], n[ok], rng)
                el = trimesh.creation.icosphere(2, radius=1.0)
                ev = el.vertices * np.array([s, 0.5 * s, 0.5 * s]) @ rand_rot(rng).T + c
                save(tag, *add(V, F, ev, el.faces))
            elif defect == "thin_fin":
                c, nr = surf_point(V, F[ok], area[ok], n[ok], rng)
                bx = trimesh.creation.box(extents=(0.2, 0.2, s))  # plate normal = z
                t = tangent(nr); R = np.stack([nr, np.cross(t, nr), t], 1)  # plate normal -> t (fin contains nr)
                bv = bx.vertices @ R.T + c
                save(tag, *add(V, F, bv, bx.faces))
            elif defect == "noise":
                vn = base.vertex_normals
                save(tag, V + vn * rng.normal(0, s, size=(len(V), 1)), F)
            elif defect == "crossing_sheet":
                c, nr = surf_point(V, F[ok], area[ok], n[ok], rng)
                pf = patch(V, F, area, c, s)
                used, inv = np.unique(F[pf], return_inverse=True)
                ax = tangent(nr)
                R = trimesh.transformations.rotation_matrix(np.radians(30), ax, point=c)
                pv = trimesh.transform_points(V[used], R)
                save(tag, *add(V, F, pv, inv.reshape(-1, 3)))
            elif defect == "hidden_shell":
                if k == 1:
                    cand = rng.uniform(V.min(0), V.max(0), size=(20000, 3))
                    try:
                        inside = base.contains(cand)
                        sd = trimesh.proximity.signed_distance(base, cand[inside]) if inside.any() else np.array([])
                    except Exception as ex:
                        inside, sd = np.zeros(0, bool), np.array([])
                    good = sd > 0
                    if good.sum() == 0:
                        best = None
                    else:
                        j = np.argmax(np.where(good, sd, -np.inf))
                        best = (cand[inside][j], float(sd[j]))
                    meta["inscribed"] = None if best is None else dict(center=best[0].tolist(), r_in=best[1])
                if meta.get("inscribed") is None:
                    meta["skipped"].append(tag); continue
                sp = trimesh.creation.icosphere(2, radius=s * meta["inscribed"]["r_in"])
                sp.apply_translation(meta["inscribed"]["center"])
                save(tag, *add(V, F, sp.vertices, sp.faces))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(name, "ok", meta["clean_faces"], meta["skipped"])


if __name__ == "__main__":
    main(sys.argv[1])
