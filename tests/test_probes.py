"""Unit tests of geometric probes on synthetic meshes with known ground truth."""
import numpy as np
import trimesh
import pytest

from geomcheck.probes import compute_all


def sphere(r=1.0, center=(0, 0, 0), sub=3):
    s = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    s.apply_translation(center)
    return s


def cat(*ms):
    return trimesh.util.concatenate(list(ms))


@pytest.fixture(scope="module")
def clean():
    return compute_all(sphere())


def test_clean_sphere(clean):
    r = clean
    assert r["n_components"] == 1 and r["n_small_components"] == 0
    assert r["watertight"] and r["boundary_edge_frac"] == 0 and r["n_boundary_loops"] == 0
    assert r["nonmanifold_edge_frac"] == 0 and r["winding_inconsistent_edge_frac"] == 0
    assert r["euler_characteristic"] == 2
    assert r["self_intersect_face_frac"] == 0
    assert r["hidden_surface_frac"] < 0.001
    assert r["thin_frac_0.005"] < 0.001
    assert r["tri_quality_median"] > 0.8 and r["sliver_face_frac"] == 0
    assert r["dihedral_gt30_lenfrac"] == 0


def test_floater():
    r = compute_all(cat(sphere(), sphere(0.05, (2.5, 0, 0), sub=2)))
    assert r["n_components"] == 2 and r["n_small_components"] == 1
    assert 0 < r["small_component_area_frac"] < 0.01
    assert r["largest_component_area_frac"] > 0.99


def test_hole():
    s = sphere()
    c = s.triangles_center
    keep = ~(c[:, 2] > 0.9)  # remove a polar cap -> one boundary loop
    h = trimesh.Trimesh(s.vertices, s.faces[keep], process=False)
    r = compute_all(h)
    assert not r["watertight"] and r["boundary_edge_frac"] > 0 and r["n_boundary_loops"] == 1
    assert r["euler_characteristic"] == 1


def test_nonmanifold_edge():
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]], float)
    F = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    r = compute_all(trimesh.Trimesh(V, F, process=False), do_rays=False)
    assert r["nonmanifold_edge_frac"] > 0 and not r["watertight"]


def test_winding_flip():
    s = sphere()
    F = s.faces.copy()
    F[:10] = F[:10, ::-1]
    r = compute_all(trimesh.Trimesh(s.vertices, F, process=False), do_rays=False, do_self_intersection=False)
    assert r["winding_inconsistent_edge_frac"] > 0


def test_overlapping_spheres_self_intersection_and_hidden():
    r = compute_all(cat(sphere(1.0), sphere(1.0, (1.0, 0, 0))))
    assert r["self_intersect_face_frac"] > 0.005
    assert r["hidden_surface_frac"] > 0.1  # interior caps are enclosed by the other sphere


def test_nested_hidden_internal():
    r = compute_all(cat(sphere(1.0), sphere(0.5)))
    # inner sphere area = 0.25 / 1.25 = 20% of total, fully hidden
    assert 0.17 < r["hidden_surface_frac"] < 0.23
    assert r["self_intersect_face_frac"] == 0


def test_thin_walls():
    thin = compute_all(trimesh.creation.box(extents=(1.0, 1.0, 0.003)))
    thick = compute_all(trimesh.creation.box(extents=(1.0, 1.0, 0.5)))
    assert thin["thin_frac_0.005"] > 0.9
    assert thick["thin_frac_0.005"] < 0.01
    assert thin["thickness_median"] < 0.005 < thick["thickness_median"]


def test_roughness():
    s = sphere(sub=4)
    rng = np.random.default_rng(0)
    V = s.vertices + rng.normal(scale=0.03, size=s.vertices.shape)
    noisy = compute_all(trimesh.Trimesh(V, s.faces, process=False), do_rays=False, do_self_intersection=False)
    clean = compute_all(s, do_rays=False, do_self_intersection=False)
    assert noisy["dihedral_mean_deg"] > 3 * clean["dihedral_mean_deg"]
    assert noisy["angle_defect_p95"] > clean["angle_defect_p95"]


def test_slivers():
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.001, 0], [0.5, -0.001, 0]], float)
    F = np.array([[0, 1, 2], [0, 3, 1]])
    r = compute_all(trimesh.Trimesh(V, F, process=False), do_rays=False, do_self_intersection=False)
    assert r["sliver_face_frac"] == 1.0


def test_duplicate_and_degenerate():
    s = sphere(sub=2)
    F = np.vstack([s.faces, s.faces[:5], [[0, 0, 1]]])
    r = compute_all(trimesh.Trimesh(s.vertices, F, process=False), do_rays=False, do_self_intersection=False)
    assert r["duplicate_face_frac"] > 0 and r["degenerate_face_frac"] > 0


def test_scale_and_translation_invariance(clean):
    s = sphere()
    s.apply_scale(37.0)
    s.apply_translation((100, -5, 3))
    r = compute_all(s)
    for k in ["hidden_surface_frac", "thin_frac_0.005", "thickness_median", "boundary_length",
              "tri_quality_median", "dihedral_mean_deg", "small_component_area_frac"]:
        assert abs(r[k] - clean[k]) < 1e-4, k


def test_unwelded_uv_seams_are_welded():
    s = sphere()
    # split every face into its own vertices (like a GLB with per-face UVs)
    V = s.vertices[s.faces.ravel()]
    F = np.arange(len(V)).reshape(-1, 3)
    r = compute_all(trimesh.Trimesh(V, F, process=False), do_rays=False, do_self_intersection=False)
    assert r["n_components"] == 1 and r["watertight"]


def test_determinism():
    m = cat(sphere(1.0), sphere(1.0, (1.0, 0, 0)))
    a, b = compute_all(m), compute_all(m)
    assert a == b


def test_decimation_matches_target_and_keeps_clean_sphere_clean():
    r = compute_all(sphere(sub=6), decimate_to=10000)
    assert r["orig_n_faces"] == 81920 and 9000 <= r["n_faces"] <= 10000
    assert r["watertight"] and r["n_components"] == 1 and r["self_intersect_face_frac"] == 0
