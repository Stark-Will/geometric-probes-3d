# Pre-registration, milestone 2 (written 2026-09-27 ~17:00 UTC+8, committed before any MATE-3D scoring
# and before any GSO injection result or any ablation was computed)
Parent plan: analysis/PREREGISTRATION.md (milestone 1). Code: geomcheck (probes unchanged since milestone 1
except an added optional decimation step, see below). Frozen milestone-1 models:
results/frozen_models/geomprobe_lr_frozen.json (sha256 bc99da10…; refit reproduces milestone-1 thresholds exactly).

Disclosure of what has been seen: the MATE-3D MOS spreadsheet (printed head/tail rows while inspecting the
format) and probe values for the 8 meshes of the prompt "A blue vase" (engineering test of decimation).
No correlation between any probe and MOS has been computed. No GSO injection has been run.

## Common preprocessing change
`compute_all(..., decimate_to=10000)`: weld → PyMeshLab quadric edge collapse (preservetopology, preserveboundary,
preservenormal, planarquadric; parameters frozen in geomcheck/probes.py::decimate) → re-normalise → probes.
Rationale: 3D-DefectBench GLBs are ≤10.7k faces (median 8.1k); the frozen models were trained in that regime.
(fast_simplification was tried first and rejected because it broke topology on the blue-vase test meshes
— a preprocessing artefact, decided before any scoring.)

## 1. MATE-3D external validation (1,280 meshes = 8 generators × 160 prompts; human Geometry MOS 0–10)
Primary input: meshes decimated to 10,000 faces. Secondary: raw meshes (if compute allows).
Pre-registered predictors (post hoc signals from milestone 1 + frozen models); predicted sign: **higher value →
lower Geometry MOS** (negative correlation) for all.
- P1 thin_frac_0.005   - P2 genus_proxy   - P3 dihedral_mean_deg (roughness)
- P4 self_intersect_face_frac   - P5 hidden_surface_frac
- P6 frozen GeomProbe-LR P(form_surface)  - P7 P(fused_incomplete)  - P8 P(extra_geometry)
- P9 mean of P6–P8 ("defect score")
Secondary/exploratory predictors: angle_defect_mean, n_small_components, n_components, log orig_n_faces
(resolution control), and all other probe features (BH across them, labelled exploratory).
Analyses (all Spearman ρ unless stated; Kendall τ-b also reported):
- A (pooled, confounded): ρ over all 1,280 meshes.
- **B (PRIMARY, within-generator)**: Pearson correlation of within-generator percentile ranks of predictor and MOS
  (i.e. generator-stratified Spearman); also ρ per generator (8 values).
- C (within-prompt): Kendall τ-b between predictor and MOS across the 8 generators of each prompt, averaged over
  160 prompts.
- D (two-way FE): global ranks, residualised on generator + prompt dummies (OLS), Pearson of residuals.
- Generator-level (n=8) means: descriptive only.
CIs: percentile bootstrap over prompts (cluster = prompt, all 8 meshes move together), 10,000 resamples, seed 2026.
p-values for B: permutation of MOS within generator, 10,000 permutations, seed 2027 (two-sided).
Multiplicity: BH q=0.05 over P1–P9 for analysis B.
Replication criterion (fixed now): a signal **replicates** if B has ρ<0, 95% CI excluding 0 and BH q<0.05;
**partial** if only A (pooled) or only C/D is significant in the predicted direction; **fails** otherwise,
and **reverses** if B is significant with ρ>0.

## 2. GSO synthetic defect injection
Objects: 120 GSO models (CC BY 4.0), random.Random(0).sample of the 1,033 GoogleResearch Fuel models
(data/gso/subset_seed0.json). Each: load model.obj, weld, decimate to 10,000 faces (same function) = "clean".
Injected on the clean decimated mesh (units = bbox diagonal 1), per-object RNG seed = sha256(name) mod 2^32,
3 severities each (S1<S2<S3):
- holes: remove k geodesic-ish patches (faces within radius r of a random surface point), total removed area
  ≈ {0.2%, 1%, 4%} of surface (k=3 patches).
- floaters: {1, 3, 10} icospheres of radius 0.01, centred at a surface point + 0.05·normal.
- interpenetrating part ("fused-parts approximation"): an ellipsoid (radii 1:0.5:0.5 × s, s ∈ {0.05, 0.10, 0.20})
  centred on a surface point, randomly oriented, concatenated without boolean union.
- thin fin: a box plate of 0.2×0.2 extent and thickness t ∈ {0.008, 0.004, 0.002}, half-embedded at a surface point.
- noise: vertex displacement along vertex normal, N(0, σ²), σ ∈ {0.001, 0.003, 0.01}.
- crossing sheet (self-intersection): a surface patch of area fraction {0.5%, 2%, 5%} duplicated, rotated 30° about
  a tangent axis through its centre, concatenated.
- hidden internal shell: icosphere of radius f·r_in, f ∈ {0.25, 0.5, 0.9}, at the interior point of maximal
  distance to the surface (r_in = that distance; from 20k seeded grid/sample candidates; objects with no interior
  point found are skipped and counted).
Target checks: holes→boundary_edge_frac; floaters→n_small_components; interpenetrating→self_intersect_face_frac
(+hidden_surface_frac); thin fin→thin_frac_0.005; noise→dihedral_mean_deg (+angle_defect_mean);
crossing sheet→self_intersect_face_frac; hidden shell→hidden_surface_frac.
Metrics per check × defect × severity: AUROC (injected vs the 120 clean meshes); sensitivity/specificity at two
pre-registered operating points: (i) strict τ=0 ("any", for boundary/components/self-intersection/hidden/thin);
(ii) benchmark-calibrated τ_c = 95th percentile of the feature over 3D-DefectBench silver-holdout assets whose crowd
majority vote is 0 for all 5 geometry defects. Full cross-talk matrix (every check × every defect) reported.
Bootstrap CIs over objects (10,000, seed 2028).

## 3. Ablations on 3D-DefectBench (golden agreement cells; training on silver as in milestone 1)
- Leave-one-family-out (LOFO) and family-only models; families: topology {n_components, largest_component_area_frac,
  n_small_components, small_component_area_frac, n_large_components, genus_proxy}; boundary/manifold
  {boundary_edge_frac, boundary_length, n_boundary_loops, winding_inconsistent_edge_frac, duplicate_face_frac};
  self-intersection {self_intersect_face_frac, self_intersect_area_frac}; hidden {hidden_surface_frac};
  thin {thickness_defined_frac, thin_frac_0.005, thin_frac_0.01, thickness_median}; roughness {dihedral_mean_deg,
  dihedral_gt30/60/90_lenfrac, angle_defect_mean, angle_defect_p95}; triangle quality {tri_quality_median,
  tri_quality_p05, sliver_face_frac, sliver_area_frac, edge_length_cv}; size {n_faces}.
- Single-feature rule baselines (threshold = silver OOF-MCC-optimal, same procedure):
  fused_incomplete: genus_proxy, self_intersect_face_frac, hidden_surface_frac; form_surface: thin_frac_0.005,
  dihedral_mean_deg; extra_geometry: n_small_components, small_component_area_frac.
  **Caveat stated now: these features were chosen after seeing milestone-1 golden AUROCs, so their golden numbers
  are optimistic; their unbiased tests are MATE-3D and GSO.**
- Silver vs expert training: repeated (50×) stratified 5-fold CV on golden agreement cells (expert-trained LR,
  threshold = MCC-optimal on inner 5-fold OOF of the training folds); the silver-trained model is scored on the
  same test folds → paired ΔAUROC/ΔMCC distributions. Plus expert-trained (all golden cells) → scored on silver holdout.
- Fusion with repeated CV (50× stratified 5-fold on golden agreement cells, seed 0..49): LR on
  [frozen silver-probe logit, VLM vote] trained on the training folds vs VLM alone on the same test folds,
  for each of the 12 VLMs × 3 primary defects; also training-free OR/AND rules. Report mean ΔMCC (pooled
  predictions per repeat) and the fraction of repeats with Δ>0.
