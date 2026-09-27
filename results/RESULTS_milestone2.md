# Milestone 2 results (pre-registration: analysis/PREREGISTRATION_M2.md, git 78d6623, committed 16:44 UTC+8
# before any MATE-3D scoring, GSO injection or ablation)
All numbers are copied from the CSVs named in each section; nothing tuned on test data.

## 1. MATE-3D external validation (results/mate3d/mate3d_dec10k_correlations.csv; 1,280/1,280 meshes OK)
Spearman with human Geometry MOS; predicted sign negative. B = within-generator (PRIMARY), 95% CI = prompt-cluster
bootstrap (10k); p = within-generator permutation (10k); BH over the 9 primary predictors.
| Predictor | A pooled ρ | **B within-gen ρ [95% CI]** | q_BH | C within-prompt τ | D two-way FE | verdict |
|---|---|---|---|---|---|---|
| thin_frac_0.005 | -0.181 | **-0.195 [-0.255, -0.132]** | 0.001 | -0.114 | -0.144 | replicates |
| genus_proxy | -0.322 | **-0.110 [-0.165, -0.055]** | 0.001 | -0.263 | -0.118 | replicates |
| self_intersect_face_frac | -0.291 | **-0.094 [-0.144, -0.041]** | 0.003 | -0.245 | -0.057 | replicates |
| dihedral_mean_deg (roughness) | -0.293 | -0.041 [-0.103, 0.020] | 0.191 | -0.241 | -0.052 | partial (pooled only → generator-confounded) |
| hidden_surface_frac | -0.297 | +0.028 [-0.033, 0.088] | 0.318 | -0.257 | -0.041 | partial (pooled only → generator-confounded) |
| frozen P(form_surface) | **+0.150** | -0.046 [-0.108, 0.018] | 0.161 | +0.120 | -0.052 | fails |
| frozen P(fused_incomplete) | -0.052 | -0.035 [-0.092, 0.023] | 0.234 | -0.060 | -0.042 | partial |
| frozen P(extra_geometry) | -0.218 | -0.073 [-0.132, -0.012] | 0.018 | -0.181 | -0.094 | replicates |
| frozen mean defect score | +0.011 | -0.081 [-0.134, -0.027] | 0.010 | -0.000 | -0.114 | replicates |
Effects are small (|ρ| ≤ 0.2 within generator) and heterogeneous across generators (per-generator table:
strongest in latentnerf/one2345++/sjc, reversed sign for several predictors in magic3d/textmesh).
Exploratory best (not pre-registered as primary): n_components ρ_B = -0.214, n_large_components -0.199,
largest_component_area_frac +0.192 (all BH q<0.001).
Secondary, raw (undecimated) meshes (mate3d_raw_correlations.csv): same verdicts for thin (-0.178), genus (-0.109),
self-intersection (-0.063), mean defect score (-0.211); dihedral and hidden again fail within generator; P(form_surface)
fails; P(fused_incomplete) now replicates weakly (-0.060, q=0.048).

## 2. GSO synthetic injection (results/gso/gso_detection.csv; 120 objects × 22 variants, 0 failures, 0 skips)
Target check, AUROC vs 120 clean meshes [95% CI], S1/S2/S3:
| Defect → check | S1 | S2 | S3 | sens@τ=0 (S1–S3) | spec@τ=0 |
|---|---|---|---|---|---|
| holes → boundary_edge_frac | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 |
| floaters → n_small_components | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 |
| interpenetrating part → self_intersect_face_frac | 0.971 | 0.978 | 0.986 | 1.00 | 0.79 |
| interpenetrating part → hidden_surface_frac | 0.835 | 0.897 | 0.937 | 1.00 | 0.58 |
| crossing sheet → self_intersect_face_frac | 0.940 | 0.965 | 0.977 | 1.00 | 0.79 |
| hidden internal shell → hidden_surface_frac | 0.811 | 0.869 | 0.912 | 1.00 | 0.58 |
| thin fin (t=0.008/0.004/0.002) → thin_frac_0.005 | 0.651* | 0.952 | 0.952 | 0.98–1.00 | 0.27 |
| noise σ=0.001/0.003/0.01 → dihedral_mean_deg | 0.753 | 0.971 | 0.999 | n/a | n/a |
*S1 fin is thicker than the 0.005 threshold by design; thin_frac_0.01 detects it.
- Specificity at τ=0 is limited because clean decimated scans already contain the signal (21% self-intersect,
  hidden>0 in 42%, thin>0 in 73%).
- Benchmark-calibrated τ_c (P95 over 200 crowd-"defect-free" 3D-DefectBench assets) is very high
  (self-intersection 4.4% of faces, hidden 12.8%, 3 small components, thin 5.0%): sensitivity at τ_c is low for
  self-intersection (≤0.10), hidden (≤0.32), floaters (0 for 1–3 floaters) → generated assets that humans call
  "defect-free" are geometrically dirty; human judgement and geometric validity diverge.
- Cross-talk (S3 AUROC, gso_crosstalk_auroc.csv): the checks are sensitive but not type-specific:
  hidden_surface_frac responds to interpenetration 0.94, crossing sheets 0.89, noise 0.86, thin fins 0.89;
  self-intersection responds to noise 1.00 and fins 0.96; boundary/floater checks are specific.

## 3. Ablations (results/ablations/)
Golden agreement cells, trained on silver (ablation CIs: 2,000 bootstrap resamples).
- Family-only models: topology-only (6 features) AUROC fused 0.801, form 0.770, extra 0.786 (MCC 0.395/0.572/0.188)
  ≥ full 30-feature model (0.695/0.710/0.742). Boundary/manifold-only ≈ chance (0.53–0.57: meshes are watertight).
  NB: choosing the best ablation on golden would be test-set selection; reported as ablation only.
- LOFO: no single family removal changes AUROC by more than ±0.06; MCC is unstable because the silver
  OOF threshold is fragile (e.g. form_surface −self_intersection / −tri_quality → threshold 1.0, MCC 0).
- Single-feature rules (features chosen post hoc in M1, so optimistic): genus 0.763, hidden 0.755,
  self-intersection 0.698 (fused); thin 0.849, dihedral 0.691 (form); n_small_components 0.770 (extra).
- Silver vs expert training (50× 5-fold CV on golden cells): expert-trained AUROC fused 0.719 vs silver 0.695
  (Δ +0.024, >0 in 68% of repeats); form 0.729 vs 0.710 (+0.019, 72%); extra 0.502 vs 0.742 (−0.240, 0%; 6 positives).
  Expert-trained → silver holdout: AUROC 0.60/0.56/0.47 (vs silver OOF 0.70/0.52/0.69). 88–121 expert cells are too
  few to beat 548 noisy silver labels; no evidence expert training helps.
- Learned fusion (LR on probe logit + VLM vote, 50× 5-fold CV on golden): mean ΔMCC over 12 VLMs fused −0.008,
  form +0.107, extra 0.000; for the 4 strongest VLMs Δ = −0.060/−0.014/−0.021 → learned fusion does not help strong VLMs.
- Training-free AND rule (VLM flag ∧ probe flag; probe threshold frozen from silver): ΔMCC > 0 in 29/36 VLM×defect
  pairs (4 negative, 3 zero); mean over the 4 strongest VLMs fused 0.305→0.354, form 0.334→0.366, extra 0.301→0.518;
  but only 1/36 significant after BH (Mistral, fused). Consistent direction, underpowered.
