# Milestone 3 results: Hi3DBench within-generator scale-up (SECONDARY evidence)
Pre-registration: analysis/PREREGISTRATION_M3_hi3dbench.md (git c003104, 17:36 UTC+8); analysis script committed in
b528343 before scoring. Numbers copied from results/hi3dbench/hi3dbench_correlations.csv, hi3dbench_per_generator.csv,
hi3dbench_meta.json. **Targets are Hi3DEval M²AP multi-agent MLLM pseudo-labels (object-level `geometry_score` =
"Geometry Plausibility"; `geo_detail_score` = geometric detail), not human ratings.**

Data: 3,060 meshes probed (spard, trellis, triposr, instant-mesh, hunyuan: 510 each; crm: 510 of which 247 labelled),
0 probe failures; 2,797 labelled meshes analysed over 510 image prompts. Preprocessing identical to MATE-3D
(weld → PyMeshLab topology-preserving QEM to 10k faces). Median probe time 1.36 s/mesh (P95 3.02 s), peak RSS ≤ 345 MB.

## Primary target: geometry_score (B = within-generator, PRIMARY; 95% CI prompt-cluster bootstrap 10k; p = within-generator permutation 10k; BH over 4)
| Predictor | A pooled ρ [CI] | **B within-gen ρ [CI]** | q_BH | C within-prompt τ [CI] | D two-way FE [CI] | verdict |
|---|---|---|---|---|---|---|
| thin_frac_0.005 | +0.009 [-0.033, 0.051] | **-0.111 [-0.157, -0.066]** | <0.001 | **+0.085** [0.051, 0.119] | -0.023 [-0.067, 0.022] | replicates |
| genus_proxy | -0.028 [-0.068, 0.015] | **-0.063 [-0.106, -0.018]** | 0.001 | +0.033 [-0.003, 0.069] | +0.005 [-0.035, 0.046] | replicates |
| self_intersect_face_frac | +0.022 [-0.019, 0.062] | **-0.042 [-0.084, -0.001]** | 0.026 | **+0.074** [0.034, 0.114] | **+0.044** [0.005, 0.083] | replicates (B only) |
| n_components | **-0.220** [-0.258, -0.183] | **-0.066 [-0.108, -0.025]** | 0.001 | **-0.175** [-0.209, -0.141] | -0.006 [-0.046, 0.034] | replicates |

- All four pre-registered predictors meet the replication rule on B, but effects are small (|ρ_B| ≤ 0.111).
- Heterogeneous: per-generator ρ is strongest for crm (-0.221 to -0.312, n=247) and instant-mesh (thin -0.214, genus -0.170);
  near zero for hunyuan, spard, trellis.
- Not robust to prompt adjustment: the two-way FE estimand D (generator + prompt) is ≈0 for thin, genus and components and
  **positive** for self-intersection (+0.044, CI excludes 0).
- Between-generator direction is opposite for thin walls and self-intersection: within a prompt, the generator whose mesh has
  more thin walls/self-intersection tends to receive a *higher* geometry score (C = +0.085 / +0.074). E.g. trellis has by far
  the highest mean thin fraction (0.185 vs ≤0.058 for the others) and the second-highest mean geometry score (5.863).
- n_components (floaters/fragments; M²AP's Geometry Plausibility criterion explicitly mentions floating parts) has the largest
  pooled and within-prompt associations (A -0.220, C -0.175) but only -0.066 within generator.

## Secondary target: geo_detail_score — all four REVERSE (positive within generator)
thin +0.097 [0.047, 0.147], genus +0.134 [0.086, 0.180], self-intersection +0.115 [0.071, 0.158], n_components +0.046 [0.001, 0.092]
(all q ≤ 0.016). Pooled: +0.159, +0.231, +0.216, -0.135. I.e. the "defect" signals co-occur with geometric detail/complexity as
judged by the MLLM pipeline — a complexity confound: a validity gate that penalises thin walls, genus or self-intersection will
also penalise detailed meshes.

## Interpretation (secondary; agreement with an automated judge, not human validation)
Direction on the primary target matches MATE-3D within generator (thin, genus, self-intersection negative), with smaller effects,
strong generator heterogeneity, and no robustness to prompt adjustment. Together with the reversal on geo_detail_score this
strengthens the paper's thesis that geometric-validity signals are only weakly and inconsistently related to perceived
(or MLLM-perceived) quality.
