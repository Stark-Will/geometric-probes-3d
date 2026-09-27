# Pre-registration, milestone 3: Hi3DBench within-generator scale-up (SECONDARY evidence)
Written and committed 2026-09-27 (UTC+8) before any Hi3DBench mesh was probed or scored (zips still downloading).
Labels: Hi3DBench `object-level.json` scores. Per the dataset documentation these are produced by the Hi3DEval
automated (MLLM-based) pipeline, i.e. **pseudo-labels, not human ratings**; results are framed as secondary
(agreement with an automated judge), never as human validation.

Data: generators with object-level labels whose meshes are in the Hi3DBench repo and fit the time/disk budget:
spard (510), trellis (510), triposr (510), instant-mesh (510), hunyuan (510), crm (247 labelled). unique3d (5 GB zip)
skipped for budget (decided before scoring). Mesh i of generator g ↔ key `image2shape_{g}_{i}`; prompt id = i
(shared image prompts across generators). Only meshes with a label are analysed.
Preprocessing: identical to milestone 2 (weld → PyMeshLab topology-preserving quadric decimation to 10,000 faces).
Target: `geometry_score` (primary); `geo_detail_score` (secondary).
Predictors (the four signals that replicated on MATE-3D or its exploratory component-count signal), predicted sign
negative (more → worse): thin_frac_0.005, genus_proxy, self_intersect_face_frac, n_components.
Analyses: A pooled Spearman; **B within-generator (PRIMARY)** = Pearson of within-generator percentile ranks;
C within-prompt Kendall τ-b across generators (prompts with ≥3 generators), averaged; D OLS residualisation of global
ranks on generator + prompt dummies. CIs: prompt-cluster bootstrap (10,000 for A/B/C; 2,000 for D), seed 2029.
p for B: within-generator permutation (10,000, seed 2030). BH over the 4 predictors (B).
Replication criterion identical to PREREGISTRATION_M2 §1.
