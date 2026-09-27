# Phase 2 / milestone 1: first real results (3D-DefectBench, expert agreement cells)
Produced by `analysis/main_comparison.py` following `analysis/PREREGISTRATION.md`
(sha256 42280714…, committed in git 024f7eb before any golden evaluation). All numbers are copied from
`main_comparison_expert_agreement.csv`, `paired_tests_probe_vs_vlm.csv` and `macro_mcc.csv`.
GeomProbe-LR = 30 deterministic geometric features -> L2 logistic regression trained on the silver holdout
(548 assets, crowd majority vote); threshold = OOF-MCC-optimal on silver. Nothing was tuned on the golden set.
95% CIs: asset-level bootstrap, 10k resamples.

## Primary defects: MCC [95% CI] (AUROC; VLM AUROC = balanced accuracy)
| Predictor | fused_incomplete (17/101) | form_surface (51/88) | extra_geometry (6/121) | macro (3) |
|---|---|---|---|---|
| **GeomProbe-LR** | 0.241 [0.041, 0.430] (0.695) | 0.288 [0.081, 0.489] (0.710) | 0.287 [0.024, 0.505] (0.742) | 0.272 |
| FaceCount-LR (control) | 0.013 | 0.000 | 0.132 | 0.048 |
| Generator-only (control) | 0.040 | **0.273** | 0.158 | 0.157 |
| Gemini 3.1 Pro | **0.463** [0.350, 0.578] | 0.362 | 0.179 | 0.335 |
| Gemini 2.5 Pro | 0.299 | 0.332 | 0.393 | 0.341 |
| GPT-5.4 | 0.240 | 0.384 | 0.270 | 0.298 |
| Gemini 3.1 Flash-Lite | 0.308 | 0.148 | 0.433 | 0.296 |
| Claude Opus 4.7 | 0.217 | 0.259 | 0.360 | 0.279 |
| GPT-5 Mini | 0.162 | -0.025 | 0.412 | 0.183 |
| Claude Sonnet 4.6 | 0.126 | 0.244 | 0.155 | 0.175 |
| GPT-4o | 0.045 | 0.080 | 0.400 | 0.175 |
| Qwen 3.5 397B | 0.110 | 0.254 | 0.136 | 0.166 |
| Mistral Small 3.1 | 0.022 | -0.154 | 0.225 | 0.031 |
| Qwen2.5-VL-7B | -0.045 | 0.130 | 0.000 | 0.028 |
| Claude Haiku 4.5 | 0.000 | -0.005 | -0.047 | -0.017 |

Negative controls (prompt-conditioned): GeomProbe-LR missing_parts MCC 0.024 [-0.164, 0.229],
pose_placement -0.047 (2 positives). Null, as pre-registered; VLMs are clearly better on missing_parts
(Gemini 2.5 Pro 0.430).
Macro over all 5 geometry defects (paper headline format): GeomProbe-LR 0.158 (rank 8/15), between
Qwen 3.5 397B 0.162 and Claude Sonnet 4.6 0.145; top is Gemini 3.1 Pro 0.298.

## Paired tests (36 primary McNemar tests, BH)
- Significant after BH (q<0.05): probe > Mistral Small (form_surface, q=0.044) and probe < GPT-4o on
  extra_geometry accuracy (q=0.044; the probe over-predicts positives: 22 predicted vs 6 true; ΔMCC CI [-0.48, 0.41]).
- ΔMCC bootstrap CI excluding 0: Gemini 3.1 Pro better on fused_incomplete (Δ=-0.222 [-0.437, -0.014]);
  probe better than Haiku 4.5, Qwen2.5-VL-7B, GPT-5 Mini, Mistral on single defects.
- Versus all strong VLMs (Gemini 2.5/3.1 Pro, GPT-5.4, Opus 4.7): no significant difference (underpowered).
- Caveat: McNemar tests per-cell accuracy, not MCC; with rare positives it penalises the (balanced) probe.

## Controls / caveats
- Generator identity alone reaches MCC 0.273 on form_surface -> the probe's form_surface result is
  largely confounded with generator (A vs B). Within-generator probe AUROC: form_surface 0.796 (A) / 0.664 (B);
  fused_incomplete 0.786 / 0.597; extra_geometry 0.785 / 0.705 (1 positive in B).
- Silver OOF (training-distribution) AUROC is low for form_surface (0.521) but golden AUROC 0.710:
  crowd labels for form_surface are very noisy; training on them is the bottleneck.
- Classic validity checks are nearly vacuous here: 98.4% of the 678 GLBs are watertight, 0% have
  non-manifold edges, 0.4% have winding errors (the benchmark meshes are decimated to <=10.7k faces).
  Informative signals: thin walls, genus, dihedral roughness/angle defect, self-intersection (47% of meshes),
  hidden internal surface (49% have >1%), small components.
- Exploratory (post hoc, selected on the test set, must be confirmed elsewhere): single-feature AUROC
  thin_frac_0.005 = 0.849 for form_surface, 0.778 for fused_incomplete; n_small_components = 0.770 for
  extra_geometry; 33 feature×defect pairs pass BH q<0.05. The same "messiness" features predict several
  defects, i.e. they are not defect-specific.
- Exploratory fusion (probe + VLM vote, trained on silver): helps weak VLMs (mean ΔMCC +0.08 fused,
  +0.11 extra), does not improve the best VLM (Gemini 3.1 Pro fused 0.463 -> 0.416).
