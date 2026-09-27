# Pre-registered analysis plan — Phase 2, milestone 1
Written 2026-09-27 ~16:30 (UTC+8) **before** any geometric probe was run on a 3D-DefectBench expert (golden)
asset for evaluation purposes and before any probe-vs-label statistic was computed.
What was already looked at when writing this: (i) VLM prediction files (to replicate the paper's
macro-MCC numbers exactly = label-handling check, done: all 12 match to 3 decimals), (ii) label
prevalences, (iii) probe output on 11 golden GLBs **for engineering smoke-testing only** (values, no labels).
The SHA256 of this file is logged in `logs/prereg_sha256.txt` at the time of writing.

## Data
- Train/threshold split: 3D-DefectBench **silver holdout** (549 assets with released GLBs; labels =
  crowd majority vote). Asset `609` (byte-identical mesh to golden `608`) is **excluded** from training.
- Test split: 3D-DefectBench **golden** (129 assets, 2 experts).
- Primary expert target: cells where both experts agree (`agreement_rate == 1`; label = `majority_vote`),
  exactly the cells for which the benchmark releases VLM predictions (877 cells; config `c004`).
- Secondary expert target (probes only; VLM predictions not released for disagreement cells):
  union = positive if either expert flags (`mean_value > 0`), all 129 assets.

## Defects
- Primary (not prompt-conditioned, geometry): `q_fused_incomplete`, `q_form_surface`, `q_extra_geometry`.
- Negative controls (prompt-conditioned; a prompt-blind geometric probe is expected to be ~null):
  `q_missing_parts`, `q_pose_placement` (the latter has 2 positives in the agreement set; reported, not tested).
- Texture defects: out of scope for geometry probes (not evaluated).

## Probe features (geomcheck v0.1.0, config in geomcheck/probes.py::CONFIG, frozen)
All scalar outputs of `compute_all` except bookkeeping (`orig_bbox_diag`, `status`, `path`, `seconds`,
`peak_rss_mb`, `n_edges`, `n_vertices`, `watertight`, `self_intersect_area_frac` duplicates are kept).
Transform: `euler_characteristic` -> `genus_proxy = log1p(max(0, 2*n_components - euler)/2)`;
all other non-negative features -> `log1p(x)`; then z-score (fit on train only). Features with zero
variance on train are dropped. Missing values (probe failures) -> train median.

## Primary predictor ("GeomProbe-LR"), one per defect
`StandardScaler + LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=5000)`.
Threshold: maximise MCC over the 5-fold stratified out-of-fold probabilities on the silver holdout
(`StratifiedKFold(5, shuffle=True, random_state=0)`; candidate thresholds = unique OOF probabilities).
Refit on all silver-holdout assets; apply model + threshold unchanged to golden assets.
No tuning of C, features, or thresholds on golden. Any later change = new, separately reported analysis.

## Baselines / controls (pre-registered)
- 12 VLM judges (released predictions, c004, parse_ok only).
- `FaceCount-LR`: same pipeline with `log1p(n_faces)` only (mesh resolution confound).
- `Generator-only`: predict with `model_version` (model A/B) only (generator-identity confound);
  and within-generator AUROC of GeomProbe-LR reported separately for model A and model B.

## Metrics
Per defect: MCC, F1 (positive class), AUROC (probability score for probes; for binary VLM predictions
AUROC = balanced accuracy, stated as such). Also macro-MCC over the 5 scored geometry defects to
compare with the paper's headline.

## Statistics
- 95% CIs: asset-level (cluster) percentile bootstrap, 10,000 resamples, seed 12345.
- Paired comparison GeomProbe-LR vs each VLM per defect: bootstrap CI of ΔMCC (same resamples) and
  exact McNemar test on per-cell correctness (two-sided, binomial on discordant pairs).
- Multiplicity: Benjamini–Hochberg (q = 0.05) over the 36 primary McNemar tests (12 VLMs × 3 primary
  defects); negative-control tests corrected separately.
- Exploratory (labelled as such): per-feature AUROC on golden agreement cells, BH over features×defects.
- Exploratory: late fusion (probe probability + VLM binary vote, LR trained on silver holdout).

## Power caveat (stated a priori)
Agreement-cell positives: fused_incomplete 17/101, form_surface 51/88, extra_geometry 6/121,
missing_parts 12/104, pose 2/125. `q_extra_geometry` in particular is severely under-powered; a null
result there is uninformative, not evidence of absence.
