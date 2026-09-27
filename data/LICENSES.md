# Data licences and provenance (verified 2026-09-27, UTC+8)

| Dataset | Source | Licence (where verified) | Gated? | Obtained | Use in paper |
|---|---|---|---|---|---|
| **3D-DefectBench** | HF `zzhao0500/3D-DefectBench` (arXiv 2607.10826) | **CC BY-NC 4.0** (`LICENSE` file + README YAML + README "License & citation": covers GLBs, labels, prompts, predictions) | no | all 41 files in `SHA256SUMS` + 678 GLBs in `GLB_SHA256SUMS` (129 golden, 549 silver-holdout), SHA256-verified; `scripts/dl_defectbench.sh` | main validation set (expert + silver labels, 12 VLM predictions) |
| **Hi3DBench** | HF `3DTopia/Hi3DBench` (NeurIPS 2025 D&B) | **MIT** (README YAML) | no | README, object-level.json, part-level.json, material-subject.json; zips for spard, trellis, triposr, instant-mesh, hunyuan, crm (3,060 meshes probed, 2,797 labelled analysed; not redistributed, only per-asset statistics) | scale-up (W6). NB: its scores are MLLM pseudo-labels, not human |
| 3DGen-Bench | HF `3DTopia/3DGen-Bench` | MIT (`LICENSE` file downloaded) | **yes: data files return 401** | LICENSE only | **needs the user's HF login + accepting the gate; not downloaded** |
| MATE-3D | HF `ccccby/MATE-3D` (ICCV 2025, arXiv 2412.11170) | **CC BY 4.0** (HF README YAML; GitHub repo has no LICENSE file) | no | README, `MOS_MATE_3D.xlsx`, prompt list, all 8 mesh zips (2.26 GB; local sha256 in ZIP_SHA256_local.txt; `scripts/dl_mate3d.sh`), 1,280 OBJ extracted | **external validation (milestone 2)**; attribution required |
| DB-3DME | HF `nsjia/DB-3DME` (arXiv 2606.10142, CVPRW 2026) | **MIT** (HF README YAML); GitHub has no LICENSE file; paper says "licence that prohibits misuse and requires attribution" | no | README, metadata.csv | **GIFs only, no meshes released** -> cannot run geometric probes; usable only as a VLM-side reference. Correction to proposal (which assumed meshes / CC BY 4.0) |
| GSO (Google Scanned Objects) | Gazebo Fuel, owner GoogleResearch | **CC BY 4.0** (verified per model via Fuel API `license_name` for all 1,033 models) | no | 120 seeded models (`scripts/dl_gso.py`, `data/gso/subset_seed0.json`) | synthetic defect injection (milestone 2); attribution required |

## Obligations
- **3D-DefectBench (CC BY-NC 4.0)**: attribution required (cite the paper and dataset). This repository does **not**
  redistribute any 3D-DefectBench GLB, render, label file or prompt; `scripts/dl_defectbench.sh` downloads them from the
  original source with SHA256 verification. The derived per-asset probe statistics (`results/probes/defectbench.csv`),
  the frozen probe classifiers in `results/frozen_models/` (fitted on 3D-DefectBench labels) and the gallery panels of
  Fig. 1b that show 3D-DefectBench meshes are derived from CC BY-NC 4.0 material and are provided for
  **non-commercial research use only**, under the same terms.
- **MATE-3D, Google Scanned Objects (CC BY 4.0)**: attribution required; `data/mate-3d/MOS_MATE_3D.xlsx` and
  `prompt_MATE_3D.json` are redistributed unchanged from HF `ccccby/MATE-3D` with attribution.
- **Hi3DBench, DB-3DME (MIT)**: only per-asset statistics / metadata are included; no meshes are redistributed.
- A byte-identical mesh appears in two 3D-DefectBench splits: `glb/golden/608.glb` == `glb/silver/609.glb` (same SHA256).
  Asset 609 is excluded from all training/threshold fitting.
- No modelfy.art generations, models or user data were used in the study.
