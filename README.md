# geometric-probes-3d

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22995915.svg)](https://doi.org/10.5281/zenodo.22995915)

**Deterministic, CPU-only mesh probes for generated 3D assets — and a pre-registered study of what they can and cannot measure.**

Code, configs, per-asset statistics and figures for the paper

> Miles Carter. *Geometric validity is not perceptual quality: what deterministic probes can and cannot measure in generated 3D assets.* 2026. PDF: [`paper/main.pdf`](paper/main.pdf) (26 pages) · Zenodo: [doi:10.5281/zenodo.22995915](https://doi.org/10.5281/zenodo.22995915) · v1.0 Release: see *Releases*.

## TL;DR

- 30 deterministic probe features in six families: **open boundaries, connected components / floaters, self-intersection, hidden (never-visible) surface, thin walls, surface roughness**. Pure Python (Trimesh, PyMeshLab, Embree), no GPU, seconds per mesh.
- **Injected defects on real scans (Google Scanned Objects):** holes and floaters detected perfectly (AUROC 1.000); interpenetrating parts / crossing sheets 0.940–0.986; hidden shells 0.811–0.912; thin fins 0.952 (two larger severities); vertex noise 0.753–0.999.
- **Expert labels (3D-DefectBench):** a logistic regression on the 30 probe features, trained only on crowd labels, reached macro Matthews correlation 0.272 over three primary geometry defects vs 0.335–0.341 for the best vision-language judges; no paired McNemar test against the four strongest judges was significant.
- **Perceived quality:** within generators, correlations with MATE-3D human geometry ratings had |r| ≤ 0.2; on 2,797 Hi3DBench meshes, |r| ≤ 0.111 with an MLLM geometry-plausibility score.
- **Take-away:** use probes as **validity gates and confound checks** (is the file printable / riggable / exportable?), not as perceptual judges.

All numbers above are copied from the paper and trace to CSV files in `results/` (see `paper/number_trace_report.md`).

## Repository layout

| Path | What |
|---|---|
| `geomcheck/probes.py` | the probe implementation (single source of truth for all thresholds, `CONFIG`) |
| `geomcheck/visualize.py` | per-face defect flags + dependency-light renderer used for the gallery figures |
| `scripts/` | dataset downloaders (SHA256-verified), defect injection, probe runner |
| `analysis/` | pre-registrations (`PREREGISTRATION*.md`) and the analysis scripts that produce `results/` |
| `results/` | per-asset probe statistics (`results/probes/*.csv`), summary tables, frozen classifiers, milestone reports |
| `paper/` | LaTeX source, figures, bibliography, compiled PDF, number-trace report |
| `tests/` | synthetic-mesh unit tests for every probe |
| `data/` | licence/provenance notes and small metadata files only — **no third-party meshes** |

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m pytest -q tests/                       # synthetic-mesh unit tests

# Milestone 1 (3D-DefectBench, CC BY-NC 4.0 — downloaded from the source, not redistributed)
bash scripts/dl_defectbench.sh
.venv/bin/python scripts/run_probes.py --glob 'data/3d-defectbench/glb/*/*.glb' --out results/probes/defectbench --workers 4
.venv/bin/python analysis/main_comparison.py

# Milestone 2 (MATE-3D + GSO defect injection)
bash scripts/dl_mate3d.sh && .venv/bin/python scripts/dl_gso.py 120 && bash scripts/run_gso_pipeline.sh
.venv/bin/python analysis/mate3d_validation.py && .venv/bin/python analysis/gso_injection.py && .venv/bin/python analysis/ablations.py

# Milestone 3 (Hi3DBench scale-up)
bash scripts/dl_hi3dbench.sh && bash scripts/run_hi3dbench_pipeline.sh && .venv/bin/python analysis/hi3dbench_validation.py
```

Probe a single mesh of your own:

```bash
.venv/bin/python scripts/run_probes.py --glob 'my_models/*.glb' --out my_results --workers 2
```

## Pre-registration

Each milestone's analysis plan was written and committed before scoring (`analysis/PREREGISTRATION.md`, `PREREGISTRATION_M2.md`, `PREREGISTRATION_M3_hi3dbench.md`; SHA256 of the first plan in `logs/prereg_sha256.txt`). Deviations are reported in the paper appendix.

## Data and licences

See [`data/LICENSES.md`](data/LICENSES.md). In short: **no 3D-DefectBench meshes, renders or labels are redistributed** (CC BY-NC 4.0); derived per-asset statistics, the frozen classifiers fitted on its labels and the Fig. 1b gallery panels are for non-commercial research use only. MATE-3D and GSO are CC BY 4.0 (attribution), Hi3DBench and DB-3DME MIT. Code in this repository is MIT (`LICENSE`); the paper text and the author's own figures are CC BY 4.0.

## Cite

Preprint on Zenodo: [https://doi.org/10.5281/zenodo.22995915](https://doi.org/10.5281/zenodo.22995915) (record: <https://zenodo.org/records/22995915>; all versions: [10.5281/zenodo.22995914](https://doi.org/10.5281/zenodo.22995914)). See also `CITATION.cff`.

```bibtex
@misc{carter2026geomprobes,
  author    = {Carter, Miles},
  title     = {Geometric validity is not perceptual quality: what deterministic probes can and cannot measure in generated 3D assets},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22995915},
  url       = {https://zenodo.org/records/22995915},
  note      = {Code: https://github.com/Stark-Will/geometric-probes-3d}
}
```

## Resources

- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- Author affiliation: [modelfy.art](https://modelfy.art) — online image/text-to-3D service with free in-browser [3D viewer](https://modelfy.art/3d-tools/online-viewer) and [format converter](https://modelfy.art/3d-tools/file-converter) (GLB/OBJ/STL/USDZ). No modelfy.art system, model or data was evaluated in this study (see the paper's competing-interest statement).
- Contact: support@modelfy.art
