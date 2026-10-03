# Limitations: geometric validity ≠ perceptual quality

The probes answer questions with a definite geometric answer: does the mesh have holes, floaters,
self-intersections, hidden internal surface, walls too thin to print? They do **not** tell you
whether a model looks good, matches its prompt, or would be preferred by a person. The paper behind
`geomcheck` was designed to measure exactly this gap.

## What the study found

The following is quoted from the repository README, which copies every number from the paper; each
traces to a CSV file in `results/` (see `paper/number_trace_report.md`).

> - **Injected defects on real scans (Google Scanned Objects):** holes and floaters detected
>   perfectly (AUROC 1.000); interpenetrating parts / crossing sheets 0.940–0.986; hidden shells
>   0.811–0.912; thin fins 0.952 (two larger severities); vertex noise 0.753–0.999.
> - **Expert labels (3D-DefectBench):** a logistic regression on the 30 probe features, trained only
>   on crowd labels, reached macro Matthews correlation 0.272 over three primary geometry defects
>   vs 0.335–0.341 for the best vision-language judges; no paired McNemar test against the four
>   strongest judges was significant.
> - **Perceived quality:** within generators, correlations with MATE-3D human geometry ratings had
>   |r| ≤ 0.2; on 2,797 Hi3DBench meshes, |r| ≤ 0.111 with an MLLM geometry-plausibility score.
> - **Take-away:** use probes as **validity gates and confound checks** (is the file printable /
>   riggable / exportable?), not as perceptual judges.

In the paper's own words: "The honest summary is that the probe is comparable to mid-tier VLM
judges, not that it beats them."

## Practical consequences

- **Use the probes as gates, not scores.** A mesh with open boundaries, floaters or hidden shells has
  a concrete problem for printing, rigging or export. A mesh that passes every check can still be an
  ugly or wrong model.
- **Do not rank generators by probe values.** Within-generator correlations with human geometry
  ratings were weak, and on Hi3DBench they were heterogeneous across generators. The paper also
  notes that a validity gate fitted to geometry plausibility can punish complexity.
- **Watch for confounds.** Probe features differ systematically between generators; the paper
  reports generator-only and face-count-only controls and within-generator AUROCs so that a
  between-generator difference is not mistaken for a geometric effect. Do the same in your own
  comparisons.

## Caveats from the paper

- **Small expert set.** The expert-agreement cells are small (extra geometry has six positives,
  pose/placement two), and the paired tests against strong VLMs were underpowered; non-significance
  is absence of evidence, not evidence of equivalence.
- **Injections are simpler than real failures.** Each injection operator adds one kind of damage to
  a cleaned, decimated scan. Generated meshes are often broken in several ways at once, and
  detection against clean scans is easier than ranking two generated meshes.
- **Decimation changes some features.** Thickness and intersection counts can move with decimation.
  The undecimated MATE-3D run did not change the verdicts, but the injection study was not repeated
  without decimation.
- **Library dependence.** Ray probes depend on a fixed 20,000-point sample and Embree;
  self-intersection depends on PyMeshLab. Bit-identity is not claimed across library versions that
  were not run.
- **Geometry only.** Texture, materials and prompt alignment are out of scope.
- **Secondary evidence.** Hi3DBench labels are MLLM pseudo-labels, not human ratings.

See Section 7 (Limitations) of the [paper](https://doi.org/10.5281/zenodo.22995915) for the full
discussion.
