#!/bin/bash
# Inject defects into the 120 seeded GSO objects (one process per object), then run probes (no further decimation).
cd "$(dirname "$0")/.."
.venv/bin/python -c "import json;print('\n'.join(json.load(open('data/gso/subset_seed0.json'))))" | \
  xargs -P 3 -I{} sh -c '[ -f "results/gso_meshes/{}/meta.json" ] || timeout 600 .venv/bin/python scripts/gso_inject.py "{}" || echo "INJECT_FAILED {}"'
.venv/bin/python scripts/run_probes.py --glob 'results/gso_meshes/*/*.ply' --out results/probes/gso --strip results/gso_meshes --workers 3
echo GSO_PIPELINE_DONE
