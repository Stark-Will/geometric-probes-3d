#!/bin/bash
# Second pass: OBJ-format generators (triposr, instant-mesh, crm) — zips contain .obj, not .glb.
cd "$(dirname "$0")/.."
for g in triposr instant-mesh crm; do
  mkdir -p data/hi3dbench/meshes/$g; unzip -q -o -j data/hi3dbench/$g.zip '*.obj' -d data/hi3dbench/meshes/$g/; echo "$g $(ls data/hi3dbench/meshes/$g | wc -l)"
done
while ! grep -q HI3D_PIPELINE_DONE logs/hi3d_pipeline.log; do sleep 20; done
.venv/bin/python scripts/run_probes.py --glob 'data/hi3dbench/meshes/*/*.*' --out results/probes/hi3dbench_dec10k --strip data/hi3dbench/meshes --decimate 10000 --workers 4 --timeout 900
echo HI3D_PIPELINE2_DONE
