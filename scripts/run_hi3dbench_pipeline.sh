#!/bin/bash
# Wait for downloads, extract GLBs per generator, run probes (decimate to 10k) — Hi3DBench scale-up (M3).
cd "$(dirname "$0")/.."
while ! grep -q ALLDONE logs/dl_hi3dbench.log; do sleep 20; done
mkdir -p data/hi3dbench/meshes
for g in spard trellis triposr instant-mesh hunyuan crm; do
  mkdir -p data/hi3dbench/meshes/$g
  if [ $g = spard ]; then cp -n data/hi3dbench/spard/spard_glb/*.glb data/hi3dbench/meshes/$g/;
  else unzip -q -o -j data/hi3dbench/$g.zip "*.glb" "*.obj" -d data/hi3dbench/meshes/$g/ || echo "UNZIP_FAIL $g"; fi
  echo "$g $(ls data/hi3dbench/meshes/$g | wc -l)"
done
.venv/bin/python scripts/run_probes.py --glob 'data/hi3dbench/meshes/*/*.glb' --out results/probes/hi3dbench_dec10k --strip data/hi3dbench/meshes --decimate 10000 --workers 4 --timeout 900
echo HI3D_PIPELINE_DONE
