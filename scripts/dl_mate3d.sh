#!/bin/bash
# MATE-3D (CC BY 4.0, HF ccccby/MATE-3D) mesh zips via direct resolve URLs (no login).
set -u
cd "$(dirname "$0")/../data/"; mkdir -p mate-3d && cd mate-3d
B=https://huggingface.co/datasets/ccccby/MATE-3D/resolve/main
for f in 3dtopia consistent3d dreamfusion latentnerf magic3d one2345++ sjc textmesh; do
  [ -s "$f.zip" ] && continue
  for t in 1 2 3 4 5; do curl -sfL -m 1800 -o "$f.zip.part" "$B/$f.zip" && mv "$f.zip.part" "$f.zip" && break; sleep $((t*10)); done
  [ -s "$f.zip" ] || echo "FAILED $f"
done
sha256sum *.zip > ZIP_SHA256_local.txt
echo ALLDONE
