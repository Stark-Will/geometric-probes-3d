#!/bin/bash
# Reproducible download of 3D-DefectBench via direct resolve URLs (no HF login), with SHA256 verification.
set -u
cd "$(dirname "$0")/../data/"; mkdir -p 3d-defectbench && cd 3d-defectbench
B=https://huggingface.co/datasets/zzhao0500/3D-DefectBench/resolve/main
for s in SHA256SUMS GLB_SHA256SUMS; do [ -s "$s" ] || curl -sfL -m 120 -o "$s" "$B/$s"; done
for f in $(tr -d '\r' < SHA256SUMS | awk '{print $2}') $(tr -d '\r' < GLB_SHA256SUMS | awk '{print $2}'); do
  [ -s "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  for t in 1 2 3 4 5; do curl -sfL -m 300 -o "$f.part" "$B/$f" && mv "$f.part" "$f" && break; sleep $((t*5)); done
  [ -s "$f" ] || echo "FAILED $f"
  sleep 0.3
done
tr -d '\r' < SHA256SUMS | sha256sum -c --quiet && echo "SHA256SUMS OK"
tr -d '\r' < GLB_SHA256SUMS | sha256sum -c --quiet && echo "GLB_SHA256SUMS OK"
echo ALLDONE
