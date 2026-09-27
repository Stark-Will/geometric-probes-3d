#!/bin/bash
# Hi3DBench (MIT, HF 3DTopia/Hi3DBench) generator zips with object-level labels (unique3d 5 GB skipped for disk/time).
set -u
cd "$(dirname "$0")/../data/"; mkdir -p hi3dbench && cd hi3dbench
B=https://huggingface.co/datasets/3DTopia/Hi3DBench/resolve/main
for f in trellis triposr instant-mesh hunyuan crm; do
  [ -s "$f.zip" ] && continue
  for t in 1 2 3 4 5; do curl -sfL -m 3600 -o "$f.zip.part" "$B/$f.zip" && mv "$f.zip.part" "$f.zip" && break; sleep $((t*10)); done
  [ -s "$f.zip" ] || echo "FAILED $f"
done
sha256sum *.zip > ZIP_SHA256_local.txt; echo ALLDONE
