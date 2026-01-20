#!/usr/bin/env bash
set -euo pipefail

DATASETS=(cifar10)
COMPRESSION_RATES=(1)
DP_SCALES=(0)
BATCH_SIZES=(8)

for ds in "${DATASETS[@]}"; do
  for cr in "${COMPRESSION_RATES[@]}"; do
    for dp in "${DP_SCALES[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        echo "Submitting H100 job for dataset=${ds}, compression_rate=${cr}, dp_scale=${dp}, batch_size=${bs}"
        sbatch job_dp_h100.sbatch "${ds}" "${cr}" "${dp}" "${bs}"
        # If needed to slow down submissions:
        # sleep 0.2
      done
    done
  done
done
