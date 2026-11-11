#!/usr/bin/env bash
# Only process-local library settings; no pip/conda mutation.
set -euo pipefail
DUCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DUCO_PYTHON="${DUCO_PYTHON:-python}"
DUCO_PREFIX="$("$DUCO_PYTHON" -c 'import sys; print(sys.prefix)')"
if [[ -f "$DUCO_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$DUCO_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
cd "$DUCO_ROOT"
exec "$DUCO_PYTHON" "$DUCO_ROOT/script/duco_wipe.py" "$@"
