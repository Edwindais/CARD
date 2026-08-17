#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/03_select_on_validation.sh <nnUNet-preprocessed-root> <output-root>

Builds ACDC source-validation caches and selects the four CARD parameters.
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
PREPROCESSED_ROOT="${1:?nnUNet preprocessed root required}"
OUTPUT_ROOT="${2:?output root required}"
PRIMARY_CHECKPOINT="${CARD_PRIMARY_CHECKPOINT:-${OUTPUT_ROOT}/checkpoints/model-base.pt}"
REFERENCE_CHECKPOINT="${CARD_REFERENCE_CHECKPOINT:-${OUTPUT_ROOT}/checkpoints/model-reference.pt}"
CACHE_ROOT="${OUTPUT_ROOT}/source_val_cache"
SELECTION_ROOT="${OUTPUT_ROOT}/selection"

for checkpoint in "${PRIMARY_CHECKPOINT}" "${REFERENCE_CHECKPOINT}"; do
  [[ -f "${checkpoint}" ]] || { echo "ERROR: checkpoint not found: ${checkpoint}" >&2; exit 2; }
done

"${PYTHON_BIN}" "${PACKAGE_ROOT}/inference/build_source_validation_cache.py" \
  --family cardiac --preprocessed-root "${PREPROCESSED_ROOT}" \
  --primary-checkpoint "${PRIMARY_CHECKPOINT}" \
  --reference-checkpoint "${REFERENCE_CHECKPOINT}" \
  --output-root "${CACHE_ROOT}" --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-12}"

"${PYTHON_BIN}" "${PACKAGE_ROOT}/evaluation/select_card_parameters.py" \
  --cache-root "${CACHE_ROOT}" --out-dir "${SELECTION_ROOT}" \
  --dataset acdc --run-kind formal \
  --protocol-manifest "${PACKAGE_ROOT}/metadata/card_source_val_manifest.json"
