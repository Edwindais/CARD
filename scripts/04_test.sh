#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/04_test.sh <nnUNet-preprocessed-root> <output-root>

Evaluates the selected CARD parameters on ACDC-C and M&Ms.
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
PREPROCESSED_ROOT="${1:?nnUNet preprocessed root required}"
PIPELINE_ROOT="${2:?output root required}"
PRIMARY_CHECKPOINT="${CARD_PRIMARY_CHECKPOINT:-${PIPELINE_ROOT}/checkpoints/model-base.pt}"
REFERENCE_CHECKPOINT="${CARD_REFERENCE_CHECKPOINT:-${PIPELINE_ROOT}/checkpoints/model-reference.pt}"
SELECTION_RESULT="${PIPELINE_ROOT}/selection/selected_tuple.json"
OUT_ROOT="${OUT_ROOT:-${PIPELINE_ROOT}/test}"
TUPLES_FILE="${OUT_ROOT}/selected_tuple.json"
MNM_ROOT="${PREPROCESSED_ROOT}/MNM/nnUNetPlans_3d_fullres"

for path in "${PRIMARY_CHECKPOINT}" "${REFERENCE_CHECKPOINT}" "${SELECTION_RESULT}"; do
  [[ -f "${path}" ]] || { echo "ERROR: required file not found: ${path}" >&2; exit 2; }
done
[[ -d "${MNM_ROOT}" ]] || { echo "ERROR: preprocessed M&Ms data not found: ${MNM_ROOT}" >&2; exit 2; }

"${PYTHON_BIN}" "${PACKAGE_ROOT}/evaluation/assemble_selected_tuples.py" \
  --cardiac "${SELECTION_RESULT}" --output "${TUPLES_FILE}"
mkdir -p "${OUT_ROOT}/metrics" "${OUT_ROOT}/logs"
MAX_CASE_ARGS=()
[[ -n "${MAX_CASES:-}" ]] && MAX_CASE_ARGS+=(--max-cases "${MAX_CASES}")

run_block() {
  local block="$1"
  local data_args=()
  case "${block}" in
    acdc-c) data_args=(--cardiac-root "${PREPROCESSED_ROOT}") ;;
    mnm) data_args=(--mnm-root "${MNM_ROOT}") ;;
  esac
  "${PYTHON_BIN}" -u "${PACKAGE_ROOT}/inference/reproduce_results.py" \
    --block "${block}" "${data_args[@]}" \
    --primary-checkpoint "${PRIMARY_CHECKPOINT}" \
    --reference-checkpoint "${REFERENCE_CHECKPOINT}" \
    --tuples-file "${TUPLES_FILE}" --output-dir "${OUT_ROOT}/metrics/${block}" \
    --batch-size "${BATCH_SIZE:-16}" --num-workers "${NUM_WORKERS:-12}" \
    "${MAX_CASE_ARGS[@]}" 2>&1 | tee "${OUT_ROOT}/logs/${block}.log"
}

run_block acdc-c
run_block mnm

if [[ -z "${MAX_CASES:-}" ]]; then
  "${PYTHON_BIN}" "${PACKAGE_ROOT}/evaluation/collect_results.py" \
    "${OUT_ROOT}"/metrics/*/per_case_metrics.csv --output-dir "${OUT_ROOT}/summary"
else
  echo "Partial test complete; aggregation skipped because MAX_CASES is set."
fi

echo "Evaluation complete: ${OUT_ROOT}"
