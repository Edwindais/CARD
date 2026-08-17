#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/01_preprocess.sh <acdc-raw-root> <mnm-raw-root> <work-root>

Converts ACDC and M&Ms into the data layout used by training and evaluation.
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
ACDC_RAW_ROOT="${1:?ACDC raw root required}"
MNM_RAW_ROOT="${2:?M&Ms raw root required}"
WORK_ROOT="${3:?work root required}"

for path in "${ACDC_RAW_ROOT}" "${MNM_RAW_ROOT}"; do
  [[ -d "${path}" ]] || { echo "ERROR: data directory not found: ${path}" >&2; exit 2; }
done

export nnUNet_raw="${WORK_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${WORK_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${WORK_ROOT}/nnUNet_results"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

cd "${PACKAGE_ROOT}"
"${PYTHON_BIN}" preprocess_data.py --acdc_root "${ACDC_RAW_ROOT}"
"${PYTHON_BIN}" preprocessing/preprocess_mnm.py \
  --raw-root "${MNM_RAW_ROOT}" --work-root "${WORK_ROOT}"
"${PYTHON_BIN}" preprocessing/verify_preprocessed.py --work-root "${WORK_ROOT}"
