#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/02_train.sh <nnUNet-preprocessed-root> <output-root> [primary|reference|both]

Trains the ACDC primary model, reference model, or both. The default is both.
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
PREPROCESSED_ROOT="${1:?nnUNet preprocessed root required}"
OUTPUT_ROOT="${2:?output root required}"
ROLE="${3:-both}"
DATA_ROOT="${PREPROCESSED_ROOT}/Dataset002_ACDC/nnUNetPlans_3d_fullres"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: preprocessed ACDC data not found: ${DATA_ROOT}" >&2; exit 2; }
case "${ROLE}" in primary|reference|both) ;; *) echo "ERROR: role must be primary, reference, or both" >&2; exit 2 ;; esac

export nnUNet_preprocessed="${PREPROCESSED_ROOT}"
export nnUNet_raw="$(dirname "${PREPROCESSED_ROOT}")/nnUNet_raw"
export nnUNet_results="${OUTPUT_ROOT}/checkpoints"
cd "${PACKAGE_ROOT}"

train_primary() {
  "${PYTHON_BIN}" train.py \
    --config-name joint_cfg dataset=acdc_nnunet_dataset \
    model.diffusion_type=categorical dataset.categorical_use_background=true \
    model.categorical_aux_loss_type=dice_ce model.categorical_aux_loss_weight=1.0 \
    model.categorical_aux_dice_weight=0.5 model.categorical_aux_ce_weight=0.5 \
    model.categorical_aux_boundary_weight=0.1 model.categorical_kl_loss_weight=0.1 \
    model.results_folder="${OUTPUT_ROOT}/checkpoints" \
    model.results_folder_postfix=acdc_base_swinunetr \
    model.train_num_steps="${PRIMARY_TRAIN_STEPS:-100000}" \
    model.save_and_sample_every="${SAVE_EVERY:-1000}" model.timesteps=50 \
    model.train_lr="${TRAIN_LR:-1e-4}" dataset.batch_size="${PRIMARY_BATCH_SIZE:-8}" \
    dataset.num_workers="${NUM_WORKERS:-4}" ++dataset.val_num_workers="${VAL_NUM_WORKERS:-2}" \
    model.gradient_accumulate_every="${PRIMARY_GRAD_ACCUM:-2}" \
    model.sampler=categorical model.sampling_steps=5 model.cfg_condition_mask_ratio=0.2 \
    model.denoising_fn=swin_unetr model.swinunetr_type=base model.sample_mode=one_hot
}

train_reference() {
  "${PYTHON_BIN}" train.py \
    --config-name joint_cfg dataset=acdc_nnunet_dataset \
    model.diffusion_type=categorical dataset.categorical_use_background=true \
    model.categorical_aux_loss_type=dice_ce model.categorical_aux_loss_weight=1.0 \
    model.categorical_aux_dice_weight=0.3 model.categorical_aux_ce_weight=0.7 \
    model.categorical_aux_boundary_weight=0.1 model.categorical_kl_loss_weight=0.1 \
    model.results_folder="${OUTPUT_ROOT}/checkpoints" \
    model.results_folder_postfix=acdc_ref_swinunetr_mini \
    model.train_num_steps="${REFERENCE_TRAIN_STEPS:-25000}" \
    model.save_and_sample_every="${SAVE_EVERY:-1000}" model.timesteps=50 \
    model.train_lr="${TRAIN_LR:-1e-4}" dataset.batch_size="${REFERENCE_BATCH_SIZE:-4}" \
    dataset.num_workers="${NUM_WORKERS:-4}" ++dataset.val_num_workers="${VAL_NUM_WORKERS:-2}" \
    model.gradient_accumulate_every="${REFERENCE_GRAD_ACCUM:-4}" \
    model.sampler=categorical model.sampling_steps=5 model.denoising_fn=swin_unetr \
    model.swinunetr_type=mini model.sample_mode=one_hot
}

case "${ROLE}" in
  primary) train_primary ;;
  reference) train_reference ;;
  both) train_primary; train_reference ;;
esac
