# CARD: Calibration via Agreement in Reverse Diffusion for Out-of-Domain MRI Segmentation

This repository provides the official implementation of CARD on ACDC-C and
M&Ms. CARD parameters are fitted exclusively on ACDC source-validation data
before evaluation. Pretrained checkpoints are provided for inference, and
training code is also included.

**Paper:** link to be added · **Models:** [Hugging Face](https://huggingface.co/edwinsssssa/CARD) ·
**License:** [MIT](LICENSE)

## Setup

```bash
conda create -n card-cardiac python=3.10
conda activate card-cardiac
pip install -r requirements.txt
```

Training and inference require a CUDA GPU. `TorchIO==1.2.0` is pinned because
the ACDC-C transformations depend on its implementation.

## Quick start

1. Download ACDC and M&Ms.
2. Download the CARD primary and reference checkpoints.
3. Preprocess both datasets with `scripts/01_preprocess.sh`.
4. Fit CARD parameters on ACDC source validation with
   `scripts/03_select_on_validation.sh`.
5. Evaluate ACDC-C and M&Ms with `scripts/04_test.sh`.

## Data

Download [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/onlinePlatform.html)
and [M&Ms](https://www.ub.edu/mnms/), accept their respective data terms, and
extract both archives locally. ACDC-C is generated from ACDC test images during
evaluation and does not require a separate download.

The preprocessing command expects these roots:

```text
ACDC/
└── training/
    └── patient001/
        ├── Info.cfg
        ├── patient001_frame01.nii.gz
        └── patient001_frame01_gt.nii.gz

MNM/
├── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
├── Training/
├── Validation/
└── Testing/
```

## Evaluation

Choose one work directory for preprocessed data and one output directory for
models and results:

```bash
WORK=/path/to/work
OUTPUT=/path/to/output
```

### 1. Preprocess ACDC and M&Ms

```bash
bash scripts/01_preprocess.sh /path/to/ACDC /path/to/MNM "$WORK"
```

The command creates `nnUNet_raw/`, `nnUNet_preprocessed/`, and
`nnUNet_results/` below `$WORK`, then checks the expected splits and data layout.

### 2. Download the models

Download `model-base.pt` and `model-reference.pt` from the accompanying Hugging
Face model repository, then place them at the following paths:

```text
$OUTPUT/
└── checkpoints/
    ├── model-base.pt
    └── model-reference.pt
```

### 3. Fit the four CARD parameters

Fit the four CARD parameters on the ACDC source-validation set:

```bash
bash scripts/03_select_on_validation.sh \
  "$WORK/nnUNet_preprocessed" "$OUTPUT"
```

The fitted parameters are saved to `$OUTPUT/selection/selected_tuple.json`.

### 4. Evaluate ACDC-C and M&Ms

```bash
bash scripts/04_test.sh "$WORK/nnUNet_preprocessed" "$OUTPUT"
```

Per-case metrics are saved under `$OUTPUT/test/metrics/`. The final group-equal
Dice, ROI-ECE, ROI-SCE, ROI-ACE, and ROI-NLL values are written to
`$OUTPUT/test/summary/summary.csv`.
ACDC-C is averaged over Clean, Bias, Motion, Ghosting, and Spike; M&Ms is
averaged over scanner vendors A, B, C, and D.

## Expected results

Evaluation produces the following group-equal results. Calibration errors are
reported in percent.

| Dataset | Dice ↑ | ROI-ECE ↓ | ROI-SCE ↓ | ROI-ACE ↓ | ROI-NLL ↓ |
|---|---:|---:|---:|---:|---:|
| ACDC-C | 0.8763 | 5.41 | 3.36 | 3.12 | 0.371 |
| M&Ms | 0.7689 | 4.98 | 3.19 | 2.93 | 0.377 |

## Optional model training

To train primary and reference models instead of using the pretrained
checkpoints:

```bash
bash scripts/02_train.sh "$WORK/nnUNet_preprocessed" "$OUTPUT" both
```

The primary model trains for 100,000 steps. The smaller reference model trains
for 25,000 steps. Checkpoints are saved every 1,000 steps.

## Repository structure

```text
scripts/
  01_preprocess.sh              preprocessing entrypoint
  02_train.sh                   optional training entrypoint
  03_select_on_validation.sh    CARD parameter selection
  04_test.sh                    ACDC-C and M&Ms evaluation

preprocess_data.py              ACDC preprocessing
preprocessing/                  M&Ms preprocessing and data checks
train.py                        primary/reference model training
model/                          categorical diffusion and denoising networks
inference/                      cache generation and fixed inference
evaluation/                     parameter optimization and result aggregation
config/                         dataset and model configurations
metadata/                       locked validation split and perturbations
```

## Citation

The citation will be added when the paper record becomes publicly available.

## License

This project is released under the [MIT License](LICENSE).
