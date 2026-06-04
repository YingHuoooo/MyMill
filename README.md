# Calibrated Risk Learning for Neural Accessibility Prediction

This repository contains the cleaned research code for a calibrated neural accessibility prediction pipeline for subtractive manufacturing. The code builds on an octree UNet accessibility predictor and adds a post hoc risk decision layer designed for dual head accessibility outputs.

The repository is intentionally code only. Pretrained checkpoints, generated logs, raw manufacturing data, processed point files, visualization outputs, and manuscript drafts are not tracked.

## Repository Layout

```text
.
├── projects/
│   ├── configs/seg_deepmill.yaml          # Main segmentation configuration
│   ├── datasets/                          # Dataset wrappers
│   ├── ocnn/                              # Octree network modules
│   ├── thsolver/                          # Training, testing, and config utilities
│   ├── tools/seg_deepmill_cutter.py       # Data preprocessing helper
│   ├── segmentation.py                    # Training, testing, and calibrated evaluation logic
│   ├── run_seg_deepmill.py                # Training and standard testing launcher
│   ├── run_mc_cp_crc.py                   # Calibrated risk evaluation launcher
│   ├── run_paper_experiments.py           # Multi seed experiment runner
│   ├── summarize_paper_experiments.py     # Paper table summarizer
│   └── dataset_statistics.py              # Dataset statistics utility
├── environment.yml                        # Conda environment template
├── requirements.txt                       # Pip dependency list
└── README.md
```

## Method Overview

The core predictor produces two accessibility heads. In this manufacturing setting, the two heads have opposite risk semantics:

- the red head treats class `0` as the risk class;
- the green head treats class `1` as the risk class.

The calibrated decision layer therefore handles the two heads separately instead of applying one global threshold. The implemented pipeline contains:

1. Per head temperature scaling to improve probability calibration before thresholding.
2. Class conditional risk control for risk class decisions in each head.
3. A recall clamp on the CRC threshold to avoid overly conservative thresholds that miss risk points.
4. Selective Risk Rescue, which uses a small calibrated budget to recover high confidence risk points after the primary CRC decision.
5. Fixed threshold and calibrated fixed threshold baselines for review and ablation comparisons.

## Environment

The recommended setup is a Python 3.9 conda environment. On Windows with an NVIDIA GPU, create the environment with:

```bash
conda env create -f environment.yml
conda activate dm
```

If you prefer pip, install PyTorch first for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The original experiments were run with PyTorch 2.0 and CUDA 11.8. If your machine uses another CUDA version, install the matching PyTorch build before running the project.

## Local Data And Checkpoints

The following paths are expected locally but are ignored by Git:

```text
pretrained/00840solver/00840.solver.tar
projects/data/filelist/models_train_val.txt
projects/data/filelist/models_test.txt
projects/data/points/
projects/data/raw_data/
projects/logs/
```

The raw data should follow the original DeepMill layout. After placing raw accessibility files under `projects/data/raw_data`, preprocess them from the `projects` directory:

```bash
cd projects
python tools/seg_deepmill_cutter.py
```

This creates the processed point files and file lists used by the training and evaluation scripts.

## Standard Training

Run commands from the `projects` directory.

```bash
python run_seg_deepmill.py \
  --depth 5 \
  --model unet \
  --conditioning concat \
  --alias train_concat_d5 \
  --ratios 1.0 \
  --max-epoch 300 \
  --test-every-epoch 50
```

To fine tune from the original checkpoint:

```bash
python run_seg_deepmill.py \
  --depth 5 \
  --model unet \
  --conditioning concat \
  --alias finetune_concat_d5 \
  --ratios 1.0 \
  --max-epoch 80 \
  --test-every-epoch 10 \
  --ckpt ../pretrained/00840solver/00840.solver.tar \
  --strict-load true \
  --resume-optimizer false \
  --reset-epoch
```

## Calibrated Risk Evaluation

Run the calibrated evaluation from `projects`:

```bash
python run_mc_cp_crc.py \
  --alias paper_seed_123 \
  --depth 5 \
  --model unet \
  --conditioning concat \
  --ckpt ../pretrained/00840solver/00840.solver.tar \
  --strict-load true \
  --seed 123 \
  --split-mode random \
  --split-seed 123 \
  --mc-samples 1 \
  --temperature-scaling \
  --calibration-baselines \
  --cp-method threshold \
  --alpha 0.1 \
  --red-crc-alpha 0.03 \
  --green-crc-alpha 0.03 \
  --crc-max-threshold 0.5 \
  --risk-rescue \
  --risk-rescue-budget 0.002 \
  --risk-rescue-min-prob 0.25 \
  --calibration-ratio 0.2 \
  --red-risk-class 0 \
  --green-risk-class 1 \
  --test-take -1
```

The script writes:

```text
projects/logs/seg_deepmill/<alias>/mc_cp_crc_summary.csv
projects/logs/seg_deepmill/<alias>/mc_cp_crc_shapes.csv
projects/logs/seg_deepmill/<alias>/mc_cp_crc_results.json
```

When `--calibration-baselines` is enabled, the summary additionally reports:

- `ats/*` for entropy based Adaptive Temperature Scaling.

For multi seed paper experiments:

```bash
python run_paper_experiments.py \
  --suite all \
  --alias-prefix paper_v1 \
  --seeds "123,456,789" \
  --calibration-baselines
```

For the review threshold comparison:

```bash
python run_paper_experiments.py \
  --suite calibrated_fixed \
  --alias-prefix review_threshold \
  --seeds "123,456,789"
```

Summarize experiment CSV files with:

```bash
python summarize_paper_experiments.py \
  logs/paper_experiments/multi_seed_summary.csv \
  --out logs/paper_experiments/paper_tables.csv
```

## Dataset Statistics

To compute dataset and calibration split statistics:

```bash
python dataset_statistics.py \
  --split-seed 123 \
  --calibration-ratio 0.2
```

The outputs are saved under `projects/logs/dataset_statistics/`.

## Notes

- Generated logs, datasets, checkpoints, paper drafts, and qualitative visualization scripts are excluded from this repository.
- The tracked code is enough to reproduce the training, calibrated decision evaluation, ablation, and summary table generation once the local data and checkpoint paths are restored.
- If using the original DeepMill model, cite the original DeepMill paper in addition to any work derived from this calibrated risk pipeline.
