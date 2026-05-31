# DeepMill Improvement Experiments

## Baselines

- `concat`: original DeepMill tool conditioning. Each decoder scale receives a 256-dimensional tool feature that is expanded to octree nodes and concatenated with geometry features.
- `film`: experimental FiLM conditioning. Each decoder scale receives tool-conditioned affine parameters and modulates decoder features with `feature * (1 + gamma) + beta`.
- `film_skip`: modulates the U-Net skip features before concatenating them into the decoder.
- `film_both`: applies both decoder FiLM and skip FiLM.

## Run Examples

Baseline-compatible run:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning concat --alias unet_d5_concat
```

FiLM ablation run:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning film --alias unet_d5_film
```

Skip-FiLM ablation run:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning film_skip --alias unet_d5_film_skip
```

Fast trend check before full training:

```bash
cd projects
python run_seg_deepmill.py --quick --depth 5 --model unet --conditioning concat --alias quick_concat_d5
python run_seg_deepmill.py --quick --depth 5 --model unet --conditioning film --alias quick_film_d5
```

The quick mode uses 5% of the training split, at most 30 epochs, tests every 5
epochs, evaluates 100 test shapes, and keeps visualization disabled. Use it only
to decide whether an idea is promising enough for a full run.

## Planned Next Ablations

- `FiLM`: compare `concat` and `film` with the same split, seed, depth, and training schedule.
- `CRC`: calibrate decision thresholds on a held-out calibration split to control false negative risk for inaccessible and occluded point labels.
- `MC-CP + CRC`: use Monte Carlo dropout to estimate predictive uncertainty, then use conformal calibration to produce risk-controlled prediction sets.

## MC-CP + CRC Evaluation

Use the author checkpoint as the main backbone for MC-CP + CRC. The script
stores the original checkpoint argmax metrics, MC-dropout argmax metrics,
conformal prediction metrics, CRC metrics, and per-shape records under the
experiment log directory.

```bash
cd projects
python run_mc_cp_crc.py --alias mccp_crc_author --ckpt ../pretrained/00840solver/00840.solver.tar --seed 123 --mc-samples 8 --alpha 0.1 --crc-alpha 0.05 --calibration-ratio 0.2
```

Saved files:

- `mc_cp_crc_results.json`: full configuration, thresholds, and metrics.
- `mc_cp_crc_summary.csv`: flat metric table for copying into reports.
- `mc_cp_crc_shapes.csv`: per-shape split and metrics.

To compare against the FiLM branch, keep the same calibration settings and
change only the checkpoint/model options:

```bash
cd projects
python run_mc_cp_crc.py --alias mccp_crc_film_skip --conditioning film_skip --film-scale 0.1 --ckpt ../pretrained/00840solver/00840.solver.tar --strict-load false --seed 123 --mc-samples 8 --alpha 0.1 --crc-alpha 0.05 --calibration-ratio 0.2
```

## FiLM Fine-Tuning

Initialize FiLM from the author checkpoint while skipping the newly introduced
FiLM parameters:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning film --film-scale 0.1 --alias film_ft_from_840 --ratios 1.0 --max-epoch 100 --test-every-epoch 10 --ckpt ../pretrained/00840solver/00840.solver.tar --strict-load false --resume-optimizer false --reset-epoch --lr 0.00001 --trainable-keywords film_conditioners
```

Skip-FiLM fine-tuning keeps the author model fixed and trains only the new
skip conditioners:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning film_skip --film-scale 0.1 --alias film_skip_ft_s01 --ratios 1.0 --max-epoch 150 --test-every-epoch 10 --ckpt ../pretrained/00840solver/00840.solver.tar --strict-load false --resume-optimizer false --reset-epoch --lr 0.00001 --trainable-keywords skip_film_conditioners
```

Combined decoder + skip FiLM:

```bash
cd projects
python run_seg_deepmill.py --depth 5 --model unet --conditioning film_both --film-scale 0.1 --alias film_both_ft_s01 --ratios 1.0 --max-epoch 150 --test-every-epoch 10 --ckpt ../pretrained/00840solver/00840.solver.tar --strict-load false --resume-optimizer false --reset-epoch --lr 0.00001 --trainable-keywords film_conditioners,skip_film_conditioners
```

## Suggested Metrics

- Primary: `f1_red`, `f1_green`, `f1_avg`, `mIoU`.
- Safety-oriented: false negative rate for inaccessible points, false negative rate for occluded points.
- Efficiency-oriented: percentage of points flagged as inaccessible/occluded after CRC calibration.
