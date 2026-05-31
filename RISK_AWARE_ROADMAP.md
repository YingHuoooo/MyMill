# Risk-Aware DeepMill Roadmap

## Current Evidence

- FiLM is useful for low-data trend checks, but it does not improve the strong
  author checkpoint.
- MC dropout does not add useful uncertainty for the current DeepMill checkpoint.
  With `mc_samples=1`, the metrics are effectively unchanged while evaluation is
  much faster.
- APS produces valid but overly conservative prediction sets on this binary
  task. It is useful as an ablation, but not the main contribution unless later
  regularized.
- Head-specific CRC is the strongest current result: red and green false-negative
  risks are controlled below the target with almost no loss in F1/mIoU.

## Main Paper Direction

Frame the project as **risk-aware accessibility learning for subtractive
manufacturing** rather than as an architecture-only improvement.

Recommended method stack:

1. DeepMill baseline checkpoint.
2. Head-specific CRC for manufacturing-risk control.
3. Adaptive CRC to reduce over-conservatism across easy and hard samples.
4. Risk-aware fine-tuning loss to improve the base model before calibration.
5. Optional APS/RAPS ablations for set-valued uncertainty.

## Method 1: Head-Specific CRC

Keep this as the current main result.

Use separate risk definitions:

- Red head: risk class `0`.
- Green head: risk class `1`.

Use separate risk levels:

- `red_crc_alpha`: default `0.03`.
- `green_crc_alpha`: default `0.05`.

Primary metrics:

- `crc_red/fn_rate`
- `crc_green/fn_rate`
- `crc/f1_avg`
- `crc_red/mIoU`
- `crc_green/mIoU`
- `predicted_risk_rate`

## Method 2: Adaptive CRC

Motivation: one global threshold can be too conservative on easy shapes and not
conservative enough on hard shapes.

Candidate stratification variables:

- Mean confidence: `max(prob).mean()`.
- Mean entropy.
- CRC margin around the risk threshold.
- Shape-level baseline error proxy from calibration.

Implementation idea:

1. Compute a difficulty score for each calibration shape.
2. Split calibration shapes into easy/hard bins, or low/medium/high uncertainty.
3. Calibrate CRC thresholds independently per bin.
4. At test time, assign each shape to a bin and use the bin threshold.

Expected effect:

- Preserve false-negative risk control.
- Reduce unnecessary predicted-risk area.
- Improve CRC F1/mIoU relative to global CRC.

## Method 3: Modern Risk-Aware Fine-Tuning Loss

The older Tversky/Focal-Tversky family is still relevant, but the next round
should prioritize newer reliability- and boundary-oriented losses.

### 3.1 Calibration-Aware Auxiliary Loss

Recent segmentation work proposes adding a differentiable calibration loss such
as marginal L1 Average Calibration Error to improve pixel-wise reliability
without sacrificing segmentation quality. This is well aligned with CRC because
CRC works better when probabilities are meaningful.

Experiment:

```bash
loss = cross_entropy + lambda_calib * calibration_loss
```

Implemented setting:

```bash
--loss-name risk_aware --calib-weight 0.05
```

Expected effect:

- Better calibrated probabilities.
- More stable CRC thresholds.
- Lower false-negative risk at similar F1/mIoU.

### 3.2 Uncertainty-Aware Cross-Entropy

Uncertainty-aware CE dynamically weights CE with predictive uncertainty. This is
newer and closer to our failure mode than plain class weighting: the hard,
ambiguous points receive more training signal.

Experiment:

```bash
loss = uncertainty_weight * cross_entropy
```

Current practical approximation:

```bash
loss = weighted_cross_entropy + lambda_fn * (1 - p_risk)
```

Implemented setting:

```bash
--loss-name risk_aware --risk-class-weight 2.0 --fn-penalty-weight 0.2
```

Expected effect:

- Better hard-point learning.
- Potential improvement in green F1 and risk-sensitive red/green recall.

### 3.3 Boundary-Wise / Narrow-Band Loss

For manufacturing accessibility, many important mistakes occur near transition
regions between accessible and inaccessible zones. Newer boundary-wise losses
focus optimization near target boundaries instead of treating all points equally.

Candidate losses:

- Boundary-wise loss based on fuzzy rough sets.
- Narrow-band boundary loss.
- Boundary difference over union loss.

Experiment:

```bash
loss = cross_entropy + lambda_boundary * boundary_loss
```

Expected effect:

- Better boundary quality.
- Potential improvement in red/green IoU and fewer local false negatives.

## Method 4: APS / RAPS Ablation

Current APS result:

- Coverage is extremely high.
- Average set size is too large.
- Doubleton rate remains high even at `alpha=0.3`.

Keep APS as an ablation. If it remains too conservative at `alpha=0.5`, switch
to RAPS-style regularization.

Metrics:

- `cp_red/coverage`
- `cp_green/coverage`
- `cp_red/avg_set_size`
- `cp_green/avg_set_size`
- `doubleton_rate`
- `risk_coverage`
- `risk_set_rate`

## Recommended Experiment Order

1. Finish APS alpha sweep: `alpha=0.5`.
2. Fix NaN aggregation for `risk_coverage`.
3. Implement adaptive CRC.
4. Implement calibration-aware or uncertainty-aware fine-tuning loss.
5. Combine the best fine-tuned checkpoint with adaptive CRC.

## Suggested Paper Tables

Core table:

| Method | mIoU red | mIoU green | F1 avg | FN risk red | FN risk green |
| --- | --- | --- | --- | --- | --- |
| DeepMill checkpoint | | | | | |
| + FiLM | | | | | |
| + MC dropout | | | | | |
| + APS | | | | | |
| + CRC | | | | | |
| + Adaptive CRC | | | | | |
| + Risk-aware loss | | | | | |
| + Risk-aware loss + Adaptive CRC | | | | | |

Reliability table:

| Method | Coverage red | Coverage green | Avg set size red | Avg set size green | Predicted risk rate |
| --- | --- | --- | --- | --- | --- |
| Threshold CP | | | | | |
| APS | | | | | |
| RAPS | | | | | |
| CRC | | | | | |
