# Paper First Candidate: Risk-Calibrated DeepMill

## Positioning

The current first-choice paper direction is **risk-calibrated milling
accessibility segmentation**. The goal is not only to improve conventional
segmentation metrics, but to reduce high-risk false negatives in the two
DeepMill prediction heads while preserving the strong author checkpoint.

Recommended title direction:

> Risk-Calibrated DeepMill: Reliability-Aware Accessibility Segmentation for
> Subtractive Manufacturing

Detailed chapter outline:

- [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md)

## Current Best Technical Route

```text
Author DeepMill checkpoint
  -> dual-head manufacturing-risk formulation
  -> head-specific temperature scaling
  -> risk-recall constrained head-specific CRC
  -> optional selective risk rescue
  -> risk-aware evaluation protocol
```

### Module 1: Dual-Head Manufacturing Risk Formulation

The original DeepMill model already has two prediction heads. We reinterpret
them as two different manufacturing-risk predictors:

- Red head: class `0` is the risk class.
- Green head: class `1` is the risk class.

This makes the evaluation and calibration head-specific instead of using one
shared decision rule for two different failure modes.

### Module 2: Head-Specific Temperature Scaling

One temperature is learned for each prediction head on the calibration split:

- `temperature/red = 0.85`
- `temperature/green = 0.70`

The calibrated probabilities are used by the downstream CRC module. This mainly
improves reliability metrics, not argmax segmentation.

### Module 3: Risk-Recall Constrained CRC

CRC is applied separately to red and green risk classes. The current best setting
uses:

- `red_crc_alpha = 0.03`
- `green_crc_alpha = 0.05`
- `crc_max_threshold = 0.5`

The max-threshold constraint prevents CRC from becoming more conservative than
the original binary argmax rule. This keeps risk recall from degrading.

### Optional Module: Selective Risk Rescue

Selective Risk Rescue is the next experimental extension. It keeps the current
CRC decision and adds a budgeted rescue rule for near-boundary non-risk points:

```text
if p(risk) >= rescue_threshold:
  force risk prediction
```

The rescue threshold is selected on the calibration split under a maximum extra
predicted-risk budget. It is reported separately as `risk_rescue_*` so the method
can be compared against the current best `crc_*` result.

Recommended first setting:

```bash
--risk-rescue --risk-rescue-budget 0.002 --risk-rescue-min-prob 0.25
```

Success criterion:

```text
risk_rescue/fn_rate < 0.04134
risk_rescue/predicted_risk_rate <= about 0.220
red/green mIoU should not drop meaningfully
```

### Module 4: Risk-Aware Evaluation Protocol

The paper should report both conventional segmentation metrics and manufacturing
risk metrics:

- Segmentation: `accu`, `mIoU`, `f1`
- Calibration: `ECE`, `NLL`, `Brier`
- Risk: `false_negative`, `fn_rate`, `predicted_risk_rate`

This is important because the strongest contribution is not a large mIoU gain;
it is reduced risk-class false negatives with almost no additional predicted
risk area.

## Current Best Command

Run from `projects`:

```bash
python run_mc_cp_crc.py --alias risk_recall_crc_author --ckpt ../pretrained/00840solver/00840.solver.tar --seed 123 --split-mode random --split-seed 123 --mc-samples 1 --temperature-scaling --temperature-min 0.5 --temperature-max 5.0 --temperature-steps 91 --cp-method threshold --alpha 0.1 --crc-alpha 0.05 --red-crc-alpha 0.03 --green-crc-alpha 0.05 --crc-max-threshold 0.5 --calibration-ratio 0.2 --red-risk-class 0 --green-risk-class 1
```

## Current Best Results

### False-Negative Risk

| Method | Avg FN rate | Red FN rate | Green FN rate |
| --- | ---: | ---: | ---: |
| Author checkpoint | 0.04571 | 0.04889 | 0.04254 |
| Temperature scaling | 0.04483 | 0.04791 | 0.04175 |
| Risk-recall CRC | 0.04134 | 0.04094 | 0.04175 |

Main risk result:

```text
Average risk-class false-negative rate:
0.04571 -> 0.04134
Relative reduction: about 9.6%
```

Head-specific risk result:

```text
Red FN rate:
0.04889 -> 0.04094
Relative reduction: about 16.3%

Green FN rate:
0.04254 -> 0.04175
Relative reduction: about 1.9%
```

### Segmentation Quality

| Method | Red mIoU | Green mIoU | CRC/Temp F1 avg |
| --- | ---: | ---: | ---: |
| Author checkpoint | 0.93921 | 0.95467 | 0.93557 baseline F1 avg |
| Temperature scaling | 0.94056 | 0.95516 | 0.93251 |
| Risk-recall CRC | 0.94076 | 0.95516 | 0.93224 |

CRC does not sacrifice mIoU:

```text
Red mIoU:   0.93921 -> 0.94076
Green mIoU: 0.95467 -> 0.95516
```

### Predicted Risk Area

| Method | Avg predicted risk rate | Red predicted risk rate | Green predicted risk rate |
| --- | ---: | ---: | ---: |
| Author checkpoint | 0.21703 | 0.34792 | 0.08615 |
| Risk-recall CRC | 0.21800 | 0.35027 | 0.08573 |

The risk area increases by only about `0.00097` absolute, i.e. roughly `0.1`
percentage points, while average FN rate drops by about `9.6%` relatively.

### Probability Calibration

| Method | ECE | NLL | Brier |
| --- | ---: | ---: | ---: |
| Author checkpoint | 0.00665 | 0.03908 | 0.02278 |
| Temperature scaling | 0.00450 | 0.03712 | 0.02244 |

Temperature scaling improves ECE by about `32.4%` relatively. Since the original
ECE is already low, this should be presented as an auxiliary reliability result,
not as the main contribution.

## Four Candidate Contributions

1. **Dual-head risk formulation for milling accessibility.**
   The two DeepMill outputs are treated as different risk heads with different
   risk classes, enabling head-specific risk control.

2. **Head-specific probability calibration.**
   Red and green heads receive separate temperature scaling parameters, reducing
   calibration error before risk control.

3. **Risk-recall constrained CRC.**
   CRC thresholds are learned per head, while `crc_max_threshold=0.5` prevents
   calibrated decisions from becoming more conservative than the original argmax
   boundary. This reduces risk-class false negatives in both heads.

4. **Manufacturing risk-aware evaluation protocol.**
   The method is evaluated with segmentation, calibration, and risk metrics,
   especially `fn_rate` and `predicted_risk_rate`, which are more aligned with
   machining safety than mIoU alone.

## How to Write the Current Result Honestly

Recommended wording:

> Compared with the author checkpoint, the proposed risk-recall constrained CRC
> reduces the average risk-class false-negative rate from 4.57% to 4.13%, a
> relative reduction of 9.6%, while preserving red and green mIoU and increasing
> the predicted-risk area by only about 0.1 percentage points. Head-specific
> temperature scaling further improves probability reliability, reducing ECE
> from 0.00665 to 0.00450.

Avoid saying:

> The method substantially improves segmentation accuracy.

The current evidence supports risk reduction and reliability improvement more
strongly than large segmentation-accuracy gains.

## Failed or Secondary Ablations

- FiLM: useful in low-data trend checks, but not stronger than the author
  checkpoint.
- MC-CP: MC dropout brought little additional information in the current model.
- APS: valid but too conservative for the current binary heads.
- Adaptive CRC: current entropy/margin binning was worse than global
  risk-recall CRC.
- Risk-aware fine-tuning loss: close to baseline with light settings, but not
  better than the checkpoint plus post-hoc calibration.

## 2025 Method Candidates for the Next Improvement

### Candidate A: Calibrated Conformal Risk Adaptation / CCRA

Source:

- Conditional Conformal Risk Adaptation, arXiv:2504.07611
- Link: https://arxiv.org/abs/2504.07611

Why it fits:

- It targets segmentation.
- It focuses on false-negative-rate control.
- It explicitly addresses the weakness of marginal CRC: some samples can have
  high risk even when average risk is controlled.
- It combines probability calibration with adaptive risk control, which matches
  our current temperature scaling + CRC direction.

Practical DeepMill version:

```text
For each shape:
  compute difficulty features:
    mean entropy
    mean max confidence
    risk probability mass
    point count
    tool-parameter statistics if available
  learn or bin a difficulty-conditioned CRC threshold
  apply risk-recall clamp
```

Expected effect:

- Lower worst-case or hard-shape FN rate.
- Better green-head risk control.
- More content than plain CRC because the method becomes conditional and
  sample-adaptive.

Risk:

- Needs careful split design to avoid overfitting calibration.
- May improve conditional risk more than average mIoU.

### Candidate B: Automatically Adaptive CRC / AA-CRC

Source:

- Automatically Adaptive Conformal Risk Control, AISTATS 2025
- Link: https://vincentblot28.github.io/assets/pdf/blot2025_automatically.pdf

Why it fits:

- It is a 2025 CRC extension.
- It adapts risk control to test-sample difficulty.
- It is theoretically stronger and more modern than hand-picked bins.

Practical DeepMill version:

```text
Train a small calibration-side error/risk predictor from frozen model features.
Use the predicted difficulty to modulate the CRC threshold per shape.
Keep crc_max_threshold=0.5 to protect risk recall.
```

Expected effect:

- Better precision-risk trade-off than a single global threshold.
- Potentially less unnecessary risk area while keeping FN rate low.

Risk:

- More complex to implement and explain.
- Needs enough calibration samples for stable difficulty prediction.

### Candidate C: Signed Distance Calibration Loss / pECE

Source:

- We Care Each Pixel: Calibrating on Medical Segmentation Model, arXiv:2503.05107
- Link: https://arxiv.org/abs/2503.05107

Why it fits:

- It is a 2025 segmentation calibration method.
- It connects calibration with boundary geometry.
- Manufacturing accessibility mistakes are often near boundaries.

Practical DeepMill version:

```text
Fine-tune header or decoder with:
  CE loss
  + lightweight point-wise calibration loss
  + boundary/risk-band weighting
```

Expected effect:

- Better calibration near accessibility boundaries.
- Possible reduction of boundary false negatives.

Risk:

- Requires training, and previous fine-tuning attempts did not clearly beat the
  author checkpoint.
- Harder to implement correctly on sparse octree/point predictions than CCRA.

## Recommended Next Step

The most suitable next method is **CCRA-lite**, not another architecture change.

Proposed next experiment:

```text
Temperature scaling
+ risk-recall CRC
+ shape-conditioned CRC thresholding using confidence/entropy/risk-mass features
```

Success criteria:

- Average FN rate below `0.04134`.
- Red FN rate stays near or below `0.04094`.
- Green FN rate below `0.04175`.
- Predicted risk rate does not rise by more than about `0.3` percentage points.
- ECE remains lower than the author checkpoint.
