# Paper Outline

Working title:

> Risk-Calibrated Neural Accessibility Learning for Subtractive Manufacturing

## Chapter 1: Introduction

### 1.1 Background

Introduce subtractive manufacturing, cutter accessibility analysis, inaccessible
regions, severe occlusion regions, and their importance for design iteration,
cutter selection, setup planning, and tool-path planning.

### 1.2 Motivation

Current neural manufacturability prediction methods can greatly accelerate
accessibility analysis, but they are usually evaluated as ordinary segmentation
models. In manufacturing, however, false negatives on high-risk regions are more
critical than ordinary pixel/point errors because they may cause unsafe or
invalid downstream machining decisions.

### 1.3 Problem Statement

The paper focuses on reducing the probability that true manufacturing-risk
regions are predicted as safe regions, while preserving the strong segmentation
ability and real-time efficiency of existing neural accessibility models.

### 1.4 Contributions

1. **Dual-head manufacturing-risk formulation.** The two prediction heads are
   treated as different manufacturing-risk predictors with head-specific risk
   classes.
2. **Head-specific probability calibration.** Separate temperatures are learned
   for the two heads to improve the reliability of predicted probabilities.
3. **Risk-recall constrained conformal risk control.** Head-specific CRC
   thresholds are constrained to avoid stricter-than-argmax risk decisions.
4. **Selective Risk Rescue.** A budgeted rescue stage recovers near-boundary
   risk points that remain missed after CRC.

### 1.5 Paper Organization

Briefly describe the remaining chapters.

## Chapter 2: Related Work

This chapter should be concise and organized around the problem rather than a
single prior model.

### 2.1 Accessibility Analysis in Subtractive Manufacturing

Discuss cutter accessibility, collision detection, inaccessible regions,
occlusion regions, and traditional geometric methods. Emphasize that geometric
methods are accurate but often computationally expensive for complex or
high-resolution models.

### 2.2 Learning-based Manufacturability Prediction

Discuss learning-based methods for manufacturability analysis and accessibility
prediction, including point cloud, voxel/octree, graph, and B-rep based
representations. Highlight that these methods improve efficiency, but most are
still optimized and evaluated mainly by accuracy, IoU, or F1, rather than by
explicit manufacturing-risk control.

### 2.3 Reliable and Risk-aware Prediction

Discuss probability calibration, uncertainty estimation, conformal prediction,
and conformal risk control. Introduce ECE, NLL, Brier score, false-negative
risk, and selective/risk-aware prediction. Emphasize that these reliability
ideas have been studied in safety-critical domains, but are still underexplored
for learning-based manufacturing accessibility prediction.

## Chapter 3: Overall Framework

### 3.1 Task Definition

Define the input shape representation, cutter parameters, point-wise binary
labels, two output heads, and head-specific risk classes.

### 3.2 Baseline Neural Accessibility Model

Introduce the baseline octree-based encoder-decoder framework, cutter parameter
embedding, and dual-head segmentation structure at a high level. This section can
refer to the existing neural accessibility model without making the whole paper
look like a narrow extension of one specific work.

### 3.3 Proposed Risk-Calibrated Framework

Present the full pipeline:

```text
Input shape and cutter parameters
  -> neural accessibility model
  -> red/green logits
  -> head-specific temperature scaling
  -> risk-recall constrained CRC
  -> selective risk rescue
  -> final risk-aware predictions
```

### 3.4 Risk-sensitive Prediction Objective

Explain that the framework aims to reduce missed manufacturing-risk regions
rather than only improve average segmentation accuracy.

### 3.5 Framework Advantages

Summarize practical advantages:

- no need to retrain the heavy backbone,
- can be applied to an existing neural accessibility model,
- provides interpretable risk-control parameters,
- improves safety-oriented behavior with limited extra predicted-risk area.

## Chapter 4: Methodology

### 4.1 Dual-head Manufacturing-risk Formulation

Define two head-specific risk tasks:

```text
red head risk class   = 0
green head risk class = 1
```

Explain why different heads need different risk definitions.

### 4.2 Head-specific Temperature Scaling

Introduce post-hoc temperature scaling:

```text
p = softmax(z / T)
```

Learn separate `T_red` and `T_green` on the calibration split. Explain that this
module improves probability reliability before downstream risk control.

### 4.3 Risk-recall Constrained CRC

Introduce conformal risk control for learning risk thresholds on a calibration
split. Then define the risk-recall constraint:

```text
threshold = min(threshold_crc, 0.5)
```

Explain that this prevents CRC from being more conservative than the original
binary argmax boundary and protects risk recall.

### 4.4 Selective Risk Rescue

Introduce the second-stage rescue rule:

```text
if p_risk >= rescue_threshold:
    predict risk
```

The rescue threshold is selected on the calibration split under a maximum
extra-risk budget. This module selectively recovers near-boundary risk points
missed by CRC.

### 4.5 Risk-aware Metrics

Define the metrics used in the paper:

- false-negative rate,
- predicted-risk rate,
- ECE,
- NLL,
- Brier score,
- mIoU,
- F1.

Clarify the practical meaning of each metric in manufacturing accessibility
prediction.

## Chapter 5: Experiments

### 5.1 Dataset and Implementation Details

Describe dataset splits, model checkpoint, octree depth, cutter parameters,
calibration ratio, random seed, GPU environment, and main hyperparameters.

### 5.2 Main Comparison

Compare:

- original neural accessibility baseline,
- temperature scaling,
- risk-recall constrained CRC,
- selective risk rescue.

Report the key result:

```text
Average risk-class false-negative rate:
0.04571 -> 0.03486
Relative reduction: about 23.7%
```

### 5.3 Head-wise Risk Analysis

Report red and green heads separately:

```text
Red FN rate:
0.04889 -> 0.03754

Green FN rate:
0.04254 -> 0.03219
```

Explain that both risk heads benefit from the final method.

### 5.4 Probability Calibration Analysis

Report ECE, NLL, and Brier score before and after temperature scaling:

```text
ECE:
0.00665 -> 0.00450
```

Explain that calibration improves probability reliability, while the main risk
reduction comes from CRC and Selective Risk Rescue.

### 5.5 Ablation Study

Build an incremental table:

```text
Baseline
+ Temperature Scaling
+ Risk-recall CRC
+ Selective Risk Rescue
```

Show what each component contributes.

### 5.6 Parameter Sensitivity

Analyze:

- `red_crc_alpha`,
- `green_crc_alpha`,
- `crc_max_threshold`,
- `risk_rescue_budget`,
- `risk_rescue_min_prob`.

Focus on the trade-off between false-negative rate and predicted-risk rate.

### 5.7 Visual Results

Show representative cases:

- baseline missed risk regions,
- CRC-corrected regions,
- selectively rescued risk regions,
- extra predicted-risk regions.

Use visualization to explain why the method is useful for safety-oriented
manufacturing decisions.

### 5.8 Discussion and Limitations

Discuss:

- why the method reduces risky misses rather than mainly improving mIoU,
- why slightly more conservative prediction is acceptable in manufacturing,
- dependence on calibration data,
- limitations of post-hoc risk control,
- potential extension to conditional CRC or training-time risk optimization.

## Chapter 6: Conclusion

### 6.1 Summary

Summarize the proposed risk-calibrated neural accessibility framework.

### 6.2 Main Findings

State that the method reduces high-risk false negatives, preserves the baseline
segmentation ability, and improves probability reliability.

### 6.3 Limitations

Mention calibration split dependency, limited validation scope, and the fact
that the method is currently post-hoc rather than end-to-end trained.

### 6.4 Future Work

Potential directions:

- conditional conformal risk adaptation,
- end-to-end risk-aware training,
- cross-dataset validation,
- integration with real tool-path planning systems.

