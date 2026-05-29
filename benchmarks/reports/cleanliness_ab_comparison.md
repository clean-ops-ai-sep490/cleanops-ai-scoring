# Cleanliness A/B Benchmark Comparison

Baseline variant: `current_active_sam3_disabled`

## Variant Metrics

| Variant | Evaluated | Skipped | Accuracy | False pass | False fail | Pending | Avg latency | Coverage source counts | SAM3 status counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_active_sam3_disabled | 18 | 1 | 44.44% | 11.11% | 11.11% | 55.56% | 708.58 ms | {'equal': 3, 'unet': 15} | {'disabled': 18} |
| candidate_unet_sam3_disabled | 18 | 1 | 44.44% | 38.89% | 0.00% | 22.22% | 543.76 ms | {'unet': 17, 'equal': 1} | {'disabled': 18} |

## Verdict Stability Versus Baseline

| Variant | Common | Verdict flips | Improved | Regressed | Same correctness |
| --- | --- | --- | --- | --- | --- |
| candidate_unet_sam3_disabled | 18 | 11 | 4 | 4 | 10 |

## Verdict Flips: candidate_unet_sam3_disabled

| Image | Expected | Baseline | Candidate | Coverage source | U-Net % | SAM3 % | Combined % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scoring_02 | PASS | FAIL | PENDING | unet | 17.257 | 0.0 | 17.257 |
| scoring_03 | PASS | PENDING | PASS | unet | 1.48 | 0.0 | 1.48 |
| scoring_07 | PASS | PENDING | PASS | unet | 0.199 | 0.0 | 0.199 |
| scoring_09 | PENDING | PENDING | PASS | unet | 6.196 | 0.0 | 6.196 |
| scoring_10 | FAIL | PENDING | PASS | unet | 3.2 | 0.0 | 3.2 |
| scoring_11 | PENDING | PENDING | PASS | unet | 0.432 | 0.0 | 0.432 |
| scoring_12 | FAIL | PENDING | PASS | unet | 2.578 | 0.0 | 2.578 |
| scoring_14 | PASS | PENDING | PASS | equal | 0.0 | 0.0 | 0.0 |
| scoring_15 | PENDING | PENDING | PASS | unet | 8.395 | 0.0 | 8.395 |
| scoring_16 | PASS | PASS | PENDING | unet | 14.754 | 0.0 | 14.754 |
| scoring_17 | PASS | FAIL | PASS | unet | 0.273 | 0.0 | 0.273 |
