# Pilot Benchmark Summary

## Core Metrics

| Metric | Value |
| --- | --- |
| Input rows | 19 |
| Evaluated samples | 18 |
| Skipped samples | 1 |
| Verdict accuracy | 22.22% |
| False pass rate | 0.00% |
| False fail rate | 0.00% |
| Pending review rate | 100.00% |
| Calibrated count | 13 |
| Calibrated rate | 72.22% |
| Manual correction rate | 77.78% |
| Average latency | 2726.9 ms |
| Average quality score | 53.46 |
| Average U-Net dirty coverage | 22.712% |
| Average SAM3 dirty coverage | 47.118% |
| Average combined dirty coverage | 55.715% |
| Average SAM3 elapsed | 2031.83 ms |

## Confusion Matrix

| Expected \ Predicted | PASS | PENDING | FAIL |
| --- | --- | --- | --- |
| PASS | 0 | 10 | 0 |
| PENDING | 0 | 4 | 0 |
| FAIL | 0 | 4 | 0 |

## By Environment

| Environment | Samples | Accuracy | False pass | False fail | Pending review | Avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| LOBBY_CORRIDOR | 13 | 23.08% | 0.00% | 0.00% | 100.00% | 2719.38 ms |
| OUTDOOR_LANDSCAPE | 2 | 50.00% | 0.00% | 0.00% | 100.00% | 3113.68 ms |
| RESTROOM | 3 | 0.00% | 0.00% | 0.00% | 100.00% | 2501.62 ms |

## By Dirty Level

| Dirty level | Samples | Accuracy |
| --- | --- | --- |
| clean | 10 | 0.00% |
| obviously_dirty | 4 | 0.00% |
| slightly_dirty | 4 | 100.00% |

## Notes

- `false_pass`: predicted PASS while expected verdict is not PASS.
- `false_fail`: predicted FAIL while expected verdict is PASS.
- `pending_review_rate`: share of samples predicted as PENDING.
- `calibrated_rate`: share of evaluated samples changed or reviewed by safety calibration.
- Dirty coverage sources: `{'sam3': 3, 'merged': 14, 'unet': 1}`.
- SAM3 statuses: `{'ok': 18}`.

## Skipped Rows

| Image ID | Expected | Predicted | Reason |
| --- | --- | --- | --- |
| scoring_18 | PENDING |  | missing_or_invalid_verdict |
