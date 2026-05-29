# Pilot Benchmark Summary

## Core Metrics

| Metric | Value |
| --- | --- |
| Input rows | 19 |
| Evaluated samples | 18 |
| Skipped samples | 1 |
| Verdict accuracy | 44.44% |
| False pass rate | 0.00% |
| False fail rate | 0.00% |
| Pending review rate | 77.78% |
| Calibrated count | 5 |
| Calibrated rate | 27.78% |
| Manual correction rate | 55.56% |
| Average latency | 2323.86 ms |
| Average quality score | 74.72 |
| Average U-Net dirty coverage | 22.721% |
| Average SAM3 dirty coverage | 8.284% |
| Average combined dirty coverage | 25.599% |
| Average SAM3 elapsed | 1732.67 ms |

## Confusion Matrix

| Expected \ Predicted | PASS | PENDING | FAIL |
| --- | --- | --- | --- |
| PASS | 4 | 6 | 0 |
| PENDING | 0 | 4 | 0 |
| FAIL | 0 | 4 | 0 |

## By Environment

| Environment | Samples | Accuracy | False pass | False fail | Pending review | Avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| LOBBY_CORRIDOR | 13 | 53.85% | 0.00% | 0.00% | 69.23% | 2222.39 ms |
| OUTDOOR_LANDSCAPE | 2 | 50.00% | 0.00% | 0.00% | 100.00% | 3402.22 ms |
| RESTROOM | 3 | 0.00% | 0.00% | 0.00% | 100.00% | 2044.71 ms |

## By Dirty Level

| Dirty level | Samples | Accuracy |
| --- | --- | --- |
| clean | 10 | 40.00% |
| obviously_dirty | 4 | 0.00% |
| slightly_dirty | 4 | 100.00% |

## Notes

- `false_pass`: predicted PASS while expected verdict is not PASS.
- `false_fail`: predicted FAIL while expected verdict is PASS.
- `pending_review_rate`: share of samples predicted as PENDING.
- `calibrated_rate`: share of evaluated samples changed or reviewed by safety calibration.
- Dirty coverage sources: `{'equal': 3, 'unet': 12, 'sam3': 3}`.
- SAM3 statuses: `{'ok': 18}`.

## Skipped Rows

| Image ID | Expected | Predicted | Reason |
| --- | --- | --- | --- |
| scoring_18 | PENDING |  | missing_or_invalid_verdict |
