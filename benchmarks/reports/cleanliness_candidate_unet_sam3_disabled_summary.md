# Pilot Benchmark Summary

## Core Metrics

| Metric | Value |
| --- | --- |
| Input rows | 19 |
| Evaluated samples | 18 |
| Skipped samples | 1 |
| Verdict accuracy | 44.44% |
| False pass rate | 38.89% |
| False fail rate | 0.00% |
| Pending review rate | 22.22% |
| Manual correction rate | 55.56% |
| Average latency | 543.76 ms |
| Average quality score | 93.8 |
| Average U-Net dirty coverage | 6.202% |
| Average SAM3 dirty coverage | 0.0% |
| Average combined dirty coverage | 6.202% |
| Average SAM3 elapsed | 0.0 ms |

## Confusion Matrix

| Expected \ Predicted | PASS | PENDING | FAIL |
| --- | --- | --- | --- |
| PASS | 7 | 3 | 0 |
| PENDING | 3 | 1 | 0 |
| FAIL | 4 | 0 | 0 |

## By Environment

| Environment | Samples | Accuracy | False pass | False fail | Pending review | Avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| LOBBY_CORRIDOR | 13 | 53.85% | 23.08% | 0.00% | 30.77% | 541.41 ms |
| OUTDOOR_LANDSCAPE | 2 | 0.00% | 100.00% | 0.00% | 0.00% | 634.61 ms |
| RESTROOM | 3 | 33.33% | 66.67% | 0.00% | 0.00% | 493.38 ms |

## By Dirty Level

| Dirty level | Samples | Accuracy |
| --- | --- | --- |
| clean | 10 | 70.00% |
| obviously_dirty | 4 | 0.00% |
| slightly_dirty | 4 | 25.00% |

## Notes

- `false_pass`: predicted PASS while expected verdict is not PASS.
- `false_fail`: predicted FAIL while expected verdict is PASS.
- `pending_review_rate`: share of samples predicted as PENDING.
- Dirty coverage sources: `{'unet': 17, 'equal': 1}`.
- SAM3 statuses: `{'disabled': 18}`.

## Skipped Rows

| Image ID | Expected | Predicted | Reason |
| --- | --- | --- | --- |
| scoring_18 | PENDING |  | missing_or_invalid_verdict |
