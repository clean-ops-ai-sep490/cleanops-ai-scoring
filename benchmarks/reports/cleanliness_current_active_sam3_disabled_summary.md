# Pilot Benchmark Summary

## Core Metrics

| Metric | Value |
| --- | --- |
| Input rows | 19 |
| Evaluated samples | 18 |
| Skipped samples | 1 |
| Verdict accuracy | 44.44% |
| False pass rate | 11.11% |
| False fail rate | 11.11% |
| Pending review rate | 55.56% |
| Manual correction rate | 55.56% |
| Average latency | 708.58 ms |
| Average quality score | 77.28 |
| Average U-Net dirty coverage | 22.721% |
| Average SAM3 dirty coverage | 0.0% |
| Average combined dirty coverage | 22.721% |
| Average SAM3 elapsed | 0.0 ms |

## Confusion Matrix

| Expected \ Predicted | PASS | PENDING | FAIL |
| --- | --- | --- | --- |
| PASS | 4 | 4 | 2 |
| PENDING | 0 | 4 | 0 |
| FAIL | 2 | 2 | 0 |

## By Environment

| Environment | Samples | Accuracy | False pass | False fail | Pending review | Avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| LOBBY_CORRIDOR | 13 | 53.85% | 7.69% | 15.38% | 46.15% | 761.14 ms |
| OUTDOOR_LANDSCAPE | 2 | 50.00% | 0.00% | 0.00% | 100.00% | 567.51 ms |
| RESTROOM | 3 | 0.00% | 33.33% | 0.00% | 66.67% | 574.89 ms |

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
- Dirty coverage sources: `{'equal': 3, 'unet': 15}`.
- SAM3 statuses: `{'disabled': 18}`.

## Skipped Rows

| Image ID | Expected | Predicted | Reason |
| --- | --- | --- | --- |
| scoring_18 | PENDING |  | missing_or_invalid_verdict |
