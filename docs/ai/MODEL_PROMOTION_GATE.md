# Model Promotion Gate

## Mục tiêu

Promotion gate quyết định candidate U-Net có được đưa vào production hay không. Gate phải bảo vệ workflow khỏi model mới nhìn có vẻ tốt trên train log nhưng làm tăng rủi ro production.

## Điều kiện bắt buộc

Candidate chỉ được promote nếu:

- Có benchmark report hợp lệ.
- Có baseline report của active model trên cùng benchmark.
- `false_pass_rate` không tăng.
- `verdict_accuracy` không giảm.
- `mIoU` không thấp hơn baseline trên golden mask set.
- `average_latency_ms` không vượt ngưỡng vận hành.
- Artifact candidate, metrics và dataset snapshot được lưu lại.

Nếu thiếu baseline hoặc thiếu report, không auto-promote. Trạng thái nên là `Awaiting benchmark` hoặc `Rejected: missing baseline/report`.

## Metric ưu tiên

Thứ tự ưu tiên khi xét gate:

1. Safety: `false_pass_rate`.
2. Business correctness: `verdict_accuracy`.
3. Review workload: `pending_review_rate`.
4. Segmentation quality: `mIoU`, `IoU_stain`, `IoU_wet`.
5. Runtime: latency và lỗi inference.

Một candidate có `mIoU` cao hơn nhưng làm tăng false pass không nên lên production.

## Decision log

Mỗi decision cần ghi:

- active model id;
- candidate model id;
- benchmark dataset id;
- metric baseline;
- metric candidate;
- pass/fail từng rule;
- người hoặc hệ thống quyết định;
- timestamp;
- lý do cuối cùng.

## Rollback

Active model cũ phải được giữ lại. Nếu model mới gây lỗi runtime hoặc production review xấu đi, rollback về active model trước đó và đánh dấu candidate là failed-in-production.

