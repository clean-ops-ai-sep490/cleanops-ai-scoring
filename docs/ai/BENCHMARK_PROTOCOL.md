# Benchmark Protocol

## Mục tiêu

Benchmark dùng để trả lời câu hỏi: model hiện tại có đủ tốt để hỗ trợ production workflow không. Benchmark phải tách khỏi dữ liệu train và không được lấy từ smoke run hoặc log demo.

## Dataset tối thiểu

Pilot benchmark mục tiêu nên có 50-100 ảnh thật. Repo hiện có pilot set khởi đầu 19 ảnh public cho cleanliness và 26 ảnh public cho PPE. Cần mở rộng thêm bằng ảnh thật từ hệ thống hoặc ảnh public có nguồn rõ ràng.

- Cân bằng `PASS`, `PENDING`, `FAIL`.
- Có nhiều environment, ví dụ lobby/corridor, restroom, outdoor.
- Có nhiều mức bẩn: clean, slightly_dirty, obviously_dirty.
- Mỗi ảnh có ground truth verdict do supervisor hoặc nhóm đánh giá thống nhất.

Golden segmentation benchmark nên có 30-50 ảnh có mask/polygon kỹ hơn:

- Dùng để đo U-Net `IoU_stain`, `IoU_wet`, `mIoU`.
- Không dùng rectangle-only labels để claim pixel-level segmentation chính xác cao.
- Không dùng ảnh golden benchmark trong train/valid retrain.

## Nơi lưu trong repo

Benchmark thật nằm dưới `benchmarks/`, không nằm trong `data/` hoặc `outputs/`:

- `benchmarks/cleanliness/pilot_benchmark.csv`
- `benchmarks/cleanliness/case_studies.csv`
- `benchmarks/ppe/pilot_benchmark.csv`
- `benchmarks/reports/`
- `scripts/run_cleanliness_pilot_benchmark.py`
- `scripts/run_ppe_pilot_benchmark.py`

Chưa có manifest golden mask thật trong repo. Không tạo manifest rỗng; chỉ thêm khi đã có mask/polygon đã duyệt.

Chi tiết xem `docs/ai/BENCHMARK_STORAGE.md`.

## Trường dữ liệu khuyến nghị

Mỗi dòng benchmark CSV nên có:

- `image_id`
- `image_url` hoặc path local
- `environment_key`
- `expected_verdict`
- `dirty_level`
- `predicted_verdict`
- `quality_score`
- `stain_or_water_coverage_pct`
- `wet_surface_coverage_pct`
- `total_dirty_coverage_pct`
- `latency_ms`
- `notes`

Nếu có mask:

- `mask_path`
- `annotation_quality`: `polygon`, `mask`, hoặc `rectangle_weak`

## Metric bắt buộc

Business metrics:

- `verdict_accuracy`: tỷ lệ verdict khớp ground truth.
- `false_pass_rate`: model dự đoán `PASS` nhưng ground truth không phải `PASS`.
- `false_fail_rate`: model dự đoán `FAIL` nhưng ground truth là `PASS`.
- `pending_review_rate`: tỷ lệ model đẩy về `PENDING`.
- `average_latency_ms`: độ trễ trung bình.

U-Net metrics, chỉ tính trên ảnh có mask/polygon đủ tốt:

- `IoU_stain`
- `IoU_wet`
- `mIoU`

## Quy định không dùng

Không dùng các thứ sau làm benchmark thật:

- Smoke metrics sinh ra để kiểm tra pipeline.
- `candidate_metrics.json` từ demo/local smoke nếu không chạy trên benchmark thật.
- Log training không có dataset/version/report đi kèm.
- Rectangle-only labels để tuyên bố segmentation chính xác theo pixel.
