# Benchmark Storage

## Hiện trạng repo

Repo hiện đã bỏ các file template benchmark. Những gì còn lại trong `benchmarks/` phải là dữ liệu thật hoặc report thật.

Hiện có:

- `benchmarks/cleanliness/pilot_benchmark.csv`: 19 ảnh public có ground truth verdict.
- `benchmarks/cleanliness/case_studies.csv`: 8 case thật được chọn từ pilot set.
- `benchmarks/ppe/pilot_benchmark.csv`: 26 ảnh public có ground truth PPE status.
- `scripts/run_cleanliness_pilot_benchmark.py`: chạy API thật và sinh CSV cleanliness đã có dự đoán.
- `scripts/run_ppe_pilot_benchmark.py`: chạy API thật và sinh CSV PPE đã có dự đoán.
- `scripts/summarize_pilot_benchmark.py`: tổng hợp metric sau khi đã có kết quả inference thật.
- `scripts/summarize_ppe_benchmark.py`: tổng hợp PPE metric sau khi đã có kết quả inference thật.

## Vị trí benchmark thật

Benchmark thật của cleanliness scoring được đặt ở:

- `benchmarks/cleanliness/pilot_benchmark.csv`
- `benchmarks/reports/`

`pilot_benchmark.csv` là tập retest chính cho business verdict. Các trường `predicted_verdict`, `quality_score`, `latency_ms`, và `visualization_url` để trống cho đến khi chạy model thật.

Golden mask benchmark chưa được commit vì hiện chưa có bộ mask/polygon thật đã duyệt. Khi có mask thật, tạo thêm manifest mới dưới `benchmarks/cleanliness/` và không dùng rectangle-only labels để claim pixel-level mIoU.

## Vị trí không nên dùng

Không đặt benchmark thật ở:

- `data/`: thư mục train/retrain local và đang bị gitignore.
- `outputs/`: output tạm thời, cũng đang bị gitignore.
- `data/retrain_bridge`: dataset sinh từ approved annotations để train candidate, không phải frozen benchmark.

## Cách chạy benchmark thật

Chạy scoring API trước, sau đó sinh CSV evaluated từ ground truth:

```powershell
python scripts/run_cleanliness_pilot_benchmark.py `
  --input-csv benchmarks/cleanliness/pilot_benchmark.csv `
  --output-csv benchmarks/reports/cleanliness_pilot_evaluated.csv `
  --api-base-url http://127.0.0.1:8000
```

Với PPE:

```powershell
python scripts/run_ppe_pilot_benchmark.py `
  --input-csv benchmarks/ppe/pilot_benchmark.csv `
  --output-csv benchmarks/reports/ppe_pilot_evaluated.csv `
  --api-base-url http://127.0.0.1:8000
```

## Cách chạy summary hiện tại

Sau khi đã có CSV evaluated, có thể sinh report:

```powershell
python scripts/summarize_pilot_benchmark.py `
  --input-csv benchmarks/reports/cleanliness_pilot_evaluated.csv `
  --output-json benchmarks/reports/cleanliness_pilot_summary.json `
  --output-md benchmarks/reports/cleanliness_pilot_summary.md
```

Report trong `benchmarks/reports/` nên được commit nếu dùng làm bằng chứng trong báo cáo.

## Dữ liệu ảnh

Pilot set hiện dùng URL ảnh public trong `image_url`. Nếu sau này dùng ảnh nội bộ, không commit ảnh riêng tư; lưu blob URL/object key trong `image_url` hoặc `local_image_path` và ghi rõ nguồn trong `notes`.
