# U-Net Retrain Protocol

## Khi nào retrain

Retrain U-Net khi có bằng chứng dữ liệu, không retrain theo cảm tính. Các trigger hợp lý:

- Có đủ annotation vùng bẩn/wet đã duyệt, mặc định tối thiểu 100 mẫu.
- `false_pass_rate` tăng trên benchmark hoặc production review.
- Supervisor thường xuyên sửa verdict vì model bỏ sót vùng bẩn/wet.
- Environment mới có texture/ánh sáng làm dirty coverage sai.
- Candidate dataset chứa pattern mới mà active model chưa học.

## Dữ liệu retrain

Dataset retrain được build từ approved annotations:

- Chỉ lấy annotation đã approved.
- Loại bỏ hoặc đánh dấu riêng rectangle weak labels.
- Tách train/valid/test bằng seed ổn định.
- Không đưa benchmark/golden benchmark vào train.
- Không tái dùng mẫu đã train nếu không có chủ đích rõ ràng.

## Pipeline

Pipeline U-Net retrain:

1. Export approved annotation manifests.
2. Tải ảnh snapshot và annotation payload.
3. Convert rectangle/polygon thành mask class `0/1/2`.
4. Train candidate U-Net từ active U-Net checkpoint.
5. Evaluate candidate trên valid/test và frozen benchmark.
6. Sinh report JSON/Markdown.
7. Chạy promotion gate.
8. Chỉ promote nếu gate pass.

## YOLO trong scope này

YOLO được freeze:

- Không retrain cùng U-Net.
- Không dùng YOLO mAP làm promotion gate chính.
- Chỉ chạy sanity check nếu cần đảm bảo artifact/runtime không hỏng.

Việc freeze YOLO giúp giảm biến số khi bảo vệ: nếu candidate tốt hơn, cải thiện đến từ U-Net retrain và dữ liệu vùng bẩn đã duyệt.

## Artifact bắt buộc

Mỗi retrain run cần lưu:

- dataset snapshot id hoặc manifest path;
- số lượng train/valid/test;
- tỷ lệ rectangle weak vs polygon/mask;
- U-Net config;
- checkpoint candidate;
- benchmark report;
- promotion decision.

