# AI Scoring Overview

## Mục tiêu

CleanOps AI scoring đánh giá ảnh sau vệ sinh để hỗ trợ supervisor quyết định khu vực đã đạt hay cần xử lý lại. Hệ thống không được trình bày như một mô hình AI nghiên cứu SOTA; nên trình bày như một dịch vụ AI có benchmark, human review, retrain và promotion gate.

## Kiến trúc hiện tại

Luồng cleanliness scoring gồm:

- Auxiliary foundation segmentation: dùng Roboflow/SAM3-style prompt hoặc class như `Garbage`, `Trash`, `Debris`, `Stain`, `Wet_Floor` để tạo evidence nền cho vùng nghi ngờ bẩn. Response contract hiện vẫn giữ key `sam3` để tương thích backend.
- YOLO detector: phát hiện vật thể hoặc vùng bất thường ở mức object/region. Trong scope rebuild này, YOLO được giữ cố định và không retrain.
- U-Net segmentation: ước lượng vùng `stain_or_water` và `wet_surface`, từ đó tính dirty coverage.
- Scoring rules: dùng `max(auxiliary coverage, U-Net coverage)`, object evidence và threshold theo environment để tạo `quality_score` và verdict `PASS/PENDING/FAIL`.
- Human review: supervisor duyệt verdict, tạo ground truth nghiệp vụ.
- Annotation loop: supervisor khoanh vùng bẩn/wet để tạo dữ liệu retrain U-Net.

## Phân biệt phạm vi

Cleanliness scoring và PPE compliance là hai bài toán khác nhau:

- Cleanliness: ảnh sau vệ sinh, output là dirty coverage, quality score, verdict.
- PPE: ảnh người lao động, output là detected/missing PPE items và compliance status.

Không trộn metric PPE vào benchmark cleanliness. Nếu báo cáo có PPE, đặt thành capability riêng.

## Vai trò retrain

Retrain không phải là nút "train lại cho mới". Retrain chỉ hợp lý khi dữ liệu thực tế hoặc supervisor review cho thấy U-Net đang sai ở các mẫu mới, ví dụ vùng ướt bị bỏ sót, dirty coverage bị đánh thấp, hoặc false pass tăng.

Trong rebuild này, retrain tập trung vào U-Net vì U-Net quyết định vùng bẩn/wet chuyên biệt và ảnh hưởng trực tiếp đến dirty coverage. YOLO được freeze để giảm biến số và làm promotion gate dễ giải thích hơn. Auxiliary segmentation là lớp phụ trợ để giảm rủi ro bỏ sót, không thay thế promotion gate của U-Net.
