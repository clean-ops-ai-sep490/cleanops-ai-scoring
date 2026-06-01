# Báo Cáo AI Scoring CleanOps

## 1. Giới thiệu bài toán

CleanOps cần đánh giá chất lượng vệ sinh từ ảnh sau khi nhân viên hoàn thành công việc. Với mỗi ảnh, hệ thống cần trả lời ba câu hỏi thực tế:

- Khu vực đã đủ sạch để nghiệm thu chưa?
- Nếu chưa chắc chắn, có cần supervisor xem lại không?
- Nếu chưa đạt, bằng chứng trực quan nằm ở đâu trên ảnh?

Đầu ra chính của hệ thống không chỉ là một nhãn AI, mà là một kết quả có thể vận hành trong quy trình giám sát: `quality_score`, verdict `PASS/PENDING/FAIL`, vùng nghi ngờ trên ảnh, và link visualization để supervisor kiểm tra.

## 2. Mục tiêu thiết kế hệ thống

Hệ thống được thiết kế theo hướng software engineering thay vì chỉ tối ưu một model AI đơn lẻ. Các mục tiêu chính gồm:

- Giải thích được: kết quả phải chỉ ra vùng bẩn, vật thể gây penalty và lý do verdict.
- Vận hành được: service có health check, model storage, cache, batch inference và visualization blob URL.
- Mở rộng được: có thể thêm model phụ trợ như Roboflow/SAM3-style segmentation mà không phá contract backend.
- Cải thiện được: supervisor review tạo dữ liệu cho retrain và promotion gate trước khi model mới lên production.
- Trung thực khi báo cáo: chỉ dùng số benchmark thật, không claim metric khi chưa chạy được thực nghiệm.

## 3. System Architecture

Kiến trúc hệ thống gồm bốn lớp chính:

| Lớp | Vai trò |
|---|---|
| Backend CleanOps | Quản lý job scoring, nhận ảnh từ workflow, lưu kết quả, expose dữ liệu cho frontend/mobile |
| AI Scoring API | FastAPI service chạy inference, scoring rule, visualization và PPE endpoint |
| Model & Blob Storage | Lưu active model, candidate model, visualization overlay và cache model runtime |
| Human Review & Retrain | Supervisor review verdict, tạo annotation/candidate data, kích hoạt retrain khi đủ dữ liệu |

Luồng xử lý tổng quát:

```text
Ảnh sau vệ sinh
  -> Backend tạo scoring job
  -> AI Scoring API tải ảnh
  -> YOLO + U-Net + auxiliary segmentation
  -> Tổng hợp dirty coverage và object penalty
  -> Tính quality_score và PASS/PENDING/FAIL
  -> Sinh overlay visualization
  -> Backend lưu kết quả và supervisor review nếu cần
```

AI service hiện có các endpoint vận hành chính: `/health/ready`, `/health/sam3`, `/evaluate-batch`, `/evaluate-url-visualize-link`, `/check` và `/ppe/evaluate`. Các debug route cũ như `/predict`, `/predict-url`, `/predict-unet` không còn là public contract ưu tiên.

## 4. AI Scoring Architecture

Cleanliness scoring được tách thành nhiều nguồn evidence thay vì ép một model làm mọi việc:

| Thành phần | Vai trò | Lý do tồn tại |
|---|---|---|
| YOLO | Phát hiện vật thể/rác ở mức object | Bắt các vật thể rõ ràng như rác, chai, túi, giấy, debris |
| U-Net fine-tuned | Phân đoạn vùng `dirty_area` và `wet_surface` chuyên biệt | Học pattern bẩn/ướt trong ngữ cảnh vệ sinh sàn thực tế |
| Roboflow/SAM3-style auxiliary segmentation | Model phụ trợ tổng quát theo prompt/class như `Garbage`, `Trash`, `Debris`, `Stain`, `Wet_Floor` | Bổ sung khả năng bắt vùng bẩn tổng quát mà U-Net có thể bỏ sót |
| Scoring rules | Tổng hợp evidence thành điểm và verdict | Dễ giải thích, dễ kiểm thử, dễ điều chỉnh theo môi trường |

Trong API contract hiện tại, block phụ trợ vẫn giữ tên `sam3` để tương thích với backend và benchmark scripts. Về mặt kiến trúc báo cáo, có thể giải thích `sam3` là lớp auxiliary foundation segmentation; provider cụ thể có thể là SAM3 local hoặc Roboflow Workflow. Với máy local hiện tại, Roboflow là hướng nhẹ hơn vì không cần GPU CUDA 12.8.

## 5. Công thức scoring và verdict

U-Net sinh `unet_dirty_coverage_pct`. Auxiliary segmentation sinh `sam3_dirty_coverage_pct`. Hệ thống lấy coverage cuối cùng theo công thức:

```text
combined_dirty_coverage_pct = max(unet_dirty_coverage_pct, sam3_dirty_coverage_pct)
```

Lý do dùng `max`: nếu model phụ trợ bắt được vùng bẩn mà U-Net bỏ sót, hệ thống giảm rủi ro false-pass. Ngược lại, nếu model phụ trợ chưa bật hoặc lỗi runtime, coverage của nó bằng 0 và hệ thống vẫn chạy được bằng U-Net/YOLO.

Điểm chất lượng được tính theo quy tắc:

```text
base_clean_score = 100 - combined_dirty_coverage_pct
object_penalty = min(40, penalty_detections_count * 10)
quality_score = clamp(base_clean_score - object_penalty, 0, 100)
```

Verdict phụ thuộc vào threshold theo môi trường:

| Verdict | Điều kiện |
|---|---|
| PASS | `quality_score >= pass_threshold` của environment |
| PENDING | `quality_score < pass_threshold` và `quality_score >= 50` |
| FAIL | `quality_score < 50` |

Thiết kế này giúp hệ thống không chỉ trả lời “sạch hay bẩn”, mà còn thể hiện mức độ tin cậy và đưa những case không chắc chắn sang supervisor review.

## 6. Benchmark và kết quả thực nghiệm

### 6.1 Benchmark pixel-level cho U-Net

Benchmark pixel-level được chạy trên tập ảnh thực tế gồm 8 ảnh test. Mục tiêu là đo khả năng phân đoạn các lớp `background`, `dirty_area` và `wet_surface`.

| Metric | Baseline | Fine-tuned | Thay đổi |
|---|---:|---:|---:|
| Pixel accuracy | 0.920 | 0.911 | -0.009 |
| Mean IoU | 0.307 | 0.329 | +0.022 |
| Mean Dice/F1 | 0.319 | 0.365 | +0.046 |

Theo pixel accuracy, baseline cao hơn nhẹ vì `background` chiếm phần lớn ảnh. Tuy nhiên, Mean IoU và Dice/F1 quan trọng hơn cho bài toán này vì chúng phản ánh khả năng nhận diện các vùng ít xuất hiện như vết bẩn và vùng ướt.

| Lớp | Baseline IoU | Fine-tuned IoU | Baseline Dice/F1 | Fine-tuned Dice/F1 |
|---|---:|---:|---:|---:|
| `background` | 0.920 | 0.911 | 0.958 | 0.953 |
| `dirty_area` | 0.000 | 0.076 | 0.000 | 0.141 |
| `wet_surface` | 0.000 | 0.000 | 0.000 | 0.000 |

Kết quả cho thấy U-Net fine-tuned bắt đầu học được lớp `dirty_area`, trong khi baseline chưa nhận diện được lớp này. Lớp `wet_surface` vẫn chưa tốt do dữ liệu thực tế và annotation còn ít.

### 6.2 Pilot benchmark nghiệp vụ trong service scoring

Checkpoint được test trong service:

```text
models/candidates/unet_existing_real_ft_20260524_174256_best.pt
```

Checkpoint này được copy từ run:

```text
unet_resnet34_real_finetune_20260524_174256
```

Active blob không bị promote hoặc thay đổi. Service được chạy ở chế độ candidate local để so sánh với current active.

| Biến thể | Evaluated | Skipped | Verdict accuracy | False pass | False fail | Pending review | Latency TB | Coverage TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current active, auxiliary disabled | 18 | 1 | 0.4444 | 0.1111 | 0.1111 | 0.5556 | 708.58 ms | 22.721% |
| Candidate U-Net, auxiliary disabled | 18 | 1 | 0.4444 | 0.3889 | 0.0000 | 0.2222 | 543.76 ms | 6.202% |

Kết luận: candidate U-Net chạy được trong service thật và latency thấp hơn, nhưng chưa ổn định hơn baseline để promote. Accuracy không tăng, trong khi false-pass tăng từ 0.1111 lên 0.3889. Điều này nghĩa là candidate đánh giá nhiều ảnh là sạch hơn thực tế, làm tăng rủi ro bỏ sót khu vực chưa đạt.

Trong A/B comparison, candidate tạo 11 verdict flip trên 18 mẫu chung: 4 case cải thiện, 4 case regress, các case còn lại không đổi về đúng/sai. Vì vậy checkpoint này chỉ nên giữ ở trạng thái candidate để phân tích thêm, chưa đưa lên active model.

### 6.3 Trạng thái auxiliary segmentation

SAM3 local đã được smoke test với `SAM3_RESOLUTION=512` và `INFERENCE_BATCH_CONCURRENCY=1`, nhưng local runtime không start được container vì mismatch CUDA:

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8
```

Host hiện dùng RTX 3050 4GB và driver báo CUDA 12.3, trong khi image SAM3 local yêu cầu CUDA 12.8. Vì vậy báo cáo không claim metric định lượng SAM3 local.

Hướng thay thế nhẹ hơn là dùng Roboflow Workflow như provider external cho auxiliary segmentation, vẫn giữ block response `sam3` để tương thích. Roboflow cần được benchmark riêng trên pilot set trước khi claim hiệu quả định lượng.

## 7. Vai trò Roboflow/SAM3 auxiliary segmentation

Roboflow/SAM3-style segmentation không thay thế U-Net tự train. Vai trò của nó là lớp phụ trợ tổng quát:

- Nhận prompt/class như `Garbage`, `Trash`, `Debris`, `Stain`, `Wet_Floor`.
- Sinh vùng nghi ngờ bẩn ở mức mask/bbox/polygon tùy workflow.
- Bổ sung evidence khi U-Net bỏ sót vùng bẩn có hình dạng lạ hoặc không giống dữ liệu train.
- Giúp giảm false-pass nếu model phụ trợ phát hiện được vùng bẩn thật.

U-Net vẫn cần thiết vì bài toán CleanOps có các pattern chuyên biệt: vết bẩn sàn nhỏ, vùng ướt phản quang, vật liệu sàn khác nhau, ánh sáng camera khác nhau. Model lớn có khả năng tổng quát tốt, nhưng không đảm bảo hiểu đúng các nhãn nghiệp vụ của hệ thống. Vì vậy kiến trúc hợp lý là kết hợp:

```text
Foundation auxiliary model = bắt tín hiệu tổng quát
U-Net fine-tuned = học tín hiệu chuyên biệt của CleanOps
Scoring rule = hợp nhất evidence theo cách giải thích được
```

## 8. Human review, retrain và promotion gate

Hệ thống không promote model mới chỉ vì train log nhìn tốt. Quy trình an toàn gồm:

1. AI scoring trả verdict và visualization.
2. Supervisor review các case `PENDING`, `FAIL` hoặc case nghi ngờ.
3. Review tạo ground truth nghiệp vụ và annotation candidate.
4. Khi đủ dữ liệu đã duyệt, hệ thống retrain U-Net candidate.
5. Candidate phải qua benchmark và promotion gate trước khi lên active model.

Trong scope hiện tại, YOLO được giữ ổn định để giảm biến số. Retrain tập trung vào U-Net vì dirty coverage ảnh hưởng trực tiếp tới `quality_score` và verdict.

## 9. Hạn chế

- Pilot benchmark hiện chỉ có 18 mẫu evaluated và 1 mẫu skipped do URL 403, nên chưa đại diện cho toàn bộ production.
- Pixel-level benchmark mới có 8 ảnh thực tế, chưa đủ để claim chất lượng segmentation ở quy mô lớn.
- Lớp `wet_surface` chưa có kết quả tốt vì thiếu dữ liệu/annotation đa dạng.
- Candidate U-Net hiện tăng false-pass trong pilot, nên chưa nên promote.
- SAM3 local chưa benchmark được do CUDA runtime mismatch.
- Roboflow auxiliary segmentation chưa có metric pilot thật trong report này, nên chỉ được trình bày là hướng tích hợp/phụ trợ.

## 10. Hướng phát triển

- Chạy Roboflow Workflow trên pilot benchmark để đo ảnh hưởng tới false-pass, false-fail, pending review và latency.
- Mở rộng pilot set lên 30-50 ảnh thật, cân bằng giữa `clean`, `slightly_dirty`, `obviously_dirty`.
- Tạo golden mask/polygon cho một tập nhỏ để đo mIoU/Dice đúng nghĩa cho U-Net và auxiliary segmentation.
- Bổ sung annotation cho `wet_surface` và các loại sàn/vết bẩn khó.
- Thiết kế promotion gate dựa trên cả metric pixel-level và metric nghiệp vụ, đặc biệt là false-pass.
- Giữ PPE compliance là capability riêng, không trộn metric PPE vào cleanliness scoring.

## 11. Kết luận

CleanOps AI Scoring được thiết kế như một hệ thống AI vận hành được, không phải một model đơn lẻ. YOLO xử lý vật thể/rác ở mức object, U-Net fine-tuned xử lý vùng bẩn/ướt chuyên biệt, và Roboflow/SAM3-style auxiliary segmentation đóng vai trò model phụ trợ tổng quát. Scoring rule hợp nhất evidence thành `quality_score` và verdict `PASS/PENDING/FAIL` theo cách dễ giải thích.

Kết quả hiện tại cho thấy fine-tuning U-Net tạo tín hiệu tích cực ở lớp `dirty_area`, nhưng candidate U-Net chưa đủ ổn định để promote vì làm tăng false-pass trong pilot benchmark. Bước tiếp theo hợp lý là benchmark Roboflow auxiliary trên pilot set, mở rộng dữ liệu thật và chỉ promote model khi qua được promotion gate.

## Cách bảo vệ trước hội đồng

**Vì sao không dùng một model duy nhất?**  
Bài toán gồm nhiều loại evidence khác nhau: vật thể/rác rõ ràng, vùng bẩn nhỏ, vùng ướt phản quang và tín hiệu tổng quát từ prompt segmentation. Tách YOLO, U-Net và auxiliary segmentation giúp từng thành phần có vai trò rõ, dễ benchmark và dễ sửa lỗi.

**Vì sao dùng rule-based score thay vì black-box?**  
Vì supervisor cần hiểu lý do verdict. Công thức `quality_score = clean score - object penalty` giúp giải thích được ảnh bị trừ điểm vì dirty coverage hay vì vật thể/rác.

**Vì sao U-Net tự train vẫn cần dù có model lớn?**  
Model lớn có khả năng tổng quát, nhưng không đảm bảo học đúng các pattern vệ sinh sàn của CleanOps. U-Net fine-tuned học trực tiếp từ dữ liệu/annotation thực tế của hệ thống, nên phù hợp với case chuyên biệt hơn.

**Vì sao chưa promote candidate U-Net?**  
Pilot benchmark cho thấy candidate không tăng verdict accuracy và làm false-pass tăng từ 0.1111 lên 0.3889. Với bài toán vệ sinh, false-pass là rủi ro cao vì hệ thống có thể nghiệm thu nhầm khu vực chưa sạch.

## Tài liệu và artifact tham chiếu

- Pilot benchmark: `benchmarks/cleanliness/pilot_benchmark.csv`
- A/B comparison: `benchmarks/reports/cleanliness_ab_comparison.md`
- SAM3 local blocker: `benchmarks/reports/sam3_smoke_blocker_20260527.md`
- AI report guide: `docs/AI_SCORING_REPORT_GUIDE.md`
- AI overview: `docs/ai/AI_SCORING_OVERVIEW.md`
