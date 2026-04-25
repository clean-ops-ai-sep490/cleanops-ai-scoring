# AI Scoring Report Guide

Tai lieu nay bien ke hoach bao cao thanh bo artifact co the dung ngay cho do an.
Huong tiep can uu tien tinh trung thuc, de giai thich, va phu hop voi hoi dong nghieng ve software engineering hon la AI research.

## 1. Dinh huong viet bao cao

Diem manh cua `cleanops-ai-scoring` khong nen duoc trinh bay nhu mot mo hinh SOTA.
Nen trinh bay no nhu mot he thong AI co:

- kien truc hybrid de giai quyet 2 bai toan khac nhau
- metric ky thuat ro rang (`yolo.map`, `unet.miou`)
- metric nghiep vu ro rang (`quality_score`, `PASS/PENDING/FAIL`)
- kha nang PPE compliance check bang model rieng
- human review loop
- retrain pipeline
- promotion gate truoc khi dua model moi len runtime

Mot cau mo dau co the dung:

> He thong AI scoring duoc thiet ke de danh gia anh sau khi ve sinh va ho tro quy trinh giam sat chat luong. Muc tieu cua nhom khong phai toi uu mot bai toan AI research chuyen sau, ma la xay dung mot dich vu AI co kha nang van hanh, giai thich duoc, va co co che cai thien theo human feedback.

## 2. Nhung gi codebase hien tai da co san

Ban co the viet phan nay dua truc tiep tren implementation hien tai:

- `YOLOv8` duoc dung cho object detection trong `src/train_yolo.py` va `src/models/yolo_detector.py`
- `U-Net` duoc dung cho segmentation trong `src/train_unet.py` va `src/models/unet_segmenter.py`
- PPE duoc xu ly boi model rieng qua `src/api/routers/ppe.py` va `src/api/ppe_utils.py`
- `quality_score` duoc tinh tu dirty coverage va object penalty trong `API_USAGE_AND_SCORING.md`
- nguong verdict theo tung moi truong duoc cau hinh trong `src/config/settings.py`
- supervisor review -> annotation candidate duoc tao trong `cleanops-backend/src/Modules/Scoring/CleanOpsAi.Modules.Scoring.Infrastructure/Consumers/ScoringResultReviewedConsumer.cs`
- retrain gate va promotion logic dua tren `yolo.map` + `unet.miou` nam trong `cleanops-backend/src/Modules/Scoring/CleanOpsAi.Modules.Scoring.Infrastructure/Consumers/ScoringRetrainRequestedConsumer.cs`

Mot cau mo ta kien truc ngan gon:

> He thong tach bai toan danh gia do sach thanh hai luong: YOLO xu ly object-level noise nhu rac lon hoac vat the bat thuong, trong khi U-Net xu ly pixel-level dirt/wet coverage. Ket qua cua hai luong duoc tong hop thanh `quality_score`, sau do quy doi sang `PASS/PENDING/FAIL` theo tung moi truong van hanh.

Mot cau mo ta them cho PPE:

> Ngoai cleaning quality scoring, service con cung cap luong PPE compliance check bang mot model rieng. Dau vao cua luong nay la danh sach `image_urls` va danh sach `required_objects`, dau ra la `PASS/FAIL`, cac `detected_items`, va `missing_items`, phu hop voi bai toan kiem tra trang bi bao ho lao dong.

## 3. Nhung gi khong nen dua vao bao cao nhu metric that

Khong nen dung cac so lieu sau nhu bang chung benchmark:

- `outputs/retrain/candidate_metrics.json` neu file nay duoc tao boi smoke flow
- bat ky metric nao den tu `scripts/run_local_retrain_smoke.py`
- bat ky bang ket qua nao duoc dien bang tay ma khong co nguon anh va quy trinh danh gia ro rang

Luu y quan trong:

`cleanops-ai-scoring/scripts/run_local_retrain_smoke.py` hien tao metrics mau:

- `yolo.map = 0.999`
- `unet.miou = 0.999`

Do day la smoke artifact phuc vu kiem tra pipeline, khong phai ket qua huan luyen that.

## 4. Cach lam pilot benchmark trung thuc nhung van kip

Khuyen nghi dung `pilot benchmark` thay vi tuyen bo benchmark hoc thuat lon.

### Buoc 1: Chuan bi tap anh

Thu thap 30-50 anh that, can bang toi thieu theo:

- `clean`
- `slightly_dirty`
- `obviously_dirty`

Neu kip, chia them theo moi truong:

- `LOBBY_CORRIDOR`
- `RESTROOM`
- `OUTDOOR_LANDSCAPE`

### Buoc 2: Tao ground truth muc toi thieu

Khong can label bbox/mask day du cho toan bo tap pilot.
Chi can danh tay cho tung anh:

- `expected_verdict`
- `dirty_level`
- `has_large_trash`

Neu can, co the dung `reviewed_verdict` cua supervisor lam ground truth nghiep vu.

### Buoc 3: Chay he thong tren tap pilot

Lay toi thieu cac truong sau cho moi anh:

- `predicted_verdict`
- `quality_score`
- `latency_ms`
- `visualization_url` hoac hinh overlay luu rieng

### Buoc 4: Tong hop metric

Dung script:

```powershell
python scripts/summarize_pilot_benchmark.py `
  --input-csv docs/templates/pilot_benchmark_dataset_template.csv `
  --output-json outputs/reports/pilot_benchmark_summary.json `
  --output-md outputs/reports/pilot_benchmark_summary.md
```

Khi co du lieu that, thay file CSV template bang file da dien.

## 5. Metric nen dua vao bao cao

### 5.1 Metric ky thuat

Neu ban co train log that, dung:

- YOLO: `mAP@50` hoac `mAP`
- U-Net: `IoU_stain`, `IoU_wet`, `mIoU`

Trong codebase hien tai:

- U-Net training in ra `IoU_stain`, `IoU_wet`, `mIoU_12`
- retrain gate dung `yolo.map` va `unet.miou`

### 5.2 Metric nghiep vu

Nen uu tien dua vao bao cao:

- `verdict_accuracy`
- `false_pass_rate`
- `false_fail_rate`
- `pending_review_rate`
- `average_latency_ms`

Dinh nghia khuyen nghi:

- `false_pass`: model doan `PASS` trong khi ground truth khong phai `PASS`
- `false_fail`: model doan `FAIL` trong khi ground truth la `PASS`
- `pending_review_rate`: ty le anh bi day vao `PENDING`

### 5.3 Metric cho PPE

Neu ban dua PPE vao bao cao, nen tach no thanh mot muc rieng thay vi nhap vao scoring cleanliness.

Metric khuyen nghi:

- `ppe_status_accuracy`
- `missing_item_recall`
- `false_missing_rate`
- `average_latency_ms`

Dinh nghia khuyen nghi:

- `ppe_status_accuracy`: ty le `PASS/FAIL` cua PPE trung ground truth
- `missing_item_recall`: ti le item thieu that su duoc model bao thieu
- `false_missing_rate`: ti le model bao thieu nhung ground truth khong thieu

Capability that trong codebase hien tai:

- endpoint production: `POST /ppe/evaluate`
- endpoint xem nhan model: `GET /ppe/labels`
- model path rieng: `PPE_MODEL_PATH`
- blob key rieng: `MODEL_STORAGE_ACTIVE_PPE_KEY`

## 6. Cau truc muc bao cao nen viet

Ban co the copy truc tiep tu file:

- `docs/templates/ai_scoring_report_section_template.md`

Cau truc nay da gom:

- muc tieu danh gia
- kien truc va metric
- benchmark ky thuat toi thieu
- benchmark nghiep vu
- PPE compliance capability
- kha nang van hanh
- gioi han va huong phat trien

## 7. Bang bao cao khuyen nghi

### Bang 1. Tong quan model

| Hang muc | Noi dung |
| --- | --- |
| Bai toan | Cleaning quality assessment tu anh after |
| Dau vao | 1 anh sau khi ve sinh |
| Dau ra | YOLO detections, U-Net mask, dirty coverage, quality score, verdict |
| Metric YOLO | `mAP` |
| Metric U-Net | `mIoU` |
| Metric nghiep vu | Verdict accuracy sau review |

### Bang 2. Benchmark ky thuat toi thieu

| Metric | Nguon du lieu |
| --- | --- |
| `mAP@50` hoac `mAP` | train/valid log YOLO that |
| `IoU_stain` | train/valid log U-Net that |
| `IoU_wet` | train/valid log U-Net that |
| `mIoU` | train/valid log U-Net that |
| `average_latency_ms` | pilot benchmark CSV |

### Bang 3. Benchmark nghiep vu

| Metric | Y nghia |
| --- | --- |
| `verdict_accuracy` | muc do dung cua verdict tong hop |
| `false_pass_rate` | rui ro bo sot khu vuc chua sach |
| `false_fail_rate` | rui ro danh gia qua nghiem |
| `pending_review_rate` | ti le can can thiep thu cong |
| `cases_corrected_by_supervisor` | so case human review phai sua |

### Bang 4. Kha nang van hanh

| Nang luc | Bang chung |
| --- | --- |
| Visualization | endpoint visualize / blob URL |
| Health check | `/`, `/health/live`, `/health/ready` |
| Model storage | Azure Blob + model cache |
| Promotion gate | `yolo.map` + `unet.miou` |
| Review-to-retrain | reviewed snapshot -> candidate -> retrain |

### Bang 5. PPE compliance capability

| Hang muc | Noi dung |
| --- | --- |
| Bai toan | Kiem tra PPE compliance |
| Dau vao | `image_urls` + `required_objects` |
| Dau ra | `status`, `detected_items`, `missing_items` |
| Endpoint | `POST /ppe/evaluate` |
| Labels | `GET /ppe/labels` |
| Model path | `PPE_MODEL_PATH` hoac blob active PPE |
| Metric nghiep vu | `ppe_status_accuracy`, `false_missing_rate` |

## 8. Cach tu ve minh truoc cau hoi cua hoi dong

Neu hoi dong hoi "Tai sao benchmark AI chua sau?", ban co the tra loi:

> Do gioi han thoi gian va dinh huong software engineering, nhom uu tien danh gia muc do hoat dong o cap he thong va nghiep vu truoc. Phan benchmark hoc thuat sau hon, voi tap test lon va annotation day du, duoc de xuat la huong phat trien tiep theo.

Neu hoi "Tai sao khong dung 1 model duy nhat?", ban co the tra loi:

> Bai toan gom hai muc tieu khac nhau: phat hien vat the bat thuong va do muc do ban tren be mat. Tach thanh YOLO + U-Net giup de giai thich ket qua, de benchmark tung thanh phan, va de retrain co trong tam hon.

## 9. Artifact di kem da duoc tao

- `docs/templates/ai_scoring_report_section_template.md`: doan bao cao co the copy vao luan van
- `docs/templates/pilot_benchmark_dataset_template.csv`: file dien du lieu pilot benchmark
- `docs/templates/pilot_benchmark_case_studies_template.csv`: file danh sach case study minh hoa
- `docs/templates/ppe_pilot_benchmark_template.csv`: file dien du lieu pilot benchmark cho PPE
- `scripts/summarize_pilot_benchmark.py`: script tong hop metric business tu file CSV
- `scripts/summarize_ppe_benchmark.py`: script tong hop metric business cho PPE tu file CSV
