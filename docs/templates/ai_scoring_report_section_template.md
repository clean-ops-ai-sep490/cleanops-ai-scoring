# AI Scoring Evaluation

## 1. Muc tieu danh gia

Muc tieu cua phan AI Scoring la chung minh he thong co kha nang ho tro quy trinh danh gia chat luong ve sinh tu anh sau khi thuc hien cong viec. Trong pham vi do an nay, nhom uu tien xay dung mot dich vu AI co the tich hop, van hanh, giai thich ket qua, va cai thien dan theo human feedback, thay vi toi uu mot bai toan AI research chuyen sau.

## 2. Kien truc va metric

He thong `cleanops-ai-scoring` duoc thiet ke theo kien truc hybrid gom hai luong xu ly:

- `YOLOv8` dam nhiem object detection de phat hien rac lon, vat the bat thuong, hoac cac object co the gay anh huong den danh gia do sach.
- `U-Net` dam nhiem semantic segmentation de uoc luong ti le dien tich ban, vet nuoc, va be mat uot tren anh.

Ket qua cua hai luong duoc tong hop thanh `quality_score`, sau do anh xa sang ba muc nghiep vu `PASS`, `PENDING`, va `FAIL` theo nguong cua tung moi truong van hanh.

Trong he thong hien tai:

- metric ky thuat cua YOLO duoc theo doi duoi dang `mAP`
- metric ky thuat cua U-Net duoc theo doi duoi dang `IoU_stain`, `IoU_wet`, va `mIoU`
- metric nghiep vu duoc theo doi thong qua `verdict_accuracy`, `false_pass_rate`, `false_fail_rate`, va `pending_review_rate`
- PPE duoc cung cap bang mot model rieng qua endpoint `POST /ppe/evaluate`, tra ve `PASS/FAIL`, `detected_items`, va `missing_items`

## 3. Phuong phap danh gia

Nhom su dung `pilot benchmark` tren tap anh thuc te thay vi mot benchmark hoc thuat quy mo lon. Cach tiep can nay phu hop voi dinh huong software engineering va dam bao moi so lieu trong bao cao deu co nguon goc ro rang.

Tap pilot duoc xay dung voi [dien so anh] anh, phan bo theo cac nhom:

- `clean`
- `slightly_dirty`
- `obviously_dirty`

Moi anh duoc gan nhan toi thieu bang tay voi:

- `expected_verdict`
- `dirty_level`
- `has_large_trash`

Sau do, nhom chay scoring service de lay:

- `predicted_verdict`
- `quality_score`
- `latency_ms`
- visualization overlay de phuc vu phan tich case study

## 4. Ket qua benchmark

### 4.1 Tong quan model

| Hang muc | Gia tri |
| --- | --- |
| Bai toan | Cleaning quality assessment |
| Dau vao | Anh after |
| Dau ra | Detection, segmentation, quality score, verdict |
| Metric YOLO | `mAP = [dien neu co]` |
| Metric U-Net | `mIoU = [dien neu co]` |
| Metric nghiep vu chinh | `verdict_accuracy = [dien]` |

### 4.2 Benchmark ky thuat toi thieu

| Metric | Gia tri | Ghi chu |
| --- | --- | --- |
| YOLO `mAP@50` hoac `mAP` | [dien neu co] | Lay tu train/valid run that |
| U-Net `IoU_stain` | [dien neu co] | Lay tu train/valid run that |
| U-Net `IoU_wet` | [dien neu co] | Lay tu train/valid run that |
| U-Net `mIoU` | [dien neu co] | Lay tu train/valid run that |
| Average latency | [dien] ms | Lay tu pilot benchmark |

### 4.3 Benchmark nghiep vu

| Metric | Gia tri | Dien giai |
| --- | --- | --- |
| So anh danh gia | [dien] | Tong so anh trong tap pilot |
| Verdict accuracy | [dien] | Ti le verdict trung ground truth |
| False pass rate | [dien] | Ti le model doan PASS sai |
| False fail rate | [dien] | Ti le model doan FAIL sai |
| Pending review rate | [dien] | Ti le case can review thu cong |
| Cases corrected by supervisor | [dien] | So case human review phai sua |

### 4.4 Kha nang van hanh

| Nang luc | Mo ta |
| --- | --- |
| Visualization | He thong co the tra ve anh overlay de kiem tra AI da nhin thay gi |
| Health check | He thong co cac endpoint health de quan sat runtime |
| Model storage | Model active duoc quan ly qua object storage va local cache |
| Promotion gate | Candidate chi duoc promote khi dat dieu kien metric |
| Review-to-retrain | Ket qua review co the duoc dua vao vong lap cai thien model |

### 4.5 PPE compliance capability

Ngoai cleaning quality scoring, service con co luong PPE compliance check phuc vu bai toan kiem tra trang bi bao ho.

| Hang muc | Gia tri |
| --- | --- |
| Dau vao | `image_urls`, `required_objects`, `min_confidence` |
| Dau ra | `status`, `detected_items`, `missing_items` |
| Endpoint | `POST /ppe/evaluate` |
| Endpoint labels | `GET /ppe/labels` |
| Metric khuyen nghi | `ppe_status_accuracy`, `false_missing_rate`, `average_latency_ms` |

Neu nhom dua PPE vao bao cao, nen tach bang benchmark rieng thay vi tron voi cleaning quality scoring. Mot bo metric toi thieu gom:

- `ppe_status_accuracy`
- `missing_item_recall`
- `false_missing_rate`
- `average_latency_ms`

## 5. Case study

Nhom chon [dien so case] truong hop tieu bieu de phan tich dinh tinh, bao gom:

- truong hop model du doan dung ro rang
- truong hop model dua ve `PENDING` va can supervisor review
- truong hop model du doan sai de chi ra gioi han hien tai

Bang case study duoc trinh bay kem visualization overlay de minh hoa su khop hoac khong khop giua nhan dinh cua model va tinh trang thuc te.

## 6. Gioi han va huong phat trien

He thong hien tai van con mot so gioi han:

- tap du lieu danh gia chua lon
- ground truth segmentation chua du cho moi anh trong pilot benchmark
- van can human review doi voi cac case mo ho

Trong giai doan tiep theo, nhom de xuat:

- mo rong tap benchmark co annotation day du hon
- theo doi metric rieng theo tung moi truong
- nang cap retrain pipeline de tan dung tot hon reviewed samples
- neu pham vi PPE duoc day manh hon, bo sung tap benchmark rieng cho tung loai trang bi bao ho

## 7. Luu y ve tinh trung thuc hoc thuat

Trong qua trinh viet bao cao, nhom khong su dung cac smoke metric hoac synthetic artifact lam bang chung benchmark. Cac so lieu trinh bay chi nen den tu train run that, reviewed workflow that, hoac pilot benchmark duoc thu thap co he thong.
