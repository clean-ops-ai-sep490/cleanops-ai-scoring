# API Usage and Scoring Guide

Date: 2026-04-09

## 1) Muc tieu tai lieu
Tai lieu nay giup team:
- Chay API local nhanh.
- Hieu endpoint nao dung cho production, endpoint nao dung cho test.
- Hieu ro cach tinh phan tram ban va cach quy doi ra diem/chat luong.

## 2) Setup va chay API
### Cai thu vien
```bash
pip install -r requirements.txt
```

### Cau hinh .env (quan trong)
```bash
cp .env.example .env
```

Sau do cap nhat cac bien nhay cam trong `.env`:
- ROBOFLOW_API_KEY
- KAGGLE_USERNAME
- KAGGLE_KEY

Luu y:
- Khong commit `.env` len git.
- Chi commit `.env.example` de chia se schema cau hinh cho team.

### Chay server
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Hoac de app tu doc host/port/reload tu `.env`:
```bash
python src/api/main.py
```

### Trang can mo
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/

## 3) Danh sach endpoint
### Production
- GET /
- POST /predict
- POST /predict-url
- POST /predict-unet
- POST /predict-unet-url
- POST /evaluate-batch

### Test
- POST /predict-url-visualize
- POST /predict-unet-url-visualize

## 4) Cach su dung nhanh
### 4.1 Single image upload (YOLO)
Endpoint: POST /predict

Body:
- file: anh upload

Response chinh:
- detections_count: so object model phat hien
- results: danh sach bbox, class, confidence

### 4.2 Single image upload (U-Net)
Endpoint: POST /predict-unet

Body:
- file: anh upload

Response chinh:
- stain_or_water_coverage_pct
- wet_surface_coverage_pct
- total_dirty_coverage_pct

### 4.3 Batch image (khuyen nghi cho backend)
Endpoint: POST /evaluate-batch

Form fields:
- files: list file upload (co the bo trong)
- image_urls: list URL anh (co the bo trong)
- env: 1 trong cac gia tri moi truong

Rang buoc:
- Tong so anh toi da 5/request.
- Co the gui ket hop files + image_urls.
- Neu image_urls bi gui thanh 1 chuoi co dau phay, backend tu tach thanh nhieu URL.
- Item loi se bi skip, backend ghi log warning, khong lam fail ca request.

Response chinh:
- summary.total_requested: tong item yeu cau
- summary.processed: so item xu ly thanh cong
- summary.skipped: so item bi bo qua vi loi
- summary.pass / pending / fail: thong ke verdict
- results: chi chua item xu ly thanh cong

## 5) Giai thich phan tram va diem

## 5.1 Cac phan tram tu U-Net
- stain_or_water_coverage_pct: ty le phan tram pixel thuoc lop 1.
- wet_surface_coverage_pct: ty le phan tram pixel thuoc lop 2.
- total_dirty_coverage_pct: ty le tong phan tram ban/uot = lop1 + lop2.

Cong thuc tong quat:
- stain_pct = stain_pixels / total_pixels * 100
- wet_pct = wet_pixels / total_pixels * 100
- total_dirty_pct = (stain_pixels + wet_pixels) / total_pixels * 100

## 5.2 Cach tinh diem clean score
He thong tinh diem tren tung anh theo cong thuc:

1. base_clean_score = 100 - total_dirty_coverage_pct
2. object_penalty = min(30, detections_count * 5)
3. quality_score = clamp(base_clean_score - object_penalty, 0, 100)

Trong do:
- detections_count lay tu YOLO.
- clamp(x, 0, 100) dam bao diem nam trong [0, 100].

## 5.3 Rule PASS/PENDING/FAIL
- PASS neu quality_score >= pass_threshold cua env.
- PENDING neu quality_score < pass_threshold nhung >= 50.
- FAIL neu quality_score < 50.

Pending lower bound hien tai: 50.

Bang threshold theo env:
- LOBBY_CORRIDOR: 90
- RESTROOM: 85
- BASEMENT_PARKING: 80
- GLASS_SURFACE: 90
- OUTDOOR_LANDSCAPE: 80
- HOSPITAL_OR: 95

## 5.4 Vi du tinh diem
Gia su:
- total_dirty_coverage_pct = 12.5
- detections_count = 2
- env = LOBBY_CORRIDOR (threshold = 90)

Tinh:
- base_clean_score = 100 - 12.5 = 87.5
- object_penalty = min(30, 2*5) = 10
- quality_score = 87.5 - 10 = 77.5

Ket luan:
- 77.5 < 90 va >= 50 => verdict = PENDING

## 6) Luu y van hanh cho team
- Neu test tren Swagger ma file upload bi loi UI, restart server va hard refresh trang docs.
- Trong /evaluate-batch, tranh gui hon 5 anh moi request.
- URL loi, timeout, 404 se duoc skip item, request van tra 200 neu con item xu ly duoc.
- Neu tat ca item deu loi, summary.processed co the bang 0.
