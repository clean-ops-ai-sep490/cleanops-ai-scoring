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
### Production chinh thuc
- GET /
- POST /evaluate-batch
- POST /evaluate-url-visualize-link
- POST /ppe/evaluate

Ghi chu:
- Backend .NET hien tai chi tieu thu `POST /evaluate-batch`, `POST /evaluate-url-visualize-link`, va `POST /ppe/evaluate`.
- Route chuan de test tay anh online va lay blob URL overlay la `POST /evaluate-url-visualize-link`.

### Internal/debug only
- POST /predict
- POST /predict-url
- POST /predict-unet
- POST /predict-unet-url
- POST /predict-url-visualize
- POST /predict-unet-url-visualize
- POST /evaluate-visualize
- POST /evaluate-url-visualize
- POST /evaluate-visualize-json
- POST /evaluate-url-visualize-json
- POST /evaluate-visualize-link

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

## 7) Visual QA endpoint (anh khoanh vung)

Route khuyen nghi duy nhat cho manual testing:
- `POST /evaluate-url-visualize-link`

Muc tieu:
- Tra ve truc tiep anh JPEG da duoc ve khoanh vung de team/khach hang xem AI hoat dong dung hay sai.
- Tren 1 anh se co:
	- Bounding box tu YOLO
	- Vung segmentation tu U-Net (stain/water, wet surface)
	- Bang thong tin verdict + quality score + dirty coverage + detections

### 7.1 Upload file
Endpoint: POST /evaluate-visualize

Form fields:
- file: anh upload
- env: tuy chon, mac dinh LOBBY_CORRIDOR

PowerShell sample:
```powershell
$form = @{
	env = "LOBBY_CORRIDOR"
	file = Get-Item "E:/test-images/floor_01.jpg"
}
Invoke-WebRequest -Uri "http://localhost:8000/evaluate-visualize" -Method Post -Form $form -OutFile "E:/test-images/floor_01_qa.jpg"
```

### 7.2 URL image
Endpoint: POST /evaluate-url-visualize

Body JSON:
```json
{
	"url": "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80",
	"env": "LOBBY_CORRIDOR"
}
```

PowerShell sample:
```powershell
$payload = @{ 
	url = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
	env = "RESTROOM"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/evaluate-url-visualize" -Method Post -ContentType "application/json" -Body $payload -OutFile "E:/test-images/url_qa.jpg"
```

### 7.3 JSON + Base64 cho frontend
Muc tieu:
- Frontend nhan mot JSON duy nhat va render anh truc tiep bang chuoi base64.
- Khong can luu file tam tren server hay tren may client.

Luu y:
- Hai route JSON chi nen xem la internal/debug hoac use case dac biet.
- Flow frontend/mobile/backend uu tien blob URL tu `POST /evaluate-url-visualize-link`.

Endpoint upload: POST /evaluate-visualize-json

Response mau (rut gon):
```json
{
	"source_type": "upload",
	"source": "floor_01.jpg",
	"env": "LOBBY_CORRIDOR",
	"mime_type": "image/jpeg",
	"encoding": "base64",
	"image_base64": "...",
	"scoring": {
		"quality_score": 86.4,
		"verdict": "PENDING"
	},
	"yolo": {
		"detections_count": 2,
		"results": []
	},
	"unet": {
		"total_dirty_coverage_pct": 8.6
	}
}
```

Endpoint URL: POST /evaluate-url-visualize-json

Body JSON:
```json
{
	"url": "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80",
	"env": "LOBBY_CORRIDOR"
}
```

PowerShell sample:
```powershell
$payload = @{ 
	url = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
	env = "RESTROOM"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/evaluate-url-visualize-json" -Method Post -ContentType "application/json" -Body $payload | ConvertTo-Json -Depth 20
```

### 7.4 Docker config lien quan
- Bien moi truong moi: VISUALIZE_JPEG_QUALITY (20-100, mac dinh 92)
- Gia tri cao hon => anh ro hon, payload base64 lon hon.

Them bien cho temporary URL (legacy, khong dung trong flow backend hien tai):
- VISUALIZE_TEMP_URL_TTL_SEC: thoi gian song cua URL tam (giay), mac dinh 900.
- VISUALIZE_TEMP_MAX_ITEMS: so anh overlay tam toi da luu trong memory, mac dinh 200.
- APP_PUBLIC_BASE_URL: base URL public de tao link cho mobile app (vi du https://api-cleanops.example.com).

Them bien cho blob visualization URL:
- VISUALIZATION_BLOB_ENABLED: bat/tat upload anh overlay len blob (mac dinh true).
- VISUALIZATION_BLOB_CONNECTION_STRING: connection string blob storage.
- VISUALIZATION_BLOB_CONTAINER: ten container luu visualization (mac dinh visualizations).
- VISUALIZATION_BLOB_PREFIX: prefix path blob (mac dinh scoring/visualizations).

## 8) Endpoint metadata + blob URL (khuyen nghi cho mobile va demo)

Muc tieu:
- Giam payload so voi base64.
- Mobile app nhan metadata va 1 public blob URL de tai khi can.

### 8.1 URL image
Endpoint: POST /evaluate-url-visualize-link

Body JSON:
```json
{
	"url": "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80",
	"env": "LOBBY_CORRIDOR"
}
```

### 8.2 Luu y
- Service tra ve blob URL trong truong `visualization.url`.
- Khong con endpoint GET `/visualizations/{token}` trong API contract hien tai.
- Backend nen luu va forward `visualizationBlobUrl` cho frontend/mobile.
- Route upload `POST /evaluate-visualize-link` duoc giu lai cho noi bo, nhung khong phai route uu tien de test tay.
