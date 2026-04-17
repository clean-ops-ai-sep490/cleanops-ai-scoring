# Cleaning Quality AI — Hybrid System (PoC)

Hệ thống AI kép (Dual-Model) đánh giá chất lượng vệ sinh từ ảnh Before/After, sử dụng YOLOv8 và U-Net. Cấu hình được tối ưu chạy trên RTX 3050 (4GB/8GB VRAM).

## Danh sách Dataset sử dụng (6 Links từ Đồ án)

Hệ thống được thiết kế để tiêu thụ chính xác 6 nguồn dữ liệu bạn đã cung cấp, chia làm 2 luồng:

### 1. Luồng YOLO (Object Detection / Bounding Box)
Đếm rác nổi, tàn thuốc, vật thể dị thường (`abnormal_objects`):
- [Kaggle: Trash Detection (alyyan)](https://www.kaggle.com/datasets/alyyan/trash-detection) - *Tải tự động qua API*
- [Roboflow: Clean/Unclean Floor](https://universe.roboflow.com/compvision-bfglv/clean-unclean-floor) - *Tải tự động qua API*
- [Mendeley: Indoor Waste Dataset (V1)](https://data.mendeley.com/datasets/dy7smxr93r/1) - *Tải thủ công vào `data/raw/yolo/indoor_waste`*

### 2. Luồng U-Net (Semantic Segmentation / Pixel Mask)
Phân vùng vết bẩn, nước đọng, bùn đất (`detected_stains`, `dirt_coverage`):
- [GitHub: SSGD (GaussianOpenSource)](https://github.com/gaussianopensource/dl_active_cleaning) - *Tải thủ công*
- [Mendeley: Stagnant Water Dataset (V4)](https://data.mendeley.com/datasets/y6zyrnxbfm/4) - *Tải thủ công*

---

## Cấu trúc project mới

```
cleaning_ai/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI entrypoint (scoring production)
│   │   ├── inference_utils.py
│   │   ├── visualization_utils.py
│   │   ├── scoring_utils.py
│   │   ├── temp_store.py
│   │   ├── schemas.py
│   │   └── openapi_utils.py
│   ├── download_dataset.py   # Tải dữ liệu YOLO (API) & Cấu trúc thư mục U-Net
│   ├── models/
│   │   ├── yolo_detector.py  # Wrapper cho YOLOv8 (phát hiện rác rắn)
│   │   └── unet_segmenter.py # Wrapper cho U-Net (phân vùng vết bẩn điểm ảnh)
│   ├── train_yolo.py         # Script train riêng rẽ cho YOLO
│   ├── train_unet.py         # Script train riêng rẽ cho U-Net
├── archive/
│   └── legacy/
│       └── src/
│           ├── infer.py      # Legacy PoC inference (CleaningQualityNet)
│           ├── model.py      # Legacy model definition
│           └── train.py      # Legacy training script
├── data/
│   └── raw/
│       ├── yolo/             # Chứa 3 bộ dataset Bounding Box
│       └── unet/
│           ├── images/       # Chứa ảnh gốc của 3 bộ dataset Mask
│           └── masks/        # Chứa ảnh label đen trắng của 3 bộ Mask
├── models/                   # Checkpoints (yolo.pt, unet.pth)
├── outputs/                  # Reports
└── requirements.txt
```

---

## Bước 1: Setup môi trường & Tải Data

```bash
# Cài đặt requirements chứa ultralytics (yolo) và segmentation-models-pytorch
pip install -r requirements.txt

# Khởi chạy tải Dataset (Kaggle & Roboflow)
python src/download_dataset.py
```
> **Lưu ý:** Các dataset trên Mendeley và Github quá lớn (hàng chục GB) và không có API chính thức. Vui lòng nhấp vào các link tải file ZIP bên trên, sau đó giải nén vào đúng thư mục `data/raw/yolo/` hoặc `data/raw/unet/` tương ứng.

---

## Kiến trúc AI Kép (Hybrid Architecture)

Thay vì ép 1 model làm 2 việc trái ngược nhau, kiến trúc này chạy song song 2 luồng trên 1 ảnh After:
```text
Ảnh After [H, W, 3] 
  │
  ├─> Luồng 1: YOLOv8n (Pretrained COCO)
  │    └─> Đếm Bounding Box -> abnormal_objects (Số rác rắn)
  │
  └─> Luồng 2: U-Net (ResNet18 backbone, sigmoid)
       └─> Nhị phân hoá Pixel (0/1) -> dirt_coverage (% diện tích ố bẩn)
```

**Logic tự động duyệt (Auto-Approve):**
Nếu `abnormal_objects == 0` VÀ `dirt_coverage < 5%` => Điểm Quality Score > 90 => AUTO_APPROVE.

---

## Chay cleanops-ai-scoring (Scoring API)

### Cach 1: Chay local Python

1. Tao moi truong va cai package

```powershell
cd e:\capstone\server-side\cleanops-ai-scoring
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Chay API

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

3. Kiem tra nhanh

- Swagger: http://localhost:8000/docs

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get | ConvertTo-Json -Depth 20
```

### Cach 2: Chay bang Docker

```powershell
cd e:\capstone\server-side\cleanops-ai-scoring
docker compose up -d --build
docker compose ps
```

Mac dinh service API build voi `requirements.inference.txt` de giam footprint runtime, trainer duoc tach rieng qua `Dockerfile.train`.

Runtime image hien tai chi nham muc dich inference + retrain bridge:

- dung `requirements.inference.txt`
- dung `opencv-python-headless` de giam GUI/native baggage
- khong bake model checkpoint vao image
- healthcheck Docker bam vao `GET /health/ready`

### Strict mode (blob-first, fail-fast)

De bat stricter runtime policy cho API (require blob active model) voi 1 compose file duy nhat:

```powershell
$env:APP_RELOAD="false"
$env:MODEL_STORAGE_ENABLED="true"
$env:MODEL_REQUIRE_BLOB="true"
$env:MODEL_FORCE_REFRESH="false"
docker compose up -d --build
docker compose ps
```

Sau khi test xong, neu can quay ve dev mode:

```powershell
Remove-Item Env:APP_RELOAD -ErrorAction SilentlyContinue
Remove-Item Env:MODEL_STORAGE_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:MODEL_REQUIRE_BLOB -ErrorAction SilentlyContinue
Remove-Item Env:MODEL_FORCE_REFRESH -ErrorAction SilentlyContinue
```

### Security checklist truoc khi push/chay shared env

- Khong commit connection string that vao git.
- Dat secret qua .env local (khong commit) hoac secret manager cua CI/cloud.
- Neu secret tung lo, rotate ngay va scrub lich su git.
- Khong de APP_RELOAD=true khi chay production.
- Kiem tra startup logs de chac chan khong dump thong tin nhay cam.

Khong khuyen nghi build API image bang training requirements trong flow thuong xuyen vi lam image to hon dang ke. Chi dung cach nay khi can debug:

```powershell
$env:SCORING_REQUIREMENTS_FILE="requirements.training.txt"
docker compose build cleanops-ai-scoring-api
```

Trainer image van co san theo profile `trainer`, nhung khong con la duong deployment mac dinh:

```powershell
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer
```

### Docker-only quick flow (optional, khong phai duong uu tien)

Neu can test nhanh toan bo bang Docker, uu tien chay script CLI trong trainer container:

```powershell
# 1) Build trainer image
docker compose build cleanops-ai-scoring-trainer

# 2) Export reviewed snapshots -> bridge dataset
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer python scripts/build_retrain_bridge_dataset.py --container retrain-samples --prefix scoring/retrain-samples --output-root data/retrain_bridge --train-ratio 0.8 --valid-ratio 0.1 --fail-mode pseudo --include-fail-unlabeled

# 3) Preprocess U-Net data (neu can)
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer python src/preprocess_unet_data.py

# 4) Train YOLO (tu script, khong qua notebook)
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer python src/train_yolo.py

# 5) Train U-Net (tu script, khong qua notebook)
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer python src/train_unet.py --epochs 30 --batch 4 --workers 0
```

Luu y:

- Notebook van co the chay neu tu setup Jupyter trong container, nhung khong can thiet cho flow retrain nhanh.
- Compose da mount `./data`, `./models`, `./unet_dataset` vao trainer de script chay truc tiep tren du lieu host.
- Neu ban host trainer tren may local va expose bang `ngrok`, khong can chay trainer container.

Co the override lenh train (vi du train U-Net):

```powershell
$env:SCORING_TRAIN_COMMAND="python src/train_unet.py --epochs 30"
docker compose --profile trainer run --rm cleanops-ai-scoring-trainer
```

## Retrain API (Remote Trigger)

Scoring API da co them endpoint retrain de backend poll theo job:

- POST /retrain/jobs
- GET /retrain/jobs/{jobId}

Bien moi truong lien quan trong service `cleanops-ai-scoring-api`:

- RETRAIN_API_ENABLED=true
- RETRAIN_API_KEY= (optional, backend gui qua header X-Retrain-Api-Key)
- RETRAIN_USE_REMOTE_TRAINER=true
- RETRAIN_TRAINER_BASE_URL=https://<your-ngrok-domain>
- RETRAIN_TRAINER_SUBMIT_PATH=/trainer/jobs
- RETRAIN_TRAINER_API_KEY= (optional, API gui qua header X-Trainer-Api-Key)
- RETRAIN_TRAINER_TIMEOUT_SEC=7200
- RETRAIN_COMMAND= (legacy fallback, optional shell command khi khong dung trainer service)
- RETRAIN_COMMAND_TIMEOUT_SEC=7200
- RETRAIN_STORAGE_CONNECTION_STRING= (hoac dung MODEL_STORAGE_CONNECTION_STRING)
- RETRAIN_CONTAINER=retrain
- RETRAIN_EXTERNAL_PREFIX=scoring/external/latest
- RETRAIN_CANDIDATE_YOLO_FILE=outputs/retrain/candidate/yolo_best.pt
- RETRAIN_CANDIDATE_UNET_FILE=outputs/retrain/candidate/unet_best.pth
- RETRAIN_CANDIDATE_METRICS_FILE=outputs/retrain/candidate_metrics.json

Luu y:

- Mac dinh compose da bat RETRAIN_USE_REMOTE_TRAINER=true, nhung `RETRAIN_TRAINER_BASE_URL` can duoc set ro rang theo moi truong.
- Voi flow host trainer local + `ngrok`, scoring API trong Docker se goi trainer qua URL `https://<your-ngrok-domain>`.
- Service trainer chay command train qua bien TRAINER_COMMAND (mac dinh: `python src/train_yolo.py`).
- Neu tat RETRAIN_USE_REMOTE_TRAINER va RETRAIN_COMMAND de trong, API se fallback candidate da co san tren Blob (neu RETRAIN_ALLOW_EXISTING_BLOB_CANDIDATE=true).
- De backend promote duoc, can dam bao candidate artifacts ton tai tai prefix `scoring/external/latest` (hoac prefix ban da set).

## Model Storage (Azure Blob)

Scoring API uu tien nap model tu Azure Blob Storage, sau do cache vao volume local de restart nhanh hon.

- Model active YOLO: `scoring/active/yolo/model.pt`
- Model active U-Net: `scoring/active/unet/model.pth`
- Container mac dinh: `models`

Bien moi truong lien quan:

- `MODEL_STORAGE_ENABLED=true`
- `MODEL_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=<storage_account>;AccountKey=<account_key>;EndpointSuffix=core.windows.net`
- `MODEL_STORAGE_CONTAINER=models`
- `MODEL_STORAGE_ACTIVE_YOLO_KEY=scoring/active/yolo/model.pt`
- `MODEL_STORAGE_ACTIVE_UNET_KEY=scoring/active/unet/model.pth`
- `MODEL_CACHE_DIR=/app/model-cache`
- `MODEL_FORCE_REFRESH=false`
- `SCORING_BLOB_NAMESPACE=scoring`
- `SCORING_BLOB_MODELS_CONTAINER=models`
- `SCORING_BLOB_RETRAIN_CONTAINER=retrain`
- `SCORING_BLOB_EXTERNAL_PREFIX=scoring/external/latest`
- `SCORING_BLOB_ACTIVE_MANIFEST_KEY=scoring/manifests/active.json`

Neu object storage khong co file active, API se fallback sang model local (neu co mount `./models:/app/models:ro`).

## Local Retrain (resource nhe cho server)

Flow de xai may ca nhan train:

1. Train tren may ca nhan (ngoai server).
2. Publish run artifact len Azure Blob container `retrain` theo namespace `scoring/runs/<runId>/...`.
3. Promote run (hoac external candidate) sang key active trong container `models` (`scoring/active/*`).
4. Restart scoring API de nap active model moi.

Giai doan hien tai uu tien Blob-first; khong bat buoc server goi ngrok trigger retrain.

### Publish retrain run artifacts tu may local

Script publish run artifact (model + metrics + optional train log):

```powershell
python scripts/publish_retrain_run_to_blob.py \
  --connection-string "<azure_blob_connection_string>" \
  --container retrain \
  --namespace scoring \
  --run-id run-20260115T103000Z \
  --yolo path/to/yolo_best.pt \
  --unet path/to/unet_best.pth \
  --metrics path/to/run_metrics.json \
  --log path/to/train.log
```

Script se upload vao:

- `scoring/runs/<runId>/artifacts/yolo/model.pt`
- `scoring/runs/<runId>/artifacts/unet/model.pth`
- `scoring/runs/<runId>/metrics/metrics.json`
- `scoring/runs/<runId>/logs/train.log` (neu co)
- `scoring/runs/<runId>/manifests/run.json`

### Upload external candidate artifacts nhanh

Sau khi train tren may ca nhan, co the upload candidate model + metrics len Azure Blob bang script:

```powershell
python scripts/upload_candidate_to_blob.py \
  --connection-string "<azure_blob_connection_string>" \
  --container retrain \
  --prefix scoring/external/latest \
  --yolo path/to/yolo_best.pt \
  --unet path/to/unet_best.pth \
  --metrics path/to/candidate_metrics.json
```

Format metrics JSON can dam bao gate worker doc duoc:

```json
{
  "yolo": { "map": 0.61 },
  "unet": { "miou": 0.73 }
}
```

Worker backend dang duoc cau hinh default doc external candidate prefix `scoring/external/latest` trong Azure Blob container `retrain`.

### Promote run/external candidate sang active models

Sau khi run dat gate, promote artifact sang key active trong container `models`:

```powershell
python scripts/promote_active_scoring_model.py \
  --connection-string "<azure_blob_connection_string>" \
  --source-container retrain \
  --target-container models \
  --namespace scoring \
  --run-id run-20260115T103000Z
```

Neu khong truyen `--run-id`, script se lay external candidate theo `--external-prefix` (default `scoring/external/latest`).

Script promotion tu dong:

- backup active hien tai vao `scoring/archive/<timestamp>/...`
- copy bo model moi vao `scoring/active/...`
- cap nhat manifest `scoring/manifests/active.json`

### Verify model moi da duoc runtime su dung

Sau khi promote, can restart API roi doi chieu health + hash de xac nhan runtime dang dung model moi:

```powershell
docker compose restart cleanops-ai-scoring-api
Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method Get | ConvertTo-Json -Depth 20
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/live" -Method Get | ConvertTo-Json -Depth 20
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health/ready" -Method Get | ConvertTo-Json -Depth 20
```

Ky vong health:

- `/health/live` tra ve process status.
- `/health/ready` tra ve 200 khi YOLO/U-Net da load xong va blob config hop le trong strict mode.
- `yolo_model_source` la `object-storage:downloaded` hoac `object-storage:cache-hit`
- `unet_model_source` la `object-storage:downloaded` hoac `object-storage:cache-hit`
- `yolo_model_path`/`unet_model_path` tro den `/app/model-cache/active/...`

So hash blob active va hash cache trong container:

```powershell
# 1) Lay hash file cache trong container
docker exec cleanops-ai-scoring-cleanops-ai-scoring-api-1 sh -lc "sha256sum /app/model-cache/active/yolo/model.pt /app/model-cache/active/unet/model.pth"

# 2) Download blob active va hash local
az storage blob download --connection-string "$env:MODEL_STORAGE_CONNECTION_STRING" --container-name models --name scoring/active/yolo/model.pt --file tmp/active_yolo.pt --overwrite --output none
az storage blob download --connection-string "$env:MODEL_STORAGE_CONNECTION_STRING" --container-name models --name scoring/active/unet/model.pth --file tmp/active_unet.pth --overwrite --output none
Get-FileHash tmp/active_yolo.pt -Algorithm SHA256
Get-FileHash tmp/active_unet.pth -Algorithm SHA256
```

Neu hash blob active trung hash trong `/app/model-cache/active/*` thi xac nhan API dang dung model moi.

Neu can ep download lai bo qua cache, dat `MODEL_FORCE_REFRESH=true` roi restart lai API.

### Bridge reviewed snapshots thanh dataset train

Sau khi supervisor review `PENDING -> PASS/FAIL`, backend worker se snapshot anh vao container `retrain-samples`.

Script bridge duoi day se doc manifest reviewed va xuat dataset local cho 2 notebook train:

- YOLO: `PASS` -> negative sample (tao file label `.txt` rong)
- U-Net: `PASS` -> mask nen class `0` (toan bo background)
- `FAIL` mac dinh duoc pseudo-label bang model hien tai (confidence-gated)
- Mau `FAIL` kho (do tin cay thap) co the bo qua hoac dua vao bucket unlabeled

```powershell
python scripts/build_retrain_bridge_dataset.py \
  --container retrain-samples \
  --prefix scoring/retrain-samples \
  --output-root data/retrain_bridge \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --fail-mode pseudo \
  --include-fail-unlabeled
```

Sau khi chay xong:

- Bao cao tong hop: `data/retrain_bridge/reports/bridge_summary.json`
- Chi tiet tung mau: `data/retrain_bridge/reports/bridge_index.jsonl`

Ghi chu quan trong:

- Neu khong truyen `--connection-string`, script se tu doc tu env `MODEL_STORAGE_CONNECTION_STRING`.
- Verdict review chi la nhan cap anh, khong co bbox/mask ground truth day du cho `FAIL`.
- Pseudo-label `FAIL` giup closed-loop nhanh; de on dinh hon, nen dat confidence threshold va bo qua mau kho.

Dung lai:

```powershell
docker compose down
```

### Cach 3: Chay image da push len Docker Hub

```powershell
docker run --rm -p 8000:8000 <dockerhub-user>/cleanops-ai-scoring:<tag>
```

Kiem tra nhanh:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get | ConvertTo-Json -Depth 20
```

Ghi chu ve mount:

- Mount la map thu muc tren may host vao container (vi du `-v E:/models:/app/models`).
- Dung mount khi ban muon thay model ma khong can build lai image.
- Neu chi muon ban be pull image tu Docker Hub va chay ngay, khong bat buoc mount.

### Test endpoint evaluate-batch

```powershell
$form = @{
  env = "LOBBY_CORRIDOR"
  image_urls = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
}
Invoke-RestMethod -Uri "http://localhost:8000/evaluate-batch" -Method Post -Form $form | ConvertTo-Json -Depth 60
```

Ket qua hop le se co:

- `summary.processed` > 0
- `results[0].scoring`
- `results[0].yolo`
- `results[0].unet`

### Test endpoint visualize tong hop (YOLO + U-Net + verdict)

Endpoint nay tra ve truc tiep file JPEG da ve khoanh vung, dung de QA nhanh model.

```powershell
$payload = @{ 
  url = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
  env = "LOBBY_CORRIDOR"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/evaluate-url-visualize" -Method Post -ContentType "application/json" -Body $payload -OutFile "E:/temp/hybrid_overlay.jpg"
```

### Endpoint JSON + Base64 cho frontend

Neu frontend can nhan truc tiep chuoi base64 de render len img tag, dung endpoint:

- `POST /evaluate-visualize-json` (upload file)
- `POST /evaluate-url-visualize-json` (URL image)

Vi du URL:

```powershell
$payload = @{ 
  url = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
  env = "LOBBY_CORRIDOR"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/evaluate-url-visualize-json" -Method Post -ContentType "application/json" -Body $payload | ConvertTo-Json -Depth 20
```

Docker note:

- Co the dieu chinh chat luong anh overlay qua env `VISUALIZE_JPEG_QUALITY` (20-100, default 92).
- Sau khi sua code/env, can rebuild container:

```powershell
docker compose up -d --build
```

### Endpoint metadata + blob URL (mobile-friendly)

Neu muon giam payload hon base64, dung endpoint tra ve link blob public:

- `POST /evaluate-visualize-link` (upload file)
- `POST /evaluate-url-visualize-link` (URL image)

Vi du URL:

```powershell
$payload = @{ 
  url = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80"
  env = "LOBBY_CORRIDOR"
} | ConvertTo-Json

$resp = Invoke-RestMethod -Uri "http://localhost:8000/evaluate-url-visualize-link" -Method Post -ContentType "application/json" -Body $payload
$resp.visualization.url
```

Them env:

- `VISUALIZATION_BLOB_ENABLED=true`
- `VISUALIZATION_BLOB_CONNECTION_STRING` (fallback sang `MODEL_STORAGE_CONNECTION_STRING` neu de trong)
- `VISUALIZATION_BLOB_CONTAINER=visualizations`
- `VISUALIZATION_BLOB_PREFIX=scoring/visualizations`

Luu y:

- Link duoc tra ve la public blob URL, frontend/mobile co the mo truc tiep.
- Backend polling `GET /api/scoring/jobs/{jobId}` se expose link nay qua field `visualizationBlobUrl`.

---

## Chay full stack qua repo backend

Neu ban muon test full async flow submit/poll (API .NET + worker + queue), xem huong dan trong [cleanopsai-backend/README.md](../cleanopsai-backend/README.md).
