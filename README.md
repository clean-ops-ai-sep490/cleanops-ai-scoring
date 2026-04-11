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
│   ├── download_dataset.py   # Tải dữ liệu YOLO (API) & Cấu trúc thư mục U-Net
│   ├── models/
│   │   ├── yolo_detector.py  # Wrapper cho YOLOv8 (phát hiện rác rắn)
│   │   └── unet_segmenter.py # Wrapper cho U-Net (phân vùng vết bẩn điểm ảnh)
│   ├── train_yolo.py         # Script train riêng rẽ cho YOLO
│   ├── train_unet.py         # Script train riêng rẽ cho U-Net
│   └── infer.py              # Backend gộp kết quả 2 model -> Báo cáo PoC
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

## Chay repo cleaning_ai_poc (Scoring API)

### Cach 1: Chay local Python

1. Tao moi truong va cai package

```powershell
cd e:\capstone\folder\cleaning_ai_poc
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
cd e:\capstone\folder\cleaning_ai_poc
docker build -t cleaning-ai-poc .
docker run --rm -p 8000:8000 -e YOLO_TRAIN_PROJECT_DIR=/app/outputs/yolo_training_4_4 -v "e:/capstone/folder/cleaning_ai_poc/outputs:/app/outputs:ro" cleaning-ai-poc
```

Luu y: can mount thu muc `outputs` de API co the doc model da train (`run_poc_*/weights/best.pt`).

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

---

## Chay full stack qua repo backend

Neu ban muon test full async flow submit/poll (API .NET + worker + queue), xem huong dan trong [cleanopsai-backend/README.md](../cleanopsai-backend/README.md).
