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
