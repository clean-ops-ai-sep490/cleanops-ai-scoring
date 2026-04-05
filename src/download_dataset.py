"""
Dataset Downloader cho Hybrid Cleaning Quality AI
=================================================
Tải về các dataset cho 2 mạng song song:

1. YOLO Dataset (Trash/Objects - Bounding Box)
   - Kaggle: alyyan/trash-detection
   - Roboflow: compvision-bfglv/clean-unclean-floor

2. U-Net Dataset (Stains/Water - Segmentation Mask)
   - Do tải tự động từ Mendeley/GitHub bị giới hạn API/token
   - Hệ thống sẽ tạo sẵn thư mục để user bỏ dữ liệu tải thủ công vào.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/raw")
YOLO_DIR = DATA_DIR / "yolo"
UNET_DIR = DATA_DIR / "unet"

def download_yolo_datasets():
    print("=" * 60)
    print("[*] TẢI DỮ LIỆU YOLO (KAGGLE & ROBOFLOW)...")
    print("=" * 60)
    
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Kaggle Trash Detection
    kaggle_dest = YOLO_DIR / "trash_kaggle"
    if not (kaggle_dest / "train").exists():
        try:
            import kaggle
            print(" -> Đang tải 'alyyan/trash-detection' từ Kaggle...")
            kaggle_dest.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files(
                "alyyan/trash-detection",
                path=str(kaggle_dest),
                unzip=True
            )
            print(f" [OK] Đã tải Kaggle Trash Detection vào {kaggle_dest}")
        except Exception as e:
            print(f" [ERROR] Lỗi tải Kaggle: {e}")
    else:
        print(" [OK] Kaggle Trash Dataset đã tồn tại sẵn.")

    # 2. Roboflow Clean/Unclean Floor
    robo_dest = YOLO_DIR / "roboflow_clean"
    # Kiểm tra xem có thư mục con nào không (ví dụ train/test/valid)
    if not robo_dest.exists() or len(list(robo_dest.glob("*"))) == 0:
        try:
            from roboflow import Roboflow
            print(" -> Đang tải 'clean-unclean-floor' từ Roboflow...")
            
            # API Key do user cung cấp
            rf = Roboflow(api_key="1xdovq9T2twKEUwE8FDw")
            project = rf.workspace("compvision-bfglv").project("clean-unclean-floor")
            
            # Tải dataset với chuẩn format yolov8
            dataset = project.version(1).download("yolov8", location=str(robo_dest))
            print(f" [OK] Đã tải Roboflow Dataset vào {robo_dest}")
        except Exception as e:
            print(f" [ERROR] Lỗi tải Roboflow: {e}")
    else:
        print(" [OK] Roboflow Dataset đã tồn tại sẵn.")

def configure_unet_datasets():
    print("\n" + "=" * 60)
    print("[*] CẤU HÌNH DỮ LIỆU U-NET (MENDELEY & GITHUB)...")
    print("=" * 60)
    
    img_dir = UNET_DIR / "images"
    mask_dir = UNET_DIR / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" [INFO] Đã tạo thư mục đón dữ liệu U-Net tại:")
    print(f"        Vị trí đặt ảnh gốc (jpg/png): {img_dir.absolute()}")
    print(f"        Vị trí đặt ảnh mask nhị phân: {mask_dir.absolute()}")
    
    print("\n [!] BƯỚC THỦ CÔNG YÊU CẦU: Các bộ Segmentation quá lớn hoặc bị chặn API, bạn cần:")
    print("     1. Vào Mendeley/Github tải ZIP về (vd link Stagnant Water / SSGD)")
    print("     2. Chép toàn bộ ảnh gốc vào thư mục 'images'.")
    print("     3. Chép toàn bộ ảnh mặt nạ (masks/ground-truths) vào thư mục 'masks'.")
    print("     Script U-Net Training sẽ tự động ánh xạ Data của bạn từ 2 thư mục này.")

def main():
    download_yolo_datasets()
    configure_unet_datasets()
    print("\n[DONE] Hoàn tất Phase 2: Data Preprocessing!")

if __name__ == "__main__":
    main()
