import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.yolo_detector import YoloDetector

def main():
    print("=" * 60)
    print("YOLOv8 TRAINING PIPELINE (TRASH / ABNORMAL OBJECTS)")
    print("=" * 60)

    # Khởi tạo mô hình
    print("[*] Đang nạp mô hình YOLOv8 Nano...")
    detector = YoloDetector(weights_path=settings.yolo_weights_path)

    # Đường dẫn file data.yaml của Roboflow (sau khi tải tự động)
    # Lưu ý: Cấu trúc tải từ Roboflow thường có file data.yaml nằm sẵn ở gốc
    data_yaml = settings.yolo_data_yaml

    if not data_yaml.exists():
        print(f"[ERROR] Không tìm thấy file cấu hình YOLO tại: {data_yaml}")
        print("        Hãy chạy 'python src/download_dataset.py' trước để tải dữ liệu tự động.")
        return

    print(f"[*] Bắt đầu huấn luyện với dataset: {data_yaml}")
    print(
        "    Cau hinh: "
        f"epochs={settings.yolo_train_epochs}, "
        f"batch={settings.yolo_train_batch}, "
        f"imgsz={settings.yolo_train_imgsz}, "
        f"device={settings.yolo_device}, "
        f"half={settings.yolo_use_half}"
    )
    
    # Bắt đầu train
    detector.train(
        data_yaml_path=str(data_yaml),
        epochs=settings.yolo_train_epochs,
        batch_size=settings.yolo_train_batch,
        imgsz=settings.yolo_train_imgsz,
        project_dir=settings.yolo_project_dir,
        run_name=settings.yolo_run_name,
        device=settings.yolo_device,
        half=settings.yolo_use_half,
    )

    print("\n[DONE] Đã hoàn tất huấn luyện luồng YOLO (Bounding Box).")
    print(
        "       Mo hinh tot nhat duoc luu tai: "
        f"{settings.yolo_project_dir}/{settings.yolo_run_name}/weights/best.pt"
    )

if __name__ == "__main__":
    main()
