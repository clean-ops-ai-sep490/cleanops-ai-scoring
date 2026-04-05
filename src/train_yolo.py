import os
from pathlib import Path
from src.models.yolo_detector import YoloDetector

def main():
    print("=" * 60)
    print("YOLOv8 TRAINING PIPELINE (TRASH / ABNORMAL OBJECTS)")
    print("=" * 60)

    # Khởi tạo mô hình
    print("[*] Đang nạp mô hình YOLOv8 Nano...")
    detector = YoloDetector(weights_path='yolov8n.pt')

    # Đường dẫn file data.yaml của Roboflow (sau khi tải tự động)
    # Lưu ý: Cấu trúc tải từ Roboflow thường có file data.yaml nằm sẵn ở gốc
    roboflow_dir = Path("data/raw/yolo/roboflow_clean")
    data_yaml = roboflow_dir / "data.yaml"

    if not data_yaml.exists():
        print(f"[ERROR] Không tìm thấy file cấu hình YOLO tại: {data_yaml}")
        print("        Hãy chạy 'python src/download_dataset.py' trước để tải dữ liệu tự động.")
        return

    print(f"[*] Bắt đầu huấn luyện với dataset: {data_yaml}")
    print("    Cấu hình: RTX 3050 (Batch=16, Image Size=640, Mixed Precision=True)")
    
    # Bắt đầu train
    detector.train(
        data_yaml_path=str(data_yaml),
        epochs=50,       # Bạn có thể giảm xuống 10-20 để test thử
        batch_size=16,   # Tối ưu cho VRAM 4GB-8GB
        imgsz=640,
        project_dir="outputs/yolo"
    )

    print("\n[DONE] Đã hoàn tất huấn luyện luồng YOLO (Bounding Box).")
    print("       Mô hình tốt nhất được lưu tại: outputs/yolo/trash_detection/weights/best.pt")

if __name__ == "__main__":
    main()
