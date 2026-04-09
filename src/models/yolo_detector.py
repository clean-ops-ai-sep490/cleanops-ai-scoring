from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights_path='yolov8n.pt'):
        """
        Khởi tạo mô hình YOLO. Mặc định dùng yolov8n (Nano) để phù hợp GPU 4GB VRAM.
        """
        self.model = YOLO(weights_path)
    
    def predict(self, image_path, conf=0.25):
        """
        Chạy inference để đếm số lượng rác/vật thể.
        Trả về tổng số bounding box tìm được.
        """
        results = self.model(image_path, conf=conf, verbose=False)
        boxes = results[0].boxes
        return len(boxes)

    def train(
        self,
        data_yaml_path,
        epochs=50,
        batch_size=16,
        imgsz=640,
        project_dir="outputs/yolo",
        run_name="trash_detection",
        device="0",
        half=True,
    ):
        """
        Huấn luyện YOLOv8 trên tập dữ liệu Trash/Objects.
        (batch_size=16 dành cho RTX 3050).
        """
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project_dir,
            name=run_name,
            device=device,
            half=half,
        )
