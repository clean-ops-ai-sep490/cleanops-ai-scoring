from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import uvicorn
from ultralytics import YOLO
import io
from PIL import Image
import os
import requests
from pydantic import BaseModel

app = FastAPI(
    title="Cleaning AI POC API",
    description="API for YOLOv8 Object Detection on Trash and Dirt",
    version="1.0.0"
)

# Đường dẫn tới model đã train thành công
# (Lưu ý: Tùy thuọc vào project name trong notebooks mà folder có thể là run_poc_1, run_poc_2...)
MODEL_PATH = r"E:\capstone\cleaning_ai_poc\outputs\yolo_training_4_4\run_poc_1\weights\best.pt"

# Load model global để tái sử dụng
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("✅ Load model thành công!")
    else:
        model = None
        print(f"⚠️ Không tìm thấy model tại {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Lỗi khởi tạo model: {e}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "Welcome to Cleaning AI Object Detection API"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        # Đọc dữ liệu ảnh
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Dự đoán
        results = model.predict(source=img, conf=0.25, save=False)
        
        # Trích xuất kết quả
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # box.xyxy: [xmin, ymin, xmax, ymax], box.conf: độ tự tin, box.cls: nhãn
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                
                detections.append({
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": float(f"{conf:.3f}"),
                    "bbox": [x1, y1, x2, y2]
                })
                
        return {
            "filename": file.filename,
            "detections_count": len(detections),
            "results": detections
        }
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

class ImageURL(BaseModel):
    url: str

@app.post("/predict-url")
async def predict_image_url(payload: ImageURL):
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        # Tải ảnh từ URL
        response = requests.get(payload.url)
        response.raise_for_status() # Báo lỗi nếu URL gặp vấn đề truy cập, chặn, die link...
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Dự đoán
        results = model.predict(source=img, conf=0.25, save=False)
        
        # Trích xuất kết quả
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # box.xyxy: [xmin, ymin, xmax, ymax], box.conf: độ tự tin, box.cls: nhãn
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                
                detections.append({
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": float(f"{conf:.3f}"),
                    "bbox": [x1, y1, x2, y2]
                })
                
        return {
            "url": payload.url,
            "detections_count": len(detections),
            "results": detections
        }
        
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict-url-visualize")
async def predict_image_url_visualize(payload: ImageURL):
    """
    Nhận diện qua URL và trả về trực tiếp ảnh đã vẽ khung (ảnh kết quả)
    giúp dễ dàng xem bằng mắt thường.
    """
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        # Tải ảnh gốc từ URL
        response = requests.get(payload.url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Dự đoán
        results = model.predict(source=img, conf=0.25, save=False)
        
        # Lấy mảng dữ liệu ảnh đã được YOLO vẽ sẵn khung (numpy array)
        im_array = results[0].plot()  # raw numpy [H, W, C] (BGR)
        
        # Đảo màu BGR sang RGB cho Pillow (vì OpenCV vẽ theo chuẩn BGR)
        im_array_rgb = im_array[..., ::-1]
        
        # Chuyển numpy array thành dạng Image
        res_img = Image.fromarray(im_array_rgb)
        
        # Lưu vào Byte Buffer (bộ nhớ ảo) thay vì ghi ổ cứng
        img_byte_arr = io.BytesIO()
        res_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Trả trực tiếp file ảnh .jpeg ra trình duyệt / màn hình UI
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
        
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
