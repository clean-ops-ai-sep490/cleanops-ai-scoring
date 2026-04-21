# Cleaning AI POC - Status Update

Date: 2026-04-09

## 1) Tong quan
- Du an hien tai da chuyen sang mo hinh Hybrid YOLO + U-Net de danh gia do sach be mat.
- Backend FastAPI da ho tro cả single image va batch processing.
- Swagger da duoc dieu chinh de test tien loi hon cho file upload + URL.

## 2) Tinh trang chuc nang
### API san sang
- Health check: / (production)
- Batch evaluate: /evaluate-batch (production)
- Visualization blob URL: /evaluate-url-visualize-link (production)
- PPE evaluate: /ppe/evaluate (production)

### Internal/debug routes
- /predict
- /predict-url
- /predict-unet
- /predict-unet-url
- /predict-url-visualize
- /predict-unet-url-visualize
- /evaluate-visualize
- /evaluate-url-visualize
- /evaluate-visualize-json
- /evaluate-url-visualize-json
- /evaluate-visualize-link

Ghi chu:
- Flow backend hien tai chi can /evaluate-batch, /evaluate-url-visualize-link, va /ppe/evaluate.
- Route uu tien de test tay va lay visualization blob URL la /evaluate-url-visualize-link.

### Batch evaluate (cap nhat moi)
- Gioi han toi da 5 anh/request.
- Ho tro ket hop file upload va image_urls.
- Neu image_urls bi gui thanh 1 chuoi co dau phay, backend se tu tach thanh nhieu URL.
- Neu 1 anh loi (URL hong, file loi), backend bo qua item do va tiep tuc xu ly cac item khac.
- Loi item duoc log o backend, khong tra error chi tiet ra response.
- Response summary co them truong skipped de theo doi so item bo qua.

## 3) Tinh trang pipeline training
### U-Net multiclass
- Da chuan hoa segmentation 3 lop:
  - 0: background
  - 1: stain_or_water
  - 2: wet_surface
- Da co script preprocess du lieu:
  - src/preprocess_unet_data.py
  - Hop nhat HD10K + Stagnant Water va export ve data/processed/unet_multiclass
- Da refactor train script:
  - src/train_unet.py
  - Loss: Dice + Focal
  - Metric: IoU class 1/2 + mIoU_12
  - Luu checkpoint tot nhat
- Da cap nhat model wrapper:
  - src/models/unet_segmenter.py

### Notebook
- Da cap nhat notebook train U-Net:
  - notebooks/03_train_unet.ipynb

## 4) File chinh dang thay doi trong nhanh hien tai
- .gitignore
- .env.example (new)
- requirements.txt
- src/api/main.py
- src/config/settings.py (new)
- src/config/__init__.py (new)
- src/download_dataset.py
- src/train_yolo.py
- src/models/unet_segmenter.py
- src/models/yolo_detector.py
- src/train_unet.py
- src/preprocess_unet_data.py (new)
- notebooks/01_pipeline_manager.ipynb
- notebooks/03_train_unet.ipynb
- API_USAGE_AND_SCORING.md (new)

## 5) Ve quan ly file de push
- Da bo sung .gitignore de chan cac file khong can thiet:
  - checkpoint lon (*.pt, *.pth, *.ckpt)
  - cache Python (__pycache__, *.pyc)
  - logs va temp
  - data/output local
  - file env local (.env, .env.*), giu lai .env.example
- Da untrack cac file pycache da bi track truoc do.

## 6) Cap nhat env cho production
- Da bo hard-code API key Roboflow trong source.
- Da dua cac bien config quan trong (model path, timeout, threshold, host/port, train defaults) ve .env.
- API, downloader, train scripts va notebook manager da doc config tu .env.

## 7) Luu y truoc khi merge
- Chua co bo test tu dong cho API va training scripts.
- Nen chot them API contract (schema response) cho frontend/backend tich hop on dinh.
- Neu can reproducible training, nen bo sung file cau hinh (yaml/json) cho hyper-parameters.
