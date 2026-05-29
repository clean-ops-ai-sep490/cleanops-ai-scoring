# Cleaning AI POC - Status Update

Date: 2026-05-27

## 1) Tong quan
- Du an hien tai dung luong Hybrid Auxiliary Segmentation + YOLO + U-Net de danh gia do sach be mat.
- Block response `sam3` duoc giu de tuong thich, nhung provider demo uu tien la Roboflow/SAM3-style auxiliary segmentation; YOLO/U-Net van la evidence domain-specific.
- Gemini/LLM verification da duoc loai khoi active scoring path de giam do tre va phu thuoc external API key.
- Backend FastAPI ho tro batch processing, visualization blob URL, auxiliary prompt check va PPE evaluate.

## 2) Tinh trang chuc nang
### API san sang
- Health check: /, /health/live, /health/ready, /health/sam3 (production)
- Batch evaluate: /evaluate-batch (production)
- Visualization blob URL: /evaluate-url-visualize-link (production)
- Upload visualization blob URL: /evaluate-visualize-link (internal upload)
- Auxiliary prompt segmentation compatibility: /check (production)
- PPE evaluate: /ppe/evaluate (production)

### Routes da remove/deprecated
- Cac debug route cu nhu /predict, /predict-url, /predict-unet, /predict-unet-url va cac route visualize JSON/base64 cu da bi remove khoi active API.
- Backend .NET hien tai chi can /evaluate-batch, /evaluate-url-visualize-link va /ppe/evaluate.
- Neu can test rieng auxiliary prompt segmentation thi dung /check.

Ghi chu:
- Route uu tien de test tay va lay visualization blob URL la /evaluate-url-visualize-link.
- Response scoring giu contract cu va bo sung dirty_coverage_source, unet_dirty_coverage_pct, sam3_dirty_coverage_pct, combined_dirty_coverage_pct.
- Response co them block sam3 khi auxiliary detector tra ve ket qua/status.

### Batch evaluate (cap nhat moi)
- Gioi han toi da 5 anh/request.
- Ho tro ket hop file upload va image_urls.
- Neu image_urls bi gui thanh 1 chuoi co dau phay, backend se tu tach thanh nhieu URL.
- Neu 1 anh loi (URL hong, file loi), backend bo qua item do va tiep tuc xu ly cac item khac.
- Loi item duoc log o backend, khong tra error chi tiet ra response.
- Response summary co them truong skipped de theo doi so item bo qua.

### Runtime auxiliary segmentation
- CPU/dev runtime: giu auxiliary segmentation disabled de service van chay nhe voi YOLO/U-Net, khong goi Gemini.
- Demo nhe: dung Roboflow Workflow lam provider external, khong hardcode API key trong source/report.
- Local SAM3 chi la compatibility path; smoke test tren host nay bi chan boi CUDA 12.8 vs driver CUDA 12.3, nen khong dung lam claim dinh luong.

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
- Retrain U-Net chi nen chay khi co approved annotations du chat luong va benchmark baseline/candidate ro rang.
- YOLO duoc freeze trong scope hien tai de giam bien so khi bao ve promotion gate.

## 4) Benchmark va bao ve
- Benchmark cleanliness pilot hien nam o benchmarks/cleanliness/pilot_benchmark.csv.
- Ket qua chay model that nen ghi vao benchmarks/reports/cleanliness_pilot_evaluated.csv.
- Summary bao ve nen sinh bang scripts/summarize_pilot_benchmark.py thanh JSON va Markdown.
- PPE benchmark la capability rieng; khong tron metric PPE vao cleanliness scoring.
- Golden mask benchmark chua tao vi repo chua co polygon/mask that da duyet.

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
- Auxiliary/SAM3 compatibility config nam trong .env.example; Roboflow key phai de trong env/secret runtime, khong commit.

## 7) Luu y truoc khi demo/bao ve
- Chay Python unit tests: python -m unittest discover -s tests -v.
- Chay backend ScoringJobServiceTests de xac nhan contract sam3/visualization payload.
- Smoke test /health/ready, /health/sam3 va /evaluate-url-visualize-link voi service dang chay.
- Chay cleanliness pilot benchmark va commit report that neu dung lam bang chung.
