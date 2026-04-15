#!/bin/sh
set -eu

docker run --rm -v cleanopsai-scoring-outputs:/outputs alpine sh -c '
mkdir -p /outputs/retrain/candidate
printf fake-yolo > /outputs/retrain/candidate/yolo_best.pt
printf fake-unet > /outputs/retrain/candidate/unet_best.pth
cat > /outputs/retrain/candidate_metrics.json <<JSON
{"yolo":{"map":0.61},"unet":{"miou":0.73}}
JSON
'

ls -la /shared/scoring-outputs/retrain
ls -la /shared/scoring-outputs/retrain/candidate
cat /shared/scoring-outputs/retrain/candidate_metrics.json
