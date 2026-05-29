# CleanOps AI Scoring API

`cleanops-ai-scoring` uses a hybrid cleanliness pipeline: YOLO for trash-like object evidence, U-Net for trained dirty/wet segmentation, and an optional auxiliary foundation segmentation provider. The public response keeps the `sam3` block for compatibility; in demo/report wording this can be Roboflow/SAM3-style auxiliary segmentation.

## Production Endpoints

- `GET /`, `GET /health/live`, `GET /health/ready`, `GET /health/sam3`
- `POST /evaluate-batch`
- `POST /evaluate-url-visualize-link`
- `POST /evaluate-visualize-link`
- `POST /check`
- `POST /ppe/evaluate`

Deprecated debug routes such as `/predict`, `/predict-url`, and `/predict-unet` have been removed.

## `/evaluate-url-visualize-link`

Input:

```json
{
  "url": "https://example.com/image.jpg",
  "env": "LOBBY_CORRIDOR"
}
```

Output includes:

- `visualization.url`: uploaded overlay preview.
- `scoring`: verdict, quality score, penalty counts, and dirty coverage source.
- `yolo`: object detections for trash-like penalties.
- `unet`: trained dirty/wet segmentation summary.
- `sam3`: auxiliary prompt/class segmentation summary and predictions when enabled.

## `/check`

Compatibility endpoint for auxiliary prompt/class segmentation. The response key remains `sam3` for backend compatibility.

Form input:

- `image`: uploaded image file.
- `classes`: comma-separated prompts, default `dirty area`.
- `threshold`: default `0.3`.
- `resolution`: used by local SAM3 runtime; Roboflow-style providers may ignore it.

Response shape:

```json
{
  "predictions": [
    {
      "class": "Stain",
      "prompt": "Stain",
      "confidence": 0.46,
      "bbox_xyxy": [100, 120, 150, 150],
      "x": 100,
      "y": 120,
      "width": 50,
      "height": 30,
      "mask_area_px": 862
    }
  ],
  "outputs": {
    "overlay_url": "https://..."
  }
}
```

## Scoring Rule

- Auxiliary foundation segmentation provides broad dirty-region evidence from prompt text/classes.
- U-Net provides trained segmentation evidence for project-specific stain/wet classes.
- YOLO provides trash-like object detections and object penalties.
- Final dirty coverage is:

```text
combined_dirty_coverage_pct = max(unet_dirty_coverage_pct, sam3_dirty_coverage_pct)
```

The existing `scoring` block is preserved and now also includes:

- `dirty_coverage_source`
- `unet_dirty_coverage_pct`
- `sam3_dirty_coverage_pct`
- `combined_dirty_coverage_pct`

## Runtime Notes

- External LLM/Gemini verification is not part of the active cleanliness scoring path.
- For normal CPU/dev runtime, keep auxiliary segmentation disabled unless testing a provider.
- For the lightweight demo path, use Roboflow Workflow as the auxiliary provider and keep secrets in env only.
- Local SAM3 remains a compatibility path, but its GPU image was blocked on this host by CUDA 12.8 vs driver CUDA 12.3, so it should not be the primary demo claim.
