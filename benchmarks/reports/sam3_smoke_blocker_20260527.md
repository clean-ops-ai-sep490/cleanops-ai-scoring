# SAM3 GPU Smoke Blocker - 2026-05-27

## Result

SAM3 quantitative benchmark was not executed because the local GPU runtime could not start the SAM3 container.

## Smoke Configuration

| Field | Value |
|---|---:|
| Variant | candidate_unet_sam3_enabled |
| SAM3 resolution | 512 |
| Inference batch concurrency | 1 |
| Evaluated samples | 0 |
| Host GPU | NVIDIA GeForce RTX 3050 4GB |
| Host CUDA reported by driver | 12.3 |
| SAM3 container CUDA requirement | >=12.8 |

## Blocking Error

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8, please update your driver to a newer version, or use an earlier cuda container
```

## Conclusion

The SAM3 integration path is present, but this local runtime cannot produce SAM3 stability metrics yet. Use a CUDA/Torch stack compatible with the host driver, or upgrade the NVIDIA driver, then rerun the SAM3 smoke and pilot benchmark.
