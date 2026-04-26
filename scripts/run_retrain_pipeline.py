from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def safe_slug(raw: str, default: str) -> str:
    value = (raw or "").strip() or default
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or default


def ensure_inside_project(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to modify path outside project root: {resolved}") from exc


def recreate_dir(path: Path) -> None:
    ensure_inside_project(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"[RUN] {' '.join(args)}", flush=True)
    proc = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(args)}")
    return proc


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def list_images(path: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(sorted(path.glob(pattern)))
    return files


def move_sample(root: Path, source_split: str, target_split: str, image: Path) -> None:
    label = root / "yolo" / "labels" / source_split / f"{image.stem}.txt"
    target_image = root / "yolo" / "images" / target_split / image.name
    target_label = root / "yolo" / "labels" / target_split / f"{image.stem}.txt"
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image), str(target_image))
    if label.exists():
        shutil.move(str(label), str(target_label))

    unet_image = root / "unet" / "images" / source_split / image.name
    unet_mask = root / "unet" / "masks" / source_split / f"{image.stem}.png"
    target_unet_image = root / "unet" / "images" / target_split / image.name
    target_unet_mask = root / "unet" / "masks" / target_split / f"{image.stem}.png"
    target_unet_image.parent.mkdir(parents=True, exist_ok=True)
    target_unet_mask.parent.mkdir(parents=True, exist_ok=True)
    if unet_image.exists():
        shutil.move(str(unet_image), str(target_unet_image))
    if unet_mask.exists():
        shutil.move(str(unet_mask), str(target_unet_mask))


def ensure_train_valid_splits(dataset_root: Path) -> None:
    train_images = list_images(dataset_root / "yolo" / "images" / "train")
    valid_images = list_images(dataset_root / "yolo" / "images" / "valid")
    test_images = list_images(dataset_root / "yolo" / "images" / "test")

    if not train_images:
        source_split = "valid" if valid_images else "test"
        source_images = valid_images if valid_images else test_images
        if not source_images:
            raise RuntimeError("No exported YOLO images found in train/valid/test splits.")
        move_sample(dataset_root, source_split, "train", source_images[0])

    train_images = list_images(dataset_root / "yolo" / "images" / "train")
    valid_images = list_images(dataset_root / "yolo" / "images" / "valid")
    test_images = list_images(dataset_root / "yolo" / "images" / "test")

    if not valid_images:
        if test_images:
            move_sample(dataset_root, "test", "valid", test_images[0])
        elif len(train_images) > 1:
            move_sample(dataset_root, "train", "valid", train_images[-1])
        else:
            raise RuntimeError("At least two exported samples are required to create train and valid splits.")


def write_yolo_data_yaml(dataset_root: Path) -> Path:
    yolo_root = dataset_root / "yolo"
    data_yaml = yolo_root / "data.yaml"
    payload = "\n".join(
        [
            f"path: {yolo_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "names:",
            "  0: stain_or_water",
            "  1: wet_surface",
            "",
        ]
    )
    data_yaml.write_text(payload, encoding="utf-8")
    return data_yaml


def find_latest_best_pt(project_dir: Path) -> Path:
    candidates = sorted(
        project_dir.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"YOLO best.pt was not produced under {project_dir}")
    return candidates[0]


def read_yolo_map(run_dir: Path) -> float:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise RuntimeError(f"YOLO results.csv not found at {results_path}")

    rows = list(csv.DictReader(results_path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"YOLO results.csv is empty: {results_path}")

    row = rows[-1]
    normalized = {key.strip(): value for key, value in row.items() if key is not None}
    for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95"):
        raw = normalized.get(key)
        if raw is not None and str(raw).strip():
            return float(raw)

    raise RuntimeError(f"YOLO results.csv is missing mAP columns: {results_path}")


def read_unet_miou(stdout: str) -> float:
    matches = re.findall(r"Best mIoU_12:\s*([0-9.]+)", stdout)
    if not matches:
        raise RuntimeError("U-Net training output did not include 'Best mIoU_12'.")
    return float(matches[-1])


def resolve_device(raw: str) -> str:
    value = raw.strip().lower()
    if value != "auto":
        return raw

    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def storage_connection_string() -> str:
    return (
        os.getenv("RETRAIN_STORAGE_CONNECTION_STRING")
        or os.getenv("MODEL_STORAGE_CONNECTION_STRING")
        or os.getenv("SCORING_RETRAIN_STORAGE_CONNECTION_STRING")
        or ""
    ).strip()


def blob_service(connection_string: str) -> BlobServiceClient:
    if not connection_string:
        raise RuntimeError("Blob connection string is required for production retrain.")
    return BlobServiceClient.from_connection_string(connection_string)


def download_blob_file(service: BlobServiceClient, container_name: str, blob_name: str, destination: Path) -> dict[str, Any]:
    container = service.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        blob.download_blob().readinto(stream)
    props = blob.get_blob_properties()
    return {
        "container": container_name,
        "key": blob_name,
        "local_path": str(destination),
        "etag": str(props.etag),
        "last_modified": props.last_modified.isoformat() if props.last_modified else None,
        "size": props.size,
    }


def inspect_unet_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read active U-Net checkpoint metadata: {checkpoint_path}") from exc

    if not isinstance(ckpt, dict):
        return {}

    metadata: dict[str, Any] = {}
    if ckpt.get("encoder"):
        metadata["encoder"] = str(ckpt["encoder"])
    if ckpt.get("img_size"):
        metadata["img_size"] = int(ckpt["img_size"])
    if "model_state" in ckpt:
        metadata["has_model_state"] = True
    return metadata


def resolve_active_base_models(connection_string: str) -> tuple[Path, Path, dict[str, Any]]:
    use_active = env_bool("RETRAIN_USE_ACTIVE_BASELINE", True)
    require_active = env_bool("RETRAIN_REQUIRE_ACTIVE_BASELINE", True)
    allow_fallback = env_bool("RETRAIN_ALLOW_BASE_FALLBACK", False)
    active_root = resolve_path(env_str("RETRAIN_ACTIVE_BASE_ROOT", "data/retrain_active_base"))
    yolo_active_path = active_root / "yolo_active.pt"
    unet_active_path = active_root / "unet_active.pth"
    metadata: dict[str, Any] = {
        "use_active_baseline": use_active,
        "require_active_baseline": require_active,
        "allow_base_fallback": allow_fallback,
    }

    if not use_active:
        if require_active and not allow_fallback:
            raise RuntimeError("RETRAIN_USE_ACTIVE_BASELINE=false is not allowed while RETRAIN_REQUIRE_ACTIVE_BASELINE=true.")
        metadata["mode"] = "base-fallback-disabled"
        return resolve_path(env_str("RETRAIN_YOLO_MODEL", "yolov8n.pt")), Path(), metadata

    service = blob_service(connection_string)
    models_container = env_str("RETRAIN_ACTIVE_MODELS_CONTAINER", env_str("MODEL_STORAGE_CONTAINER", "models"))
    yolo_key = env_str("RETRAIN_ACTIVE_YOLO_KEY", env_str("MODEL_STORAGE_ACTIVE_YOLO_KEY", "scoring/active/yolo/model.pt")).strip("/")
    unet_key = env_str("RETRAIN_ACTIVE_UNET_KEY", env_str("MODEL_STORAGE_ACTIVE_UNET_KEY", "scoring/active/unet/model.pth")).strip("/")

    recreate_dir(active_root)
    try:
        yolo_info = download_blob_file(service, models_container, yolo_key, yolo_active_path)
        unet_info = download_blob_file(service, models_container, unet_key, unet_active_path)
    except ResourceNotFoundError as exc:
        if allow_fallback and not require_active:
            metadata["mode"] = "base-fallback-missing-active"
            metadata["missing_active_error"] = str(exc)
            return resolve_path(env_str("RETRAIN_YOLO_MODEL", "yolov8n.pt")), Path(), metadata
        raise RuntimeError(
            "Active model is required for production retrain but could not be downloaded "
            f"from container={models_container} yolo={yolo_key} unet={unet_key}."
        ) from exc

    unet_metadata = inspect_unet_checkpoint(unet_active_path)
    metadata.update(
        {
            "mode": "active-finetune",
            "active_models_container": models_container,
            "active_yolo": yolo_info,
            "active_unet": unet_info,
            "active_unet_metadata": unet_metadata,
        }
    )
    print(
        "[ACTIVE] Downloaded active baseline models "
        f"yolo={yolo_key} unet={unet_key}",
        flush=True,
    )
    return yolo_active_path, unet_active_path, metadata


def upload_artifact(container, blob_name: str, file_path: Path, content_type: str) -> None:
    with file_path.open("rb") as stream:
        container.upload_blob(
            name=blob_name,
            data=stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def candidate_artifact_keys(generated_at_utc: str) -> tuple[str, dict[str, dict[str, str]]]:
    container_name = env_str("RETRAIN_CONTAINER", env_str("SCORING_BLOB_RETRAIN_CONTAINER", "retrain"))
    external_prefix = env_str("RETRAIN_EXTERNAL_PREFIX", env_str("SCORING_BLOB_EXTERNAL_PREFIX", "scoring/external/latest")).strip("/")
    archive_prefix = env_str("RETRAIN_CANDIDATE_ARCHIVE_PREFIX", "scoring/external/candidates").strip("/")
    run_id = safe_slug(os.getenv("TRAINER_BATCH_ID") or os.getenv("TRAINER_JOB_ID") or "", "manual")
    timestamp = safe_slug(generated_at_utc.replace(":", "").replace("+", "Z"), "timestamp")
    immutable_prefix = f"{archive_prefix}/{run_id}/{timestamp}".strip("/")
    latest_keys = {
        "candidateYoloKey": f"{external_prefix}/yolo/model.pt",
        "candidateUnetKey": f"{external_prefix}/unet/model.pth",
        "candidateMetricsKey": f"{external_prefix}/metrics/metrics.json",
    }
    immutable_keys = {
        "candidateYoloKey": f"{immutable_prefix}/yolo/model.pt",
        "candidateUnetKey": f"{immutable_prefix}/unet/model.pth",
        "candidateMetricsKey": f"{immutable_prefix}/metrics/metrics.json",
    }
    return container_name, {"latest": latest_keys, "immutable": immutable_keys}


def upload_candidate_artifacts(
    yolo_path: Path,
    unet_path: Path,
    metrics_path: Path,
    artifact_keys: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    connection_string = storage_connection_string()
    if not connection_string:
        raise RuntimeError(
            "RETRAIN_STORAGE_CONNECTION_STRING or MODEL_STORAGE_CONNECTION_STRING is required "
            "to upload retrain candidate artifacts."
        )

    container_name = env_str("RETRAIN_CONTAINER", env_str("SCORING_BLOB_RETRAIN_CONTAINER", "retrain"))

    service = blob_service(connection_string)
    container = service.get_container_client(container_name)
    try:
        container.create_container()
    except ResourceExistsError:
        pass

    for scope, keys in artifact_keys.items():
        upload_artifact(container, keys["candidateYoloKey"], yolo_path, "application/octet-stream")
        upload_artifact(container, keys["candidateUnetKey"], unet_path, "application/octet-stream")
        upload_artifact(container, keys["candidateMetricsKey"], metrics_path, "application/json")
        print(f"[UPLOAD] Candidate artifacts uploaded scope={scope} container={container_name}", flush=True)
    return artifact_keys


def main() -> None:
    connection_string = storage_connection_string()
    if not connection_string:
        raise RuntimeError("Blob connection string is required to build the retrain dataset.")

    dataset_root = resolve_path(env_str("RETRAIN_DATASET_ROOT", "data/retrain_bridge"))
    active_yolo_path, active_unet_path, active_base_metadata = resolve_active_base_models(connection_string)
    yolo_project_dir = resolve_path(env_str("RETRAIN_YOLO_PROJECT_DIR", "outputs/retrain/yolo_training"))
    yolo_run_name = env_str("RETRAIN_YOLO_RUN_NAME", "candidate")
    candidate_yolo_path = resolve_path(env_str("RETRAIN_CANDIDATE_YOLO_FILE", "outputs/retrain/candidate/yolo_best.pt"))
    candidate_unet_path = resolve_path(env_str("RETRAIN_CANDIDATE_UNET_FILE", "outputs/retrain/candidate/unet_best.pth"))
    candidate_metrics_path = resolve_path(env_str("RETRAIN_CANDIDATE_METRICS_FILE", "outputs/retrain/candidate_metrics.json"))

    min_approved = env_int("RETRAIN_MIN_APPROVED_ANNOTATIONS", 5)
    dataset_limit = env_int("RETRAIN_DATASET_LIMIT", 5)
    if dataset_limit > 0 and dataset_limit < min_approved:
        raise RuntimeError("RETRAIN_DATASET_LIMIT must be >= RETRAIN_MIN_APPROVED_ANNOTATIONS for production retrain.")

    recreate_dir(dataset_root)
    recreate_dir(yolo_project_dir / yolo_run_name)
    candidate_yolo_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_unet_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    build_args = [
        sys.executable,
        "scripts/build_retrain_bridge_dataset.py",
        "--connection-string",
        connection_string,
        "--container",
        env_str("RETRAIN_SAMPLES_CONTAINER", "retrain-samples"),
        "--prefix",
        env_str("RETRAIN_SAMPLES_PREFIX", "scoring/retrain-samples"),
        "--output-root",
        str(dataset_root),
        "--train-ratio",
        env_str("RETRAIN_TRAIN_RATIO", "0.8"),
        "--valid-ratio",
        env_str("RETRAIN_VALID_RATIO", "0.1"),
        "--limit",
        str(dataset_limit),
    ]
    only_after_date = env_str("RETRAIN_ONLY_AFTER_DATE", "")
    if only_after_date:
        build_args.extend(["--only-after-date", only_after_date])

    run_command(build_args)
    summary = load_json(dataset_root / "reports" / "bridge_summary.json")
    exported = int(summary.get("exported_annotation_items") or 0)
    errors = int(summary.get("error_items") or 0)
    if exported < min_approved:
        raise RuntimeError(
            f"Not enough approved annotations for retrain. exported={exported}, "
            f"required={min_approved}, errors={errors}."
        )

    ensure_train_valid_splits(dataset_root)
    yolo_data_yaml = write_yolo_data_yaml(dataset_root)

    yolo_device = resolve_device(env_str("RETRAIN_YOLO_DEVICE", "auto"))
    yolo_half = env_bool("RETRAIN_YOLO_HALF", True) and yolo_device != "cpu"

    yolo_env = os.environ.copy()
    yolo_env.update(
        {
            "YOLO_DATA_YAML": str(yolo_data_yaml),
            "YOLO_WEIGHTS_PATH": str(active_yolo_path),
            "YOLO_TRAIN_EPOCHS": str(env_int("RETRAIN_YOLO_EPOCHS", 1)),
            "YOLO_TRAIN_BATCH": str(env_int("RETRAIN_YOLO_BATCH", 1)),
            "YOLO_TRAIN_IMGSZ": str(env_int("RETRAIN_YOLO_IMGSZ", 320)),
            "YOLO_PROJECT_DIR": str(yolo_project_dir),
            "YOLO_RUN_NAME": yolo_run_name,
            "YOLO_DEVICE": yolo_device,
            "YOLO_USE_HALF": str(yolo_half).lower(),
        }
    )
    run_command([sys.executable, "src/train_yolo.py"], env=yolo_env)

    yolo_best = find_latest_best_pt(yolo_project_dir)
    shutil.copy2(yolo_best, candidate_yolo_path)
    yolo_map = read_yolo_map(yolo_best.parents[1])

    unet_proc = run_command(
        [
            sys.executable,
            "src/train_unet.py",
            "--data-root",
            str(dataset_root / "unet"),
            "--epochs",
            str(env_int("RETRAIN_UNET_EPOCHS", 1)),
            "--batch",
            str(env_int("RETRAIN_UNET_BATCH", 1)),
            "--img-size",
            str(int(active_base_metadata.get("active_unet_metadata", {}).get("img_size") or env_int("RETRAIN_UNET_IMGSZ", 256))),
            "--workers",
            str(env_int("RETRAIN_UNET_WORKERS", 0)),
            "--encoder",
            str(active_base_metadata.get("active_unet_metadata", {}).get("encoder") or env_str("RETRAIN_UNET_ENCODER", "resnet18")),
            "--encoder-weights",
            env_str("RETRAIN_UNET_ENCODER_WEIGHTS", "none"),
            "--init-checkpoint",
            str(active_unet_path),
            "--save-path",
            str(candidate_unet_path),
        ]
    )
    if not candidate_unet_path.is_file():
        raise RuntimeError(f"U-Net checkpoint was not produced at {candidate_unet_path}")
    unet_miou = read_unet_miou(unet_proc.stdout or "")

    if not candidate_yolo_path.is_file():
        raise RuntimeError(f"YOLO candidate was not produced at {candidate_yolo_path}")

    generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _, uploaded_keys = candidate_artifact_keys(generated_at_utc)
    metrics = {
        "trainer": {
            "mode": "production-real",
            "generated_at_utc": generated_at_utc,
            "job_id": os.getenv("TRAINER_JOB_ID"),
            "batch_id": os.getenv("TRAINER_BATCH_ID"),
            "sample_count": exported,
            "dataset_root": str(dataset_root),
            "yolo_best_source": str(yolo_best),
            "unet_checkpoint": str(candidate_unet_path),
        },
        "activeBase": active_base_metadata,
        "dataset": summary,
        "yolo": {
            "map": yolo_map,
        },
        "unet": {
            "miou": unet_miou,
        },
    }
    metrics["candidateArtifacts"] = uploaded_keys
    write_json(candidate_metrics_path, metrics)
    upload_candidate_artifacts(candidate_yolo_path, candidate_unet_path, candidate_metrics_path, uploaded_keys)

    print("[DONE] Production retrain pipeline completed")
    print(json.dumps(metrics, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
