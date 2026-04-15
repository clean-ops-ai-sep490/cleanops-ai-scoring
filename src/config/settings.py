from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file() -> None:
    raw_env_path = os.getenv("ENV_FILE", str(DEFAULT_ENV_FILE))
    env_path = Path(raw_env_path)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path

    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        # Keep real OS env priority over .env file values.
        os.environ.setdefault(key, value)


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_path(raw: Optional[str], fallback: Optional[Path] = None) -> Optional[Path]:
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return fallback


def _find_latest_best_pt(base_dir: Path) -> Optional[str]:
    if not base_dir.exists():
        return None

    best_files = sorted(
        base_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(best_files[0]) if best_files else None


def _find_latest_run_poc_best_pt(project_dir: Path) -> Optional[str]:
    if not project_dir.exists() or not project_dir.is_dir():
        return None

    candidates = []
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue

        match = re.fullmatch(r"run_poc_(\d+)", run_dir.name)
        if not match:
            continue

        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            continue

        run_index = int(match.group(1))
        candidates.append((run_index, run_dir.stat().st_mtime, best_pt))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return str(candidates[0][2])


def _build_env_rules() -> Dict[str, Dict[str, object]]:
    env_rules = {
        "LOBBY_CORRIDOR": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_LOBBY_CORRIDOR", 90.0),
            "label": "Sanh / Hanh lang / Thang may",
        },
        "RESTROOM": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_RESTROOM", 85.0),
            "label": "Nha ve sinh",
        },
        "BASEMENT_PARKING": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_BASEMENT_PARKING", 80.0),
            "label": "Tang ham de xe",
        },
        "GLASS_SURFACE": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_GLASS_SURFACE", 90.0),
            "label": "Be mat kinh",
        },
        "OUTDOOR_LANDSCAPE": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_OUTDOOR_LANDSCAPE", 80.0),
            "label": "Ngoai canh va san vuon",
        },
        "HOSPITAL_OR": {
            "pass_threshold": _as_float("ENV_PASS_THRESHOLD_HOSPITAL_OR", 95.0),
            "label": "Phong mo / Y te",
        },
    }

    raw_json = os.getenv("ENV_RULES_JSON")
    if not raw_json:
        return env_rules

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return env_rules

    if not isinstance(parsed, dict):
        return env_rules

    for env_name, env_cfg in parsed.items():
        if env_name not in env_rules or not isinstance(env_cfg, dict):
            continue

        if "pass_threshold" in env_cfg:
            try:
                env_rules[env_name]["pass_threshold"] = float(env_cfg["pass_threshold"])
            except (TypeError, ValueError):
                pass

        if "label" in env_cfg and isinstance(env_cfg["label"], str):
            env_rules[env_name]["label"] = env_cfg["label"]

    return env_rules


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app_host: str
    app_port: int
    app_reload: bool
    request_timeout_sec: int
    app_public_base_url: str

    base_output_dir: Path
    model_cache_dir: Path
    model_path: Optional[str]
    unet_model_path: str
    unet_img_size: int
    model_storage_enabled: bool
    model_storage_connection_string: str
    model_storage_container: str
    model_storage_active_yolo_key: str
    model_storage_active_unet_key: str
    model_force_refresh: bool
    model_require_blob: bool

    yolo_conf: float
    max_batch_images: int
    pending_lower_bound: float
    visualize_jpeg_quality: int
    visualize_temp_url_ttl_sec: int
    visualize_temp_max_items: int

    kaggle_dataset: str
    roboflow_api_key: Optional[str]
    roboflow_workspace: str
    roboflow_project: str
    roboflow_version: int

    yolo_raw_dir: Path
    yolo_data_yaml: Path
    yolo_weights_path: str
    yolo_train_epochs: int
    yolo_train_batch: int
    yolo_train_imgsz: int
    yolo_project_dir: str
    yolo_run_name: str
    yolo_device: str
    yolo_use_half: bool

    hd10k_root: Path
    stagnant_root: Path
    unet_processed_root: Path
    random_seed: int


_load_env_file()


def _build_settings() -> Settings:
    base_output_dir = _resolve_path(os.getenv("BASE_OUTPUT_DIR"), PROJECT_ROOT / "outputs")
    assert base_output_dir is not None

    model_cache_dir = _resolve_path(os.getenv("MODEL_CACHE_DIR"), PROJECT_ROOT / "model-cache")
    assert model_cache_dir is not None

    model_path_raw = os.getenv("MODEL_PATH")
    resolved_model_path = _resolve_path(model_path_raw)

    model_path: Optional[str] = None
    if resolved_model_path and resolved_model_path.is_file():
        model_path = str(resolved_model_path)
    else:
        preferred_yolo_project_dir = _resolve_path(
            os.getenv("YOLO_TRAIN_PROJECT_DIR"), PROJECT_ROOT / "outputs" / "yolo_training_4_4"
        )
        legacy_yolo_project_dir = _resolve_path(os.getenv("YOLO_PROJECT_DIR"), None)

        search_roots = []
        for candidate in [preferred_yolo_project_dir, legacy_yolo_project_dir, base_output_dir]:
            if candidate is None:
                continue
            if candidate in search_roots:
                continue
            search_roots.append(candidate)

        for root in search_roots:
            model_path = _find_latest_run_poc_best_pt(root)
            if model_path:
                break

        if not model_path:
            model_path = _find_latest_best_pt(base_output_dir)

    if not model_path:
        model_path = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")

    unet_model_path = _resolve_path(
        os.getenv("UNET_MODEL_PATH"), PROJECT_ROOT / "models" / "unet_multiclass_best.pth"
    )
    assert unet_model_path is not None

    yolo_raw_dir = _resolve_path(os.getenv("YOLO_RAW_DIR"), PROJECT_ROOT / "data" / "raw" / "yolo")
    assert yolo_raw_dir is not None

    yolo_data_yaml = _resolve_path(
        os.getenv("YOLO_DATA_YAML"), yolo_raw_dir / "roboflow_clean" / "data.yaml"
    )
    assert yolo_data_yaml is not None

    hd10k_root = _resolve_path(
        os.getenv("HD10K_ROOT"), PROJECT_ROOT / "unet_dataset" / "HD10K_IROS2022"
    )
    assert hd10k_root is not None

    stagnant_root = _resolve_path(
        os.getenv("STAGNANT_ROOT"),
        PROJECT_ROOT
        / "unet_dataset"
        / "Stagnant Water and Wet Surface Dataset-fJZUTl"
        / "Stagnant Water and Wet Surface Dataset",
    )
    assert stagnant_root is not None

    unet_processed_root = _resolve_path(
        os.getenv("UNET_PROCESSED_ROOT"), PROJECT_ROOT / "data" / "processed" / "unet_multiclass"
    )
    assert unet_processed_root is not None

    return Settings(
        project_root=PROJECT_ROOT,
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=_as_int("APP_PORT", 8000),
        app_reload=_as_bool("APP_RELOAD", True),
        request_timeout_sec=_as_int("REQUEST_TIMEOUT_SEC", 30),
        app_public_base_url=os.getenv("APP_PUBLIC_BASE_URL", "").strip(),
        base_output_dir=base_output_dir,
        model_cache_dir=model_cache_dir,
        model_path=model_path,
        unet_model_path=str(unet_model_path),
        unet_img_size=_as_int("UNET_IMG_SIZE", 384),
        model_storage_enabled=_as_bool("MODEL_STORAGE_ENABLED", False),
        model_storage_connection_string=os.getenv("MODEL_STORAGE_CONNECTION_STRING", ""),
        model_storage_container=os.getenv("MODEL_STORAGE_CONTAINER", "models"),
        model_storage_active_yolo_key=os.getenv("MODEL_STORAGE_ACTIVE_YOLO_KEY", "scoring/active/yolo/model.pt"),
        model_storage_active_unet_key=os.getenv("MODEL_STORAGE_ACTIVE_UNET_KEY", "scoring/active/unet/model.pth"),
        model_force_refresh=_as_bool("MODEL_FORCE_REFRESH", False),
        model_require_blob=_as_bool("MODEL_REQUIRE_BLOB", False),
        yolo_conf=_as_float("YOLO_CONF", 0.25),
        max_batch_images=_as_int("MAX_BATCH_IMAGES", 5),
        pending_lower_bound=_as_float("PENDING_LOWER_BOUND", 50.0),
        visualize_jpeg_quality=_as_int("VISUALIZE_JPEG_QUALITY", 92),
        visualize_temp_url_ttl_sec=_as_int("VISUALIZE_TEMP_URL_TTL_SEC", 900),
        visualize_temp_max_items=_as_int("VISUALIZE_TEMP_MAX_ITEMS", 200),
        kaggle_dataset=os.getenv("KAGGLE_DATASET", "alyyan/trash-detection"),
        roboflow_api_key=os.getenv("ROBOFLOW_API_KEY"),
        roboflow_workspace=os.getenv("ROBOFLOW_WORKSPACE", "compvision-bfglv"),
        roboflow_project=os.getenv("ROBOFLOW_PROJECT", "clean-unclean-floor"),
        roboflow_version=_as_int("ROBOFLOW_VERSION", 1),
        yolo_raw_dir=yolo_raw_dir,
        yolo_data_yaml=yolo_data_yaml,
        yolo_weights_path=os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt"),
        yolo_train_epochs=_as_int("YOLO_TRAIN_EPOCHS", 50),
        yolo_train_batch=_as_int("YOLO_TRAIN_BATCH", 16),
        yolo_train_imgsz=_as_int("YOLO_TRAIN_IMGSZ", 640),
        yolo_project_dir=os.getenv("YOLO_PROJECT_DIR", "outputs/yolo"),
        yolo_run_name=os.getenv("YOLO_RUN_NAME", "trash_detection"),
        yolo_device=os.getenv("YOLO_DEVICE", "0"),
        yolo_use_half=_as_bool("YOLO_USE_HALF", True),
        hd10k_root=hd10k_root,
        stagnant_root=stagnant_root,
        unet_processed_root=unet_processed_root,
        random_seed=_as_int("RANDOM_SEED", 42),
    )


settings = _build_settings()


def get_env_rules() -> Dict[str, Dict[str, object]]:
    return _build_env_rules()
