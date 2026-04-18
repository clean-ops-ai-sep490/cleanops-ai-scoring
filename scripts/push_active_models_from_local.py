from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _find_newest_files(search_root: Path, pattern: str) -> List[Path]:
    if not search_root.exists() or not search_root.is_dir():
        return []
    return [p for p in search_root.glob(pattern) if p.is_file()]


def _pick_newest(paths: Iterable[Path]) -> Path:
    candidates: List[Tuple[float, str, Path]] = []
    for path in paths:
        candidates.append((path.stat().st_mtime, str(path), path))

    if not candidates:
        raise FileNotFoundError("No matching files were found.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _upload_file(container_client, blob_name: str, file_path: Path, content_type: str) -> None:
    with file_path.open("rb") as stream:
        container_client.upload_blob(
            name=blob_name,
            data=stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload newest local YOLO/U-Net models to active blob keys"
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("MODEL_STORAGE_CONTAINER", "models"),
        help="Target blob container",
    )
    parser.add_argument(
        "--yolo-search-root",
        default=os.getenv("YOLO_TRAIN_PROJECT_DIR", "outputs/yolo_training_4_4"),
        help="Directory used to search for YOLO best.pt",
    )
    parser.add_argument(
        "--yolo-pattern",
        default="**/weights/best.pt",
        help="Glob pattern under yolo-search-root",
    )
    parser.add_argument(
        "--yolo-blob-key",
        default=os.getenv("MODEL_STORAGE_ACTIVE_YOLO_KEY", "scoring/active/yolo/model.pt"),
        help="Target blob key for active YOLO model",
    )
    parser.add_argument(
        "--unet-search-root",
        default=str(Path(os.getenv("UNET_MODEL_PATH", "models/unet_multiclass_best.pth")).parent),
        help="Directory used to search for U-Net model files",
    )
    parser.add_argument(
        "--unet-pattern",
        default="**/*.pth",
        help="Glob pattern under unet-search-root",
    )
    parser.add_argument(
        "--unet-blob-key",
        default=os.getenv("MODEL_STORAGE_ACTIVE_UNET_KEY", "scoring/active/unet/model.pth"),
        help="Target blob key for active U-Net model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files and keys without uploading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.connection_string.strip():
        raise ValueError("Azure Blob connection string is required.")

    yolo_root = _resolve_path(args.yolo_search_root)
    unet_root = _resolve_path(args.unet_search_root)

    yolo_candidates = _find_newest_files(yolo_root, args.yolo_pattern)
    unet_candidates = _find_newest_files(unet_root, args.unet_pattern)

    yolo_path = _pick_newest(yolo_candidates)
    unet_path = _pick_newest(unet_candidates)

    print("Selected newest local model files:")
    print(f"  yolo_path={yolo_path}")
    print(f"  yolo_mtime={yolo_path.stat().st_mtime}")
    print(f"  unet_path={unet_path}")
    print(f"  unet_mtime={unet_path.stat().st_mtime}")
    print(f"  container={args.container}")
    print(f"  yolo_blob_key={args.yolo_blob_key}")
    print(f"  unet_blob_key={args.unet_blob_key}")

    if args.dry_run:
        print("Dry run enabled. No files uploaded.")
        return

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    container_client = blob_service.get_container_client(args.container)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    _upload_file(container_client, args.yolo_blob_key, yolo_path, "application/octet-stream")
    _upload_file(container_client, args.unet_blob_key, unet_path, "application/octet-stream")

    print("Uploaded active models:")
    print(f"  yolo={args.yolo_blob_key}")
    print(f"  unet={args.unet_blob_key}")


if __name__ == "__main__":
    main()