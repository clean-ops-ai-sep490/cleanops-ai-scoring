from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_schema(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    schema = load_json(path)
    detection = {name: int(idx) for idx, name in schema["detection_classes"].items()}
    segmentation = {name: int(idx) for idx, name in schema["segmentation_classes"].items()}
    return detection, segmentation


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in IMAGE_EXTS
    )


def iter_masks(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in MASK_EXTS
    )


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stem(dataset: str, image_path: Path) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{dataset}_{image_path.stem}_{digest}"


def copy_image(src: Path, dst_dir: Path, stem: str | None = None) -> Path:
    ensure_dir(dst_dir)
    name = f"{stem}{src.suffix.lower()}" if stem else src.name
    dst = dst_dir / name
    shutil.copy2(src, dst)
    return dst


def train_val_split(items: list[Any], val_ratio: float, seed: int) -> tuple[list[Any], list[Any]]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    if not items:
        return [], []
    val_count = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val_items = items[:val_count]
    train_items = items[val_count:]
    return train_items, val_items


def get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        pass

    try:
        with path.open("rb") as f:
            header = f.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker_start = f.read(1)
                    if marker_start != b"\xff":
                        return None
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                        f.read(3)
                        height, width = struct.unpack(">HH", f.read(4))
                        return int(width), int(height)
                    length = struct.unpack(">H", f.read(2))[0]
                    f.seek(length - 2, 1)
    except Exception:
        return None
    return None


def find_nearby_label(image_path: Path, label_roots: Iterable[Path], suffix: str = ".txt") -> Path | None:
    candidates = []
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}{suffix}")
    for root in label_roots:
        if root.exists():
            candidates.append(root / f"{image_path.stem}{suffix}")
            try:
                parts = list(image_path.relative_to(root.parent).parts)
                if "images" in [part.lower() for part in parts]:
                    image_idx = [part.lower() for part in parts].index("images")
                    parts[image_idx] = "labels"
                    candidates.append(root.parent.joinpath(*parts).with_suffix(suffix))
            except Exception:
                pass
            try:
                rel = image_path.relative_to(image_path.parents[1])
                candidates.append(root / rel.with_suffix(suffix))
            except Exception:
                pass
    candidates.append(image_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_simple_yaml_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    names: dict[int, str] = {}
    in_names = False
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                values = [value.strip().strip("'\"") for value in stripped.strip("[]").split(",")]
                for value in values:
                    if value:
                        names[len(names)] = value
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"[]")
                if key.isdigit() and value:
                    names[int(key)] = value
                continue
            if stripped.startswith("- "):
                names[len(names)] = stripped[2:].strip().strip("'\"")
                continue
            if not raw.startswith((" ", "\t")):
                break
    return names
