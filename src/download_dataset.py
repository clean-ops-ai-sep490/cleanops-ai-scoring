"""Dataset downloader for YOLO/U-Net pipeline with .env based configuration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import settings

DATA_DIR = settings.project_root / "data" / "raw"
YOLO_DIR = settings.yolo_raw_dir
UNET_DIR = DATA_DIR / "unet"


def download_yolo_datasets() -> None:
    print("=" * 60)
    print("[*] TAI DU LIEU YOLO (KAGGLE & ROBOFLOW)...")
    print("=" * 60)

    YOLO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Kaggle dataset
    kaggle_dest = YOLO_DIR / "trash_kaggle"
    if not (kaggle_dest / "train").exists():
        try:
            import kaggle

            print(f" -> Dang tai '{settings.kaggle_dataset}' tu Kaggle...")
            kaggle_dest.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files(
                settings.kaggle_dataset,
                path=str(kaggle_dest),
                unzip=True,
            )
            print(f" [OK] Da tai Kaggle dataset vao {kaggle_dest}")
        except Exception as e:
            print(f" [ERROR] Loi tai Kaggle: {e}")
            print(" [HINT] Hay dat KAGGLE_USERNAME va KAGGLE_KEY trong .env")
    else:
        print(" [OK] Kaggle dataset da ton tai san.")

    # 2. Roboflow dataset
    robo_dest = YOLO_DIR / "roboflow_clean"
    if not robo_dest.exists() or len(list(robo_dest.glob("*"))) == 0:
        if not settings.roboflow_api_key:
            print(" [WARN] Chua co ROBOFLOW_API_KEY trong .env, bo qua buoc tai Roboflow.")
            return

        try:
            from roboflow import Roboflow

            print(
                f" -> Dang tai '{settings.roboflow_project}' tu Roboflow "
                f"(workspace={settings.roboflow_workspace}, version={settings.roboflow_version})..."
            )

            rf = Roboflow(api_key=settings.roboflow_api_key)
            project = rf.workspace(settings.roboflow_workspace).project(settings.roboflow_project)
            project.version(settings.roboflow_version).download("yolov8", location=str(robo_dest))
            print(f" [OK] Da tai Roboflow dataset vao {robo_dest}")
        except Exception as e:
            print(f" [ERROR] Loi tai Roboflow: {e}")
    else:
        print(" [OK] Roboflow dataset da ton tai san.")


def configure_unet_datasets() -> None:
    print("\n" + "=" * 60)
    print("[*] CAU HINH DU LIEU U-NET (THU CONG)")
    print("=" * 60)

    img_dir = UNET_DIR / "images"
    mask_dir = UNET_DIR / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(" [INFO] Da tao thu muc don du lieu U-Net tai:")
    print(f"        Anh goc (jpg/png): {img_dir.resolve()}")
    print(f"        Mask segmentation: {mask_dir.resolve()}")
    print("\n [NOTE] Voi bo du lieu lon, vui long tai thu cong va copy vao 2 thu muc tren.")


def main() -> None:
    download_yolo_datasets()
    configure_unet_datasets()
    print("\n[DONE] Hoan tat phase data preparation.")


if __name__ == "__main__":
    main()
