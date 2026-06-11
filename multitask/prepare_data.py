"""
収集した生データを train / val に分割し、学習用 data/ を構築するスクリプト

入力 (raw_data/ 配下):
  raw_data/
  ├── donburi/
  │   ├── donburi_clean/
  │   │   └── *.jpg, *.png, ...
  │   └── donburi_dirty/
  │       └── *.jpg, *.png, ...
  ├── bento/
  │   ├── bento_clean/
  │   └── bento_dirty/
  └── ...（トレー種類ぶん繰り返す）

出力:
  data/
  ├── train/
  │   ├── donburi_clean/  donburi_clean_001.jpg ...
  │   ├── donburi_dirty/
  │   ├── bento_clean/
  │   ├── bento_dirty/
  │   └── ...
  └── val/
      └── （同様）

  data.zip （Colab アップロード用）
"""

import random
import shutil
import zipfile
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────
SOURCE_DIR = Path("raw_data")
DATA_DIR   = Path("data")
ZIP_NAME   = "data.zip"
VAL_RATIO  = 0.2
SEED       = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────


def split_files(files: list, val_ratio: float, seed: int):
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []  # 1枚しかない場合は全て train へ
    n_val = max(1, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)  # train に最低1枚は残す
    return shuffled[n_val:], shuffled[:n_val]


def copy_with_rename(files: list, dst_dir: Path, prefix: str):
    if not files:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files, start=1):
        dst = dst_dir / f"{prefix}_{i:03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def create_zip(data_dir: Path, zip_name: str):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(data_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(data_dir.parent))
    size_mb = Path(zip_name).stat().st_size / 1024 / 1024
    print(f"\n📦 {zip_name} を作成しました ({size_mb:.1f} MB)")


def main():
    if not SOURCE_DIR.exists():
        print(f"❌ {SOURCE_DIR}/ が見つかりません")
        print(f"   {SOURCE_DIR}/<トレー種類>/<トレー種類>_clean/ と "
              f"_dirty/ の構造で画像を置いてください")
        return

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    print("=" * 50)
    print("  データセット構築")
    print("=" * 50)

    found_any = False
    for tray_dir in sorted(SOURCE_DIR.iterdir()):
        if not tray_dir.is_dir():
            continue
        tray_name = tray_dir.name

        for cleanliness in ["clean", "dirty"]:
            src_dir = tray_dir / f"{tray_name}_{cleanliness}"
            if not src_dir.exists():
                print(f"  ⚠️  見つかりません: {src_dir}")
                continue

            files = [p for p in sorted(src_dir.iterdir())
                     if p.suffix.lower() in IMAGE_EXTS]
            if not files:
                print(f"  ⚠️  画像がありません: {src_dir}")
                continue

            found_any = True
            train_files, val_files = split_files(files, VAL_RATIO, SEED)
            label = f"{tray_name}_{cleanliness}"

            print(f"\n[{label}]  train={len(train_files)} 枚 / val={len(val_files)} 枚")
            copy_with_rename(train_files, DATA_DIR / "train" / label, label)
            copy_with_rename(val_files,   DATA_DIR / "val"   / label, label)

    if not found_any:
        print("\n❌ 画像が1枚も見つかりませんでした")
        return

    # 構成サマリ
    print("\n─── データセット構成 ───")
    for split in ["train", "val"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for folder in sorted(split_dir.iterdir()):
            n = len(list(folder.glob("*")))
            print(f"  data/{split}/{folder.name}: {n} 枚")

    create_zip(DATA_DIR, ZIP_NAME)
    print(f"👉 Google Colab には '{ZIP_NAME}' をアップロードしてください")


if __name__ == "__main__":
    main()
