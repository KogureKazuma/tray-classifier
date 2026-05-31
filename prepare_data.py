"""
デスクトップの画像をリネームしてデータセットを構築するスクリプト

実行すると以下をすべて自動でやります:
  1. 画像を clean_001.png / dirty_001.png ... にリネーム
  2. train / val に分割（デフォルト: 8:2）
  3. Colab アップロード用の data.zip を作成
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────
DESKTOP = Path.home() / "Desktop"

# clean 画像のファイル名リスト（タイムスタンプ順）
CLEAN_FILES = [
    "スクリーンショット 2026-05-28 13.47.35.png",
    "スクリーンショット 2026-05-28 13.47.40.png",
    "スクリーンショット 2026-05-28 13.47.45.png",
    "スクリーンショット 2026-05-28 13.47.51.png",
    "スクリーンショット 2026-05-28 13.47.54.png",
    "スクリーンショット 2026-05-28 13.47.59.png",
    "スクリーンショット 2026-05-28 13.48.02.png",
    "スクリーンショット 2026-05-28 13.48.07.png",
    "スクリーンショット 2026-05-28 13.48.10.png",
    "スクリーンショット 2026-05-28 13.48.14.png",
]

# dirty 画像のファイル名リスト
DIRTY_FILES = [
    "スクリーンショット 2026-05-28 13.49.04.png",
    "スクリーンショット 2026-05-28 13.49.07.png",
    "スクリーンショット 2026-05-28 13.49.12.png",
    "スクリーンショット 2026-05-28 13.49.16.png",
    "スクリーンショット 2026-05-28 13.49.19.png",
    "スクリーンショット 2026-05-28 13.49.22.png",
    "スクリーンショット 2026-05-28 13.49.27.png",
    "スクリーンショット 2026-05-28 13.49.32.png",
    "スクリーンショット 2026-05-28 13.49.35.png",
    "スクリーンショット 2026-05-28 13.49.37.png",
]

VAL_RATIO = 0.2   # 検証用に使う割合（0.2 = 20% = 2枚）
SEED      = 42
DATA_DIR  = Path("data")
ZIP_NAME  = "data.zip"
# ─────────────────────────────────────────────────────────────


def split_files(files: list, val_ratio: float, seed: int):
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


def copy_with_rename(src_files: list, src_dir: Path, dst_dir: Path, label: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, fname in enumerate(src_files, start=1):
        src = src_dir / fname
        if not src.exists():
            print(f"  ⚠️  見つかりません: {src}")
            continue
        ext = src.suffix
        dst = dst_dir / f"{label}_{i:03d}{ext}"
        shutil.copy2(src, dst)
        print(f"  {fname}  →  {dst.relative_to(DATA_DIR.parent)}")


def create_zip(data_dir: Path, zip_name: str):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(data_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(data_dir.parent))
    size_mb = Path(zip_name).stat().st_size / 1024 / 1024
    print(f"\n📦 {zip_name} を作成しました ({size_mb:.1f} MB)")


def main():
    # 既存データを削除してクリーン構築
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    print("=" * 50)
    print("  データセット構築")
    print("=" * 50)

    for label, files in [("clean", CLEAN_FILES), ("dirty", DIRTY_FILES)]:
        train_files, val_files = split_files(files, VAL_RATIO, SEED)
        print(f"\n[{label}]  train={len(train_files)} 枚 / val={len(val_files)} 枚")

        print("  train:")
        copy_with_rename(train_files, DESKTOP, DATA_DIR / "train" / label, label)
        print("  val:")
        copy_with_rename(val_files,   DESKTOP, DATA_DIR / "val"   / label, label)

    # ディレクトリ構成を表示
    print("\n─── データセット構成 ───")
    for split in ["train", "val"]:
        for lbl in ["clean", "dirty"]:
            p = DATA_DIR / split / lbl
            n = len(list(p.glob("*")))
            print(f"  data/{split}/{lbl}: {n} 枚")

    # ZIP 作成（Colab アップロード用）
    create_zip(DATA_DIR, ZIP_NAME)
    print(f"👉 Google Colab には '{ZIP_NAME}' をアップロードしてください")


if __name__ == "__main__":
    main()
