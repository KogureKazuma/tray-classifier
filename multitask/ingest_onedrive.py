"""
OneDrive からダウンロードした zip を raw_data/ 形式に変換するスクリプト

以下の 3 パターンをすべて自動検出して統一形式に変換する:
  (A) clean/ と dirty/ だけ直下にある  (zip名をトレー種類名として使用)
  (B) <name>/<name>_clean/ と <name>/<name>_dirty|duty/ がある
      ※ _duty は _dirty の typo として自動修正

.jfif は Pillow 経由で .jpg に変換してコピーする。

Usage:
    python ingest_onedrive.py ~/Downloads/OneDrive_1_2026-6-18

完了後:
    python prepare_data.py
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
OUTPUT_DIR = Path("raw_data")


def save_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".jfif":
        dst = dst.with_suffix(".jpg")
    Image.open(src).convert("RGB").save(dst)


def process_zip(zip_path: Path, out_root: Path):
    tray_name = zip_path.stem
    tmp_dir = zip_path.parent / f"_tmp_{tray_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    try:
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if "__MACOSX" in member or member.endswith(".DS_Store"):
                    continue
                z.extract(member, tmp_dir)

        top_dirs = {e.name.lower(): e for e in tmp_dir.iterdir()
                    if e.is_dir() and not e.name.startswith(".")}

        if "clean" in top_dirs or "dirty" in top_dirs:
            # パターン A: clean/ dirty/ が直下
            src_clean = top_dirs.get("clean")
            src_dirty = top_dirs.get("dirty")
        else:
            # パターン B: <name>/ の中に _clean/ _dirty|duty/ がある
            inner = next((e for e in tmp_dir.iterdir()
                          if e.is_dir() and not e.name.startswith(".")), None)
            if inner is None:
                print(f"  ⚠️  {zip_path.name}: フォルダが見つかりません（スキップ）")
                return
            src_clean = inner / f"{inner.name}_clean"
            src_dirty = inner / f"{inner.name}_dirty"
            if not src_dirty.exists():
                src_dirty = inner / f"{inner.name}_duty"  # typo 自動修正

        stats = {}
        for label, src in [("clean", src_clean), ("dirty", src_dirty)]:
            dst_dir = out_root / tray_name / f"{tray_name}_{label}"
            if src is None or not src.exists():
                print(f"  ⚠️  {tray_name}_{label}: フォルダが見つかりません")
                stats[label] = 0
                continue
            imgs = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
            for img in imgs:
                save_image(img, dst_dir / img.name)
            stats[label] = len(imgs)
            print(f"  {tray_name}_{label}: {stats[label]} 枚")

        if stats.get("clean", 0) <= 1:
            print(f"  ⚠️  clean が {stats.get('clean',0)}枚 → val 分割できない可能性があります")

    finally:
        shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onedrive_dir", help="OneDrive フォルダのパス")
    args = parser.parse_args()

    onedrive = Path(args.onedrive_dir)
    if not onedrive.exists():
        print(f"❌ {onedrive} が見つかりません")
        return

    zips = sorted(onedrive.glob("*.zip"))
    if not zips:
        print(f"❌ {onedrive} に zip が見つかりません")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"既存の {OUTPUT_DIR}/ を削除しました")

    print(f"=== OneDrive データを {OUTPUT_DIR}/ に変換 ===\n")
    for zf in zips:
        print(f"[{zf.stem}]")
        process_zip(zf, OUTPUT_DIR)
        print()

    print("完了。次のコマンドで train/val 分割 → data.zip を作成してください:")
    print("  python prepare_data.py")


if __name__ == "__main__":
    main()
