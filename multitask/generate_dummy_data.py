"""
マルチタスク分類器の動作確認用ダミー画像生成スクリプト

trayA / trayB / trayC の3種類 × clean/dirty のダミー画像を生成する
本番では実際のトレー画像に差し替えてください

フォルダ構造:
  data/train/trayA_clean/
  data/train/trayA_dirty/
  data/train/trayB_clean/
  ...
"""

import os
import shutil
import random
from PIL import Image, ImageDraw

TRAY_TYPES = {
    "trayA": {"base_color": (220, 210, 190), "shape": "rect"},
    "trayB": {"base_color": (200, 220, 230), "shape": "round"},
    "trayC": {"base_color": (230, 220, 200), "shape": "oval"},
}

SPLIT_COUNTS = {"train": 20, "val": 5}


def make_tray(base_color, shape, size=(256, 256)) -> Image.Image:
    img = Image.new("RGB", size, color=base_color)
    draw = ImageDraw.Draw(img)
    if shape == "rect":
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(120, 120, 120), width=4)
    elif shape == "round":
        draw.ellipse([10, 10, size[0]-10, size[1]-10], outline=(120, 120, 120), width=4)
    elif shape == "oval":
        draw.ellipse([10, 40, size[0]-10, size[1]-40], outline=(120, 120, 120), width=4)
    return img


def add_residue(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(5, 20)):
        x = random.randint(20, img.width - 20)
        y = random.randint(20, img.height - 20)
        r = random.randint(5, 25)
        color = random.choice([(180, 80, 30), (200, 50, 50), (60, 140, 60), (230, 200, 50)])
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    return img


def main():
    if os.path.exists("data"):
        shutil.rmtree("data")
        print("既存の data/ を削除しました")

    for split, count in SPLIT_COUNTS.items():
        for tray_name, cfg in TRAY_TYPES.items():
            for cleanliness in ["clean", "dirty"]:
                folder = f"data/{split}/{tray_name}_{cleanliness}"
                os.makedirs(folder, exist_ok=True)
                for i in range(count):
                    img = make_tray(cfg["base_color"], cfg["shape"])
                    if cleanliness == "dirty":
                        img = add_residue(img)
                    img = img.rotate(random.uniform(-10, 10), expand=False)
                    img.save(f"{folder}/{tray_name}_{cleanliness}_{i:04d}.jpg")
                print(f"生成: {folder} ({count} 枚)")

    print("\nダミーデータ生成完了（本番では実際の画像に差し替えてください）")


if __name__ == "__main__":
    main()
