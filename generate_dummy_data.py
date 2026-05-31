"""
動作確認用ダミー画像を生成するスクリプト
本番では実際のトレー画像に差し替えてください
"""

import os
import shutil
import random
from PIL import Image, ImageDraw

DIRS = {
    "data/train/clean": 80,
    "data/train/dirty": 80,
    "data/val/clean":   20,
    "data/val/dirty":   20,
}


def make_clean_tray(size=(256, 256)) -> Image.Image:
    img = Image.new("RGB", size, color=(220, 210, 190))
    draw = ImageDraw.Draw(img)
    # トレーの枠
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(160, 140, 120), width=4)
    return img


def make_dirty_tray(size=(256, 256)) -> Image.Image:
    img = make_clean_tray(size)
    draw = ImageDraw.Draw(img)
    # ランダムな食品残渣 (茶色・赤・緑の斑点)
    for _ in range(random.randint(5, 20)):
        x = random.randint(20, size[0]-20)
        y = random.randint(20, size[1]-20)
        r = random.randint(5, 25)
        color = random.choice([
            (180, 80, 30),   # 茶色
            (200, 50, 50),   # 赤
            (60, 140, 60),   # 緑
            (230, 200, 50),  # 黄
        ])
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    return img


def main():
    # 既存の data/ を丸ごと削除してからクリーンに作り直す
    if os.path.exists("data"):
        shutil.rmtree("data")
        print("既存の data/ を削除しました")

    for path, count in DIRS.items():
        os.makedirs(path, exist_ok=True)
        label = "dirty" if "dirty" in path else "clean"
        for i in range(count):
            img = make_dirty_tray() if label == "dirty" else make_clean_tray()
            # 少しノイズを加えてバリエーションを持たせる
            img = img.rotate(random.uniform(-10, 10), expand=False)
            img.save(os.path.join(path, f"{label}_{i:04d}.jpg"))
        print(f"生成: {path} ({count} 枚)")
    print("ダミーデータ生成完了")


if __name__ == "__main__":
    main()
