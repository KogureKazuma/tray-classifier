"""
トレー検出モデル (YOLO) の動作確認用ダミーデータ生成スクリプト

画像内にランダムな矩形（トレーに見立てる）を1〜3個描画し、
YOLO形式のラベルファイル（class x_center y_center width height, 0-1正規化）を生成する

本番では実際の映像から切り出したフレーム + バウンディングボックスアノテーション
（LabelImg / Roboflow など）に差し替えてください
"""

import os
import random
import shutil
from PIL import Image, ImageDraw

IMG_SIZE = 640
SPLIT_COUNTS = {"train": 30, "val": 8}


def make_image_and_labels(img_size: int = IMG_SIZE):
    img = Image.new("RGB", (img_size, img_size), color=(60, 60, 60))
    draw = ImageDraw.Draw(img)
    labels = []

    for _ in range(random.randint(1, 3)):
        w = random.randint(100, 250)
        h = random.randint(100, 250)
        x1 = random.randint(0, img_size - w)
        y1 = random.randint(0, img_size - h)
        x2, y2 = x1 + w, y1 + h
        color = tuple(random.randint(150, 230) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        xc = (x1 + x2) / 2 / img_size
        yc = (y1 + y2) / 2 / img_size
        nw = w / img_size
        nh = h / img_size
        labels.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    return img, labels


def main():
    if os.path.exists("data"):
        shutil.rmtree("data")
        print("既存の data/ を削除しました")

    for split, count in SPLIT_COUNTS.items():
        img_dir = f"data/images/{split}"
        lbl_dir = f"data/labels/{split}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(count):
            img, labels = make_image_and_labels()
            img.save(f"{img_dir}/img_{i:04d}.jpg")
            with open(f"{lbl_dir}/img_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))
        print(f"生成: {img_dir} / {lbl_dir} ({count} 枚)")

    print("\nダミーデータ生成完了（本番では実際の映像フレーム+アノテーションに差し替えてください）")


if __name__ == "__main__":
    main()
