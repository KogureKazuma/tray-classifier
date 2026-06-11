"""
トレー検出モデル (YOLOv8) の学習スクリプト

データ構造 (YOLO形式) は data.yaml を参照。
動作確認には generate_dummy_data.py でダミーデータを作成できます。

Usage:
    python train_detector.py --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="ベースモデル（nが最軽量）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="tray_detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )

    print(f"\n学習完了: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
