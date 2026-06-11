#!/bin/bash
# リアルタイムパイプライン動作確認用デモセットアップ
#
# ダミーデータでYOLO検出器とマルチタスク分類器を短時間学習し、
# pipeline.py をすぐ試せる状態にする（精度は確認用で実用品質ではない）
#
# Usage:
#   bash setup_demo.sh
#
# 完了後の動作確認:
#   python3 pipeline.py --source 0 \
#       --detector runs/tray_detector/weights/best.pt \
#       --classifier ../checkpoints/best_model.pth \
#       --conf 0.3

set -e
cd "$(dirname "$0")"

echo "==> 1/4 トレー検出用ダミーデータを生成"
python3 generate_dummy_data.py

echo "==> 2/4 YOLOv8n を短時間学習（デモ用）"
python3 train_detector.py --epochs 3 --imgsz 320 --project "$(pwd)/runs" --name tray_detector

echo "==> 3/4 トレー分類用ダミーデータを生成"
(cd .. && python3 generate_dummy_data.py)

echo "==> 4/4 マルチタスク分類器を短時間学習（デモ用）"
(cd .. && python3 train.py --epochs 3 --batch_size 16)

cat <<'EOF'

セットアップ完了！以下のコマンドで動作確認できます。

  Webカメラ:
    python3 pipeline.py --source 0 \
        --detector runs/tray_detector/weights/best.pt \
        --classifier ../checkpoints/best_model.pth \
        --conf 0.3

  静止画:
    python3 pipeline.py --source <画像パス> --image \
        --detector runs/tray_detector/weights/best.pt \
        --classifier ../checkpoints/best_model.pth \
        --conf 0.3 --output result.jpg

※ ダミーデータでの学習のため検出・分類精度は低い想定。
   パイプラインの動作確認（検出→切り出し→分類→描画）が目的。
EOF
