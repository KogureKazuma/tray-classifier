"""
動画 / Webカメラ からトレーを検出し、種類とclean/dirtyを判定して描画するパイプライン

──── 全画面モード（YOLOなし・今すぐ試せる） ────
    python pipeline.py --whole-frame --classifier ../checkpoints/best_model.pth
    python pipeline.py --whole-frame --classifier ../checkpoints/best_model.pth --source 1
        ↑ --source 1 = Continuity Camera (iPhone を Mac に接続した場合)
    python pipeline.py --whole-frame --classifier ../checkpoints/best_model.pth \
        --source "http://192.168.x.x:8080/video"
        ↑ IP Webcam 等のスマホカメラアプリ経由

──── YOLO 検出ありモード（複数トレー対応・学習済み detector が必要） ────
    python pipeline.py --source 0 \
        --detector runs/tray_detector/weights/best.pt \
        --classifier ../checkpoints/best_model.pth
    python pipeline.py --source video.mp4 --output result.mp4
    python pipeline.py --source photo.jpg --image --output out.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import MultiTaskTrayClassifier, CLEANLINESS_NAMES  # noqa: E402

IMG_SIZE = 224
INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    tray_type_names = ckpt["tray_type_names"]
    model = MultiTaskTrayClassifier(num_tray_types=len(tray_type_names), freeze_base=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, tray_type_names


def classify_crop(model, tray_type_names, crop_bgr, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    tensor = INFER_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        tray_logits, clean_logits = model(tensor)
        tray_probs = F.softmax(tray_logits, dim=1)[0]
        clean_probs = F.softmax(clean_logits, dim=1)[0]
    tray_idx = tray_probs.argmax().item()
    clean_idx = clean_probs.argmax().item()
    return (
        tray_type_names[tray_idx], tray_probs[tray_idx].item(),
        CLEANLINESS_NAMES[clean_idx], clean_probs[clean_idx].item(),
    )


# ── 描画ヘルパー ──────────────────────────────────────────────────────────────

def draw_result(frame, box, tray_type, tray_conf, cleanliness, clean_conf):
    """YOLO 検出ボックスに結果を描画（検出ありモード用）"""
    x1, y1, x2, y2 = map(int, box)
    color = (255, 144, 30) if cleanliness == "clean" else (50, 50, 230)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{tray_type} {tray_conf:.0%} | {cleanliness} {clean_conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_result_whole(frame, tray_type, tray_conf, cleanliness, clean_conf):
    """フレーム全体に枠＋大きなラベルを描画（全画面モード用）"""
    h, w = frame.shape[:2]
    color = (255, 144, 30) if cleanliness == "clean" else (50, 50, 230)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
    label = f"{tray_type}  {tray_conf:.0%}  |  {cleanliness}  {clean_conf:.0%}"
    scale, thick = 1.2, 2
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(frame, (0, 0), (tw + 20, th + base + 16), color, -1)
    cv2.putText(frame, label, (10, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


# ── フレーム処理 ──────────────────────────────────────────────────────────────

def process_frame(frame, detector, classifier, tray_type_names, device, conf_thres):
    """YOLO で検出してから分類（複数トレー対応）"""
    results = detector(frame, conf=conf_thres, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        tray_type, tray_conf, cleanliness, clean_conf = classify_crop(
            classifier, tray_type_names, crop, device)
        draw_result(frame, (x1, y1, x2, y2), tray_type, tray_conf, cleanliness, clean_conf)
    return frame


def process_frame_whole(frame, classifier, tray_type_names, device):
    """フレーム中央の正方形を切り出して分類（YOLO なし）"""
    h, w = frame.shape[:2]
    side = min(h, w)
    y1, x1 = (h - side) // 2, (w - side) // 2
    crop = frame[y1:y1 + side, x1:x1 + side]
    tray_type, tray_conf, cleanliness, clean_conf = classify_crop(
        classifier, tray_type_names, crop, device)
    draw_result_whole(frame, tray_type, tray_conf, cleanliness, clean_conf)
    return frame


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0=Mac内蔵カメラ / 1=Continuity Camera(iPhone) / "
                             "URL=IPカメラ / 動画・画像ファイルパス")
    parser.add_argument("--classifier", default="../checkpoints/best_model.pth")
    parser.add_argument("--whole-frame", action="store_true",
                        help="YOLOをスキップして画面全体を分類（今すぐ試せる）")
    parser.add_argument("--detector", default="runs/tray_detector/weights/best.pt",
                        help="YOLOモデルパス（--whole-frame 使用時は不要）")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--image", action="store_true", help="--source を静止画として扱う")
    parser.add_argument("--output", help="保存先パス（省略時は画面に表示）")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"使用デバイス: {device}")

    classifier, tray_type_names = load_classifier(args.classifier, device)
    print(f"トレー種類: {tray_type_names}")

    # YOLO は --whole-frame のときはロードしない
    detector = None
    if not args.whole_frame:
        from ultralytics import YOLO
        detector = YOLO(args.detector)

    # ── 静止画モード ──
    if args.image:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"画像を読み込めませんでした: {args.source}")
            return
        if args.whole_frame:
            process_frame_whole(frame, classifier, tray_type_names, device)
        else:
            process_frame(frame, detector, classifier, tray_type_names, device, args.conf)
        out_path = args.output or "result.jpg"
        cv2.imwrite(out_path, frame)
        print(f"結果を保存しました: {out_path}")
        return

    # ── 映像モード ──
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"映像ソースを開けませんでした: {source}")
        return

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    mode = "全画面モード" if args.whole_frame else "YOLO検出モード"
    print(f"{mode} | {'q で終了' if writer is None else f'出力先: {args.output}'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.whole_frame:
            process_frame_whole(frame, classifier, tray_type_names, device)
        else:
            process_frame(frame, detector, classifier, tray_type_names, device, args.conf)

        if writer:
            writer.write(frame)
        else:
            cv2.imshow("Tray Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
