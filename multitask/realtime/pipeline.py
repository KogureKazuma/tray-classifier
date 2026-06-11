"""
動画 / Webカメラ からトレーを検出し、種類とclean/dirtyを判定して描画するパイプライン

  1. YOLO でフレーム内のトレー位置を検出
  2. 検出した領域を切り出して MultiTaskTrayClassifier で判定
  3. 結果（トレー種類・clean/dirty・信頼度）を映像に描画

事前準備:
  - トレー検出モデル: train_detector.py で学習した runs/tray_detector/weights/best.pt
  - トレー分類モデル: ../checkpoints/best_model.pth（multitask/train.py で学習）

Usage:
    python pipeline.py --source 0                                  # Webカメラ
    python pipeline.py --source video.mp4 --output result.mp4      # 動画ファイル
    python pipeline.py --source photo.jpg --image --output out.jpg # 静止画
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

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


def draw_result(frame, box, tray_type, tray_conf, cleanliness, clean_conf):
    x1, y1, x2, y2 = map(int, box)
    color = (255, 144, 30) if cleanliness == "clean" else (50, 50, 230)  # BGR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{tray_type} {tray_conf:.0%} | {cleanliness} {clean_conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def process_frame(frame, detector, classifier, tray_type_names, device, conf_thres):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0=Webカメラ / 動画ファイル / 画像ファイル")
    parser.add_argument("--detector", default="runs/tray_detector/weights/best.pt")
    parser.add_argument("--classifier", default="../checkpoints/best_model.pth")
    parser.add_argument("--conf", type=float, default=0.5, help="検出の信頼度しきい値")
    parser.add_argument("--image", action="store_true", help="--source を静止画として扱う")
    parser.add_argument("--output", help="保存先パス（指定しなければ画面に表示）")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"使用デバイス: {device}")

    detector = YOLO(args.detector)
    classifier, tray_type_names = load_classifier(args.classifier, device)
    print(f"トレー種類: {tray_type_names}")

    if args.image:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"画像を読み込めませんでした: {args.source}")
            return
        process_frame(frame, detector, classifier, tray_type_names, device, args.conf)
        out_path = args.output or "result.jpg"
        cv2.imwrite(out_path, frame)
        print(f"結果を保存しました: {out_path}")
        return

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

    print("'q' キーで終了" if writer is None else f"出力先: {args.output}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
