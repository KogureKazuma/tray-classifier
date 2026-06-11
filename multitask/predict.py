"""
推論スクリプト: 画像 or ディレクトリを渡すと トレー種類 と clean/dirty を判定
Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --dir path/to/folder/
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import MultiTaskTrayClassifier, CLEANLINESS_NAMES

IMG_SIZE = 224

INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    tray_type_names = ckpt["tray_type_names"]
    model = MultiTaskTrayClassifier(num_tray_types=len(tray_type_names), freeze_base=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tray_type_names


def predict_single(model, tray_type_names, image_path: str, device: torch.device) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = INFER_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        tray_logits, clean_logits = model(tensor)
        tray_probs = F.softmax(tray_logits, dim=1)[0]
        clean_probs = F.softmax(clean_logits, dim=1)[0]

    tray_idx = tray_probs.argmax().item()
    clean_idx = clean_probs.argmax().item()
    return {
        "path": image_path,
        "tray_type": tray_type_names[tray_idx],
        "tray_confidence": tray_probs[tray_idx].item(),
        "cleanliness": CLEANLINESS_NAMES[clean_idx],
        "cleanliness_confidence": clean_probs[clean_idx].item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image", help="単一画像パス")
    parser.add_argument("--dir", help="画像ディレクトリパス")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model, tray_type_names = load_model(args.checkpoint, device)

    targets = []
    if args.image:
        targets = [args.image]
    elif args.dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        targets = [
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
    else:
        print("--image または --dir を指定してください")
        return

    print(f"\n{'ファイル名':<30} {'トレー種類':<12} {'清潔度':<18} {'信頼度':>16}")
    print("-" * 80)
    for path in sorted(targets):
        r = predict_single(model, tray_type_names, path, device)
        name = os.path.basename(r["path"])
        clean_label = "clean (残渣なし)" if r["cleanliness"] == "clean" else "dirty (残渣あり)"
        conf = f"tray:{r['tray_confidence']:.0%} clean:{r['cleanliness_confidence']:.0%}"
        print(f"{name:<30} {r['tray_type']:<12} {clean_label:<18} {conf:>16}")


if __name__ == "__main__":
    main()
