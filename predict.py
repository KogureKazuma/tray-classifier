"""
推論スクリプト: 画像 or ディレクトリを渡すと clean/dirty を判定
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

from train import build_model

CLASS_NAMES = ["clean (残渣なし)", "dirty (残渣あり)"]
IMG_SIZE = 224

INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device):
    model = build_model(freeze_base=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single(model, image_path: str, device: torch.device) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = INFER_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return {
        "path": image_path,
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": probs[pred_idx].item(),
        "clean_prob": probs[0].item(),
        "dirty_prob": probs[1].item(),
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
    model = load_model(args.checkpoint, device)

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

    print(f"\n{'ファイル名':<40} {'判定':<25} {'信頼度':>8}")
    print("-" * 80)
    for path in sorted(targets):
        result = predict_single(model, path, device)
        name = os.path.basename(result["path"])
        print(f"{name:<40} {result['prediction']:<25} {result['confidence']:>7.1%}")


if __name__ == "__main__":
    main()
