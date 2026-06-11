"""
検証セットで トレー種類 / clean-dirty それぞれの精度・混同行列を出力
Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from dataset import MultiTaskTrayDataset
from model import MultiTaskTrayClassifier, CLEANLINESS_NAMES

IMG_SIZE = 224

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    tray_type_names = ckpt["tray_type_names"]

    model = MultiTaskTrayClassifier(num_tray_types=len(tray_type_names), freeze_base=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_ds = MultiTaskTrayDataset(os.path.join(args.data_dir, "val"), VAL_TF, tray_type_names=tray_type_names)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    tray_labels, tray_preds = [], []
    clean_labels, clean_preds = [], []

    with torch.no_grad():
        for inputs, t_labels, c_labels in val_loader:
            inputs = inputs.to(device)
            tray_logits, clean_logits = model(inputs)
            tray_preds.extend(tray_logits.argmax(1).cpu().numpy())
            clean_preds.extend(clean_logits.argmax(1).cpu().numpy())
            tray_labels.extend(t_labels.numpy())
            clean_labels.extend(c_labels.numpy())

    print("\n=== トレー種類 Classification Report ===")
    print(classification_report(tray_labels, tray_preds, target_names=tray_type_names, zero_division=0))

    print("=== 清潔度 (clean/dirty) Classification Report ===")
    print(classification_report(clean_labels, clean_preds, target_names=CLEANLINESS_NAMES, zero_division=0))

    cm = confusion_matrix(clean_labels, clean_preds, labels=[0, 1])
    print("=== 清潔度 混同行列 ===")
    print(f"{'':>10} {'pred:clean':>12} {'pred:dirty':>12}")
    for i, name in enumerate(CLEANLINESS_NAMES):
        print(f"{name:>10} {cm[i][0]:>12} {cm[i][1]:>12}")


if __name__ == "__main__":
    main()
