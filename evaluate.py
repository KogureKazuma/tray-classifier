"""
検証セットで精度・混同行列・ROC曲線を出力
Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np

from train import build_model

IMG_SIZE = 224
CLASS_NAMES = ["clean", "dirty"]

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    model = build_model(freeze_base=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), VAL_TF)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # 分類レポート
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    print("=== Confusion Matrix ===")
    print(f"{'':>10} {'pred:clean':>12} {'pred:dirty':>12}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>10} {cm[i][0]:>12} {cm[i][1]:>12}")

    # AUC-ROC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nAUC-ROC: {auc:.4f}")

    # ROC 曲線プロット
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Tray Residue Classifier")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(args.checkpoint), "roc_curve.png")
    plt.savefig(out_path)
    print(f"ROC 曲線を保存: {out_path}")


if __name__ == "__main__":
    main()
