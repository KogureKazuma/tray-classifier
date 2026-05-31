"""
トレー食品残渣 分類モデル トレーニングスクリプト
転移学習 (EfficientNet-B0) を使ったバイナリ分類
  - clean: 食品残渣なし
  - dirty: 食品残渣あり
"""

import os
import time
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

import matplotlib.pyplot as plt


# ─── 定数 ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASS_NAMES = ["clean", "dirty"]  # data/train/ 以下のフォルダ名と一致させること


# ─── データ拡張 ──────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─── モデル構築 ──────────────────────────────────────────────────────
def build_model(freeze_base: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # 最終 classifier を差し替え
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


# ─── 学習ループ ──────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    dataloaders: dict,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}  " + "-" * 30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.float() / dataset_size

            print(f"  {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_dir, "best_model.pth"))
                print(f"  ** best model saved (val_acc={best_acc:.4f})")

        if scheduler:
            scheduler.step()

    print(f"\nBest val acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


# ─── 学習曲線プロット ────────────────────────────────────────────────
def plot_history(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"学習曲線を保存: {save_path}")


# ─── メイン ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tray Food Residue Classifier")
    parser.add_argument("--data_dir", default="data", help="データルートディレクトリ")
    parser.add_argument("--save_dir", default="checkpoints", help="モデル保存先")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--unfreeze_after", type=int, default=10,
                        help="この epoch 以降で base を unfreeze して fine-tune")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"使用デバイス: {device}")

    # データセット
    train_tf, val_tf = get_transforms()
    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf),
        "val":   datasets.ImageFolder(os.path.join(args.data_dir, "val"),   val_tf),
    }
    dataloaders = {
        k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"),
                      num_workers=0, pin_memory=False)
        for k, v in image_datasets.items()
    }
    print(f"train: {len(image_datasets['train'])} 枚  "
          f"val: {len(image_datasets['val'])} 枚")
    print(f"クラス: {image_datasets['train'].classes}")

    # Phase 1: base frozen で head のみ学習
    model = build_model(freeze_base=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.unfreeze_after)

    phase1_epochs = min(args.unfreeze_after, args.epochs)
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        device, phase1_epochs, args.save_dir
    )

    # Phase 2: 全体 fine-tune（残りエポック）
    remaining = args.epochs - phase1_epochs
    if remaining > 0:
        print("\n=== Phase 2: Fine-tuning (全層) ===")
        for param in model.parameters():
            param.requires_grad = True
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=remaining)
        model, history2 = train_model(
            model, dataloaders, criterion, optimizer2, scheduler2,
            device, remaining, args.save_dir
        )
        for k in history:
            history[k].extend(history2[k])

    plot_history(history, os.path.join(args.save_dir, "training_curve.png"))


if __name__ == "__main__":
    main()
