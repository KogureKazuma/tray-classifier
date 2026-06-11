"""
マルチタスク（トレー種類 × clean/dirty）分類モデル 学習スクリプト

データ構造:
  data/train/<トレー種類>_clean/*.jpg
  data/train/<トレー種類>_dirty/*.jpg
  data/val/<トレー種類>_clean/*.jpg
  data/val/<トレー種類>_dirty/*.jpg

例:
  data/train/trayA_clean/
  data/train/trayA_dirty/
  data/train/trayB_clean/
  data/train/trayB_dirty/
  ...
"""

import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import MultiTaskTrayDataset
from model import MultiTaskTrayClassifier

IMG_SIZE = 224


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


def run_epoch(model, loader, criterion, optimizer, device, train, cleanliness_weight):
    model.train() if train else model.eval()

    total_loss = 0.0
    tray_correct = 0
    clean_correct = 0
    n = 0

    for inputs, tray_labels, clean_labels in loader:
        inputs = inputs.to(device)
        tray_labels = tray_labels.to(device)
        clean_labels = clean_labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            tray_logits, clean_logits = model(inputs)
            loss = criterion(tray_logits, tray_labels) \
                + cleanliness_weight * criterion(clean_logits, clean_labels)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        tray_correct += (tray_logits.argmax(1) == tray_labels).sum().item()
        clean_correct += (clean_logits.argmax(1) == clean_labels).sum().item()
        n += batch_size

    return {
        "loss": total_loss / n,
        "tray_acc": tray_correct / n,
        "clean_acc": clean_correct / n,
    }


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_tray_acc"], label="train")
    axes[1].plot(epochs, history["val_tray_acc"], label="val")
    axes[1].set_title("Tray Type Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["train_clean_acc"], label="train")
    axes[2].plot(epochs, history["val_clean_acc"], label="val")
    axes[2].set_title("Cleanliness Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"学習曲線を保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-task Tray Classifier")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unfreeze_after", type=int, default=10,
                        help="この epoch 以降で全層 fine-tune")
    parser.add_argument("--cleanliness_weight", type=float, default=1.0,
                        help="clean/dirty 損失の重み（清潔判定を重視したい場合は大きくする）")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"使用デバイス: {device}")

    train_tf, val_tf = get_transforms()
    train_ds = MultiTaskTrayDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_ds = MultiTaskTrayDataset(os.path.join(args.data_dir, "val"), val_tf,
                                   tray_type_names=train_ds.tray_type_names)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    print(f"トレー種類({len(train_ds.tray_type_names)}): {train_ds.tray_type_names}")
    print(f"train: {len(train_ds)} 枚 / val: {len(val_ds)} 枚")

    os.makedirs(args.save_dir, exist_ok=True)

    model = MultiTaskTrayClassifier(num_tray_types=len(train_ds.tray_type_names), freeze_base=True).to(device)
    criterion = nn.CrossEntropyLoss()

    head_params = list(model.tray_type_head.parameters()) + list(model.cleanliness_head.parameters())
    optimizer = optim.Adam(head_params, lr=args.lr)
    phase1_epochs = min(args.unfreeze_after, args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_tray_acc": [], "train_clean_acc": [],
               "val_loss": [], "val_tray_acc": [], "val_clean_acc": []}

    for epoch in range(1, args.epochs + 1):
        if epoch == phase1_epochs + 1:
            print("\n=== Phase 2: 全層 fine-tune 開始 ===")
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - phase1_epochs)

        print(f"\nEpoch {epoch}/{args.epochs}  " + "-" * 30)

        train_m = run_epoch(model, dataloaders["train"], criterion, optimizer, device, True, args.cleanliness_weight)
        val_m = run_epoch(model, dataloaders["val"], criterion, optimizer, device, False, args.cleanliness_weight)

        print(f"  train  loss: {train_m['loss']:.4f}  "
              f"tray_acc: {train_m['tray_acc']:.4f}  clean_acc: {train_m['clean_acc']:.4f}")
        print(f"  val    loss: {val_m['loss']:.4f}  "
              f"tray_acc: {val_m['tray_acc']:.4f}  clean_acc: {val_m['clean_acc']:.4f}")

        history["train_loss"].append(train_m["loss"])
        history["train_tray_acc"].append(train_m["tray_acc"])
        history["train_clean_acc"].append(train_m["clean_acc"])
        history["val_loss"].append(val_m["loss"])
        history["val_tray_acc"].append(val_m["tray_acc"])
        history["val_clean_acc"].append(val_m["clean_acc"])

        combined_acc = (val_m["tray_acc"] + val_m["clean_acc"]) / 2
        if combined_acc > best_acc:
            best_acc = combined_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_wts,
                "tray_type_names": train_ds.tray_type_names,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ** best model saved (combined_acc={best_acc:.4f})")

        scheduler.step()

    print(f"\nBest combined val acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    plot_history(history, os.path.join(args.save_dir, "training_curve.png"))


if __name__ == "__main__":
    main()
