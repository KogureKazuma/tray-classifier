"""
マルチタスク用データセット

フォルダ名 "<トレー種類>_clean" / "<トレー種類>_dirty" から
トレー種類とclean/dirtyの2ラベルを読み取る

例:
  data/train/trayA_clean/*.jpg
  data/train/trayA_dirty/*.jpg
  data/train/trayB_clean/*.jpg
  data/train/trayB_dirty/*.jpg
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CLEANLINESS_NAMES = ["clean", "dirty"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MultiTaskTrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, tray_type_names=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        parsed = []
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue
            tray_type, cleanliness = self._parse_folder_name(folder.name)
            for img_path in sorted(folder.glob("*")):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    parsed.append((img_path, tray_type, cleanliness))

        if not parsed:
            raise ValueError(f"画像が見つかりません: {self.root_dir}")

        if tray_type_names is None:
            tray_type_names = sorted({t for _, t, _ in parsed})
        self.tray_type_names = tray_type_names
        tray_type_to_idx = {name: i for i, name in enumerate(tray_type_names)}

        self.samples = []
        for img_path, tray_type, cleanliness in parsed:
            if tray_type not in tray_type_to_idx:
                raise ValueError(
                    f"未知のトレー種類です: '{tray_type}' ({img_path})\n"
                    f"既知のトレー種類: {tray_type_names}"
                )
            tray_idx = tray_type_to_idx[tray_type]
            clean_idx = CLEANLINESS_NAMES.index(cleanliness)
            self.samples.append((img_path, tray_idx, clean_idx))

    @staticmethod
    def _parse_folder_name(name: str):
        for suffix in CLEANLINESS_NAMES:
            token = f"_{suffix}"
            if name.endswith(token):
                tray_type = name[: -len(token)]
                if not tray_type:
                    raise ValueError(f"フォルダ名が不正です: '{name}'")
                return tray_type, suffix
        raise ValueError(
            f"フォルダ名は '<トレー種類>_clean' か '<トレー種類>_dirty' にしてください: '{name}'"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tray_idx, clean_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, tray_idx, clean_idx
