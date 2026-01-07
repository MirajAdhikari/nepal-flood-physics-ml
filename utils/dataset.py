from pathlib import Path
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class Sen1Floods11Dataset(Dataset):
    def __init__(self, root_dir=r'D:\Datasets\Sen1Floods11\data\flood_events\HandLabeled', transform=None):
        self.root_dir = Path(root_dir)
        self.s1_dir = self.root_dir / 'S1Hand'
        self.label_dir = self.root_dir / 'LabelHand'
        self.transform = transform

        self.s1_files = sorted(list(self.s1_dir.glob('*.tif')))
        self.label_files = sorted(list(self.label_dir.glob('*.tif')))

        print(f"S1Hand folder: {len(self.s1_files)} files found")
        print(f"LabelHand folder: {len(self.label_files)} files found")

        if len(self.s1_files) == 0 or len(self.label_files) == 0:
            raise FileNotFoundError("No .tif files found in S1Hand or LabelHand.")

        # Map label files by base name (remove '_LabelHand')
        name_to_label = {}
        for label_path in self.label_files:
            base_name = label_path.stem.replace('_LabelHand', '')
            name_to_label[base_name] = label_path

        paired = []
        for s1_path in self.s1_files:
            base_name = s1_path.stem.replace('_S1Hand', '')
            label_path = name_to_label.get(base_name)
            if label_path:
                paired.append((s1_path, label_path))
            else:
                print(f"Warning: No matching label for {s1_path.name}")

        self.pairs = paired
        print(f"Successfully paired {len(self.pairs)} samples — ready for training!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_path, label_path = self.pairs[idx]

        with rasterio.open(s1_path) as src:
            s1 = src.read().astype(np.float32)  # (2, 512, 512)

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)  # (512, 512)

        # Clean and normalize SAR data
        s1 = np.nan_to_num(s1, nan=-30.0)
        s1 = np.clip(s1, -30.0, 0.0)
        s1 = (s1 + 30.0) / 30.0  # Scale to [0,1]

        # Clean labels to [0,1]
        label = np.clip(label, 0, 1)
        label = np.nan_to_num(label, nan=0.0)

        input_tensor = torch.from_numpy(s1)  # (2, H, W)
        label_tensor = torch.from_numpy(label).unsqueeze(0)  # (1, H, W)

        if self.transform:
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)

        return input_tensor, label_tensor