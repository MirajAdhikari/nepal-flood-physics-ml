import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils.dataset import Sen1Floods11Dataset
from models.fno_segmentation import FNOSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = Sen1Floods11Dataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

baseline = FNOSegmentation().to(device)
baseline_state = torch.load('models/best_fno_baseline.pth', map_location=device, weights_only=False)
baseline_state.pop('_metadata', None)
baseline.load_state_dict(baseline_state, strict=False)
baseline.eval()

physics = FNOSegmentation().to(device)
physics_state = torch.load('models/best_fno_physics.pth', map_location=device, weights_only=False)
physics_state.pop('_metadata', None)
physics.load_state_dict(physics_state, strict=False)
physics.eval()

print("Both models loaded — comparing 5 validation examples...\n")

fig, axes = plt.subplots(5, 4, figsize=(16, 20))
fig.suptitle("Baseline vs Physics-Informed FNO Flood Segmentation", fontsize=20)

with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= 5:
            break
        
        inputs = inputs.to(device)
        logits_base = baseline(inputs)
        logits_phys = physics(inputs)
        pred_base = torch.sigmoid(logits_base)
        pred_phys = torch.sigmoid(logits_phys)

        input_vv = inputs[0, 0].cpu().numpy()
        gt = labels[0, 0].cpu().numpy()
        base = pred_base[0, 0].cpu().numpy()
        phys = pred_phys[0, 0].cpu().numpy()

        axes[i, 0].imshow(input_vv, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Sentinel-1 VV")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        im = axes[i, 2].imshow(base, cmap='Blues', vmin=0, vmax=1)
        axes[i, 2].set_title("Baseline FNO")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(phys, cmap='Blues', vmin=0, vmax=1)
        axes[i, 3].set_title("Physics-Informed FNO")
        axes[i, 3].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Flood Probability')
plt.show()