import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset import Sen1Floods11Dataset
from models.fno_segmentation import FNOSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = Sen1Floods11Dataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model = FNOSegmentation().to(device)
state_dict = torch.load('models/best_fno_baseline.pth', map_location=device, weights_only=False)
state_dict.pop('_metadata', None)  # Remove extra metadata key
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Model loaded — visualizing 6 validation examples...\n")

fig, axes = plt.subplots(6, 3, figsize=(15, 18))
fig.suptitle("FNO Baseline Flood Segmentation (Val Loss ~0.154)", fontsize=18)

with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= 6:
            break
        
        inputs = inputs.to(device)
        logits = model(inputs)
        pred_prob = torch.sigmoid(logits)

        input_vv = inputs[0, 0].cpu().numpy()
        gt = labels[0, 0].cpu().numpy()
        pred = pred_prob[0, 0].cpu().numpy()

        axes[i, 0].imshow(input_vv, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Sentinel-1 VV (normalized)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Flood")
        axes[i, 1].axis('off')

        im = axes[i, 2].imshow(pred, cmap='Blues', vmin=0, vmax=1)
        axes[i, 2].set_title("FNO Prediction")
        axes[i, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Flood Probability')
plt.show()