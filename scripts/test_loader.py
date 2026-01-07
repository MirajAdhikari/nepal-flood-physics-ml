import sys
import os

# Add the project root to Python path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import Sen1Floods11Dataset
from torch.utils.data import DataLoader

dataset = Sen1Floods11Dataset()

# Create a DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,          # You can change this (2, 4, 8, etc.)
    shuffle=True,
    num_workers=0,         # Set to 2 or 4 if you want faster loading (Windows sometimes needs 0)
    pin_memory=True        # Faster transfer to GPU if available
)

# Test: Load one batch and print info
print("\n--- DataLoader Test ---")
for inputs, labels in loader:
    print(f"Batch input shape : {inputs.shape}")      # Expected: [4, 2, 512, 512] → (batch, channels VV/VH, H, W)
    print(f"Batch label shape : {labels.shape}")      # Expected: [4, 1, 512, 512]
    print(f"Input value range : {inputs.min().item():.2f} to {inputs.max().item():.2f}")
    print(f"Flood pixels in this batch : {labels.sum().item()}")
    print(f"Percentage flooded in batch : {100 * labels.mean().item():.2f}%")
    break  # Only show the first batch