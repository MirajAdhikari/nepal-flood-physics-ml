import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import Sen1Floods11Dataset
from models.fno_segmentation import FNOSegmentation

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
dataset = Sen1Floods11Dataset()
print(f"Total samples: {len(dataset)}")

# Train / Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# Model
model = FNOSegmentation(modes=12, width=32).to(device)

# Best stable loss for binary segmentation
criterion = nn.BCEWithLogitsLoss()

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training settings
num_epochs = 30
best_val_loss = float('inf')
save_path = 'models/best_fno_baseline.pth'

os.makedirs('models', exist_ok=True)  # Create folder if not exists

print("\nStarting training...\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()  # Important: float for BCEWithLogitsLoss

        optimizer.zero_grad()
        outputs = model(inputs)  # Raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Step scheduler
    scheduler.step(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"   → New best model saved! (val loss: {best_val_loss:.6f})")

print("\nTraining complete!")
print(f"Best model saved at: {save_path}")
print("Next: Visualize predictions and add physics-informed loss (Shallow Water Equations + Manning's friction).")