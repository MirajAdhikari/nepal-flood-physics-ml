import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import Sen1Floods11Dataset
from models.fno_segmentation import FNOSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = Sen1Floods11Dataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

model = FNOSegmentation().to(device)
criterion_data = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

lambda_physics = 0.1

def physics_loss(pred_logits):
    pred_prob = torch.sigmoid(pred_logits)
    h = pred_prob + 1e-6  # [B, C, H, W]

    # Pad
    h_pad = torch.nn.functional.pad(h, (1, 1, 1, 1), mode='replicate')

    # Central differences
    dh_dx = (h_pad[:, :, 1:-1, 2:] - h_pad[:, :, 1:-1, :-2]) / 2.0  # [B,C,H,W]
    dh_dy = (h_pad[:, :, 2:, 1:-1] - h_pad[:, :, :-2, 1:-1]) / 2.0  # [B,C,H,W]

    # Gravity / slope loss
    gravity_loss = torch.mean(dh_dx**2 + dh_dy**2)

    div_x = dh_dx[:, :, :, 1:] - dh_dx[:, :, :, :-1]   # [B,C,H,W-1]
    div_y = dh_dy[:, :, 1:, :] - dh_dy[:, :, :-1, :]   # [B,C,H-1,W]

    # Match shapes explicitly
    H = min(div_x.shape[2], div_y.shape[2])
    W = min(div_x.shape[3], div_y.shape[3])

    div = div_x[:, :, :H, :W] + div_y[:, :, :H, :W]
    mass_loss = torch.mean(div**2)

    # Friction (defined on original grid)
    friction_loss = torch.mean(h ** (4/3))

    return gravity_loss + mass_loss + friction_loss

num_epochs = 30
best_val_loss = float('inf')
save_path = 'models/best_fno_physics.pth'

os.makedirs('models', exist_ok=True)

print("Starting physics-informed training...\n")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_data = 0.0
    train_phys = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss_data = criterion_data(outputs, labels)
        loss_phys = physics_loss(outputs)

        loss = loss_data + lambda_physics * loss_phys

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_data += loss_data.item()
        train_phys += loss_phys.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs)
            val_loss += criterion_data(outputs, labels).item()

    avg_train = train_loss / len(train_loader)
    avg_data = train_data / len(train_loader)
    avg_phys = train_phys / len(train_loader)
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:2d}/{num_epochs}")
    print(f"  Train Total: {avg_train:.6f} | Data: {avg_data:.6f} | Physics: {avg_phys:.6f}")
    print(f"  Val Data: {avg_val:.6f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), save_path)
        print("  → New best model saved!")

print("\nPhysics-informed training complete!")
print(f"Best model saved at: {save_path}")