import torch
import torch.nn as nn
from neuralop.models import FNO  # Current correct import

class FNOSegmentation(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        
        self.fno = FNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=2,
            out_channels=width,
            factorization=None,
            rank=0.2
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
            # NO Sigmoid here — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        x = self.fno(x)
        x = self.head(x)
        return x  # Raw logits