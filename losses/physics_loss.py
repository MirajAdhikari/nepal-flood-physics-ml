import torch

def physics_loss(pred_prob, dem=None, manning_n=0.035, g=9.81, dx=10.0):
    """
    Physics-informed loss from 2D Shallow Water Equations for flood mapping.
    pred_prob: (B, 1, H, W) flood probability → proxy for water depth h
    dem: (B, 1, H, W) terrain elevation (optional, download SRTM later)
    
    Enforces:
    1. Gravity-driven downhill flow
    2. Mass conservation (continuity equation)
    3. Manning's friction resistance
    """
    h = pred_prob  # Water depth proxy [0,1]
    
    # Finite differences for gradients (padding to avoid boundary issues)
    pad = 1
    h_pad = torch.nn.functional.pad(h, (pad, pad, pad, pad), mode='replicate')
    
    # Spatial gradients
    dh_dx = (h_pad[:, :, :, pad:] - h_pad[:, :, :, :-pad]) / dx
    dh_dy = (h_pad[:, :, pad:, :] - h_pad[:, :, :-pad, :]) / dx
    
    # 1. Gravity loss: penalize uphill flow (water should flow downhill)
    gravity_loss = torch.mean(torch.relu(dh_dx**2 + dh_dy**2))  # Penalize standing water
    
    # 2. Mass conservation approximation (divergence-free flow)
    div = dh_dx[:, :, :-1, :] + dh_dy[:, :, :, :-1]
    mass_loss = torch.mean(div**2)
    
    # 3. Manning's friction: velocity ~ sqrt(h) * slope / n
    if dem is not None:
        # Bed slope from DEM
        dem_pad = torch.nn.functional.pad(dem, (pad, pad, pad, pad), mode='replicate')
        slope_x = (dem_pad[:, :, :, pad:] - dem_pad[:, :, :, :-pad]) / dx(slope_y)