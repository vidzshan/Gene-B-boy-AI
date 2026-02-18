
import torch

def slerp(low, high, val):
    """
    Spherical linear interpolation between two tensors.
    low: tensor of shape (..., D)
    high: tensor of shape (..., D)
    val: float or tensor of shape (...) between 0 and 1
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    
    omega = torch.acos((low_norm * high_norm).sum(dim=-1).clamp(-1, 1))
    so = torch.sin(omega)
    
    if torch.all(so < 1e-6):
        # Fallback to linear if vectors are collinear
        return (1.0 - val) * low + val * high
        
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + \
          (torch.sin(val * omega) / so).unsqueeze(-1) * high
    return res

def slerp_trajectory(z_start, z_end, steps):
    """Generates a smooth trajectory between two points."""
    vals = torch.linspace(0, 1, steps, device=z_start.device)
    return torch.stack([slerp(z_start, z_end, v) for v in vals])
