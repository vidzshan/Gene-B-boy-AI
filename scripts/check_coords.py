import torch
import numpy as np
CHECKPOINT = './pretrained_models/audio_checkpoints/expansion_sasaki_gen_e5.pth'
gen_weights = torch.load(CHECKPOINT)
# Just print a few raw numbers
print("Raw Joint Coordinates (First 5 joints, Frame 0):")
# We can't easily run the generator without full setup, 
# but let's look at the weights' magnitude.
for name, param in gen_weights.items():
    if 'joint_heads.0.weight' in name:
        print(f"Weight magnitude: {param.abs().mean().item()}")