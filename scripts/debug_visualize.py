import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import os
from generator import SasakiGenerator

# CONFIG
CHECKPOINT = './pretrained_models/generator_checkpoints_physics/sasaki_gen_ep50.pt'
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), 
    (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
    (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

def debug_skeleton():
    print(f"Visualizing {CHECKPOINT}...")
    model = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    label = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE) # Toprock
    intent = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE) # Attack
    
    with torch.no_grad():
        fake = model(z, label, intent)
    
    # Process to (Frames, Joints, 3)
    skel = fake[0].squeeze(-1).permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
    
    # Store line objects
    lines = [ax.plot([], [], linewidth=2)[0] for _ in BONES]
    
    def update(frame):
        current_pose = skel[frame] # (25, 3)
        
        for i, (u, v) in enumerate(BONES):
            x_data = [current_pose[u, 0], current_pose[v, 0]]
            y_data = [current_pose[u, 1], current_pose[v, 1]]
            
            # --- THE DIAGNOSTIC LOGIC ---
            # Calculate length of this bone
            length = np.linalg.norm(current_pose[u] - current_pose[v])
            
            # If bone is unreasonably long (> 0.4), turn RED
            if length > 0.4:
                lines[i].set_color('red')
                lines[i].set_alpha(1.0)
            # If bone is unreasonably short (< 0.01), turn YELLOW
            elif length < 0.01:
                lines[i].set_color('yellow')
            # Otherwise, standard BLUE
            else:
                lines[i].set_color('blue')
                lines[i].set_alpha(0.6)
            
            lines[i].set_data(x_data, y_data)
        
        ax.set_title(f"Frame {frame} | Red = Broken Bone")
        return lines

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    debug_skeleton()