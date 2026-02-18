import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

# --- IMPORT YOUR GENERATOR ---
from generator import SasakiGenerator

# ================= CONFIG =================
CHECKPOINT_DIR = './pretrained_models/generator_checkpoints_hybrid/'
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU BONE STRUCTURE (Blue=Center, Green=Left, Red=Right)
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),     # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
]

def load_generator(epoch):
    path = os.path.join(CHECKPOINT_DIR, f"sasaki_gen_ep{epoch}.pt")
    if not os.path.exists(path):
        print(f"❌ Error: Checkpoint not found at {path}")
        return None
    
    print(f"Loading Epoch {epoch}...")
    model = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def generate_dance(model, move_type="Toprock", intent_type="Attack"):
    """
    Generates a single sequence based on specific commands.
    """
    # 1. Inputs
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    
    # Label: 0=Toprock, 1=Footwork, 2=Powermove
    label = torch.zeros(1, 3).to(DEVICE)
    if move_type == "Toprock": label[0, 0] = 1.0
    elif move_type == "Footwork": label[0, 1] = 1.0
    elif move_type == "Powermove": label[0, 2] = 1.0
    
    # Intent: 0=Attack(Advance), 1=Retreat, 2=Hold
    intent = torch.zeros(1, 3).to(DEVICE)
    if intent_type == "Attack": intent[0, 0] = 1.0
    elif intent_type == "Retreat": intent[0, 1] = 1.0
    elif intent_type == "Hold": intent[0, 2] = 1.0

    # 2. Generate
    with torch.no_grad():
        # Output shape: (1, 3, 64, 25, 1)
        fake_motion = model(z, label, intent)
    
    # 3. Convert to Numpy for Plotting
    # Squeeze Person dim -> (3, 64, 25) -> Transpose to (64, 25, 3) for easier indexing
    skeleton = fake_motion[0, :, :, :, 0].cpu().numpy()
    skeleton = np.transpose(skeleton, (1, 2, 0)) # Now (Frames, Joints, Channels XYZ)
    
    return skeleton

def animate_skeleton(skeleton, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=14)
    
    # Setup fixed limits so we can see movement
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    
    scatter = ax.scatter([], [], c='black', s=20)
    lines = [ax.plot([], [], 'b-')[0] for _ in BONES] # Blue lines default

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for line in lines: line.set_data([], [])
        return scatter, *lines

    def update(frame):
        # skeleton[frame] is (25, 3)
        x = skeleton[frame, :, 0]
        y = skeleton[frame, :, 1] # Use Y for 2D view
        
        # Update Joints
        scatter.set_offsets(np.c_[x, y])
        
        # Update Bones
        for i, (u, v) in enumerate(BONES):
            x_pair = [x[u], x[v]]
            y_pair = [y[u], y[v]]
            lines[i].set_data(x_pair, y_pair)
            
            # Color coding (Spine=Blue, Left=Green, Right=Red)
            if i < 4: lines[i].set_color('blue')
            elif i < 8: lines[i].set_color('green')
            elif i < 12: lines[i].set_color('red')
            elif i < 16: lines[i].set_color('green')
            else: lines[i].set_color('red')

        return scatter, *lines

    ani = animation.FuncAnimation(fig, update, frames=64, init_func=init, blit=True, interval=50)
    plt.show()

if __name__ == "__main__":
    # --- INTERFACE ---
    print("--- GENERATOR CHECK ---")
    epoch = input("Enter Epoch number to check (e.g. 10): ")
    
    model = load_generator(epoch)
    
    if model:
        print("Generating 'Toprock' with 'Attack' intent...")
        data = generate_dance(model, move_type="Toprock", intent_type="Attack")
        
        print("Visualizing...")
        animate_skeleton(data, f"Epoch {epoch} | Toprock Attack")
