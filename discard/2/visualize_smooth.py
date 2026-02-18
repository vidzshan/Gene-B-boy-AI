import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter # The Magic Smoother
import os

# --- IMPORT YOUR GENERATOR ---
from generator import SasakiGenerator

# ================= CONFIG =================
CHECKPOINT_PATH = './pretrained_models/generator_checkpoints_smooth/sasaki_gen_ep50.pt'
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU BONE STRUCTURE
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),     # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
]

def load_generator():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Model not found at {CHECKPOINT_PATH}")
        exit()
    
    print(f"Loading Model...")
    model = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def smooth_sequence(skeleton_np, window_length=7, polyorder=2):
    """
    Applies temporal smoothing to remove 'Vibration'.
    Input: (Frames, Joints, 3)
    """
    smooth_skeleton = np.zeros_like(skeleton_np)
    
    # Smooth each joint and each coordinate (X, Y, Z) independently
    for j in range(skeleton_np.shape[1]): # Joints
        for c in range(3): # X, Y, Z
            # Savitzky-Golay filter preserves peaks better than Moving Average
            signal = skeleton_np[:, j, c]
            smooth_signal = savgol_filter(signal, window_length, polyorder)
            smooth_skeleton[:, j, c] = smooth_signal
            
    return smooth_skeleton

def generate_and_render(model, move_type="Toprock", intent_type="Attack"):
    print(f"🎬 Action: {move_type} | Intent: {intent_type}")
    
    # 1. Inputs
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    
    label = torch.zeros(1, 3).to(DEVICE)
    if move_type == "Toprock": label[0, 0] = 1.0
    elif move_type == "Footwork": label[0, 1] = 1.0
    elif move_type == "Powermove": label[0, 2] = 1.0
    
    intent = torch.zeros(1, 3).to(DEVICE)
    if intent_type == "Attack": intent[0, 0] = 1.0 # Advance
    elif intent_type == "Retreat": intent[0, 1] = 1.0
    elif intent_type == "Hold": intent[0, 2] = 1.0
    
    # 2. Inference
    with torch.no_grad():
        raw_out = model(z, label, intent)
    
    # 3. Process
    # (1, 3, 64, 25, 1) -> (64, 25, 3)
    skeleton = raw_out[0, :, :, :, 0].permute(1, 2, 0).cpu().numpy()
    
    # 4. SMOOTHING (The Fix)
    final_skeleton = smooth_sequence(skeleton)
    
    return final_skeleton

def animate(skeleton, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=14)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    
    scatter = ax.scatter([], [], c='black', s=20)
    lines = [ax.plot([], [], 'b-')[0] for _ in BONES]

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for line in lines: line.set_data([], [])
        return scatter, *lines

    def update(frame):
        x = skeleton[frame, :, 0]
        y = skeleton[frame, :, 1]
        
        scatter.set_offsets(np.c_[x, y])
        
        for i, (u, v) in enumerate(BONES):
            lines[i].set_data([x[u], x[v]], [y[u], y[v]])
            # Color Logic
            if i < 4: lines[i].set_color('blue') # Spine
            elif i < 12: lines[i].set_color('red') # Arms
            else: lines[i].set_color('green') # Legs

        return scatter, *lines

    ani = animation.FuncAnimation(fig, update, frames=64, init_func=init, blit=True, interval=50)
    
    # Save as GIF (Optional)
    # ani.save(f"{title}.gif", writer='pillow', fps=20)
    plt.show()

if __name__ == "__main__":
    model = load_generator()
    
    # DEMO 1: TOPROCK
    skel = generate_and_render(model, "Toprock", "Attack")
    animate(skel, "Final Result: Smooth Toprock")
