# import gradio as gr
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from scipy.ndimage import gaussian_filter1d
# from model.generator import MotionGenerator
# import os

# # Try to import diagnostics, but don't crash if missing
# try:
#     from diagnostics import get_motion_quality_report, plot_motion_trail
# except ImportError:
#     print("⚠️ 'diagnostics.py' not found. Metrics will be disabled.")
#     def get_motion_quality_report(s): return {"Jitter": 0, "Bone Error": 0, "Energy": 0}
#     def plot_motion_trail(s, p): return p

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# LATENT_DIM = 128

# # CHECK YOUR PATH HERE
# MODEL_PATH = "pretrained_models/generator_checkpoints_final/generator_clean_ep60.pt"

# # ================= 2D CONFIGURATION =================
# # We only connect the joints that exist in COCO (No empty spine)
# SAFE_EDGES = [
#     (4, 8), (12, 16), (4, 12), (8, 16), # Torso Box
#     (4, 5), (5, 6), (8, 9), (9, 10),    # Arms
#     (12, 13), (13, 14), (16, 17), (17, 18), # Legs
#     (3, 4), (3, 8) # Head
# ]

# # ================= 2D VISUALIZER =================
# def visualize_2d_centered(sequence, save_path):
#     """
#     Plots the skeleton in strict 2D (X and Y only).
#     Assumes data is Hip-Centered.
#     """
#     # Smooth slightly for better visuals
#     sequence = gaussian_filter1d(sequence, sigma=1.0, axis=1)
    
#     data_x = sequence[0, :, :]
#     data_y = sequence[1, :, :] 
#     # Note: In image data, Y is down. To stand up, we might need -Y.
    
#     fig, ax = plt.subplots(figsize=(6, 6))
    
#     def update(frame_idx):
#         ax.clear()
#         # Set fixed 2D limits (Centered at 0,0)
#         ax.set_xlim(-0.8, 0.8)
#         ax.set_ylim(-0.8, 0.8)
#         ax.set_aspect('equal')
        
#         ax.set_title(f"Frame {frame_idx}")
#         ax.grid(True, linestyle='--', alpha=0.3)
        
#         # Coordinates
#         x = data_x[frame_idx]
#         y = -data_y[frame_idx] # Invert Y to make head go UP
        
#         # Draw Bones
#         for i, j in SAFE_EDGES:
#             ax.plot([x[i], x[j]], [y[i], y[j]], c='blue', linewidth=2)
            
#         # Draw Joints (Only active ones)
#         active = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
#         ax.scatter(x[active], y[active], c='red', s=30, zorder=10)

#     ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
#     ani.save(save_path, writer='pillow', fps=20)
#     plt.close(fig)
#     return save_path

# # ================= LOAD MODEL =================
# model = MotionGenerator(latent_dim=LATENT_DIM).to(DEVICE)

# if os.path.exists(MODEL_PATH):
#     if torch.cuda.is_available():
#         model.load_state_dict(torch.load(MODEL_PATH))
#     else:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#     model.eval()
#     print(f"✅ Loaded model: {MODEL_PATH}")
# else:
#     print(f"❌ Error: Model not found at {MODEL_PATH}")

# # ================= GENERATION LOOP =================
# def generate_dance_with_stats(dance_type):
#     if dance_type is None: dance_type = "Toprock"
    
#     label_map = {"Toprock": 0, "Footwork": 1, "Power": 2}
#     target_label = label_map[dance_type]

#     # Generate
#     z = torch.randn(1, LATENT_DIM).to(DEVICE)
#     lbl = torch.tensor([target_label]).to(DEVICE)
    
#     with torch.no_grad():
#         fake_motion = model(z, lbl)

#     # Convert to Numpy
#     motion_np = fake_motion.cpu().numpy()[0] 
#     motion_squeezed = motion_np.squeeze(-1) # (3, 64, 25)

#     # 1. Generate 2D GIF (Using new function)
#     gif_path = "temp_dance_2d.gif"
#     visualize_2d_centered(motion_squeezed, save_path=gif_path)

#     # 2. Generate Trail Image
#     trail_path = "temp_trail.png"
#     # We assume plot_motion_trail handles 2D logic internally
#     plot_motion_trail(motion_squeezed, save_path=trail_path)

#     # 3. Calculate Metrics
#     report = get_motion_quality_report(motion_squeezed)
    
#     # Format Report
#     report_text = (
#         f"--- DIAGNOSTIC REPORT ---\n"
#         f"Jitter Score: {report.get('Jitter (Shakiness)', 0)}\n"
#         f"Bone Error:   {report.get('Bone Error (Stretching)', 0)}\n"
#         f"Energy Score: {report.get('Movement (Energy)', 0)}\n"
#         f"-------------------------\n"
#         f"Interpretation:\n"
#         f"• Jitter > 5.0? -> Too shaky (Increase Smoothness Penalty)\n"
#         f"• Bone Error > 2.0? -> Rubber limbs (Increase Bone Penalty)\n"
#         f"• Energy < 0.5? -> Frozen/Statue (Decrease Coord Penalty)"
#     )

#     return gif_path, trail_path, report_text

# # ================= UI =================
# iface = gr.Interface(
#     fn=generate_dance_with_stats,
#     inputs=[
#         gr.Radio(["Toprock", "Footwork", "Power"], label="Dance Style", value="Toprock"),
#     ],
#     outputs=[
#         gr.Image(label="2D Animation"), 
#         gr.Image(label="Motion Trail (Static)"),
#         gr.Textbox(label="Diagnostics")
#     ],
#     title="Breakin' AI - 2D Diagnostic Mode",
#     description="Generates artifact-free 2D breakdancing sequences."
# )

# iface.launch()


import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
from model.generator import MotionGenerator
import os
import glob
import random

# ================= CONFIGURATION =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 128
CHECKPOINT_DIR = "pretrained_models/generator_checkpoints_final"

# Try to import diagnostics, but use internal fallback if missing
try:
    from diagnostics import get_motion_quality_report, plot_motion_trail
except ImportError:
    print("⚠️ 'diagnostics.py' not found. Using internal fallbacks.")
    def get_motion_quality_report(s): 
        return {"Jitter": 0.0, "Bone Error": 0.0, "Energy": 0.0}
    
    # Internal fallback for trail plotting if file is missing
    def plot_motion_trail(sequence, save_path):
        seq = sequence.transpose(1, 2, 0) # [64, 25, 3]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.axis('off')
        x, y = seq[:, :, 0], -seq[:, :, 1]
        # Plot trails
        for t in range(0, 64, 4):
            ax.scatter(x[t], y[t], c='blue', s=10, alpha=t/64.0)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

# ================= HELPER: AUTO-LOAD MODEL =================
def load_latest_model():
    model = MotionGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    
    # Find all .pt files
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    if not files:
        print(f"❌ Error: No checkpoints found in {CHECKPOINT_DIR}")
        return model, "None"
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    
    print(f"✅ Loading latest model: {latest_file}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(latest_file))
    else:
        model.load_state_dict(torch.load(latest_file, map_location='cpu'))
    
    model.eval()
    return model, latest_file

# Load model immediately
model, loaded_filename = load_latest_model()

# ================= 2D CONFIGURATION =================
SAFE_EDGES = [
    (4, 8), (12, 16), (4, 12), (8, 16), # Torso Box
    (4, 5), (5, 6), (8, 9), (9, 10),    # Arms
    (12, 13), (13, 14), (16, 17), (17, 18), # Legs
    (3, 4), (3, 8) # Head
]

# ================= 2D VISUALIZER =================
def visualize_2d_centered(sequence, save_path):
    data_x = sequence[0, :, :]
    data_y = sequence[1, :, :] 
    
    # Calculate dynamic limits (Research Requirement: Don't cut off limbs)
    # We look at the max reach of the dancer
    max_reach = np.max(np.abs(data_x))
    limit = max(1.2, max_reach + 0.1) # Default 1.2, expand if needed
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        
        ax.set_title(f"Frame {frame_idx}")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        x = data_x[frame_idx]
        y = -data_y[frame_idx] # Invert Y
        
        # Draw Bones
        for i, j in SAFE_EDGES:
            ax.plot([x[i], x[j]], [y[i], y[j]], c='blue', linewidth=2)
            
        # Draw Joints
        active = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        ax.scatter(x[active], y[active], c='red', s=30, zorder=10)

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save(save_path, writer='pillow', fps=20)
    plt.close(fig)
    return save_path

# ================= GENERATION LOOP =================
def generate_dance_with_stats(dance_type, seed, smoothness):
    if dance_type is None: dance_type = "Toprock"
    
    # REPRODUCIBILITY: Set the seed
    if seed == -1:
        seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    
    label_map = {"Toprock": 0, "Footwork": 1, "Power": 2}
    target_label = label_map[dance_type]

    # Generate
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    lbl = torch.tensor([target_label]).to(DEVICE)
    
    with torch.no_grad():
        fake_motion = model(z, lbl)

    # Convert to Numpy
    motion_np = fake_motion.cpu().numpy()[0] 
    motion_squeezed = motion_np.squeeze(-1) # (3, 64, 25)

    # Apply Smoothing (Controlled by User)
    if smoothness > 0:
        motion_squeezed = gaussian_filter1d(motion_squeezed, sigma=smoothness, axis=1)

    # 1. Generate 2D GIF
    gif_path = "temp_dance_2d.gif"
    visualize_2d_centered(motion_squeezed, save_path=gif_path)

    # 2. Generate Trail Image
    trail_path = "temp_trail.png"
    plot_motion_trail(motion_squeezed, save_path=trail_path)

    # 3. Calculate Metrics
    report = get_motion_quality_report(motion_squeezed)
    
    # Format Report
    report_text = (
        f"--- DIAGNOSTIC REPORT ---\n"
        f"Used Seed:    {seed}\n"
        f"Loaded Model: {os.path.basename(loaded_filename)}\n"
        f"-------------------------\n"
        f"Jitter Score: {report.get('Jitter (Shakiness)', 0)}\n"
        f"Bone Error:   {report.get('Bone Error (Stretching)', 0)}\n"
        f"Energy Score: {report.get('Movement (Energy)', 0)}\n"
    )

    return gif_path, trail_path, report_text

# ================= UI =================
iface = gr.Interface(
    fn=generate_dance_with_stats,
    inputs=[
        gr.Radio(["Toprock", "Footwork", "Power"], label="Dance Style", value="Toprock"),
        gr.Number(label="Random Seed (-1 for random)", value=-1),
        gr.Slider(0.0, 3.0, value=1.0, label="Smoothness (Sigma)")
    ],
    outputs=[
        gr.Image(label="2D Animation"), 
        gr.Image(label="Motion Trail"),
        gr.Textbox(label="Research Diagnostics")
    ],
    title="Breakin' AI - Research Dashboard",
    description=f"Generating with model: {os.path.basename(loaded_filename)}. Use Seed to replicate results."
)

iface.launch()
