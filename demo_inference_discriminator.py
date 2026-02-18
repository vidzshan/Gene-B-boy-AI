import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.HPI_GCN_OP import Model
# import Pillow here if you want to be explicit, but it works as a backend writer

# ================= CONFIG =================
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
MODEL_PATH = './pretrained_models/brace_final_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSES = {0: 'Toprock', 1: 'Footwork', 2: 'Powermove'}
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),     # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
]

def load_data_and_model():
    # 1. Load Data
    data = np.load(DATA_PATH)
    x_data = data['x_data'] if 'x_data' in data else data['x_train']
    y_data = data['y_data'] if 'y_data' in data else data['y_train']
    if y_data.ndim > 1: y_data = np.argmax(y_data, axis=1)

    # 2. Load Model
    # Note: num_person=1 because we fixed it in training
    model = Model(num_class=120, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    
    return x_data, y_data, model

def visualize_prediction(x_data, y_data, model):
    # Pick random sample
    idx = np.random.randint(0, len(x_data))
    sample = x_data[idx] # (3, 64, 25, 1)
    label = y_data[idx]
    
    # Predict
    tensor = torch.from_numpy(sample).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0, pred_idx].item()
    
    pred_name = CLASSES[pred_idx]
    true_name = CLASSES[label]
    
    print(f"\nSample #{idx}")
    print(f"TRUE LABEL: {true_name}")
    print(f"AI PREDICT: {pred_name} ({confidence*100:.1f}%)")
    
    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(f"Truth: {true_name} | AI: {pred_name} ({confidence*100:.0f}%)")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    
    # Prepare data (Time, Joints, XY)
    # Sample is (3, 64, 25, 1) -> (64, 25, 3)
    motion = sample.squeeze(-1).transpose(1, 2, 0)
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        pose = motion[frame] # (25, 3)
        x = pose[:, 0]
        y =-pose[:, 1]
        
        # Draw Bones
        for b in BONES:
            ax.plot([x[b[0]], x[b[1]]], [y[b[0]], y[b[1]]], 'blue', linewidth=2)
        
        # Draw Joints
        ax.scatter(x, y, c='red', s=20)

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    
    # --- ADDED SINGLE LINE HERE ---
    print(f"Saving animation_sample_{idx}.gif...")
    ani.save(f'animation_sample_{idx}.gif', writer='pillow', fps=30) 
    
    plt.show()

if __name__ == "__main__":
    x, y, m = load_data_and_model()
    while True:
        visualize_prediction(x, y, m)
        if input("Again? (Enter/n): ") == 'n': break
