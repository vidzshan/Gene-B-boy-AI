import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from model.HPI_GCN_OP import Model
import traceback

# ==========================================
# 1. CONFIGURATION
# ==========================================
weights_path = './pretrained_models/ntu120_pretrained.pt'
data_path = './data/ntu120/NTU120_CSub.npz'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU Action Classes
action_names = {
    0: "Drink water", 1: "Eat meal/snack", 2: "Brushing teeth",
    3: "Brushing hair", 4: "Drop", 5: "Pickup", 6: "Throw",
    7: "Sitting down", 8: "Standing up", 49: "Punch/Slap",
    50: "Kicking", 51: "Pushing", 54: "Hug", 55: "Handshake",
    117: "Random/Noise"
}

bones = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
]

# ==========================================
# 2. MODEL LOADING
# ==========================================
def load_model():
    print(f"Loading Model on {device}...")
    model = Model(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_dict)
    except Exception as e:
        print(f"❌ Error: {e}"); exit()
    model.to(device).eval()
    return model

# ==========================================
# 3. DATA PROCESSING
# ==========================================
def ensure_4d_shape(sample):
    """
    FIXED: Correctly parses the 150 features as (Person -> Joint -> XYZ).
    """
    # If already 4D, return it
    if sample.ndim == 4: 
        return sample

    # If 2D, we need to reshape carefully
    if sample.ndim == 2:
        d1, d2 = sample.shape
        
        # 1. Normalize input to (Time, 150)
        # If shape is (150, Time), transpose it first
        if d1 == 150: 
            sample = sample.T  # Now (T, 150)
        
        # Get Time dimension
        T = sample.shape[0]
        
        # 2. Reshape with the correct semantic order:
        # The 150 numbers are: 2 Persons * 25 Joints * 3 Channels (XYZ)
        # Order: [Person 1 (75)... Person 2 (75)...]
        # Inside Person: [Joint 1 (3)... Joint 2 (3)...]
        # Inside Joint: [X, Y, Z]
        
        # Reshape to: (Time, Person, Joint, Channel)
        sample = sample.reshape(T, 2, 25, 3)
        
        # 3. Transpose to the Model's expected format: (Channel, Time, Joint, Person)
        # (T, M, V, C) -> (C, T, V, M)
        # Permute: (3, 0, 2, 1)
        sample = sample.transpose(3, 0, 2, 1)
        
        return sample

    raise ValueError(f"Shape Error: {sample.shape}")

def resize_sequence(sample, target_frames=64):
    sample = ensure_4d_shape(sample)
    C, T, V, M = sample.shape
    if T == target_frames: return sample
    sample_flat = sample.transpose(0, 2, 3, 1).reshape(-1, T)
    sample_torch = torch.from_numpy(sample_flat).unsqueeze(0).float()
    resized = F.interpolate(sample_torch, size=target_frames, mode='linear', align_corners=False).squeeze(0).numpy()
    return resized.reshape(C, V, M, target_frames).transpose(0, 3, 1, 2)

# ==========================================
# 4. MULTI-VIEW VISUALIZATION (The Fix)
# ==========================================
def visualize_inference(model, npz_data):
    if 'x_test' in npz_data: x_data, y_data = npz_data['x_test'], npz_data['y_test']
    else: x_data, y_data = npz_data['x'], npz_data['y']

    idx = np.random.randint(0, len(x_data))
    raw_sample = x_data[idx]
    
    # Label Handling
    try:
        raw_label = y_data[idx]
        if isinstance(raw_label, np.ndarray): label_int = int(raw_label.item()) if raw_label.size == 1 else int(np.argmax(raw_label))
        else: label_int = int(raw_label)
    except: label_int = 0

    try: processed_sample = resize_sequence(raw_sample, 64)
    except: return

    # Inference
    tensor_input = torch.from_numpy(processed_sample).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(tensor_input)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_name = action_names.get(pred_idx, f"Class {pred_idx}")
    true_name = action_names.get(label_int, f"Class {label_int}")
    print(f"Sample #{idx} | Truth: {true_name} | Pred: {pred_name} ({confidence*100:.1f}%)")

    # --- 3-VIEW SETUP ---
    # Shape: (3, 64, 25, 2) -> Person 0 -> (3, 64, 25) -> Transpose to (64, 25, 3)
    skeleton = processed_sample[:, :, :, 0].transpose(1, 2, 0)
    
    # Calculate Auto-Limits to fix the "Squashed" look
    all_x, all_y, all_z = skeleton[:,:,0], skeleton[:,:,1], skeleton[:,:,2]
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    min_z, max_z = np.min(all_z), np.max(all_z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Truth: {true_name} | Pred: {pred_name}", fontsize=14)

    def setup_ax(ax, title, x_lims, y_lims):
        ax.set_title(title)
        ax.set_xlim(x_lims[0]-0.2, x_lims[1]+0.2)
        ax.set_ylim(y_lims[0]-0.2, y_lims[1]+0.2)
        ax.set_aspect('equal')
        ax.axis('off')

    def update(frame):
        pose = skeleton[frame] # (25, 3) -> [x, y, z]
        
        # Clear axes
        ax1.clear(); ax2.clear(); ax3.clear()
        
        # 1. Front View (X vs Y)
        setup_ax(ax1, "Front View (X vs Y)", (min_x, max_x), (min_y, max_y))
        ax1.scatter(pose[:, 0], pose[:, 1], c='red', s=20)
        for b in bones: ax1.plot([pose[b[0]-1,0], pose[b[1]-1,0]], [pose[b[0]-1,1], pose[b[1]-1,1]], c='blue')

        # 2. Top View (X vs Z) - Often helps if Y is depth
        setup_ax(ax2, "Top View (X vs Z)", (min_x, max_x), (min_z, max_z))
        ax2.scatter(pose[:, 0], pose[:, 2], c='green', s=20)
        for b in bones: ax2.plot([pose[b[0]-1,0], pose[b[1]-1,0]], [pose[b[0]-1,2], pose[b[1]-1,2]], c='green')

        # 3. Side View (Z vs Y)
        setup_ax(ax3, "Side View (Z vs Y)", (min_z, max_z), (min_y, max_y))
        ax3.scatter(pose[:, 2], pose[:, 1], c='purple', s=20)
        for b in bones: ax3.plot([pose[b[0]-1,2], pose[b[1]-1,2]], [pose[b[0]-1,1], pose[b[1]-1,1]], c='purple')

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    plt.show()

if __name__ == "__main__":
    model = load_model()
    try: npz_data = np.load(data_path)
    except: exit()
    while True:
        try:
            visualize_inference(model, npz_data)
            if input("Next? (Enter=Yes, n=No): ").lower() == 'n': break
        except: traceback.print_exc(); break
