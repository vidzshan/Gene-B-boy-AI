
import numpy as np

DATA_PATH = './data/brace/BRACE_synced_v2.npz'
OUTPUT_PATH = './data/brace/mean_pose.npy'

def calc_mean():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    x_data = data['x_data'] # (N, 3, 64, 25, 1)
    
    # Calculate Mean across Batch (0) and Time (2)
    # Result: (3, 25, 1) -> (3, 25)
    mean_pose = np.mean(x_data, axis=(0, 2, 4))
    
    print(f"Mean Pose Shape: {mean_pose.shape}")
    print(f"Sample Joint 0 (Nose): {mean_pose[:, 0]}")
    
    np.save(OUTPUT_PATH, mean_pose)
    print(f"Saved Mean Pose to {OUTPUT_PATH}")

if __name__ == "__main__":
    calc_mean()
