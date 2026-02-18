import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
# REPLACE THIS WITH THE PATH TO ONE REAL JSON FILE ON YOUR LAPTOP
TEST_FILE = r"brace/dataset/2014/-OFL9l7_y2g/-OFL9l7_y2g_1559-1666_toprock.json.json" 
# (Or any valid path to a .json file you have)

# COCO -> NTU MAPPING
COCO_TO_NTU = {
    0: 3, 5: 4, 6: 8, 7: 5, 8: 9, 9: 6, 10: 10, 
    11: 12, 12: 16, 13: 13, 14: 17, 15: 14, 16: 18
}

def test_preprocessing(json_path):
    if not os.path.exists(json_path):
        print(f"ERROR: Could not find file {json_path}")
        print("Please edit the TEST_FILE path in the script.")
        return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # Get first frame ID
    frame_ids = sorted(raw_data.keys(), key=lambda x: int(x.split('-')[-1].split('.')[0]))
    first_frame = frame_ids[0]
    
    # 1. LOAD RAW
    kpts = np.array(raw_data[first_frame]['keypoints'])
    box = raw_data[first_frame]['box']
    print(f"Raw Keypoints Shape: {kpts.shape}")
    print(f"Bounding Box: {box}")

    # 2. NORMALIZE TO BOX
    bx, by, bw, bh = box[0], box[1], box[2], box[3]
    kpts[:, 0] = (kpts[:, 0] - bx) / bw
    kpts[:, 1] = (kpts[:, 1] - by) / bh
    
    # 3. MAP TO NTU (25 Joints)
    ntu_frame = np.zeros((25, 3))
    for coco_idx, ntu_idx in COCO_TO_NTU.items():
        ntu_frame[ntu_idx, :] = kpts[coco_idx, :]

    # 4. APPLY HIP CENTERING (The New Math)
    # Hip indices in NTU are 12 and 16
    left_hip = ntu_frame[12, :2]
    right_hip = ntu_frame[16, :2]
    
    # Check if hips exist (are not 0,0)
    if np.sum(left_hip) == 0 or np.sum(right_hip) == 0:
        print("WARNING: Hips are missing in this frame! Math might fail.")
        hip_center = np.array([0.5, 0.5]) # Fallback
    else:
        hip_center = (left_hip + right_hip) / 2.0
    
    print(f"Calculated Hip Center: {hip_center}")

    # Shift all joints
    ntu_centered = ntu_frame.copy()
    # Subtract hip center from X and Y
    # We leave Z (confidence) alone
    ntu_centered[:, 0] -= hip_center[0]
    ntu_centered[:, 1] -= hip_center[1]

    # ================= VISUALIZE =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # PLOT 1: ORIGINAL (Should look like a person in a box)
    ax1.set_title("Step 2: Box Normalized (0 to 1)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0) # Invert Y for image coords
    valid_x = kpts[:, 0]
    valid_y = kpts[:, 1]
    ax1.scatter(valid_x, valid_y, c='blue')
    
    # PLOT 2: CENTERED (Should be centered at 0,0)
    ax2.set_title("Step 4: Hip Centered (Target)")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(1, -1)
    
    # Plot active joints
    active_x = []
    active_y = []
    for idx in COCO_TO_NTU.values():
        if ntu_centered[idx, 0] != -hip_center[0]: # If it's not just the shifted zero
            active_x.append(ntu_centered[idx, 0])
            active_y.append(ntu_centered[idx, 1])
            
    ax2.scatter(active_x, active_y, c='red')
    ax2.scatter([0], [0], c='green', marker='x', label='Origin (0,0)')
    ax2.legend()
    
    plt.show()
    print("Check the popup image. Does the Red Skeleton look like a Human centered at 0,0?")

if __name__ == "__main__":
    test_preprocessing(TEST_FILE)
