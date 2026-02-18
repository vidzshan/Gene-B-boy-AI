# import os
# import json
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm

# # ==========================================
# # CONFIGURATION
# # ==========================================
# # PATH TO YOUR BRACE DATASET FOLDER (The one containing 2011, 2013, etc.)
# BRACE_ROOT = r"brace\dataset"
# OUTPUT_PATH = "data/brace/BRACE_processed.npz"

# TARGET_FRAMES = 64
# NUM_NTU_JOINTS = 25
# NUM_COCO_JOINTS = 17

# # Classes: 0=toprock, 1=footwork, 2=powermove
# CLASS_MAP = {'toprock': 0, 'footwork': 1, 'powermove': 2}

# # MAPPING DICTIONARY (COCO Index -> NTU Index)
# # Unmapped NTU joints (0,1,2,7,11,15,19,20,21,22,23,24) will remain 0
# COCO_TO_NTU = {
#     0: 3,   # Nose -> Head
#     5: 4,   # L Shoulder -> L Shoulder
#     6: 8,   # R Shoulder -> R Shoulder
#     7: 5,   # L Elbow -> L Elbow
#     8: 9,   # R Elbow -> R Elbow
#     9: 6,   # L Wrist -> L Wrist
#     10: 10, # R Wrist -> R Wrist
#     11: 12, # L Hip -> L Hip
#     12: 16, # R Hip -> R Hip
#     13: 13, # L Knee -> L Knee
#     14: 17, # R Knee -> R Knee
#     15: 14, # L Ankle -> L Ankle
#     16: 18  # R Ankle -> R Ankle
# }

# def resize_sequence(data, target_frames):
#     """
#     Resizes data (T, C, V) -> (Target, C, V)
#     """
#     # Convert to tensor: (1, C*V, T) for interpolation
#     T, V, C = data.shape
#     data_torch = torch.tensor(data, dtype=torch.float32)

#     # Reshape to (Batch, Channels, Time)
#     # We treat Joints*Channels as the "Channel" dimension
#     data_torch = data_torch.view(T, V*C).permute(1, 0).unsqueeze(0)

#     # Interpolate
#     data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)

#     # Reshape back to (Target, V, C)
#     data_resized = data_resized.squeeze(0).permute(1, 0).view(target_frames, V, C)
#     return data_resized.numpy()

# def process_brace_file(json_path):
#     with open(json_path, 'r') as f:
#         raw_data = json.load(f)

#     # raw_data is a dict: {"frame_id": {"keypoints": [...], "box": [...]}}
#     # Sort keys to ensure temporal order
#     frame_ids = sorted(raw_data.keys(), key=lambda x: int(x.split('-')[-1].split('.')[0]))

#     if len(frame_ids) < 10:
#         return None # Skip very short clips

#     frames_coco = []

#     for fid in frame_ids:
#         # Extract keypoints (17, 3) -> [x, y, score]
#         kpts = np.array(raw_data[fid]['keypoints'])

#         # Normalize using the bounding box (Crucial for GCN)
#         # box: [x, y, w, h]
#         box = raw_data[fid]['box']
#         bx, by, bw, bh = box[0], box[1], box[2], box[3]

#         # Normalize to [-1, 1] range relative to box center
#         kpts[:, 0] = (kpts[:, 0] - bx) / bw
#         kpts[:, 1] = (kpts[:, 1] - by) / bh

#         # Use score as the 3rd channel (Confidence)
#         # If format is just [x,y], add a dummy 1.0 for confidence
#         if kpts.shape[1] == 2:
#             kpts = np.column_stack((kpts, np.ones(17)))

#         frames_coco.append(kpts)

#     frames_coco = np.array(frames_coco) # Shape (T, 17, 3)

#     # 1. Resize Temporal Dimension
#     frames_coco = resize_sequence(frames_coco, TARGET_FRAMES) # Shape (64, 17, 3)

#     # 2. Map to NTU Topology (Zero Padding)
#     frames_ntu = np.zeros((TARGET_FRAMES, NUM_NTU_JOINTS, 3))

#     for coco_idx, ntu_idx in COCO_TO_NTU.items():
#         frames_ntu[:, ntu_idx, :] = frames_coco[:, coco_idx, :]

#     return frames_ntu

# def main():
#     all_data = []
#     all_labels = []
#     sample_names = []

#     print(f"Scanning {BRACE_ROOT}...")

#     # Recursive walk through year/video folders
#     files_found = 0
#     for root, dirs, files in os.walk(BRACE_ROOT):
#         for file in files:
#             if file.endswith(".json"):
#                 files_found += 1
#                 # Parse Label from Filename
#                 # Format: {video}_{start}-{end}_{label}.json.json (sometimes double json extension in brace)
#                 try:
#                     # Split by underscore, take last part, remove extension
#                     # Example: ..._powermove.json.json -> powermove
#                     label_str = file.split('_')[-1].split('.')[0]

#                     if label_str not in CLASS_MAP:
#                         continue # Skip if label is weird

#                     label_id = CLASS_MAP[label_str]

#                     full_path = os.path.join(root, file)
#                     processed_skeleton = process_brace_file(full_path)

#                     if processed_skeleton is not None:
#                         all_data.append(processed_skeleton)
#                         all_labels.append(label_id)
#                         sample_names.append(file)

#                 except Exception as e:
#                     print(f"Error parsing {file}: {e}")

#     print(f"Found {files_found} files. Successfully processed {len(all_data)} sequences.")

#     # Stack into final arrays
#     # Shape: (N, T, V, C) -> (N, C, T, V, M)
#     # HPI-GCN Expects: (N, C, T, V, M)

#     X = np.array(all_data, dtype=np.float32) # (N, 64, 25, 3)
#     y = np.array(all_labels, dtype=np.int64)

#     # Transpose to (N, 3, 64, 25)
#     X = np.transpose(X, (0, 3, 1, 2))

#     # Add "Person" dimension (M=1) -> (N, 3, 64, 25, 1)
#     X = np.expand_dims(X, axis=-1)

#     print(f"Final Data Shape: {X.shape}")
#     print(f"Final Label Shape: {y.shape}")

#     # Save
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     np.savez(OUTPUT_PATH, x_data=X, y_data=y, names=sample_names)
#     print(f"Saved to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import re

# ================= CONFIG =================
BRACE_ROOT = r"brace/dataset"  # CHECK THIS PATH
OUTPUT_PATH = "data/brace/BRACE_centered.npz"
TARGET_FRAMES = 64
NUM_NTU_JOINTS = 25

CLASS_MAP = {'toprock': 0, 'footwork': 1, 'powermove': 2}

COCO_TO_NTU = {
    0: 3, 5: 4, 6: 8, 7: 5, 8: 9, 9: 6, 10: 10, 
    11: 12, 12: 16, 13: 13, 14: 17, 15: 14, 16: 18
}

def resize_sequence_linear(data, target_frames):
    source_frames = data.shape[0]
    t_source = np.linspace(0, 1, source_frames)
    t_target = np.linspace(0, 1, target_frames)
    data_resized = np.zeros((target_frames, data.shape[1], data.shape[2]))
    for j in range(data.shape[1]):
        for c in range(data.shape[2]):
            data_resized[:, j, c] = np.interp(t_target, t_source, data[:, j, c])
    return data_resized

def extract_frame_number(filename_key):
    # Extracts "1559" from "-OFL9l7_y2g/img-001559.png"
    # Looks for the number after 'img-'
    try:
        match = re.search(r'img-(\d+)', filename_key)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

def process_brace_file(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    # Robust Sorting
    frame_ids = sorted(raw_data.keys(), key=extract_frame_number)
    if len(frame_ids) < 10: return None

    frames_list = []
    for fid in frame_ids:
        kpts = np.array(raw_data[fid]['keypoints']) 
        
        # FIX: Handle 5-element box [x, y, w, h, score]
        box = raw_data[fid]['box']
        bx, by, bw, bh = box[0], box[1], box[2], box[3]
        
        # Normalize
        kpts[:, 0] = (kpts[:, 0] - bx) / bw
        kpts[:, 1] = (kpts[:, 1] - by) / bh
        
        if kpts.shape[1] == 2:
            kpts = np.column_stack((kpts, np.ones(17)))
            
        frames_list.append(kpts)
        
    frames = np.array(frames_list) 

    # Resize
    frames = resize_sequence_linear(frames, TARGET_FRAMES)

    # Map to NTU
    frames_ntu = np.zeros((TARGET_FRAMES, NUM_NTU_JOINTS, 3))
    for coco_idx, ntu_idx in COCO_TO_NTU.items():
        frames_ntu[:, ntu_idx, :] = frames[:, coco_idx, :]

    # *** HIP CENTERING ***
    # Calculate Average of L_Hip(12) and R_Hip(16)
    # Note: We must handle cases where hips might be 0.0 (missing)
    # But for simplicity in this pass, we assume hips exist or model handles 0.0
    hip_center = (frames_ntu[:, 12, :2] + frames_ntu[:, 16, :2]) / 2.0 
    
    for j in range(NUM_NTU_JOINTS):
        # Subtract hip center from X and Y
        frames_ntu[:, j, 0] -= hip_center[:, 0]
        frames_ntu[:, j, 1] -= hip_center[:, 1]

    return frames_ntu

def main():
    all_data = []
    all_labels = []
    
    print("Scanning and Centering Data...")
    for root, dirs, files in os.walk(BRACE_ROOT):
        for file in files:
            if file.endswith(".json"):
                try:
                    # Robust Label Parsing (folder name or filename)
                    label_str = file.split('_')[-1].split('.')[0]
                    if label_str in CLASS_MAP:
                        p = process_brace_file(os.path.join(root, file))
                        if p is not None:
                            all_data.append(p)
                            all_labels.append(CLASS_MAP[label_str])
                except: pass

    if len(all_data) == 0:
        print("ERROR: No files found! Check BRACE_ROOT path.")
        return

    X = np.array(all_data, dtype=np.float32) 
    y = np.array(all_labels, dtype=np.int64)
    
    # (N, 3, 64, 25, 1)
    X = np.transpose(X, (0, 3, 1, 2))
    X = np.expand_dims(X, axis=-1)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, x_data=X, y_data=y)
    print(f"Data Centered! Shape: {X.shape}")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

