
import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# CONFIG
DATA_ROOT = './brace/dataset'
OUTPUT_PATH = './data/brace/BRACE_synced_v2.npz'
SEGMENTS_PATH = './brace/annotations/segments.csv'
FRAME_LENGTH = 64

# Load segments for lookup
# Load segments for lookup
segments_df = pd.read_csv(SEGMENTS_PATH)
# Load Video Metadata for FPS
meta_df = pd.read_csv('./brace/videos_info.csv')

def get_video_fps(vid_id):
    try:
        row = meta_df[meta_df['video_id'] == vid_id]
        if not row.empty:
            return float(row['fps'].values[0])
    except: pass
    return 30.0 # Default

def map_coco_to_ntu25(coco_kps):
    T = coco_kps.shape[0]
    ntu = np.zeros((T, 25, 3))
    mapping = {3:0, 4:5, 5:7, 6:9, 8:6, 9:8, 10:10, 12:11, 13:13, 14:15, 16:12, 17:14, 18:16}
    for ntu_idx, coco_idx in mapping.items():
        ntu[:, ntu_idx, :] = coco_kps[:, coco_idx, :]
    ntu[:, 0, :] = (ntu[:, 12, :] + ntu[:, 16, :]) / 2
    ntu[:, 20, :] = (ntu[:, 4, :] + ntu[:, 8, :]) / 2
    ntu[:, 1, :] = (ntu[:, 0, :] + ntu[:, 20, :]) / 2
    vec = ntu[:, 20, :] - ntu[:, 1, :]
    ntu[:, 2, :] = ntu[:, 20, :] + vec * 0.2
    ntu[:, 7, :] = ntu[:, 6, :]; ntu[:, 21, :] = ntu[:, 6, :]; ntu[:, 22, :] = ntu[:, 6, :]
    ntu[:, 11, :] = ntu[:, 10, :]; ntu[:, 23, :] = ntu[:, 10, :]; ntu[:, 24, :] = ntu[:, 10, :]
    ntu[:, 15, :] = ntu[:, 14, :]; ntu[:, 19, :] = ntu[:, 18, :]
    return ntu

def normalize_skeleton(skeletons):
    base = skeletons[:, 0:1, :]
    skeletons = skeletons - base
    return skeletons / 1000.0

def find_segment_info(video_id, frame_idx):
    """Finds which seq_idx and sequence_start_frame this frame belongs to."""
    subset = segments_df[segments_df['video_id'] == video_id]
    # SORTING FIX: Ensure we map to 0, 1, 2 in order
    subset = subset.sort_values('start_frame')
    
    # Iterate with index tracking
    for i, (_, row) in enumerate(subset.iterrows()):
        if row['start_frame'] <= frame_idx <= row['end_frame']:
            # Use 'i' as the seq_idx to match file ordering (0.npz, 1.npz)
            # Instead of the CSV's potentially broken 'seq_idx' column
            return i, int(row['start_frame']), int(row['dance_type_id'])
    return -1, -1, 0

def process_sequence(video_folder):
    parts = os.path.normpath(video_folder).split(os.sep)
    video_id = parts[-1]
    fps = get_video_fps(video_id)
    
    json_files = glob.glob(os.path.join(video_folder, "*.json"))
    clips_data = []
    
    for j_file in json_files:
        with open(j_file, 'r') as f:
            data = json.load(f)
        sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('img-')[-1].split('.')[0]))
        
        coco_kps = []
        frame_indices = []
        for key in sorted_keys:
            coco_kps.append(data[key]['keypoints'])
            frame_indices.append(int(key.split('img-')[-1].split('.')[0]))
        
        ntu_kps = normalize_skeleton(map_coco_to_ntu25(np.array(coco_kps)))
        
        num_frames = len(ntu_kps)
        if num_frames < FRAME_LENGTH: continue
            
        stride = FRAME_LENGTH // 2
        for start in range(0, num_frames - FRAME_LENGTH + 1, stride):
            abs_start_frame = frame_indices[start]
            seq_idx, seq_start, label = find_segment_info(video_id, abs_start_frame)
            
            if seq_idx == -1: continue # Skip if not in a known sequence
            
            # Critical Audio Sync Metadata
            audio_offset = abs_start_frame - seq_start
            
            vis_chunk = ntu_kps[start:start+FRAME_LENGTH].transpose(2,0,1)[:,:,:,np.newaxis]
            
            clips_data.append({
                'data': vis_chunk,
                'video_id': video_id,
                'seq_idx': seq_idx,
                'audio_offset': audio_offset,
                'label': label,
                'fps': fps  # Save FPS
            })
    return clips_data

def build_v2_final():
    print("--- Building Final Synced Dataset (With FPS) ---")
    video_folders = glob.glob(os.path.join(DATA_ROOT, "*", "*"))
    all_clips = []
    for folder in tqdm(video_folders):
        all_clips.extend(process_sequence(folder))
        
    x_data = np.stack([c['data'] for c in all_clips])
    video_ids = np.array([c['video_id'] for c in all_clips])
    seq_indices = np.array([c['seq_idx'] for c in all_clips])
    audio_offsets = np.array([c['audio_offset'] for c in all_clips])
    labels = np.array([c['label'] for c in all_clips])
    fps_arr = np.array([c['fps'] for c in all_clips])
    
    np.savez_compressed(OUTPUT_PATH, 
                        x_data=x_data, 
                        video_ids=video_ids, 
                        seq_indices=seq_indices,
                        audio_offsets=audio_offsets,
                        labels=labels,
                        fps=fps_arr) # New Key
    print(f"✔ Done. Shape: {x_data.shape}. Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_v2_final()