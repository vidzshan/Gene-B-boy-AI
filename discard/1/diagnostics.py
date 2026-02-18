import numpy as np
import matplotlib.pyplot as plt
import torch

# NTU/COCO Mapped Bone Pairs (Shoulders, Arms, Legs, Hips)
BONES_FOR_METRICS = [
    (4, 5), (5, 6),   # Left Arm
    (8, 9), (9, 10),  # Right Arm
    (12, 13), (13, 14), # Left Leg
    (16, 17), (17, 18), # Right Leg
    (4, 8), (12, 16)    # Torso width
]

def get_motion_quality_report(sequence):
    """
    Input: sequence numpy array (3, 64, 25)
    Returns: Dictionary of metrics
    """
    # Sequence is (Channels, Frames, Joints) -> (Frames, Joints, Channels)
    seq = sequence.transpose(1, 2, 0) # [64, 25, 3]
    
    # 1. STABILITY SCORE (Jitter/Shake)
    # Calculate velocity (difference between frames)
    velocity = np.diff(seq, axis=0)
    # Acceleration (difference in velocity) - high accel means shaking
    acceleration = np.diff(velocity, axis=0)
    jitter_score = np.mean(np.abs(acceleration)) * 1000

    # 2. DYNAMICS SCORE (Is it moving or frozen?)
    # Calculate total distance traveled by the center of mass (hips)
    # Hips are approx indices 12 and 16
    hips = (seq[:, 12, :] + seq[:, 16, :]) / 2.0
    total_displacement = np.sum(np.linalg.norm(np.diff(hips, axis=0), axis=1))
    
    # 3. ANATOMY SCORE (Bone Consistency)
    # Do bones stretch? They shouldn't.
    bone_variances = []
    for j1, j2 in BONES_FOR_METRICS:
        # Calculate distance between joints for all frames
        dist = np.linalg.norm(seq[:, j1, :] - seq[:, j2, :], axis=1)
        # Standard Deviation should be 0 (bones don't change length)
        bone_variances.append(np.std(dist))
    anatomy_score = np.mean(bone_variances) * 1000

    return {
        "Jitter (Shakiness)": round(jitter_score, 2),
        "Movement (Energy)": round(total_displacement, 2),
        "Bone Error (Stretching)": round(anatomy_score, 2)
    }

def plot_motion_trail(sequence, save_path="trail.png"):
    """
    Creates a static image showing the path of the movement.
    """
    seq = sequence.transpose(1, 2, 0) # [64, 25, 3]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Invert Y for plotting
    x = seq[:, :, 0]
    y = 1.0 - seq[:, :, 1]

    # Plot every 5th frame with increasing opacity
    num_frames = seq.shape[0]
    active_joints = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

    for t in range(0, num_frames, 4):
        alpha = (t / num_frames) ** 2 # Fade in
        color = plt.cm.viridis(t / num_frames)
        
        # Plot Joints
        # Check if data is valid (not 0.0)
        valid_mask = x[t, active_joints] > 0.01
        
        if np.any(valid_mask):
            # Only plot active joints
            ax.scatter(x[t, active_joints], y[t, active_joints], c=[color], s=10, alpha=alpha)

    plt.title("Motion Trail (Time Lapse)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path
