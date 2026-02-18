
# Chapter 12: The Bug Chronicles (Development Log)

This chapter details the specific, critical failures encountered during the development of the Sasaki-GAN, serving as a technical reference for common pitfalls in Neuro-Symbolic Dance Generation.

## 12.1 The "Frozen Epoch" (Epoch 0-10)
**Date:** November 4, 2025
**Symptom:** The Loss curve for the Generator remained exactly at `0.693` (ln 2) while the Discriminator dropped to `0.0`.
**Error Log:**
```text
Epoch 10/50
Settings: L_adv=1.0, L_phys=0.0
Generator Loss: 0.6931 (Static)
Discriminator Loss: 0.0000 (Perfect)
WARNING: Gradient Norm for Generator is 0.00000e+00
```
**Diagnosis:** Vanishing Gradients. The Discriminator was perfect too early.
**Fix:** implemented `Spectral Normalization` in the Discriminator and added `Gaussian Noise` ($\sigma=0.1$) to the real inputs.

## 12.2 The "Sliding Foot" (Epoch 25)
**Date:** December 12, 2025
**Symptom:** The character performed a Toprock, but the global root position drifted 2 meters to the left.
**Code Trace:**
In `sasaki_brain.py`:
```python
# Old Logic
root_pos += predicted_velocity * delta_t
```
The `predicted_velocity` had a constant bias of `0.001` due to ReLU activation causing a positive mean shift.
**Fix:** Switched to LeakyReLU (negative slope 0.2) to center the distribution at zero.

## 12.3 The "Memory Leak" (Live Run)
**Date:** January 5, 2026
**Symptom:** After 4 minutes of playback, the FPS dropped from 30 to 5.
**Error Log:**
```text
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 8.00 GiB total capacity; 7.20 GiB already allocated)
```
**Diagnosis:** The `memory_buffer` list in Python was appending tensors that were still attached to the Computation Graph (`requires_grad=True`).
**Fix:** Added `.detach().cpu().numpy()` before taking snapshots for the history buffer.

## 12.4 The "UDP Packet Loss" (Blender)
**Date:** January 15, 2026
**Symptom:** The Blender character would "teleport" 10 frames into the future.
**Diagnosis:** We were sending packets faster than Blender's Modal Operator could tick. The OS network buffer was filling up, and Blender was reading the *oldest* packet first (FIFO).
**Fix:** Changed behavior to LIFO (Last In First Out). The Blender script now drains the entire socket buffer and only processes the *last* received packet.

## 12.5 The "Silence Panic"
**Date:** January 20, 2026
**Symptom:** When the music stopped, the GAN outputted random high-frequency noise.
**Reason:** Z-Score Normalization divided by zero (std_dev = 0).
**Fix:** Added `epsilon=1e-6` to the denominator and a "Silence Gate" that forces the output vector to `zeros` if RMS < 0.001.

---

# Appendix B: Supplementary Code Repository

## B.1 Data Augmentation (`augment_data.py`)
This script applies random rotation (Y-axis) and mirroring to the dataset to effectively quadruple the training data size.

```python
def augment_data(data):
    # 1. Mirroring
    mirrored = data.copy()
    mirrored[:, :, 0] *= -1 # Flip X
    # Swap Left/Right Joints (Indices based on NTU)
    mirrored = swap_joints(mirrored, left=[5,6,7], right=[9,10,11])
    
    # 2. Rotation
    rotated = []
    for angle in [90, 180, 270]:
        r_data = rotate_y(data, angle)
        rotated.append(r_data)
        
    return np.concatenate([data, mirrored] + rotated)
```

## B.2 COCO-17 Graph Definition (`coco_17.py`)
The minimal graph structure used for the Zero-Padding bridge.

```python
# Edge List for COCO-17
edges = [
    (0, 1), (0, 2), # Nose to Eyes
    (1, 3), (2, 4), # Eyes to Ears
    (5, 7), (7, 9), # Left Arm
    (6, 8), (8, 10), # Right Arm
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16), # Right Leg
    (5, 6), (11, 12), # Shoulder/Hip Bridges
    (5, 11), (6, 12)  # Torso
]
```

## B.3 3D Visualization Tool (`visualize_3d.py`)
Used for offline verification using Matplotlib.

```python
def plot_3d(pose, ax):
    ax.cla()
    for edge in edges:
        p1 = pose[:, edge[0]]
        p2 = pose[:, edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
```
