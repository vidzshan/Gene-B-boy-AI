
import numpy as np
import os

INPUT = "data/brace/BRACE_centered.npz"
OUTPUT = "data/brace/BRACE_centered_aug.npz"

if not os.path.exists(INPUT):
    print("Please run Step 1 first.")
else:
    print("Augmenting...")
    data = np.load(INPUT)
    x = data['x_data']
    y = data['y_data']

    # Mirroring (Flip X)
    x_flip = x.copy()
    x_flip[:, 0, :, :, :] *= -1 

    # Swap Left/Right Joints
    pairs = [(4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18)]
    for a, b in pairs:
        temp = x_flip[:, :, :, a, :].copy()
        x_flip[:, :, :, a, :] = x_flip[:, :, :, b, :]
        x_flip[:, :, :, b, :] = temp

    x_final = np.concatenate((x, x_flip), axis=0)
    y_final = np.concatenate((y, y), axis=0)

    np.savez(OUTPUT, x_data=x_final, y_data=y_final)
    print(f"Augmented. Final Shape: {x_final.shape}")
