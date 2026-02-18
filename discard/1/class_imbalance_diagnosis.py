import numpy as np
import os

DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'

def check_imbalance():
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found.")
        return

    print(f"Loading {DATA_PATH}...")
    npz_data = np.load(DATA_PATH)
    y_key = 'y_data' if 'y_data' in npz_data else 'y_train'
    y = npz_data[y_key]

    # Handle labels if they are one-hot encoded or arrays
    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    unique, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    
    classes = {0: 'Toprock', 1: 'Footwork', 2: 'Powermove'}
    
    print("\n--- CLASS DISTRIBUTION ---")
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"{classes.get(cls, cls)}: {count} samples ({percentage:.2f}%)")
    
    if max(counts) / min(counts) > 1.5:
        print("\n⚠️ Imbalance Detected! The majority class is >1.5x larger than the minority.")
        print("   Action: We MUST use Weighted CrossEntropyLoss.")
    else:
        print("\n✅ Dataset is fairly balanced. Weighted Loss is optional but still recommended.")

if __name__ == "__main__":
    check_imbalance()
