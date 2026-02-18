import numpy as np

DATA_PATH = "data/brace/BRACE_fixed_topology_v3.npz"

def check_zeros():
    data = np.load(DATA_PATH)
    x = data['x_data'] # (N, 3, 64, 25, 1)
    
    # Check average value of specific joints across all frames/samples
    # Sum absolute values. If sum is 0, the joint is empty.
    abs_sum = np.sum(np.abs(x), axis=(0, 1, 2, 4))
    
    print("--- Joint Activity Report ---")
    missing_critical = []
    for i in range(25):
        status = "ACTIVE" if abs_sum[i] > 0 else "DEAD (0.0)"
        print(f"Joint {i}: {status}")
        
        # Check specific critical spine joints
        if i in [0, 1, 20, 2] and abs_sum[i] == 0:
            missing_critical.append(i)

    if missing_critical:
        print("\n🚨 CRITICAL ISSUE FOUND 🚨")
        print(f"The Graph Bridge is broken. Missing Joints: {missing_critical}")
        print("The GCN cannot transfer info from Hips to Shoulders.")
    else:
        print("\n✅ Spine seems active. Problem might be elsewhere.")

if __name__ == "__main__":
    check_zeros()
