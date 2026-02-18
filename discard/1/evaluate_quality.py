import torch
import numpy as np
from model.generator import MotionGenerator

# ================= CONFIG =================
GENERATOR_PATH = r"pretrained_models/generator_checkpoints_dynamic/generator_dynamic_ep100.pt"
LATENT_DIM = 128

# Pairs of joints that form rigid bones (e.g., Knee to Ankle)
# We exclude the Spine/Head because they are flexible
RIGID_BONES = [
    (4, 5), (5, 6),    # Left Arm (Upper, Lower)
    (8, 9), (9, 10),   # Right Arm
    (12, 13), (13, 14), # Left Leg
    (16, 17), (17, 18)  # Right Leg
]

def load_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    if not torch.cuda.is_available():
        gen.load_state_dict(torch.load(GENERATOR_PATH, map_location='cpu'))
    else:
        gen.load_state_dict(torch.load(GENERATOR_PATH))
    gen.eval()
    return gen, device

def calculate_metrics(gen, device):
    print(f"📊 Analyzing Generator Quality ({GENERATOR_PATH})...")
    
    # Generate a batch of 100 samples (Mixed classes)
    batch_size = 100
    z = torch.randn(batch_size, LATENT_DIM).to(device)
    # Create random labels (0, 1, 2)
    labels = torch.randint(0, 3, (batch_size,)).to(device)
    
    with torch.no_grad():
        # Shape: [Batch, 3, 64, 25, 1]
        generated = gen(z, labels)
        # Reshape to [Batch, Time, Joints, Coords] -> [100, 64, 25, 3]
        data = generated.squeeze().permute(0, 2, 3, 1).cpu().numpy()

    # --- METRIC 1: BONE LENGTH CONSISTENCY ---
    # Bones should not stretch. Variance should be near 0.
    bone_variances = []
    for b_start, b_end in RIGID_BONES:
        # Vector between joints
        # [100, 64, 3]
        vec = data[:, :, b_start, :] - data[:, :, b_end, :]
        # Length = Sqrt(x^2 + y^2 + z^2)
        lengths = np.linalg.norm(vec, axis=-1)
        # Variance over Time (how much does length change during the dance?)
        var = np.var(lengths, axis=1) 
        bone_variances.append(np.mean(var))
    
    avg_bone_error = np.mean(bone_variances) * 1000 # Scale up for readability

    # --- METRIC 2: SMOOTHNESS (JERK) ---
    # Acceleration change. High jerk = Shaky/Jittery.
    # Velocity = frame[t+1] - frame[t]
    velocity = np.diff(data, axis=1)
    # Accel = vel[t+1] - vel[t]
    accel = np.diff(velocity, axis=1)
    # Jerk = accel[t+1] - accel[t]
    jerk = np.diff(accel, axis=1)
    
    avg_jerk = np.mean(np.abs(jerk)) * 1000

    # --- METRIC 3: DIVERSITY (Movement Range) ---
    # How far does the skeleton travel?
    # We look at the Pelvis/Root (Joint 0 or 1)
    root_pos = data[:, :, 1, :] # Shape [100, 64, 3]
    # Max position - Min position for each sample
    ranges = np.max(root_pos, axis=1) - np.min(root_pos, axis=1)
    avg_range = np.mean(np.linalg.norm(ranges, axis=-1))

    print("-" * 40)
    print(f"🦴 Bone Consistency Error: {avg_bone_error:.4f}")
    print(f"   (Target: < 1.0. If high -> Rubber limbs)")
    print("-" * 40)
    print(f"〰️ Shakiness (Jerk):      {avg_jerk:.4f}")
    print(f"   (Target: < 5.0. If high -> Jitter/Seizure)")
    print("-" * 40)
    print(f"💃 Movement Magnitude:    {avg_range:.4f}")
    print(f"   (Target: > 0.1. If low -> Statue/Frozen)")
    print("-" * 40)

    # Final Verdict
    if avg_bone_error < 2.0 and avg_jerk < 10.0:
        if avg_range > 0.1:
            print("✅ VERDICT: REALISTIC & DYNAMIC")
        else:
            print("⚠️ VERDICT: REALISTIC BUT STATIC (Needs more movement)")
    else:
        print("❌ VERDICT: UNSTABLE (Needs more training or smoothness loss)")

if __name__ == "__main__":
    gen, device = load_generator()
    calculate_metrics(gen, device)
