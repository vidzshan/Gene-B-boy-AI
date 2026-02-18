import torch
import numpy as np
import os

# --- IMPORTS ---
from model.HPI_GCN_OP import Model
from generator import SasakiGenerator

# ================= CONFIG =================
GEN_PATH = './pretrained_models/generator_checkpoints_physics/sasaki_gen_ep50.pt' # Change to your best epoch
DISC_PATH = './pretrained_models/brace_final_model.pt'
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU BONES LIST
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), 
    (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
    (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

def load_models():
    # 1. Load Generator
    gen = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(DEVICE)
    try:
        gen.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
        print(f"✅ Generator Loaded: {GEN_PATH}")
    except:
        print(f"❌ Error: Could not load {GEN_PATH}")
        return None, None

    # 2. Load Discriminator
    # Note: We must load it exactly as it was trained (120 classes initially, usually chopped to 3)
    # However, your saved brace_final_model.pt likely has the head already chopped to 3.
    # We try loading with 3 classes first.
    disc = Model(num_class=3, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True).to(DEVICE)
    try:
        disc.load_state_dict(torch.load(DISC_PATH, map_location=DEVICE))
        print(f"✅ Discriminator Loaded: {DISC_PATH}")
    except:
        print("⚠️ Warning: Weight mismatch. Ensure Discriminator architecture matches saved weights.")
    
    gen.eval()
    disc.eval()
    return gen, disc

def audit_move(gen, disc, move_name, label_vec):
    print(f"\n--- AUDIT: {move_name} ---")
    
    # 1. Generate
    z = torch.randn(16, LATENT_DIM).to(DEVICE) # Batch of 16 to get average stats
    label = torch.tensor([label_vec]).repeat(16, 1).to(DEVICE)
    intent = torch.tensor([[1.0, 0.0, 0.0]]).repeat(16, 1).to(DEVICE) # Attack Intent
    
    with torch.no_grad():
        fake_skel = gen(z, label, intent) # (16, 3, 64, 25, 1)
        
        # 2. Discriminator Opinion
        disc_out = disc(fake_skel)
        probs = torch.softmax(disc_out, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
        avg_conf = torch.mean(confidence).item()
        
    print(f"1. DISCRIMINATOR OPINION:")
    print(f"   Avg Confidence: {avg_conf*100:.2f}%")
    print(f"   (If this is high but skeleton looks bad, the Discriminator is 'fooled')")

    # 3. Geometry Check (Bone Integrity)
    # Convert to Numpy for math
    skel_np = fake_skel.cpu().numpy() # (16, 3, 64, 25, 1)
    
    avg_bone_errors = []
    
    print(f"2. ANATOMY CHECK (Bone Lengths):")
    # We check Frame 0 vs Frame 30 to see if bones stretch
    for b_idx, (u, v) in enumerate(BONES):
        # Calculate length of this bone across all frames/batches
        joint_u = skel_np[:, :, :, u, 0]
        joint_v = skel_np[:, :, :, v, 0]
        
        # Euclidean distance
        dist = np.linalg.norm(joint_u - joint_v, axis=1) # (16, 64)
        mean_len = np.mean(dist)
        std_len = np.std(dist)
        
        # Valid bones shouldn't change length (Std Dev should be low)
        if std_len > 0.05:
            print(f"   ⚠️ Bone {u}-{v} is Elastic! (Var: {std_len:.4f})")
        
        avg_bone_errors.append(mean_len)

    avg_len = np.mean(avg_bone_errors)
    print(f"   Average Bone Length: {avg_len:.4f} (Should be ~0.15 - 0.25)")
    
    # 4. Physics Check (Velocity)
    diff = skel_np[:, :, 1:, :, :] - skel_np[:, :, :-1, :, :]
    velocity = np.linalg.norm(diff, axis=1) # (16, 63, 25, 1)
    avg_vel = np.mean(velocity)
    max_vel = np.max(velocity)
    
    print(f"3. PHYSICS CHECK (Velocity):")
    print(f"   Avg Speed: {avg_vel:.4f}")
    print(f"   Max Burst: {max_vel:.4f}")
    
    if max_vel > 0.3:
        print("   ❌ CRITICAL: Teleporting detected (Max > 0.3)")
    elif avg_vel < 0.01:
        print("   ❌ CRITICAL: Statue detected (Avg < 0.01)")
    else:
        print("   ✅ Motion looks fluid.")

if __name__ == "__main__":
    gen, disc = load_models()
    if gen and disc:
        # Audit Toprock
        audit_move(gen, disc, "Toprock", [1.0, 0.0, 0.0])
        # Audit Powermove
        audit_move(gen, disc, "Powermove", [0.0, 0.0, 1.0])