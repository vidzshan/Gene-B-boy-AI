import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# --- IMPORT YOUR MODULES ---
from model.HPI_GCN_OP import Model 
from generator import SasakiGenerator 
from sasaki_system_full import SasakiSystem, CoherenceLoss 
from train_brace_final import BraceDataset, load_and_split_data 

# ================= CONFIG =================
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
DISCRIMINATOR_PATH = './pretrained_models/brace_final_model.pt'
SAVE_DIR = './pretrained_models'

BATCH_SIZE = 16
LATENT_DIM = 128
EPOCHS = 100 
LR_GEN = 0.0002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU BONE DEFINITIONS (For Bone Loss)
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),     # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
]

# ================= HELPER: BONE LOSS =================
def calculate_bone_loss(generated_skeleton):
    """
    Punishes the generator if bones are unnaturally long (The Box Effect).
    Input: (Batch, 3, Frames, 25, 1)
    """
    total_bone_loss = 0.0
    
    # Iterate over defined bones
    for (u, v) in BONES:
        # Get Joint U and Joint V
        # Shape: (Batch, 3, Frames, 1)
        joint_u = generated_skeleton[:, :, :, u, :]
        joint_v = generated_skeleton[:, :, :, v, :]
        
        # Calculate Euclidean Distance (Bone Length)
        # Norm over channel dim (1)
        length = torch.norm(joint_u - joint_v, dim=1)
        
        # PENALTY THRESHOLD: 0.4
        # A human bone in normalized space (0.5 box) rarely exceeds 0.3-0.4.
        # The "Box" diagonal is > 1.0.
        # We use ReLU so we only punish if it exceeds the limit.
        penalty = torch.relu(length - 0.4)
        
        total_bone_loss += torch.mean(penalty)
        
    return total_bone_loss

# ================= MAIN TRAINING =================
def train_gan():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("--- 1. LOADING DATA ---")
    X_train, _, y_train, _ = load_and_split_data()
    dataset = BraceDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("--- 2. LOADING DISCRIMINATOR (THE CRITIC) ---")
    critic = Model(num_class=3, num_point=25, num_person=1, 
                   graph='graph.ntu_rgb_d.Graph', 
                   graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    
    checkpoint = torch.load(DISCRIMINATOR_PATH, map_location=DEVICE)
    critic.load_state_dict(checkpoint)
    critic.to(DEVICE).eval() 
    for param in critic.parameters(): param.requires_grad = False

    print("--- 3. INITIALIZING SASAKI SYSTEM (THE ARTIST) ---")
    gen_body = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
    gen_body.to(DEVICE)
    
    sasaki_system = SasakiSystem(generator_model=gen_body, discriminator_model=critic)
    sasaki_system.to(DEVICE)
    
    coherence_criterion = CoherenceLoss() 
    adversarial_criterion = nn.CrossEntropyLoss() 

    optimizer_G = optim.Adam(gen_body.parameters(), lr=LR_GEN)

    print(f"--- STARTING GENERATION LOOP ({EPOCHS} Epochs) ---")
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for real_motion, real_labels in dataloader:
            real_motion = real_motion.to(DEVICE) 
            real_labels = real_labels.to(DEVICE)
            batch_size = real_motion.size(0)

            # 1. Inputs
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            label_onehot = torch.zeros(batch_size, 3).to(DEVICE)
            label_onehot.scatter_(1, real_labels.unsqueeze(1), 0.9) # Soft Targets

            # 2. Sasaki Decision
            generated_skeleton, intent_vector = sasaki_system(z, label_onehot, real_motion)

            # 3. LOSS CALCULATION
            
            # A. Realism (Priority #1)
            critic_pred = critic(generated_skeleton) 
            loss_realism = adversarial_criterion(critic_pred, real_labels)

            # B. Intent (Priority #3 - Reduced Weight)
            loss_intent = coherence_criterion(generated_skeleton, intent_vector)
            
            # C. Anatomy (Priority #2 - The Bone Breaker)
            loss_bone = calculate_bone_loss(generated_skeleton)

            # TOTAL LOSS MIX
            # 0.9 Realism: Look Human first.
            # 0.1 Intent: Obey Sasaki second.
            # 1.0 Bone: DO NOT BE A BOX.
            loss_G = (0.9 * loss_realism) + (0.1 * loss_intent) + (1.0 * loss_bone)

            # 4. Update
            optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(gen_body.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            total_loss += loss_G.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
        if (epoch+1) % 10 == 0:
            filename = f"sasaki_generator_ep{epoch+1}.pt"
            torch.save(gen_body.state_dict(), os.path.join(SAVE_DIR, filename))
            torch.save(gen_body.state_dict(), os.path.join(SAVE_DIR, "sasaki_generator.pt"))
            print(f" >>> Saved Checkpoint: {filename}")

    print("--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    train_gan()
