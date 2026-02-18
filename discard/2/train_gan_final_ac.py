import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# --- IMPORTS ---
from model.HPI_GCN_OP import Model
from generator import SasakiGenerator
from sasaki_system_full import SasakiSystem, CoherenceLoss
from train_brace_final import BraceDataset, load_and_split_data

# ================= CONFIG =================
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
DISCRIMINATOR_PATH = './pretrained_models/brace_final_model.pt'
SAVE_DIR = './pretrained_models/generator_checkpoints_ac/'
BATCH_SIZE = 16
LATENT_DIM = 128
EPOCHS = 60
LR_GEN = 0.0002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# BONES (NTU Topology)
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), 
    (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
    (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

def get_mean_skeleton(dataloader, device):
    print(" [Setup] Computing Mean Skeleton (Anchor)...")
    accum = None
    count = 0
    for data, _ in dataloader:
        curr_mean = data.mean(dim=(0, 2)) 
        if accum is None: accum = torch.zeros_like(curr_mean).to(device)
        accum += curr_mean.to(device)
        count += 1
    mean_pose = accum / count
    return mean_pose.unsqueeze(0).unsqueeze(2).repeat(1, 1, 64, 1, 1)

# --- PHYSICS LOSSES (Standardized) ---
def calculate_physics_loss(fake):
    # 1. Bone Loss (Quadratic)
    bone_loss = 0.0
    for (u, v) in BONES:
        d = torch.norm(fake[:,:,:,u,:] - fake[:,:,:,v,:], dim=1)
        bone_loss += torch.mean((d - 0.2)**2)
        
    # 2. Velocity Loss (Quadratic)
    diff = fake[:,:,1:,:,:] - fake[:,:,:-1,:,:]
    vel = torch.norm(diff, dim=1)
    vel_loss = torch.mean(torch.relu(vel - 0.15)**2) # Relaxed slightly to 0.15
    
    # 3. Centering (Prevents Box)
    center_loss = torch.mean(fake ** 2)
    
    return bone_loss, vel_loss, center_loss

def train_gan_ac():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    print("--- Loading Data ---")
    X_train, _, y_train, _ = load_and_split_data()
    dataset = BraceDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print("--- Loading Discriminator (Frozen Teacher) ---")
    critic = Model(num_class=3, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True).to(DEVICE)
    critic.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=DEVICE))
    critic.eval()
    for p in critic.parameters(): p.requires_grad = False
    
    print("--- Initializing Sasaki Generator ---")
    gen_body = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
    sasaki_system = SasakiSystem(gen_body, critic).to(DEVICE)
    optimizer_G = optim.Adam(gen_body.parameters(), lr=LR_GEN)
    
    classification_criterion = nn.CrossEntropyLoss()
    coherence_criterion = CoherenceLoss()
    
    anchor_skeleton = get_mean_skeleton(dataloader, DEVICE)
    
    print(f"--- STARTING AC-GAN TRAINING (Class Diversity Focused) ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_div = 0
        total_cls = 0
        
        # Anchor decays to 0.0 eventually
        anchor_weight = max(0.0, 1.0 - (epoch / 25.0))
        
        for real_motion, real_labels in dataloader:
            real_motion, real_labels = real_motion.to(DEVICE), real_labels.to(DEVICE)
            B = real_motion.size(0)
            
            # --- INPUT PREPARATION ---
            # We create TWO random noises for the SAME label to force diversity
            z1 = torch.randn(B, LATENT_DIM).to(DEVICE)
            z2 = torch.randn(B, LATENT_DIM).to(DEVICE)
            
            label_onehot = torch.zeros(B, 3).to(DEVICE)
            label_onehot.scatter_(1, real_labels.unsqueeze(1), 0.9) # Soft labels
            
            # --- GENERATION ---
            fake1, intent1 = sasaki_system(z1, label_onehot, real_motion)
            fake2, _       = sasaki_system(z2, label_onehot, real_motion) # Same label, diff noise
            
            # --- LOSS 1: AC-GAN CLASSIFICATION (The "Meaning" Loss) ---
            # Does the Discriminator recognize the move?
            pred_logits = critic(fake1)
            loss_class = classification_criterion(pred_logits, real_labels)
            
            # --- LOSS 2: DIVERSITY REGULARIZATION (The "Mode Collapse" Killer) ---
            # Calculate distance between the two generated skeletons
            gen_diff = torch.mean(torch.abs(fake1 - fake2))
            # We want to MAXIMIZE difference, so we minimize negative difference
            # But strictly maximizing leads to explosion, so we cap it
            loss_diversity = -1.0 * torch.clamp(gen_diff, max=0.5) 
            
            # --- LOSS 3: PHYSICS & ANCHOR (The "Structure" Loss) ---
            bone_l, vel_l, center_l = calculate_physics_loss(fake1)
            loss_intent = coherence_criterion(fake1, intent1)
            loss_anchor = torch.mean((fake1 - anchor_skeleton) ** 2)
            
            # === THE FINAL BALANCED FORMULA ===
            # Classification (20.0): Highest Priority. Must look like the target Class.
            # Physics (10.0): Keep it valid.
            # Diversity (5.0): Force variations.
            # Anchor: Decays.
            
            loss_G = (20.0 * loss_class) + \
                     (5.0 * loss_diversity) + \
                     (10.0 * bone_l) + \
                     (10.0 * vel_l) + \
                     (1.0 * center_l) + \
                     (0.5 * loss_intent) + \
                     (anchor_weight * 5.0 * loss_anchor)
            
            optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(gen_body.parameters(), 1.0)
            optimizer_G.step()
            
            total_loss += loss_G.item()
            total_div += gen_diff.item()
            total_cls += loss_class.item()

        # LOGGING
        print(f"Epoch {epoch+1} | Total Loss: {total_loss/len(dataloader):.4f} | Class Loss: {total_cls/len(dataloader):.4f} | Diversity: {total_div/len(dataloader):.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(gen_body.state_dict(), os.path.join(SAVE_DIR, f"sasaki_gen_ep{epoch+1}.pt"))

if __name__ == "__main__":
    train_gan_ac()