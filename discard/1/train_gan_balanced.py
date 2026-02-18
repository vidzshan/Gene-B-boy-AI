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
SAVE_DIR = './pretrained_models/generator_checkpoints_balanced/' # FINAL FOLDER
BATCH_SIZE = 16
LATENT_DIM = 128
EPOCHS = 60
LR_GEN = 0.0002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# BONES
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

def calculate_bone_loss(generated_skeleton):
    loss = 0.0
    for (u, v) in BONES:
        joint_u = generated_skeleton[:, :, :, u, :]
        joint_v = generated_skeleton[:, :, :, v, :]
        dist = torch.norm(joint_u - joint_v, dim=1)
        # Relaxed threshold slightly to 0.35
        loss += torch.mean(torch.relu(dist - 0.35))
    return loss

def calculate_velocity_loss(generated_skeleton):
    diff = generated_skeleton[:, :, 1:, :, :] - generated_skeleton[:, :, :-1, :, :]
    velocity = torch.norm(diff, dim=1)
    # Relaxed threshold: Allow up to 0.15 movement per frame
    loss = torch.mean(torch.relu(velocity - 0.15))
    return loss

def train_gan_balanced():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    print("--- Loading Data ---")
    X_train, _, y_train, _ = load_and_split_data()
    dataset = BraceDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print("--- Loading Critic ---")
    critic = Model(num_class=3, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    critic.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=DEVICE))
    critic.to(DEVICE).eval()
    for p in critic.parameters(): p.requires_grad = False 
    
    print("--- Initializing Sasaki System ---")
    gen_body = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
    sasaki_system = SasakiSystem(gen_body, critic).to(DEVICE)
    
    optimizer_G = optim.Adam(gen_body.parameters(), lr=LR_GEN)
    adversarial_criterion = nn.CrossEntropyLoss()
    coherence_criterion = CoherenceLoss()
    
    anchor_skeleton = get_mean_skeleton(dataloader, DEVICE)
    
    print(f"--- STARTING BALANCED TRAINING ({EPOCHS} Epochs) ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # ANCHOR: Decay to 0.0 at Epoch 30
        anchor_weight = max(0.0, 1.0 - (epoch / 30.0)) 
        
        for real_motion, real_labels in dataloader:
            real_motion, real_labels = real_motion.to(DEVICE), real_labels.to(DEVICE)
            B = real_motion.size(0)
            
            z = torch.randn(B, LATENT_DIM).to(DEVICE)
            label_onehot = torch.zeros(B, 3).to(DEVICE)
            label_onehot.scatter_(1, real_labels.unsqueeze(1), 0.9)
            
            fake_skel, intent_vec = sasaki_system(z, label_onehot, real_motion)
            
            # LOSSES
            pred = critic(fake_skel)
            loss_realism = adversarial_criterion(pred, real_labels)
            
            loss_intent = coherence_criterion(fake_skel, intent_vec)
            
            # PHYSICS LOSSES
            loss_bone = calculate_bone_loss(fake_skel)
            loss_velocity = calculate_velocity_loss(fake_skel)
            
            loss_anchor = torch.mean((fake_skel - anchor_skeleton) ** 2)
            
            # === THE BALANCED FORMULA ===
            # Realism (1.0) is now comparable to Physics (2.0 + 2.0)
            # The Statue effect is killed because Realism demands a skeleton shape.
            loss_G = (1.0 * loss_realism) + \
                     (0.5 * loss_intent) + \
                     (2.0 * loss_bone) + \
                     (2.0 * loss_velocity) + \
                     (anchor_weight * 5.0 * loss_anchor) 
            
            optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(gen_body.parameters(), 1.0)
            optimizer_G.step()
            
            total_loss += loss_G.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Anchor W: {anchor_weight:.2f}")
        
        if (epoch+1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"sasaki_gen_ep{epoch+1}.pt")
            torch.save(gen_body.state_dict(), save_path)
            print(f" >>> Saved: {save_path}")

if __name__ == "__main__":
    train_gan_balanced()   # <--- Call the function with parentheses