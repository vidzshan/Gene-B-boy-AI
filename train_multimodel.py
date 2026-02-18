import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from audio_brace_loader import AudioBraceDataset
from model.generator import AudioAwareGenerator 
from model.HPI_GCN_OP import Model as Discriminator
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
SKELETON_NPZ = './data/brace/BRACE_synced_v2.npz'
AUDIO_DIR = './brace/audio_features/'
DISC_PATH = './pretrained_models/brace_final_model.pt'
SAVE_DIR = './pretrained_models/audio_checkpoints/'
BATCH_SIZE = 8 
LR_GEN = 0.0001  
LR_DISC = 0.0002 # Slightly slower D for better balance
EPOCHS = 50 
DEVICE = torch.device("cuda")

BONES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (20, 8), (8, 9), (9, 10), 
         (0, 12), (12, 13), (13, 14), (0, 16), (16, 17), (17, 18)]

def stabilized_expansion_loss(gen_skel):
    p1 = gen_skel.squeeze(-1) # (B, 3, 64, 25)
    
    # 1. ROOT LOCK: Joint 0 (SpineBase) must stay at (0,0)
    # This stops the skeleton from flying off-screen
    root_pos = p1[:, :2, :, 0] # (B, 2, 64)
    loss_root = torch.mean(root_pos**2) * 100.0

    # 2. BONE HINGE: Keep limbs in a human range
    loss_bone = 0
    for u, v in BONES:
        dist = torch.norm(p1[:, :, :, u] - p1[:, :, :, v], dim=1)
        loss_bone += torch.mean(torch.relu(0.15 - dist)**2 + torch.relu(dist - 0.4)**2)
    
    return loss_root + loss_bone

def max_correlation_sync_loss(gen_velocity, audio_beats, window=10):
    B, T = gen_velocity.shape
    gen_v = (gen_velocity - gen_velocity.mean(dim=1, keepdim=True)) / (gen_velocity.std(dim=1, keepdim=True) + 1e-6)
    aud_b = (audio_beats - audio_beats.mean(dim=1, keepdim=True)) / (audio_beats.std(dim=1, keepdim=True) + 1e-6)
    corrs = []
    for shift in range(-window, window + 1):
        if shift < 0:
            c = torch.mean(gen_v[:, :shift] * aud_b[:, -shift:], dim=1)
        elif shift > 0:
            c = torch.mean(gen_v[:, shift:] * aud_b[:, :-shift], dim=1)
        else:
            c = torch.mean(gen_v * aud_b, dim=1)
        corrs.append(c)
    max_corr = torch.stack(corrs, dim=1).max(dim=1)[0]
    return 1.0 - torch.mean(max_corr)

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dataset = AudioBraceDataset(SKELETON_NPZ, AUDIO_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    generator = AudioAwareGenerator(audio_dim=33).to(DEVICE)
    discriminator = Discriminator(num_class=3, num_point=25, num_person=1, 
                                 graph='graph.ntu_rgb_d.Graph', 
                                 graph_args={'labeling_mode': 'spatial'}).to(DEVICE)
    discriminator.load_state_dict(torch.load(DISC_PATH))
    
    MEAN_POSE = torch.from_numpy(np.load('./data/brace/mean_pose.npy')).float().to(DEVICE) 
    
    opt_g = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_DISC, betas=(0.5, 0.999))
    criterion_cls = nn.CrossEntropyLoss()
    
    print(f"🚀 Launching STABILIZED Sasaki-GAN Training...")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        # Phase 2 Weights: Fixed Structure
        anch_w = 1.0 
        
        for i, (real_skel, audio_feats, labels) in enumerate(pbar):
            audio_feats = audio_feats.to(DEVICE); labels = labels.to(DEVICE).long(); real_skel = real_skel.to(DEVICE)

            # 1. Update Discriminator
            opt_d.zero_grad()
            z = torch.randn(BATCH_SIZE, 128).to(DEVICE)
            label_oh = torch.eye(3, device=DEVICE)[labels]
            with torch.no_grad():
                fake_skel = generator(z, audio_feats, label_oh)
            d_real = discriminator(real_skel)
            d_fake = discriminator(fake_skel.detach())
            loss_d = criterion_cls(d_real, labels) + criterion_cls(d_fake, torch.zeros_like(labels)) 
            loss_d.backward()
            opt_d.step()

            # 2. Update Generator
            opt_g.zero_grad()
            fake_skel = generator(z, audio_feats, label_oh)
            g_out = discriminator(fake_skel)
            
            # --- REALISM (WEIGHT: 5.0) ---
            loss_adv = criterion_cls(g_out, labels) * 5.0
            
            # --- PHYSICS (WEIGHT: 10.0) ---
            p1 = fake_skel.squeeze(-1)
            root_pos = p1[:, :2, :, 0] 
            loss_root = torch.mean(root_pos**2) * 10.0 
            
            loss_bone = 0
            for u, v in BONES:
                dist = torch.norm(p1[:, :, :, u] - p1[:, :, :, v], dim=1)
                loss_bone += torch.mean(torch.relu(0.15 - dist)**2 + torch.relu(dist - 0.4)**2)

            loss_physics = (loss_root + loss_bone) * 10.0

            # --- DIVERSITY (WEIGHT: 10.0) ---
            flat_fake = fake_skel.view(BATCH_SIZE, -1)
            dist_matrix = torch.cdist(flat_fake, flat_fake, p=2)
            num_features = flat_fake.shape[1]
            loss_div = -torch.mean(dist_matrix) / num_features
            loss_div = loss_div * 10.0 

            # --- SYNC (WEIGHT: 10.0) ---
            diff = fake_skel[:, :, 1:] - fake_skel[:, :, :-1]
            velocity = torch.norm(diff, dim=1).mean(dim=(2, 3)) 
            velocity = torch.cat([velocity, velocity[:, -1:]], dim=1)
            loss_sync = max_correlation_sync_loss(velocity, audio_feats[:,:,-1], window=10) * 10.0
            
            # --- ANCHOR (WEIGHT: 1.0) ---
            seq_mean = torch.mean(p1, dim=2) 
            loss_anchor = torch.mean((seq_mean - MEAN_POSE.unsqueeze(0))**2) * anch_w

            # COMBINED LOSS
            loss_g = loss_adv + loss_physics + loss_sync + loss_div + loss_anchor
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5) 
            opt_g.step()
            
            pbar.set_description(f"E{epoch+1}|Adv:{loss_adv:.2f}|Phys:{loss_physics:.1f}|Div:{loss_div:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f'{SAVE_DIR}/structured_sasaki_gen_e{epoch+1}.pth')

if __name__ == "__main__":
    train()