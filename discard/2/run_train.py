# Direct Coordinate Matching between fake and real:
# code
# Python
# # From your failure script
# loss_coord = torch.mean((fake_masked - real_masked) ** 2)
# The Fatal Flaw:
# In a GAN, the input z (noise) is random. The real batch is also random.
# The Generator creates a "Windmill."
# The Real Batch happens to contain a "Toprock."
# You tell the Generator: "Error! You should have been a Toprock!"
# Next batch, it generates a "Toprock," but the Real Batch has a "Flare."
# You tell the Generator: "Error! You should have been a Flare!"
# Result: The Generator gets confused and learns to output the mathematical average of all moves to minimize error. This results in Blurry Motion, Jitter, or the "Cloud" artifact.

print("DEBUG: VERY START", flush=True)
import torch
print("DEBUG: torch imported", flush=True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
print("DEBUG: std libs imported", flush=True)
from model.HPI_GCN_OP import Model as Discriminator
print("DEBUG: Discriminator imported", flush=True)
from model.generator import MotionGenerator
print("DEBUG: Generator imported", flush=True)

# --- CONFIG ---
DATA_PATH = "data/brace/BRACE_centered_aug.npz"
DISCRIMINATOR_PATH = "pretrained_models/brace_finetuned.pt"
SAVE_DIR = "pretrained_models/generator_checkpoints_final"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
LR = 0.0002
EPOCHS = 60
LATENT_DIM = 128

# BALANCED WEIGHTS for CENTERED DATA
# Since data is centered, Coord loss is naturally smaller and cleaner.
LAMBDA_CLASS = 2.0   
LAMBDA_COORD = 5.0   # Relaxed: Stop forcing it to be a statue
LAMBDA_BONE  = 10.0  # Moderate: Maintain structure but allow flex
LAMBDA_VEL   = 5.0   # Increased: Force smooth motion (No Jitter)
LAMBDA_DIV   = 5.0   # Force variety

ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
BONES = [(4, 5), (5, 6), (8, 9), (9, 10), (4, 8), (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)]

class BraceDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = data['x_data']
        self.y = data['y_data']
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        data = torch.from_numpy(self.x[idx]).float()
        C, T, V, M = data.shape
        data_padded = torch.zeros(C, T, V, 2)
        data_padded[:, :, :, 0] = data[:, :, :, 0]
        return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

def calculate_losses(fake, real):
    fake_p1 = fake[:, :, :, :, 0]
    real_p1 = real[:, :, :, :, 0]
    
    # Mask: Only calculate loss on joints that exist in COCO
    mask = torch.zeros_like(fake_p1)
    mask[:, :, :, ACTIVE_IDXS] = 1.0
    
    fake_masked = fake_p1 * mask
    real_masked = real_p1 * mask
    
    # Coord Loss
    loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
    # Bone Loss
    f_p = fake_masked.permute(0, 2, 3, 1)
    r_p = real_masked.permute(0, 2, 3, 1)
    loss_bone = 0
    for (j1, j2) in BONES:
        f_l = torch.norm(f_p[:, :, j1, :] - f_p[:, :, j2, :], dim=-1)
        r_l = torch.norm(r_p[:, :, j1, :] - r_p[:, :, j2, :], dim=-1)
        loss_bone += torch.mean((f_l - r_l) ** 2)
        
    # Vel Loss
    fake_vel = f_p[:, 1:] - f_p[:, :-1]
    real_vel = r_p[:, 1:] - r_p[:, :-1]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from model.HPI_GCN_OP import Model as Discriminator
from model.generator import MotionGenerator

print("DEBUG: Script started", flush=True)

# --- CONFIG ---
DATA_PATH = "data/brace/BRACE_centered_aug.npz"
DISCRIMINATOR_PATH = "pretrained_models/brace_finetuned.pt"
SAVE_DIR = "pretrained_models/generator_checkpoints_final"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
LR = 0.0002
EPOCHS = 60
LATENT_DIM = 128

# BALANCED WEIGHTS for CENTERED DATA
# Since data is centered, Coord loss is naturally smaller and cleaner.
LAMBDA_CLASS = 2.0   
LAMBDA_COORD = 5.0   # Relaxed: Stop forcing it to be a statue
LAMBDA_BONE  = 10.0  # Moderate: Maintain structure but allow flex
LAMBDA_VEL   = 5.0   # Increased: Force smooth motion (No Jitter)
LAMBDA_DIV   = 5.0   # Force variety

ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
BONES = [(4, 5), (5, 6), (8, 9), (9, 10), (4, 8), (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)]

class BraceDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = data['x_data']
        self.y = data['y_data']
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        data = torch.from_numpy(self.x[idx]).float()
        C, T, V, M = data.shape
        data_padded = torch.zeros(C, T, V, 2)
        data_padded[:, :, :, 0] = data[:, :, :, 0]
        return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

def calculate_losses(fake, real):
    fake_p1 = fake[:, :, :, :, 0]
    real_p1 = real[:, :, :, :, 0]
    
    # Mask: Only calculate loss on joints that exist in COCO
    mask = torch.zeros_like(fake_p1)
    mask[:, :, :, ACTIVE_IDXS] = 1.0
    
    fake_masked = fake_p1 * mask
    real_masked = real_p1 * mask
    
    # Coord Loss
    loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
    # Bone Loss
    f_p = fake_masked.permute(0, 2, 3, 1)
    r_p = real_masked.permute(0, 2, 3, 1)
    loss_bone = 0
    for (j1, j2) in BONES:
        f_l = torch.norm(f_p[:, :, j1, :] - f_p[:, :, j2, :], dim=-1)
        r_l = torch.norm(r_p[:, :, j1, :] - r_p[:, :, j2, :], dim=-1)
        loss_bone += torch.mean((f_l - r_l) ** 2)
        
    # Vel Loss
    fake_vel = f_p[:, 1:] - f_p[:, :-1]
    real_vel = r_p[:, 1:] - r_p[:, :-1]
    loss_vel = torch.mean((fake_vel - real_vel)**2) # Force it to move like Real Data
    
    return loss_coord, loss_bone, loss_vel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Device: {device}", flush=True)
    
    # Resume Logic
    start_epoch = 0
    resume_path = os.path.join(SAVE_DIR, "generator_clean_ep15.pt") # Hardcoded based on user request/findings
    
    print("DEBUG: Initializing Dataset...", flush=True)
    dataset = BraceDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("DEBUG: Dataset Initialized", flush=True)
    
    print("DEBUG: Initializing Generator...", flush=True)
    generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    
    if os.path.exists(resume_path):
        print(f"DEBUG: Resuming from {resume_path}", flush=True)
        generator.load_state_dict(torch.load(resume_path))
        start_epoch = 15
    else:
        print("DEBUG: No checkpoint found, starting fresh", flush=True)

    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    
    print("DEBUG: Initializing Discriminator...", flush=True)
    critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    critic.fc = nn.Linear(256, 3)
    critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
    critic = critic.to(device)
    for p in critic.parameters(): p.requires_grad = False
    critic.eval()
    print("DEBUG: Discriminator Loaded", flush=True)
    
    ce_loss = nn.CrossEntropyLoss()
    
    print(f"DEBUG: Starting Loop from epoch {start_epoch+1}...", flush=True)
    for epoch in range(start_epoch, EPOCHS):
        t_loss = 0
        for i, (real, label) in enumerate(dataloader):
            real, label = real.to(device), label.to(device)
            
            z1 = torch.randn(real.size(0), LATENT_DIM).to(device)
            z2 = torch.randn(real.size(0), LATENT_DIM).to(device)
            
            fake1 = generator(z1, label)
            fake2 = generator(z2, label)
            
            fake_pad1 = torch.zeros_like(real); fake_pad1[:,:,:,:,0] = fake1[:,:,:,:,0]
            
            l_style = ce_loss(critic(fake_pad1), label)
            l_coord, l_bone, l_vel = calculate_losses(fake_pad1, real)
            
            diff_mot = torch.mean(torch.abs(fake1 - fake2))
            l_div = -diff_mot
            
            loss = (LAMBDA_CLASS * l_style) + (LAMBDA_COORD * l_coord) + (LAMBDA_BONE * l_bone) + (LAMBDA_VEL * l_vel) + (LAMBDA_DIV * l_div)
            
            optimizer_G.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0) # Clip Gradients
            optimizer_G.step()
            t_loss += loss.item()
            
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}: Loss {t_loss/len(dataloader):.4f}", flush=True)
            torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_clean_ep{epoch+1}.pt"))
            
    print("✅ Clean Training Complete.", flush=True)

if __name__ == "__main__":
    main()
