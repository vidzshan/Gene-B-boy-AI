# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.optim as optim
# # # # # # from torch.utils.data import DataLoader, Dataset
# # # # # # import numpy as np
# # # # # # import os

# # # # # # # Import models
# # # # # # from model.HPI_GCN_OP import Model as Discriminator
# # # # # # from model.generator import MotionGenerator

# # # # # # # ================= CONFIG =================
# # # # # # DATA_PATH = r"data/brace/BRACE_augmented.npz"
# # # # # # DISCRIMINATOR_PATH = r"pretrained_models/brace_finetuned.pt"
# # # # # # SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# # # # # # os.makedirs(SAVE_DIR, exist_ok=True)

# # # # # # BATCH_SIZE = 16
# # # # # # LR = 0.0002
# # # # # # EPOCHS = 50 # Reduced to 50 because we want to see results fast
# # # # # # LATENT_DIM = 128

# # # # # # # # --- THE "UNFREEZE" WEIGHTS (ADJUSTED FOR MOVEMENT) ---
# # # # # # # LAMBDA_CLASS = 5.0   # <--- INCREASED: Listen to the Critic!
# # # # # # # LAMBDA_COORD = 0.01  # <--- SLASHED: Stop trying to match exact pixels. Just dance.
# # # # # # # LAMBDA_BONE  = 2.0   # Keep this to prevent rubber limbs.
# # # # # # # LAMBDA_VEL   = 0.1   # <--- SLASHED: Allow the dancer to move fast.

# # # # # # # --- THE "HUMAN STRUCTURE" CONFIG ---
# # # # # # LAMBDA_CLASS = 1.0   # Low: Focus less on "Style" for now.
# # # # # # LAMBDA_COORD = 10.0  # VERY HIGH: You MUST stand where the human stands.
# # # # # # LAMBDA_BONE  = 20.0  # EXTREME: If you stretch a bone, you get punished hard.
# # # # # # LAMBDA_VEL   = 0.5   # Low: You are allowed to move, just don't jitter.
# # # # # # #We are telling the AI: "You must look like a human (High Coord/Bone), but you are free to move (Low Vel)."



# # # # # # # ACTIVE JOINTS (COCO Mapped indices)
# # # # # # ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

# # # # # # # Bone Definitions for Active Joints
# # # # # # BONES = [
# # # # # #     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
# # # # # #     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# # # # # # ]

# # # # # # # ================= HELPER =================
# # # # # # def calculate_losses(fake, real, device):
# # # # # #     # fake/real shape: [Batch, 3, 64, 25, 2]
    
# # # # # #     # Extract the first person (the actual dancer)
# # # # # #     fake_p1 = fake[:, :, :, :, 0] # [Batch, 3, 64, 25]
# # # # # #     real_p1 = real[:, :, :, :, 0] # [Batch, 3, 64, 25]
    
# # # # # #     # 1. CREATE MASK
# # # # # #     # We only want to calculate error on the Active Joints
# # # # # #     mask = torch.zeros_like(fake_p1)
# # # # # #     mask[:, :, :, ACTIVE_IDXS] = 1.0
    
# # # # # #     # Apply Mask
# # # # # #     fake_masked = fake_p1 * mask
# # # # # #     real_masked = real_p1 * mask
    
# # # # # #     # A. Coordinate Loss (Masked)
# # # # # #     loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
# # # # # #     # B. Bone Loss (Only Active Bones)
# # # # # #     # Shape: [B, 3, T, V] -> [B, T, V, 3]
# # # # # #     fake_p = fake_masked.permute(0, 2, 3, 1)
# # # # # #     real_p = real_masked.permute(0, 2, 3, 1)
    
# # # # # #     loss_bone = 0
# # # # # #     for (j1, j2) in BONES:
# # # # # #         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
# # # # # #         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
# # # # # #         loss_bone += torch.mean((f_len - r_len) ** 2)
        
# # # # # #     # C. Velocity Loss (Smoothness)
# # # # # #     # Penalize jerky movements (large changes between frames)
# # # # # #     # vel = pos[t] - pos[t-1]
# # # # # #     fake_vel = fake_p[:, 1:, :, :] - fake_p[:, :-1, :, :]
# # # # # #     loss_vel = torch.mean(fake_vel ** 2)
    
# # # # # #     return loss_coord, loss_bone, loss_vel

# # # # # # # ================= DATASET =================
# # # # # # class BraceDataset(Dataset):
# # # # # #     def __init__(self, data_path):
# # # # # #         data = np.load(data_path)
# # # # # #         self.x = data['x_data']
# # # # # #         self.y = data['y_data']
# # # # # #     def __len__(self): return len(self.y)
# # # # # #     def __getitem__(self, idx):
# # # # # #         data = torch.from_numpy(self.x[idx]).float()
# # # # # #         C, T, V, M = data.shape
# # # # # #         data_padded = torch.zeros(C, T, V, 2)
# # # # # #         data_padded[:, :, :, 0] = data[:, :, :, 0]
# # # # # #         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # # # # # # ================= MAIN =================
# # # # # # def main():
# # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # #     print(f"💎 Starting FINAL POLISHED Training on {device}...")

# # # # # #     dataset = BraceDataset(DATA_PATH)
# # # # # #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# # # # # #     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
# # # # # #     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

# # # # # #     critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
# # # # # #     critic.fc = nn.Linear(256, 3)
# # # # # #     critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
# # # # # #     critic = critic.to(device)
# # # # # #     for param in critic.parameters(): param.requires_grad = False
# # # # # #     critic.eval()

# # # # # #     criterion_class = nn.CrossEntropyLoss()

# # # # # #     for epoch in range(EPOCHS):
# # # # # #         total_loss = 0
        
# # # # # #         for i, (real_motion, real_labels) in enumerate(dataloader):
# # # # # #             real_motion = real_motion.to(device)
# # # # # #             real_labels = real_labels.to(device)
# # # # # #             batch_size = real_motion.size(0)
# # # # # #             optimizer_G.zero_grad()

# # # # # #             z = torch.randn(batch_size, LATENT_DIM).to(device)
# # # # # #             fake_motion = generator(z, real_labels)
            
# # # # # #             fake_padded = torch.zeros_like(real_motion)
# # # # # #             fake_padded[:, :, :, :, 0] = fake_motion[:, :, :, :, 0]

# # # # # #             # 1. Style Loss
# # # # # #             critic_pred = critic(fake_padded)
# # # # # #             loss_style = criterion_class(critic_pred, real_labels)
            
# # # # # #             # 2. Physical Losses (Masked + Smoothness)
# # # # # #             loss_coord, loss_bone, loss_vel = calculate_losses(fake_padded, real_motion, device)

# # # # # #             # Combine
# # # # # #             loss = (LAMBDA_CLASS * loss_style) + \
# # # # # #                    (LAMBDA_COORD * loss_coord) + \
# # # # # #                    (LAMBDA_BONE * loss_bone) + \
# # # # # #                    (LAMBDA_VEL * loss_vel)

# # # # # #             loss.backward()
# # # # # #             torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
# # # # # #             optimizer_G.step()
            
# # # # # #             total_loss += loss.item()

# # # # # #         if (epoch+1) % 10 == 0:
# # # # # #             print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")
# # # # # #             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

# # # # # #     print("✅ Final Training Complete.")

# # # # # # if __name__ == "__main__":
# # # # # #     main()


# # # # # # Physics First

# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from torch.utils.data import DataLoader, Dataset
# # # # # import numpy as np
# # # # # import os

# # # # # from model.generator import MotionGenerator

# # # # # # ================= CONFIG =================
# # # # # DATA_PATH = r"data/brace/BRACE_augmented.npz"
# # # # # SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# # # # # os.makedirs(SAVE_DIR, exist_ok=True)

# # # # # BATCH_SIZE = 16
# # # # # LR = 0.001 # Increased LR because we removed the Discriminator (faster learning)
# # # # # EPOCHS = 50
# # # # # LATENT_DIM = 128

# # # # # # --- PHYSICS ONLY CONFIG ---
# # # # # LAMBDA_CLASS = 0.0   # <--- TURNED OFF: Ignore the critic.
# # # # # LAMBDA_COORD = 10.0  # High: Copy the real data positions.
# # # # # LAMBDA_BONE  = 5.0   # Moderate: Maintain structure.
# # # # # LAMBDA_VEL   = 1.0   # Moderate: Copy the speed of real dancers.

# # # # # ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
# # # # # BONES = [
# # # # #     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
# # # # #     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# # # # # ]

# # # # # # ================= DATASET =================
# # # # # class BraceDataset(Dataset):
# # # # #     def __init__(self, data_path):
# # # # #         data = np.load(data_path)
# # # # #         self.x = data['x_data']
# # # # #         self.y = data['y_data']
# # # # #     def __len__(self): return len(self.y)
# # # # #     def __getitem__(self, idx):
# # # # #         data = torch.from_numpy(self.x[idx]).float()
# # # # #         C, T, V, M = data.shape
# # # # #         data_padded = torch.zeros(C, T, V, 2)
# # # # #         data_padded[:, :, :, 0] = data[:, :, :, 0]
# # # # #         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # # # # # ================= HELPER =================
# # # # # def calculate_losses(fake, real):
# # # # #     fake_p1 = fake[:, :, :, :, 0]
# # # # #     real_p1 = real[:, :, :, :, 0]
    
# # # # #     mask = torch.zeros_like(fake_p1)
# # # # #     mask[:, :, :, ACTIVE_IDXS] = 1.0
    
# # # # #     fake_masked = fake_p1 * mask
# # # # #     real_masked = real_p1 * mask
    
# # # # #     # Coordinate Loss
# # # # #     loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
# # # # #     # Bone Loss
# # # # #     fake_p = fake_masked.permute(0, 2, 3, 1)
# # # # #     real_p = real_masked.permute(0, 2, 3, 1)
# # # # #     loss_bone = 0
# # # # #     for (j1, j2) in BONES:
# # # # #         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
# # # # #         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
# # # # #         loss_bone += torch.mean((f_len - r_len) ** 2)
        
# # # # #     # Velocity Loss
# # # # #     fake_vel = fake_p[:, 1:, :, :] - fake_p[:, :-1, :, :]
# # # # #     real_vel = real_p[:, 1:, :, :] - real_p[:, :-1, :, :]
# # # # #     loss_vel = torch.mean((fake_vel - real_vel) ** 2) # Match Real Velocity!
    
# # # # #     return loss_coord, loss_bone, loss_vel

# # # # # # ================= MAIN =================
# # # # # def main():
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     print(f"🚑 Starting PHYSICS REPAIR (No Discriminator) on {device}...")

# # # # #     dataset = BraceDataset(DATA_PATH)
# # # # #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# # # # #     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
# # # # #     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

# # # # #     # Note: We DO NOT load the Discriminator. We don't need it yet.

# # # # #     for epoch in range(EPOCHS):
# # # # #         total_loss = 0
# # # # #         d_coord, d_bone, d_vel = 0, 0, 0
        
# # # # #         for i, (real_motion, real_labels) in enumerate(dataloader):
# # # # #             real_motion = real_motion.to(device)
# # # # #             real_labels = real_labels.to(device)
# # # # #             batch_size = real_motion.size(0)
# # # # #             optimizer_G.zero_grad()

# # # # #             z = torch.randn(batch_size, LATENT_DIM).to(device)
# # # # #             fake_motion = generator(z, real_labels)
            
# # # # #             fake_padded = torch.zeros_like(real_motion)
# # # # #             fake_padded[:, :, :, :, 0] = fake_motion[:, :, :, :, 0]

# # # # #             # PURE PHYSICS LOSS
# # # # #             loss_coord, loss_bone, loss_vel = calculate_losses(fake_padded, real_motion)

# # # # #             loss = (LAMBDA_COORD * loss_coord) + \
# # # # #                    (LAMBDA_BONE * loss_bone) + \
# # # # #                    (LAMBDA_VEL * loss_vel)

# # # # #             loss.backward()
# # # # #             optimizer_G.step()
            
# # # # #             total_loss += loss.item()
# # # # #             d_coord += loss_coord.item()
# # # # #             d_bone += loss_bone.item()

# # # # #         # Logging Breakdown
# # # # #         avg_loss = total_loss / len(dataloader)
# # # # #         avg_coord = d_coord / len(dataloader)
# # # # #         avg_bone = d_bone / len(dataloader)

# # # # #         if (epoch+1) % 5 == 0:
# # # # #             print(f"Epoch [{epoch+1}/{EPOCHS}] Total: {avg_loss:.4f} | Coord: {avg_coord:.4f} | Bone: {avg_bone:.4f}")
# # # # #             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

# # # # #     print("✅ Physics Repair Complete.")

# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # #turn on discriminator
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # from torch.utils.data import DataLoader, Dataset
# # # # import numpy as np
# # # # import os

# # # # from model.HPI_GCN_OP import Model as Discriminator
# # # # from model.generator import MotionGenerator

# # # # # ================= CONFIG =================
# # # # DATA_PATH = r"data/brace/BRACE_augmented.npz"
# # # # DISCRIMINATOR_PATH = r"pretrained_models/brace_finetuned.pt"
# # # # # WE LOAD THE "STATUE" WEIGHTS TO START
# # # # START_WEIGHTS = r"pretrained_models/generator_checkpoints_final/generator_final_ep50.pt"

# # # # SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# # # # os.makedirs(SAVE_DIR, exist_ok=True)

# # # # BATCH_SIZE = 16
# # # # LR = 0.0002 
# # # # EPOCHS = 50
# # # # LATENT_DIM = 128

# # # # # --- THE "WAKE UP" CONFIG ---
# # # # LAMBDA_CLASS = 2.0   # <--- TURNED ON: "Look like a Toprock move!"
# # # # LAMBDA_COORD = 1.0   # Lowered from 10.0: You are free to move away from the average.
# # # # LAMBDA_BONE  = 5.0   # Kept Moderate: Don't break the bones we just fixed.
# # # # LAMBDA_VEL   = 0.1   # Low: Moving fast is allowed now.

# # # # ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
# # # # BONES = [
# # # #     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
# # # #     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# # # # ]

# # # # # ================= HELPER =================
# # # # def calculate_losses(fake, real):
# # # #     fake_p1 = fake[:, :, :, :, 0]
# # # #     real_p1 = real[:, :, :, :, 0]
    
# # # #     mask = torch.zeros_like(fake_p1)
# # # #     mask[:, :, :, ACTIVE_IDXS] = 1.0
    
# # # #     fake_masked = fake_p1 * mask
# # # #     real_masked = real_p1 * mask
    
# # # #     # Coordinate Loss
# # # #     loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
# # # #     # Bone Loss
# # # #     fake_p = fake_masked.permute(0, 2, 3, 1)
# # # #     real_p = real_masked.permute(0, 2, 3, 1)
# # # #     loss_bone = 0
# # # #     for (j1, j2) in BONES:
# # # #         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
# # # #         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
# # # #         loss_bone += torch.mean((f_len - r_len) ** 2)
        
# # # #     # Velocity Loss
# # # #     fake_vel = fake_p[:, 1:, :, :] - fake_p[:, :-1, :, :]
# # # #     loss_vel = torch.mean(fake_vel ** 2)
    
# # # #     return loss_coord, loss_bone, loss_vel

# # # # # ================= DATASET =================
# # # # class BraceDataset(Dataset):
# # # #     def __init__(self, data_path):
# # # #         data = np.load(data_path)
# # # #         self.x = data['x_data']
# # # #         self.y = data['y_data']
# # # #     def __len__(self): return len(self.y)
# # # #     def __getitem__(self, idx):
# # # #         data = torch.from_numpy(self.x[idx]).float()
# # # #         C, T, V, M = data.shape
# # # #         data_padded = torch.zeros(C, T, V, 2)
# # # #         data_padded[:, :, :, 0] = data[:, :, :, 0]
# # # #         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # # # # ================= MAIN =================
# # # # def main():
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     print(f"⏰ Waking up the Dancer on {device}...")

# # # #     dataset = BraceDataset(DATA_PATH)
# # # #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# # # #     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    
# # # #     # --- LOAD THE STATUE WEIGHTS ---
# # # #     print(f"Loading weights from {START_WEIGHTS}...")
# # # #     generator.load_state_dict(torch.load(START_WEIGHTS))
    
# # # #     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

# # # #     # LOAD DISCRIMINATOR (It is back!)
# # # #     critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
# # # #     critic.fc = nn.Linear(256, 3)
# # # #     critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
# # # #     critic = critic.to(device)
# # # #     for param in critic.parameters(): param.requires_grad = False
# # # #     critic.eval()

# # # #     criterion_class = nn.CrossEntropyLoss()

# # # #     for epoch in range(EPOCHS):
# # # #         total_loss = 0
        
# # # #         for i, (real_motion, real_labels) in enumerate(dataloader):
# # # #             real_motion = real_motion.to(device)
# # # #             real_labels = real_labels.to(device)
# # # #             batch_size = real_motion.size(0)
# # # #             optimizer_G.zero_grad()

# # # #             z = torch.randn(batch_size, LATENT_DIM).to(device)
# # # #             fake_motion = generator(z, real_labels)
            
# # # #             fake_padded = torch.zeros_like(real_motion)
# # # #             fake_padded[:, :, :, :, 0] = fake_motion[:, :, :, :, 0]

# # # #             # 1. Style Loss
# # # #             critic_pred = critic(fake_padded)
# # # #             loss_style = criterion_class(critic_pred, real_labels)
            
# # # #             # 2. Physical Losses
# # # #             loss_coord, loss_bone, loss_vel = calculate_losses(fake_padded, real_motion)

# # # #             loss = (LAMBDA_CLASS * loss_style) + \
# # # #                    (LAMBDA_COORD * loss_coord) + \
# # # #                    (LAMBDA_BONE * loss_bone) + \
# # # #                    (LAMBDA_VEL * loss_vel)

# # # #             loss.backward()
# # # #             optimizer_G.step()
            
# # # #             total_loss += loss.item()

# # # #         if (epoch+1) % 5 == 0:
# # # #             print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")
# # # #             # Overwrite the checkpoints or save new ones
# # # #             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

# # # #     print("✅ Wake Up Training Complete.")

# # # # if __name__ == "__main__":
# # # #     main()


# # # #Listen to the discriminator
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader, Dataset
# # # import numpy as np
# # # import os

# # # from model.HPI_GCN_OP import Model as Discriminator
# # # from model.generator import MotionGenerator

# # # # ================= CONFIG =================
# # # DATA_PATH = r"data/brace/BRACE_augmented.npz"
# # # DISCRIMINATOR_PATH = r"pretrained_models/brace_finetuned.pt"

# # # # LOAD YOUR LATEST "STATUE" WEIGHTS (From the Physics Repair run)
# # # START_WEIGHTS = r"pretrained_models/generator_checkpoints_final/generator_final_ep50.pt"

# # # SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# # # os.makedirs(SAVE_DIR, exist_ok=True)

# # # BATCH_SIZE = 16
# # # LR = 0.0002
# # # EPOCHS = 50
# # # LATENT_DIM = 128

# # # # --- THE "UNLEASHED" CONFIG ---
# # # LAMBDA_CLASS = 10.0  # High: Force the model to satisfy the style Critic
# # # LAMBDA_COORD = 0.0   # Off: Stop enforcing exact pixel matching
# # # LAMBDA_VEL   = 0.0   # Off: Allow fast movement
# # # LAMBDA_BONE  = 5.0   # Moderate: Keep the skeleton together

# # # ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
# # # BONES = [
# # #     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
# # #     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# # # ]

# # # # ================= HELPER =================
# # # def calculate_losses(fake, real):
# # #     fake_p1 = fake[:, :, :, :, 0]
# # #     real_p1 = real[:, :, :, :, 0]
    
# # #     mask = torch.zeros_like(fake_p1)
# # #     mask[:, :, :, ACTIVE_IDXS] = 1.0
# # #     fake_masked = fake_p1 * mask
# # #     real_masked = real_p1 * mask
    
# # #     # Coordinate Loss
# # #     loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
# # #     # Bone Loss
# # #     fake_p = fake_masked.permute(0, 2, 3, 1)
# # #     real_p = real_masked.permute(0, 2, 3, 1)
# # #     loss_bone = 0
# # #     for (j1, j2) in BONES:
# # #         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
# # #         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
# # #         loss_bone += torch.mean((f_len - r_len) ** 2)
        
# # #     # Velocity Loss
# # #     fake_vel = fake_p[:, 1:, :, :] - fake_p[:, :-1, :, :]
# # #     loss_vel = torch.mean(fake_vel ** 2)
    
# # #     return loss_coord, loss_bone, loss_vel

# # # # ================= DATASET (RESTORED) =================
# # # class BraceDataset(Dataset):
# # #     def __init__(self, data_path):
# # #         data = np.load(data_path)
# # #         self.x = data['x_data']
# # #         self.y = data['y_data']
# # #     def __len__(self): return len(self.y)
# # #     def __getitem__(self, idx):
# # #         data = torch.from_numpy(self.x[idx]).float()
# # #         C, T, V, M = data.shape
# # #         data_padded = torch.zeros(C, T, V, 2)
# # #         data_padded[:, :, :, 0] = data[:, :, :, 0]
# # #         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # # # ================= MAIN =================
# # # def main():
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     print(f"🚀 UNLEASHING THE DANCER on {device}...")

# # #     dataset = BraceDataset(DATA_PATH)
# # #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# # #     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    
# # #     # Load the Statue Weights
# # #     if os.path.exists(START_WEIGHTS):
# # #         print(f"Loading Statue weights from {START_WEIGHTS}...")
# # #         generator.load_state_dict(torch.load(START_WEIGHTS))
# # #     else:
# # #         print("⚠️ WARNING: Could not find start weights. Starting from scratch.")

# # #     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

# # #     critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
# # #     critic.fc = nn.Linear(256, 3)
# # #     critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
# # #     critic = critic.to(device)
# # #     for param in critic.parameters(): param.requires_grad = False
# # #     critic.eval()

# # #     criterion_class = nn.CrossEntropyLoss()

# # #     for epoch in range(EPOCHS):
# # #         total_loss = 0
        
# # #         for i, (real_motion, real_labels) in enumerate(dataloader):
# # #             real_motion = real_motion.to(device)
# # #             real_labels = real_labels.to(device)
# # #             batch_size = real_motion.size(0)
# # #             optimizer_G.zero_grad()

# # #             z = torch.randn(batch_size, LATENT_DIM).to(device)
# # #             fake_motion = generator(z, real_labels)
# # #             fake_padded = torch.zeros_like(real_motion)
# # #             fake_padded[:, :, :, :, 0] = fake_motion[:, :, :, :, 0]

# # #             # 1. Style Loss (THE BOSS)
# # #             critic_pred = critic(fake_padded)
# # #             loss_style = criterion_class(critic_pred, real_labels)
            
# # #             # 2. Physical Losses
# # #             loss_coord, loss_bone, loss_vel = calculate_losses(fake_padded, real_motion)

# # #             loss = (LAMBDA_CLASS * loss_style) + \
# # #                    (LAMBDA_COORD * loss_coord) + \
# # #                    (LAMBDA_BONE * loss_bone) + \
# # #                    (LAMBDA_VEL * loss_vel)

# # #             loss.backward()
# # #             optimizer_G.step()
# # #             total_loss += loss.item()

# # #         if (epoch+1) % 5 == 0:
# # #             print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")
# # #             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

# # #     print("✅ Unleashed Training Complete.")

# # # if __name__ == "__main__":
# # #     main()

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, Dataset
# # import numpy as np
# # import os

# # from model.HPI_GCN_OP import Model as Discriminator
# # from model.generator import MotionGenerator

# # # ================= CONFIG =================
# # DATA_PATH = r"data/brace/BRACE_augmented.npz"
# # DISCRIMINATOR_PATH = r"pretrained_models/brace_finetuned.pt"
# # # LOAD YOUR LATEST WEIGHTS (The ones that gave you the images above)
# # START_WEIGHTS = r"pretrained_models/generator_checkpoints_final/generator_final_ep50.pt"

# # SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# # os.makedirs(SAVE_DIR, exist_ok=True)

# # BATCH_SIZE = 16
# # LR = 0.0002
# # EPOCHS = 50
# # LATENT_DIM = 128

# # # --- THE "FORCED MOVEMENT" CONFIG ---
# # LAMBDA_CLASS = 5.0   
# # LAMBDA_COORD = 0.0   # Keep 0. Let it wander.
# # LAMBDA_BONE  = 5.0   # Keep structure.
# # LAMBDA_VEL   = 0.0   
# # LAMBDA_DIV   = 10.0  # <--- NEW: Diversity Penalty (Don't be boring!)

# # ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
# # BONES = [
# #     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
# #     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# # ]

# # # ================= DATASET =================
# # class BraceDataset(Dataset):
# #     def __init__(self, data_path):
# #         data = np.load(data_path)
# #         self.x = data['x_data']
# #         self.y = data['y_data']
# #     def __len__(self): return len(self.y)
# #     def __getitem__(self, idx):
# #         data = torch.from_numpy(self.x[idx]).float()
# #         C, T, V, M = data.shape
# #         data_padded = torch.zeros(C, T, V, 2)
# #         data_padded[:, :, :, 0] = data[:, :, :, 0]
# #         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # # ================= HELPER =================
# # def calculate_losses(fake, real):
# #     fake_p1 = fake[:, :, :, :, 0]
# #     real_p1 = real[:, :, :, :, 0]
    
# #     mask = torch.zeros_like(fake_p1)
# #     mask[:, :, :, ACTIVE_IDXS] = 1.0
# #     fake_masked = fake_p1 * mask
# #     real_masked = real_p1 * mask
    
# #     # Bone Loss
# #     fake_p = fake_masked.permute(0, 2, 3, 1)
# #     real_p = real_masked.permute(0, 2, 3, 1)
# #     loss_bone = 0
# #     for (j1, j2) in BONES:
# #         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
# #         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
# #         loss_bone += torch.mean((f_len - r_len) ** 2)
        
# #     return 0.0, loss_bone, 0.0 # Return 0 for coord/vel since we turned them off

# # # ================= MAIN =================
# # def main():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"⚡ ADDING CHAOS (Diversity Training) on {device}...")

# #     dataset = BraceDataset(DATA_PATH)
# #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# #     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    
# #     if os.path.exists(START_WEIGHTS):
# #         print(f"Loading weights from {START_WEIGHTS}...")
# #         generator.load_state_dict(torch.load(START_WEIGHTS))
    
# #     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

# #     critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
# #     critic.fc = nn.Linear(256, 3)
# #     critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
# #     critic = critic.to(device)
# #     for param in critic.parameters(): param.requires_grad = False
# #     critic.eval()

# #     criterion_class = nn.CrossEntropyLoss()

# #     for epoch in range(EPOCHS):
# #         total_loss = 0
        
# #         for i, (real_motion, real_labels) in enumerate(dataloader):
# #             real_motion = real_motion.to(device)
# #             real_labels = real_labels.to(device)
# #             batch_size = real_motion.size(0)
# #             optimizer_G.zero_grad()

# #             # --- 1. Generate BATCH A ---
# #             z1 = torch.randn(batch_size, LATENT_DIM).to(device)
# #             fake_motion1 = generator(z1, real_labels)
            
# #             # --- 2. Generate BATCH B (Same labels, Different Noise) ---
# #             z2 = torch.randn(batch_size, LATENT_DIM).to(device)
# #             fake_motion2 = generator(z2, real_labels)
            
# #             # Pad for Discriminator
# #             fake_padded1 = torch.zeros_like(real_motion)
# #             fake_padded1[:, :, :, :, 0] = fake_motion1[:, :, :, :, 0]

# #             # --- LOSS CALCULATION ---
            
# #             # A. Style Loss (Using Batch A)
# #             critic_pred = critic(fake_padded1)
# #             loss_style = criterion_class(critic_pred, real_labels)
            
# #             # B. Bone Loss
# #             _, loss_bone, _ = calculate_losses(fake_padded1, real_motion)
            
# #             # C. DIVERSITY LOSS (The Magic Ingredient)
# #             # Calculate how different the inputs were
# #             diff_z = torch.mean(torch.abs(z1 - z2)) 
# #             # Calculate how different the outputs are
# #             diff_motion = torch.mean(torch.abs(fake_motion1 - fake_motion2))
            
# #             # If inputs are different (diff_z > 0), outputs MUST be different.
# #             # We maximize diff_motion, so we minimize -diff_motion
# #             # OR: loss = 1 / (diff_motion + epsilon)
# #             loss_div = -diff_motion

# #             # Combine
# #             loss = (LAMBDA_CLASS * loss_style) + \
# #                    (LAMBDA_BONE * loss_bone) + \
# #                    (LAMBDA_DIV * loss_div)

# #             loss.backward()
# #             optimizer_G.step()
# #             total_loss += loss.item()

# #         if (epoch+1) % 5 == 0:
# #             print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")
# #             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

# #     print("✅ Chaos Training Complete.")

# # if __name__ == "__main__":
# #     main()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import os

# from model.HPI_GCN_OP import Model as Discriminator
# from model.generator import MotionGenerator

# # ================= CONFIG =================
# DATA_PATH = r"data/brace/BRACE_augmented.npz"
# DISCRIMINATOR_PATH = r"pretrained_models/brace_finetuned.pt"

# # CHANGE THIS LINE IN THE SCRIPT
# START_WEIGHTS = r"pretrained_models/generator_checkpoints_final/generator_exploded.pt"

# SAVE_DIR = r"pretrained_models/generator_checkpoints_final"
# os.makedirs(SAVE_DIR, exist_ok=True)

# BATCH_SIZE = 16
# LR = 0.0001  # <--- SLOW DOWN! We need precision now.
# EPOCHS = 50
# LATENT_DIM = 128

# # --- THE "REPAIR KIT" CONFIG ---
# LAMBDA_CLASS = 2.0   # Keep style focus
# LAMBDA_COORD = 10.0  # <--- HIGH: Snaps the skeleton back to the box
# LAMBDA_BONE  = 50.0  # <--- MASSIVE: Forces the limbs to connect immediately
# LAMBDA_VEL   = 1.0   # Keep movement valid
# LAMBDA_DIV   = 1.0   # Low: Just enough to keep it alive, but stop the explosions

# ACTIVE_IDXS = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
# BONES = [
#     (4, 5), (5, 6), (8, 9), (9, 10), (4, 8),
#     (12, 13), (13, 14), (16, 17), (17, 18), (12, 16)
# ]

# # ================= DATASET =================
# class BraceDataset(Dataset):
#     def __init__(self, data_path):
#         data = np.load(data_path)
#         self.x = data['x_data']
#         self.y = data['y_data']
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx):
#         data = torch.from_numpy(self.x[idx]).float()
#         C, T, V, M = data.shape
#         data_padded = torch.zeros(C, T, V, 2)
#         data_padded[:, :, :, 0] = data[:, :, :, 0]
#         return data_padded, torch.tensor(self.y[idx], dtype=torch.long)

# # ================= HELPER =================
# def calculate_losses(fake, real):
#     fake_p1 = fake[:, :, :, :, 0]
#     real_p1 = real[:, :, :, :, 0]
    
#     mask = torch.zeros_like(fake_p1)
#     mask[:, :, :, ACTIVE_IDXS] = 1.0
#     fake_masked = fake_p1 * mask
#     real_masked = real_p1 * mask
    
#     # Coordinate Loss
#     loss_coord = torch.mean((fake_masked - real_masked) ** 2)
    
#     # Bone Loss
#     fake_p = fake_masked.permute(0, 2, 3, 1)
#     real_p = real_masked.permute(0, 2, 3, 1)
#     loss_bone = 0
#     for (j1, j2) in BONES:
#         f_len = torch.norm(fake_p[:, :, j1, :] - fake_p[:, :, j2, :], dim=-1)
#         r_len = torch.norm(real_p[:, :, j1, :] - real_p[:, :, j2, :], dim=-1)
#         loss_bone += torch.mean((f_len - r_len) ** 2)
        
#     # Velocity Loss
#     fake_vel = fake_p[:, 1:, :, :] - fake_p[:, :-1, :, :]
#     loss_vel = torch.mean(fake_vel ** 2)
    
#     return loss_coord, loss_bone, loss_vel

# # ================= MAIN =================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🔧 REPAIRING THE SKELETON on {device}...")

#     dataset = BraceDataset(DATA_PATH)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#     generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    
#     # Load the EXPLODING weights
#     if os.path.exists(START_WEIGHTS):
#         print(f"Loading Chaos weights from {START_WEIGHTS}...")
#         generator.load_state_dict(torch.load(START_WEIGHTS))
    
#     optimizer_G = optim.Adam(generator.parameters(), lr=LR)

#     critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
#     critic.fc = nn.Linear(256, 3)
#     critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
#     critic = critic.to(device)
#     for param in critic.parameters(): param.requires_grad = False
#     critic.eval()

#     criterion_class = nn.CrossEntropyLoss()

#     for epoch in range(EPOCHS):
#         total_loss = 0
        
#         for i, (real_motion, real_labels) in enumerate(dataloader):
#             real_motion = real_motion.to(device)
#             real_labels = real_labels.to(device)
#             batch_size = real_motion.size(0)
#             optimizer_G.zero_grad()

#             # Batch A
#             z1 = torch.randn(batch_size, LATENT_DIM).to(device)
#             fake_motion1 = generator(z1, real_labels)
            
#             # Batch B (For Diversity Check)
#             z2 = torch.randn(batch_size, LATENT_DIM).to(device)
#             fake_motion2 = generator(z2, real_labels)
            
#             fake_padded1 = torch.zeros_like(real_motion)
#             fake_padded1[:, :, :, :, 0] = fake_motion1[:, :, :, :, 0]

#             # 1. Style Loss
#             critic_pred = critic(fake_padded1)
#             loss_style = criterion_class(critic_pred, real_labels)
            
#             # 2. Physical Losses (THE REPAIR)
#             loss_coord, loss_bone, loss_vel = calculate_losses(fake_padded1, real_motion)
            
#             # 3. Diversity (Low weight now)
#             diff_motion = torch.mean(torch.abs(fake_motion1 - fake_motion2))
#             loss_div = -diff_motion

#             loss = (LAMBDA_CLASS * loss_style) + \
#                    (LAMBDA_COORD * loss_coord) + \
#                    (LAMBDA_BONE * loss_bone) + \
#                    (LAMBDA_VEL * loss_vel) + \
#                    (LAMBDA_DIV * loss_div)

#             loss.backward()
#             optimizer_G.step()
#             total_loss += loss.item()

#         if (epoch+1) % 5 == 0:
#             print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")
#             torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_final_ep{epoch+1}.pt"))

#     print("✅ Repair Training Complete.")

# if __name__ == "__main__":
#     main()


print("DEBUG: Script started", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from model.HPI_GCN_OP import Model as Discriminator
from model.generator import MotionGenerator

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
    print("🌟 STARTING CLEAN TRAINING...")
    
    dataset = BraceDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    generator = MotionGenerator(latent_dim=LATENT_DIM).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    
    critic = Discriminator(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    critic.fc = nn.Linear(256, 3)
    critic.load_state_dict(torch.load(DISCRIMINATOR_PATH))
    critic = critic.to(device)
    for p in critic.parameters(): p.requires_grad = False
    critic.eval()
    
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
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
            print(f"Epoch {epoch+1}: Loss {t_loss/len(dataloader):.4f}")
            torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_clean_ep{epoch+1}.pt"))
            
    print("✅ Clean Training Complete.")

if __name__ == "__main__":
    main()
