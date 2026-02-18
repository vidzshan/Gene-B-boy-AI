import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# --- IMPORTS ---
from model.HPI_GCN_OP import Model
from generator import SasakiGenerator
from train_brace_final import BraceDataset, load_and_split_data

# ================= CONFIG =================
GEN_PATH = './pretrained_models/generator_checkpoints_ac/sasaki_gen_ep60.pt' # Check your folder for the latest file
DISC_PATH = './pretrained_models/brace_final_model.pt'
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ForensicAuditor:
    def __init__(self):
        self.device = DEVICE
        self._load_models()
        self.features_cache = []
        
        # --- FIX: USE .fc INSTEAD OF .fcn ---
        # We attach a "Wiretap" (Hook) to the final classification layer.
        # We want to steal the data *before* it becomes a label.
        self.discriminator.fc.register_forward_hook(self.hook_fn)

    def _load_models(self):
        print("--- Loading Models for Forensics ---")
        # Generator
        self.generator = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(self.device)
        try:
            self.generator.load_state_dict(torch.load(GEN_PATH, map_location=self.device))
            print(f"✅ Generator loaded: {GEN_PATH}")
        except FileNotFoundError:
            print(f"❌ Error: Generator file not found at {GEN_PATH}")
            exit()
        self.generator.eval()
        
        # Discriminator
        self.discriminator = Model(num_class=3, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True).to(self.device)
        self.discriminator.load_state_dict(torch.load(DISC_PATH, map_location=self.device))
        self.discriminator.eval()

    def hook_fn(self, module, input, output):
        # input[0] contains the High-Level Features just before classification
        # Shape is usually (Batch, Feature_Size)
        self.features_cache.append(input[0].detach().cpu().numpy())

    def get_features(self, inputs):
        self.features_cache = [] # Clear cache
        with torch.no_grad():
            _ = self.discriminator(inputs)
        # Concatenate batch
        return np.concatenate(self.features_cache, axis=0)

    def calculate_frechet_distance(self, real_feats, fake_feats):
        """
        The GOLD STANDARD for GAN Evaluation (FMD).
        """
        mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def analyze_joint_covariance(self, skeleton_data):
        """
        Deep Analysis of Joint Correlation.
        """
        # Data: (Batch, 3, 64, 25, 1) -> (Batch*Time, 25, 3)
        B, C, T, V, M = skeleton_data.shape
        data_flat = skeleton_data.permute(0, 2, 3, 1, 4).reshape(-1, V, C).cpu().numpy()
        
        # Calculate velocity magnitude for every joint
        # (N, 25)
        velocity = np.linalg.norm(data_flat, axis=2)
        
        # Calculate Correlation Matrix (25x25)
        cov_matrix = np.corrcoef(velocity, rowvar=False)
        return cov_matrix

    def run_full_audit(self):
        print("--- 1. Gathering Real Data Features ---")
        X_train, _, _, _ = load_and_split_data()
        
        # Take a subset of 100 samples to save time and RAM
        subset_size = min(100, len(X_train))
        real_sample = torch.from_numpy(X_train[:subset_size]).float().to(self.device)
        
        real_features = self.get_features(real_sample)
        # Flatten: (100, 256, 1, 1) -> (100, 256) or similar
        real_features = real_features.reshape(real_features.shape[0], -1)

        print("--- 2. Gathering Generator Features ---")
        z = torch.randn(subset_size, LATENT_DIM).to(self.device)
        label = torch.tensor([[1.0, 0.0, 0.0]]).repeat(subset_size, 1).to(self.device) # Toprock
        intent = torch.tensor([[1.0, 0.0, 0.0]]).repeat(subset_size, 1).to(self.device) # Attack
        
        with torch.no_grad():
            fake_sample = self.generator(z, label, intent)
        
        fake_features = self.get_features(fake_sample)
        fake_features = fake_features.reshape(fake_features.shape[0], -1)

        # --- ANALYSIS 1: FRECHET DISTANCE ---
        fid_score = self.calculate_frechet_distance(real_features, fake_features)
        print(f"\n>>> FRECHET MOTION DISTANCE (FMD): {fid_score:.4f}")
        print("    (Lower is Better. 0-50: Great, >200: Poor)")

        # --- ANALYSIS 2: MANIFOLD VISUALIZATION (PCA) ---
        print("\n--- Visualizing Manifold Space ---")
        pca = PCA(n_components=2)
        # Ensure equal sizes for fitting
        min_len = min(len(real_features), len(fake_features))
        combined = np.concatenate([real_features[:min_len], fake_features[:min_len]], axis=0)
        
        pca.fit(combined)
        real_pca = pca.transform(real_features[:min_len])
        fake_pca = pca.transform(fake_features[:min_len])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.5, label='Real Motion')
        plt.scatter(fake_pca[:, 0], fake_pca[:, 1], c='red', alpha=0.5, label='Generated Motion')
        plt.legend()
        plt.title("The 'Brain' of the Discriminator: Real vs Fake Manifold")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

        # --- ANALYSIS 3: JOINT COVARIANCE MATRIX ---
        print("\n--- Visualizing Joint Structural Integrity ---")
        real_cov = self.analyze_joint_covariance(real_sample)
        fake_cov = self.analyze_joint_covariance(fake_sample)
        
        # Handle potential NaNs in correlation if variance is 0
        real_cov = np.nan_to_num(real_cov)
        fake_cov = np.nan_to_num(fake_cov)
        
        diff_cov = np.abs(real_cov - fake_cov)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.heatmap(real_cov, ax=ax1, cmap="coolwarm", vmin=-1, vmax=1)
        ax1.set_title("Real Joint Correlation")
        
        sns.heatmap(fake_cov, ax=ax2, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_title("Generator Joint Correlation")
        
        sns.heatmap(diff_cov, ax=ax3, cmap="Reds", vmin=0, vmax=1)
        ax3.set_title("Mismatch (Dark Red = Broken Physics)")
        
        plt.show()
        
        print("\n--- DIAGNOSIS REPORT ---")
        avg_diff = np.mean(diff_cov)
        print(f"Average Structural Deviation: {avg_diff:.4f}")

if __name__ == "__main__":
    auditor = ForensicAuditor()
    auditor.run_full_audit()