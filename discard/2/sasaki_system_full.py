import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. STILLNESS LAYER (Fixed Dimensions)
# ==========================================
class StillnessLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, skeleton_seq):
        # skeleton_seq: (Batch, 3, Frames, 25, 1)
        
        # Calculate Velocity (Frame-to-Frame difference)
        diff = skeleton_seq[:, :, 1:, :, :] - skeleton_seq[:, :, :-1, :, :]
        
        # Calculate Magnitude per sample
        # Norm over channels(1), Sum over Frames(2), Joints(3), Person(4)
        # Result shape: (Batch,)
        velocity = torch.norm(diff, dim=1).sum(dim=(1,2,3))
        
        # Calculate Stillness
        global_stillness = 1.0 / (velocity + 1e-4)
        global_stillness = torch.sigmoid(global_stillness)
        
        return global_stillness

# ==========================================
# 2. MOTION LAYER (Fixed Entropy Calculation)
# ==========================================
class MotionLayer(nn.Module):
    def __init__(self, pretrained_discriminator):
        super().__init__()
        self.encoder = pretrained_discriminator
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_context_entropy(self, skeleton_seq):
        """
        Calculates Shannon Entropy per sample in the batch.
        """
        with torch.no_grad():
            # 1. Flatten the data per sample
            # (Batch, 3, 64, 25, 1) -> (Batch, 4800)
            batch_size = skeleton_seq.size(0)
            flat_data = skeleton_seq.view(batch_size, -1)
            
            # 2. Calculate Variance PER SAMPLE (dim=1)
            # This fixes the scalar crash. We get a vector of size (Batch,)
            variance = torch.var(flat_data, dim=1)
            
            # 3. Map to Entropy
            entropy = torch.tanh(variance * 10) * 1.5
            return entropy

# ==========================================
# 3. IMPROVISATION SLM
# ==========================================
class SasakiSLM(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, entropy, stillness_level, current_state_idx=None):
        # entropy shape: (Batch,)
        batch_size = entropy.shape[0] 
        decisions = []
        
        for i in range(batch_size):
            H = entropy[i].item()
            S = stillness_level[i].item()
            
            # --- LOGIC ---
            if H > 1.1:
                probs = torch.tensor([0.2, 0.2, 0.6]) # Hold
            elif S > 0.8:
                probs = torch.tensor([0.7, 0.2, 0.1]) # Advance
            else:
                probs = torch.tensor([0.33, 0.33, 0.33]) # Random

            # Sample decision
            decision = torch.multinomial(probs, 1)
            
            # One-Hot Encode
            vec = torch.zeros(3)
            vec[decision] = 1.0
            decisions.append(vec)
            
        return torch.stack(decisions) # (Batch, 3)

# ==========================================
# 4. COHERENCE LAYER
# ==========================================
class CoherenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, generated_motion, intent_vector):
        diff = generated_motion[:, :, 1:, :, :] - generated_motion[:, :, :-1, :, :]
        
        loss = 0.0
        
        # HOLD CHECK
        hold_mask = intent_vector[:, 2] == 1.0
        if hold_mask.sum() > 0:
            hold_loss = torch.mean(torch.norm(diff[hold_mask], dim=1)) 
            loss += hold_loss * 2.0 

        # ADVANCE CHECK
        adv_mask = intent_vector[:, 0] == 1.0
        if adv_mask.sum() > 0:
            adv_velocity = torch.mean(torch.norm(diff[adv_mask], dim=1))
            adv_loss = 1.0 / (adv_velocity + 0.1)
            loss += adv_loss * 0.5

        return loss

# ==========================================
# THE FULL SYSTEM
# ==========================================
class SasakiSystem(nn.Module):
    def __init__(self, generator_model, discriminator_model):
        super().__init__()
        self.stillness_layer = StillnessLayer()
        self.motion_layer = MotionLayer(discriminator_model)
        self.slm = SasakiSLM()
        self.generator = generator_model
        
    def forward(self, z, labels, prev_motion):
        # 1. SENSE (Now returns Vectors, not Scalars)
        entropy = self.motion_layer.get_context_entropy(prev_motion)
        stillness = self.stillness_layer(prev_motion)
        
        # 2. DECIDE
        intent_vector = self.slm(entropy, stillness)
        intent_vector = intent_vector.to(z.device)
        
        # 3. ACT
        generated_skeleton = self.generator(z, labels, intent_vector)
        
        return generated_skeleton, intent_vector
