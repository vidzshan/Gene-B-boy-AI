import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# SASAKI IMPROVISATION SLM (Stochastic Logic Model)
# based on improvisation_slm.yml & improvisationdecision.yml
# ==========================================

class SasakiBrain(nn.Module):
    def __init__(self, skill_level=0.5):
        super().__init__()
        
        # STATES: 0=Advance (F), 1=Retreat (B), 2=Hold (S/Stillness)
        self.states = ['Advance', 'Retreat', 'Hold']
        self.current_state_idx = 0 # Default start: Advance
        
        # CONFIG
        self.skill_level = skill_level # Ability to control "Stillness"
        
        # BASE TRANSITION MATRIX (3x3)
        # Rows: From [F, B, S] -> Cols: To [F, B, S]
        # Initial bias: flowing motion (F->F) is common
        self.transition_matrix = torch.tensor([
            [0.5, 0.3, 0.2], # From Advance
            [0.4, 0.4, 0.2], # From Retreat
            [0.6, 0.1, 0.3]  # From Hold (Explosive exit from freeze)
        ])

    def compute_context_entropy(self, motion_sequence):
        """
        Calculates Shannon Entropy based on motion variance.
        High Variance = High Chaos = High Entropy.
        """
        # motion_sequence shape: (Batch, Frames, Joints, 3)
        if motion_sequence is None:
            return 0.5 # Default medium entropy
            
        # Calculate velocity (Frame-to-Frame difference)
        velocity = motion_sequence[:, 1:] - motion_sequence[:, :-1]
        
        # Variance as a proxy for 'Information Content'
        variance = torch.var(velocity)
        
        # Normalize to 0.0 - 1.5 range (approx log2(3))
        entropy = torch.tanh(variance) * 1.5
        return entropy.item()

    def decide_next_intent(self, motion_history):
        """
        The Core Logic: Decides F (Attack), B (Retreat), or S (Freeze).
        """
        H = self.compute_context_entropy(motion_history)
        
        # --- POLICY RULES (from improvisationdecision.yml) ---
        # 1. Low Entropy (Boring/Clean context) -> Confident Flow (Attack)
        if H < 0.6:
            action_probs = torch.tensor([0.45, 0.25, 0.30]) # Bias to Advance
            
        # 2. Mid Entropy (Confusing) -> Superposition (Random)
        elif 0.6 <= H < 1.1:
            action_probs = torch.tensor([0.34, 0.33, 0.33]) # Equal
            
        # 3. High Entropy (Chaos) -> Listen and Hold (Freeze)
        else:
            # Skill Check: Only skilled dancers can 'Freeze' effectively in chaos
            hold_prob = 0.50 * self.skill_level
            advance_prob = (1.0 - hold_prob) / 2
            action_probs = torch.tensor([advance_prob, advance_prob, hold_prob])

        # Apply Temperature (Entropy modulates randomness)
        # High entropy = High temp = More random
        temp = max(0.1, H) 
        logits = torch.log(action_probs + 1e-9) / temp
        probs = F.softmax(logits, dim=0)

        # Sample the decision
        next_state_idx = torch.multinomial(probs, 1).item()
        self.current_state_idx = next_state_idx
        
        intent = self.states[next_state_idx]
        
        # Return the Intent Vector (One-Hot Encoded) to feed the Generator
        # [1,0,0] = Advance, [0,1,0] = Retreat, [0,0,1] = Hold
        intent_vec = torch.zeros(3)
        intent_vec[next_state_idx] = 1.0
        
        return intent, intent_vec

# ==========================================
# TEST HARNESS (Verify Sasaki Logic)
# ==========================================
if __name__ == "__main__":
    brain = SasakiBrain(skill_level=0.8)
    
    # Simulate some motion input
    dummy_motion = torch.randn(1, 64, 25, 3) 
    
    print("--- SASAKI BRAIN TEST ---")
    for i in range(5):
        intent_name, intent_vec = brain.decide_next_intent(dummy_motion * (i*0.5)) # Increasing chaos
        print(f"Chaos Level {i}: Decision -> {intent_name} {intent_vec.tolist()}")
