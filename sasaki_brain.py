import torch
import numpy as np
import torch.nn.functional as F
import math
import random
from dataclasses import dataclass, field

# --- Partner's Core Theory Implementation ---
class KQIEngine:
    def __init__(self, theta=0.1): # Lowered threshold
        self.theta = theta

    def process(self, r, omega, sigma, t, rho):
        # SENSITIVITY CALIBRATION
        # Add a baseline 0.1 to G so the gate is never permanently locked
        G = (r + 0.1) * (omega + 0.1) * 2.0 
        safe_sigma = max(sigma, 0.1)
        Q = (G ** 2) / safe_sigma
        
        M = 1.0 if Q >= self.theta else 0.2 # Baseline 20% activity
        tau = t * rho
        C = math.exp(min(tau * 0.1, 2.0)) 
        
        # --- THE FIX: BASELINE DEPTH ---
        D = (Q * C * M) + 0.1 # Minimum depth 0.1 to keep dancer alive
        
        return {"G": G, "Q": Q, "M": M, "C": C, "D": D}

class MineLife:
    def __init__(self, w=10, h=10):
        self.w, self.h = w, h
        self.alive = [[1 if random.random() < 0.2 else 0 for _ in range(w)] for _ in range(h)]
        self.t = 0

    def tick(self):
        self.t += 1
        # Simplified cellular automata to simulate "Neural Firing"
        alive_ratio = sum(sum(r) for r in self.alive) / (self.w * self.h)
        novelty = abs(math.sin(self.t * 0.05)) # Slower curiosity drift
        return {"alive_ratio": alive_ratio, "novelty": novelty, "t": self.t}

class SasakiLiveBrain:
    def __init__(self):
        self.current_state = 0 # 0:TR, 1:FW, 2:PW, 3:IDLE
        self.kqi = KQIEngine(theta=0.1) # High sensitivity
        self.latent_will = np.random.randn(128)
        self.t = 0

    def decide_style(self, audio_vector):
        self.t += 1
        # 1. PERCEPTUAL ENERGY GATE
        energy = audio_vector[-1] # Assuming Onset is last
        
        # --- THE HEARTBEAT FIX ---
        # If energy is near 0, provide a fake 'Subconscious' pulse
        # This keeps the brain 'alive' even in silence
        if energy < 0.01:
            # Use sine wave to create breathing rhythm
            energy = 0.02 * (1.0 + math.sin(self.t * 0.1)) 

        # 2. LOGIC CALCULATION (KQI)
        # Non-linear gain: energy squared makes loud music 'explosive'
        r_val = (energy ** 2) * 50.0 # Increased gain
        omega_val = np.var(audio_vector[:20]) * 20.0
        
        kqi_res = self.kqi.process(r=r_val, omega=omega_val, sigma=0.1, t=self.t, rho=0.5)
        
        # FORCE DEPTH TO BE VISIBLE
        kqi_res["D"] = max(kqi_res["D"], 0.1) # Minimum 0.1
        
        # 3. STOCHASTIC DECISION
        # Fresh random will for every move to break the 'same posture' lock
        self.latent_will = np.random.randn(128) 
        
        if kqi_res["D"] > 2.0:
            self.current_state = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
        else:
            self.current_state = 0 # Default to Toprock for mid-energy

        drive = np.clip(kqi_res["D"] / 2.0, 1.0, 5.0)
        return self.current_state, drive, kqi_res, self.latent_will