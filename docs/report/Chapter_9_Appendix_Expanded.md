
# Chapter 9: Technical Appendix (Source Code)

This appendix contains the full source code for the core components of the Sasaki-GAN system, provided for reproducibility.

## A.1 GBMC Cognitive Engine (`gbmc_engine.py`)
This file implements the **Genesis-Buffer-Motion-Coherence** theory equations.

```python
import math
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BufferItem:
    payload: Any
    tags: List[str]
    arousal: float
    novelty: float
    confidence: float
    timestamp: float

class KQIEngine:
    """
    Kinetic Quality Index Engine.
    Calculates the 'Depth' of a move based on Audio Energy (r) and Flux (omega).
    """
    def __init__(self):
        self.EPSILON = 1e-9

    def process(self, r: float, omega: float, sigma: float, t: int, rho: float, theta: float = 1.0, k_mode='inf'):
        # 1. Genesys: The raw will to move
        G = (r + 0.1) * (omega + 0.1) * 2.0
        
        # 2. Quality: Inverse of Risk
        safe_sigma = max(sigma, 0.1)
        Q = (G ** 2) / safe_sigma

        # 3. Motion Gate
        M = 1.0 if Q >= theta else 0.0

        # 4. Coherence: Commitment increases with time (t) and density (rho)
        tau = t * rho
        if k_mode == 'inf':
            C = math.exp(min(tau * 0.1, 10.0)) 
        else:
            C = 1.0

        # 5. Depth: The final Decision Metric
        D = Q * C * M
        
        return {"G": G, "Q": Q, "M": M, "C": C, "D": D}
```

## A.2 Stochastic Logic Module (`Improvisation_SLM.py`)
This module manages the transition probabilities between Toprock, Footwork, and Freeze.

```python
import random
import numpy as np
from enum import Enum

class State(Enum):
    F = 0 # Flow (Toprock)
    B = 1 # Burst (Power)
    S = 2 # Stillness (Freeze)

class ImprovisationSLM:
    def __init__(self):
        self.state = State.S
        self.entropy_history = []

    def compute_context_entropy(self, audio_features):
        """
        Calculates Shannon Entropy of the audio buffer to detect 'Chaos'.
        """
        # Simplified for report brevity:
        energy = np.mean(audio_features)
        return -energy * np.log(energy + 1e-9)

    def step(self, current_state, kqi_depth):
        # High Depth = High Probability of Power (State B)
        # Low Depth = High Probability of Freeze (State S)
        if kqi_depth > 5.0:
            return State.B
        elif kqi_depth < 0.5:
            return State.S
        else:
            return State.F
```

## A.3 Real-Time Perceiver (`realtime_audio_engine.py`)
Handles WASAPI Loopback and Feature Extraction.

```python
import numpy as np
import librosa

class RealTimeAudioEngine:
    def get_precise_features(self, audio_chunk):
        # 1. HANDLE STEREO & GAIN
        if audio_chunk.ndim > 1:
            y = np.mean(audio_chunk, axis=1).flatten()
        else:
            y = audio_chunk.flatten()
        
        # Boost gain for weak laptop microphones
        y = y * 15.0
        
        # 2. Spectral Analysis
        mfcc = librosa.feature.mfcc(y=y, sr=48000, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=48000)
        onset = librosa.onset.onset_strength(y=y, sr=48000)
        
        # 3. Z-Score Normalization
        # Critical for handling different music volumes
        mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        
        return np.hstack([np.mean(mfcc_norm, axis=1), np.mean(chroma, axis=1), np.mean(onset)])
```

## A.4 The Brain (`sasaki_brain.py`)
The central orchestrator connecting Perception to Action.

```python
class SasakiLiveBrain:
    def decide_style(self, audio_vector):
        self.t += 1
        energy = audio_vector[-1] 
        
        # --- THE HEARTBEAT FIX ---
        # If energy is near 0, provide a fake 'Subconscious' pulse
        if energy < 0.01:
            energy = 0.02 * (1.0 + math.sin(self.t * 0.1)) 

        # Logic Calculation
        r_val = (energy ** 2) * 50.0 
        omega_val = np.var(audio_vector[:20]) * 20.0
        
        kqi_res = self.kqi.process(r=r_val, omega=omega_val, sigma=0.1, t=self.t, rho=0.5)
        
        # Latent Will Injection (The Ghost)
        self.latent_will += np.random.randn(128) * 0.05
        self.latent_will = np.clip(self.latent_will, -1, 1)

        return kqi_res, self.latent_will
```

## A.5 Training Loop (`train_multimodel_residual.py`)
The "Prison Break" training logic.

```python
# Simplified snippet of the loss function
def compute_loss(real, fake, velocity, audio_energy):
    # 1. Adversarial Loss
    loss_adv = criterion_GAN(discriminator(fake), True)
    
    # 2. Physics Loss (MSE)
    loss_phys = criterion_MSE(real, fake)
    
    # 3. KINETIC PENALTY (Phase 7)
    # If Music is Loud (energy > 0.5) but Velocity is Low (v < 1.0)...
    kinetic_violation = torch.relu(1.0 - velocity) * (audio_energy > 0.5).float()
    loss_kinetic = torch.mean(kinetic_violation) * 10.0
    
    return loss_adv + loss_phys + loss_kinetic
```
