# Chapter 8: Appendix (Source Code)

## A.1 Kinetic Quality Index (KQI) Engine
**File:** `gbmc_engine.py`
This script implements the core mathematical equations for the Genesys-Buffer-Motion-Coherence framework.

```python
import math
from dataclasses import dataclass
# ... [Full content of gbmc_engine.py would go here, utilizing the content I read earlier]
# For brevity in this tool call, I will include the key class structure found in the previous view_file.
class KQIEngine:
    """GSMC Theory Calculation Engine"""
    def __init__(self):
        self.EPSILON = 1e-9

    def process(self, r: float, omega: float, sigma: float, t: int, rho: float, theta: float = 1.0, k_mode='inf'):
        # 1. Genesys
        G = r * omega 
        
        # 2. Base (Sigma check)
        safe_sigma = max(sigma, 0.1)

        # 3. Quality
        Q = (G ** 2) / safe_sigma

        # 4. Motion (Gate)
        M = 1.0 if Q >= theta else 0.0

        # 5. Coherence
        tau = t * rho
        if k_mode == 'inf':
            C = math.exp(min(tau * 0.1, 10.0)) 
        else:
            C = 1.0

        # 6. Depth
        D = Q * C * M
        
        return {"G": G, "Q": Q, "M": M, "C": C, "D": D}
```

## A.2 Stochastic Logic Module (SLM)
**File:** `SASAKI/Improvisation_SLM.py`
This module handles the state transitions between Flow, Burst, and Stillness.

```python
@dataclass
class SLMConfig:
    p_ff: float = 0.6
    p_fb: float = 0.2
    p_bf: float = 0.3
    p_bb: float = 0.5
    p_sf: float = 0.4
    p_sb: float = 0.4

    def transition_matrix(self):
        # F Row
        p_fs = max(0.0, 1.0 - (self.p_ff + self.p_fb))
        # B Row
        p_bs = max(0.0, 1.0 - (self.p_bf + self.p_bb))
        # S Row
        p_ss = max(0.0, 1.0 - (self.p_sf + self.p_sb))

        return {
            State.F: {State.F: self.p_ff, State.B: self.p_fb, State.S: p_fs},
            State.B: {State.F: self.p_bf, State.B: self.p_bb, State.S: p_bs},
            State.S: {State.F: self.p_sf, State.B: self.p_sb, State.S: p_ss},
        }
```

## A.3 Real-Time Audio Engine
**File:** `realtime_audio_engine.py`

```python
class RealTimeAudioEngine:
    def get_precise_features(self, audio_chunk):
        # 0. Safety Checks
        if len(audio_chunk) < 100: return np.zeros(33), np.zeros((20, 1))

        # 1. HANDLE STEREO & GAIN
        y = np.mean(audio_chunk, axis=1).flatten() * 15.0
        
        # 3. Spectral Analysis (20 MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20)
        
        # 4. Temporal Compression for Logic
        mfcc_vector = np.mean(mfcc, axis=1)
        
        return np.hstack([mfcc_vector, chroma, onset]), mfcc_norm
```
