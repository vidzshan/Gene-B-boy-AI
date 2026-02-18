# Chapter 13: Complete Source Code Repository

This chapter contains the complete source code for the critical components of the system.

## File: gbmc_engine.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\gbmc_engine.py`

```python
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 0) 前提: Minimal Buffer Implementation (仮置き)
# ============================================================
@dataclass
class Item:
    payload: Any
    tags: List[str]
    arousal: float
    novelty: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def as_dict(self):
        return {
            "payload": self.payload,
            "tags": self.tags,
            "metrics": {"arousal": self.arousal, "novelty": self.novelty, "conf": self.confidence},
            "ts": self.timestamp
        }

class Buffer:
    def __init__(self, cap: int = 64):
        self.cap = cap
        self.data: List[Item] = []

    def push(self, payload, tags=None, arousal=0.0, novelty=0.0, confidence=0.5):
        it = Item(payload, tags or [], arousal, novelty, confidence)
        self.data.append(it)
        if len(self.data) > self.cap:
            self.data.pop(0)
        return it

    def view(self, n=10):
        return self.data[-n:]

# ============================================================
# 1) KQI Engine (The Logic Core)
# ============================================================
class KQIEngine:
    """GSMC理論に基づく計算エンジン"""
    def __init__(self):
        self.EPSILON = 1e-9

    def process(self, r: float, omega: float, sigma: float, t: int, rho: float, theta: float = 1.0, k_mode='inf'):
        # 1. Genesys
        G = r * omega #omega = minesweeper's landmine
        
        # 2. Base (Sigma check)
        safe_sigma = max(sigma, 0.1) # 0.1を下限として安全マージン確保

        # 3. Quality
        Q = (G ** 2) / safe_sigma

        # 4. Motion (Gate)
        M = 1.0 if Q >= theta else 0.0 #theta = minesweeper's threshold

        # 5. Coherence
        tau = t * rho
        if k_mode == 'inf':
            # tauが大きすぎるとexpが爆発するのでクリップするか、対数スケールで調整
            # ここでは味付けとして tau * 0.1 を入力とする
            C = math.exp(min(tau * 0.1, 10.0)) 
        elif k_mode == 1:
            C = 1.0 + tau
        else:
            C = 1.0

        # 6. Depth
        D = Q * C * M
        
        return {"G": G, "Q": Q, "M": M, "C": C, "D": D}

# ============================================================
# 2) MineLife (The Body / Eco-system)
#    (User's provided code, slightly compacted)
# ============================================================
@dataclass
class MineLifeConfig:
    w: int = 16
    h: int = 16
    mine_ratio: float = 0.12
    birth: Tuple[int, ...] = (3,)
    survive: Tuple[int, ...] = (2, 3)
    reveal_per_tick: int = 2
    danger_spread: float = 0.35
    seed: Optional[int] = None

class MineLife: #if NTU use, embeddied here.
    def __init__(self, cfg: MineLifeConfig):
        self.cfg = cfg
        if cfg.seed is not None: random.seed(cfg.seed)
        self.t = 0
        self.alive = [[1 if random.random() < 0.25 else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.mine = [[1 if random.random() < cfg.mine_ratio else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.revealed = [[0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.risk = [[0.0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self._update_risk()

    def _n8(self, y, x):
        out = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                ny, nx = y+dy, x+dx
                if 0<=ny<self.cfg.h and 0<=nx<self.cfg.w: out.append((ny, nx))
        return out

    def _count(self, grid, y, x):
        return sum(grid[ny][nx] for (ny, nx) in self._n8(y, x))

    def _update_risk(self):
        base = [[min(1.0, self._count(self.mine, y, x)/8.0) for x in range(self.cfg.w)] for y in range(self.cfg.h)]
        new_r = [[0.0]*self.cfg.w for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                neigh = self._n8(y, x) #if GCN use, embeddied here.
                avg = sum(base[ny][nx] for ny, nx in neigh)/max(1, len(neigh))
                new_r[y][x] = base[y][x]*(1.0-self.cfg.danger_spread) + avg*self.cfg.danger_spread
        self.risk = new_r

    def _life_step(self):
        nxt = [[0]*self.cfg.w for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                n = self._count(self.alive, y, x)
                state = self.alive[y][x]
                nxt[y][x] = 1 if (state==1 and n in self.cfg.survive) or (state==0 and n in self.cfg.birth) else 0
        self.alive = nxt

    def _auto_reveal(self):
        cand = [(y, x) for y in range(self.cfg.h) for x in range(self.cfg.w) if self.revealed[y][x] == 0]
        if not cand: return []
        cand.sort(key=lambda p: self.risk[p[0]][p[1]]) # Sort by risk (safer first)
        picks = []
        for _ in range(min(self.cfg.reveal_per_tick, len(cand))):
            # 30% chance to pick from high risk (tail), 70% from low risk (head)
            p = cand.pop(-1) if random.random() > 0.7 else cand.pop(0)
            self.revealed[p[0]][p[1]] = 1
            picks.append((p[0], p[1], self.mine[p[0]][p[1]], self._count(self.mine, p[0], p[1])))
        return picks

    def tick(self) -> Dict[str, Any]:
        self.t += 1
        self._life_step()
        self._update_risk()
        reveals = self._auto_reveal()

        alive_ratio = sum(sum(r) for r in self.alive) / (self.cfg.w * self.cfg.h)
        revealed_ratio = sum(sum(r) for r in self.revealed) / (self.cfg.w * self.cfg.h)
        risk_mean = sum(sum(r) for r in self.risk) / (self.cfg.w * self.cfg.h)
        mine_hits = sum(1 for (_, _, hit, _) in reveals if hit == 1)
        
        # MineLife Internal Metrics
        tension = min(1.0, 0.15 + 0.6 * risk_mean + 0.25 * mine_hits)
        novelty = max(0.0, 1.0 - revealed_ratio) * 0.7 + abs(0.5 - alive_ratio) * 0.3
        
        tags = ["MineLife"]
        if tension > 0.7: tags.append("high_tension")
        if alive_ratio > 0.55: tags.append("dense_motion")

        return {
            "t": self.t,
            "alive_ratio": alive_ratio,
            "revealed_ratio": revealed_ratio,
            "risk_mean": risk_mean,
            "tension": tension,
            "novelty": novelty,
            "reveals": reveals,
            "tags": tags,
            "mine_hits": mine_hits
        }

# ============================================================
# 3) The Integration: KQI-Powered Brain Buffer
# ============================================================
class BrainBufferOneFile:
    """
    MineLife (無意識/身体) -> KQI Engine (意識/論理) -> Buffer (記憶)
    """
    def __init__(self, buffer_cap: int = 64, mine_cfg: Optional[MineLifeConfig] = None):
        self.buf = Buffer(cap=buffer_cap)

        self.mine = MineLife(mine_cfg or MineLifeConfig())
        
        self.kqi = KQIEngine() # ここにKQIエンジンを搭載

        # KQIパラメータ設定
        self.kqi_theta = 0.5  # Motion閾値
        self.kqi_mode = 'inf' # 共鳴モード

    def tick(self, n: int = 1) -> List[Dict[str, Any]]:
        outs = []
        for _ in range(max(0, n)):
            # 1. MineLifeの状態更新 (Body Update)
            s = self.mine.tick()
            
            # 2. 変数マッピング (Sensory Input -> KQI Variables)
            # r (Range/Power): 生命密度が高いほどパワーがある (0.0 - 10.0 scale)
            r_val = s["alive_ratio"] * 10.0
            
            # omega (Velocity): 新規性が高い（変化が激しい）ほど回転が速い (0.0 - 5.0 scale)
            omega_val = s["novelty"] * 5.0
            
            # sigma (Noise): リスクが高い、または地雷を踏んだ瞬間に急増する
            sigma_val = s["risk_mean"] * 5.0 + (s["mine_hits"] * 10.0)
            
            # t, rho: 時間と密度
            t_val = s["t"]
            rho_val = s["alive_ratio"]

            # 3. KQI Engine Process (Cognitive Processing)
            kqi_res = self.kqi.process(
                r=r_val, 
                omega=omega_val, 
                sigma=sigma_val, 
                t=t_val, 
                rho=rho_val, 
                theta=self.kqi_theta, 
                k_mode=self.kqi_mode
            )

            # 4. 統合ペイロードの作成
            # MineLifeの生データ + KQIの解釈データ
            full_payload = {
                "source": "MineLife",
                "raw_stats": {k: s[k] for k in ["t", "alive_ratio", "risk_mean", "tension"]},
                "kqi_analysis": kqi_res, # ここに D, Q, M が入る
                "events": s["reveals"]
            }
            
            # タグ付けの強化 (KQIの結果に基づいてタグを追加)
            final_tags = s["tags"] + ["KQI_Processed"]
            if kqi_res["M"] > 0:
                final_tags.append("MOTION_ACTIVE")
            if kqi_res["D"] > 100.0:
                final_tags.append("DEEP_RESONANCE")

            # 5. BufferへPush (Memory Storage)
            # KQIの算出値(D)を arousal (覚醒度) として扱うのが自然
            self.buf.push(
                payload=full_payload,
                tags=final_tags,
                arousal=min(1.0, kqi_res["D"] / 50.0), # Dを正規化して覚醒度へ
                novelty=s["novelty"],
                confidence=kqi_res["M"] # Motionゲートが開いている＝確信がある
            )
            
            outs.append({"tick": s["t"], "kqi": kqi_res, "tags": final_tags})
            
        return outs

    def view(self, n: int = 5):
        return [it.as_dict() for it in self.buf.view(n)]

# ============================================================
# テスト実行
# ============================================================
if __name__ == "__main__":
    brain = BrainBufferOneFile(mine_cfg=MineLifeConfig(w=10, h=10, mine_ratio=0.15))
    
    print("--- 脳内シミュレーション開始 (KQI Integrated) ---")
    
    # 10ターン回してみる
    results = brain.tick(n=10)
    
    for res in results:
        kqi = res["kqi"]
        print(f"Tick {res['tick']:02d} | "
              f"G:{kqi['G']:.2f} σ:{kqi['Q']:.2f} " # Qは便宜上sigmaの逆数的な位置づけだがここではQを表示
              f"-> Motion:{kqi['M']} "
              f"-> Depth(D):{kqi['D']:.4f} "
              f"| Tags: {res['tags']}")
```

<div style='page-break-after: always;'></div>

## File: sasaki_brain.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\sasaki_brain.py`

```python
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
```

<div style='page-break-after: always;'></div>

## File: realtime_audio_engine.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\realtime_audio_engine.py`

```python
import numpy as np
import librosa

class RealTimeAudioEngine:
    def __init__(self, sr=15360):
        self.sr = sr

    def get_precise_features(self, audio_chunk):
        """
        Extracts High-Fidelity Audio DNA for Logic (Vector) and UI (Matrix).
        """
        # 0. Safety Checks
        if len(audio_chunk) < 100: return np.zeros(33), np.zeros((20, 1))

        # 1. HANDLE STEREO & GAIN (15x Boost for Intel Arrays)
        if audio_chunk.ndim > 1:
            y = np.mean(audio_chunk, axis=1).flatten().astype(np.float32)
        else:
            y = audio_chunk.flatten().astype(np.float32)
        y = y * 15.0
        
        # 2. Check for signal presence
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.00005: 
             return np.zeros(33), np.zeros((20, 1))

        try:
            # 3. Spectral Analysis (20 MFCCs)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20)
            
            # Precise Normalization: Z-score scaling
            # This makes the texture 'dance' in the UI
            mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            
            # 4. Temporal Compression for Logic
            mfcc_vector = np.mean(mfcc_norm, axis=1)
            
            # 5. Multi-modal stack (33-dim)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=self.sr), axis=1)
            onset = np.mean(librosa.onset.onset_strength(y=y, sr=self.sr))
            
            # Return both the Vector (for Brain) and Matrix (for UI)
            return np.hstack([mfcc_vector, chroma, onset]), mfcc_norm
            
        except Exception as e:
            return np.zeros(33), np.zeros((20, 1))

    def get_features(self, audio_chunk):
        # Wrapper for backward compatibility with Brain Thread
        vec, _ = self.get_precise_features(audio_chunk)
        return vec
```

<div style='page-break-after: always;'></div>

## File: blender_receiver.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\blender_receiver.py`

```python
import bpy
import socket
import struct
import mathutils
import os

# --- SETTINGS ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
JOINT_COUNT = 25
SCALE_FACTOR = 1.0 

# Path to your refinement GLB (Absolute Path to UltraShape Source)
GLB_PATH = r"C:\Users\kos04\OneDrive\Desktop\vidz\GCN\UltraShape-1.0\outputs\refine_demo\charachter_refined.glb"

class UpdReceiverModal(bpy.types.Operator):
    """Runs a modal timer to listen for UDP packets"""
    bl_idname = "wm.udp_receiver_modal"
    bl_label = "Sasaki UDP Receiver"
    
    _timer = None
    _sock = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                data, addr = self._sock.recvfrom(1024) 
                if len(data) >= 300:
                    count = len(data) // 4
                    floats = struct.unpack(f'{count}f', data)
                    
                    # Update Blender Debug Spheres
                    for i in range(JOINT_COUNT):
                        if i*3 + 2 < len(floats):
                            # HPI-GCN output is typically (X, Y, Z) normalized [-1, 1]
                            # Blender is Z-Up. We usually map:
                            # Python X -> Blender X
                            # Python Y -> Blender Z (Height)
                            # Python Z -> Blender Y (Depth)
                            x = floats[i*3] * SCALE_FACTOR
                            y = floats[i*3+1] * SCALE_FACTOR # Height
                            z = floats[i*3+2] * SCALE_FACTOR # Depth
                            
                            # Update Empty position
                            obj_name = f"J{i}"
                            if obj_name in bpy.data.objects:
                                obj = bpy.data.objects[obj_name]
                                # Correct Mapping for Blender Viewport
                                obj.location = (x, z, y) 
                                
            except BlockingIOError:
                pass 
            except Exception as e:
                print(f"UDP Error: {e}")
                
        elif event.type == 'ESC':
            return self.cancel(context)
            
        return {'PASS_THROUGH'}

    def execute(self, context):
        # 0. Load the UltraShape Character if not present
        if not any("charachter" in o.name for o in bpy.data.objects):
            print(f"📂 Loading UltraShape: {GLB_PATH}")
            if os.path.exists(GLB_PATH):
                try:
                    bpy.ops.import_scene.gltf(filepath=GLB_PATH)
                    print("✅ UltraShape Loaded.")
                except RuntimeError as e:
                    print(f"⚠️ GLB Import Failed (Likely Corrupted): {e}")
                    print("➡️ Proceeding to start UDP Receiver anyway (Debug Spheres only).")
            else:
                print(f"⚠️ GLB Not Found: {GLB_PATH}")

        # 1. Setup Socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_IP, UDP_PORT))
        self._sock.setblocking(False)
        
        # 2. Setup Debug Skeleton Handles
        collection = bpy.data.collections.get("Sasaki_Rig")
        if not collection:
            collection = bpy.data.collections.new("Sasaki_Rig")
            bpy.context.scene.collection.children.link(collection)
            
        for i in range(JOINT_COUNT):
            name = f"J{i}"
            if name not in bpy.data.objects:
                bpy.ops.object.empty_add(type='SPHERE', radius=0.03)
                obj = bpy.context.active_object
                obj.name = name
                for coll in obj.users_collection:
                    coll.objects.unlink(obj)
                collection.objects.link(obj)
                
        # 3. Start Timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        print(f"✅ Listening on {UDP_IP}:{UDP_PORT}")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self._sock.close()
        print("🛑 UDP Receiver Stopped")
        return {'CANCELLED'}

def register():
    bpy.utils.register_class(UpdReceiverModal)

def unregister():
    bpy.utils.unregister_class(UpdReceiverModal)

if __name__ == "__main__":
    register()
    bpy.ops.wm.udp_receiver_modal()

```

<div style='page-break-after: always;'></div>

## File: RUN_SASAKI_LIVE.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\RUN_SASAKI_LIVE.py`

```python
import torch

import numpy as np

import cv2

import sounddevice as sd

import queue

import threading

import json

import time
import socket
import struct

from collections import deque

from scipy.spatial.transform import Slerp, Rotation as R

import scipy.interpolate as interpolate

import warnings

warnings.simplefilter("ignore")



from model.generator import AudioAwareGenerator

from realtime_audio_engine import RealTimeAudioEngine

from sasaki_brain import SasakiLiveBrain



# --- SETTINGS ---

DEVICE = torch.device("cuda")

INPUT_DEVICE = 14

SAMPLING_RATE = 48000

GOLDEN_LIMIT = 150

BONES = [(0,1), (1,20), (20,2), (2,3), (20,4), (4,5), (5,6), (20,8), (8,9), (9,10), (0,12), (12,13), (13,14), (0,16), (16,17), (17,18)]



class KineticPostProcessor:

    """Solves Size Consistency and Seamless Stitching."""

    def __init__(self, db_poses):

        self.db_poses = db_poses

        spine_lengths = np.linalg.norm(db_poses[:, 20] - db_poses[:, 0], axis=1)

        self.ref_scale = np.mean(spine_lengths)

        print(f"📏 Reference Human Scale established: {self.ref_scale:.4f}")



    def normalize_pose(self, pose):

        """Forces all skeletons to have the same height/size."""

        curr_scale = np.linalg.norm(pose[20] - pose[0])

        if curr_scale < 1e-6: return pose

        return pose * (self.ref_scale / curr_scale)



    def convert_to_global(self, relative_poses):

        """

        CondMDI Requirement: Cumulatively sum displacements to fix the dancer in world space.

        """

        global_poses = relative_poses.copy()

        # Assume joint 0 is the root

        # We cumulatively sum the X and Z (floor) displacements across the 64 frames

        # Note: Input must be (T, V, C) or similar. Assuming (T, 25, 3)

        if global_poses.ndim == 3: # (T, 25, 3)

             global_poses[:, 0, [0, 2]] = np.cumsum(relative_poses[:, 0, [0, 2]], axis=0)

        elif global_poses.ndim == 2: # (25, 3) - Single frame? Cumsum doesn't apply well.

             pass

        return global_poses



        return pose * (self.ref_scale / curr_scale)



class BracePatternPlanner:

    def __init__(self):

        # Your specific BRACE data counts

        self.patterns = [

            ([0, 2, 1], 172), # TR -> PW -> FW

            ([0, 1], 70),    # TR -> FW

            ([0, 2], 69),    # TR -> PW

            ([0, 1, 2], 60), # TR -> FW -> PW

            ([2], 28),       # PW ONLY

            ([1], 20),       # FW ONLY

        ]

        # Normalize weights

        counts = np.array([p[1] for p in self.patterns])

        self.weights = counts / counts.sum()



    def select_new_story(self):

        # Pick a full sequence 'Story' to follow

        idx = np.random.choice(len(self.patterns), p=self.weights)

        return self.patterns[idx][0]



class UnityStreamer:
    def __init__(self, ip="127.0.0.1", port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)

    def send_pose(self, pose_array):
        # pose_array shape: (25, 3)
        # Flatten to 75 floats
        data = pose_array.flatten().tolist()
        # Pack into binary struct (75 floats)
        msg = struct.pack(f'{len(data)}f', *data)
        self.sock.sendto(msg, self.addr)

class SasakiMasterControlCenter:

    def __init__(self):

        print("🧬 Building Full Research Suite...")

        self.gen = AudioAwareGenerator(audio_dim=33).to(DEVICE)

        self.gen.load_state_dict(torch.load('./pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth'))

        self.gen.eval()

       

        # Load Dataset

        db = np.load('./data/brace/BRACE_synced_v2.npz')

        self.db_poses = db['x_data'] if 'x_data' in db else db['x_train']

        self.db_poses = self.db_poses.transpose(0, 2, 3, 1, 4).reshape(-1, 75)

        self.db_labels = db['labels'] if 'labels' in db else db['y_data']

        self.db_tensor = torch.from_numpy(self.db_poses).float().to(DEVICE)

       

        # Calculate Velocity Manifold (Diff between frames)

        db_vel = np.diff(self.db_poses, axis=0, prepend=self.db_poses[0:1])

        self.db_velocity_tensor = torch.from_numpy(db_vel).float().to(DEVICE)

       

        # Style Buckets

        expanded_labels = np.repeat(self.db_labels, 64)

        self.bucket_indices = {i: np.where(expanded_labels == i)[0] for i in range(3)}



        self.mean_pose_raw = np.load('./data/brace/mean_pose.npy')

        self.mean_pose = torch.from_numpy(self.mean_pose_raw).float().to(DEVICE).view(1, 3, 1, 25, 1)

        self.audio_engine = RealTimeAudioEngine(sr=SAMPLING_RATE)

        self.brain = SasakiLiveBrain()

        self.audio_queue = queue.Queue()

       

        self.post_processor = KineticPostProcessor(self.db_poses.reshape(-1, 25, 3))

       

        # --- FLOW BUFFERS ---

        self.move_queue = queue.Queue(maxsize=2)

        self.is_running = True

        self.prev_pose = None

        self.history = deque(maxlen=20)

        self.boredom = 0.0

        self.current_planned_start_idx = 0 # Track for continuity

        self.current_momentum = torch.zeros(1, 75).to(DEVICE)

       

        # --- PILLAR 2: GLOBAL ROOT VELOCITY ---

        self.world_root_pos = np.array([0.0, 0.0, 0.0])



        # Forensic Buffers

        self.audio_log = deque(maxlen=GOLDEN_LIMIT)

        self.kinetic_log = deque(maxlen=GOLDEN_LIMIT)

        self.kqi_log = deque(maxlen=GOLDEN_LIMIT)

        self.pose_history = deque(maxlen=5)

       

        self.pattern_planner = BracePatternPlanner()
        self.latest_live_audio = np.zeros(33) # Real-Time Ear Buffer
        self.latest_mfcc_matrix = np.zeros((20, 1)) # Spectral Matrix Buffer
        self.streamer = UnityStreamer() # UDP Bridge to Blender/Unity



    def get_playback_params(self, audio_vec):

        # energy = onset, complexity = mfcc variance

        energy = audio_vec[-1] # Assuming last dim is energy or similar

        complexity = np.var(audio_vec[:20])

       

        # 1. STILLNESS (Freeze)

        if energy < 0.01 and complexity < 0.1:

            return 0.0 # Index does not move

       

        # 2. SPEED (Conductance)

        # High entropy/energy makes moves fast

        speed = np.clip(energy * 2.0 + complexity * 0.5, 0.5, 2.0)

        return speed



    def get_tta_embedding(self, current_frame, total_frames=64, d_model=33):

        """

        Harvey et al. 2020: Time-to-Arrival Embedding.

        """

        tta = total_frames - current_frame

        # Sinusoidal encoding

        indices = torch.arange(0, d_model // 2).float().to(DEVICE)

        div_term = torch.exp(indices * -(np.log(10000.0) / (d_model / 2)))

       

        z_tta = torch.zeros(1, 1, d_model).to(DEVICE)

        z_tta[0, 0, 0::2] = torch.sin(tta * div_term)

        z_tta[0, 0, 1::2] = torch.cos(tta * div_term)

        return z_tta



    def capture_snapshot(self):

        filename = f"GOLDEN_5S_{int(time.time())}.json"

        with open(filename, 'w') as f:

            json.dump({"audio_dna": [a.tolist() for a in self.audio_log], "kinetic_vectors": [k.tolist() for k in self.kinetic_log], "logic_metrics": list(self.kqi_log)}, f)

        print(f"\n📸 [SNAPSHOT SAVED]: {filename}")



    def draw_precise_mfcc(self, img, mfcc_matrix):
        """
        Draws a professional-grade spectral monitoring zone.
        mfcc_matrix: (20, T) normalized coefficients.
        """
        # Define UI Area: Top Right Sidebar
        start_x, start_y = 820, 50
        
        # 1. Draw Frequency Bands
        for i in range(20):
            # Calculate intensity for this band
            val = np.mean(mfcc_matrix[i, :]) if mfcc_matrix.shape[1] > 0 else 0
            # Map Z-score (-2 to 2) to brightness (0 to 255)
            intensity = int(np.clip((val + 2) * 64, 0, 255))
            
            # Color Logic: High frequency (Top) is Cyan, Low (Bottom) is Pink
            color = (255, 255 - intensity, intensity) 
            
            cv2.rectangle(img, 
                        (int(start_x + (i * 18)), start_y), 
                        (int(start_x + (i * 18) + 15), start_y + 40), 
                        color, -1)
        
        cv2.putText(img, "SPECTRAL PERCEPTION (MFCC 1-20)", (start_x, start_y - 15), 
                    1, 0.6, (200, 200, 200), 1)

    def draw_dashboard(self, img, kqi, style_idx, drive, audio_vec):

        cv2.rectangle(img, (800, 0), (1200, 800), (30, 30, 30), -1)

        # 1. SPECTRAL HEATMAP (New Professional Logic)
        self.draw_precise_mfcc(img, self.latest_mfcc_matrix)



        # 2. METRICS

        cv2.putText(img, f"SASAKI DEPTH (D): {kqi.get('D',0):.2f}", (820, 130), 1, 1.2, (0, 255, 255), 2)

        cv2.putText(img, f"KINETIC DRIVE: {drive:.2f}x", (820, 170), 1, 1.0, (255, 255, 0), 1)

        cv2.putText(img, f"CREATIVE DRIFT: {self.boredom:.2f}", (820, 210), 1, 1.0, (255, 0, 255), 1)

       

        # 3. STYLE

        styles = ["TOPROCK", "FOOTWORK", "POWERMOVE", "IDLE"]

        s_idx = style_idx if style_idx < 4 else 3

        for i, name in enumerate(styles):

            color = (0, 255, 0) if i == s_idx else (80, 80, 80)

            indicator = "[X]" if i == s_idx else "[ ]"

            cv2.putText(img, f"{indicator} {name}", (840, 250 + (i*30)), 1, 1, color, 2)

           

        return img



    def apply_masked_sum(self, xt, ground_truth_c, mask_m):

        """

        Formula from PDF: xt_tilde = m * c + (1 - m) * xt

        This 'pins' your real breakdancing moves while the AI denoises the transition.

        """

        # Ensure mask dimensions match

        # mask_m is (T), need (T, 25, 3) or similar broadcasting

        if mask_m.ndim == 1:

            mask_m = mask_m.view(-1, 1, 1).to(DEVICE)

        return (mask_m * ground_truth_c) + ((1 - mask_m) * xt)



    def diffusion_model(self, xt, t, text_p=None):

        """

        Mockup/Wrapper for the HPI-GCN acting as UNet.

        """

        # Our generator takes (z, audio, label).

        # For this pivotal upgrade, we treat 'xt' as the latent/input.

        # We need to adapt arguments. This is a PLACHOLDER for the full integration.

        # Returning random noise or simplified prediction for now to satisfy the interface.

        return torch.randn_like(xt) * 0.01



    def update_step(self, xt, eps, t):

        """

        Standard DDPM update step (simplified).

        """

        alpha = 0.9 # placeholder

        return (xt - eps) * alpha + torch.randn_like(xt) * 0.001



    def generate_diffusion_bridge(self, start_move, end_move, text_p):

        """

        Algorithm 2: Sampling.

        start_move: 32 frames of Toprock

        end_move: 32 frames of Power

        """

        # 1. Create Canvas (128 frames)

        # [Start Move (32)] + [Gap (64)] + [End Move (32)]

        c = torch.cat([start_move, torch.zeros(64, 25, 3).to(DEVICE), end_move])

        m = torch.cat([torch.ones(32).to(DEVICE), torch.zeros(64).to(DEVICE), torch.ones(32).to(DEVICE)]) # Mask

       

        # 2. Start with pure noise

        xt = torch.randn(128, 25, 3).to(DEVICE)

       

        # 3. Denoising Loop (Simplified for speed)

        # Reducing to 10 steps for realtime plausibility

        for t in reversed(range(10)):

            # Force the AI to see your real moves

            xt_tilde = self.apply_masked_sum(xt, c, m)

           

            # Predict the 'Clean' motion using text prompt p

            eps = self.diffusion_model(xt_tilde, t, text_p)

           

            # Update xt-1

            xt = self.update_step(xt, eps, t)

           

        return xt # The final 128-frame seamless dance



    def bi_objective_rag_search(self, query_intent, anchor_pose, prev_velocity, style_idx):

        """

        UPGRADED: Sasaki Kinetic Gum Search.

        prev_velocity: (1, 75) tensor representing the 'momentum' of the last move.

        """

        indices = self.bucket_indices[style_idx]

        manifold = self.db_tensor[indices]

        vel_manifold = self.db_velocity_tensor[indices]

       

        # 1. Position Distance (Shape)

        d_pose = torch.cdist(anchor_pose, manifold)

        # 2. Music Distance (Intent)

        d_music = torch.cdist(query_intent, manifold)

        # 3. Velocity Distance (Momentum / The 'Gum')

        d_vel = torch.cdist(prev_velocity, vel_manifold)



        # Normalize

        d_pose /= (torch.max(d_pose) + 1e-6)

        d_music /= (torch.max(d_music) + 1e-6)

        d_vel /= (torch.max(d_vel) + 1e-6)



        # --- PHASE 3: CATEGORY TRANSITION PENALTY ---

        velocity_magnitude = torch.norm(prev_velocity)

        if velocity_magnitude > 0.4: # If moving fast

            # High Speed: prioritize the 'swing' (40%) to ensure flow

            total_dist = (0.4 * d_music) + (0.2 * d_pose) + (0.4 * d_vel)

        else:

            # Low Speed: focus on pose shape and music

            total_dist = (0.5 * d_music) + (0.4 * d_pose) + (0.1 * d_vel)

       

        # 5. STOCHASTIC TOP-K (Prison Break)
        k = 100 
        _, top_k_indices = torch.topk(total_dist, k, largest=False)
        
        # Randomly pick from the top 10% of matches
        random_pick = np.random.randint(0, 10) 
        try:
             selection_idx = indices[top_k_indices[0, random_pick].item()]
        except:
             selection_idx = indices[top_k_indices[0, 0].item()]

        # Tabu history...
        self.history.append(selection_idx)
        return selection_idx



    def brain_thread_loop(self):

        print("🧠 Brain Thread: Choreographic Conductor Online.")

        while self.is_running:

            # 1. Select the 'Story' (Pattern) based on BRACE stats

            current_story = self.pattern_planner.select_new_story()

           

            for style_idx in current_story:

                # 2. Accumulate audio context specifically for THIS segment

                chunks = []

                while len(chunks) < 4: # Wait for ~0.4s of audio for valid texture

                    try: chunks.append(self.audio_queue.get(timeout=0.5))

                    except queue.Empty: break

               

                if not chunks: continue

                audio_vec = self.audio_engine.get_features(np.concatenate(chunks).flatten())

               

                # 3. Decide Kinetic Drive and Intent

                _, drive, kqi, will = self.brain.decide_style(audio_vec)

                speed = self.get_playback_params(audio_vec)



                # 4. Generate Intent & Bi-Objective Search

                z = torch.from_numpy(will).float().to(DEVICE).unsqueeze(0)

                audio_t = torch.from_numpy(audio_vec).float().to(DEVICE).view(1,1,33).repeat(1,64,1)

                label_oh = torch.eye(3, device=DEVICE)[style_idx].unsqueeze(0)

               

                with torch.no_grad():

                    delta = self.gen(z, audio_t, label_oh)

                    push = 15.0 + (drive * 20.0)

                    messy = (self.mean_pose + (delta * push)).squeeze().permute(1, 2, 0).reshape(64, 75)

                    query_intent = messy[0:1]



                # Momentum-Aware Search

                anchor_pose = torch.from_numpy(self.prev_pose).float().to(DEVICE).reshape(1, 75) if self.prev_pose is not None else torch.zeros(1, 75).to(DEVICE)

                global_idx = self.bi_objective_rag_search(query_intent, anchor_pose, self.current_momentum, style_idx)

               

                # 5. Push Move to Performance Queue

                self.move_queue.put({

                    "style": style_idx,

                    "global_idx": global_idx,

                    "speed": speed,

                    "kqi": kqi,

                    "audio": audio_vec

                })



    def start(self):

        cv2.namedWindow("SASAKI-GAN CONDUCTOR", cv2.WINDOW_NORMAL)

       

        # Launch Brain Thread

        threading.Thread(target=self.brain_thread_loop, daemon=True).start()



        # --- FIX: ROBUST DEVICE QUERY ---
        try:
            # Query without kind='input' because Loopback is technically an Output-Input hybrid
            dev_info = sd.query_devices(INPUT_DEVICE)
            # WASAPI Loopback REQUIRES the hardware's native channel count (usually 2)
            ch = int(dev_info['max_input_channels'])
            if ch == 0: # If it claims 0, it's definitely a loopback of an output
                ch = int(dev_info['max_output_channels'])
            
            print(f"🎧 Loopback Active: {dev_info['name']}")
            print(f"📊 Hardware Channels: {ch} | Rate: {SAMPLING_RATE}Hz")
        except Exception as e:
            print(f"⚠️ Device Query Warning: {e}. Defaulting to Stereo.")
            ch = 2

        def audio_callback(indata, frames, time, status):
             self.audio_queue.put(indata.copy())
             # IMMEDIATELY update the live perception for the Conductor
             features, matrix = self.audio_engine.get_precise_features(indata.flatten())
             self.latest_live_audio = features
             self.latest_mfcc_matrix = matrix

        with sd.InputStream(device=INPUT_DEVICE, samplerate=SAMPLING_RATE, channels=ch, callback=audio_callback):

            print("🚀 PERFORMANCE ACTIVE. ZERO-LATENCY FLOW.")

           

            while True:

                # 1. Wait for the Brain to provide a pre-calculated Move

                try:

                    move = self.move_queue.get(timeout=5.0)

                except queue.Empty:

                    print("⚠️ Waiting for Brain Thread...")

                    continue



                global_idx = move["global_idx"]

                speed = move["speed"]

                prev_move_end = torch.from_numpy(self.prev_pose).float().to(DEVICE).reshape(25, 3) if self.prev_pose is not None else None



                # 2. Execute the 64-frame sequence with Dynamic Continuity

                f_pointer = 0.0

                while f_pointer < 64:

                    f_idx = int(f_pointer)

                    f_pointer += max(0.2, speed) # Advance pointer (Min 0.2 ensures move doesn't freeze)

                    if f_idx >= 64: break

                   

                    # 3. Get Pose & Apply Global Root Velocity (CondMDI Inertial Drift)

                    raw_pose = self.db_poses[(global_idx + f_idx) % len(self.db_poses)].reshape(25, 3)

                   

                    if f_idx == 0 and self.prev_pose is not None:

                        # Align hips to prevent ghost-snapping

                        target_offset = self.prev_pose[0] - raw_pose[0]

                        # EMA Filter for smooth root transition

                        self.root_offset = 0.8 * getattr(self, 'root_offset', target_offset) + 0.2 * target_offset

                   

                    # 4. Kinetic Gum Blending

                    if f_idx < 15 and prev_move_end is not None:

                        m = f_idx / 15.0

                        c_tensor = torch.from_numpy(raw_pose).float().to(DEVICE)

                        render_pose = (m * c_tensor + (1 - m) * prev_move_end).cpu().numpy()

                    else:

                        render_pose = raw_pose



                    # 5. Transform & Velocity Tracking

                    final_output = self.post_processor.normalize_pose(render_pose + getattr(self, 'root_offset', 0))

                   

                    if self.prev_pose is not None:

                        mom = (final_output - self.prev_pose).flatten()

                        self.current_momentum = torch.from_numpy(mom).float().to(DEVICE).unsqueeze(0)



                    # 6. Render Dashboard & Skeleton (Use LIVE Audio)
                    img = np.zeros((800, 1200, 3), dtype=np.uint8) + 20
                    
                    # Update parameters based on REAL-TIME MIC data
                    live_speed = self.get_playback_params(self.latest_live_audio)
                    # Use live speed for conducting
                    speed = live_speed 

                    img = self.draw_dashboard(img, move["kqi"], move["style"], speed, self.latest_live_audio)
                    
                    # 7. UDP STREAM TO 3D ENGINE
                    self.streamer.send_pose(final_output)

                    coords = (final_output[:, :2] * 450 + 400).astype(int)

                   

                    for u, v in BONES:

                        cv2.line(img, (coords[u,0], coords[u,1]), (coords[v,0], coords[v,1]), (0, 255, 120), 3, cv2.LINE_AA)

                   

                    cv2.imshow("SASAKI-GAN CONDUCTOR", img)

                    self.prev_pose = final_output.copy()

                   

                    # Manual Kill Switch

                    if cv2.waitKey(30) == ord('q'):

                        self.is_running = False

                        return



if __name__ == "__main__":

    SasakiMasterControlCenter().start()
```

<div style='page-break-after: always;'></div>

## File: audio_brace_loader.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\audio_brace_loader.py`

```python
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
import pandas as pd

class AudioBraceDataset(Dataset):
    def __init__(self, skeleton_path, audio_dir, fps=30):
        print(f"Loading Synced Skeleton Data from: {skeleton_path}")
        self.skel_data = np.load(skeleton_path)
        
        self.skeleton_data = self.skel_data['x_data']
        self.video_ids = self.skel_data['video_ids']
        self.seq_indices = self.skel_data['seq_indices']
        self.audio_offsets = self.skel_data['audio_offsets']
        self.labels = self.skel_data['labels']
        
        # Load FPS Metadata (New)
        if 'fps' in self.skel_data:
            self.fps_data = self.skel_data['fps']
        else:
            print("Warning: FPS metadata not found. Defaulting to 30.")
            self.fps_data = np.full(len(self.video_ids), 30.0)

        self.audio_dir = audio_dir
        
        print("Indexing segment audio files...")
        self.audio_map = {}
        all_audio_files = glob.glob(os.path.join(audio_dir, "**", "*.npz"), recursive=True)
        for path in all_audio_files:
            fname = os.path.basename(path)
            parts = fname.replace(".npz", "").split(".")
            if len(parts) >= 2:
                vid_id = parts[0]
                try:
                    seq_idx = int(parts[1])
                    self.audio_map[(vid_id, seq_idx)] = path
                except: continue
        
        print(f"Found {len(self.audio_map)} mapped audio segments.")
        
    def __len__(self):
        return len(self.skeleton_data)

    def __getitem__(self, index):
        skeleton = self.skeleton_data[index]
        label = self.labels[index]
        
        vid_id = self.video_ids[index]
        seq_idx = self.seq_indices[index]
        audio_offset = self.audio_offsets[index]
        video_fps = self.fps_data[index]
        
    def __getitem__(self, index):
        skeleton = self.skeleton_data[index]
        label = self.labels[index]
        
        vid_id = self.video_ids[index]
        seq_idx = self.seq_indices[index]
        audio_offset = self.audio_offsets[index]
        
        audio_path = self.audio_map.get((vid_id, seq_idx))
        TARGET_FRAMES = 64
        
        # Audio Sampling Rate Investigation Result:
        # Majority of dataset features are 1:1 mapped to Video Frames.
        # (25fps video -> 25Hz audio, 30fps video -> 30Hz audio).
        # Therefore, direct indexing is the most robust method.
        # Mean Lag was -1.66 frames. Applying +2 correction.
        
        if audio_path:
            try:
                audio_data = np.load(audio_path)
                
                mfcc = audio_data['mfcc'].squeeze().T 
                chroma = audio_data['chroma_cqt'].squeeze().T
                onset = audio_data['onset_env'].ravel()[:, np.newaxis]
                combined = np.hstack([mfcc, chroma, onset]) # (Total_T, 33)
                
                # Correction based on Audit v5 (Mean Lag -35)
                # Motion is Early -> We need to shift audio retrieval LATER to match.
                SYSTEMIC_CORRECTION = 35 
                
                # Raw start index
                raw_start = audio_offset + SYSTEMIC_CORRECTION
                
                # Safety Clamping (Prevent Crash on Short Audio)
                max_start = len(combined) - TARGET_FRAMES
                start_idx = max(0, min(raw_start, max_start))
                end_idx = start_idx + TARGET_FRAMES
                
                crop = combined[start_idx : end_idx]
                
                # Final check for length (padding if absolutely necessary)
                if len(crop) < TARGET_FRAMES:
                    pad_amt = TARGET_FRAMES - len(crop)
                    crop = np.pad(crop, ((0, pad_amt), (0, 0)), mode='constant')
                
                audio_vector = torch.from_numpy(crop).float()

            except Exception as e:
                audio_vector = torch.zeros(TARGET_FRAMES, 33)
        else:
            audio_vector = torch.zeros(TARGET_FRAMES, 33)

        skeleton = torch.from_numpy(skeleton).float()
        return skeleton, audio_vector, label



if __name__ == "__main__":
    SKELETON_PATH = './data/brace/BRACE_synced_v2.npz'
    AUDIO_DIR = './brace/audio_features/'
    
    if os.path.exists(SKELETON_PATH):
        ds = AudioBraceDataset(SKELETON_PATH, AUDIO_DIR)
        print(f"Dataset Length: {len(ds)}")
        s, a, l = ds[0]
        print(f"Sample 0 Shapes - Skeleton: {s.shape}, Audio: {a.shape}")
        print(f"Sample 0 Meta - Vid: {ds.video_ids[0]}, Seq: {ds.seq_indices[0]}, Offset: {ds.audio_offsets[0]}")
    else:
        print(f"Synced dataset not found. Run build_brace_v2.py first.")



```

<div style='page-break-after: always;'></div>

## File: calc_mean_pose.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\calc_mean_pose.py`

```python

import numpy as np

DATA_PATH = './data/brace/BRACE_synced_v2.npz'
OUTPUT_PATH = './data/brace/mean_pose.npy'

def calc_mean():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    x_data = data['x_data'] # (N, 3, 64, 25, 1)
    
    # Calculate Mean across Batch (0) and Time (2)
    # Result: (3, 25, 1) -> (3, 25)
    mean_pose = np.mean(x_data, axis=(0, 2, 4))
    
    print(f"Mean Pose Shape: {mean_pose.shape}")
    print(f"Sample Joint 0 (Nose): {mean_pose[:, 0]}")
    
    np.save(OUTPUT_PATH, mean_pose)
    print(f"Saved Mean Pose to {OUTPUT_PATH}")

if __name__ == "__main__":
    calc_mean()

```

<div style='page-break-after: always;'></div>

## File: SASAKI/Improvisation_SLM.py
**Location:** `c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\SASAKI/Improvisation_SLM.py`

```python
from dataclasses import dataclass, asdict
from enum import Enum
import math
import random
from typing import List, Dict, Optional


class State(str, Enum):
    F = "F"  # 前進
    B = "B"  # 後退
    S = "S"  # 停滞


@dataclass
class SLMConfig:
    # 自由パラメータ6個（残りは 1 - (p_xy + p_xz)）
    p_ff: float = 0.6
    p_fb: float = 0.2
    p_bf: float = 0.3
    p_bb: float = 0.5
    p_sf: float = 0.4
    p_sb: float = 0.4

    temperature_tau: float = 1.0    # ゆらぎ量
    skill_level: float = 0.5       # 停滞のコントロール力
    learning_rate_eta: float = 0.1 # 学習率

    def transition_matrix(self):
        """
        3x3行列 P[state_from][state_to]
        行の残りは 1 - (p_xy + p_xz) で S行きにする。
        順番は [F, B, S]
        """
        # F行
        p_fs = max(0.0, 1.0 - (self.p_ff + self.p_fb))
        # B行
        p_bs = max(0.0, 1.0 - (self.p_bf + self.p_bb))
        # S行
        p_ss = max(0.0, 1.0 - (self.p_sf + self.p_sb))

        return {
            State.F: {State.F: self.p_ff, State.B: self.p_fb, State.S: p_fs},
            State.B: {State.F: self.p_bf, State.B: self.p_bb, State.S: p_bs},
            State.S: {State.F: self.p_sf, State.B: self.p_sb, State.S: p_ss},
        }


@dataclass
class SLMState:
    current_state: State = State.F
    internal_clock: float = 0.0
    last_entropy: float = 0.0
    step_index: int = 0


@dataclass
class LogEntry:
    t: int
    state_from: State
    state_to: State
    entropy: float
    reward: float
    note: str = ""


class ImprovisationSLM:
    """
    3状態マルコフ連鎖 + 温度パラメータ + 熟練度ゲート で構成された
    必要最小限の即興エンジン。
    """

    def __init__(self, config: SLMConfig):
        self.config = config
        self.log: List[LogEntry] = []

    @staticmethod
    def softmax(logits, tau: float):
        if tau <= 0:
            # τ→0 なら argmax に近づける
            max_idx = max(range(len(logits)), key=lambda i: logits[i])
            out = [0.0] * len(logits)
            out[max_idx] = 1.0
            return out

        exps = [math.exp(l / tau) for l in logits]
        s = sum(exps)
        if s == 0:
            return [1.0 / len(logits)] * len(logits)
        return [e / s for e in exps]

    def compute_context_entropy(
        self,
        audio_features,
        pose_features,
    ) -> float:
        """
        シャノンエントロピーの超ざっくり版。
        実際には BRACE の音特徴・ポーズ変化量から確率分布を作って
        H = - Σ p log p を計算するイメージ。
        ここでは placeholder 実装。
        """
        # 例: 正規化エネルギーっぽいものをまとめて 0〜1 に押し込む
        val = float(abs(audio_features)) + float(abs(pose_features))
        # 適当にスケーリングして 0〜1 くらいに
        return max(0.0, min(1.0, val / 10.0))

    def step(
        self,
        state: SLMState,
        audio_entropy_source,
        pose_entropy_source,
        reward: float = 0.0,
    ) -> SLMState:
        """
        SLM を1ステップ進める。
        - audio_entropy_source / pose_entropy_source は
          BRACEから切ったウィンドウの特徴量から計算した指標を想定。
        """
        cfg = self.config
        P = cfg.transition_matrix()

        # 文脈エントロピー（本当はここでシャノンエントロピーを計算）
        context_entropy = self.compute_context_entropy(
            audio_entropy_source, pose_entropy_source
        )

        # 温度をエントロピーで少し動かす例
        # エントロピーが高いほどバラける
        tau = cfg.temperature_tau * (0.5 + context_entropy)

        # 現状態の遷移確率を logits にして softmax
        row = P[state.current_state]
        states = [State.F, State.B, State.S]
        base_probs = [row[s] for s in states]
        # logを取ることで softmax の入力に
        logits = [math.log(max(p, 1e-8)) for p in base_probs]

        probs = self.softmax(logits, tau)

        # 熟練度で S 行きの確率をゲート
        # skill=0 → Sはほぼ出ない
        # skill=1 → Sの確率そのまま
        idx_S = states.index(State.S)
        probs[idx_S] *= cfg.skill_level

        # 正規化し直す
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)

        # サンプリング
        r = random.random()
        cum = 0.0
        next_state = states[-1]
        for s_state, p in zip(states, probs):
            cum += p
            if r <= cum:
                next_state = s_state
                break

        # ログ
        self.log.append(
            LogEntry(
                t=state.step_index,
                state_from=state.current_state,
                state_to=next_state,
                entropy=context_entropy,
                reward=reward,
            )
        )

        # 学習（超シンプルな報酬付き微調整；本格的なRLにするならここ拡張）
        if reward != 0.0:
            # 報酬が正 → 選んだ遷移の確率を少し上げる、負 → 下げる
            eta = cfg.learning_rate_eta
            from_row = P[state.current_state]
            old_p = from_row[next_state]
            delta = eta * reward
            new_p = max(0.0, min(1.0, old_p + delta))
            # 簡易的に、選んだ遷移だけ書き換えて normalize は省略
            if state.current_state == State.F and next_state == State.F:
                cfg.p_ff = new_p
            elif state.current_state == State.F and next_state == State.B:
                cfg.p_fb = new_p
            elif state.current_state == State.B and next_state == State.F:
                cfg.p_bf = new_p
            elif state.current_state == State.B and next_state == State.B:
                cfg.p_bb = new_p
            elif state.current_state == State.S and next_state == State.F:
                cfg.p_sf = new_p
            elif state.current_state == State.S and next_state == State.B:
                cfg.p_sb = new_p

        # 状態更新
        return SLMState(
            current_state=next_state,
            internal_clock=(state.internal_clock + 0.25) % 1.0,
            last_entropy=context_entropy,
            step_index=state.step_index + 1,
        )

```

<div style='page-break-after: always;'></div>

