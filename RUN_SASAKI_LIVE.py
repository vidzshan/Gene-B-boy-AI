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