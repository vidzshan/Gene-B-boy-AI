
# Chapter 10: User Manual and Operational Guide

This chapter provides the necessary documentation for future students or researchers to operate the **Sasaki-GAN** system.

## 10.1 System Requirements
### 10.1.1 Hardware Specifications
*   **CPU:** Intel Core i7-10750H or better (AVX2 support required for NumPy).
*   **GPU:** NVIDIA RTX 3060 (6GB VRAM) minimum. RTX 4070 (8GB) recommended for >30 FPS.
*   **RAM:** 16GB DDR4 (32GB recommended for Unity Editor + Python simultaneous execution).
*   **Audio:** WASAPI-compatible Sound Card (Loopback support required).

### 10.1.2 Software Dependencies
*   **Operating System:** Windows 10/11 (Required for WASAPI).
*   **Python:** Version 3.10.x (Torch 2.0 compatibility).
*   **Blender:** Version 3.6 LTS (Python 3.10 internal).
*   **CUDA:** Toolkit 11.8 (cuDNN 8.5).

## 10.2 Installation Guide
### 10.2.1 Environment Setup
We recommend using `conda` to manage the environment.
```bash
conda create -n gcn python=3.10
conda activate gcn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa sounddevice numpy matplotlib scipy pyyaml
pip install faiss-cpu  # For Retrieval Decoder
```

### 10.2.2 Dataset Preparation
1.  Download the **BRACE Dataset** content (`brace_audio/` and `brace_video/`).
2.  Run the synchronization script:
    ```bash
    python tools/synchronize_brace.py --shift_ms 1160
    ```
    *Note: The 1160ms shift fixes the inherent lag in the original dataset.*
3.  Run the topology fixer:
    ```bash
    python tools/fix_topology.py --target nturgbd120
    ```
    This will generate `BRACE_25J_Fixed.npy`.

## 10.3 Running the System
### 10.3.1 Training Mode
To retrain the model from scratch (Warning: Takes ~72 Hours):
```bash
python train_multimodel_residual.py --config configs/sasaki_residual.yaml
```
**Key Flags:**
*   `--resume_epoch N`: Resume from a checkpoint.
*   `--prison_break`: Enable the Kinetic Penalty (use only after Epoch 20).
*   `--ghost_weight 0.1`: Set the variance of the Latent Will.

### 10.3.2 Live Performance Mode
This is the main interaction mode.
1.  **Start Blender Bridge:**
    *   Open `Sasaki_Visualizer.blend`.
    *   Go to Scripture Editor -> Open `blender_receiver.py`.
    *   Click "Run Script". The console should say "Listening on :5000".
2.  **Start the Brain:**
    ```bash
    python run_sasaki_live.py --mode production
    ```
3.  **Play Music:**
    *   Play any audio on the system (Spotify/YouTube).
    *   The terminal will show the **KQI Dashboard** (Depth, Quality, State).

## 10.4 Troubleshooting
### 10.4.1 Error: "Input Overflow" (PyAudio)
**Symptom:** The terminal spouts "Input Overflow" and the animation stutters.
**Cause:** The `ListenerThread` is processing data slower than the sampling rate (48kHz). This usually happens if `print()` statements are left in the audio callback.
**Fix:** Remove all I/O from the callback. Ensure `AudioQ` has a maxsize (e.g., 1024) and drop packets if full.

### 10.4.2 Error: "Bone Length Explosion"
**Symptom:** The character's arm stretches to infinity.
**Cause:** The GAN generated a NaN (Not a Number) or Infinity value, usually due to a division by zero in the Normalization layer.
**Fix:** We implemented `torch.clamp(val, -10, 10)` in `HPI_GCN` output. If this persists, restart the script; the internal LSTM hidden state might be corrupted.

### 10.4.3 Feature: "The Zombie Drift"
**Symptom:** The character slowly rotates off-center over 10 minutes.
**Cause:** The root velocity accumulation ($P_t = P_{t-1} + V_t$) accumulates floating point error.
**Fix:** The system automatically "Center Resets" if audio energy $<0.01$ (Silence) for more than 5 seconds. You can force a reset by pressing `R` in the OpenCV window.

---

# Chapter 11: Full API Reference

## 11.1 gbmmc_engine.py
### Class `KQIEngine`
**Methods:**
*   `__init__(self)`: Initializes epsilon.
*   `process(r, omega, sigma, t, rho)`: The main calculation loop.
    *   **Returns:** Dictionary `{'G', 'Q', 'M', 'C', 'D'}`.
    *   **Parameters:**
        *   `r` (float): Audio RMS Energy (0.0 - 1.0).
        *   `omega` (float): Spectral Flux Variance.
        *   `sigma` (float): Signal-to-Noise Ratio inverse.

## 11.2 sasaki_brain.py
### Class `SasakiLiveBrain`
**Attributes:**
*   `self.kqi`: Instance of KQIEngine.
*   `self.memory_buffer`: A deque of size 64 storing past latent vectors.
*   `self.current_state`: Enum (Flow/Burst/Still).

**Methods:**
*   `decide_style(audio_vector)`:
    *   Logic:
        1. Check Audio Energy.
        2. If Silent -> Apply Heartbeat (Sinewave).
        3. Else -> Run KQI.
        4. If Depth > 2.0 -> Burst.
        5. Else -> Flow.
    *   **Returns:** `(state, drive, latent_will)`

## 11.3 realtime_audio_engine.py
### Class `RealTimeAudioEngine`
**Attributes:**
*   `self.stream`: PyAudio InputStream.
*   `self.buffer`: Rate-limited Queue.

**Methods:**
*   `callback(in_data, frame_count, time_info, status)`:
    *   Non-blocking callback. Captures raw bytes, converts to Int16, normalizes to Float32 (-1.0 to 1.0).
*   `get_features()`:
    *   Runs Librosa MFCC extraction on the last 2048 samples.
    *   Apply Z-Score Normalization using a running mean/std buffer.

## 11.4 blender_receiver.py (Python-in-Blender)
### Operator `ModalTimerOperator`
This class hooks into Blender's internal event loop.
*   `modal(self, context, event)`: Called every frame.
    1.  Polls UDP socket (Non-blocking).
    2.  If data: Unpacks 75 floats.
    3.  Updates `bpy.data.objects['Bone_Target'].location`.
    4.  Calls `bpy.context.view_layer.update()` to refresh the viewport.
