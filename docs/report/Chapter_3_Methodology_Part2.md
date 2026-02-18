
## 3.4 System Architecture

The overall system architecture is designed as a **Neuro-Symbolic Pipeline** consisting of three main stages: **Perception (Ears)**, **Cognition (Brain)**, and **Action (Body)**.

### 3.4.1 Perception: HPI-GCN-OP Backbone
The "Critic" (Discriminator) of our GAN is based on the **HPI-GCN-OP** architecture.
*   **Input Data:** Skeletal sequences of $T=64$ frames (approx. 2 seconds).
*   **Topology:** A graph of $V=25$ nodes (NTU-120 layout).
*   **Layer Structure:**
    1.  **Over-Parameterization Block:** A specialized graph convolution layer that learns two separate weight matrices—one for the physical connections (bones) and one for the global correlations (e.g., hand-foot synchro).
    2.  **Re-Parameterization Block:** During inference, these matrices are summed element-wise into a single $25\times25$ adjacency matrix.
    3.  **Output:** A 256-dimensional latent feature vector $z_{motion}$ that encapsulates the "style" of the movement.

This backbone serves two roles:
1.  **Orchestrator:** It provides the "Feature Space" for our Retrieval Decoder.
2.  **Teacher:** It provides the gradient feedback during GAN training to ensure physical realism.

### 3.4.2 Action: Neural Retrieval Decoder
A common issue in generative animation is **"Bone Stretching"**—where the AI generates coordinates that imply a bone has grown or shrunk. To solve this, we discarded direct coordinate generation in favor of a **Retrieval-based** approach.

**The Algorithm:**
1.  **Intent Generation:** The Neuro-Symbolic GAN outputs a target feature vector $v_{intent} \in \mathbb{R}^{256}$. This represents "Ideal Motion."
2.  **Manifold Search:** We utilize a **FAISS** (Facebook AI Similarity Search) index containing 50,000 encoded clips from the BRACE dataset.
    $$ \text{idx} = \arg\min_i || v_{intent} - v_{database}^{(i)} ||_2 $$
3.  **Momentum Locking:** To prevent the character from jittering (e.g., retrieving a "Turn Left" clip immediately after a "Turn Right" clip), we enforce a **Momentum Penalty**.
    $$ \text{Cost} = ||v_{intent} - v_i|| + \lambda \cdot (1 - \text{Cosine}(v_{prev\_velocity}, v_{new\_velocity})) $$
    If the new clip reverses the direction of momentum, its cost skyrockets, and the system chooses a smoother alternative.
4.  **Blending:** Once a clip is selected, we perform **Spherical Linear Interpolation (SLERP)** over 5 frames to stitch it seamlessly to the current pose.

**Result:** The character *never* displays invalid anatomy because every frame displayed is a real human frame. The AI controls the *sequence*, not the *pixels*.

### 3.4.3 Feature Engineering: Real-Time Audio
The system uses `PyAudio` to capture the **WASAPI Loopback** (System Audio) stream at 48kHz.
We process the audio in rolling windows of 1024 samples (approx. 21ms).

**Feature Vector Construction:**
We extract a 33-dimensional feature vector for every frame:
*   **Indices 0-19 (MFCC):** 20 coefficients representing the spectral envelope (Instrument type).
*   **Indices 20-31 (Chroma):** 12 semi-tones representing the key/harmony.
*   **Index 32 (Onset Strength):** A scalar representing percussive attack.

**Normalization Strategy:**
Raw audio values fluctuate wildly (quiet intro vs loud drop). We implemented a Z-Score normalization that adapts over a 5-second sliding window.
$$ x_{norm} = \frac{x - \mu_{window}}{\sigma_{window} + 1e^{-6}} $$
This ensures that the AI perceives "Relative Change" rather than absolute volume. A quiet song with a sudden violin swell can trigger a "Drop" just as effectively as a loud EDM track, preserving the "Artistic Quality" of interpretation.

---
