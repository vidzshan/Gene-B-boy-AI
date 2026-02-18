# Chapter 2: Related Work

## 2.1 Skeleton-Based Action Recognition
In the domain of Computer Vision, understanding human movement has evolved from analyzing pixel-level differences (Optical Flow) to modeling the geometric structure of the body (Skeleton-based approach). Skeleton data, typically represented as a sequence of joint coordinates $(x, y, z)$, is robust to changes in lighting, clothing, and background background, making it ideal for dance analysis.

### 2.1.1 Graph Convolutional Networks (GCNs)
Traditional Deep Learning classifiers like Convolutional Neural Networks (CNNs) were designed for Euclidean data (regular grids like images). However, the human skeleton is a **Graph** structure, where nodes (joints) are connected by edges (bones) in a non-Euclidean topology.

**ST-GCN (Spatial-Temporal GCN):**
Proposed by Yan et al. [6] in 2018, ST-GCN was the first model to apply graph convolution directly to action recognition. The core operation is defined as:
$$ f_{out}(v_i) = \sum_{v_j \in N(v_i)} \frac{1}{Z_{ij}} f_{in}(v_j) \cdot w(l_i(v_j)) $$
Where $N(v_i)$ is the neighbor set of joint $i$, and $Z_{ij}$ is a normalization term. ST-GCN demonstrated that modeling the natural connectivity of the body (e.g., Hand connected to Elbow) significantly outperforms LSTM-based approaches which treat the body as a linear vector.

**HPI-GCN (High-Performance Inference GCN):**
While ST-GCN is accurate, it is computationally heavy due to the complex adjacency matrix multiplications. Liu et al. [5] introduced HPI-GCN to solve the trade-off between accuracy and inference speed.
*   **Structural Re-parameterization (RP):** During training, the model uses a multi-branch architecture to learn diverse features. During inference, these branches are mathematically collapsed into a single convolution kernel.
    $$ W_{fused} = W_{conv} + W_{1x1} + W_{identity} $$
    This allows the deployment model to run as a simple linear stack, drastically reducing FLOPs (Floating Point Operations).
*   **Over-Parameterization (OP):** The "OP" variant introduces additional learnable weight matrices to the adjacency graph, effectively allowing the model to learn "hidden connections" (e.g., the correlation between the Left Hand and Right Foot during a spin, which are not physically connected by a bone).

We selected **HPI-GCN-OP** as the backbone for our project because its lightweight inference structure is the only architecture capable of sustaining >30FPS on a consumer GPU while maintaining the high accuracy needed to distinguish subtle breakdancing moves.

## 2.2 Generative Models for Motion
### 2.2.1 GANs (Generative Adversarial Networks)
Introduced by Goodfellow et al. [7], GANs consist of two competing networks:
*   **Generator ($G$):** Tries to create realistic data from random noise $z$.
*   **Discriminator ($D$):** Tries to distinguish between real data and generated data.

In the context of motion, **MotionGAN** and **LstmGAN** have been explored. However, they suffer from **Mode Collapse**. If the dataset contains multimodal distributions (e.g., for input "Kick," the output can be "High Kick" or "Low Kick"), the Generator often collapses to the mean, producing a "Medium Kick" that looks unrealistic. In dance, this results in "floating" or "sliding" movements because the model averages the foot positions.

### 2.2.2 Diffusion Models
Recently, Denoising Diffusion Probabilistic Models (DDPMs) like **EDGE (Editable Dance Generation)** have set a new standard for quality. They work by iteratively denoising a sequence of Gaussian noise into a clean motion sequence.
While Diffusion models produce superior diversity and physical realism compared to GANs, their **Inference Latency** is a major bottleneck. A standard Diffusion model requires 50-100 denoising steps to generate one second of motion, taking several seconds of computation time. This makes them unsuitable for our **Live Improvisation** objective, where the system must react to a musical beat within milliseconds.

## 2.3 Music Information Retrieval (MIR)
To generate dance, the machine must "hear" the music. Music Information Retrieval (MIR) is the field of extracting meaningful features from audio signals.

### 2.3.1 Feature Extraction Strategies
*   **RMSE (Root Mean Square Energy):** Measures the loudness. Useful for detecting "Drops" or "Breaks."
*   **Spectral Flux:** Measures the rate of change in the power spectrum. High flux indicates percussive onsets (drum hits).
*   **MFCCs (Mel-Frequency Cepstral Coefficients):** A representation of the short-term power spectrum essentially describing the "Timbre" or "Color" of the sound.

Most existing dance generation systems (e.g., AIST++) rely exclusively on **Beat Tracking**—finding the timestamps of the kick drum. However, Müller [8] argues that musicality is more than rhythm; it includes **Dynamics** (swells and fades) and **Structure** (Verse vs. Chorus).
Our algorithm extends this logic. Instead of just "hitting the beat," we calculate the **Entropy of the Feature Vector**.
*   **High Entropy:** Complex, chaotic music $\to$ High variance movement (Toprock).
*   **Low Entropy:** Simple, sustained tones $\to$ Low variance movement (Freeze).

## 2.4 The Research Gap
The intersection of HPI-GCN (Efficiency) and Neuro-Symbolic Logic (Decision Making) remains unexplored in dance generation.
1.  **Efficiency vs. Quality:** Existing high-quality models (Diffusion) are too slow. Fast models (GANs) are too unstable.
2.  **Context Awareness:** Existing models map Audio $\to$ Motion directly. They lack a "Buffer" state to decide *not* to move.
3.  **Topology Mismatch:** Most research uses the NTU-120 topology (25 joints). The best breakdancing dataset (BRACE) uses COCO (17 joints). There is no standard "Adapter" to bridge high-performance GCNs to in-the-wild dance datasets.

Our project fills these gaps by proposing the **GBMC Framework**, which uses HPI-GCN-OP for efficient physics and a Symbolic Logic Module for "Ma"-aware decision making.
