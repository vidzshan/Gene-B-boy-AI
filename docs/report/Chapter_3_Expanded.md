
# Chapter 3: Methodology and Mathematical Framework

This chapter details the theoretical underpinnings of the **Sasaki-GAN** system. We introduce the **GBMC Framework** for cognitive circulation and derive the **KQI Equations** that govern the improvisational restrictions.

## 3.1 The GBMC Framework
Computational creativity often struggles with the "Stop Condition." A standard recursive neural network will generate motion infinitely. To model the active decision to stop (Ma), we propose the **Genesis-Buffer-Motion-Coherence (GBMC)** cycle.

![System Architecture](file:///c:/Users/kos04/OneDrive/Desktop/vidz/GCN/HPI-GCN/img/Architecture.png)
*Fig 3.0: The Complete Sasaki-GAN Architecture, illustrating the flow from Audio Input (Genesis) to the GBMC Cycle and final Output.*

### 3.1.1 Genesis ($G$)
Genesis is the "Will to Move." It is a function of external auditory stimuli.
$$ G(t) = f_{genesis}(E(t), \Omega(t)) $$
Where $E(t)$ is the Energy (RMS) and $\Omega(t)$ is the Spectral Flux.
Crucially, $G(t) > 0$ even if $E(t) = 0$. This represents the "Internal Clock" or the dancer's pulse. We modeled this using a low-frequency sine wave $\sin(0.1t)$ added to the baseline.

### 3.1.2 Buffer ($B$)
The Buffer represents potential energy. It accumulates $G(t)$ over time until a threshold is reached.
$$ B(t) = B(t-1) + G(t) - M(t) $$
Where $M(t)$ is the released Kinetic Energy.
This "Kaplan Turbine" model allows us to simulate **Tension**. During a long silence (Build-up), $G(t)$ is small but positive. $M(t)$ is zero. Thus $B(t)$ rises. When the "Drop" hits, the stored $B(t)$ is released in an explosive burst (Powermove).

![Hypothetical KQI Curve](file:///c:/Users/kos04/OneDrive/Desktop/vidz/GCN/HPI-GCN/img/plot_hypothetical_curve.png)
*Fig 3.2: The Idealized GBMC Cycle. The Blue line represents Audio Energy, while the Red line represents Internal Tension (Buffer). Note how Tension peaks exactly when Energy is lowest (The Drop).*


### 3.1.3 Motion ($M$)
The Kinetic Release. This is where the GAN operates.
$$ M_{state} = \text{SasakiGAN}(z_{latent} | B(t)) $$
The magnitude of the generated motion vector is constrained by the Buffer level.

### 3.1.4 Coherence ($C$)
The physical validity check.
$$ C = \text{PhysicsLoss}(M_{state}) + \text{AnatomyLoss}(M_{state}) $$
If $C$ is too high (invalid motion), the system rejects $M_{state}$ and does not decrement $B(t)$. The energy remains "trapped," forcing the system to try again in the next frame with a different latent seed $z$.

## 3.2 The Kinetic Quality Index (KQI)
To implement GBMC, we derived the **KQI Equations** (as implemented in `gbmc_engine.py`).

### 3.2.1 Derivation of Depth ($D$)
The central variable is "Depth" ($D$), which determines the commitment to a move.
$$ D = Q \cdot C \cdot M $$
$$ Q = \frac{G^2}{\sigma} $$
The variable $\sigma$ denotes **Environmental Risk** (or Noise). In a "Cypher" (Battle), the risk is the judgment of the crowd.
*   **Low Noise (Clear Music):** $\sigma \to 0 \implies Q \to \infty$. The dancer is confident.
*   **High Noise (Ambiguous Music):** $\sigma \to 1 \implies Q \to G^2$. The dancer is hesitant.

This equation provides a mathematical basis for **Confidence**. Confidence is not just having high skill ($G$); it is having high clarity ($1/\sigma$).

## 3.3 The Zero-Padding Topology Bridge
One of the most significant engineering challenges was mapping the 2D COCO dataset to the 3D NTU-120 model.

### 3.3.1 Graph Isomorphism Problem
Let $G_{COCO} = (V_{17}, E_{16})$ and $G_{NTU} = (V_{25}, E_{24})$.
There is no direct isomorphism because $V_{NTU}$ contains nodes like "SpineMid" which are implicit in $V_{COCO}$.

![Joint Topology Mapping](file:///c:/Users/kos04/OneDrive/Desktop/vidz/GCN/HPI-GCN/img/Joint%20Topology%20mapping.png)
*Fig 3.3: Conceptual mapping between the 17-point COCO standard and the 25-point NTU standard.*

### 3.3.2 The Learnable Adjacency Matrix
Instead of a fixed mapping, we used a **Learnable Adjacency Matrix**.
Standard GCN: $Y = \Lambda^{-\frac{1}{2}} (A + I) \Lambda^{-\frac{1}{2}} X W$
HPI-GCN-OP with Learnable Edge:
$$ A_{new} = A_{physical} + A_{learned} $$
We initialized the input tensor $X$ with zeros for the missing 8 joints.
$$ X_{in} = [X_{coco} \oplus 0_{8}] $$
During backpropagation, the gradients flowed into $A_{learned}$. The network discovered that:
*   $Node_{SpineMid} \approx 0.5 \cdot (Node_{L\_Shoulder} + Node_{R\_Hip})$
This effective "interpolation" allowed the 25-joint model to function correctly on 17-joint data without manual feature engineering.

### 3.3.3 Topology Visualization
**Figure 3.4** demonstrates the successful validation of our mapping strategy. On the **Left**, we see the standard NTU-RGB+D skeleton (25 joints). On the **Right**, we display our processed BRACE data, which has been reconstructed from 17 joints to the 25-joint standard. The alignment confirms that the Zero-Padding and Graph Inference successfully hallucinated the missing anatomical features.

![Topology Visualization](file:///c:/Users/kos04/OneDrive/Desktop/vidz/GCN/HPI-GCN/img/joints%20mapped%20into%20this%2025-joint%20structure.png)
*Fig 3.4: Comparison of skeletal topologies. Left: The Target NTU-25 structure. Right: The Reconstructed BRACE structure, showing successful adaptation to the 25-joint format.*

## 3.4 The Neural Retrieval Decoder (RAG)
To prevent bone stretching, we implemented a Retrieval Augmented Generation (RAG) system.

![RAG Neural Retrieval Decoder](file:///c:/Users/kos04/OneDrive/Desktop/vidz/GCN/HPI-GCN/img/RAG%20Neural%20retrivel%20decoder.png)
*Fig 3.5: The Neural Retrieval Mechanism. The GAN outputs a 'Query Vector', which searches the 'Motion Manifest' (Database) for the closest valid human pose.*

## 3.5 The Stochastic Logic Module (SLM)
The SLM (Appendix A.2) is the "Conductor" of the system.
It uses a **Temperature-Scaled Softmax** to decide the next state.
$$ P(S_{t+1}|S_t) = \text{Softmax}(\frac{\log \pi}{\tau}) $$
Where $\tau$ is the Temperature, derived from Entropy.
*   High Entropy Music $\implies$ High $\tau$ $\implies$ Uniform Distribution (Random Motion).
*   Low Entropy Music $\implies$ Low $\tau$ $\implies$ ArgMax Distribution (Deterministically following the beat).
This mimics human behavior: When the beat is clear, we follow it. When the beat is chaotic, we get creative.
