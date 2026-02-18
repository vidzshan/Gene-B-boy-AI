# Chapter 3: Methodology

## 3.1 Theoretical Framework: GBMC Circulation
This research proposes a novel circulation structure for computational choreography called **GBMC (Genesis-Buffer-Motion-Coherence)**. While traditional AI pipelines are linear (Input $\to$ Process $\to$ Output), GBMC is cyclic, mirroring the cognitive process of a human improviser.

### 3.1.1 The Four Stages
1.  **Genesis (G):** The initial impulse. This is the "Subconscious" layer of the system. Even in complete silence, a dancer has a latent desire to move. We model this as a low-frequency oscillator (LFO) that provides a baseline "Will" to the system.
    *   *Input:* Audio Energy ($E$), Audio Velocity ($\Omega$).
    *   *Output:* Raw Drive Vector $v_{drive}$.

2.  **Buffer (B):** The filter. This is the "Conscious" layer. The system receives the impulse from Genesis but does not act immediately. It holds the impulse in a buffer and evaluates the "Atmospheric Tension" (Entropy).
    *   *Logic:* If the Entropy is ambiguous, the Buffer fills up.
    *   *Effect:* This creates "Ma" (The Space). The energy accumulates without physical motion, creating visual tension.

3.  **Motion (M):** The release. When the Buffer overflows (or a specific trigger like a Drop occurs), the potential energy is converted into kinetic energy.
    *   *Action:* The GAN generates a sequence of coordinates.
    *   *Constraint:* The motion magnitude is proportional to the accumulated Buffer pressure.

4.  **Coherence (C):** The verification. This is the "Physical" layer (The Body). The generated motion is checked against anatomical constraints (bone length, joint limits).
    *   *Review:* If the motion is physically impossible (e.g., knee bending backwards), the Coherence layer rejects it and requests a new path from the Buffer.

## 3.2 The KQI (Kinetic Quality Index) Engine
The GBMC framework is powered by the **KQI Engine**, a mathematical model that translates sensory data into decision metrics. The equations are derived from the `gbmc_engine.py` implementation.

### 3.2.1 Variable Definitions
*   $r$ (Range): The amplitude of the audio signal (RMS). Represents "Power."
*   $\omega$ (Omega): The spectral flux variance. Represents "Speed" or "Chaos."
*   $\sigma$ (Sigma): The risk/noise factor.
*   $t$ (Time): The global clock tick.
*   $\rho$ (Rho): The density of motion history (how much gained movement in the last $N$ frames).

### 3.2.2 The Equations
**1. Genesys ($G$):**
The raw drive to move is the product of Power and Speed. We add a baseline constant ($\epsilon=0.1$) to prevent the system from "dying" in total silence.
$$ G = (r + 0.1) \cdot (\omega + 0.1) \cdot 2.0 $$

**2. Quality ($Q$):**
Quality is defined as the Drive squared, inversely proportional to the Risk ($\sigma$). This is a critical inversion: As risk goes down (cleaner signal), Quality goes up quadratically.
$$ Q = \frac{G^2}{\max(\sigma, 0.1)} $$

**3. Depth ($D$):**
Depth is the final decision metric. It combines Quality ($Q$) with Coherence ($C$) and the Motion Gate ($M$).
$$ D = Q \cdot C \cdot M $$
Where $C = e^{\min(t \cdot \rho \cdot 0.1, 2.0)}$. This exponential term means that "The longer the dance goes on ($t$), and the denser it becomes ($\rho$), the deeper the commitment to the next move."

### 3.2.3 Interpretation
*   **Case A (High Energy):** Loud, fast music ($r \uparrow, \omega \uparrow$) results in massive $G$, leading to high $D$. The system triggers a **Powermove**.
*   **Case B (The Drop):** Silence ($r \approx 0$) but with accumulated history ($t \uparrow$). The formula for Quality uses $\max(\sigma, 0.1)$. If the signal is clean (low noise $\sigma$), $Q$ remains stable. Combined with high Coherence $C$, $D$ can remain high even without input. This triggers a **Freeze** (Active Stillness).

## 3.3 Stochastic Logic Module (SLM)
Deep Learning models are deterministic. To introduce "Creative Will," we implemented the **SLM (Improvisation_SLM.py)**.

### 3.3.1 State Transition Matrix
We define three discrete states for the dancer:
1.  **State F (Flow/Forward):** Toprock, Rhythm.
2.  **State B (Burst/Backward):** Power, Acrobatics.
3.  **State S (Stillness/Stagnation):** Freeze, Pause.

The transition probabilities $P_{ij}$ (probability of moving from state $i$ to $j$) are dynamic.
$$ P_{matrix} = \begin{bmatrix} P_{FF} & P_{FB} & P_{FS} \\ P_{BF} & P_{BB} & P_{BS} \\ P_{SF} & P_{SB} & P_{SS} \end{bmatrix} $$

### 3.3.2 Entropy Modulation
The SLM calculates the **Context Entropy** ($H$) of the last 30 frames.
$$ H = - \sum p_i \log p_i $$
*   **If $H > \tau_{high}$ (High Chaos):** The SLM boosts $P_{FB}$ and $P_{BB}$ (Increases chance of Power).
*   **If $H < \tau_{low}$ (High Order):** The SLM boosts $P_{FS}$ and $P_{SS}$ (Increases chance of Freeze).

This logic ensures that the "Decision to Freeze" is probabilistic but biased by the environment. It is not hard-coded "if silence then stop," but rather "if silence then the *temptation* to stop increases." This allows for "Fake-outs" where the dancer ignores the silence and keeps moving—a hallmark of high-level improvisation.
