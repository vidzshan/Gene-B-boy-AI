import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_professional_architecture():
    """
    Generates a publication-quality architecture diagram for SASAKI-GAN,
    following conventions seen in top-tier ML research papers.
    """
    # --- Style Configuration ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 8

    FIG_SIZE = (12, 7) # Slightly wider to fit the flow
    BG_COLOR = 'white'
    TEXT_COLOR = '#111111'
    BORDER_COLOR = '#333333'
    BORDER_WIDTH = 1.0

    # Module Colors
    COLOR_GEN = '#eef3ff'      # Light blue for generator path
    COLOR_DISC = '#ffefef'     # Light red for discriminator path
    COLOR_REAL = '#eef7ee'     # Light green for real data
    COLOR_CORE = '#f5f5f5'     # Gray for SASAKI core
    COLOR_INFER = '#fafafa'    # Off-white for inference

    # Arrow Properties
    ARROW_PROPS = dict(
        arrowstyle='-|>',
        mutation_scale=15,
        lw=1.0,
        color='#444444',
        shrinkA=5,
        shrinkB=5
    )

    # --- Diagram Setup ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_aspect('equal')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor(BG_COLOR)

    # --- Helper Functions ---
    def draw_box(x, y, w, h, label, sublabel="", color=COLOR_CORE):
        # Shadow effect
        ax.add_patch(patches.Rectangle((x+0.05, y-0.05), w, h,
                                       facecolor='#dddddd', edgecolor='none', alpha=0.5))
        # Main Box
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       facecolor=color,
                                       edgecolor=BORDER_COLOR,
                                       linewidth=BORDER_WIDTH))
        ax.text(x + w / 2, y + h / 2 + 0.1, label,
                ha='center', va='center',
                fontweight='bold', fontsize=9, color=TEXT_COLOR)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.2, sublabel,
                    ha='center', va='center',
                    fontsize=7, style='italic', color=TEXT_COLOR)

    def draw_arrow(start, end, label="", offset=0, style='solid', rad=0.0):
        style_props = dict(ls=style, connectionstyle=f"arc3,rad={rad}")
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={**ARROW_PROPS, **style_props})
        if label:
            # Simple midpoint calculation
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + offset
            # Adjust label position for curved lines roughly
            if rad != 0: mid_y += rad 
            
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                    fontsize=7, style='italic', color=TEXT_COLOR,
                    bbox=dict(fc='white', ec='none', pad=0, alpha=0.8))

    # ==============================
    # 1. INPUTS (LEFT)
    # ==============================
    draw_box(0.5, 5.0, 1.2, 1.0, "Inputs", "Noise $z$, Label $c$")
    draw_box(0.5, 3.0, 1.2, 1.0, "Context", "Entropy $H$")

    # ==============================
    # 2. THE BRAIN (SASAKI & GEN)
    # ==============================
    # Sasaki Logic
    draw_box(2.5, 3.2, 1.8, 0.8, "Generative Logic", "Stochastic Policy")
    draw_arrow((1.7, 3.5), (2.5, 3.5), "Entropy Signal")

    # Generator
    draw_box(2.5, 5.0, 1.8, 1.0, "Generator", "GRU-RNN", color=COLOR_GEN)
    
    # Arrows to Generator
    draw_arrow((1.7, 5.5), (2.5, 5.5), rad=0) # From Inputs
    draw_arrow((3.4, 4.0), (3.4, 5.0), "Intent Vector", rad=0) # From Sasaki

    # ==============================
    # 3. TRAINING LOOP (TOP RIGHT)
    # ==============================
    # Real Data Source
    draw_box(6.0, 6.0, 1.5, 0.8, "Real Data", "NTU/BRACE", color=COLOR_REAL)

    # Discriminator
    draw_box(8.5, 5.0, 1.8, 1.0, "Discriminator", "HPI-GCN-OP", color=COLOR_DISC)

    # Flows to Discriminator
    draw_arrow((4.3, 5.5), (8.5, 5.5), "Raw Trajectory (Fake)", rad=0)
    draw_arrow((7.5, 6.4), (9.4, 6.0), "Real Samples", rad=-0.2)

    # Losses
    draw_arrow((9.4, 5.0), (3.4, 6.0), "Hybrid Loss\n(Adv + Class + Div)", style='dashed', rad=0.3)

    # ==============================
    # 4. INFERENCE PIPELINE (BOTTOM)
    # ==============================
    # Container Box for Inference
    ax.add_patch(patches.Rectangle((4.8, 0.5), 6.5, 2.5, ls='dashed', ec='#999999', fc='none', lw=1))
    ax.text(5.0, 2.8, "Inference / Production Pipeline", ha='left', va='top', fontsize=8, fontweight='bold', color='#555')

    # Motion Matching
    draw_box(5.5, 1.0, 2.0, 1.0, "Neural Retrieval", "Motion Matching", color=COLOR_INFER)
    
    # Smoothing
    draw_box(8.5, 1.0, 2.0, 1.0, "Kinematics", "Savitzky-Golay", color=COLOR_INFER)

    # Connections to Inference
    # 1. Generator to Retrieval (The Query)
    draw_arrow((4.3, 5.2), (5.5, 1.8), "Latent Query", rad=-0.1)
    
    # 2. Real Data to Retrieval (The Codebook)
    draw_arrow((6.75, 6.0), (6.5, 2.0), "Reference DB", style='dotted', rad=-0.2)

    # 3. Retrieval to Smoothing
    draw_arrow((7.5, 1.5), (8.5, 1.5), "Matched Pose")

    # ==============================
    # 5. OUTPUT (FAR RIGHT)
    # ==============================
    ax.text(11.2, 1.5, "FINAL\nANIMATION", ha='center', va='center', fontsize=9, fontweight='bold')
    draw_arrow((10.5, 1.5), (10.9, 1.5))

    # --- Title ---
    plt.suptitle("Figure 1: GAN Neuro-Symbolic Architecture", fontsize=14, fontweight='bold', y=0.98)

    # --- Display and Save ---
    plt.tight_layout()
    plt.savefig("SASAKI_GAN_Professional_Architecture.png", dpi=300, bbox_inches='tight')
    print("✅ Professional architecture diagram saved: SASAKI_GAN_Professional_Architecture.png")
    plt.show()

if __name__ == '__main__':
    draw_professional_architecture()