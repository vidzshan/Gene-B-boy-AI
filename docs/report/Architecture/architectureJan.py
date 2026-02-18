import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fixed_final_architecture():
    """
    SASAKI-GAN Architecture: Final Layout Fix with Fine-tuning Labels.
    - Extended x-axis to 16 to prevent 'Final Motion' overlap.
    - Added 'Fine-tuned: BRACE' to the Discriminator.
    - Rerouted Inference Query for maximum clarity.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11

    # Extended width to 16 for better horizontal spacing
    FIG_SIZE = (16, 10) 
    BG_COLOR = 'white'
    TEXT_COLOR = '#111111'
    BORDER_COLOR = '#2c3e50'
    BORDER_WIDTH = 1.6

    # Palette
    COLOR_AUDIO = '#e8f5e9' 
    COLOR_LOGIC = '#eeeeee' 
    COLOR_GEN   = '#e3f2fd' 
    COLOR_DISC  = '#ffebee' 
    COLOR_LOSS  = '#fff3e0' 

    ARROW_PROPS = dict(arrowstyle='-|>', mutation_scale=20, lw=1.3, color='#34495e', shrinkA=5, shrinkB=5)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_aspect('equal')
    ax.set_xlim(0, 16) # Extended canvas
    ax.set_ylim(0, 10) 
    ax.axis('off')
    fig.patch.set_facecolor(BG_COLOR)

    def draw_box(x, y, w, h, label, sublabel="", color='#f5f5f5', extra=""):
        # Shadow effect
        ax.add_patch(patches.Rectangle((x+0.05, y-0.05), w, h, facecolor='#bdc3c7', edgecolor='none', alpha=0.3))
        # Main Box
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=BORDER_COLOR, linewidth=BORDER_WIDTH))
        
        # Label text
        ax.text(x + w / 2, y + h / 2 + 0.25, label, ha='center', va='center', fontweight='bold', fontsize=12)
    
        if sublabel:
            # FIXED: Added 'sublabel' as the third positional argument
            ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha='center', va='center', fontsize=9, style='italic')
        
        if extra:
            # Extra labels (like "Pre-trained")
            ax.text(x + w / 2, y + 0.3, extra, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#c0392b')
    
    def draw_arrow(start, end, label="", offset=0.35, style='solid', rad=0.0, color='#34495e', text_align='center'):
        style_props = dict(ls=style, connectionstyle=f"arc3,rad={rad}")
        ax.annotate("", xy=end, xytext=start, arrowprops={**ARROW_PROPS, **style_props, 'color': color})
        if label:
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + offset + rad, label, ha=text_align, va='center', fontsize=9.5, 
                    fontweight='bold', bbox=dict(fc='white', ec='none', pad=1, alpha=0.9))

    # ==========================================
    # ROW 1 & 2: Core Components
    # ==========================================
    draw_box(0.5, 7.5, 2.5, 1.5, "Audio Features", "MFCC, Chroma, Onsets", color=COLOR_AUDIO)
    draw_box(4.0, 7.5, 3.0, 1.5, "Generative Logic", "Stochastic Policy (SLM)", color=COLOR_LOGIC)
    draw_arrow((3.0, 8.25), (4.0, 8.25), "Spectral Flux")
    
    draw_box(8.0, 5.0, 2.8, 3.5, "Generator (Artist)", "Recurrent GRU-RNN", color=COLOR_GEN)
    
    # Updated Discriminator Box with Fine-tuning Note
    draw_box(12.0, 5.5, 3.2, 2.2, "Discriminator (Critic)", "HPI-GCN-OP", 
             color=COLOR_DISC, extra="Pre-trained: NTU-120\nFine-tuned: BRACE")
    
    draw_arrow((7.0, 8.25), (8.0, 8.0), "Intent Vector", rad=0.1)
    draw_arrow((10.8, 7.0), (12.0, 7.0), "Skeleton $\hat{x}$")
    
    # ==========================================
    # ROW 3: Training / Loss
    # ==========================================
    draw_box(4.0, 2.5, 6.0, 1.5, "Multi-Objective Loss Function", "Hybrid Adversarial Learning", color=COLOR_LOSS)
    
    draw_arrow((1.75, 7.5), (4.0, 3.5), "Beat Sync Loss", rad=-0.3, color='#27ae60')
    draw_arrow((13.6, 5.5), (10.0, 3.5), "Kinetic Realism", rad=0.2, color='#c0392b')
    draw_arrow((7.0, 4.0), (8.0, 5.0), "Backpropagation", style='dashed', rad=0.1)

    # ==========================================
    # ROW 4: Inference (Resolved Overlap)
    # ==========================================
    # Moved Neural Retrieval slightly left to 10.5
    draw_box(10.5, 0.5, 3.2, 1.2, "Neural Retrieval", "Kinematic Refinement", color='#fafafa')
    
    # Clearer path for Inference Query
    draw_arrow((9.4, 5.0), (10.5, 1.3), "Inference Query", rad=-0.15, offset=-0.7) 
    
    # Final Motion text moved to 15.5 to eliminate all overlap
    ax.text(15.5, 1.1, "FINAL\nMOTION", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    draw_arrow((13.7, 1.1), (14.8, 1.1))

    # --- Grouping Titles ---
    ax.text(1.7, 9.5, "I. PERCEPTION", ha='center', fontweight='bold', color='#7f8c8d')
    ax.text(5.5, 9.5, "II. REASONING", ha='center', fontweight='bold', color='#7f8c8d')
    ax.text(9.4, 9.5, "III. SYNTHESIS", ha='center', fontweight='bold', color='#7f8c8d')
    ax.text(13.6, 9.5, "IV. EVALUATION", ha='center', fontweight='bold', color='#7f8c8d')

    plt.suptitle("Figure 2: Hierarchical Neuro-Symbolic Architecture for Improvisational Breaking", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    draw_fixed_final_architecture()