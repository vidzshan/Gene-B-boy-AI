# Transfer Learning to BRACE

Status: Not started
Summary: What you have covered:
• Set your transfer strategy: "Pre-train, Replace Head, Fine-tune" approach.
• Saved best epoch weights for transfer.
• Addressed skeleton topology mismatch (NTU 25 joints vs. BRACE, likely COCO 17 joints) and recommended zero-padding.
• Outlined target data shape: (N, 3, 64, 25, 1).
• Code modifications to load pretrained weights, freeze early layers, and replace the classifier head.
• Noted how HPI-GCN can later function as a "critic" for generative models.
• Listed immediate action items: data conversion, class definitions, joint mapping.
Parent-task: HPI-GCN-OP (https://www.notion.so/HPI-GCN-OP-2af1439453128082b0a9ee7a948c5ec3?pvs=21)
Sub-tasks: X SECRET (https://www.notion.so/X-SECRET-2b2143945312809a9046ee87a31e245a?pvs=21), Zero-Padding (https://www.notion.so/Zero-Padding-2b21439453128048a200dd620d9cd299?pvs=21), DISCARD GCN (https://www.notion.so/DISCARD-GCN-2b214394531280ad8be3d32e4d6f752d?pvs=21), file structure of Brace (https://www.notion.so/file-structure-of-Brace-2871439453128059ad1fe939a417603b?pvs=21), Brace dance data (https://www.notion.so/Brace-dance-data-27f1439453128064860ff50004b11a33?pvs=21), Zero-Padding-Hack (https://www.notion.so/Zero-Padding-Hack-2bd143945312804bb6f9fd92dd96c0e9?pvs=21), The Data Bridge (Audio-Video Alignment) (https://www.notion.so/The-Data-Bridge-Audio-Video-Alignment-2df1439453128077879bc0ecd1b7df7c?pvs=21)

## Stretagy

- Pre-train, Replace Head, Fine-tune

*Fine-tune on BRACE*.

BRACE is a smaller dataset than NTU, Let’s use the **"Pre-train, Replace Head, Fine-tune"** strategy.

![](https://media1.giphy.com/media/v1.Y2lkPTc5NDFmZGM2YWpjY3Jsb2V5YTZmMG5xcXF5eWFoNXkwem5naGx2bDc0MWYwb3I5MSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/lUedOXZaqNJoqHkmFP/giphy.gif)

### Phase 2.1: Save Your Weights

Ensure you have the `.pt` file from the best epoch.

- Look in: `work_dir/ntu120/xsub/HPI_GCN_OP_joint0/runs/`
- Find the file corresponding to **Epoch 48** (since that was the file you have best accuracy).
- Rename it to `ntu120_pretrained.pt` and move it to a safe folder.(pretrained_model)
    - Test 1:
        
        **Goal:**
        
        Verify the file isn't corrupted and that the architecture matches your weights.
        
        **Method:**
        
        We will feed it "dummy" data (random noise) just to see if it crashes.
        
        sanity_check.py
        
        - -- STARTING SANITY CHECK ---
        [1] Building Model Architecture...
        [2] Loading weights from ./pretrained_models/ntu120_pretrained.pt...
        ✔ Weights loaded successfully!
        [3] Generated dummy input shape: torch.Size([1, 3, 64, 25, 2])
        [4] Running Forward Pass...
        ✔ Inference Successful!
        Output Shape: torch.Size([1, 120]) (Should be [1, 120])
        Predicted Class (Random Noise): 117
        --- SANITY CHECK PASSED ---
    - Test 3: Understanding the Output (For your Thesis/Research)
        
        You are about to move to Domain Adaptation. It is crucial to understand what the output actually *is* right now.
        
        Your model currently outputs a vector of size **120**. These correspond to the NTU-120 classes.
        
        - Class 0: Drink water
        - Class 12: Writing
        - Class 50: Kicking other person
        - Class 119: ...
        
        **Why this matters for BRACE (Breakdancing):**
        
        When we start "Transfer Learning" in the next phase, we are going to:
        
        1. Keep the **Backbone** (which understands physics, joints, gravity).
        2. **Cut off** the "Head" (the final layer that outputs 120 numbers of classes).
        3. **Attach** a new "Head" (that outputs BRACE classes: e.g., Powermove, Toprock and Footwork).
        
        ### Next Immediate Action
        
        Run **Test 1 (Sanity Check)** first.
        
        If it passes, reply with **"Sanity Check Passed."**
        
        **Domain Adaptation**, where we will restructure the model to ingest your BRACE data.
        
        **visualize the "Brain" in action.**
        
        We can write a script that:
        
        1. Takes a raw skeleton sequence from your dataset.
        2. Feeds it to your model.
        3. **Animates the skeleton** on your screen (stick figure).
        4. Prints what the model "thinks" the action is (e.g., "Predicted: Jump Kick").
            
            ### What this proves for your Research
            
            When you see the skeleton moving and the model correctly labeling it, you are witnessing **Feature Extraction**.
            
            The model is looking at the trajectory of the wrist joint relative to the head joint and calculating: *"The probability of this trajectory matching 'Drink Water' is 99%."*
            
            **For Breakdancing (Phase 3):**
            
            We will eventually use this same tool, but instead of seeing "Drink Water," you will see a skeleton doing a **Windmill**, and the model will output **"Power Move: Windmill."**
            
            `ntu_demo_view.py`:
            
            Features:
            
            - 2D visualization
                - Standard X/Y plane view. This is often faster to render and easier to verify for dance forms (checking silhouettes).
            - Multi-View Visualization
                
                To fix this, we need to see the skeleton from all angles to find the correct "Front View."
                
                I have updated the code to open **3 Windows side-by-side**:
                
                1. **Front View** (X vs Y)
                2. **Top View** (X vs Z)
                3. **Side View** (Z vs Y)
                
                This script also includes **Auto-Scaling**, so the dancer will never be too big or too small for the screen.
                
            
            ![image.png](image.png)
            
            ![image.png](image%201.png)
            
            - **Front View:** Shows a clearly recognizable human skeleton. The arms, legs, and spine are connected correctly.
            - **Geometry:** The "Spiderweb" effect is gone. The fix to `ensure_4d_shape` worked.
            - **Prediction:** It guessed "Class 74" while the truth was "Class 98". This is normal for a pre-trained model on a random sample without context, but the *physics* are now correct.

### Phase 2.2: The Topology Mismatch (Critical Risk)

**The Risk:** NTU uses Kinect format (25 joints). BRACE likely uses COCO format (17 joints).

**The Fix:** Align the BRACE data to the NTU graph structure before feeding it to the model.

If BRACE is **COCO (17 joints)**, you have two options:

1. **Zero-Padding (Easier):** Map the 17 COCO joints to the corresponding NTU indices and fill the missing 8 joints (like spine, tips of hands) with zeros.
2. **Graph Redefinition (Harder):** Change the `A` matrix in the code to support 17 nodes instead of 25.

*Recommendation for now:* Use **Zero-Padding** (Joint Mapping). It allows you to reuse the weights you just spent 2 days training without breaking the matrix shapes.

### Phase 2.3: Preparing BRACE Data

You need to create a `.npz` file for BRACE that mimics the NTU structure.

Target Shape: `(N, C, T, V, M)`

- **N:** Number of Clips (Breakdancing sequences)
- **C:** 3 (X, Y, Confidence/Z)
- **T:** 64 (Frames - you must sample/cut your clips to 64 frames)
- **V:** 25 (Joints - mapped from BRACE)
- **M:** 1 (Breakdancing is usually solo, unlike NTU. Set second person to 0).

### Phase 2.4: Modifying the Code for Fine-Tuning

You need to modify `model/HPI_GCN_OP.py` (or create a new wrapper script) to load your pre-trained weights but change the final classification layer.

Create a file `finetune_brace.py`:

```python
import torch
import torch.nn as nn
from model.HPI_GCN_OP import Model as HPI_Model

def get_finetuning_model(pretrained_path, num_brace_classes=10):
    # 1. Initialize the model with NTU settings (to match saved weights)
    # num_class=120 (Original NTU classes)
    model = HPI_Model(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    # 2. Load the weights you just trained
    print(f"Loading weights from {pretrained_path}...")
    state_dict = torch.load(pretrained_path)
    model.load_state_dict(state_dict)

    # 3. Freeze Early Layers (Optional but recommended for small data)
    # This keeps the "Human Motion Physics" knowledge locked
    for param in model.parameters():
        param.requires_grad = False

    # 4. Replace the Classification Head (The "Surgery")
    # We enable gradients ONLY for this new layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_brace_classes)

    return model

# Example Usage
if __name__ == "__main__":
    model = get_finetuning_model("ntu120_pretrained.pt", num_brace_classes=5) # e.g., Toprock, Footwork, Power, Freeze, Transition
    print("Model ready for BRACE fine-tuning.")

```

### 3. How this connects to "Generative Dance" (Your Ultimate Goal)

You might ask: *"I am building a Generator. Why did I just train a Classifier?"*

This HPI-GCN model is the **"Critic"** (or the Perceptual Feature Extractor) for your future Generative Model (like Diffusion model).

1. **Motion Consistency:** When you generate a dance, you run it through this HPI-GCN. If the HPI-GCN gets confused or gives low confidence, your generated dance looks "fake" or "physically impossible."
2. **Style Transfer:** You can extract the "feature vector" (the layer before the classification) to mathematically represent the *style* of the dance (e.g., "Aggressive Power Move" vs "Smooth Footwork").
3. **Beat Alignment:** Later, you will fuse these GCN features with the Audio features (Librosa/Essentia) to ensure the "Go Downs" happen exactly on the beat.

### Immediate Action Items

1. **Stop Training:** You have enough data.
2. **Data Conversion:** Write a script to convert your BRACE JSONs into the `.npy` format `(N, 3, 64, 25, 1)`.
3. **Define Classes:** Decide exactly what your BRACE labels are (Powermove, Toprock and Footwork).
4. **Map Joints:** Create a dictionary mapping BRACE skeleton IDs to NTU skeleton IDs.

You are proceeding exactly as a high-level researcher would. The "Foundation Model" (NTU) is built. Now you specialize.

**Very Important Points/Pitfalls to Review or May Be Missing:**

- **Data Consistency & Cleaning:**
    - Confirm all BRACE data (JSONs) have valid and complete pose sequences with proper joint ordering.
    - Handle sequences shorter than 64 frames (padding/truncation policy).
    - Address missing joints (should be set to zero in padded array, but make sure no corrupt data slips in).
- **Label Definitions & Balance:**
    - Ensure your class definitions in BRACE are finalized and balanced enough for training (avoid high class imbalance).
    - Double-check label mapping—no mislabels or duplicated class indices.
- **Evaluation/Validation Split:**
    - Create (and document) your validation/test split strategy for BRACE (cross-validation/folds or holdout set).
    - Do not fine-tune and test on the same data.
- **Hyperparameter Check:**
    - The learning rates for fine-tuning should be lower than for pre-training.
    - Confirm optimizer and scheduler settings for the new dataset size and convergence.
- **Model Compatibility:**
    - Ensure that loading NTU weights (with changed classifier head) does not skip, misalign, or throw warnings in layer loading.
    - Confirm all necessary parameters in HPI-GCN-OP are compatible with new data shapes.
- **Feature Extraction & Logging:**
    - Set up logging for accuracy, loss, class-wise metrics during fine-tuning.
    - Save checkpoints and ensure model reproducibility.
- **Document & Automate:**
    - Document your data conversion code and joint-mapping logic.
    - Where possible, script or automate data prep/augmentation.
- **Additional Generalization:**
    - Consider lightweight data augmentation if your BRACE set is very small (temporal, spatial, or noise).
    - Plan for error analysis (confusion matrix, misclassified motion types).
- **Long-term Vision (Optional, but significant):**
    - If going toward generative models, carefully store “feature vectors” for style/identity separation.
    - Plan/track future fusion with audio descriptors for beat alignment.

---

**If any of these points remain unchecked, missing, or undocumented, they are very important to address before moving fully into fine-tuning and experimentation.**