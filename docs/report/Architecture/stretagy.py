import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. DEFINE THE HPI-GCN-OP ARCHITECTURE
# ==========================================

class HPIGCN_Backbone(nn.Module):
    """
    The 'Backbone': This part understands physics, joints, and gravity.
    It takes skeletal data and extracts rich features.
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super(HPIGCN_Backbone, self).__init__()
        # Simulating Graph Convolutional Layers (The "Physics" Engine)
        self.graph_conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.graph_conv2 = nn.Conv2d(64, feature_dim, kernel_size=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1) # Pools data into a feature vector

    def forward(self, x):
        # Input x shape: (Batch, Channels, Frames, Joints)
        
        # --- FEATURE EXTRACTION LOGIC ---
        # Internally, this calculates relative positions (like Wrist vs Head)
        x = F.relu(self.graph_conv1(x))
        x = F.relu(self.graph_conv2(x))
        
        # Output a single feature vector representing the motion's "DNA"
        x = self.global_pooling(x)
        return x.view(x.size(0), -1) # Flatten to (Batch, 256)

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.backbone = HPIGCN_Backbone()
        # The "Head": Maps features to specific class labels
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# ==========================================
# 2. PHASE 1: PRE-TRAINED ON NTU RGB+D 120
# ==========================================

print("--- PHASE 1: Loading Pre-trained NTU Model ---")

# Initialize model with 120 classes (NTU Standard)
model = ActionRecognitionModel(num_classes=120)

# Simulate a "Drink Water" input (Batch=1, Channels=3, Frames=64, Joints=25)
dummy_input_ntu = torch.randn(1, 3, 64, 25) 

# Forward pass
output_ntu = model(dummy_input_ntu)
print(f"Model Architecture: Backbone + Head (Output Size: {output_ntu.shape[1]})")

# Visualize Prediction (Simulated)
ntu_classes = {0: "Drink Water", 12: "Writing", 50: "Kicking", 119: "Rock-paper-scissors"}
predicted_index = 0 # Let's pretend the model predicted class 0
print(f"Prediction: Class {predicted_index} -> '{ntu_classes[predicted_index]}'")
print(f"Logic: The backbone saw the wrist joint move towards the head joint.\n")


# ==========================================
# 3. PHASE 2: FREEZE BACKBONE & CUT HEAD
# ==========================================

print("--- PHASE 2: Transfer Learning Setup ---")

# A. FREEZE THE BACKBONE
# We stop calculating gradients for the backbone. 
# We want to KEEP the physics knowledge it learned from NTU.
for param in model.backbone.parameters():
    param.requires_grad = False

print("Step A: HPI-GCN-OP Backbone is now FROZEN. (Physics knowledge preserved)")

# B. REPLACE THE HEAD
# We cut off the 120-class layer and attach a new 3-class layer.
num_brace_classes = 3 # Powermove, Toprock, Footwork
model.head = nn.Linear(256, num_brace_classes)

print(f"Step B: Old 'Head' discarded. New 'Head' attached with output size: {num_brace_classes}.\n")


# ==========================================
# 4. PHASE 3: FINE-TUNE FOR BRACE (Breakdancing)
# ==========================================

print("--- PHASE 3: Fine-Tuning for Breakdancing ---")

# Simulate a "Windmill" input (Same skeleton structure, different movement)
dummy_input_brace = torch.randn(1, 3, 64, 25)

# Forward pass with the NEW Head
output_brace = model(dummy_input_brace)

# Define new classes
brace_classes = ["Powermove (Windmill)", "Toprock", "Footwork"]

# Get prediction
pred_prob = F.softmax(output_brace, dim=1)
predicted_index_brace = 0 # Pretending it predicts Powermove

print(f"New Model Output Vector Size: {output_brace.shape[1]}")
print(f"Prediction: Class {predicted_index_brace} -> '{brace_classes[predicted_index_brace]}'")

# ==========================================
# 5. VISUALIZING THE FEATURE LOGIC (Wrist vs Head)
# ==========================================
print("\n--- INTERNAL LOGIC: Why it works ---")

def calculate_trajectory_logic(skeleton_data):
    # Simulating what the GCN calculates mathematically
    # Index 3 = Head, Index 7 = Wrist (Example indices)
    head_pos = skeleton_data[:, :, 3] 
    wrist_pos = skeleton_data[:, :, 7]
    
    # Calculate distance
    distance = torch.norm(head_pos - wrist_pos)
    
    print(f"Mathematical Feature Extraction:")
    print(f"1. Tracking Wrist Joint (x,y,z)")
    print(f"2. Tracking Head Joint (x,y,z)")
    print(f"3. Calculated Trajectory Delta: {distance:.4f}")
    print("In Phase 1, Low Delta = 'Drink Water'.")
    print("In Phase 3, High Velocity + Low Delta (on ground) = 'Windmill'.")

calculate_trajectory_logic(dummy_input_brace[0])