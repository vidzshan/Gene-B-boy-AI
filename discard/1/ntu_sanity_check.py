import torch
import numpy as np
from model.HPI_GCN_OP import Model

# 1. Configuration (Must match your Training Config)
config = {
    'num_class': 120,
    'num_point': 25,
    'num_person': 2,
    'graph': 'graph.ntu_rgb_d.Graph',
    'graph_args': {'labeling_mode': 'spatial'}
}

def clean_state_dict(state_dict):
    """
    Removes 'module.' prefix if the model was trained with DataParallel
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    return new_state_dict

print("--- STARTING SANITY CHECK ---")

# 2. Initialize the Model Architecture
print("[1] Building Model Architecture...")
model = Model(
    num_class=config['num_class'],
    num_point=config['num_point'],
    num_person=config['num_person'],
    graph=config['graph'],
    graph_args=config['graph_args']
)
model.cuda() # Move to RTX 4070
model.eval() # Set to Inference Mode (turns off Dropout)

# 3. Load Your Pretrained Weights
weights_path = './pretrained_models/ntu120_pretrained.pt'
print(f"[2] Loading weights from {weights_path}...")

try:
    # Load the file
    checkpoint = torch.load(weights_path)
    
    # Sometimes checkpoints save extra info (epoch, optimizer). We just want weights.
    # If you saved the whole dictionary, the weights might be under 'model' or 'state_dict'
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # It might be just the weights directly

    # Clean the keys (remove 'module.')
    state_dict = clean_state_dict(state_dict)
    
    # Apply weights
    model.load_state_dict(state_dict)
    print("✔ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

# 4. Generate Dummy Input (Batch=1, Channels=3, Frames=64, Joints=25, Person=2)
# This simulates one 2-second video clip.
dummy_input = torch.randn(1, 3, 64, 25, 2).cuda()
print(f"[3] Generated dummy input shape: {dummy_input.shape}")

# 5. Run Inference
print("[4] Running Forward Pass...")
with torch.no_grad():
    output = model(dummy_input)

# 6. Analyze Output
# Output shape should be (1, 120) -> Probability for each class
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

print("✔ Inference Successful!")
print(f"   Output Shape: {output.shape} (Should be [1, 120])")
print(f"   Predicted Class (Random Noise): {predicted_class}")
print("--- SANITY CHECK PASSED ---")
