import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from model.HPI_GCN_OP import Model

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
PRETRAINED_WEIGHTS = './pretrained_models/ntu120_pretrained.pt'
SAVE_MODEL_PATH = './pretrained_models/brace_final_model.pt'

NUM_CLASSES_BRACE = 3
BATCH_SIZE = 16 
LEARNING_RATE = 0.01 
EPOCHS = 50 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. DATASET
# ==========================================
class BraceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.labels[index]
        
        data_tensor = torch.from_numpy(data_numpy).float()
        
        # Ensure label is a simple integer
        if isinstance(label, np.ndarray):
            label = int(np.argmax(label)) if label.size > 1 else int(label.item())
        else:
            label = int(label)

        return data_tensor, label

# ==========================================
# 3. HELPERS
# ==========================================
def load_and_split_data():
    print(f"Loading {DATA_PATH}...")
    npz_data = np.load(DATA_PATH)
    
    x_key = 'x_data' if 'x_data' in npz_data else 'x_train'
    y_key = 'y_data' if 'y_data' in npz_data else 'y_train'
    
    X = npz_data[x_key]
    y = npz_data[y_key]
    
    if y.ndim > 1: y = np.argmax(y, axis=1)

    print(f"Total Samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def get_class_weights(y_train):
    print("Calculating Class Weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"⚖️ Class Weights: {weights}")
    return weights_tensor

# ==========================================
# 4. TRAINING LOOP (The Fixed Version)
# ==========================================
def train():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    train_dataset = BraceDataset(X_train, y_train)
    test_dataset = BraceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Compute Imbalance Fix
    class_weights = get_class_weights(y_train)

    # 3. Initialize Model (WITH ROBUST WEIGHT SURGERY)
    print("Initializing HPI-GCN-OP for Single Person...")
    
    # Set num_person=1 to match BRACE
    model = Model(num_class=120, num_point=25, num_person=1, 
                  graph='graph.ntu_rgb_d.Graph', 
                  graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    
    # Load Weights
    print(f"Loading weights from {PRETRAINED_WEIGHTS}...")
    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove 'module.' prefix
    new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # --- WEIGHT SURGERY: 2 Person -> 1 Person ---
    print("Performing Weight Surgery (Slicing input channels 150 -> 75)...")
    vars_sliced = 0
    for key in list(new_dict.keys()):
        # FIX: We now check "len(new_dict[key].shape) > 0" to ensure it's not a scalar
        # This prevents the IndexError you saw.
        if 'data_bn' in key and len(new_dict[key].shape) > 0 and new_dict[key].shape[0] == 150:
            new_dict[key] = new_dict[key][:75] 
            vars_sliced += 1
            
    print(f"✂️ Surgery Complete: Sliced {vars_sliced} layers.")
    
    # Load state_dict
    keys = model.load_state_dict(new_dict, strict=False)
    print(f"Weights loaded. (Expected mismatches in Head/BN ignored)")
    
    # Replace Head for BRACE classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES_BRACE)
    model.to(DEVICE)

    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0004)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 5. Lists for Dashboard
    loss_history = []
    acc_history = []

    best_acc = 0.0
    print("--- START TRAINING ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # SAFETY VALVE: GRADIENT CLIPPING
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Store history
        loss_history.append(avg_loss)
        acc_history.append(acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f" >>> Saved Best Model: {acc:.2f}%")

    print(f"--- DONE. Final Best Accuracy: {best_acc:.2f}% ---")
    
    print("\n=== COPY THESE LISTS FOR PLOTTING SCRIPT ===")
    print(f"losses = {loss_history}")
    print(f"accuracies = {acc_history}")

if __name__ == "__main__":
    train()
