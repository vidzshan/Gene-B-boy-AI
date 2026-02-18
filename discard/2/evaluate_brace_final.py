import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from model.HPI_GCN_OP import Model

# CONFIG
DATA_PATH = './data/brace/BRACE_fixed_topology_v3.npz'
MODEL_PATH = './pretrained_models/brace_final_model.pt'
CLASSES = ['Toprock', 'Footwork', 'Powermove']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate():
    # 1. Load Data & Re-Split (Must use same random_state=42 to get same test set)
    npz_data = np.load(DATA_PATH)
    x_key = 'x_data' if 'x_data' in npz_data else 'x_train'
    y_key = 'y_data' if 'y_data' in npz_data else 'y_train'

    X = npz_data[x_key]
    y = npz_data[y_key]

    # Get the 20% Test Set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 2. Load Model
    model = Model(num_class=120, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}, Is_joint=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE).eval()
    # do not need to change the model file. You simply ensure that any new code you write (like the Generator or Demo) uses num_person=1 when loading that file.
    
    # 3. Inference
    print(f"Evaluating on {len(X_test)} unseen samples...")
    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    y_tensor = torch.from_numpy(y_test).long().to(DEVICE)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()
    targets = y_tensor.cpu().numpy()

    # 4. Report
    print(classification_report(targets, preds, target_names=CLASSES))

    # 5. Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('HPI-GCN-OP on BRACE (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    evaluate()
