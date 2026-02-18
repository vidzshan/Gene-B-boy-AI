import matplotlib.pyplot as plt

def plot_results(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Training Loss (Red)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Create a second y-axis for Accuracy (Blue)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color)
    ax2.plot(epochs, test_accuracies, color=color, linewidth=2, linestyle='dashed', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    # The "Research Grade" Target Line
    ax2.axhline(y=85, color='green', linestyle='-', linewidth=2, label='Target (85%)')
    # Place text slightly above line
    ax2.text(1, 86, 'Research Standard (85%)', color='green', fontweight='bold')

    plt.title('HPI-GCN-OP on BRACE: Training Dynamics')
    fig.tight_layout()
    
    save_path = 'training_health_chart.png'
    plt.savefig(save_path)
    print(f"✅ Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- PASTE YOUR RESULTS HERE AFTER TRAINING ---
    # Example data:
    losses = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3]
    accuracies = [35, 45, 60, 70, 75, 78]
    
    plot_results(losses, accuracies)
