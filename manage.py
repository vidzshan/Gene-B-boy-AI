import os
import sys
import shutil
import subprocess
from pathlib import Path

# ==========================================
# SASAKI-GAN MISSION CONTROL
# ==========================================
# A unified CLI for managing the Neuro-Symbolic Repository.
# Usage: python manage.py [command]

MODES = {
    "clean": "Remove pycache and temp artifacts",
    "tree": "Visualize Neuromorphic Structure",
    "audit": "Run Forensic Data Analysis",
    "live": "Launch the SASAKI Live Core",
    "lab": "Launch the Gradio Research Lab"
}

def clean():
    print("🧹 Purging Synaptic Debris (__pycache__)...")
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                print(f"   - Cleared {os.path.join(root, d)}")
    print("✨ Cortex Clean.")

def tree():
    print("\n🧠 SASAKI-GAN NEUROMORPHIC ANATOMY")
    print("====================================")
    print("📂 CORTEX (Decision Logic)")
    print(" ├── 🧠 SASAKI/                 [Stochastic Logic Modules]")
    print(" ├── 📜 slerp_utils.py          [Latent Interpolation]")
    print(" └── ⚙️ config/                 [Hyperparameter Synapses]")
    print("\n💪 MOTOR CORTEX (Generative Core)")
    print(" ├── 🕸️ net/                    [HPI-GCN Architecture]")
    print(" ├── 🏋️ main.py                 [Training Harness]")
    print(" └── 📦 pretrained_models/      [Muscle Memory (Checkpoints)]")
    print("\n👁️ PERCEPTION (Sensory Input)")
    print(" ├── 🎤 realtime_audio_engine.py [Auditory Cortex]")
    print(" └── 📂 audio_brace_loader.py   [Dataloader]")
    print("\n🦴 SPINE (Infrastructure)")
    print(" ├── 🛠️ scripts/                [Reflex Utilities]")
    print(" ├── 📚 docs/                   [Long-term Memory]")
    print(" └── 🚀 manage.py               [Central Nervous System]")
    print("====================================\n")

def audit():
    print("🔍 Running Forensic Audit...")
    try:
        subprocess.run(["python", "scripts/analyze_correlation.py"], check=True)
    except Exception as e:
        print(f"⚠️ Audit failed: {e}")

def live():
    print("⚡ Igniting SASAKI Live Core...")
    os.system("python RUN_SASAKI_LIVE.py")

def lab():
    print("🧪 Opening Research Lab...")
    os.system("python app.py")

def help():
    print("🎮 MISSION CONTROL COMMANDS:")
    for key, val in MODES.items():
        print(f"   python manage.py {key:<10} : {val}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help()
        sys.exit()
    
    cmd = sys.argv[1]
    if cmd == "clean": clean()
    elif cmd == "tree": tree()
    elif cmd == "audit": audit()
    elif cmd == "live": live()
    elif cmd == "lab": lab()
    else: help()
