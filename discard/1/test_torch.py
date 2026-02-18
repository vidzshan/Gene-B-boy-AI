import time
print("Start import...")
start = time.time()
import torch
print(f"Imported torch in {time.time() - start:.2f}s")
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")
