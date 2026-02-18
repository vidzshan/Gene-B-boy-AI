print("Start test_imports.py", flush=True)
import torch
print("Imported torch", flush=True)
import torch.nn as nn
print("Imported nn", flush=True)
import numpy as np
print("Imported numpy", flush=True)
import torch.optim as optim
print("Imported optim", flush=True)
from torch.utils.data import DataLoader, Dataset
print("Imported DataLoader, Dataset", flush=True)
import os
print("Imported os", flush=True)

print("Importing model.generator...", flush=True)
from model.generator import MotionGenerator
print("Imported model.generator", flush=True)

print("Importing model.HPI_GCN_OP...", flush=True)
from model.HPI_GCN_OP import Model as Discriminator
print("Imported model.HPI_GCN_OP", flush=True)

print("All imports successful", flush=True)
