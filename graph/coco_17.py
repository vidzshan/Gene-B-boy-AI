import sys
import numpy as np
sys.path.extend(['../'])
from graph import tools

# COCO has 17 joints
num_node = 17

# 1. Self Loops (Every joint connects to itself)
self_link = [(i, i) for i in range(num_node)]

# 2. Anatomical Connections (The "Bones")
# Format: (Child, Parent) -> Direction of information flow
neighbor_base = [
    (0, 1), (0, 2), # Nose -> Eyes
    (1, 3), (2, 4), # Eyes -> Ears
    (0, 5), (0, 6), # Nose -> Shoulders (Connecting head to body)
    (5, 7), (7, 9), # L Shoulder -> L Elbow -> L Wrist
    (6, 8), (8, 10),# R Shoulder -> R Elbow -> R Wrist
    (5, 11), (6, 12), # Shoulders -> Hips (Torso)
    (11, 12),       # Hip -> Hip (Pelvis connection)
    (11, 13), (13, 15), # L Hip -> L Knee -> L Ankle
    (12, 14), (14, 16)  # R Hip -> R Knee -> R Ankle
]

# 3. Create Inward (Centripetal) and Outward (Centrifugal) edges
inward = [(i, j) for (i, j) in neighbor_base]
outward = [(j, i) for (i, j) in neighbor_base]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_dim1':
            A = tools.get_graph(num_node, self_link, neighbor)
        else:
            raise ValueError()
        return A
