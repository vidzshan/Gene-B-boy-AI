import numpy as np
import torch

class NeuralRetrievalDecoder:
    def __init__(self, database_path):
        print("Loading Real Motion Database for RAG...")
        db = np.load(database_path)
        # Flatten database to (Total_Frames, 75)
        self.real_poses = db['x_data'].transpose(0, 2, 3, 1, 4).reshape(-1, 75)
        self.real_poses_tensor = torch.from_numpy(self.real_poses).float().cuda()

    def decode(self, ai_output):
        """
        Projects messy AI output onto the real human manifold.
        ai_output: (64, 25, 3)
        """
        query = torch.from_numpy(ai_output).float().cuda().reshape(64, 75)
        
        # Calculate distance between AI query and all Real Poses
        # This is the 'Neural Retrieval' step
        dists = torch.cdist(query, self.real_poses_tensor)
        
        # Find index of the closest real pose for every frame
        closest_indices = torch.argmin(dists, dim=1)
        
        # Retrieve anatomically perfect poses
        perfect_skeleton = self.real_poses[closest_indices.cpu().numpy()]
        return perfect_skeleton.reshape(64, 25, 3)

# Usage in your visualization script:
# messy_ai_skel = gen(z, a, l)
# rag_decoder = NeuralRetrievalDecoder('./data/brace/BRACE_synced_v2.npz')
# clean_skel = rag_decoder.decode(messy_ai_skel)