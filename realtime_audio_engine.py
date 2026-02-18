import numpy as np
import librosa

class RealTimeAudioEngine:
    def __init__(self, sr=15360):
        self.sr = sr

    def get_precise_features(self, audio_chunk):
        """
        Extracts High-Fidelity Audio DNA for Logic (Vector) and UI (Matrix).
        """
        # 0. Safety Checks
        if len(audio_chunk) < 100: return np.zeros(33), np.zeros((20, 1))

        # 1. HANDLE STEREO & GAIN (15x Boost for Intel Arrays)
        if audio_chunk.ndim > 1:
            y = np.mean(audio_chunk, axis=1).flatten().astype(np.float32)
        else:
            y = audio_chunk.flatten().astype(np.float32)
        y = y * 15.0
        
        # 2. Check for signal presence
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.00005: 
             return np.zeros(33), np.zeros((20, 1))

        try:
            # 3. Spectral Analysis (20 MFCCs)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20)
            
            # Precise Normalization: Z-score scaling
            # This makes the texture 'dance' in the UI
            mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            
            # 4. Temporal Compression for Logic
            mfcc_vector = np.mean(mfcc_norm, axis=1)
            
            # 5. Multi-modal stack (33-dim)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=self.sr), axis=1)
            onset = np.mean(librosa.onset.onset_strength(y=y, sr=self.sr))
            
            # Return both the Vector (for Brain) and Matrix (for UI)
            return np.hstack([mfcc_vector, chroma, onset]), mfcc_norm
            
        except Exception as e:
            return np.zeros(33), np.zeros((20, 1))

    def get_features(self, audio_chunk):
        # Wrapper for backward compatibility with Brain Thread
        vec, _ = self.get_precise_features(audio_chunk)
        return vec