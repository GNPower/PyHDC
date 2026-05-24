import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def CosineSimilarity(a, b):
    """Cosine similarity: (A · B) / (‖A‖ · ‖B‖), in [-1, 1]."""
    if TORCH_AVAILABLE and torch.is_tensor(a):
        af = a.float()
        bf = b.float()
        return (torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))).item()
    else:
        af = a.astype(np.float32)
        bf = b.astype(np.float32)
        return np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf))
