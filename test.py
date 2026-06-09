import torch

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
