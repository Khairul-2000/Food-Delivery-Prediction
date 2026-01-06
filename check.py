import torch

# ------------------------------------------------------------------------------------------------------------------------

# Combined dataset: bikes for short distances, cars for longer ones
distances = torch.tensor([
    [1.0, 1.2,12], [1.5, 1.7,12] ])



print(distances.shape)