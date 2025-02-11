import torch
from models.aspp import ASPP

# Example input: batch_size=1, channels=2048, height=32, width=32
input_tensor = torch.randn(1, 2048, 32, 32)

# Initialize ASPP
aspp = ASPP(in_channels=2048, out_channels=256)

# Forward pass
output = aspp(input_tensor)

# Print output shape
print("Output shape:", output.shape)
