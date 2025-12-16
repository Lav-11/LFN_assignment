import torch
import torch_geometric

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")

try:
    from torch_sparse import SparseTensor
    print("SUCCESS: torch-sparse is installed and working!")
except ImportError as e:
    print(f"ERROR: torch-sparse failed to import. Reason: {e}")