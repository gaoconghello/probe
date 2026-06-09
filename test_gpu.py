import torch


import torch
print(torch.version.cuda)


# 首先检查CUDA(GPU)是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU for acceleration")
    # 检查CUDA是否可用
    print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "NA")
    print("GPU设备:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "NA")
# 然后检查MPS是否可用
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for acceleration")
# 如果都不可用,则使用CPU
else:
    device = torch.device("cpu")
    print("Neither GPU nor MPS available. Using CPU")
