import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("GPU 架构（Compute Capability）:", torch.cuda.get_device_capability(0))


x = torch.rand(3, 3).cuda()
print("成功使用 GPU:", x)