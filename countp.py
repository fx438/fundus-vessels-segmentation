import torch
from LightMUnet import LightMUNet

# 加载模型
model = LightMUNet()

# 逐模块统计参数量
total_params = 0
print("="*60)
print("模块名                          参数量（M）  占比")
print("="*60)

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        ratio = (params / 1e6) / (total_params / 1e6) * 100 if total_params !=0 else 0
        print(f"{name:<30} {params/1e6:.4f} M    {ratio:.1f}%")

print("="*60)
print(f"总参数量: {total_params/1e6:.4f} M")
print("="*60)