# 测试代码（单独运行）
import torch
from mamba_tiny import Mamba

# 初始化 Mamba 层（参数与 LightM-UNet 中一致）
mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
# 模拟输入：(batch=2, seq_len=100, d_model=64)
dummy_input = torch.randn(2, 100, 64)
# 前向传播
dummy_output = mamba(dummy_input)

# 验证输出维度
print(f"输入维度: {dummy_input.shape}")
print(f"输出维度: {dummy_output.shape}")
assert dummy_output.shape == dummy_input.shape, "Mamba 层输出维度错误"
print("Mamba 层测试通过！")