import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_tiny import Mamba  # 确保已修正返回值为output

class VisionStateSpaceModule(nn.Module):
    def __init__(self, dim, expansion_factor=1, groups=4):  # expand=1（无膨胀），groups=4（分组卷积）
        super().__init__()
        self.dim = dim
        self.expand_dim = dim * expansion_factor  # 隐藏层维度=dim（无膨胀）
        self.groups = groups

        # 核心修改：用1x1分组卷积替代Linear（参数=dim×expand_dim / groups）
        self.linear1 = nn.Conv2d(dim, self.expand_dim, 1, groups=groups, bias=False)
        self.dwconv = nn.Conv2d(self.expand_dim, self.expand_dim, 3, padding=1, groups=self.expand_dim, bias=False)
        self.silu = nn.SiLU()
        # Mamba层：d_model=expand_dim=dim，保持维度一致
        self.mamba = Mamba(d_model=self.expand_dim, d_state=8, d_conv=3, expand=1, bias=False)
        self.norm1 = nn.LayerNorm(self.expand_dim)
        self.linear2 = nn.Conv2d(dim, self.expand_dim, 1, groups=groups, bias=False)
        self.output_proj = nn.Conv2d(self.expand_dim, dim, 1, groups=groups, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        seq_len = H * W
        # 转置为CNN格式：(B, C, H, W)
        x_cnn = x.permute(0, 3, 1, 2)  # (B, dim, H, W)

        # 分支1：1x1 Conv → DWConv → SiLU → Mamba → LayerNorm
        x1 = self.linear1(x_cnn)  # (B, expand_dim, H, W)
        x1_conv = self.dwconv(x1)
        x1_conv = self.silu(x1_conv)
        # 转置为Mamba输入格式：(B, seq_len, expand_dim)
        x1_seq = x1_conv.permute(0, 2, 3, 1).reshape(B, seq_len, self.expand_dim)
        x1_seq = self.mamba(x1_seq)
        x1 = x1_seq.reshape(B, H, W, self.expand_dim).permute(0, 3, 1, 2)  # 转回CNN格式
        x1 = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm适配

        # 分支2：1x1 Conv → SiLU
        x2 = self.linear2(x_cnn)
        x2 = self.silu(x2)

        # Hadamard乘积融合+投影
        x_fuse = x1 * x2
        x_out = self.output_proj(x_fuse)  # (B, dim, H, W)
        # 转回(B, H, W, C)格式
        return x_out.permute(0, 2, 3, 1)

class ResidualVisionMambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vss_module = VisionStateSpaceModule(dim)
        self.adjust_factor = nn.Parameter(torch.ones(1, 1, 1, dim))  # 无参数增量

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        x_vss = self.vss_module(x_norm)
        x = x_vss + self.adjust_factor * residual
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_rvm_layers, groups=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rvm_layers = nn.Sequential(*[ResidualVisionMambaLayer(in_dim) for _ in range(num_rvm_layers)])
        # 分组卷积：参数=in_dim×out_dim / groups
        self.channel_proj = nn.Conv2d(in_dim, out_dim, 1, groups=groups, bias=False)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x, name=""):
        x = x.permute(0, 2, 3, 1)  # (B, H, W, in_dim)
        x = self.rvm_layers(x)  # (B, H, W, in_dim)
        x = x.permute(0, 3, 1, 2)  # (B, in_dim, H, W)
        x = self.channel_proj(x)  # (B, out_dim, H, W)
        x_skip = x
        x_down = self.max_pool(x)
        return x_down, x_skip

class BottleneckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rvm_layers = nn.Sequential(*[ResidualVisionMambaLayer(dim) for _ in range(4)])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.rvm_layers(x)
        x = x.permute(0, 3, 1, 2)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, name):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name = name
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dwconv = nn.Conv2d(in_dim, out_dim, 3, padding=1, groups=out_dim)  # 深度可分离卷积，参数极少
        self.relu = nn.ReLU(inplace=True)
        # 核心优化：adjust_factor通道数=out_dim，直接与x_conv匹配（无需额外投影层）
        self.adjust_factor = nn.Parameter(torch.ones(1, out_dim, 1, 1))
        # 新增：1x1卷积压缩x_fuse通道数（参数极少，比动态层更高效）
        self.fuse_proj = nn.Conv2d(in_dim, out_dim, 1)  # 仅1个卷积核，参数=in_dim×out_dim + out_dim

    def forward(self, x, skip_x):
        x_up = self.upsample(x)
        x_fuse = x_up + skip_x
        # 用1x1卷积将x_fuse（in_dim通道）压缩到out_dim通道，与x_conv匹配
        x_fuse_proj = self.fuse_proj(x_fuse)  # (B, out_dim, H, W)
        x_conv = self.dwconv(x_fuse)  # (B, out_dim, H, W)
        # 现在通道数一致：x_conv（out_dim） + adjust_factor×x_fuse_proj（out_dim）
        x_out = self.relu(x_conv + self.adjust_factor * x_fuse_proj)
        return x_out

class LightMUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # 浅层卷积：3→32（参数=3×32×3×3 + 32=896）
        self.shallow_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.shallow_norm = nn.BatchNorm2d(32)
        self.shallow_relu = nn.ReLU(inplace=True)
        
        # Encoder（参数主要来自RVM层和channel_proj）
        self.encoder1 = EncoderBlock(32, 64, 1)  # RVM×1 + 32→64线性层
        self.encoder2 = EncoderBlock(64, 128, 2)  # RVM×2 + 64→128线性层
        self.encoder3 = EncoderBlock(128, 256, 2) # RVM×2 + 128→256线性层
        
        self.bottleneck = BottleneckBlock(256)  # RVM×4（参数占比最高，但Mamba是线性复杂度）
        
        # Decoder（参数主要来自fuse_proj和dwconv）
        self.decoder1 = DecoderBlock(256, 128, name="1")  # 256→128投影 + 深度卷积
        self.decoder2 = DecoderBlock(128, 64, name="2")   # 128→64投影 + 深度卷积
        self.decoder3 = DecoderBlock(64, 32, name="3")    # 64→32投影 + 深度卷积
        
        # 输出层：32→1（参数=32×1×1 + 1=33）
        self.output_conv = nn.Conv2d(32, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 浅层特征
        x_shallow = self.shallow_conv(x)
        x_shallow = self.shallow_norm(x_shallow)
        x_shallow = self.shallow_relu(x_shallow)
        
        # Encoder
        x1_down, x1_skip = self.encoder1(x_shallow, name="1")
        x2_down, x2_skip = self.encoder2(x1_down, name="2")
        x3_down, x3_skip = self.encoder3(x2_down, name="3")
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x3_down)
        
        # Decoder
        x_dec1 = self.decoder1(x_bottleneck, x3_skip)
        x_dec2 = self.decoder2(x_dec1, x2_skip)
        x_dec3 = self.decoder3(x_dec2, x1_skip)
        
        # 输出
        x_logits = self.output_conv(x_dec3)
        x_pred = self.sigmoid(x_logits)
        return x_pred

if __name__ == "__main__":
    model = LightMUNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")  # 输出：~1.3M
    dummy_input = torch.randn(1, 3, 576, 576)
    dummy_output = model(dummy_input)
    print(f"输入维度: {dummy_input.shape}, 输出维度: {dummy_output.shape}")
    print("✅ 模型轻量化且前向成功！")