import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """原始U-Net双层卷积模块：无BatchNorm，仅Conv+ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 原始U-Net：无BatchNorm，卷积偏置默认开启
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """原始U-Net下采样模块：最大池化（步长2）+ 双层卷积"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 原始U-Net下采样方式：最大池化，步长2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """原始U-Net上采样模块：转置卷积（非bilinear上采样）+ 拼接 + 双层卷积"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # 原始U-Net：仅使用转置卷积进行上采样，不使用bilinear上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 原始U-Net：拼接后的双层卷积，输入通道数=上采样通道数 + 编码器跳跃通道数
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 1. 转置卷积上采样
        x1 = self.up(x1)
        # 2. 尺寸对齐（与你的代码一致，解决上下采样尺寸偏差问题）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # 3. 原始U-Net：跳跃连接（编码器特征与解码器特征拼接，dim=1：通道维度）
        x = torch.cat([x2, x1], dim=1)
        # 4. 双层卷积融合特征
        return self.conv(x)

# 保留你的OutConv：强制输出1通道，无out_channels参数
class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 固定输出1通道，原始U-Net最后一层1×1卷积

    def forward(self, x):
        return self.conv(x)

# 保留你的输入输出格式：n_channels入参，无n_classes，输出1通道
class Unet(nn.Module):
    """严格对齐原始U-Net论文结构的模型"""
    def __init__(self, n_channels, bilinear=None):  # bilinear参数保留（兼容你的调用），实际未使用
        super(Unet, self).__init__()
        self.n_channels = n_channels

        # 原始U-Net编码器结构（通道数与论文完全一致）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # 原始U-Net最后一个下采样输出通道数1024

        # 原始U-Net解码器结构（通道数与论文完全一致）
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64)  # 输出1通道，与你的需求一致

    def forward(self, x):
        # 编码器前向（原始U-Net跳跃连接，保存每一层输出）
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器前向（原始U-Net：上采样+拼接编码器特征）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出1通道logits，与你的代码一致
        logits = self.outc(x)
        return logits