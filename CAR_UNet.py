import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------- 1. 修复MECA（对齐论文，仅全局池化） --------------------------
class MECA(nn.Module):
    def __init__(self, channel):
        super(MECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 论文k=3，仅3个参数
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.shape
        # 双池化+1D卷积
        avg = self.avg_pool(x).squeeze(-1).permute(0, 2, 1)  # (B,1,C)
        max = self.max_pool(x).squeeze(-1).permute(0, 2, 1)
        avg_att = self.conv(avg).permute(0, 2, 1).unsqueeze(-1)  # (B,C,1,1)
        max_att = self.conv(max).permute(0, 2, 1).unsqueeze(-1)
        return x * self.sigmoid(avg_att + max_att)

# -------------------------- 2. 修复CADRB（输入通道=拼接后通道数） --------------------------
class CADRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CADRB, self).__init__()
        # 主路径：2个3×3卷积（输入通道=拼接后通道数，输出=目标通道数）
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # Shortcut路径：1×1卷积（匹配输入输出通道）
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # MECA注意力（作用于输出通道）
        self.meca = MECA(channel=out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 双残差融合+MECA
        residual = self.path1(x) + self.path2(x)
        return self.relu(self.meca(residual))

# -------------------------- 3. 修复Down模块（通道数逐步翻倍） --------------------------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样：尺寸减半
            CADRB(in_channels, out_channels)  # 输入=上一层输出，输出=翻倍通道
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# -------------------------- 4. 修复Up模块（关键：拼接后通道数=CADRB输入通道） --------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            # 双线性上采样：尺寸翻倍，通道数不变
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # CADRB输入通道=上采样通道 + 跳跃连接通道（in_channels=上采样通道，需手动计算）
            self.conv = CADRB(in_channels + in_channels//2, out_channels)
        else:
            # 转置卷积上采样：尺寸翻倍，通道数减半
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = CADRB(in_channels, out_channels)  # 输入=上采样通道（in_channels//2）+ 跳跃连接通道（in_channels//2）= in_channels

    def forward(self, x1, x2):
        # x1：上采样前的特征（深层），x2：跳跃连接特征（浅层）
        x1 = self.up(x1)
        # 对齐尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        # 拼接：深层上采样特征 + 浅层跳跃连接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -------------------------- 5. 修复OutConv（输出1通道） --------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------- 6. 修复CAR_UNet（通道数全链路匹配） --------------------------
class CAR_UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(CAR_UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # 论文第一卷积层：3→16通道（轻量化）
        self.inc = CADRB(n_channels, 16)
        
        # 收缩路径：通道数逐步翻倍（16→32→64→128→256）
        self.down1 = Down(16, 32)    # 16→32（下采样后）
        self.down2 = Down(32, 64)    # 32→64
        self.down3 = Down(64, 128)   # 64→128
        self.down4 = Down(128, 256)  # 128→256（最深层）
        
        # 扩展路径：通道数逐步减半（256→128→64→32→16）
        # Up参数说明：in_channels=深层通道数，out_channels=目标通道数
        self.up1 = Up(256, 128, bilinear)  # 深层256→上采样后256（双线性）+ 浅层128 → 拼接384 → 输出128
        self.up2 = Up(128, 64, bilinear)   # 深层128→上采样后128 + 浅层64 → 拼接192 → 输出64
        self.up3 = Up(64, 32, bilinear)    # 深层64→上采样后64 + 浅层32 → 拼接96 → 输出32
        self.up4 = Up(32, 16, bilinear)    # 深层32→上采样后32 + 浅层16 → 拼接48 → 输出16
        
        # 输出层：16→1通道
        self.outc = OutConv(16)
        
        # 跳跃连接MECA（作用于浅层特征）
        self.meca_32 = MECA(32)
        self.meca_64 = MECA(64)
        self.meca_128 = MECA(128)
        self.meca_256 = MECA(256)

    def forward(self, x):
        # 收缩路径：CADRB + 跳跃连接MECA
        x1 = self.inc(x)               # 3→16
        x2 = self.down1(x1)            # 16→32
        x2_att = self.meca_32(x2)      # 跳跃连接MECA
        x3 = self.down2(x2)            # 32→64
        x3_att = self.meca_64(x3)      # 跳跃连接MECA
        x4 = self.down3(x3)            # 64→128
        x4_att = self.meca_128(x4)     # 跳跃连接MECA
        x5 = self.down4(x4)            # 128→256
        x5_att = self.meca_256(x5)     # 跳跃连接MECA
        
        # 扩展路径：上采样 + 拼接 + CADRB（通道数匹配）
        x = self.up1(x5_att, x4_att)   # 256（深层）+ 128（浅层）→ 拼接384 → 输出128
        x = self.up2(x, x3_att)        # 128 + 64 → 拼接192 → 输出64
        x = self.up3(x, x2_att)        # 64 + 32 → 拼接96 → 输出32
        x = self.up4(x, x1)            # 32 + 16 → 拼接48 → 输出16（x1无MECA）
        
        logits = self.outc(x)          # 16→1
        return logits
