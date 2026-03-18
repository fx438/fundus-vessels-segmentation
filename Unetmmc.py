import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedMMC(nn.Module):
    def __init__(self, channel=256, d_state=64):
        super().__init__()
        self.channel = channel
        self.d_state = d_state
        self.noise_gate = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        # 1. 动态形态卷积（修复：深度卷积权重维度，groups=C时权重应为(C,1,3,3)）
        # 深度卷积：每个输入通道对应1个独立3×3卷积核，输出通道数=输入通道数=channel
        self.morph_conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,  # 输出通道数=输入通道数（深度卷积要求）
            kernel_size=3,
            padding=1,
            groups=channel,  # 分组数=通道数（深度卷积）
            bias=False
        )
        # 修复：权重初始化（确保权重形状为(channel, 1, 3, 3)，而非(channel, channel, 3, 3)）
        nn.init.kaiming_normal_(self.morph_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.x_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)  # 替代水平池化
        self.y_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)  # 替代垂直池化
        # 2. 增大动态偏移幅度，适配亚像素级血管（关键修改）
        self.offset = nn.Parameter(torch.randn(256, 1, 3, 3) * 0.1)  # 初始偏移从0.05→0.1
        self.offset.data.clamp_(-0.3, 0.3)  # 偏移范围从±0.2→±0.3
        
        # 2. SSM模块（2D操作，无修改）
        self.ssm = nn.Sequential(
            nn.Conv2d(channel, channel + d_state, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel + d_state),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel + d_state, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)
        )
        
        # 3. 多视角融合（无修改）
        self.fusion_conv = nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(channel)

    def forward(self, x):
       
        B, C, H, W = x.shape
        
        # 1. 动态形态卷积（修复：偏移与权重形状匹配，groups=C正确生效）
        # 权重形状：(C, 1, 3, 3)，偏移形状：(1, C, 3, 3)→广播后匹配
        morph_weight = self.morph_conv.weight + self.offset  # 形状：(C, 1, 3, 3)
        # 深度卷积：groups=C，每个通道用自己的1×3×3卷积核，输入输出通道数均为C
        x_morph = F.conv2d(x, morph_weight, padding=1, groups=C)  # 输出：(B, C, H, W)
        
        # 2. 多视角特征提取（无修改）
        x_x = self.x_conv(x_morph)  # 水平方向特征（无池化，不丢失1-2像素血管）
        x_y = self.y_conv(x_morph)  # 垂直方向特征（无池化）
        
        # 3. SSM模块与融合（无修改）
        x_x_ssm = self.ssm(x_x)
        x_y_ssm = self.ssm(x_y)
        x_fused = torch.cat([x_x_ssm, x_y_ssm], dim=1)
        x_out = self.fusion_conv(x_fused)
        x_out = self.fusion_bn(x_out)
        channel_attn = torch.sigmoid(x_out.mean(dim=(2,3), keepdim=True))  # 对每个通道求均值→sigmoid得权重
        x_out = x_out * channel_attn  # 用权重突出血管特征通道

        gate = self.noise_gate(x_out)
        x_out = x_out * gate
        return x_out

# -------------------------- MECA模块（无修改，确保通道匹配） --------------------------
class MECA(nn.Module):
    def __init__(self, channel):
        super(MECA, self).__init__()
        self.channel = channel
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, bias=True),
            nn.Sigmoid()  
        )
        self.res_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.res_bn = nn.BatchNorm2d(channel)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_pool(x)  
        global_feat = self.global_pool(x)  
        global_feat = global_feat.expand_as(local_feat)  
        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        channel_weight = self.gate(concat_feat)
        enhanced_feat = x * channel_weight  

        residual = self.res_conv(x)
        residual = self.res_bn(residual)
        enhanced_feat = enhanced_feat + residual  
        enhanced_feat = self.dropout(enhanced_feat)
        return enhanced_feat

# -------------------------- U-Net基础模块（无修改，确保inplace禁用） --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k_size = int(abs((np.log2(channel) + b) / gamma))
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

# -------------------------- 最终Unet模型（无修改，确保模块初始化正确） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64)

        self.eca_256 = MECA(channel=256)
        self.simplified_mmc = SimplifiedMMC(channel=256, d_state=64)  # 显式传参，通道匹配
        self.mmc_meca_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # 输入MMC的特征：(B, 256, H, W)

        # MECA与MMC融合（通道均为256，无冲突）
        x3_channel = self.eca_256(x3)
        x3_mmc = self.simplified_mmc(x3)
        weight = torch.softmax(self.mmc_meca_weight, dim=0)
        x3_enhanced = x3_channel * weight[0] + x3_mmc * weight[1]

        x4 = self.down3(x3_enhanced)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3_enhanced)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits