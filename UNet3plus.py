import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 保留你原有的所有核心模块 --------------------------
# MECA模块（无修改）
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
        self.dropout = nn.Dropout(p=0.05)

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
        enhanced_feat = x * (0.5 + 0.5 * channel_weight)  # 弱化版
        return enhanced_feat

# DoubleConv模块（无修改，保留Dropout）
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

# Down模块（无修改，保留你的卷积下采样）
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_down(x)

# -------------------------- 恢复原始Up模块（删除LGAG，保留双线性/转置卷积分支） --------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 保留你原有的if-else上采样分支
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        # 保留原始DoubleConv（无LGAG）
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样 + 尺寸对齐（完全保留你的原始逻辑）
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 恢复原始直接拼接（删除LGAG相关逻辑）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# OutConv模块（无修改）
class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ECA模块（无修改，保留备用）
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

# -------------------------- UNet3+核心：多尺度特征融合模块（新增） --------------------------
class UNet3Plus_Fusion(nn.Module):
    """UNet3+的多尺度特征融合模块：融合编码器所有层级的特征，添加残差连接"""
    def __init__(self, channels_list=[64, 128, 256, 512, 512], out_channel=64):
        super(UNet3Plus_Fusion, self).__init__()
        self.channels_list = channels_list  # 编码器各层输出通道：[x1, x2, x3, x4, x5]
        self.out_channel = out_channel

        # 1. 各尺度特征降维（确保所有特征通道一致，便于融合）
        self.reduce_dims = nn.ModuleList([
            nn.Conv2d(c, out_channel, kernel_size=1, bias=False)
            for c in channels_list
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(out_channel) for _ in channels_list
        ])

        # 2. 多尺度融合卷积（整合所有降维后的特征）
        self.fusion_conv = DoubleConv(
            in_channels=out_channel * len(channels_list),  # 64*5=320
            out_channels=out_channel
        )

        # 3. 残差连接（连接融合后的特征与最浅层特征x1，保留细节）
        self.residual_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channel)

    def forward(self, features):
        """
        Args:
            features: 编码器各层输出的特征列表，顺序为 [x1, x2, x3, x4, x5]
                      x1: (B,64,H,W), x2: (B,128,H/2,W/2), x3: (B,256,H/4,W/4)
                      x4: (B,512,H/8,W/8), x5: (B,512,H/16,W/16)
        Return:
            fused_feat: 多尺度融合后的特征（B,64,H,W）
        """
        B, _, H, W = features[0].shape  # 以最浅层x1的尺寸为基准
        reduced_feats = []

        # 对每个尺度的特征进行“降维+上采样到x1尺寸”
        for i, (feat, reduce_conv, bn) in enumerate(zip(features, self.reduce_dims, self.bns)):
            # 降维到out_channel（64）
            reduced = reduce_conv(feat)
            reduced = bn(reduced)
            reduced = F.relu(reduced, inplace=False)
            # 上采样到x1的尺寸（H,W）
            if i > 0:  # x2-x5需要上采样，x1无需
                scale = 2 ** i  # x2上采样2倍，x3上采样4倍，以此类推
                reduced = F.interpolate(
                    reduced, size=(H, W), mode='bilinear', align_corners=True
                )
            reduced_feats.append(reduced)

        # 拼接所有降维+上采样后的特征（64*5=320通道）
        concat_feat = torch.cat(reduced_feats, dim=1)
        # 融合卷积（320→64通道）
        fused = self.fusion_conv(concat_feat)
        # 残差连接（融合特征 + 降维后的x1）
        residual = self.residual_conv(reduced_feats[0])  # x1降维后的残差
        residual = self.residual_bn(residual)
        fused_with_res = fused + residual  # 残差连接，缓解梯度消失
        fused_with_res = F.relu(fused_with_res, inplace=False)

        return fused_with_res

# -------------------------- 最终UNet3+模型（保留现有模块 + 多尺度融合 + 残差连接） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # 1. 保留你原有的编码器结构（含MECA）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.eca_256 = MECA(channel=256)  # 保留你的MECA模块

        # 2. UNet3+核心：多尺度特征融合模块（新增）
        self.multiscale_fusion = UNet3Plus_Fusion(
            channels_list=[64, 128, 256, 512, 512],  # 编码器各层通道
            out_channel=64  # 融合后输出通道（与x1一致）
        )

        # 3. 保留你原有的解码器结构（恢复后的Up模块）
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 4. 新增：解码器残差连接（连接融合特征与解码器各层输出，强化细节）
        self.decoder_residuals = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=1, bias=False) for _ in range(4)
        ])
        self.decoder_bns = nn.ModuleList([
            nn.BatchNorm2d(64) for _ in range(4)
        ])

        # 5. 输出层（无修改）
        self.outc = OutConv(64)

    def forward(self, x):
        # 1. 编码器前向（保留你的原始逻辑 + MECA）
        x1 = self.inc(x)  # (B,64,H,W) - 最浅层特征（细血管细节）
        x2 = self.down1(x1)  # (B,128,H/2,W/2)
        x3 = self.down2(x2)  # (B,256,H/4,W/4)
        x3_enhanced = self.eca_256(x3)  # MECA增强
        x4 = self.down3(x3_enhanced)  # (B,512,H/8,W/8)
        x5 = self.down4(x4)  # (B,512,H/16,W/16) - 最深层特征（全局语义）

        # 2. UNet3+多尺度融合（输入编码器所有层特征）
        fused_feat = self.multiscale_fusion([x1, x2, x3_enhanced, x4, x5])  # (B,64,H,W)

        # 3. 解码器前向（保留原始逻辑 + 新增残差连接）
        x = self.up1(x5, x4)  # (B,256,H/8,W/8)
        x = self.up2(x, x3_enhanced)  # (B,128,H/4,W/4)
        x = self.up3(x, x2)  # (B,64,H/2,W/2) - 上采样到H/2
        # 解码器残差连接：将融合特征（H,W）下采样到当前尺寸，与解码器输出相加
        res3 = self.decoder_residuals[2](fused_feat)
        res3 = self.decoder_bns[2](res3)
        res3 = F.interpolate(res3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = x + res3  # 残差连接，强化细节

        x = self.up4(x, x1)  # (B,64,H,W)
        # 最终残差连接：融合特征直接与解码器输出相加
        res4 = self.decoder_residuals[3](fused_feat)
        res4 = self.decoder_bns[3](res4)
        x = x + res4  # 残差连接，提升特征表达

        # 4. 输出
        logits = self.outc(x)
        return logits

