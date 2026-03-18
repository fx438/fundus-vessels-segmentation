import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 修复维度不匹配的MMC（Morph Mamba Convolution）层 --------------------------
class MMC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super(MMC, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 动态形态偏移学习（仅学习x、y两个方向的偏移，避免维度不匹配）
        self.morph_offset = nn.Parameter(torch.randn(2) * 0.1)  # 修复1：仅2维（x、y方向），全局共享偏移
        # 多视角特征提取（x轴、y轴独立卷积核）
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                                padding=(self.padding, 0), groups=groups, bias=False)
        self.conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), 
                                padding=(0, self.padding), groups=groups, bias=False)
        # 状态空间建模（简化版SSM）
        self.ssm = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 输出特征调整
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 步骤1：动态形态偏移（修复维度匹配问题）
        # 1.1 生成标准坐标网格（2, H, W）：第0维为x坐标，第1维为y坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), 
                                        torch.arange(W, device=x.device), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # 尺寸：(2, H, W)，对应(x,y)
        
        # 1.2 调整偏移量维度（修复核心：确保offset与grid第0维均为2）
        offset = self.morph_offset.unsqueeze(1).unsqueeze(2)  # 从(2,) → (2, 1, 1)
        grid = grid + offset  # 现在维度匹配：(2, H, W) + (2, 1, 1) → (2, H, W)
        
        # 1.3 归一化网格到[-1,1]（适配F.grid_sample的输入要求）
        grid = grid / torch.tensor([W-1, H-1], device=x.device).unsqueeze(1).unsqueeze(2) * 2 - 1
        grid = grid.permute(1, 2, 0).unsqueeze(0).repeat(B, 1, 1, 1)  # 最终尺寸：(B, H, W, 2)
        
        # 1.4 双线性插值获取偏移后的特征
        x_offset = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # 步骤2：多视角特征提取（x轴、y轴独立卷积）
        feat_x = self.conv_x(x_offset)  # x轴方向特征（捕捉水平血管）
        feat_y = self.conv_y(x_offset)  # y轴方向特征（捕捉垂直血管）
        feat_multi = torch.cat([feat_x, feat_y], dim=1)  # 拼接多视角特征：(B, 2*out_channels, H, W)
        
        # 步骤3：状态空间建模（整合多视角信息）
        feat_ssm = self.ssm(feat_multi)  # 输出：(B, out_channels, H, W)
        
        # 步骤4：输出调整（匹配原DoubleConv的激活与Dropout逻辑）
        out = self.bn(feat_ssm)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out

# -------------------------- 2. 原有辅助模块保持不变（FocalAttention、RCSA等） --------------------------
class FocalAttention(nn.Module):
    def __init__(self, channel, kernel_size=3, gamma=1.2):
        super(FocalAttention, self).__init__()
        self.gamma = gamma
        self.local_conv = nn.Conv2d(channel, channel, kernel_size=kernel_size, 
                                    padding=kernel_size//2, groups=channel, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                        padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        local_feat = self.local_conv(x)
        local_feat = self.bn(local_feat)
        
        avg_feat = torch.mean(local_feat, dim=1, keepdim=True)
        max_feat, _ = torch.max(local_feat, dim=1, keepdim=True)
        sig_feat = torch.cat([avg_feat, max_feat], dim=1)
        
        attention_map = self.attention_conv(sig_feat)
        attention_map = self.sigmoid(attention_map)
        focal_mask = (1 - attention_map) ** self.gamma * attention_map
        
        enhanced_feat = x * focal_mask + x
        return enhanced_feat

class RCSA(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(RCSA, self).__init__()
        self.meca = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Identity()
        )
        self.meca_gate = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.meca_res_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.meca_res_bn = nn.BatchNorm2d(channel)
        self.meca_dropout = nn.Dropout(p=0.2)
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
        
        self.deep_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.deep_gate = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.meca[0](x)
        global_feat = nn.AdaptiveAvgPool2d(1)(x).expand_as(local_feat)
        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        channel_weight = self.meca_gate(concat_feat)
        enhanced_feat = x * channel_weight
        
        residual = self.meca_res_conv(x)
        residual = self.meca_res_bn(residual)
        enhanced_feat = enhanced_feat + residual
        enhanced_feat = self.meca_dropout(enhanced_feat)
        
        avg_out = torch.mean(enhanced_feat, dim=1, keepdim=True)
        max_out = torch.max(enhanced_feat, dim=1, keepdim=True)[0]
        concat_out = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(concat_out))
        enhanced_feat = enhanced_feat * spatial_weight
        
        deep_y = self.deep_avg_pool(enhanced_feat)
        deep_channel_weight = self.deep_gate(deep_y)
        final_feat = enhanced_feat * deep_channel_weight
        
        return final_feat

# -------------------------- 3. Down、GroupDoubleConv、Up、OutConv模块保持不变 --------------------------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MMC(out_channels, out_channels, kernel_size=3)  # 调用修复后的MMC
        )

    def forward(self, x):
        return self.conv_down(x)

class GroupDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(GroupDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels, groups=4):
        super(Up, self).__init__()
        self.groups = groups
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.conv = GroupDoubleConv(concat_channels, out_channels, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------- 4. 最终MMC替换版Unet模型（保持不变） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.n_channels = n_channels

        self.inc = MMC(n_channels, 64, kernel_size=3)  # 修复后的MMC
        self.focal_attention_64 = FocalAttention(channel=64, kernel_size=3, gamma=1.2)
        
        self.down1 = Down(64, 128)
        self.focal_attention_128 = FocalAttention(channel=128, kernel_size=3, gamma=1.2)
        
        self.down2 = Down(128, 256)
        self.focal_attention_256 = FocalAttention(channel=256, kernel_size=3, gamma=1.2)
        self.rcsa = RCSA(channel=256, kernel_size=3)
        
        self.cross_res_x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.bottleneck_enhance = nn.Sequential(
            RCSA(channel=1024, kernel_size=3),
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_res = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(1024)

        self.up1 = Up(1024, 512, 512+512, groups=4)
        self.focal_attention_512 = FocalAttention(channel=512, kernel_size=3, gamma=1.2)
        
        self.up2 = Up(512, 256, 256+256, groups=4)
        self.up3 = Up(256, 128, 128+128, groups=4)
        self.up4 = Up(128, 64, 64+64, groups=4)

        self.final_enhance = MMC(64, 64, kernel_size=3)  # 修复后的MMC
        self.focal_attention_final = FocalAttention(channel=64, kernel_size=3, gamma=1.2)
        self.outc = OutConv(64)

    def forward(self, x):
        B, C, H_in, W_in = x.shape

        x1 = self.inc(x)
        x1_focal = self.focal_attention_64(x1)
        
        x2 = self.down1(x1_focal)
        x2_focal = self.focal_attention_128(x2)
        
        x3 = self.down2(x2_focal)
        x3_focal = self.focal_attention_256(x3)
        x3_enhanced = self.rcsa(x3_focal)
        
        x4 = self.down3(x3_enhanced)
        x3_cross = self.cross_res_x3(x3_enhanced)
        x4 = x4 + x3_cross
        x5 = self.down4(x4)

        x5_enhanced = self.bottleneck_enhance(x5)
        x5_residual = self.bottleneck_res(x5)
        x5_residual = self.bottleneck_bn(x5_residual)
        x5_final = x5_enhanced + x5_residual

        x = self.up1(x5_final, x4)
        x = self.focal_attention_512(x)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_up4 = self.up4(x, x1)

        x_final = self.final_enhance(x_up4)
        x_final_focal = self.focal_attention_final(x_final)

        logits = self.outc(x_final_focal)
        if logits.shape[2:] != (H_in, W_in):
            logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return logits

