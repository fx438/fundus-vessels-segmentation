import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 4方向梯度计算模块（保持修复后逻辑） --------------------------
class FourDirGradientConv(nn.Module):
    def __init__(self, in_channels):
        super(FourDirGradientConv, self).__init__()
        self.in_channels = in_channels
        
        # 东北(NE)方向：右上像素 - 当前像素
        ne_kernel = torch.tensor(
            [[[[0, 0, 1], [0, -1, 0], [0, 0, 0]]]],
            dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.conv_ne = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.conv_ne.weight = nn.Parameter(ne_kernel, requires_grad=False)
        
        # 西北(NW)方向：左上像素 - 当前像素
        nw_kernel = torch.tensor(
            [[[[1, 0, 0], [0, -1, 0], [0, 0, 0]]]],
            dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.conv_nw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.conv_nw.weight = nn.Parameter(nw_kernel, requires_grad=False)
        
        # 东南(SE)方向：右下像素 - 当前像素
        se_kernel = torch.tensor(
            [[[[0, 0, 0], [0, -1, 0], [0, 0, 1]]]],
            dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.conv_se = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.conv_se.weight = nn.Parameter(se_kernel, requires_grad=False)
        
        # 西南(SW)方向：左下像素 - 当前像素
        sw_kernel = torch.tensor(
            [[[[0, 0, 0], [0, -1, 0], [1, 0, 0]]]],
            dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.conv_sw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.conv_sw.weight = nn.Parameter(sw_kernel, requires_grad=False)
        
        # 通道压缩：4*in_channels → 4
        self.channel_compress = nn.Conv2d(
            in_channels * 4, 4, kernel_size=1, padding=0, bias=False
        )
        self.bn_compress = nn.BatchNorm2d(4)

    def forward(self, x):
        feat_ne = self.conv_ne(x)
        feat_nw = self.conv_nw(x)
        feat_se = self.conv_se(x)
        feat_sw = self.conv_sw(x)
        four_dir_feat = torch.cat([feat_ne, feat_nw, feat_se, feat_sw], dim=1)
        four_dir_feat = self.bn_compress(self.channel_compress(four_dir_feat))
        return four_dir_feat

# -------------------------- 2. 管状结构滤波器（保持不变） --------------------------
class TubularStructureFilter(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(TubularStructureFilter, self).__init__()
        self.hessian_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        hessian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], 
                                      dtype=torch.float32).repeat(in_channels, 1, 1, 1)
        self.hessian_conv.weight = nn.Parameter(hessian_kernel, requires_grad=False)
        
        self.prob_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        tubular_resp = torch.abs(self.hessian_conv(x))
        return self.prob_conv(tubular_resp)

# -------------------------- 3. 焦点注意力模块（保持不变） --------------------------
class FocalAttention(nn.Module):
    def __init__(self, channel, kernel_size=3, gamma=1.2):
        super(FocalAttention, self).__init__()
        self.gamma = gamma
        self.channel = channel
        self.four_dir_grad = FourDirGradientConv(in_channels=channel)
        self.fuse_conv = nn.Conv2d(channel + 4, channel, kernel_size=1, padding=0, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channel)
        self.local_conv = nn.Conv2d(channel, channel, kernel_size=kernel_size, 
                                    padding=kernel_size//2, groups=channel, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                        padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        four_dir_feat = self.four_dir_grad(x)
        x_with_grad = torch.cat([x, four_dir_feat], dim=1)
        x_fused = self.fuse_bn(self.fuse_conv(x_with_grad))
        local_feat = self.bn(self.local_conv(x_fused))
        avg_feat = torch.mean(local_feat, dim=1, keepdim=True)
        max_feat, _ = torch.max(local_feat, dim=1, keepdim=True)
        sig_feat = torch.cat([avg_feat, max_feat], dim=1)
        attention_map = self.sigmoid(self.attention_conv(sig_feat))
        focal_mask = (1 - attention_map) ** self.gamma * attention_map
        return x * focal_mask + x

# -------------------------- 4. RCSA模块（保持不变，适配1024通道） --------------------------
class RCSA(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(RCSA, self).__init__()
        self.meca = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Identity()
        )
        self.four_dir_grad = FourDirGradientConv(in_channels=channel)
        self.meca_gate = nn.Sequential(
            nn.Conv2d(channel * 2 + 4, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.meca_res_conv = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False)
        self.meca_res_bn = nn.BatchNorm2d(channel)
        self.meca_dropout = nn.Dropout(p=0.1)
        self.spatial_conv = nn.Conv2d(6, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
        self.tubular_filter = TubularStructureFilter(in_channels=channel)
        self.deep_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.deep_gate = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.meca[0](x)
        four_dir_feat = self.four_dir_grad(x)
        global_feat = nn.AdaptiveAvgPool2d(1)(x).expand_as(local_feat)
        concat_feat = torch.cat([local_feat, global_feat, four_dir_feat], dim=1)
        
        channel_weight = self.meca_gate(concat_feat)
        enhanced_feat = x * channel_weight
        residual = self.meca_res_bn(self.meca_res_conv(x))
        enhanced_feat = enhanced_feat + residual
        enhanced_feat = self.meca_dropout(enhanced_feat)
        
        avg_out = torch.mean(enhanced_feat, dim=1, keepdim=True)
        max_out = torch.max(enhanced_feat, dim=1, keepdim=True)[0]
        four_dir_feat_rcsa = self.four_dir_grad(enhanced_feat)
        concat_out = torch.cat([avg_out, max_out, four_dir_feat_rcsa], dim=1)
        tubular_prob = self.tubular_filter(enhanced_feat)
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(concat_out))
        spatial_weight = spatial_weight * tubular_prob
        enhanced_feat = enhanced_feat * spatial_weight
        
        deep_y = self.deep_avg_pool(enhanced_feat)
        deep_channel_weight = self.deep_gate(deep_y)
        return enhanced_feat * deep_channel_weight

# -------------------------- 5. 基础卷积模块（保持不变） --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.four_dir_grad = FourDirGradientConv(in_channels=in_channels)
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels + 4, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        four_dir_feat = self.four_dir_grad(x)
        x_with_grad = torch.cat([x, four_dir_feat], dim=1)
        return self.conv_down(x_with_grad)

class GroupDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(GroupDoubleConv, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels, groups=2):
        super(Up, self).__init__()
        self.groups = groups
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.conv = GroupDoubleConv(concat_channels + 4, out_channels, groups=groups)
        self.four_dir_grad = FourDirGradientConv(in_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        x2_grad = self.four_dir_grad(x2)
        x2_with_grad = torch.cat([x2, x2_grad], dim=1)
        x = torch.cat([x2_with_grad, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------- 6. 最终Unet模型（已加回瓶颈层RSCA） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.n_channels = n_channels

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
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

        # 瓶颈层（加回RSCA，通道数1024）
        self.bottleneck_enhance = nn.Sequential(
            RCSA(channel=1024, kernel_size=3),  # 重新加入瓶颈层RSCA
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_res = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(1024)

        # 解码器
        self.up1 = Up(1024, 512, 512+512, groups=2)
        self.focal_attention_512 = FocalAttention(channel=512, kernel_size=3, gamma=1.2)
        
        self.up2 = Up(512, 256, 256+256, groups=2)
        self.up3 = Up(256, 128, 128+128, groups=2)
        self.up4 = Up(128, 64, 64+64, groups=2)

        # 最终增强层
        self.final_enhance = DoubleConv(64, 64)
        self.focal_attention_final = FocalAttention(channel=64, kernel_size=3, gamma=1.2)
        self.outc = OutConv(64)

    def forward(self, x):
        B, C, H_in, W_in = x.shape

        # 编码器前向
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

        # 瓶颈层前向（含RSCA增强）
        x5_enhanced = self.bottleneck_enhance(x5)  # 包含RSCA特征提纯
        x5_residual = self.bottleneck_res(x5)
        x5_residual = self.bottleneck_bn(x5_residual)
        x5_final = x5_enhanced + x5_residual

        # 解码器前向
        x = self.up1(x5_final, x4)
        x = self.focal_attention_512(x)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_up4 = self.up4(x, x1)

        # 最终输出
        x_final = self.final_enhance(x_up4)
        x_final_focal = self.focal_attention_final(x_final)
        logits = self.outc(x_final_focal)
        
        if logits.shape[2:] != (H_in, W_in):
            logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return logits

