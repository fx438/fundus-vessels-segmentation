import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 焦点注意力模块（保留不变） --------------------------
class FocalAttention(nn.Module):
    def __init__(self, channel, kernel_size=3, gamma=1.2):  # gamma从2.0下调至1.2
        super(FocalAttention, self).__init__()
        self.gamma = gamma  # 弱化难例加权，避免过度聚焦局部噪声
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
        
        enhanced_feat = x * focal_mask + x  # 残差连接，保证特征不丢失
        return enhanced_feat

# -------------------------- RCSA模块（保留不变） --------------------------
class RCSA(nn.Module):
    def __init__(self, channel, kernel_size=2, pool_scales=[1, 2, 3]):  # 核心改1：缩小池化/卷积核
        super(RCSA, self).__init__()
        # 1. 多尺度池化优化：极细尺度+混合池化（平均+最大）+非对称池化
        self.multi_pool = nn.ModuleList()
        self.pool_scales = pool_scales
        self.num_pool_branches = len(pool_scales)*2 + 2  # 对称池化(6) + 非对称池化(2) = 8个分支
        for scale in pool_scales:
            # 对称池化（适配圆形细小血管）+非对称池化（适配线性分支）
            self.multi_pool.append(nn.AvgPool2d(scale, stride=1, padding=scale//2))  # 平均池化保留纹理
            self.multi_pool.append(nn.MaxPool2d(scale, stride=1, padding=scale//2))  # 最大池化保留边缘
        # 新增：非对称池化分支（适配线性细小血管）
        self.asym_pool_h = nn.AvgPool2d((1, 3), stride=1, padding=(0, 1))  # 横向池化
        self.asym_pool_v = nn.AvgPool2d((3, 1), stride=1, padding=(1, 0))  # 纵向池化

        # 2. 尺度注意力权重：优先激活小尺度特征（聚焦细小血管）
        # 关键修复：输入通道数改为 C//4 * 分支数（PSP降维后）
        self.scale_attn = nn.Sequential(
            nn.Conv2d((channel//4)*self.num_pool_branches, self.num_pool_branches, kernel_size=1),
            nn.Softmax(dim=1)  # 生成各尺度权重，小尺度权重更高
        )

        # 3. 通道注意力重构：Softmax替代Sigmoid + 通道权重系数
        # 关键修复：输入通道数 = 多尺度特征通道(C//4) + 全局特征通道(C)
        self.meca_gate = nn.Sequential(
            nn.Conv2d((channel//4) + channel, channel, kernel_size=1, bias=True),
            nn.Softmax(dim=1)  # 核心改：平衡低响应细小血管通道权重
        )
        self.small_vessel_coeff = nn.Parameter(torch.ones(channel))  # 可学习的细小血管通道系数

        # 4. 残差连接+正则化优化：低概率Dropout + 加权残差融合
        self.meca_res_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.meca_res_bn = nn.BatchNorm2d(channel)
        self.meca_dropout = nn.Dropout(p=0.08)  # 核心改：降低Dropout概率，保留弱特征

        # 5. 空间注意力升级：小卷积核 + Attention Gate（抑制背景噪声）
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)  # 2×2卷积核
        self.spatial_sigmoid = nn.Sigmoid()
        # Attention Gate：前景（血管）-背景二分类，抑制背景噪声
        self.attn_gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 6. 轻量PSP + 上采样恢复分辨率（补充上下文+恢复细小血管细节）
        self.psp_conv = nn.ModuleList([
            nn.Conv2d(channel, channel//4, kernel_size=1) for _ in range(self.num_pool_branches)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样恢复像素细节
        self.fusion_conv = nn.Conv2d(channel*2, channel, kernel_size=1)  # 跨尺度融合

    def forward(self, x):
        B, C, H, W = x.shape
        x_ori = x  # 保存原始特征，用于后续融合

        # -------------------------- 步骤1：多尺度特征提取（强制尺寸对齐） --------------------------
        # 对称混合池化（平均+最大）
        multi_local_feats = [pool(x) for pool in self.multi_pool]
        # 非对称池化（捕捉线性细小血管）
        multi_local_feats.append(self.asym_pool_h(x))
        multi_local_feats.append(self.asym_pool_v(x))
        
        # 核心修复1：强制所有池化特征图尺寸与原始x一致
        multi_local_feats = [
            F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            for feat in multi_local_feats
        ]

        # 轻量PSP：降维减少冗余，聚焦细小血管特征
        multi_local_feats = [self.psp_conv[i](feat) for i, feat in enumerate(multi_local_feats)]
        
        # 核心修复2：尺度注意力输入通道对齐
        multi_feat_cat = torch.cat(multi_local_feats, dim=1)  # [B, (C//4)*8, H, W]
        scale_weights = self.scale_attn(multi_feat_cat)  # [B, 8, H, W]
        
        # 按尺度加权（强制权重尺寸匹配）
        multi_local_feat = 0
        for i, feat in enumerate(multi_local_feats):
            weight = scale_weights[:, i:i+1, :, :]
            # 强制权重尺寸与特征图一致
            weight = F.interpolate(weight, size=(H, W), mode='bilinear', align_corners=False)
            multi_local_feat += feat * weight
        
        # 对齐多尺度特征尺寸
        multi_local_feat = F.interpolate(multi_local_feat, size=(H, W), mode='bilinear', align_corners=False)

        # -------------------------- 步骤2：全局特征+通道注意力（Softmax+通道系数） --------------------------
        global_feat = nn.AdaptiveAvgPool2d(1)(x).expand_as(x)
        # 核心修复3：拼接通道数对齐
        concat_feat = torch.cat([multi_local_feat, global_feat], dim=1)  # [B, (C//4)+C, H, W]
        channel_weight = self.meca_gate(concat_feat)
        # 通道权重系数：强化细小血管对应通道
        channel_weight = channel_weight * self.small_vessel_coeff.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        enhanced_feat = x * channel_weight

        # -------------------------- 步骤3：残差连接（加权融合+低概率Dropout） --------------------------
        residual = self.meca_res_conv(x)
        residual = self.meca_res_bn(residual)
        # 加权残差融合：保留原始弱特征（细小血管）
        enhanced_feat = 0.7 * x + 0.3 * enhanced_feat
        enhanced_feat = enhanced_feat + residual
        enhanced_feat = self.meca_dropout(enhanced_feat)

        # -------------------------- 步骤4：空间注意力（小卷积核+Attention Gate） --------------------------
        avg_out = torch.mean(enhanced_feat, dim=1, keepdim=True)
        max_out = torch.max(enhanced_feat, dim=1, keepdim=True)[0]
        concat_out = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(concat_out))  # 2×2卷积核，保留细节
        # Attention Gate：抑制背景噪声，聚焦前景血管
        gate = self.attn_gate(enhanced_feat)
        spatial_weight = spatial_weight * gate  # 仅在前景区域激活空间注意力
        enhanced_feat = enhanced_feat * spatial_weight

        # -------------------------- 步骤5：上采样+跨尺度融合（恢复细小血管分辨率） --------------------------
        enhanced_feat_up = self.upsample(enhanced_feat)  # 2×上采样，恢复像素细节
        x_ori_up = self.upsample(x_ori)
        fusion_feat = torch.cat([enhanced_feat_up, x_ori_up], dim=1)
        final_feat = self.fusion_conv(fusion_feat)
        # 降回原始尺寸，避免维度不匹配
        final_feat = F.interpolate(final_feat, size=(H, W), mode='bilinear', align_corners=False)

        return final_feat

# -------------------------- 基础注意力门（原始版本，保留不变） --------------------------
class AttentionGate(nn.Module):
    """
    轻量型注意力门：区分细小血管和背景噪声，减少假阳性（原始版本）
    """
    def __init__(self, encoder_channels, decoder_channels, hidden_channels=None):
        super(AttentionGate, self).__init__()
        # 默认隐藏通道数 = 编码器通道数的一半（轻量设计）
        self.hidden_channels = hidden_channels if hidden_channels is not None else encoder_channels // 2
        
        # 通道调整卷积
        self.encoder_conv = nn.Conv2d(encoder_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.decoder_conv = nn.Conv2d(decoder_channels, self.hidden_channels, kernel_size=1, bias=False)
        
        # 注意力掩码生成
        self.attention_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 残差连接：保护细小血管弱信号
        self.residual_conv = nn.Conv2d(encoder_channels, encoder_channels, kernel_size=1, bias=False)

    def forward(self, encoder_feat, decoder_feat):
        # 尺寸对齐：解码器特征对齐到编码器特征尺寸
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, size=encoder_feat.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # 通道映射
        e_feat = self.encoder_conv(encoder_feat)
        d_feat = self.decoder_conv(decoder_feat)
        
        # 生成注意力掩码
        attention_logits = self.attention_conv(torch.relu(e_feat + d_feat))
        attention_mask = self.sigmoid(attention_logits)
        
        # 应用掩码+残差连接
        gated_encoder_feat = encoder_feat * attention_mask
        gated_encoder_feat = gated_encoder_feat + self.residual_conv(encoder_feat)
        
        return gated_encoder_feat

# -------------------------- 方案2：带血管连续性强化的循环跳跃连接（保留不变） --------------------------
class RecurrentSkipConnectionWithAGAndContinuity(nn.Module):
    """
    带注意力门+血管连续性强化的递归跳跃连接：恢复低对比度血管连续性，兼顾低假阳性
    核心改进：1. 跨轮特征保留 2. 连续性权重 3. 避免弱信号在迭代中丢失
    """
    def __init__(self, encoder_channels, decoder_channels, hidden_channels=None, num_recur_rounds=2):
        super(RecurrentSkipConnectionWithAGAndContinuity, self).__init__()
        self.hidden_channels = hidden_channels if hidden_channels is not None else decoder_channels
        self.num_recur_rounds = num_recur_rounds
        
        # 编码器端注意力门（使用基础版本AG）
        self.attention_gate = AttentionGate(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            hidden_channels=self.hidden_channels // 2
        )
        
        # 通道调整卷积
        self.encoder_adjust = nn.Conv2d(encoder_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.decoder_adjust = nn.Conv2d(decoder_channels, self.hidden_channels, kernel_size=1, bias=False)
        
        # 递归融合卷积
        self.recur_conv = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        
        # 最终通道适配
        self.final_adjust = nn.Conv2d(self.hidden_channels, decoder_channels, kernel_size=1, bias=False)
        
        # -------------------------- 方案2核心：连续性强化机制 --------------------------
        # 1. 跨轮特征保留权重：防止低对比度血管信号在迭代中被稀释
        self.continuity_weight = nn.Parameter(torch.ones(1))  # 可学习参数，初始为1
        # 2. 迭代特征融合系数：平衡当前轮特征与历史特征
        self.iter_fusion_coeff = nn.Parameter(torch.tensor(0.5))  # 初始平衡权重

    def forward(self, encoder_feat, decoder_feat):
        # 步骤1：编码器特征降噪（注意力门筛选）
        gated_encoder_feat = self.attention_gate(encoder_feat, decoder_feat)
        
        # 步骤2：尺寸对齐
        if gated_encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            h, w = decoder_feat.shape[2:]
            gated_encoder_feat = F.adaptive_avg_pool2d(gated_encoder_feat, (h, w))
        
        # 步骤3：通道调整
        e_feat = self.encoder_adjust(gated_encoder_feat)
        d_feat = self.decoder_adjust(decoder_feat)
        
        # 步骤4：初始化融合 + 保存初始特征（关键：保留低对比度血管的初始弱信号）
        f_prev = e_feat + d_feat
        f_init = f_prev.clone()  # 保存初始融合特征，用于跨轮连续性强化
        
        # 步骤5：多轮递归融合（方案2核心：加入跨轮特征保留）
        for _ in range(self.num_recur_rounds):
            # 融合当前轮特征 + 初始特征 + 连续性权重：避免弱信号被迭代稀释
            fusion_input = f_prev + e_feat + d_feat + self.continuity_weight * f_init
            # 平衡当前特征与历史特征
            f_current = self.activation(self.recur_conv(fusion_input))
            f_prev = self.iter_fusion_coeff * f_prev + (1 - self.iter_fusion_coeff) * f_current
        
        # 步骤6：最终通道适配
        fusion_feat = self.final_adjust(f_prev)
        
        return fusion_feat

# -------------------------- 基础模块（DoubleConv/Down/GroupDoubleConv/OutConv） --------------------------
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
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DoubleConv(out_channels, out_channels)
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

# -------------------------- Up模块（保留不变，使用方案2） --------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels, groups=4, num_recur_rounds=2):
        super(Up, self).__init__()
        self.groups = groups
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )
        self.bn_up = nn.BatchNorm2d(out_channels)
        
        # 核心修改：替换为方案2的循环跳跃连接
        self.recurrent_skip = RecurrentSkipConnectionWithAGAndContinuity(
            encoder_channels=out_channels,
            decoder_channels=out_channels,
            hidden_channels=out_channels,
            num_recur_rounds=num_recur_rounds
        )
        
        # 核心修复：GroupDoubleConv的输入通道数改为out_channels
        self.conv = GroupDoubleConv(out_channels, out_channels, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.bn_up(x1)
        
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        # 用方案2的循环跳跃连接替换原有模块
        fusion_feat = self.recurrent_skip(x2, x1)
        x = self.conv(fusion_feat)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------- 完整Unet模型（核心修复：添加rcsa_aux_conv属性） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.n_channels = n_channels

        # 编码器（保留不变）
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

        # 瓶颈层（保留不变）
        self.bottleneck_enhance = nn.Sequential(
            RCSA(channel=1024, kernel_size=3),
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_res = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(1024)

        # 解码器（使用替换后的Up模块，含方案2）
        self.up1 = Up(1024, 512, 512+512, groups=4, num_recur_rounds=2)
        self.focal_attention_512 = FocalAttention(channel=512, kernel_size=3, gamma=1.2)
        
        self.up2 = Up(512, 256, 256+256, groups=4, num_recur_rounds=2)
        self.up3 = Up(256, 128, 128+128, groups=4, num_recur_rounds=2)
        self.up4 = Up(128, 64, 64+64, groups=4, num_recur_rounds=2)

        # 最终增强层（保留不变）
        self.final_enhance = DoubleConv(64, 64)
        self.focal_attention_final = FocalAttention(channel=64, kernel_size=3, gamma=1.2)
        self.outc = OutConv(64)

        # -------------------------- 核心修复：提前初始化rcsa_aux_conv --------------------------
        # 与rcsa_aux_loss函数中的定义一致，避免属性缺失
        self.rcsa_aux_conv = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        B, C, H_in, W_in = x.shape

        # 编码器前向（保留不变）
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

        # 瓶颈层前向（保留不变）
        x5_enhanced = self.bottleneck_enhance(x5)
        x5_residual = self.bottleneck_res(x5)
        x5_residual = self.bottleneck_bn(x5_residual)
        x5_final = x5_enhanced + x5_residual

        # 解码器前向（使用方案2模块）
        x = self.up1(x5_final, x4)
        x = self.focal_attention_512(x)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_up4 = self.up4(x, x1)

        # 最终增强（保留不变）
        x_final = self.final_enhance(x_up4)
        x_final_focal = self.focal_attention_final(x_final)
        logits = self.outc(x_final_focal)

        # 尺寸还原
        if logits.shape[2:] != (H_in, W_in):
            logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return logits