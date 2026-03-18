import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 新激活函数 SUGRA --------------------------
class SUGRA(nn.Module):
    """SUGRA激活函数：ReLU的优化版本，含可学习门控参数"""
    def __init__(self, in_features=None):
        super(SUGRA, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # 初始斜率0.1
        self.beta = nn.Parameter(torch.zeros(1))        # 初始偏置0
    
    def forward(self, x):
        gate = torch.sigmoid(self.alpha * x + self.beta)
        return x * gate

# -------------------------- 2. MECA模块（无修改，保持原有功能） --------------------------
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
        enhanced_feat = x * (0.5 + 0.5 * channel_weight)  
        return enhanced_feat

# -------------------------- 3. ECA模块（无修改，保持原有功能） --------------------------
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

# -------------------------- 4. 新增：Inception-like跳跃连接模块 --------------------------
class InceptionSkipModule(nn.Module):
    """
    优化后模块：融合AFF模块优势 + 原有核心设计
    核心改进：
    1. 新增全局特征分支（借鉴AFF的avg pooling，提升血管连通性）
    2. 权重归一化改用exp（借鉴AFF，避免小众分支被抑制）
    3. 保留多尺度卷积分支+通道注意力（适配细小血管提取）
    """
    def __init__(self, in_channels, out_channels=None, cross_level_channels=None):
        super(InceptionSkipModule, self).__init__()
        # 输出通道数默认与输入一致（保证跳跃连接拼接无维度冲突）
        self.out_channels = out_channels if out_channels else in_channels
        self.mid_channels = self.out_channels // 2  # 分支降维，轻量化设计
        # 跨层级特征通道数（若传入，启用跨层级特征融合；默认None，仅单层级处理）
        self.cross_level_channels = cross_level_channels

        # -------------------------- 1. 多尺度卷积分支（保留原有核心，适配血管形态） --------------------------
        # 分支1：1×1卷积 → 通道融合（适配主干血管，粗尺度）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            SUGRA()  # 搭配SUGRA，避免神经元死亡
        )

        # 分支2：3×3卷积 → 细尺度特征（适配细小血管、毛细血管）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            SUGRA()
        )

        # 分支3：1×1+3×3卷积 → 中尺度特征（适配血管分叉、弯曲结构）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            SUGRA(),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            SUGRA()
        )

        # -------------------------- 2. 新增全局特征分支（借鉴AFF，提升血管连通性） --------------------------
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：捕捉血管整体走势（解决连通性差问题）
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            SUGRA(),
            # 上采样恢复原分辨率（与其他分支对齐）
            nn.Upsample(scale_factor=None, mode='bilinear', align_corners=True)  # 分辨率在forward中动态适配
        )

        # -------------------------- 3. 跨层级特征融合分支（可选，借鉴AFF跨层级思想） --------------------------
        if self.cross_level_channels is not None:
            self.cross_level_conv = nn.Sequential(
                nn.Conv2d(self.cross_level_channels, self.mid_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.mid_channels),
                SUGRA(),
                # 上/下采样动态适配当前层级分辨率
                nn.Upsample(scale_factor=None, mode='bilinear', align_corners=True)
            )
            self.total_branches = 5  # 4个基础分支 + 1个跨层级分支
        else:
            self.total_branches = 4  # 仅4个基础分支

        # -------------------------- 4. 自适应权重（改用exp归一化，借鉴AFF，更温和） --------------------------
        self.branch_weights = nn.Parameter(torch.ones(self.total_branches) * (1 / self.total_branches))
        # 无需Softmax，用exp归一化（避免小众分支权重被过度抑制）

        # -------------------------- 5. 通道注意力（保留原有，抑制背景噪声） --------------------------
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        # -------------------------- 6. 多分支融合卷积（适配分支总数） --------------------------
        self.fusion_conv = nn.Conv2d(
            self.mid_channels * self.total_branches,  # 所有分支通道数总和
            self.out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.fusion_bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, cross_level_feat=None):
        """
        Args:
            x: 当前层级浅层特征（如Encoder的x1/x2/x3，必传）
            cross_level_feat: 跨层级特征（如Encoder的x3→x2的上采样特征，可选，需与cross_level_channels匹配）
        Returns:
            enhanced_feat: 融合后增强的特征（用于跳跃连接拼接）
        """
        B, C, H, W = x.shape  # 当前层级特征分辨率
        branch_outputs = []

        # -------------------------- 1. 基础分支前向（4个） --------------------------
        # 分支1：1×1卷积
        b1 = self.branch1(x)
        branch_outputs.append(b1)

        # 分支2：3×3卷积
        b2 = self.branch2(x)
        branch_outputs.append(b2)

        # 分支3：1×1+3×3卷积
        b3 = self.branch3(x)
        branch_outputs.append(b3)

        # 分支4：全局特征（动态适配分辨率）
        self.branch4[4].scale_factor = (H, W)  # 上采样恢复当前层级分辨率
        b4 = self.branch4(x)
        branch_outputs.append(b4)

        # -------------------------- 2. 跨层级分支前向（可选） --------------------------
        if self.cross_level_channels is not None and cross_level_feat is not None:
            # 动态适配跨层级特征分辨率
            self.cross_level_conv[4].scale_factor = (H, W)
            cross_feat = self.cross_level_conv(cross_level_feat)
            branch_outputs.append(cross_feat)

        # -------------------------- 3. 自适应权重融合（exp归一化） --------------------------
        # 权重归一化：exp(w_i) / sum(exp(w_j))（借鉴AFF，更温和）
        exp_weights = torch.exp(self.branch_weights)
        normalized_weights = exp_weights / exp_weights.sum(dim=0)

        # 分支加权（每个分支乘对应归一化权重）
        weighted_branches = []
        for i in range(self.total_branches):
            weighted = branch_outputs[i] * normalized_weights[i]
            weighted_branches.append(weighted)

        # -------------------------- 4. 多分支拼接+融合 --------------------------
        concat_feat = torch.cat(weighted_branches, dim=1)  # 通道维度拼接
        fused_feat = self.fusion_conv(concat_feat)
        fused_feat = self.fusion_bn(fused_feat)

        # -------------------------- 5. 通道注意力增强（抑制背景噪声） --------------------------
        attn_weight = self.channel_attention(x)  # 基于原始输入计算注意力，更贴合浅层特征
        enhanced_feat = fused_feat * attn_weight

        return enhanced_feat

# -------------------------- 5. U-Net基础模块（DoubleConv）：替换ReLU为SUGRA --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            SUGRA(),  # 替换原nn.ReLU(inplace=False)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            SUGRA(),  # 替换原nn.ReLU(inplace=False)
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

# -------------------------- 6. Down模块（无修改，保留卷积下采样） --------------------------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SUGRA(),  # 替换原nn.ReLU(inplace=False)
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_down(x)

# -------------------------- 7. 改造Up模块：融入InceptionSkipModule增强跳跃连接 --------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        # 跳跃连接增强：加入InceptionSkipModule
        self.inception_skip = InceptionSkipModule(skip_channels)
        # 保留原DoubleConv融合特征
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样Decoder特征
        x1 = self.up(x1)
        
        # 尺寸对齐（保持原始逻辑）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 关键修改：用InceptionSkipModule增强跳跃特征x2（Encoder输出）
        x2_enhanced = self.inception_skip(x2)
        
        # 特征拼接（增强后的跳跃特征 + 上采样特征）
        x_concat = torch.cat([x2_enhanced, x1], dim=1)
        return self.conv(x_concat)

# -------------------------- 8. OutConv模块（无修改） --------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------- 9. 最终Unet模型（整合所有模块） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # 编码器（Encoder）：保持原有通道数设计
        self.inc = DoubleConv(n_channels, 64)  # 输入→64
        self.down1 = Down(64, 128)            # 64→128
        self.down2 = Down(128, 256)           # 128→256（MECA输入层）
        self.down3 = Down(256, 512)           # 256→512
        self.down4 = Down(512, 512)           # 512→512

        # 注意力模块（保持原有配置）
        self.eca_256 = MECA(channel=256)  # 对256通道特征增强

        # 解码器（Decoder）：改造Up模块，传入跳跃连接通道数
        self.up1 = Up(1024, 256, skip_channels=512, bilinear=bilinear)  # 跳跃通道=512（x4）
        self.up2 = Up(512, 128, skip_channels=256, bilinear=bilinear)   # 跳跃通道=256（x3_enhanced）
        self.up3 = Up(256, 64, skip_channels=128, bilinear=bilinear)    # 跳跃通道=128（x2）
        self.up4 = Up(128, 64, skip_channels=64, bilinear=bilinear)     # 跳跃通道=64（x1）

        # 输出层
        self.outc = OutConv(64)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.inc(x)                          # 64×H/2×W/2
        x2 = self.down1(x1)                       # 128×H/4×W/4
        x3 = self.down2(x2)                       # 256×H/8×W/8
        x3_enhanced = self.eca_256(x3)            # MECA增强256通道特征
        x4 = self.down3(x3_enhanced)              # 512×H/16×W/16
        x5 = self.down4(x4)                       # 512×H/32×W/32

        # 解码器前向传播（传入增强后的跳跃特征）
        x = self.up1(x5, x4)                      # 上采样x5 + 跳跃特征x4（512通道）
        x = self.up2(x, x3_enhanced)              # 上采样 + 跳跃特征x3_enhanced（256通道）
        x = self.up3(x, x2)                       # 上采样 + 跳跃特征x2（128通道）
        x = self.up4(x, x1)                       # 上采样 + 跳跃特征x1（64通道）

        # 输出预测
        logits = self.outc(x)
        return logits