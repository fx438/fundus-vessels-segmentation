import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 所有基础模块保持不变 --------------------------
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

# -------------------------- 核心修改：Mamba模块添加3处Dropout --------------------------
class CustomMamba2D(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=3, expand=1, dropout_rate=0.15):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dropout_rate = dropout_rate  # 新增dropout率参数（默认0.15，可调整）

        self.in_proj = nn.Conv2d(d_model, self.d_inner, kernel_size=1, bias=False)
        self.in_norm = nn.BatchNorm2d(self.d_inner)
        self.in_dropout = nn.Dropout(p=self.dropout_rate)  # 1. 输入投影后dropout（破坏初始特征相关性）
        
        self.conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv//2,
            groups=self.d_inner, bias=False
        )
        self.conv_norm = nn.BatchNorm2d(self.d_inner)
        self.conv_act = nn.SiLU()
        
        self.B = nn.Conv2d(self.d_inner, self.d_state, kernel_size=1, bias=False)
        self.C = nn.Conv2d(self.d_state, self.d_inner, kernel_size=1, bias=False)
        self.A = nn.Parameter(torch.ones(1, self.d_state, 1) * -0.1)
        self.D = nn.Parameter(torch.ones(1, self.d_inner, 1, 1) * 0.1)

        self.gate_proj = nn.Conv2d(self.d_inner, self.d_inner * 2, kernel_size=1, bias=False)
        self.gate_norm = nn.BatchNorm2d(self.d_inner * 2)
        
        self.ssm_dropout = nn.Dropout(p=self.dropout_rate)  # 2. SSM输出后dropout（核心正则化，破坏过强依赖）
        self.out_dropout = nn.Dropout(p=self.dropout_rate)  # 3. 输出投影前dropout（最终特征正则化）

        self.out_proj = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        assert C == self.d_model, f"Mamba输入通道数必须为{self.d_model}，当前为{C}"

        x_proj = self.in_proj(x)
        x_proj = self.in_norm(x_proj)
        x_proj = self.conv_act(x_proj)
        x_proj = self.in_dropout(x_proj)  # 应用1：输入投影后dropout
        
        x_conv = self.conv(x_proj)
        x_conv = self.conv_norm(x_conv)
        x_conv = self.conv_act(x_conv)
        B, C_inner, H_conv, W_conv = x_conv.shape

        x_flat = x_conv.flatten(start_dim=2)  # [B, d_inner, L]
        L = x_flat.shape[-1]
        B_proj = self.B(x_conv).flatten(start_dim=2)  # [B, d_state, L]

        # 前缀和+广播并行计算（保持不变）
        exp_A = torch.exp(self.A)
        exp_A_pows = exp_A ** torch.arange(L, device=x.device).reshape(1, 1, L)
        exp_A_pows_flip = torch.flip(exp_A_pows, dims=[-1])
        exp_A_cumprod = torch.cumprod(exp_A_pows_flip, dim=-1)
        exp_A_weights = torch.flip(exp_A_cumprod, dims=[-1])
        
        B_weighted = B_proj * exp_A_pows
        state_seq = torch.cumsum(B_weighted, dim=-1)

        C_proj = self.C(state_seq.reshape(B, self.d_state, H_conv, W_conv))
        D_proj = self.D * x_conv
        ssm_out = C_proj + D_proj
        ssm_out = self.ssm_dropout(ssm_out)  # 应用2：SSM核心输出后dropout（关键！）

        gate_shift = self.gate_proj(x_proj)
        gate_shift = self.gate_norm(gate_shift)
        gate, shift = gate_shift.chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        shift = torch.tanh(shift)
        x_gated = gate * (ssm_out + shift)

        x_gated = self.out_dropout(x_gated)  # 应用3：输出投影前dropout
        x_out = self.out_proj(x_gated)
        x_out = self.out_norm(x_out)
        
        if x_out.shape[2:] != (H_in, W_in):
            x_out = F.interpolate(x_out, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return x_out

# -------------------------- 编码器单个Mamba层（保持不变，传入dropout率） --------------------------
class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.n_channels = n_channels

        # 编码器（单个Mamba层，传入dropout_rate=0.15）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        self.mamba_encoder = CustomMamba2D(
            d_model=512,
            d_state=16,
            d_conv=3,
            expand=1,
            dropout_rate=0.15  # 编码器Mamba dropout率（可调整）
        )
        self.mamba_encoder_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.down4 = Down(512, 1024)

        # 瓶颈层（保持不变）
        self.bottleneck_enhance = nn.Sequential(
            RCSA(channel=1024, kernel_size=3),
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_res = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(1024)

        # 解码器（保持不变）
        self.up1 = Up(1024, 512, 512+512, groups=4)
        self.up2 = Up(512, 256, 256+256, groups=4)
        self.up3 = Up(256, 128, 128+128, groups=4)
        self.up4 = Up(128, 64, 64+64, groups=4)

        # 解码器Mamba（传入dropout_rate=0.1）
        self.mamba_decoder = CustomMamba2D(
            d_model=64,
            d_state=16,
            d_conv=3,
            expand=1,
            dropout_rate=0.1  # 解码器Mamba dropout率（略低，避免破坏高频细节）
        )
        self.mamba_decoder_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.outc = OutConv(64)
        self.rcsa = RCSA(channel=256, kernel_size=3)
        self.cross_res_x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        B, C, H_in, W_in = x.shape

        # 编码器前向
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x4 = self.down3(x3)
        x3_enhanced = self.rcsa(x3)
        x3_cross = self.cross_res_x3(x3_enhanced)
        x4 = x4 + x3_cross
        
        # Mamba前向 + 残差连接
        x4_mamba = self.mamba_encoder(x4)
        x4_res = self.mamba_encoder_res(x4)
        x4_enhanced = x4_mamba + x4_res

        x5 = self.down4(x4_enhanced)

        # 瓶颈层
        x5_enhanced = self.bottleneck_enhance(x5)
        x5_residual = self.bottleneck_res(x5)
        x5_final = x5_enhanced + x5_residual

        # 解码器前向
        x = self.up1(x5_final, x4_enhanced)
        x = self.up2(x, x3_enhanced)
        x = self.up3(x, x2)
        x_up4 = self.up4(x, x1)

        # 解码器Mamba
        x_mamba = self.mamba_decoder(x_up4)
        x_mamba_res = self.mamba_decoder_res(x_up4)
        x_final = x_mamba + x_mamba_res

        # 输出层
        logits = self.outc(x_final)
        if logits.shape[2:] != (H_in, W_in):
            logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return logits