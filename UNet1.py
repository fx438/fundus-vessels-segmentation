import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
import cv2
import numpy as np

# -------------------------- 通道注意力模块（强化有效特征通道） --------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//4),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//4, in_ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        att = self.fc(avg).view(b, c, 1, 1)
        return x * att

# -------------------------- 空间注意力模块（强化血管区域空间特征） --------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (kernel_size - 1) * 2 // 2  # 适配dilation=2，保持尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False, dilation=2)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat_out = torch.cat([max_out, avg_out], dim=1)
        att_map = self.conv(concat_out)
        att_map = self.bn(att_map)
        att_weight = self.sigmoid(att_map)
        att_weight = att_weight * 1.5
        att_weight = torch.clamp(att_weight, 0, 1)
        return x * att_weight

# -------------------------- 多尺度卷积模块（捕捉不同尺度细小血管） --------------------------
class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 3  # 15//3=5，确保整除
        self.conv1 = nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=3, dilation=3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        out = torch.cat([feat1, feat2, feat3], dim=1)  # 5+5+5=15通道
        out = self.bn(out)
        out = self.relu(out)
        return out

# -------------------------- DoubleConv模块（Unet基础卷积块） --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# -------------------------- 核心Unet模型（无索引越界，适配细小血管分割） --------------------------
class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(Unet, self).__init__()

        # 编码器
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)

        # 解码器
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

        # 辅助模块（无冗余，索引明确）
        self.att1 = SpatialAttention(kernel_size=3)  # c1特征空间注意力
        self.detail_residual = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  # 残差连接
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 短路径融合权重

        # 短路径分支（6层：0-5，索引绝对不越界）
        self.small_vessel_branch = nn.Sequential(
            MultiScaleConv(32, 15),          # 0: 32→15（多尺度特征）
            ChannelAttention(15),            # 1: 通道注意力
            nn.Conv2d(15, 15, kernel_size=3, padding=1),  # 2: 15→15卷积
            nn.BatchNorm2d(15),              # 3: BN层
            nn.ReLU(inplace=True),           # 4: ReLU激活
            nn.Conv2d(15, out_ch, kernel_size=1)  # 5: 输出层（15→1）
        )

        # 分支血管注意力（筛选血管区域）
        self.branch_vessel_att = nn.Sequential(
            nn.Conv2d(15, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self._init_weights()  # 初始化权重

    def _init_weights(self):
        """初始化所有层权重，确保训练稳定"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化输出层
        nn.init.xavier_uniform_(self.conv10.weight, gain=0.2)
        nn.init.constant_(self.conv10.bias, -1.5)

    def forward(self, x):
        # 编码器前向传播
        c1 = self.conv1(x)          # (batch, 32, H, W)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)         # (batch, 64, H/2, W/2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)         # (batch, 128, H/4, W/4)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)         # (batch, 256, H/8, W/8)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)         # (batch, 512, H/16, W/16)

        # 解码器前向传播
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1_att = self.att1(c1)  # 对c1特征应用空间注意力
        merge9 = torch.cat([up_9, c1_att], dim=1)
        c9 = self.conv9(merge9)

        # 残差连接（保留细小血管细节）
        c1_residual = self.detail_residual(c1_att)
        c9 = c9 + c1_residual
        c9 = F.relu(c9, inplace=True)

        # 短路径前向传播（索引0-5，无越界）
        branch_feat = self.small_vessel_branch[0](c1_att)  # 0: 多尺度卷积
        branch_feat = self.small_vessel_branch[1](branch_feat)  # 1: 通道注意力
        branch_feat = self.small_vessel_branch[2](branch_feat)  # 2: 卷积
        branch_feat = self.small_vessel_branch[3](branch_feat)  # 3: BN
        branch_feat = self.small_vessel_branch[4](branch_feat)  # 4: ReLU

        # 分支血管注意力（筛选血管区域）
        vessel_att = self.branch_vessel_att(branch_feat)
        vessel_prior = (torch.mean(c1_att, dim=1, keepdim=True) > 0).float()  # 血管先验
        vessel_att = vessel_att * vessel_prior
        branch_feat = branch_feat * vessel_att

        # 短路径输出（索引5，最终输出）
        branch_out = self.small_vessel_branch[5](branch_feat)

        # 主路径+短路径加权融合
        c10 = (1 - self.alpha) * self.conv10(c9) + self.alpha * branch_out

        return c10

# -------------------------- 损失函数（对齐Miou，优化细小血管） --------------------------
def miou_aligned_loss(pred, target):
    batch_size = pred.shape[0]
    h, w = pred.shape[2], pred.shape[3]
    pred_prob = torch.sigmoid(pred)

    # 1. 基础Dice损失
    intersection = (pred_prob * target).sum(dim=(2,3))
    union = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1e-8
    base_dice = 1 - (2 * intersection) / union
    base_dice = base_dice.mean() * 0.8 + 1e-6

    # 2. 细小血管Dice损失
    target_np = target.cpu().numpy().astype(np.uint8)
    small_vessel_mask = []
    for i in range(batch_size):
        target_single = np.squeeze(target_np[i])
        eroded = cv2.erode(target_single, np.ones((3,3), np.uint8), iterations=1)
        small_vessel = (target_single - eroded) > 0
        small_vessel = np.expand_dims(np.expand_dims(small_vessel, 0), 0)
        small_vessel_mask.append(small_vessel)
    small_vessel_mask = torch.tensor(np.concatenate(small_vessel_mask, 0), 
                                     dtype=torch.float32, device=pred.device)

    small_dice = 0.0
    if small_vessel_mask.sum() > 0:
        small_pred = pred_prob * small_vessel_mask
        small_target = target * small_vessel_mask
        small_inter = (small_pred * small_target).sum(dim=(2,3))
        small_union = small_pred.sum(dim=(2,3)) + small_target.sum(dim=(2,3)) + 1e-8
        small_dice = 1 - (2 * small_inter) / small_union
        small_dice = small_dice.mean() * 3.0 + 1e-6

    # 3. 边界损失
    boundary_mask = []
    for i in range(batch_size):
        target_single = np.squeeze(target_np[i])
        grad = cv2.morphologyEx(target_single, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
        boundary = (grad > 0).astype(np.float32)
        boundary = np.expand_dims(np.expand_dims(boundary, 0), 0)
        boundary_mask.append(boundary)
    boundary_mask = torch.tensor(np.concatenate(boundary_mask, 0), 
                                 dtype=torch.float32, device=pred.device)

    boundary_loss = 0.0
    if boundary_mask.sum() > 0:
        boundary_pred = pred * boundary_mask
        boundary_target = target * boundary_mask
        boundary_loss = torch.nn.BCEWithLogitsLoss()(boundary_pred, boundary_target) * 0.5 + 1e-6

    # 4. Miou损失（直接对齐评价指标）
    miou = intersection / (union - intersection + 1e-8)
    miou_loss = 1 - miou.mean() * 1.2 + 1e-6

    # 总损失
    total_loss = small_dice + miou_loss + base_dice + boundary_loss
    return total_loss

# -------------------------- 训练相关配置示例（可选） --------------------------
if __name__ == "__main__":
    # 模型初始化
    model = Unet(in_ch=3, out_ch=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器和调度器（无报错）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=3e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=1,  # 整数≥1，符合要求
        eta_min=1e-5,
        verbose=False
    )

    # 测试前向传播（无报错）
    dummy_input = torch.randn(2, 3, 576, 576).to(device)  # batch_size=2，输入尺寸576×576
    dummy_target = torch.randn(2, 1, 576, 576).to(device)
    output = model(dummy_input)
    loss = miou_aligned_loss(output, dummy_target)