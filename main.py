import os
import cv2
import torch
import numpy as np
import logging
import torch.utils.data as data
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 导入模型和数据集（确保路径正确）
from UNetdice08 import Unet
from dataset import DRIVEDataset_Paper
# 导入指标函数（复用之前的metrics.py，含IoU/Dice/F1）
from metrics import get_acc, get_auc, get_iou, get_sp, get_precision, get_recall, get_f1, get_sen, get_sp
import torch.nn.functional as F
# -------------------------- 1. 全局配置（与你的目录完全对齐） --------------------------
ROOT_DIR = '/root/autodl-tmp/UNET-ZOO-master'
# 日志/模型/预测结果保存路径
LOG_DIR = os.path.join(ROOT_DIR, 'result', 'log')
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_model')
PRED_DIR = os.path.join(ROOT_DIR, 'predictions')
# 创建目录（避免路径不存在报错）
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# -------------------------- 2. 日志配置（同时输出到控制台和文件） --------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'unet_paper_train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# -------------------------- 3. 设备配置（自动适配CPU/GPU） --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'使用设备：{device}')

# -------------------------- 4. 数据集加载（严格按论文参数） --------------------------
def load_dataset():
    # 数据预处理：ToTensor+归一化（与论文预处理衔接）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3通道输入适配
    ])
    target_transform = transforms.ToTensor()  # 标签转Tensor（单通道）

    # 加载训练集（论文：10000个48×48样本块）
    train_dataset = DRIVEDataset_Paper(
        state='train',
        patch_size=48,
        num_train_patches=10000,
        transform=transform,
        target_transform=target_transform
    )
    # 加载验证集（8:2拆分，完整图像）
    val_dataset = DRIVEDataset_Paper(
        state='val',
        transform=transform,
        target_transform=target_transform
    )
    # 加载测试集（完整图像，论文评估用）
    test_dataset = DRIVEDataset_Paper(
        state='test',
        transform=transform,
        target_transform=target_transform
    )

    # DataLoader（批量大小32，符合论文训练效率）
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # 加速GPU数据传输
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # 日志输出数据集信息
    logging.info(f'\n数据集加载完成：')
    logging.info(f'训练集：{len(train_dataset)}个样本块（48×48）')
    logging.info(f'验证集：{len(val_dataset)}张完整图像')
    logging.info(f'测试集：{len(test_dataset)}张完整图像')
    return train_loader, val_loader, test_loader


# 训练时，每个epoch后加这段代码（打印MMC关键数值）
def simple_check_mmc(model):
    # 1. 获取MMC的动态偏移（offset）和最终权重（morph_weight）
    mmc = model.simplified_mmc
    offset = mmc.offset.detach().cpu()  # 可学习偏移：(256,1,3,3)
    morph_weight = (mmc.morph_conv.weight + mmc.offset).detach().cpu()  # 最终权重：(256,1,3,3)
    
    # 2. 打印2个核心数值
    print("\n【MMC简单数值验证】")
    print(f"1. 偏移幅度均值：{offset.abs().mean():.6f}")  # 偏移是否“动起来”
    print(f"2. 权重分布标准差：{morph_weight.std():.6f}")  # 通道间权重是否有差异
    print("-"*40)


# -------------------------- 5. 损失函数（论文：BCE+Focal混合损失） --------------------------
# def paper_hybrid_loss(pred, target):
#     """
#     论文3.2节混合损失：L_FC = α*L_F + (1-α)*L_CE
#     α=0.7（论文实验最优值，表1验证）
#     """
#     alpha = 0.4
#     # 1. 交叉熵损失（L_CE）：稳定梯度
#     ce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
    
#     # 2. Focal损失（L_F）：缓解类不平衡（论文公式8）
#     pred_sigmoid = torch.sigmoid(pred)
#     focal_loss = - (1 - pred_sigmoid) ** 2 * target * torch.log(pred_sigmoid + 1e-8) - \
#                  pred_sigmoid ** 2 * (1 - target) * torch.log(1 - pred_sigmoid + 1e-8)
#     focal_loss = focal_loss.mean()
    
#     intersection = (pred_sigmoid * target).sum()  # 交集（预测正确的血管像素）
#     union = pred_sigmoid.sum() + target.sum()     # 并集（所有预测血管+真实血管）
#     dice_loss = 1 - (2 * intersection) / (union + 1e-8)  # 1-Dice系数（损失越小，重叠越好）
    
#     # 4. 混合损失加权求和（平衡三类损失）
#     total_loss = alpha * focal_loss + + 0.2* ce_loss+0.4 * dice_loss
#     return total_loss


# def paper_hybrid_loss(pred, target):
#     # 1. 固定权重和，避免尺度波动（核心改1）
#     alpha_focal = 0.4  # 替代动态alpha，避免极端值
#     alpha_ce = 0.2
#     alpha_dice = 0.35  # 原0.4→0.35，为一致性损失腾出0.05权重
#     alpha_consistency = 0.05  # 新增：血管区域一致性损失权重
#     gamma = 1.5  # 新增Focal的gamma，强化难例加权
    
#     # BCE Loss（保持不变）
#     ce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
#     pred_sigmoid = torch.sigmoid(pred)
    
#     # 修正Focal Loss计算（核心改2：标准Focal公式，适配难例）
#     # p_t：对正样本（血管）是pred_sigmoid，负样本是1-pred_sigmoid
#     p_t = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
#     # 难例加权：(1 - p_t)^gamma，越难分的样本权重越高
#     focal_weight = (1 - p_t) ** gamma
#     # 带alpha的Focal Loss（alpha平衡正负样本，gamma平衡难例）
#     focal_loss = -alpha_focal * focal_weight * torch.log(p_t + 1e-8)
#     focal_loss = focal_loss.mean()
    
#     # Dice Loss（保持不变，可加平滑项避免分母为0）
#     intersection = (pred_sigmoid * target).sum()
#     union = pred_sigmoid.sum() + target.sum()
#     dice_loss = 1 - (2 * intersection + 1e-8) / (union + 2e-8)  # 加平滑
    
#     # -------------------------- 新增：血管区域一致性损失（核心） --------------------------
#     def vessel_region_consistency_loss(pred_s, mask, temp=0.1):
#         """
#         pred_s: 经过sigmoid的预测图 (B,1,H,W)
#         mask: 金标准掩码 (B,1,H,W)
#         temp: 温度系数，越小越强制血管区域平滑
#         """
#         # 1. 仅聚焦标注为血管的区域（mask=1）
#         vessel_mask = (mask > 0.5).float()
#         if vessel_mask.sum() == 0:  # 无血管的样本跳过
#             return torch.tensor(0.0, device=pred_s.device)
        
#         # 2. 对血管区域内的预测值做3×3局部平均（模拟“内部一致”先验）
#         pred_smoothed = F.avg_pool2d(pred_s, kernel_size=3, stride=1, padding=1)
#         pred_smoothed = pred_smoothed * vessel_mask  # 仅保留血管区域的平滑结果
        
#         # 3. 惩罚原始预测与平滑预测的差异（孤立背景像素差异最大）
#         consistency_loss = F.mse_loss(pred_s * vessel_mask, pred_smoothed) / temp
#         return consistency_loss
    
#     # 计算一致性损失
#     consistency_loss = vessel_region_consistency_loss(pred_sigmoid, target)
    
#     # 总Loss：权重和=0.4+0.2+0.35+0.05=1，保持尺度稳定
#     total_loss = (alpha_focal * focal_loss + 
#                   alpha_ce * ce_loss + 
#                   alpha_dice * dice_loss + 
#                   alpha_consistency * consistency_loss)
#     return total_loss
import torch


def paper_hybrid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    改进的混合损失函数（Focal + BCE + Dice），固定权重和为1保证尺度稳定
    
    Args:
        pred: 模型预测输出（未经过sigmoid），shape: [B, C, H, W] 或 [B, H, W]
        target: 标签值，shape与pred一致，值为0或1
        
    Returns:
        total_loss: 加权后的总损失值
    """
    # 1. 固定权重和，避免尺度波动（核心改1）
    alpha_focal = 0.4  # 替代动态alpha，避免极端值
    alpha_ce = 0.2
    alpha_dice = 0.4
    gamma = 1.5  # 新增Focal的gamma，强化难例加权

    # BCE Loss（保持不变）
    ce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
    
    pred_sigmoid = torch.sigmoid(pred)
    
    # 修正Focal Loss计算（核心改2：标准Focal公式，适配难例）
    # p_t：对正样本（血管）是pred_sigmoid，负样本是1-pred_sigmoid
    p_t = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    # 难例加权：(1 - p_t)^gamma，越难分的样本权重越高
    focal_weight = (1 - p_t) ** gamma
    # 带alpha的Focal Loss（alpha平衡正负样本，gamma平衡难例）
    focal_loss = -alpha_focal * focal_weight * torch.log(p_t + 1e-8)
    focal_loss = focal_loss.mean()
    
    # Dice Loss（保持不变，可加平滑项避免分母为0）
    intersection = (pred_sigmoid * target).sum()
    union = pred_sigmoid.sum() + target.sum()
    dice_loss = 1 - (2 * intersection + 1e-8) / (union + 2e-8)  # 加平滑
    
    # 总Loss：权重和=1，尺度稳定（核心改3）
    total_loss = alpha_focal * focal_loss + alpha_ce * ce_loss + alpha_dice * dice_loss
    
    return total_loss



# -------------------------- 6. 训练函数（含验证，论文100轮训练） --------------------------
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100,grad_clip=1.0):
    best_val_f1 = 0.0  # 以F1为最优模型评判标准（论文核心指标）
    train_loss_list = []
    val_metrics_list = []  # 存储验证集指标：[IoU, Dice, F1]

    logging.info(f'\n开始训练（共{epochs}轮）：')
    for epoch in range(epochs):
        # -------------------------- 训练阶段 --------------------------
        model.train()
        train_total_loss = 0.0
        train_sample_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            # 数据送设备
            images, masks = images.to(device), masks.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, masks)
            # 反向传播+参数更新
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if batch_idx % 10 == 0:  # 每10个batch打印一次，避免日志冗余
                print(f'\nEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
                # 遍历模型参数，筛选eca_512_final的参数（重点关注gate层的卷积核）
                # for param_name, param in model.named_parameters():
                #     if 'deep_channel' in param_name and 'weight' in param_name:  # 只打印权重梯度（偏置梯度可选）
                #         if param.grad is not None:
                #             grad_mean = param.grad.abs().mean().item()  # 梯度绝对值的平均值（更易读）
                #             grad_max = param.grad.abs().max().item()    # 梯度最大值（可选，看极端值）
                #             print(f'[512梯度] {param_name}: 平均梯度={grad_mean:.6f}, 最大梯度={grad_max:.6f}')
                #         else:
                #             print(f'[512MECA梯度] {param_name}: 梯度为None（未更新）')
            optimizer.step()

            # 累计损失
            train_total_loss += loss.item() * images.size(0)
            train_sample_count += images.size(0)

            # 日志输出训练进度（每10个batch打印一次）
            if (batch_idx + 1) % 10 == 0:
                avg_batch_loss = loss.item()
                logging.info(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {avg_batch_loss:.4f}')

        # 计算本轮平均训练损失
        avg_train_loss = train_total_loss / train_sample_count if train_sample_count > 0 else 0.0
        train_loss_list.append(avg_train_loss)
        logging.info(f'\nEpoch {epoch+1}/{epochs} | 平均训练损失：{avg_train_loss:.4f}')

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        val_iou_total = 0.0
        val_sp_total = 0.0
        val_f1_total = 0.0
        val_sample_count = 0

        with torch.no_grad():  # 关闭梯度计算
            for images, masks, img_paths, mask_paths in val_loader:
                # 数据送设备
                images, masks = images.to(device), masks.to(device)
                # 前向传播
                outputs = model(images)
                # 预测结果处理（sigmoid+二值化）
                pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
                pred_binary = (pred_prob >= 0.5).astype(np.int16)
                # 真实标签处理
                mask_gt = masks.cpu().numpy().squeeze().astype(np.int16)

                # 计算验证集指标（复用metrics.py函数）
                iou = get_iou(mask_paths[0], pred_prob)  # 传入概率图计算更准确
                sp = get_sp(mask_paths[0], pred_prob)
                f1 = get_f1(mask_paths[0], pred_prob)

                # 累计指标
                val_iou_total += iou
                val_sp_total += sp
                val_f1_total += f1
                val_sample_count += 1

        # 计算本轮平均验证指标
        if val_sample_count > 0:
            avg_val_iou = val_iou_total / val_sample_count
            avg_val_sp = val_sp_total / val_sample_count
            avg_val_f1 = val_f1_total / val_sample_count
            val_metrics_list.append([avg_val_iou, avg_val_sp, avg_val_f1])
            # 日志输出验证指标
            logging.info(f'Epoch {epoch+1}/{epochs} | 验证集指标：')
            logging.info(f'平均IoU：{avg_val_iou:.4f} | 平均sp：{avg_val_sp:.4f} | 平均F1：{avg_val_f1:.4f}')

            # 保存最优模型（以F1为基准，论文核心评估指标）
            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                model_save_path = os.path.join(MODEL_DIR, 'unet_paper_best.pth')
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'====== 保存最优模型（F1：{best_val_f1:.4f}）到 {model_save_path} ======')
        else:
            logging.warning(f'Epoch {epoch+1}/{epochs} | 验证集无有效样本')

        # 学习率调度（基于训练损失，论文余弦退火/ReduceLROnPlateau均可）
        scheduler.step(avg_train_loss)

    # 训练结束日志
    logging.info(f'\n训练完成！最佳验证集F1：{best_val_f1:.4f}')
    return model, train_loss_list, val_metrics_list

# -------------------------- 7. 测试函数（论文评估逻辑，保存预测结果） --------------------------
def test(model, test_loader, model_path):
    # 加载最优模型
    if not os.path.exists(model_path):
        logging.error(f'未找到模型文件：{model_path}')
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f'\n加载模型：{model_path}，开始测试')

    # 测试集指标累计
    test_acc_total=0.0
    test_sen_total=0.0
    test_iou_total = 0.0
    test_sp_total = 0.0
    test_f1_total = 0.0
    test_auc_total = 0.0
    test_sample_count = 0

    with torch.no_grad():
        for idx, (images, masks, img_paths, mask_paths) in enumerate(test_loader):
            # 数据送设备
            images, masks = images.to(device), masks.to(device)
            # 前向传播
            outputs = model(images)
            # 预测结果处理
            pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
            pred_binary = (pred_prob >= 0.5).astype(np.uint8) * 255  # 转255便于保存

            # 真实标签处理（转为255灰度图，便于可视化）
            mask_gt = masks.cpu().numpy().squeeze().astype(np.uint8)  # 原是int16，转uint8
            mask_gt = mask_gt * 255  # 标签原值0=背景、1=血管，转255后血管更清晰

            # -------------------------- 新增：原图恢复与处理 --------------------------
            # 原数据预处理：Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])，反归一化恢复原图
            img_origin = images.cpu().numpy().squeeze()  # 形状：(3, H, W)（C, H, W）
            img_origin = (img_origin * 0.5 + 0.5) * 255  # 反归一化：从(-1~1)→(0~255)
            img_origin = img_origin.transpose(1, 2, 0).astype(np.uint8)  # 转为(H, W, 3)的RGB图

            # -------------------------- 核心：生成三图对比图 --------------------------
            # 1. 统一所有图像尺寸（避免因模型padding导致尺寸不一致）
            H, W = img_origin.shape[:2]  # 以原图尺寸为基准
            mask_gt = cv2.resize(mask_gt, (W, H), interpolation=cv2.INTER_NEAREST)  # 标签 resize
            pred_binary = cv2.resize(pred_binary, (W, H), interpolation=cv2.INTER_NEAREST)  # 预测图 resize

            # 2. 把单通道的标签、预测图转为3通道（与原图一致，便于拼接）
            mask_gt_3ch = cv2.cvtColor(mask_gt, cv2.COLOR_GRAY2BGR)
            pred_binary_3ch = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)

            # 3. 横向拼接三图（顺序：原图 → 真实标签 → 预测结果）
            comparison_img = np.hstack([img_origin, mask_gt_3ch, pred_binary_3ch])

            # 4. 添加文字标注（区分三幅图，字体清晰不遮挡）
            cv2.putText(
                comparison_img, 'Origin',  # 标注文字
                (20, 30),  # 文字位置（左上角）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1.0,  # 字体大小
                (0, 255, 0),  # 文字颜色（绿色）
                2  # 文字粗细
            )
            cv2.putText(
                comparison_img, 'Ground Truth',
                (W + 20, 30),  # 标签图文字位置（原图宽度+偏移）
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),  # 文字颜色（蓝色）
                2
            )
            cv2.putText(
                comparison_img, 'Prediction',
                (2 * W + 20, 30),  # 预测图文字位置（2×原图宽度+偏移）
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 文字颜色（红色）
                2
            )

            # -------------------------- 保存三图对比图（替换原单图保存） --------------------------
            img_name = os.path.basename(img_paths[0]).replace('.tif', '_comparison.png')  # 文件名改为“对比图”
            pred_save_path = os.path.join(PRED_DIR, img_name)
            cv2.imwrite(pred_save_path, comparison_img)

          # 计算测试集指标（新增ACC和SEN）
            iou = get_iou(mask_paths[0], pred_prob)
            sp = get_sp(mask_paths[0], pred_prob)
            f1 = get_f1(mask_paths[0], pred_prob)
            acc = get_acc(mask_paths[0], pred_prob)  # 新增：计算ACC
            sen = get_sen(mask_paths[0], pred_prob)  # 新增：计算SEN
            auc = get_auc(mask_paths[0], pred_prob)
            # 累计指标（新增ACC和SEN的累计）
            test_iou_total += iou
            test_sp_total += sp
            test_f1_total += f1
            test_acc_total += acc  # 新增：累计ACC
            test_sen_total += sen  # 新增：累计SEN
            test_auc_total +=auc
            test_sample_count += 1

            # 日志输出（更新为“保存三图对比”，并添加ACC和SEN）
            logging.info(f'测试样本 {idx+1}/{len(test_loader)} | 保存三图对比：{pred_save_path}')
            logging.info(f'样本指标：IoU={iou:.4f} | sp={sp:.4f} | F1={f1:.4f} | ACC={acc:.4f} | SEN={sen:.4f}')

    # 计算测试集整体指标（原有逻辑不变）
    if test_sample_count > 0:
        avg_test_iou = test_iou_total / test_sample_count
        avg_test_sp = test_sp_total / test_sample_count
        avg_test_f1 = test_f1_total / test_sample_count
        avg_test_acc=test_acc_total/test_sample_count
        avg_test_sen=test_sen_total/test_sample_count
        avg_test_auc=test_auc_total/test_sample_count
        print(f'\n====== 测试集整体指标 ======')
        print(f'平均IoU：{avg_test_iou:.4f}')
        print(f'平均sp：{avg_test_sp:.4f}')
        print(f'平均F1：{avg_test_f1:.4f}')
        print(f'平均sen：{avg_test_sen:.4f}')
        print(f'平均acc：{avg_test_acc:.4f}')
        print(f'平均auc：{avg_test_auc:.4f}')
    else:
        logging.error('测试集无有效样本')

# -------------------------- 8. 主函数（串联训练+测试） --------------------------
if __name__ == '__main__':
    # 1. 加载数据集
    train_loader, val_loader, test_loader = load_dataset()

    # 2. 初始化模型（输出通道1，适配二分类）
    model = Unet(n_channels=3).to(device)  # 确保UNet.py输出通道为1

    # 3. 初始化优化器和学习率调度器（论文Adam优化器）
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,  # 论文初始学习率
        weight_decay=1e-5  # 权重衰减防过拟合
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # 学习率减半
        patience=5,  # 5轮无改善则降学习率
        verbose=True  # 打印学习率变化
    )

    # 4. 初始化损失函数（论文混合损失）
    criterion = paper_hybrid_loss

    #5. 开始训练（论文100轮）
    trained_model, train_loss, val_metrics = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=40,
        grad_clip=1.0
    )

    # 6. 开始测试（加载最优模型）
    best_model_path = os.path.join(MODEL_DIR, 'unet_paper_best.pth')
    test(model, test_loader, best_model_path)