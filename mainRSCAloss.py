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
from UNetnewRSCA import Unet
from dataset import DRIVEDataset_Paper
# 导入指标函数
from metrics import get_acc, get_auc, get_iou, get_sp, get_precision, get_recall, get_f1, get_sen
import torch.nn.functional as F
import torch.nn as nn

# -------------------------- 1. 全局配置 --------------------------
ROOT_DIR = '/root/autodl-tmp/UNET-ZOO-master'
LOG_DIR = os.path.join(ROOT_DIR, 'result', 'log')
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_model')
PRED_DIR = os.path.join(ROOT_DIR, 'predictions')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# -------------------------- 2. 日志配置 --------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'unet_paper_train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# -------------------------- 3. 设备配置 --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'使用设备：{device}')

# -------------------------- 4. 数据集加载 --------------------------
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    target_transform = transforms.ToTensor()

    train_dataset = DRIVEDataset_Paper(
        state='train',
        patch_size=48,
        num_train_patches=10000,
        transform=transform,
        target_transform=target_transform
    )
    val_dataset = DRIVEDataset_Paper(
        state='val',
        transform=transform,
        target_transform=target_transform
    )
    test_dataset = DRIVEDataset_Paper(
        state='test',
        transform=transform,
        target_transform=target_transform
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
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

    logging.info(f'\n数据集加载完成：')
    logging.info(f'训练集：{len(train_dataset)}个样本块（48×48）')
    logging.info(f'验证集：{len(val_dataset)}张完整图像')
    logging.info(f'测试集：{len(test_dataset)}张完整图像')
    return train_loader, val_loader, test_loader

# -------------------------- 恢复RCSA特征获取函数 --------------------------
def get_unet_with_rcsa_output(model, x):
    B, C, H_in, W_in = x.shape

    # 编码器前向（恢复焦点注意力和RCSA调用，获取rcsa_feat）
    x1 = model.inc(x)
    x1_focal = model.focal_attention_64(x1)
    
    x2 = model.down1(x1_focal)
    x2_focal = model.focal_attention_128(x2)
    
    x3 = model.down2(x2_focal)
    x3_focal = model.focal_attention_256(x3)
    x3_enhanced = model.rcsa(x3_focal)  # 获取RCSA特征
    rcsa_feat = x3_enhanced  # 定义rcsa_feat，用于辅助损失
    
    x4 = model.down3(x3_enhanced)
    x3_cross = model.cross_res_x3(x3_enhanced)
    x4 = x4 + x3_cross
    x5 = model.down4(x4)

    # 瓶颈层前向
    x5_enhanced = model.bottleneck_enhance(x5)
    x5_residual = model.bottleneck_res(x5)
    x5_residual = model.bottleneck_bn(x5_residual)
    x5_final = x5_enhanced + x5_residual

    # 解码器前向
    x = model.up1(x5_final, x4)
    x = model.focal_attention_512(x)
    
    x = model.up2(x, x3)
    x = model.up3(x, x2)
    x_up4 = model.up4(x, x1)

    # 最终增强
    x_final = model.final_enhance(x_up4)
    x_final_focal = model.focal_attention_final(x_final)
    logits = model.outc(x_final_focal)

    # 尺寸还原
    if logits.shape[2:] != (H_in, W_in):
        logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)

    return logits, rcsa_feat

# -------------------------- 恢复RCSA辅助损失函数 --------------------------
def rcsa_aux_loss(rcsa_feat, target, model):
    # RCSA特征降维
    rcsa_feat_2d = model.rcsa_aux_conv(rcsa_feat)
    # 尺寸匹配
    rcsa_feat_2d = F.interpolate(
        rcsa_feat_2d,
        size=target.shape[2:],
        mode='bilinear',
        align_corners=False
    )
    
    # 计算Dice损失
    rcsa_prob = torch.sigmoid(rcsa_feat_2d)
    intersection = (rcsa_prob * target).sum()
    union = rcsa_prob.sum() + target.sum()
    dice_loss = 1 - (2 * intersection + 1e-8) / (union + 2e-8)
    
    return dice_loss

# -------------------------- 训练指标简单验证 --------------------------
def simple_check_metrics():
    print("\n【训练指标简单验证】")
    print("1. 训练损失是否持续下降")
    print("2. 验证F1是否持续提升")
    print("3. 验证损失是否出现停滞/上升")
    print("-"*40)

# -------------------------- 混合损失函数 --------------------------
def paper_hybrid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    alpha_focal = 0.4
    alpha_ce = 0.2
    alpha_dice = 0.4
    gamma = 1.5

    # BCE Loss
    ce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
    
    pred_sigmoid = torch.sigmoid(pred)
    
    # Focal Loss
    p_t = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_weight = (1 - p_t) ** gamma
    focal_loss = -alpha_focal * focal_weight * torch.log(p_t + 1e-8)
    focal_loss = focal_loss.mean()
    
    # Dice Loss
    intersection = (pred_sigmoid * target).sum()
    union = pred_sigmoid.sum() + target.sum()
    dice_loss = 1 - (2 * intersection + 1e-8) / (union + 2e-8)
    
    # 总Loss
    total_loss = alpha_focal * focal_loss + alpha_ce * ce_loss + alpha_dice * dice_loss
    
    return total_loss

# -------------------------- 训练函数 --------------------------
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100,grad_clip=1.0):
    best_val_f1 = 0.0
    train_loss_list = []
    val_loss_list = []
    val_metrics_list = []
    aux_loss_weight = 0.3  # 恢复RCSA辅助损失权重

    logging.info(f'\n开始训练（共{epochs}轮）：')
    logging.info(f'RCSA辅助损失权重：{aux_loss_weight}')
    logging.info(f'学习率调度：监控验证损失，patience=5，factor=0.5')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_total_loss = 0.0
        train_main_loss = 0.0
        train_aux_loss = 0.0
        train_sample_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            # 获取主输出和RCSA特征
            outputs, rcsa_feat = get_unet_with_rcsa_output(model, images)
            # 主损失
            main_loss = criterion(outputs, masks)
            # RCSA辅助损失
            aux_loss = rcsa_aux_loss(rcsa_feat, masks, model)
            # 总损失
            total_loss = main_loss + aux_loss_weight * aux_loss
            
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # 累计损失
            train_total_loss += total_loss.item() * images.size(0)
            train_main_loss += main_loss.item() * images.size(0)
            train_aux_loss += aux_loss.item() * images.size(0)
            train_sample_count += images.size(0)

            # 日志输出
            if (batch_idx + 1) % 10 == 0:
                logging.info(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | '
                             f'Total Loss: {total_loss.item():.4f} | Main Loss: {main_loss.item():.4f} | '
                             f'Aux Loss (RCSA): {aux_loss.item():.4f}')

        # 平均训练损失
        avg_train_loss = train_total_loss / train_sample_count if train_sample_count > 0 else 0.0
        avg_main_loss = train_main_loss / train_sample_count if train_sample_count > 0 else 0.0
        avg_aux_loss = train_aux_loss / train_sample_count if train_sample_count > 0 else 0.0
        
        train_loss_list.append(avg_train_loss)
        logging.info(f'\nEpoch {epoch+1}/{epochs} | 平均训练总损失：{avg_train_loss:.4f} | '
                     f'平均主损失：{avg_main_loss:.4f} | 平均RCSA辅助损失：{avg_aux_loss:.4f}')

        simple_check_metrics()

        # 验证阶段
        model.eval()
        val_loss_total = 0.0
        val_iou_total = 0.0
        val_sp_total = 0.0
        val_f1_total = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    images, masks, img_paths, mask_paths = batch_data
                elif len(batch_data) == 2:
                    images, masks = batch_data
                    img_paths = [f'val_img_{val_sample_count}.tif']
                    mask_paths = [f'val_mask_{val_sample_count}.tif']
                else:
                    logging.warning(f'验证集数据格式异常，跳过该批次')
                    continue
                    
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_batch_loss = criterion(outputs, masks)
                val_loss_total += val_batch_loss.item() * images.size(0)

                # 预测结果处理
                pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
                mask_gt = masks.cpu().numpy().squeeze().astype(np.int16)

                # 计算指标
                try:
                    iou = get_iou(mask_paths[0], pred_prob)
                    sp = get_sp(mask_paths[0], pred_prob)
                    f1 = get_f1(mask_paths[0], pred_prob)
                except Exception as e:
                    logging.warning(f'计算验证集指标失败：{e}，跳过该样本')
                    continue

                # 累计指标
                val_iou_total += iou
                val_sp_total += sp
                val_f1_total += f1
                val_sample_count += 1

        # 平均验证指标
        if val_sample_count > 0:
            avg_val_loss = val_loss_total / val_sample_count
            avg_val_iou = val_iou_total / val_sample_count
            avg_val_sp = val_sp_total / val_sample_count
            avg_val_f1 = val_f1_total / val_sample_count

            val_loss_list.append(avg_val_loss)
            val_metrics_list.append([avg_val_iou, avg_val_sp, avg_val_f1])

            logging.info(f'Epoch {epoch+1}/{epochs} | 验证集指标：')
            logging.info(f'平均验证损失：{avg_val_loss:.4f} | 平均IoU：{avg_val_iou:.4f} | 平均sp：{avg_val_sp:.4f} | 平均F1：{avg_val_f1:.4f}')

            # 保存最优模型
            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                model_save_path = os.path.join(MODEL_DIR, 'unet_paper_best.pth')
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'====== 保存最优模型（F1：{best_val_f1:.4f}）到 {model_save_path} ======')
        else:
            avg_val_loss = 0.0
            logging.warning(f'Epoch {epoch+1}/{epochs} | 验证集无有效样本')

        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1}/{epochs} | 当前学习率：{current_lr:.6f}')

    logging.info(f'\n训练完成！最佳验证集F1：{best_val_f1:.4f}')
    return model, train_loss_list, val_loss_list, val_metrics_list

# -------------------------- 测试函数 --------------------------
def test(model, test_loader, model_path):
    if not os.path.exists(model_path):
        logging.error(f'未找到模型文件：{model_path}')
        return
    # 加载模型
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logging.info(f'\n加载模型：{model_path}，开始测试')

    # 指标累计
    test_acc_total=0.0
    test_sen_total=0.0
    test_iou_total = 0.0
    test_sp_total = 0.0
    test_f1_total = 0.0
    test_auc_total = 0.0
    test_sample_count = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if len(batch_data) == 4:
                images, masks, img_paths, mask_paths = batch_data
            elif len(batch_data) == 2:
                images, masks = batch_data
                img_paths = [f'test_img_{idx}.tif']
                mask_paths = [f'test_mask_{idx}.tif']
            else:
                logging.warning(f'测试集数据格式异常，跳过该批次')
                continue
                
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
            pred_binary = (pred_prob >= 0.5).astype(np.uint8) * 255

            # 标签处理
            mask_gt = masks.cpu().numpy().squeeze().astype(np.uint8)
            mask_gt = mask_gt * 255

            # 原图恢复
            img_origin = images.cpu().numpy().squeeze()
            img_origin = (img_origin * 0.5 + 0.5) * 255
            img_origin = img_origin.transpose(1, 2, 0).astype(np.uint8)

            # 三图对比
            H, W = img_origin.shape[:2]
            mask_gt = cv2.resize(mask_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            pred_binary = cv2.resize(pred_binary, (W, H), interpolation=cv2.INTER_NEAREST)

            mask_gt_3ch = cv2.cvtColor(mask_gt, cv2.COLOR_GRAY2BGR)
            pred_binary_3ch = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)
            comparison_img = np.hstack([img_origin, mask_gt_3ch, pred_binary_3ch])

            # 添加标注
            cv2.putText(comparison_img, 'Origin', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(comparison_img, 'Ground Truth', (W+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            cv2.putText(comparison_img, 'Prediction', (2*W+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # 保存
            img_name = os.path.basename(img_paths[0]).replace('.tif', '_comparison.png')
            pred_save_path = os.path.join(PRED_DIR, img_name)
            cv2.imwrite(pred_save_path, comparison_img)

            # 计算指标
            try:
                iou = get_iou(mask_paths[0], pred_prob)
                sp = get_sp(mask_paths[0], pred_prob)
                f1 = get_f1(mask_paths[0], pred_prob)
                acc = get_acc(mask_paths[0], pred_prob)
                sen = get_sen(mask_paths[0], pred_prob)
                auc = get_auc(mask_paths[0], pred_prob)
            except Exception as e:
                logging.warning(f'计算测试集指标失败：{e}，跳过该样本')
                continue

            # 累计
            test_iou_total += iou
            test_sp_total += sp
            test_f1_total += f1
            test_acc_total += acc
            test_sen_total += sen
            test_auc_total += auc
            test_sample_count += 1

            logging.info(f'测试样本 {idx+1}/{len(test_loader)} | 保存三图对比：{pred_save_path}')
            logging.info(f'样本指标：IoU={iou:.4f} | sp={sp:.4f} | F1={f1:.4f} | ACC={acc:.4f} | SEN={sen:.4f} | AUC={auc:.4f}')

    # 整体指标
    if test_sample_count > 0:
        avg_test_iou = test_iou_total / test_sample_count
        avg_test_sp = test_sp_total / test_sample_count
        avg_test_f1 = test_f1_total / test_sample_count
        avg_test_acc=test_acc_total/test_sample_count
        avg_test_sen=test_sen_total/test_sample_count
        avg_test_auc=test_auc_total/test_sample_count
        
        log_info = f'\n====== 测试集整体指标 ======\n' \
                   f'平均IoU：{avg_test_iou:.4f}\n' \
                   f'平均sp：{avg_test_sp:.4f}\n' \
                   f'平均F1：{avg_test_f1:.4f}\n' \
                   f'平均SEN：{avg_test_sen:.4f}\n' \
                   f'平均ACC：{avg_test_acc:.4f}\n' \
                   f'平均AUC：{avg_test_auc:.4f}'
        print(log_info)
        logging.info(log_info)
    else:
        logging.error('测试集无有效样本')

# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    # 加载数据集
    train_loader, val_loader, test_loader = load_dataset()

    # 初始化模型
    model = Unet(n_channels=3).to(device)
    logging.info(f'Unet模型初始化完成，输入通道数：3')

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # 损失函数
    criterion = paper_hybrid_loss

    # 训练（如需训练，取消注释下面代码）
    trained_model, train_loss, val_loss, val_metrics = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=60,
        grad_clip=1.0
    )

    # 测试：修正参数传递顺序（model -> test_loader -> model_path）
    best_model_path = os.path.join(MODEL_DIR, 'unet_paper_best.pth')
    # 关键修正：第一个参数传入模型对象model，不是模型路径
    test(model, test_loader, best_model_path)            