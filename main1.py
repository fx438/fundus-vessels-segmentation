import argparse
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from PIL import Image
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
# 模型导入（与原代码一致）
from UNet import Unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s

# 数据集导入（与原代码一致）
from dataset import DriveEyeDataset

# 指标和工具函数（新增 get_precision、get_recall、get_f1）
from metrics import get_iou, get_hd, get_dice, get_precision, get_recall, get_f1
from plot import loss_plot, metrics_plot

plt.rcParams["figure.max_open_warning"] = 0  # 关闭多图警告


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0, type=int)  # 明确类型为int
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=200)  # 按你的配置改为200轮
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn8s/cenet')
    parse.add_argument("--batch_size", type=int, default=2)  # 按你的配置改为2
    parse.add_argument('--dataset', default='drive',
                       help='dataset name: liver/isbiCell/drive')
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    parse.add_argument("--resume", type=str, default=None, help="path to resume training weight file")
    args = parse.parse_args()
    return args


def getLog(args):
    dirname = os.path.join(args.log_dir, args.arch, str(args.batch_size), str(args.dataset), str(args.epoch))
    os.makedirs(dirname, exist_ok=True)  # 简化目录创建
    filename = os.path.join(dirname, 'log.log')
    # 配置日志（同时输出到控制台和文件）
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,  # 降低冗余
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    return logging


def getModel(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    elif args.arch == 'unet++':
        args.deepsupervision = False  # 关键：关闭深度监督（通过args控制，不传递给模型初始化）
        model = NestedUNet(args, 3, 1).to(device)  # 移除 deep_supervision 参数，适配你的unetpp.py
    elif args.arch == 'Attention_UNet':
        model = AttU_Net(3, 1).to(device)
    elif args.arch == 'segnet':
        model = SegNet(3, 1).to(device)
    elif args.arch == 'r2unet':
        model = R2U_Net(3, 1).to(device)
    elif args.arch == 'myChannelUnet':
        model = myChannelUnet(3, 1).to(device)
    elif args.arch == 'fcn8s':
        assert args.dataset != 'esophagus', "fcn8s模型不能用于数据集esophagus"
        model = get_fcn8s(1).to(device)
    elif args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    else:
        raise ValueError(f"不支持的模型架构：{args.arch}")
    logging.info(f"成功加载模型：{args.arch}，深度监督状态：{args.deepsupervision}")
    return model


def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    # 全局transform（与dataset.py兼容）
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()

    if args.dataset == 'drive':
        train_dataset = DriveEyeDataset('train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_dataset = DriveEyeDataset('val', transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        test_dataset = DriveEyeDataset('test', transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        raise ValueError(f"不支持的数据集：{args.dataset}")
    
    # 日志输出数据集信息
    logging.info(f"数据集：{args.dataset}")
    logging.info(f"训练集样本数：{len(train_dataset) if train_dataset else 0}")
    logging.info(f"验证集样本数：{len(val_dataset) if val_dataset else 0}")
    logging.info(f"测试集样本数：{len(test_dataset) if test_dataset else 0}")
    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders, args):
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 局部设备定义（避免全局依赖）
    with torch.no_grad():
        miou_total = 0
        hd_total = 0
        dice_total = 0
        precision_total = 0  # 新增：Precision累加
        recall_total = 0     # 新增：Recall累加
        f1_total = 0         # 新增：F1累加
        num = len(val_dataloaders)
        valid_num = 0        # 新增：有效样本数（避免计算失败影响均值）
        if num == 0:
            logging.warning("验证集为空，跳过验证")
            return best_iou, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for x, _, pic_path, mask_path in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()

            # 计算指标（确保mask路径正确，新增三个指标）
            try:
                iou = get_iou(mask_path[0], img_y)
                hd = get_hd(mask_path[0], img_y)
                dice = get_dice(mask_path[0], img_y)
                precision = get_precision(mask_path[0], img_y)  # 新增
                recall = get_recall(mask_path[0], img_y)        # 新增
                f1 = get_f1(mask_path[0], img_y)                # 新增

                # 累加指标（只累加计算成功的样本）
                miou_total += iou
                hd_total += hd
                dice_total += dice
                precision_total += precision
                recall_total += recall
                f1_total += f1
                valid_num += 1
            except Exception as e:
                logging.error(f"计算指标失败，路径：{mask_path[0]}，错误：{e}")
                continue
        
        # 计算平均指标（用有效样本数计算）
        aver_iou = miou_total / valid_num if valid_num > 0 else 0.0
        aver_hd = hd_total / valid_num if valid_num > 0 else 0.0
        aver_dice = dice_total / valid_num if valid_num > 0 else 0.0
        aver_precision = precision_total / valid_num if valid_num > 0 else 0.0  # 新增
        aver_recall = recall_total / valid_num if valid_num > 0 else 0.0        # 新增
        aver_f1 = f1_total / valid_num if valid_num > 0 else 0.0                # 新增

        # 打印验证集指标（包含新指标）
        print(f'\n验证集指标：')
        print(f'Miou={aver_iou:.4f}, aver_hd={aver_hd:.4f}, aver_dice={aver_dice:.4f}')
        print(f'aver_precision={aver_precision:.4f}, aver_recall={aver_recall:.4f}, aver_f1={aver_f1:.4f}')
        logging.info(f'\n验证集指标：')
        logging.info(f'Miou={aver_iou:.4f}, aver_hd={aver_hd:.4f}, aver_dice={aver_dice:.4f}')
        logging.info(f'aver_precision={aver_precision:.4f}, aver_recall={aver_recall:.4f}, aver_f1={aver_f1:.4f}')
    
        # 保存最佳模型（仍以IoU为基准，不改变原有逻辑）
        if aver_iou > best_iou:
            print(f'当前aver_iou:{aver_iou:.4f} > 最佳best_iou:{best_iou:.4f}，保存模型！')
            best_iou = aver_iou
            os.makedirs('./saved_model', exist_ok=True)
            # 修正模型文件名（添加 _best 后缀，与test函数加载逻辑一致）
            model_path = os.path.join('./saved_model', 
                                     f'{args.arch}_{args.batch_size}_{args.dataset}_{args.epoch}_best.pth')
            torch.save(model.state_dict(), model_path)
        
        # 返回值新增3个平均指标（保持原有返回顺序，新增在最后）
        return best_iou, aver_iou, aver_dice, aver_hd, aver_precision, aver_recall, aver_f1


def train(model, criterion, optimizer, train_dataloader, val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = 0.0, 0.0, 0.0, 0.0
    # 新增：新指标的记录列表（仅用于打印，不传入plot.py）
    aver_precision_list, aver_recall_list, aver_f1_list = [], [], []
    num_epochs = args.epoch
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    # 学习率调度器（基于验证集IoU）
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,  # 初始周期30轮
        T_mult=1,  # 每次重启后周期变为1倍
        eta_min=1e-5,  # 最低 lr=1e-5
        verbose=True
    )

    for epoch in range(num_epochs):
        model = model.train()
        print(f'\nEpoch {epoch}/{num_epochs-1}')
        logging.info(f'\nEpoch {epoch}/{num_epochs-1}')
        print('-' * 20)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0.0
        step = 0

        for x, y, _, _ in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()

            # 前向传播（适配深度监督关闭后的单输出）
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output, labels)

            # 打印血管占比（辅助监控）
            if step <= 2 and epoch % 5 == 0:
                if args.deepsupervision:
                    pred_prob = torch.sigmoid(outputs[-1])
                else:
                    pred_prob = torch.sigmoid(output)
                pred_vessel_ratio = (pred_prob > 0.5).sum().item() / pred_prob.numel()
                print(f"Epoch {epoch} Step {step}：预测血管占比={pred_vessel_ratio:.4f}")

            # 反向传播
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 打印训练进度
            progress = f"{step}/{(dt_size - 1) // train_dataloader.batch_size + 1}"
            print(f"进度：{progress}，train_loss:{loss.item():.4f}")
            logging.info(f"进度：{progress}，train_loss:{loss.item():.4f}")

        # 计算平均损失
        avg_epoch_loss = epoch_loss / step
        loss_list.append(avg_epoch_loss)
        print(f"Epoch {epoch} 平均损失：{avg_epoch_loss:.4f}")
        logging.info(f"Epoch {epoch} 平均损失：{avg_epoch_loss:.4f}")

        # 验证 + 学习率调整（接收新增的3个指标）
        best_iou, aver_iou, aver_dice, aver_hd, aver_precision, aver_recall, aver_f1 = val(model, best_iou, val_dataloader, args)
        scheduler.step()  # 基于验证IoU调整学习率

        # 记录指标（原有指标列表不变，新指标单独记录）
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        aver_precision_list.append(aver_precision)  # 新增
        aver_recall_list.append(aver_recall)        # 新增
        aver_f1_list.append(aver_f1)                # 新增

        # 打印新指标记录（不传入plot.py，仅日志输出）
        print(f"Epoch {epoch} 验证集新指标：")
        print(f"Precision={aver_precision:.4f}, Recall={aver_recall:.4f}, F1={aver_f1:.4f}")
        logging.info(f"Epoch {epoch} 验证集新指标：")
        logging.info(f"Precision={aver_precision:.4f}, Recall={aver_recall:.4f}, F1={aver_f1:.4f}")

    # 绘制图表（保持原样，只传入原有指标，不改动plot.py）
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice', iou_list, dice_list)
    metrics_plot(args, 'hd', hd_list)
    
    # 额外打印新指标的训练过程均值（总结用）
    avg_train_precision = np.mean(aver_precision_list)
    avg_train_recall = np.mean(aver_recall_list)
    avg_train_f1 = np.mean(aver_f1_list)
    logging.info(f"\n训练过程验证集新指标均值：")
    logging.info(f"平均Precision={avg_train_precision:.4f}, 平均Recall={avg_train_recall:.4f}, 平均F1={avg_train_f1:.4f}")
    logging.info(f"训练结束，最佳Miou：{best_iou:.4f}")
    return model


def test(val_dataloaders, args, save_predict=False):
    logging.info('开始测试...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 局部设备定义
    # 创建保存目录
    if save_predict:
        save_dir = os.path.join('./saved_predict', args.arch, str(args.batch_size), str(args.epoch), args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"预测结果保存目录：{save_dir}")
    
    # 加载最佳模型
    model_path = os.path.join('./saved_model', f'{args.arch}_{args.batch_size}_{args.dataset}_{args.epoch}_best.pth')
    if not os.path.exists(model_path):
        print(f"警告：未找到最佳模型文件 {model_path}，测试跳过")
        logging.error(f"未找到最佳模型文件 {model_path}")
        return
    
    # 加载模型
    model = getModel(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f"成功加载模型：{model_path}")
    
    with torch.no_grad():
        miou_total = 0
        hd_total = 0
        dice_total = 0
        precision_total = 0  # 新增：Precision累加
        recall_total = 0     # 新增：Recall累加
        f1_total = 0         # 新增：F1累加
        num = len(val_dataloaders)
        valid_num = 0        # 新增：有效样本数
        if num == 0:
            print("警告：测试集为空")
            logging.warning("测试集为空")
            return
        
        for idx, (pic, _, pic_path, mask_path) in enumerate(val_dataloaders):
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()

            # 归一化 + 二值化
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            predict_binary = (predict >= 0.5).astype(np.uint8) * 255

            # 计算指标（新增三个指标）
            try:
                iou = get_iou(mask_path[0], predict)
                dice = get_dice(mask_path[0], predict)
                hd = get_hd(mask_path[0], predict)
                precision = get_precision(mask_path[0], predict)  # 新增
                recall = get_recall(mask_path[0], predict)        # 新增
                f1 = get_f1(mask_path[0], predict)                # 新增

                # 累加指标
                miou_total += iou
                hd_total += hd
                dice_total += dice
                precision_total += precision
                recall_total += recall
                f1_total += f1
                valid_num += 1
            except Exception as e:
                print(f"样本 {idx} 计算指标失败：{e}")
                logging.error(f"样本 {idx} 计算指标失败：{e}")
                continue

            # 绘制三图对比（保持原样）
            fig = plt.figure(figsize=(15, 5))
            # 原图
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('Input ', fontsize=12)
            try:
                img_input = cv2.imread(pic_path[0])
                img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                img_input = cv2.resize(img_input, (576, 576))
                ax1.imshow(img_input)
            except:
                ax1.set_title('Input', fontsize=12)
                ax1.imshow(np.zeros((576, 576, 3), dtype=np.uint8))
            ax1.axis('off')
            # 预测图
            ax2 = fig.add_subplot(1, 3, 2)
            # 预测图标题添加F1指标（可选，直观展示）
            ax2.set_title(f'Predict (IoU={iou:.4f}, F1={f1:.4f})', fontsize=12)
            ax2.imshow(predict_binary, cmap='gray', vmin=0, vmax=255)
            ax2.axis('off')
            # 真实标注
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('Mask truth', fontsize=12)
            try:
                img_mask = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
                img_mask = cv2.resize(img_mask, (576, 576))
                ax3.imshow(img_mask, cmap='gray', vmin=0, vmax=255)
            except:
                ax3.set_title('Mask', fontsize=12)
                ax3.imshow(np.zeros((576, 576), dtype=np.uint8), cmap='gray')
            ax3.axis('off')
            plt.tight_layout()

            # 保存结果（保持原样）
            if save_predict:
                mask_filename = os.path.basename(mask_path[0])
                compare_path = os.path.join(save_dir, f'{os.path.splitext(mask_filename)[0]}_compare.png')
                plt.savefig(compare_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
                print(f'保存对比图：{compare_path}')
            plt.close()

            # 打印单样本指标（包含新指标）
            print(f'样本 {idx+1}/{num}：{os.path.basename(pic_path[0])} → ')
            print(f'IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            logging.info(f'样本 {idx+1}/{num}：{os.path.basename(pic_path[0])} → ')
            logging.info(f'IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
        
        # 输出整体指标（包含新指标）
        if valid_num == 0:
            print("无有效测试样本，无法计算整体指标")
            logging.warning("无有效测试样本，无法计算整体指标")
            return
        
        aver_iou = miou_total / valid_num
        aver_hd = hd_total / valid_num
        aver_dice = dice_total / valid_num
        aver_precision = precision_total / valid_num  # 新增
        aver_recall = recall_total / valid_num        # 新增
        aver_f1 = f1_total / valid_num                # 新增

        print(f'\n====== 测试集整体指标 ======')
        print(f'平均IoU={aver_iou:.4f}, 平均HD={aver_hd:.4f}, 平均Dice={aver_dice:.4f}')
        print(f'平均Precision={aver_precision:.4f}, 平均Recall={aver_recall:.4f}, 平均F1={aver_f1:.4f}')
        logging.info(f'\n====== 测试集整体指标 ======')
        logging.info(f'平均IoU={aver_iou:.4f}, 平均HD={aver_hd:.4f}, 平均Dice={aver_dice:.4f}')
        logging.info(f'平均Precision={aver_precision:.4f}, 平均Recall={aver_recall:.4f}, 平均F1={aver_f1:.4f}')


def miou_aligned_loss(pred, target):
    batch_size = pred.shape[0]
    h, w = pred.shape[2], pred.shape[3]
    pred_prob = torch.sigmoid(pred)
    
    # 1. 基础 Dice 损失（保留，但权重降低，只保证整体分割）
    intersection = (pred_prob * target).sum(dim=(2,3))
    union = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1e-8
    base_dice = 1 - (2 * intersection) / union
    base_dice = base_dice.mean() * 0.8  # 权重从1.5→0.8，不占主导
    
    # 2. 细小血管 Dice 损失（权重翻倍，强制模型优化关键区域）
    # 提取细小血管掩码（不变）
    target_np = target.cpu().numpy().astype(np.uint8)
    small_vessel_mask = []
    for i in range(batch_size):
        target_single = np.squeeze(target_np[i])
        eroded = cv2.erode(target_single, np.ones((3,3), np.uint8), iterations=1)
        small_vessel = (target_single - eroded) > 0
        small_vessel = np.expand_dims(np.expand_dims(small_vessel, 0), 0)
        small_vessel_mask.append(small_vessel)
    small_vessel_mask = torch.tensor(np.concatenate(small_vessel_mask, 0), 
                                     dtype=torch.float32, device=target.device)
    
    small_dice = 0.0
    if small_vessel_mask.sum() > 0:
        small_pred = pred_prob * small_vessel_mask
        small_target = target * small_vessel_mask
        small_inter = (small_pred * small_target).sum(dim=(2,3))
        small_union = small_pred.sum(dim=(2,3)) + small_target.sum(dim=(2,3)) + 1e-8
        small_dice = 1 - (2 * small_inter) / small_union
        small_dice = small_dice.mean() * 3.0  # 权重从1.2→3.0，成为核心损失
    
    # 3. 边界损失（权重降低，避免干扰核心目标）
    boundary_mask = []
    for i in range(batch_size):
        target_single = np.squeeze(target_np[i])
        grad = cv2.morphologyEx(target_single, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
        boundary = (grad > 0).astype(np.float32)
        boundary = np.expand_dims(np.expand_dims(boundary, 0), 0)
        boundary_mask.append(boundary)
    boundary_mask = torch.tensor(np.concatenate(boundary_mask, 0), 
                                 dtype=torch.float32, device=target.device)
    
    boundary_loss = 0.0
    if boundary_mask.sum() > 0:
        boundary_pred = pred * boundary_mask
        boundary_target = target * boundary_mask
        boundary_loss = torch.nn.BCEWithLogitsLoss()(boundary_pred, boundary_target) * 0.5  # 从0.8→0.5
    
    # 4. 新增：Miou 损失（直接优化 Miou，彻底对齐指标）
    miou = (intersection) / (union - intersection + 1e-8)  # Miou = TP/(TP+FP+FN)
    miou_loss = 1 - miou.mean() * 1.2  # 直接惩罚低 Miou，权重1.2
    
    base_dice = base_dice + 1e-6
    small_dice = small_dice + 1e-6
    boundary_loss = boundary_loss + 1e-6
    miou_loss = miou_loss + 1e-6
    # 总损失：核心（细小血管+Miou）+ 辅助（基础Dice+边界）
    total_loss = small_dice + miou_loss + base_dice + boundary_loss
    return total_loss


if __name__ == "__main__":
    # 全局transform定义
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 解析参数 + 初始化日志
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print(f'模型：{args.arch},\n训练轮数：{args.epoch},\n批量大小：{args.batch_size},\n数据集：{args.dataset}')
    logging.info(f'\n======= 训练配置 =======')
    logging.info(f'模型：{args.arch},\n训练轮数：{args.epoch},\n批量大小：{args.batch_size},\n数据集：{args.dataset}')
    print('**************************')

    # 加载模型
    model = getModel(args)
    # 加载断点（如果有）
    if args.resume is not None and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"成功加载断点权重：{args.resume}")
        logging.info(f"成功加载断点权重：{args.resume}")
    elif args.resume is not None:
        print(f"警告：未找到权重文件 {args.resume}，将从头训练")
        logging.warning(f"未找到权重文件 {args.resume}，将从头训练")

    # 加载数据集
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)

    # 定义损失函数和优化器
    criterion = miou_aligned_loss
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=2e-5)  # 优化学习率和权重衰减
    # 训练 + 测试
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders, val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, args, save_predict=True)