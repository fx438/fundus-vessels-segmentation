# quick_test.py
import os
import argparse
import torch
import numpy as np
import cv2
import torch.utils.data as data
from torchvision import transforms
from UNet import Unet
from dataset import DRIVEDataset_Paper
from metrics import get_iou, get_dice, get_f1

def main(model_path):
    # 1. 基础配置
    ROOT_DIR = '/root/autodl-tmp/UNET-ZOO-master'
    PRED_DIR = os.path.join(ROOT_DIR, 'predictions')
    os.makedirs(PRED_DIR, exist_ok=True)  # 确保预测结果目录存在
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== 测试配置 ===')
    print(f'设备：{device}')
    print(f'模型路径：{model_path}')
    print(f'预测结果保存目录：{PRED_DIR}')

    # 2. 加载测试集（和训练时保持一致的预处理）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    target_transform = transforms.ToTensor()

    test_dataset = DRIVEDataset_Paper(
        state='test',
        transform=transform,
        target_transform=target_transform
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    print(f'测试集加载完成：共 {len(test_dataset)} 张图像')

    # 3. 加载模型（确保和训练时的模型结构一致）
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'模型文件不存在：{model_path}')
    
    model = Unet(n_channels=3).to(device)  # 无n_classes参数，输出1通道
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式（禁用Dropout/BatchNorm更新）
    print('模型加载成功！开始测试...')

    # 4. 执行测试（计算指标+保存预测结果）
    total_iou = 0.0
    total_dice = 0.0
    total_f1 = 0.0
    sample_count = len(test_loader)

    with torch.no_grad():  # 关闭梯度计算，加速测试
        for idx, (images, masks, img_paths, mask_paths) in enumerate(test_loader):
            # 数据送设备
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播预测
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()  # 概率图（0~1）
            pred_binary = (pred_prob >= 0.5).astype(np.uint8) * 255  # 二值化（0=背景，255=血管）

            # 计算核心指标
            iou = get_iou(mask_paths[0], pred_prob)
            dice = get_dice(mask_paths[0], pred_prob)
            f1 = get_f1(mask_paths[0], pred_prob)

            # 累计指标
            total_iou += iou
            total_dice += dice
            total_f1 += f1

            # 保存预测结果（按原图名称命名，便于对比）
            img_basename = os.path.basename(img_paths[0]).replace('.tif', '')
            pred_save_path = os.path.join(PRED_DIR, f'{img_basename}_pred.png')
            cv2.imwrite(pred_save_path, pred_binary)

            # 实时输出进度
            print(f'[{idx+1:2d}/{sample_count}] 保存预测图：{os.path.basename(pred_save_path)} | IoU: {iou:.4f} | Dice: {dice:.4f} | F1: {f1:.4f}')

    # 5. 输出整体测试结果
    avg_iou = total_iou / sample_count
    avg_dice = total_dice / sample_count
    avg_f1 = total_f1 / sample_count

    print('\n' + '='*50)
    print('测试完成！整体指标：')
    print(f'平均IoU：{avg_iou:.4f}')
    print(f'平均Dice：{avg_dice:.4f}')
    print(f'平均F1：{avg_f1:.4f}')
    print('='*50)

if __name__ == '__main__':
    # 解析命令行参数（指定模型路径）
    parser = argparse.ArgumentParser(description='快速测试训练中断的模型')
    parser.add_argument('--model_path', type=str, required=True, help='已保存的模型路径（如 saved_model/unet_paper_best.pth）')
    args = parser.parse_args()
    
    # 运行测试
    main(args.model_path)