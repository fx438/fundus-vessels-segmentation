import os
import cv2
import torch
import numpy as np
import logging
import torch.utils.data as data
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from LightMUnet import LightMUNet  # 确保路径正确
from dataset import DRIVEDataset_Paper
from metrics import get_iou, get_dice, get_precision, get_recall, get_f1

# 全局配置
ROOT_DIR = '/root/autodl-tmp/UNET-ZOO-master'
LOG_DIR = os.path.join(ROOT_DIR, 'result', 'log')
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_model')
PRED_DIR = os.path.join(ROOT_DIR, 'predictions')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'lightmunet_paper_train.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'使用设备：{device}')

# 数据集加载
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

# 损失函数
def paper_hybrid_loss(pred, target):
    alpha = 0.7
    ce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
    pred_sigmoid = torch.sigmoid(pred)
    focal_loss = - (1 - pred_sigmoid) ** 2 * target * torch.log(pred_sigmoid + 1e-8) - \
                 pred_sigmoid ** 2 * (1 - target) * torch.log(1 - pred_sigmoid + 1e-8)
    focal_loss = focal_loss.mean()
    total_loss = alpha * focal_loss + (1 - alpha) * ce_loss
    return total_loss

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
    best_val_f1 = 0.0
    train_loss_list = []
    val_metrics_list = []

    logging.info(f'\n开始训练（共{epochs}轮）：')
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0
        train_sample_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * images.size(0)
            train_sample_count += images.size(0)

            if (batch_idx + 1) % 10 == 0:
                logging.info(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}')

        avg_train_loss = train_total_loss / train_sample_count if train_sample_count > 0 else 0.0
        train_loss_list.append(avg_train_loss)
        logging.info(f'\nEpoch {epoch+1}/{epochs} | 平均训练损失：{avg_train_loss:.4f}')

        model.eval()
        val_iou_total = 0.0
        val_dice_total = 0.0
        val_f1_total = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for images, masks, img_paths, mask_paths in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()

                iou = get_iou(mask_paths[0], pred_prob)
                dice = get_dice(mask_paths[0], pred_prob)
                f1 = get_f1(mask_paths[0], pred_prob)

                val_iou_total += iou
                val_dice_total += dice
                val_f1_total += f1
                val_sample_count += 1

        if val_sample_count > 0:
            avg_val_iou = val_iou_total / val_sample_count
            avg_val_dice = val_dice_total / val_sample_count
            avg_val_f1 = val_f1_total / val_sample_count
            val_metrics_list.append([avg_val_iou, avg_val_dice, avg_val_f1])
            logging.info(f'Epoch {epoch+1}/{epochs} | 验证集指标：')
            logging.info(f'平均IoU：{avg_val_iou:.4f} | 平均Dice：{avg_val_dice:.4f} | 平均F1：{avg_val_f1:.4f}')

            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                model_save_path = os.path.join(MODEL_DIR, 'lightmunet_paper_best.pth')
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'====== 保存最优模型（F1：{best_val_f1:.4f}）到 {model_save_path} ======')

        scheduler.step(avg_train_loss)

    logging.info(f'\n训练完成！最佳验证集F1：{best_val_f1:.4f}')
    return model, train_loss, val_metrics

# 测试函数
def test(model, test_loader, model_path):
    if not os.path.exists(model_path):
        logging.error(f'未找到模型文件：{model_path}')
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f'\n加载模型：{model_path}，开始测试')

    test_iou_total = 0.0
    test_dice_total = 0.0
    test_f1_total = 0.0
    test_sample_count = 0

    with torch.no_grad():
        for idx, (images, masks, img_paths, mask_paths) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
            pred_binary = (pred_prob >= 0.5).astype(np.uint8) * 255

            iou = get_iou(mask_paths[0], pred_prob)
            dice = get_dice(mask_paths[0], pred_prob)
            f1 = get_f1(mask_paths[0], pred_prob)

            test_iou_total += iou
            test_dice_total += dice
            test_f1_total += f1
            test_sample_count += 1

            img_name = os.path.basename(img_paths[0]).replace('.tif', '_lightmunet_pred.png')
            pred_save_path = os.path.join(PRED_DIR, img_name)
            cv2.imwrite(pred_save_path, pred_binary)
            logging.info(f'测试样本 {idx+1}/{len(test_loader)} | 保存预测图：{pred_save_path}')
            logging.info(f'样本指标：IoU={iou:.4f} | Dice={dice:.4f} | F1={f1:.4f}')

    if test_sample_count > 0:
        avg_test_iou = test_iou_total / test_sample_count
        avg_test_dice = test_dice_total / test_sample_count
        avg_test_f1 = test_f1_total / test_sample_count
        logging.info(f'\n====== 测试集整体指标 ======')
        logging.info(f'平均IoU：{avg_test_iou:.4f}')
        logging.info(f'平均Dice：{avg_test_dice:.4f}')
        logging.info(f'平均F1：{avg_test_f1:.4f}')
        print(f'\n====== 测试集整体指标 ======')
        print(f'平均IoU：{avg_test_iou:.4f}')
        print(f'平均Dice：{avg_test_dice:.4f}')
        print(f'平均F1：{avg_test_f1:.4f}')
    else:
        logging.error('测试集无有效样本')

# 主函数
if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_dataset()

    model = LightMUNet(in_channels=3, num_classes=1).to(device)
    logging.info(f'模型初始化完成：LightM-UNet（参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f} M）')

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    criterion = paper_hybrid_loss

    trained_model, train_loss, val_metrics = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=100
    )

    best_model_path = os.path.join(MODEL_DIR, 'lightmunet_paper_best.pth')
    test(trained_model, test_loader, best_model_path)