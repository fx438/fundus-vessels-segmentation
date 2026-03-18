import torch.utils.data as data
import numpy as np
import cv2
import imageio
from sklearn.model_selection import train_test_split
from glob import glob
import os
import random

class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = (state == 'train')  # 训练集自动开启增强
        self.root = r'/root/autodl-tmp/UNET-ZOO-master/drive'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform  # 保留外部transform（如ToTensor、归一化）
        self.target_transform = target_transform
        self.img_size = (576, 576)  # 保持原有尺寸适配

    def _lightm_unet_preprocess(self, pic):
        """无裁切模式下的优化：增强细小血管对比度，让模型看清细节"""
        # 1. RGB转灰度（保留亮度差异，血管和背景的核心区别）
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
        
        # 2. 局部对比度增强（关键！突出细小血管，不放大噪声）
        # 核尺寸=3（越小，越关注细小结构），clipLimit=1.5（比原来低，避免噪声过度放大）
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
        clahe_pic = clahe.apply(gray_pic)
        
        # 3. 细节锐化（进一步强化细小血管的边缘）
        # 拉普拉斯锐化：突出高频细节（细小血管属于高频特征）
        laplacian = cv2.Laplacian(clahe_pic, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        # 融合锐化后的图像（原图80% + 锐化图20%，平衡细节和噪声）
        enhanced_gray = cv2.addWeighted(clahe_pic, 0.8, laplacian, 0.2, 0)
        
        # 4. 自适应Gamma矫正（解决光照不均，避免部分区域细小血管被掩盖）
        mean_val = np.mean(enhanced_gray)
        gamma = np.clip(1.0 - (mean_val / 255.0 - 0.5) * 2, 0.8, 1.2)  # 缩小Gamma范围，更温和
        gamma_pic = np.power(enhanced_gray / 255.0, gamma) * 255.0
        gamma_pic = gamma_pic.astype(np.uint8)
        
        # 5. 恢复三通道（和你无裁切时的输入格式一致，模型无需适配）
        preprocessed_pic = cv2.cvtColor(gamma_pic, cv2.COLOR_GRAY2RGB)
        return preprocessed_pic

    def _safe_augment(self, pic, mask):
        """保留原有安全增强：水平翻转+亮度微调，不破坏血管特征"""
        h, w = pic.shape[:2]
        # 水平翻转（50%概率）
        if random.random() > 0.5:
            pic = cv2.flip(pic, 1)
            mask = cv2.flip(mask, 1)
        # 亮度微调（±5%）
        alpha = random.uniform(0.95, 1.05)
        pic = np.clip(pic * alpha, 0, 255).astype(np.uint8)
        return pic, mask

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]

        # 1. 读取原始图像（BGR转RGB，保持原有格式逻辑）
        pic = cv2.imread(pic_path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        pic = cv2.resize(pic, self.img_size)  # 统一缩放尺寸

        # 2. （灰度化→均衡化→Gamma矫正）
        pic = self._lightm_unet_preprocess(pic)

        # 3. 读取并处理标签（保持原有逻辑：处理gif格式+单通道）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 处理DRIVE数据集的gif格式标签
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]  # 取第一帧
            if len(mask.shape) == 3:
                mask = mask[..., 0]  # 转单通道
        mask = cv2.resize(mask, self.img_size)  # 标签与图像尺寸匹配

        # 4. 训练集增强（仅训练集执行，与预处理步骤衔接）
        if self.aug:
            pic, mask = self._safe_augment(pic, mask)

        # 5. 归一化（保持原有逻辑：图像和标签归一到[0,1]）
        pic = pic.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0  # 血管=1，背景=0

        # 6. 应用外部transform（如ToTensor、额外归一化等，兼容原有流程）
        if self.transform is not None:
            pic = self.transform(pic)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # 确保标签为单通道（适配模型输出格式）
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return pic, mask, pic_path, mask_path

    def getDataPath(self):
        """保留原有路径管理逻辑：按文件名排序确保图-标对应，拆分训练/验证/测试集"""
        # 读取训练集路径
        all_train_imgs = glob(self.root + r'/training/images/*')
        all_train_masks = glob(self.root + r'/training/1st_manual/*')
        
        # 按文件名数字排序（核心：确保原图与mask一一对应）
        def sort_by_filename(path_list):
            return sorted(path_list, key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        all_train_imgs = sort_by_filename(all_train_imgs)
        all_train_masks = sort_by_filename(all_train_masks)
        
        # 拆分训练集/验证集（保持原有比例和随机种子）
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_train_imgs, all_train_masks,
            test_size=0.2, random_state=42, shuffle=True
        )

        # 读取并排序测试集路径
        test_imgs = glob(self.root + r'/test/images/*')
        test_masks = glob(self.root + r'/test/1st_manual/*')
        test_imgs = sort_by_filename(test_imgs)
        test_masks = sort_by_filename(test_masks)

        # 根据state返回对应数据集
        if self.state == 'train':
            return train_imgs, train_masks
        elif self.state == 'val':
            return val_imgs, val_masks
        elif self.state == 'test':
            return test_imgs, test_masks
        else:
            raise ValueError("state must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.pics)