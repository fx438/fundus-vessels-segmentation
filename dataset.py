import logging
import torch.utils.data as data
import numpy as np
import cv2
import imageio
from sklearn.model_selection import train_test_split
from glob import glob
import os
import random
from scipy.ndimage import gaussian_filter

class DRIVEDataset_Paper(data.Dataset):
    def __init__(self, state, patch_size=48, num_train_patches=15000,
                 transform=None, target_transform=None):
        self.state = state  # 'train'/'val'/'test'
        self.aug = (state == 'train')
        self.root = r'/root/autodl-tmp/UNET-ZOO-master/drive'
        self.pics, self.masks = self._get_data_path()
        self.transform = transform
        self.current_epoch = 0
        self.target_transform = target_transform
        self.patch_size = patch_size
        self.num_train_patches = num_train_patches
        
        # 初始化时确保变量有值
        if self.state == 'train':
            self.train_patches = self._generate_paper_patches()
        else:
            self.img_size = (576, 576)

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_data_path(self):
        # 读取训练集路径
        all_train_imgs = glob(os.path.join(self.root, 'training', 'images', '*'))
        all_train_masks = glob(os.path.join(self.root, 'training', '1st_manual', '*'))
        
        # 按文件名排序
        def sort_by_filename(path_list):
            return sorted(path_list, key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        all_train_imgs = sort_by_filename(all_train_imgs)
        all_train_masks = sort_by_filename(all_train_masks)
        
        # 拆分训练集/验证集
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_train_imgs, all_train_masks, test_size=0.2, random_state=42, shuffle=True
        )

        # 读取测试集路径
        test_imgs = glob(os.path.join(self.root, 'test', 'images', '*'))
        test_masks = glob(os.path.join(self.root, 'test', '1st_manual', '*'))
        test_imgs = sort_by_filename(test_imgs)
        test_masks = sort_by_filename(test_masks)

        if self.state == 'train':
            return train_imgs, train_masks
        elif self.state == 'val':
            return val_imgs, val_masks
        elif self.state == 'test':
            return test_imgs, test_masks
        else:
            raise ValueError("state must be 'train', 'val', or 'test'")

    def _paper_preprocess(self, img, mask):
        # RGB转灰度
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Z-score标准化+Min-Max归一化
        img_norm = (img_gray - np.mean(img_gray)) / np.std(img_gray)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
        img_norm = (img_norm * 255).astype(np.uint8)
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_norm)
        
        # 伽马校正（动态调整gamma值，增加多样性）
        if self.aug and random.random() > 0.5:
            gamma = random.uniform(0.9, 1.3)
        else:
            gamma = 1.1
        img_gamma = np.power(img_clahe / 255.0, gamma) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        
        # 标签二值化
        mask_binary = (mask > 127).astype(np.uint8) * 1
        
        # 恢复三通道
        img_final = cv2.cvtColor(img_gamma, cv2.COLOR_GRAY2RGB)
        return img_final, mask_binary

    def _generate_paper_patches(self):
        patches = []
        patches_per_img = self.num_train_patches // len(self.pics)
        remaining_patches = self.num_train_patches % len(self.pics)
        
        for idx, (img_path, mask_path) in enumerate(zip(self.pics, self.masks)):
            # 读取原图
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"训练集图像读取失败：{img_path}，跳过该图")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 读取掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                try:
                    mask = imageio.mimread(mask_path)
                    mask = np.array(mask)[0]
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                except Exception as e:
                    logging.warning(f"训练集掩码读取失败：{mask_path}，错误：{e}，跳过该图")
                    continue
            
            # 论文预处理
            img_prep, mask_prep = self._paper_preprocess(img, mask)
            h, w = img_prep.shape[:2]
            
            current_patches = 0
            target = patches_per_img + (1 if idx < remaining_patches else 0)
            
            # 硬负样本比例
            hard_negative_ratio = 0.1
            hard_negative_target = int(target * hard_negative_ratio)
            normal_target = target - hard_negative_target
            
            # 生成正常样本（含血管）
            while current_patches < normal_target:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                
                img_patch = img_prep[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask_prep[y:y+self.patch_size, x:x+self.patch_size]
                
                if np.sum(mask_patch) > 0:
                    patches.append((img_patch, mask_patch))
                    current_patches += 1
            
            # 生成硬负样本（无血管）
            hard_negative_count = 0
            while hard_negative_count < hard_negative_target:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                
                img_patch = img_prep[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask_prep[y:y+self.patch_size, x:x+self.patch_size]
                
                if np.sum(mask_patch) == 0:
                    patches.append((img_patch, mask_patch))
                    hard_negative_count += 1
        
        random.shuffle(patches)
        return patches[:self.num_train_patches]

    def _safe_augment(self, img, mask):
        """
        调整后三阶段增强策略：
        1. 前25轮：温和增强（保留基础特征，避免模型初期学偏）
        2. 25轮后：低对比度血管专项增强（核心优化提前，聚焦断裂修复）
                   （原25-55轮超强增强与55轮后专项增强合并，简化逻辑且提前介入）
        """
        h, w = img.shape[:2]
        current_epoch = self.current_epoch

        # -------------------------- 阶段1：前15轮 - 温和增强 --------------------------
        if current_epoch < 15:
            # 1. 水平/垂直翻转（概率70%）
            if random.random() > 0.3:
                flip_type = random.choice([0, 1])
                img = cv2.flip(img, flip_type)
                mask = cv2.flip(mask, flip_type)
            
            # 2. 小角度旋转（±12°，概率60%）
            if random.random() > 0.4:
                angle = random.randint(-12, 12)
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h), borderValue=(0,0,0))
                mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)
            
            # 3. 轻微亮度/对比度扰动（概率50%）
            if random.random() > 0.5:
                img = self._adjust_brightness_contrast(
                    img, alpha=random.uniform(0.9, 1.1), beta=random.randint(-8, 8)
                )
            
            # 4. 轻微高斯噪声（概率40%）
            if random.random() > 0.6:
                img = self._add_gaussian_noise(img, mean=0, var=10)

        # -------------------------- 阶段2：25轮后 - 低对比度血管专项增强（核心调整） --------------------------
        else:
            # 基础操作：翻转+小角度旋转（保护血管连续性，避免强旋转导致断裂）
            flip_type = random.choice([-1, 0, 1])
            img = cv2.flip(img, flip_type)
            mask = cv2.flip(mask, flip_type)
            
            angle = random.randint(-10, 10)  # 适度控制旋转角度，平衡多样性与连续性
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(0,0,0))
            mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)

            # 核心1：低对比度血管对比度增强（解决亮度不均导致的断裂）
            if random.random() > 1.5:  # 概率80%，高频介入
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # 大核自适应阈值（突出低对比度血管边缘，避免小核过度分割）
                img_thresh = cv2.adaptiveThreshold(
                    img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 15, 8
                )
                # 融合原图像和阈值图像，保留细节+增强血管显著性
                img_gray_enhance = cv2.addWeighted(img_gray, 0.7, img_thresh, 0.3, 0)
                img = cv2.cvtColor(img_gray_enhance, cv2.COLOR_GRAY2RGB)

            # 核心2：形态学闭运算（修复低对比度血管微小断裂，仅作用于图像特征层）
            if random.random() > 0.15:  # 概率80%，重点优化断裂
                # 椭圆结构元（适配血管管状形态，避免方形核破坏血管连续性）
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # 闭运算：先膨胀填充间隙，再腐蚀恢复原尺寸，修复微小断裂
                img_gray_close = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
                img = cv2.cvtColor(img_gray_close, cv2.COLOR_GRAY2RGB)

            # 核心3：定向亮度补偿（针对低对比度血管暗区，避免暗区漏检）
            if random.random() > 0.35:  # 概率60%，精准提升暗区血管可见性
                img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                v_channel = img_hsv[..., 2]
                # 仅提升低亮度区域（<100），高亮度区域保持不变，避免过度曝光
                v_channel_enhance = np.where(v_channel < 100, v_channel * 1.2, v_channel)
                v_channel_enhance = np.clip(v_channel_enhance, 0, 255).astype(np.uint8)
                img_hsv[..., 2] = v_channel_enhance
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

            # 核心4：血管边缘强化（避免低对比度血管边缘模糊导致预测断裂）
            if random.random() > 0.3:  # 概率70%，强化边缘特征
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # 拉普拉斯边缘检测（突出血管轮廓，抑制背景噪声）
                laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
                laplacian = np.uint8(np.absolute(laplacian))
                # 融合边缘与原图，平衡细节保留与边缘显著性
                img_gray_edge = cv2.addWeighted(img_gray, 0.8, laplacian, 0.2, 0)
                img = cv2.cvtColor(img_gray_edge, cv2.COLOR_GRAY2RGB)

            # 辅助增强1：适度弹性变形（增加血管形态多样性，避免过拟合）
            if random.random() > 0.6:  # 概率40%，温和变形不破坏结构
                alpha = random.uniform(15, 25)
                sigma = random.uniform(4, 5)
                img = self._elastic_transform(img, alpha=alpha, sigma=sigma)
                mask = self._elastic_transform(mask, alpha=alpha, sigma=sigma)

            # 辅助增强2：轻度高斯模糊+噪声（提升模型鲁棒性，不破坏血管特征）
            if random.random() > 0.7:  # 概率30%，低频率介入避免干扰
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.5, 0.8))
                img = self._add_gaussian_noise(img, mean=0, var=5)

        return img, mask

    # -------------------------- 辅助工具函数 --------------------------
    def _adjust_brightness_contrast(self, img, alpha=1.0, beta=0):
        """调整亮度和对比度"""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _elastic_transform(self, img, alpha=25, sigma=5):
        """弹性变形"""
        h, w = img.shape[:2]
        # 生成随机位移场
        dx = np.random.randn(h, w) * 0.01
        dy = np.random.randn(h, w) * 0.01
        # 高斯平滑位移场
        dx = gaussian_filter(dx, sigma) * alpha
        dy = gaussian_filter(dy, sigma) * alpha
        # 生成网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        # 应用变形
        if len(img.shape) == 3:
            return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        else:
            return cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    def _add_gaussian_noise(self, img, mean=0, var=10):
        """添加高斯噪声"""
        if len(img.shape) == 3:
            h, w, c = img.shape
            noise = np.random.normal(mean, var**0.5, (h, w, c)).astype(np.float32)
        else:
            h, w = img.shape
            noise = np.random.normal(mean, var**0.5, (h, w)).astype(np.float32)
        
        img_noisy = img.astype(np.float32) + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
        return img_noisy

    def __getitem__(self, index):
        # 初始化img和mask
        img = None
        mask = None
        
        if self.state == 'train':
            # 训练集：从预生成的块中读取
            img_patch, mask_patch = self.train_patches[index]
            img, mask = img_patch.copy(), mask_patch.copy()
            # 训练集增强
            if self.aug:
                img, mask = self._safe_augment(img, mask)
            
            # 归一化
            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
        
        else:
            # 验证/测试集：读取完整图像
            img_path = self.pics[index]
            mask_path = self.masks[index]
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"验证/测试集图像读取失败：{img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # 读取掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                try:
                    mask = imageio.mimread(mask_path)
                    mask = np.array(mask)[0]
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                except Exception as e:
                    raise ValueError(f"验证/测试集掩码读取失败：{mask_path}，错误：{e}")
            mask = cv2.resize(mask, self.img_size)
            
            # 论文预处理
            img, mask = self._paper_preprocess(img, mask)
            
            # 归一化
            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
        
        # 应用外部transform
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        # 确保标签为单通道
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        # 返回对应数据
        if self.state == 'train':
            return img, mask
        else:
            return img, mask, self.pics[index], self.masks[index]

    def __len__(self):
        if self.state == 'train':
            return len(self.train_patches)
        else:
            return len(self.pics)