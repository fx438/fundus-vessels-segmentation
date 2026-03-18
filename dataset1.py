import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio

import cv2
import numpy as np
import imageio
from torch.utils.data import Dataset

class DriveEyeDataset(Dataset):
    def __init__(self, pics, masks, transform=None, target_transform=None, aug=False):
        self.pics = pics
        self.masks = masks
        self.transform = transform  # 接收main.py的transform（图像增强+归一化）
        self.target_transform = target_transform  # 接收main.py的target_transform（标签处理）
        self.aug = aug  # 禁用自定义增强，统一用main.py的transform（避免双重增强）

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, index):
        imgx, imgy = (576, 576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]

        # 1. 读取图像：cv2读为BGR，转RGB（和PIL格式一致），不手动归一化
        pic = cv2.imread(pic_path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # 转RGB（和main.py的transform兼容）
        pic = cv2.resize(pic, (imgx, imgy))  # Resize到576x576

        # 2. 读取标签：确保单通道，血管=255，背景=0（不手动归一化）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 处理gif格式标签（DRIVE部分标签是gif）
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]  # 取第一帧
            if len(mask.shape) == 3:
                mask = mask[..., 0]  # 转单通道
        mask = cv2.resize(mask, (imgx, imgy))  # Resize到576x576

        # 3. 禁用自定义增强（统一用main.py的transform，确保图像和标签同步增强）
        # if self.aug:
        #     pic, mask = self._safe_augment(pic, mask)  # 注释掉，避免双重增强

        # 4. 应用transform（main.py中定义的增强+归一化）
        if self.transform is not None:
            pic = self.transform(pic)  # 图像：ToTensor() + 增强 + Normalize
        if self.target_transform is not None:
            mask = self.target_transform(mask)  # 标签：转张量 + 二值化

        # 确保标签是单通道（[1, 576, 576]），和模型输出匹配
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return pic, mask, pic_path, mask_path

class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = (state == 'train')  # 训练集自动开启增强，其他集关闭（安全开关）
        self.root = r'/root/autodl-tmp/UNET-ZOO-master/drive'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform  # 保留 main.py 传入的 transform，不替换！
        self.target_transform = target_transform  # 保留原始转换

    # 新增：安全的增强函数（只对训练集生效）
    def _safe_augment(self, pic, mask):
        h, w = pic.shape[:2]
        # 只加“水平翻转”（最安全，不会破坏特征位置）
        if random.random() > 0.5:
            pic = cv2.flip(pic, 1)
            mask = cv2.flip(mask, 1)
        #亮度微调
        alpha = random.uniform(0.95, 1.05)
        pic = np.clip(pic * alpha, 0, 255).astype(np.uint8)
        return pic, mask

    def __getitem__(self, index):
        imgx, imgy = (576, 576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
            if len(mask.shape) == 3:
                mask = mask[..., 0]
        
        # 训练集增强（安全开启）
        if self.aug:
            pic, mask = self._safe_augment(pic, mask)
        
        # 原有预处理逻辑不变（保持和 main.py 兼容）
        pic = cv2.resize(pic, (imgx, imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def getDataPath(self):
        # 1. 读取原图和 mask 路径
        all_train_imgs = glob(self.root + r'/training/images/*')
        all_train_masks = glob(self.root + r'/training/1st_manual/*')
        
        # 核心修复：按文件名排序，确保原图和 mask 一一对应（关键！）
        def sort_by_filename(path_list):
            # 提取文件名中的数字（如 01、02），按数字排序
            return sorted(path_list, key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        all_train_imgs = sort_by_filename(all_train_imgs)
        all_train_masks = sort_by_filename(all_train_masks)
        
        # 2. 拆分训练集/验证集（此时顺序完全对应）
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_train_imgs,
            all_train_masks,
            test_size=0.2,
            random_state=42,
            shuffle=True  # 允许打乱，但保持原图和 mask 的对应关系
        )

        # 测试集同样排序
        test_imgs = glob(self.root + r'/test/images/*')
        test_masks = glob(self.root + r'/test/1st_manual/*')
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
    def __len__(self):
        return len(self.pics)
class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"D:\2345Downloads\UNET-ZOO-master\UNET-ZOO-master\liver\train"  # 请根据你的实际路径修改
        self.val_root = r"D:\2345Downloads\UNET-ZOO-master\UNET-ZOO-master\liver\val"      # 请根据你的实际路径修改
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
        return pics, masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)

class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r"D:\2345Downloads\UNET-ZOO-master\UNET-ZOO-master\isbi"  # 请根据你的实际路径修改
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def crop_with_retinal_mask(self, pic, mask, pic_path):
        """
        使用视网膜掩膜裁剪原始图像和血管标注，仅保留眼底有效区域
        :param pic: 原始图像（BGR格式，numpy数组）
        :param mask: 血管标注（灰度图，numpy数组）
        :param pic_path: 图像路径，用于提取编号并拼接掩膜路径
        :return: 裁剪后的图像、裁剪后的血管标注
        """
        # 提取图像编号（如从"21_training.tif"中提取"21"）
        img_num = os.path.basename(pic_path).split('_')[0]
        # 拼接视网膜掩膜路径
        retina_mask_path = os.path.join(
            os.path.dirname(pic_path).replace("images", "mask"),
            f"{img_num}_training_mask.gif"
        )
        # 读取视网膜掩膜并归一化
        retina_mask = imageio.imread(retina_mask_path) / 255.0  # 归一化到0-1范围
        # 裁剪：仅保留掩膜内的区域（图像为3通道，掩膜为单通道，需扩展维度后相乘）
        cropped_pic = pic * retina_mask[..., np.newaxis]
        cropped_mask = mask * retina_mask
        return cropped_pic, cropped_mask

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        #眼底去背景去噪
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        pic, mask = self.crop_with_retinal_mask(pic, mask, pic_path)
        #cache增强
        lab = cv2.cvtColor(pic.astype(np.uint8), cv2.COLOR_BGR2LAB)  # 转LAB色彩空间
        l_channel = lab[:, :, 0]  # 提取亮度通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 论文推荐参数
        l_channel_enhanced = clahe.apply(l_channel)  # 增强亮度通道
        lab[:, :, 0] = l_channel_enhanced
        pic = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #归一化
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)