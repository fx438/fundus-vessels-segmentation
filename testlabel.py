import cv2
import imageio
import numpy as np

# 替换为你的标签路径
mask_path = "/root/autodl-tmp/UNET-ZOO-master/drive/training/1st_manual/21_manual1.gif"
# 用你分块模式的读取逻辑
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    mask = imageio.mimread(mask_path)[0][..., 0]
# 打印关键信息
print(f"标签像素范围：{mask.min()}~{mask.max()}")  # 正确应为0~255
print(f"血管像素数（>127的像素）：{np.sum(mask > 127)}")  # 正确应>0（通常几千~几万）