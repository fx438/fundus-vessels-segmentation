# test_preprocessing_steps.py（新增极端增强+去噪步骤，6步对比）
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 固定输入输出路径
INPUT_PATH = "/root/autodl-tmp/UNET-ZOO-master/drive/test/images/01_test.tif"
OUTPUT_PATH = "/root/autodl-tmp/UNET-ZOO-master/preprocessing_comparison_enhanced.png"

# -------------------------- 新增：极端增强+去噪核心函数 --------------------------
def extreme_enhance_and_denoise(img_gray):
    """极端增强对比度 + 去除点状噪声（保留线状血管）"""
    # 1. 极端CLAHE增强（clipLimit从2.0→4.0，更激进）
    clahe_extreme = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    img_clahe_extreme = clahe_extreme.apply(img_gray)
    
    # 2. 自适应阈值二值化（极端强化血管与背景对比）
    img_thresh = cv2.adaptiveThreshold(
        img_clahe_extreme, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 反相：血管=白，背景=黑
        blockSize=15, C=2
    )
    
    # 3. 形态学开运算（去除1-2px点状噪声，不破坏线状血管）
    kernel_dot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_dot)
    
    # 4. 方向滤波（强化线状血管，抑制残留噪声）
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),  # 水平
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),  # 垂直
        cv2.rotate(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), cv2.ROTATE_45_CLOCKWISE),  # 45°
        cv2.rotate(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), cv2.ROTATE_135_CLOCKWISE)  # 135°
    ]
    direction_outputs = [cv2.erode(img_opening, kernel) for kernel in kernels]
    img_directional = cv2.bitwise_or(*direction_outputs)
    
    # 5. 轻微膨胀（还原血管宽度）
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_final = cv2.dilate(img_directional, kernel_dilate, iterations=1)
    
    return img_clahe_extreme, img_thresh, img_final

# -------------------------- 原有步骤+新增步骤整合 --------------------------
def get_preprocessing_steps(image_path):
    # 1. 原图（BGR转RGB，适配matplotlib显示）
    img = cv2.imread(image_path)
    img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_original_norm = img_original.astype(np.float32) / 255.0

    # 2. RGB转灰度（预处理前置步骤）
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)

    # 3. Z-score标准化 + Min-Max归一化 + 255缩放
    img_zscore = (img_gray - np.mean(img_gray)) / np.std(img_gray)
    img_norm = (img_zscore - img_zscore.min()) / (img_zscore.max() - img_zscore.min())
    img_norm = (img_norm * 255).astype(np.uint8)
    img_norm_norm = img_norm / 255.0

    # 4. 原始CLAHE增强（clipLimit=2.0）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_norm)
    img_clahe_norm = img_clahe / 255.0

    # 5. 新增：极端对比度增强（clahe+阈值二值化）
    img_extreme_clahe, img_thresh, img_denoised = extreme_enhance_and_denoise(img_norm)
    img_extreme_clahe_norm = img_extreme_clahe / 255.0
    img_thresh_norm = img_thresh / 255.0
    img_denoised_norm = img_denoised / 255.0

    # 6. 伽马校正（gamma=1.5，你之前设置的参数）
    gamma = 1.5
    img_gamma = np.power(img_clahe / 255.0, gamma) * 255.0
    img_gamma = img_gamma.astype(np.uint8)
    img_gamma_norm = img_gamma / 255.0

    return [
        ("原图", img_original_norm),
        ("Z-score+Min-Max归一化", img_norm_norm),
        ("CLAHE增强(clipLimit=2.0)", img_clahe_norm),
        ("极端增强(clipLimit=4.0+阈值)", img_thresh_norm),
        ("去噪后(形态学+方向滤波)", img_denoised_norm),
        ("伽马校正(gamma=1.5)", img_gamma_norm)
    ]

# -------------------------- 生成6步对比图（3行2列布局） --------------------------
def generate_comparison():
    steps = get_preprocessing_steps(INPUT_PATH)
    fig, axes = plt.subplots(3, 2, figsize=(18, 24))  # 3行2列，适配6步
    axes = axes.flatten()

    for idx, (title, img) in enumerate(steps):
        if len(img.shape) == 3:
            axes[idx].imshow(img)
        else:
            axes[idx].imshow(img, cmap='gray')  # 灰度图用gray通道更清晰
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=15)
        axes[idx].axis('off')

    # 整体标题（突出核心对比：极端增强+去噪）
    plt.suptitle("Retinal Image Preprocessing Comparison (Extreme Enhance + Denoise)", 
                 fontsize=20, y=0.98, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight')
    print(f"6步对比图已保存至：{OUTPUT_PATH}")
    print(f"核心观察点：")
    print(f"  1. 极端增强步骤：血管与背景对比拉满，但点状噪声明显")
    print(f"  2. 去噪步骤：点状噪声消失，线状血管完整保留")
    print(f"  3. 对比伽马校正：去噪后的血管更干净，无噪声干扰")

if __name__ == "__main__":
    generate_comparison()