import numpy as np
import cv2

def geodesic_voting(post_prob, img_gray, geo_radius=3, vote_thresh=0.4):
    """
    测地线投票优化视网膜血管分割概率图
    Args:
        post_prob: 模型输出的概率图（H, W），值范围0-1
        img_gray: 原始灰度图像（H, W），用于计算测地线距离
        geo_radius: 测地线邻域半径（3-5像素，适配细血管）
        vote_thresh: 投票阈值（0.3-0.5）
    Returns:
        voted_prob: 投票优化后的概率图（H, W）
    """
    H, W = post_prob.shape
    voted_prob = np.zeros_like(post_prob)
    
    # 1. 预处理：计算图像梯度（用于测地线距离权重）
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())  # 归一化
    weight_map = 1 - grad_mag  # 梯度小的区域（血管）权重高，梯度大的区域（边缘/噪声）权重低
    
    # 2. 对每个像素计算测地线邻域投票（优化：用矩阵运算替代双重循环，提升速度）
    for i in range(H):
        for j in range(W):
            # 构建欧氏邻域（再通过测地线权重过滤）
            min_i = max(0, i - geo_radius)
            max_i = min(H, i + geo_radius + 1)
            min_j = max(0, j - geo_radius)
            max_j = min(W, j + geo_radius + 1)
            
            # 邻域内像素坐标
            ni, nj = np.meshgrid(range(min_i, max_i), range(min_j, max_j), indexing='ij')
            ni_flat = ni.flatten()
            nj_flat = nj.flatten()
            
            # 计算测地线权重（结合欧氏距离和梯度权重）
            euclid_dist = np.sqrt((ni_flat - i)**2 + (nj_flat - j)**2)  # 欧氏距离
            geo_weight = weight_map[ni_flat, nj_flat] * np.exp(-euclid_dist / geo_radius)  # 测地线权重
            geo_weight = geo_weight / (geo_weight.sum() + 1e-6)  # 归一化（避免除零）
            
            # 投票：邻域内概率的加权和
            vote_sum = (post_prob[ni_flat, nj_flat] * geo_weight).sum()
            voted_prob[i, j] = vote_sum
    
    # 3. 阈值过滤：低于阈值的设为0（抑制噪声）
    voted_prob[voted_prob < vote_thresh] = 0
    return voted_prob