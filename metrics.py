import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
from sklearn.metrics import roc_auc_score  # 引入AUC计算依赖
flag=0.5
# -------------------------- 原有类和函数（保持不变） --------------------------
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

def _preprocess(mask_name, predict):
    """统一预处理：读取掩码、转单通道、统一尺寸、二值化（避免重复代码）"""
    # 1. 读取掩码并强制转单通道
    image_mask = cv2.imread(mask_name, 0)  # 0=单通道模式
    if image_mask is None:
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        # 容错：若读取后是3通道，转单通道
        if len(image_mask.shape) == 3:
            image_mask = image_mask[..., 0]  # 取第1通道
        image_mask = cv2.resize(image_mask, (576, 576))
    
    # 2. 统一预测图和掩码的形状
    predict = cv2.resize(predict, (image_mask.shape[1], image_mask.shape[0]))
    
    # 3. 掩码二值化（0/1），预测图保留原始概率（AUC需概率值，非二值）
    image_mask = (image_mask >= 125).astype(np.int16)  # 掩码转0/1
    predict = np.clip(predict, 0, 1)  # 预测概率裁剪到[0,1]，避免异常值
    
    return image_mask, predict

# -------------------------- 原有指标函数（保持不变） --------------------------
def get_iou(mask_name, predict):
    image_mask, predict = _preprocess(mask_name, predict)
    predict_bin = (predict >= flag).astype(np.int16)  # 二值化预测图
    
    # 计算IoU（避免除0错误）
    inter = np.sum(predict_bin & image_mask)
    union = np.sum(predict_bin | image_mask)
    iou_tem = inter / (union + 1e-8)  # 添加1e-8避免除0
    
    print(f'{mask_name}: IoU={iou_tem:.4f}')
    return iou_tem

def get_sp(mask_name, predict):
    """
    计算特异性（Specificity, SP）：
    公式：Specificity = TN / (TN + FP)
    其中：
    - TN（True Negative，真负例）：真实为背景（0）且预测为背景（0）的像素数
    - FP（False Positive，假正例）：真实为背景（0）但预测为血管（1）的像素数
    """
    # 1. 预处理：获取二值化真实掩码和预测概率图
    image_mask, predict = _preprocess(mask_name, predict)
    # 2. 预测图二值化（与你原有Dice逻辑一致，阈值0.5）
    predict_bin = (predict >= flag).astype(np.int16)  # 1=预测为血管，0=预测为背景
    
    # 3. 计算特异性核心指标（TN和FP）
    # TN：真实背景（image_mask==0）且预测背景（predict_bin==0）
    TN = np.sum((image_mask == 0) & (predict_bin == 0))
    # FP：真实背景（image_mask==0）但预测血管（predict_bin==1）
    FP = np.sum((image_mask == 0) & (predict_bin == 1))
    
    # 4. 计算特异性（添加1e-8避免分母为0，与你原有Dice的防错逻辑一致）
    specificity = TN / (TN + FP + 1e-8)
    
    return specificity

def get_hd(mask_name, predict):
    image_mask, predict = _preprocess(mask_name, predict)
    predict_bin = (predict >= flag).astype(np.int16)  # 二值化预测图
    
    # 计算Hausdorff距离（避免空集错误）
    try:
        # 提取前景像素坐标（只计算血管区域的距离）
        mask_coords = np.argwhere(image_mask == 1)
        pred_coords = np.argwhere(predict_bin == 1)
        
        # 处理空集情况（无血管时返回0）
        if len(mask_coords) == 0 or len(pred_coords) == 0:
            return 0.0
        
        hd1 = directed_hausdorff(mask_coords, pred_coords)[0]
        hd2 = directed_hausdorff(pred_coords, mask_coords)[0]
        return max(hd1, hd2)
    except Exception as e:
        print(f"计算HD时出错：{e}")
        return np.inf  # 空集时返回无穷大，不影响整体均值

def get_precision(mask_name, predict):
    """精准度：预测为血管且实际为血管的比例（TP/(TP+FP)）"""
    image_mask, predict = _preprocess(mask_name, predict)
    predict_bin = (predict >= flag).astype(np.int16)  # 二值化预测图
    
    TP = np.sum(predict_bin & image_mask)  # 真阳性
    FP = np.sum(predict_bin & (1 - image_mask))  # 假阳性
    precision = TP / (TP + FP + 1e-8)  # 避免除0
    
    print(f'{mask_name}: Precision={precision:.4f}')
    return precision

def get_recall(mask_name, predict):
    """召回率：实际为血管且预测为血管的比例（TP/(TP+FN)）"""
    image_mask, predict = _preprocess(mask_name, predict)
    predict_bin = (predict >= flag).astype(np.int16)  # 二值化预测图
    
    TP = np.sum(predict_bin & image_mask)  # 真阳性
    FN = np.sum((1 - predict_bin) & image_mask)  # 假阴性
    recall = TP / (TP + FN + 1e-8)  # 避免除0
    
    print(f'{mask_name}: Recall={recall:.4f}')
    return recall

def get_f1(mask_name, predict):
    """F1-Score：Precision和Recall的调和平均"""
    precision = get_precision(mask_name, predict)
    recall = get_recall(mask_name, predict)
    
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)  # 避免除0
    print(f'{mask_name}: F1-Score={f1:.4f}')
    return f1

def get_acc(mask_name, predict):
    """准确率：所有预测正确的像素占总像素的比例（(TP+TN)/(TP+TN+FP+FN)）"""
    image_mask, predict = _preprocess(mask_name, predict)
    predict_bin = (predict >= flag).astype(np.int16)  # 二值化预测图
    
    TP = np.sum(predict_bin & image_mask)        # 真阳性（血管预测正确）
    TN = np.sum((1 - predict_bin) & (1 - image_mask))  # 真阴性（背景预测正确）
    total_pixels = image_mask.size           # 总像素数
    acc = (TP + TN) / (total_pixels + 1e-8)  # 避免除0
    
    print(f'{mask_name}: ACC={acc:.4f}')
    return acc

def get_sen(mask_name, predict):
    """灵敏度（Sensitivity）：实际为血管且被正确预测的比例（同Recall，用于医学影像统一术语）"""
    # 直接复用Recall的逻辑，避免重复代码
    sen = get_recall(mask_name, predict)
    print(f'{mask_name}: SEN={sen:.4f}')  # 单独打印SEN指标名
    return sen

# -------------------------- 新增：AUC（ROC曲线下面积）计算函数 --------------------------
def get_auc(mask_name, predict, use_mask=True):
    """
    计算AUC（ROC曲线下面积）：评估模型区分血管与背景的能力
    Args:
        mask_name: 掩码文件路径（.png/.jpg等）
        predict: 模型预测概率图（需为[0,1]概率值，非二值化）
        use_mask: 是否使用掩码的有效区域（默认True，仅计算掩码内像素）
    Returns:
        auc: AUC值（范围[0,1]，越接近1表示区分能力越强）
    """
    # 1. 统一预处理：获取二值掩码和原始预测概率
    image_mask, predict_prob = _preprocess(mask_name, predict)
    
    # 2. 展平数据（适配sklearn.roc_auc_score输入格式）
    y_true = image_mask.flatten()  # 真实标签（0=背景，1=血管）
    y_score = predict_prob.flatten()  # 预测概率（需[0,1]）
    
    # 3. （可选）仅保留掩码有效区域（若掩码含无效标记，此处可扩展，当前默认全有效）
    if use_mask:
        # 过滤掉标签异常值（若有），仅保留0/1像素
        valid_idx = (y_true == 0) | (y_true == 1)
        y_true = y_true[valid_idx]
        y_score = y_score[valid_idx]
    
    # 4. 处理极端情况（避免计算错误）
    try:
        # 情况1：无血管或无背景（所有标签相同），AUC无意义，返回1.0（完美区分）
        if len(np.unique(y_true)) == 1:
            print(f'{mask_name}: 标签仅含单一类别（无血管/无背景），AUC=1.0000')
            return 1.0
        # 情况2：正常计算AUC
        auc = roc_auc_score(y_true, y_score)
        auc = np.clip(auc, 0, 1)  # 确保AUC在[0,1]（避免数值异常）
        print(f'{mask_name}: AUC={auc:.4f}')
        return auc
    except Exception as e:
        print(f'{mask_name}: 计算AUC时出错：{e}，返回0.5000（随机猜测水平）')
        return 0.5