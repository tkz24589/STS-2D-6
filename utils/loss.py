import torch
import torch.nn.functional as F
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

def iou_loss(pred, target):
    # 计算预测和目标的交集和并集
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection

    # 计算IOU
    iou = intersection / (union + 1e-6)

    # 计算IOU损失
    iou_loss = 1.0 - iou

    return iou_loss

def h_loss(pred, target):
    # # 计算两个批次中所有mask之间的距离矩阵
    # pred = (pred > thr).float()
    # dists = torch.cdist(pred.reshape(pred.shape[0], -1), target.reshape(target.shape[0], -1), p=1.0)
    # # 计算每个mask与另一个批次中mask的最小距离
    # d1 = torch.min(dists, dim=1).values
    # d2 = torch.min(dists, dim=0).values
    # # 计算距离矩阵中的最大距离
    # hd = torch.tensor(max(np.percentile(d1.cpu().numpy(), 95), np.percentile(d2.cpu().numpy(), 95))).to(pred.device)
    # # 将Hausdorff距离归一化到[0,1]
    # hd = hd / (pred.shape[-1] * pred.shape[-2])
    # return 1 - hd
    # 计算两个批次中所有mask之间的距离矩阵
    # 归一化
    # transform = transforms.Normalize(mean=[0], std=[1])
     # 计算每个点对之间的欧氏距离
    distances = torch.cdist(pred, target, p=1)

    # 计算set1中的每个点到set2的最小距离
    min_dist_set1_to_set2, _ = torch.min(distances, dim=2)

    # 计算set2中的每个点到set1的最小距离
    min_dist_set2_to_set1, _ = torch.min(distances, dim=3)

    min_dist_set = min_dist_set1_to_set2 + min_dist_set2_to_set1

    # 计算最大值和最小值
    max_dist = torch.max(min_dist_set)
    min_dist = torch.min(min_dist_set)

    # 归一化距离
    normalized_dist = min_dist / max(max_dist, 1e-5)
    # normalized_dist = transform(min_dist_set)

    # 取两个集合中的最小距离之和作为二维豪斯多夫距离
    hausdorff_dist = torch.min(normalized_dist) 

    return hausdorff_dist

class DHILoss():
    def __init__(self, dice_wight=0.4, iou_weight=0.3, h_weight=0.3, bce_weight=0.3):
        self.dice_wight = dice_wight
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.h_weight = h_weight
        self.dice = DiceLoss(mode='binary')
        self.bce = nn.BCELoss()

    def do(self, pred, true):
        loss = self.dice(pred, true) * self.dice_wight + iou_loss(pred, true) * self.iou_weight + self.bce(pred, true) * self.bce_weight + self.h_weight * h_loss(pred, true)
        return loss
