import torch


def dice_coef(preds, targets, thr=0.5, beta=0.5, smooth=1e-5):

    #comment out if your model contains a sigmoid or equivalent activation layer
    # flatten label and prediction tensors
    preds = (preds > thr).view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def IoU(pred, target, thr=0.5):
    smooth = 1.
    p_flat = (pred > thr).view(-1).float()
    t_flat = target.view(-1).float()
    intersection = (p_flat * t_flat).sum()
    union = (p_flat + t_flat).sum() - intersection
    return (intersection + smooth) / (union + smooth)


def h_score(pred, target, thr=0.5):
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
    pred = (pred > thr).float()
    #  # 计算每个点对之间的欧氏距离
    # distances = torch.cdist(pred, target)

    # # 计算set1中的每个点到set2的最小距离
    # min_dist_set1_to_set2, _ = torch.min(distances, dim=1)

    # # 计算set2中的每个点到set1的最小距离
    # min_dist_set2_to_set1, _ = torch.min(distances, dim=0)

    # # 取两个集合中的最小距离之和作为二维豪斯多夫距离
    # hausdorff_dist = torch.min(min_dist_set1_to_set2) + torch.min(min_dist_set2_to_set1)

    # # 将Hausdorff距离归一化到[0,1]

    # hausdorff_dist = hausdorff_dist / (torch.max(min_dist_set1_to_set2) + torch.max(min_dist_set2_to_set1))
    # 计算横轴和纵轴上的绝对距离之和
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

    return 1 - hausdorff_dist

class Score():
    def __init__(self, dice_wight=0.5, iou_weight=0.5):
        self.dice_wight = dice_wight
        self.iou_weight = iou_weight

    def score(self, pred, true, th):
        dice = dice_coef(pred, true, th)
        iou = IoU(pred, true, th)
        score = self.dice_wight * dice + self.iou_weight * iou
        return score

class DHIScore():
    def __init__(self, dice_wight=0.4, iou_weight=0.3, h_weight=0.3):
        self.dice_wight = dice_wight
        self.iou_weight = iou_weight
        self.h_weight = h_weight


    def score(self, pred, true, th):
        dice = dice_coef(pred, true, th)
        iou = IoU(pred, true, th)
        h = h_score(pred, true, th)
        score = self.dice_wight * dice + self.iou_weight * iou + self.h_weight * h
        return score
