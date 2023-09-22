# Copyright (c) OpenMMLab. All rights reserved.
import functools
import numpy as np
import torch.nn.functional as F


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

task_name = {
    "01": "Task01_BrainTumour",
    "02": "Task02_Heart",
    "03": "Task03_Liver",
    "04": "Task04_Hippocampus",
    "05": "Task05_Prostate",
    "06": "Task06_Lung",
    "07": "Task07_Pancreas",
    "08": "Task08_HepaticVessel",
    "09": "Task09_Spleen",
    "10": "Task10_Colon",
}

patch_size = {
    "01": [128, 128, 128],
    "02": [160, 192, 80],
    "03": [128, 128, 128],
    "04": [40, 56, 40],
    "05": [320, 256],
    "06": [192, 160, 80],
    "07": [224, 224, 40],
    "08": [192, 192, 64],
    "09": [192, 160, 64],
    "10": [192, 160, 56],
}

spacing = {
    "01": [1.0, 1.0, 1.0],
    "02": [1.25, 1.25, 1.37],
    "03": [0.77, 0.77, 1],
    "04": [1.0, 1.0, 1.0],
    "05": [0.62, 0.62],
    "06": [0.79, 0.79, 1.24],
    "07": [0.8, 0.8, 2.5],
    "08": [0.8, 0.8, 1.5],
    "09": [0.79, 0.79, 1.6],
    "10": [0.78, 0.78, 3],
}

clip_values = {
    "01": [0, 0],
    "02": [0, 0],
    "03": [-17, 201],
    "04": [0, 0],
    "05": [0, 0],
    "06": [-1024, 325],
    "07": [-96, 215],
    "08": [-3, 243],
    "09": [-41, 176],
    "10": [-30, 165.82],
}

normalize_values = {
    "01": [0, 0],
    "02": [0, 0],
    "03": [99.40, 39.36],
    "04": [0, 0],
    "05": [0, 0],
    "06": [-158.58, 324.7],
    "07": [77.99, 75.4],
    "08": [104.37, 52.62],
    "09": [99.29, 39.47],
    "10": [62.18, 32.65],
}

data_loader_params = {
    "01": {"batch_size": 8},
    "02": {"batch_size": 2},
    "03": {"batch_size": 8},
    "04": {"batch_size": 9},
    "05": {"batch_size": 2},
    "06": {"batch_size": 2},
    "07": {"batch_size": 2},
    "08": {"batch_size": 2},
    "09": {"batch_size": 2},
    "10": {"batch_size": 2},
}

deep_supr_num = {
    "01": 3,
    "02": 3,
    "03": 3,
    "04": 1,
    "05": 4,
    "06": 3,
    "07": 3,
    "08": 3,
    "09": 3,
    "10": 3,
}

def get_kernels_strides(task_id):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size[task_id], spacing[task_id]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides
