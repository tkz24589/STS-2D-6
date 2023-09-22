import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchvision.utils import make_grid
from utils.dataset import TSTCustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from utils.score import DHIScore
import albumentations as A
from datasets.split_train_val_smp import split_train_val
from utils.loss import DHILoss as Loss
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os
from sklearn.model_selection import KFold
from utils.scheduler import get_scheduler, scheduler_step
from argparse import ArgumentParser
from albumentations.pytorch import ToTensorV2

class CFG:
    # training related
    parser = ArgumentParser(description='e-Lab Segmentation Script')
    arg = parser.add_argument
    arg('--bs', type=int, default=32, help='batch size')
    arg('--lr', type=float, default=5e-5, help='learning rate, default is 5e-4')
    arg('--lrd', type=float, default=1e-7, help='learning rate decay (in # samples)')
    arg('--wd', type=float, default=2e-5, help='L2 penalty on the weights, default is 2e-4')

    # device related
    arg('--workers', type=int, default=8, help='# of cpu threads for data-loader')
    arg('--maxepoch', type=int, default=50, help='maximum number of training epochs')

    # data set related:
    arg('--datapath', type=str, default='datasets/', help='dataset location')
    arg('--img_size', type=int, default=640, help='image height')

    # model related
    """
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 
    'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 
    'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
    'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 
    'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 
    'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 
    'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 
    'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 
    'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 
    'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 
    'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 
    'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080',
      'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 
      'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 
      'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 
      'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 
      'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 
'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4']"""
    arg('--encoder_name', type=str, default='resnet18', help='encoder类别如上')#训练时encoder指定此处，主要支持单网络训练，jame
    arg('--model_type', type=str, default='Unet', help='decoder的类别 Unet、PSP、FPN、PAN、UnetPlusPlus')#训练时decoder指定此处，主要支持单网络训练，jame
    arg('--checkpoint', type=str, default='result/logs/tst/last/efficientnet-b5/1/checkpoint/best.pth', help='model file path')#此处在模型推理代码中重新编写，实际意义为预训练模式jame。
    arg('--pretrain', type=bool, default=False , help='load pretrain model')

    args = parser.parse_args()

    # 指定device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # =======================训练通用配置=================================================
    pretrain = args.pretrain
    checkpoint = args.checkpoint
    data_path = args.datapath
    size = args.img_size
    classes = 1
    lr = args.lr
    min_lr = args.lrd
    epochs = args.maxepoch
    batch_size = args.bs
    min_loss = np.inf
    weight_decay = args.wd
    num_workers = args.workers
    train_list = ['labelled'] # 训练文件夹名称（标注好的数据集）， datasets/ 下，当需要切换数据源是需要更换。-----改哟unlabelled
    use_amp = True # 采用自动梯度
    onnx_path = 'model.onnx' # onnx文件保存路径，根目录下
    is_crop = False # 训练是否随机裁剪
    crop_ratio = 0.25 # 随机裁剪概率
    all_best_score = 0 # 记录一轮中最好的dice分数
    best_th = 0.5 # 记录一轮训练中最好dice分数的threshold
    is_loop = True # 是否批量训练jame，通过此控制是否批量训练
    fold_num = 4 # 5-fold交叉验证

    #=============================================指定单网络训练+测试=====================================================
    encoder_name = args.encoder_name
    model_type = args.model_type # Unet、PSP、FPN、PAN、UnetPlusPlus
    decoder_channels = (256, 128, 64, 32, 16)
    encoder_depth = 5
    # 指定输出目录 ---------------------别忘记改哟
    log_dir = 'result/logs/sts/'

    #=============================================配置批量训练+测试=====================================================
    # encoder 列表
    encoder_list = ['se_resnext101_32x4d', 'efficientnet-b5', 'mobileone_s4']
    # 训练batch_size, 根据机器性能自行调整， 与decoder_list一一对应
    bs_list = [3, 4, 4]
    # decoder 列表
    decoder_list = ['Unet++']

    #=============================================训练、测试数据归一化和增强=====================================================
    train_aug_list = [
        # A.RandomCrop(height=size, width=size, p=0.5),
        A.Resize(size, size),
        # A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                        mask_fill_value=0, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(transpose_mask=True),
    ]
    test_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(transpose_mask=True),
    ]


def train_step(model, train_dataloader, criterion, optimizer, writer, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=CFG.use_amp)
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iter, (img, true_mask) in bar:
        optimizer.zero_grad()
        img = img.to(CFG.device)
        true_mask = true_mask.to(CFG.device)
        pred_mask = model(img)
        loss = criterion.do(pred_mask, true_mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch, gpu_mem=f'{mem:0.2f} GB',
                        lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}', type='train')
        epoch_loss += loss.item()
    # writer.add_scalar('Train/Loss', epoch_loss / len(train_dataloader), epoch)
    return epoch_loss / len(train_dataloader)


def valid_step(model, valid_dataloader, criterion, writer, epoch, dih_score, log_dir, fold):
    model.eval()
    model_path = f'{log_dir}/valid/{CFG.encoder_name}/{fold}/checkpoint/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    epoch_loss = 0
    best_th = 0
    best_score = 0
    socres = {}
    for th in np.arange(1, 10, 0.5) / 10:
        socres[th] = []

    bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for step, (image, true_mask, org_image) in bar:
        image = image.to(CFG.device)
        true_mask = true_mask.to(CFG.device)
        org_image = org_image.numpy()
        with torch.no_grad():
            pred_mask = model(image)
            loss = criterion.do(pred_mask, true_mask)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch, gpu_mem=f'{mem:0.2f} GB', type='valid')
        for th in np.arange(1, 10, 0.5) / 10:
            score = dih_score.score(pred_mask, true_mask, th=th).item()
            socres[th].append(score)
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(valid_dataloader)

    # 提取区域名称和对应的统计值
    scores_means = {}
    for th, score in socres.items():
        scores_mean = sum(score) / len(score)
        scores_means[th] = scores_mean
        if scores_mean > best_score:
            best_score = scores_mean
            best_th = th

    print(best_score, best_th)

    if CFG.all_best_score < best_score:
        message = 'best_th={:2f}'.format(best_th), "score up: {:2f}->{:2f}".format(CFG.all_best_score, best_score)
        print(message)
        with open(f'{model_path}result.txt', 'w+') as file:
            # 写入内容
            file.write(f'{message}\n') 
        CFG.all_best_score = best_score
        CFG.checkpoint = model_path + 'best.pth'
        torch.save({'model_state_dict': model.state_dict()},
                   CFG.checkpoint)
    # torch.save({'model_state_dict': model.state_dict()},
    #                'last.pth')
    return avg_loss

# 循环训练+测试多个模型
def train_loop():
    # encoder 列表
    encoder_list = CFG.encoder_list
    # 训练batch_size
    bs = CFG.bs_list
    # decoder 列表
    decoder_list = CFG.decoder_list

    if len(bs) != len(decoder_list):
        assert('bs_list必须和decoder_list长度相同')

    for encoder_name, batch_size in zip(encoder_list, bs):
        for decoder_name in decoder_list:
            # 对当前网络训练
            CFG.encoder_name = encoder_name
            CFG.model_type = decoder_name
            train_single(model_type=decoder_name, encoder_name=encoder_name, batch_size=batch_size)

def train_single(model_type, encoder_name, batch_size):
    
    # 每次训练一个模型， 更新过程参数
    start_epoch = 0
    CFG.best_th = 0
    CFG.all_best_score = 0
    # log_dir = f'{CFG.log_dir}{model_type}/{encoder_name}-{model_type}'
    log_dir = CFG.log_dir

    criterion = Loss(0.4, 0.3, 0.3, 0.3)
    dih_score = DHIScore()

    Fold = KFold(shuffle=True, n_splits=CFG.fold_num, random_state=42)

    # train_data_list, valid_data_list, _ = split_train_val(CFG.data_path, CFG.train_list, 0.2)
    _, data_list, _ = split_train_val(CFG.data_path, CFG.train_list, 1, file_uint='.png')
    data_fold = Fold.split(data_list)

    for fold, (train_data_index, valid_data_index) in enumerate(data_fold):
        try:
            if model_type == 'Unet':
                model = smp.Unet(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
            elif model_type == 'Linknet':
                model = smp.Linknet(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
            elif model_type == 'Unet++':
                model = smp.UnetPlusPlus(encoder_name=encoder_name, activation='sigmoid', decoder_channels=CFG.decoder_channels, encoder_depth=CFG.encoder_depth).to(CFG.device)
            elif model_type == 'MAnet':
                model = smp.MAnet(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
            elif model_type == 'FPN':
                model = smp.FPN(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
            elif model_type == 'PSPNet':
                model = smp.PSPNet(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
            elif model_type == 'PAN':
                model = smp.PAN(encoder_name=encoder_name, activation='sigmoid').to(CFG.device)
        except:
            message = f'{encoder_name}和{model_type}不匹配'
            print(message)
            return message
        
        if CFG.pretrain:
            checkpoint = torch.load(CFG.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Checkpoint load')
        optimizer = optim.AdamW(model.parameters(),
                                lr=CFG.lr,
                                betas=(0.9, 0.999),
                                weight_decay=CFG.weight_decay
                                )
        scheduler = get_scheduler(CFG, optimizer)
        writer = None
        print(f'{fold} fold')
        CFG.all_best_score = 0
        train_data_list = [data_list[i] for i in train_data_index]
        valid_data_list = [data_list[i] for i in valid_data_index]
        train_dataset = TSTCustomDataset(CFG.data_path, train_data_list, A.Compose(CFG.train_aug_list), CFG.is_crop, CFG.crop_ratio, is_train=True, is_valid=False)
        valid_dataset = TSTCustomDataset(CFG.data_path, valid_data_list, A.Compose(CFG.valid_aug_list), False, 0, is_train=True, is_valid=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)

        for epoch in range(start_epoch, CFG.epochs):
            train_step(model, train_dataloader, criterion, optimizer, writer, epoch)
            valid_step(model, valid_dataloader, criterion, writer, epoch, dih_score, log_dir, fold)
            scheduler_step(scheduler)
    return 'ok'

if __name__ == '__main__':
    if CFG.is_loop:
        train_loop()
    else:
        train_single(CFG.model_type, CFG.encoder_name, CFG.batch_size)
