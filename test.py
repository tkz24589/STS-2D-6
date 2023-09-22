import torch
import numpy as np
from utils.dataset import TSTCustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from datasets.split_train_val_smp import split_train_val
import segmentation_models_pytorch as smp
import os
from albumentations.pytorch import ToTensorV2
from argparse import ArgumentParser
import torch.nn.functional as F
import cv2   
import zipfile


class CFG:
    parser = ArgumentParser(description='e-Lab Segmentation Script')
    arg = parser.add_argument
    arg('--bs', type=int, default=2, help='batch size')
    arg('--workers', type=int, default=0, help='# of cpu threads for data-loader')
    arg('--img_size', type=int, default=640, help='image height')
    arg('--datapath', type=str, default='datasets/', help='dataset location')
    arg('--encoder_name', type=str, default='mit_b3', help='encoder类别，如上')
    arg('--model_type', type=str, default='Unet++', help='decoder的类别 Unet、PSP、FPN、PAN、UnetPlusPlus')
    args = parser.parse_args()

    # 指定device
    device = torch.device('cuda')
    print(device)
    # device = torch.device('cpu')
    # =======================测试通用配置=================================================
    data_path = args.datapath
    size = args.img_size
    batch_size = args.bs
    classes = 1
    num_workers = args.workers
    valid_list = ['test'] # 文件夹名称
    chepoint_dir = 'result/logs/sts/last/' # 模型文件目录
    test_encoder_list = ['efficientnet-b5', 'mobileone_s4', 'se_resnext101_32x4d'] # 混合模型

    #=============================================指定单网络训练+测试=====================================================
    encoder_name = args.encoder_name
    model_type = args.model_type # Unet、PSP、FPN、PAN、UnetPlusPlus


    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(transpose_mask=True),
    ]

def valid_step(model, valid_dataloader, shape):
    save_path_list = []
    bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for step, (image, org_img, save_path) in bar:
        image = image.to(CFG.device)
        with torch.no_grad():
            pred_mask = model(image)
            # pred_mask = (pred_mask > 0.2).float()
            org_img_cpu = org_img.numpy()
            # pred_mask_cpu = (pred_mask_cpu > 0.2).astype(np.uint8) * 255
            for org_im, mask, path in zip(org_img_cpu, pred_mask, save_path):
                mask_cpu = F.interpolate(mask.unsqueeze(0), size=org_im.shape[:2], mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()
                mask_cpu = (mask_cpu > 0.7).astype(np.uint8) * 255
                cv2.imwrite(path, mask_cpu[0])
                save_path_list.append(path)
    return save_path_list

class EnsembleModel:
    def __init__(self):
        self.models = []

    def __call__(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        avg_preds = torch.mean(outputs, dim=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)

def build_ensemble_model():
    model = EnsembleModel()
    for encoder in CFG.test_encoder_list:
        model_list = os.listdir(CFG.chepoint_dir + encoder + '/')
        for model_dir in model_list:
            model_path = CFG.chepoint_dir + encoder + '/' + model_dir + '/checkpoint/best.pth'
            _model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None)
            _model.to(CFG.device)
            print(model_path)
            try:
                state = torch.load(model_path, map_location=CFG.device)
                _model.load_state_dict(state['model_state_dict'])
    #                 _model.load_state_dict(state)
            except:
                print('checpoint not load')
            _model.eval()
            model.add_model(_model)
    return model

#对保存的图像进行打包
def zip_files(file_paths, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file)          

def valid(batch_size):

    model = build_ensemble_model()

    valid_data_list, _, shape = split_train_val(CFG.data_path, CFG.valid_list, 1.0, is_train=False, file_uint='.png')

    valid_dataset = TSTCustomDataset(CFG.data_path, valid_data_list, A.Compose(CFG.valid_aug_list), False, None, is_train=False, is_valid=False)

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers)

    file_paths = valid_step(model, valid_dataloader, shape)

    #打包图片
    output_path = 'infers.zip'
    zip_files(file_paths, output_path)

    return 'ok'

if __name__ == '__main__':
    valid(CFG.batch_size)