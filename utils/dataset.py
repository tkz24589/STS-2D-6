import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
        
class TSTCustomDataset(Dataset):
    def __init__(self, data_root, data_list, transform, is_crop, crop_ratio, is_train, is_valid):
        super(TSTCustomDataset, self).__init__()
        self.transform = transform
        self.data_list = data_list
        self.data_root = data_root
        self.is_train = is_train
        self.is_valid = is_valid
        # [380:380+224, 640:640+224, :]
        self.selection = None
        self.is_crop = is_crop
        self.crop_ratio = crop_ratio
        if is_train:
            self.type = '/train/'
        else:
            self.type = '/test/'

    def __len__(self):
        return len(self.data_list)
    
    def get_randon_cnter_crop(self, center, shape, min_size):
        max_width = shape[1]
        max_height = shape[0]
        max_size = min(min(max_width - center[0], center[0]), min(max_height - center[1], center[1])) * 2
        # 生成逐渐增加的边框大小
        size = random.randint(min(min_size, max_size), max_size)
        x = center[0] - size // 2
        y = center[1] - size // 2
        self.selection = [y, y+size, x ,x+size]


    def __getitem__(self, index):
        if self.is_train:
            img_path = self.data_root + list(self.data_list[index].keys())[0] + '.png'
            label_type = list(self.data_list[index].values())[0]
        else:
            img_path = self.data_root + self.data_list[index] + '.png'
        org_image = cv2.imread(img_path)
        if self.is_train:
            org_label = (cv2.imread(img_path.replace('image', 'label'), 0) > 0).astype('uint8')
            if label_type == 0:
                org_label = (org_label * 0.9) + 0.05
            label = org_label[:, :, np.newaxis]
            crop_ratio = np.random.rand()
            if crop_ratio > self.crop_ratio:
                if self.is_crop:
                    # 获取白色像素的位置
                    white_pixels = np.where(org_label == 1.0)
                    if len(white_pixels[0]) != 0 and len(white_pixels[1]) != 0:
                        # 计算几何中心坐标
                        center_x = np.mean(white_pixels[1])
                        center_y = np.mean(white_pixels[0])
                        min_x = np.min(white_pixels[1])
                        min_y = np.min(white_pixels[0])
                        max_x = np.max(white_pixels[1])
                        max_y = np.max(white_pixels[0])
                        min_size = max(max_x - min_x, max_y - min_y)
                        center = (int(center_x), int(center_y))
                        #     print("中心点坐标：", center)
                        # else:
                        #     print("未找到有效的轮廓")
                        self.get_randon_cnter_crop(center, label.shape, min_size)
                        org_image = org_image[self.selection[0]:self.selection[1], self.selection[2]:self.selection[3], :]
                        label = label[self.selection[0]:self.selection[1], self.selection[2]:self.selection[3], :]
                    # cv2.imshow('img', org_image)
                    # cv2.imshow('mask', label * 255)
                    # cv2.waitKey(0)
        if self.transform:
            if self.is_train:
                data = self.transform(image=org_image, mask=label)
            else:
                data = self.transform(image=org_image)
            image = data['image']
            if self.is_train:
                label = data['mask'].float()
        if self.is_train:
            if self.is_valid:
                return image, label, org_image
            else:
                return image, label
        else:
            save_path = 'infers/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            label_name = self.data_list[index].split('image')[1] + '.png'
            return image, org_image, save_path + label_name


