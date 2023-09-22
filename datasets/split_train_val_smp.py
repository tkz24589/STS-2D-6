import os
import random
import cv2

def split_train_val(data_root, data_list, val_ratio=0.1, is_train=True, file_uint='.jpg'):
    train_data_list = []
    valid_data_list = []
    for index in data_list:
        if is_train:
            type = f'train/{index}'
            ids = os.listdir(data_root + type + '/image/')
        else:
            type = f'{index}'
            ids = os.listdir(data_root + type + '/image/')
        ids = [type + '/image/' + i[:-4] for i in ids]
        if index == 'unlabelled':
            data_type = 0
        else:
            data_type = 1
        if is_train:
            random.shuffle(ids)
            val_len = int(len(ids) * val_ratio)
            train_ids = ids[:-val_len]

            for i in ids:
                if i in train_ids:
                    train_data_list.append({i:data_type})
                else:
                    valid_data_list.append({i:data_type})
    path = data_root + ids[0] + file_uint
    shape = cv2.imread(path).shape
    if is_train:
        return train_data_list, valid_data_list, shape
    else:
        return ids, None, shape

