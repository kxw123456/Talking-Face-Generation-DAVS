# coding=utf-8
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Options
import cv2
import shutil
config = Options.Config()


def find_classes(dir, config=config):
    # dir: /data/train
    classes = [str(d) for d in range(config.label_size)]        # label_size = 500   生成列表 字符格式
    class_to_idx = {classes[i]: i for i in range(len(classes))} # 字符格式 ---》 数字 字典
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, mode, config=config):
    '''
    arg:
        dir:/data/train   class_to_idx:字典     mode:train
    return：
        [('./data/train/0/0',0),('./data/train/0/1',0), ...
         ('./data/train/1/0, 1),('./data/train/1/1', 1), ...
        ...
         ('./data/train/499/0,499),('./data/train/499/1,499), ... ]
    '''
    videos = []         # 列表
    dir = os.path.expanduser(dir)     
    classes = [str(d) for d in range(config.label_size)]        # ['0','1','2',...,'499']    wid
    for target in classes:                      # 遍历 每个类文件   word   
        d = os.path.join(dir, target)           # /data/train/0/  

        if not os.path.isdir(d):
            continue
        listd = sorted(os.listdir(d))           # 列表  ['0','1',...]            pid
        #if mode == 'val':
            #listd = random.sample(listd, 10)
        for fnames in listd:                    # 遍历类文件中的每个文件  person
            path = os.path.join(d, fnames)      # /data/train/0/0
            if os.path.isdir(os.path.join(d, fnames)):
                if os.path.exists(path):
                    item = (path, class_to_idx[target])     # (/data/train/0/0, 0)   元组
                    videos.append(item)
    return videos 

def lip_reading_loader(path, config=config, mode='train', random_crop=True,
                       ini='fan'):
    '''
    arg:
        path: ./data/train/0/0 
    return：
        {'video':(25,3,256,256)
         'mfcc20': (25,1,20,12)
         'A_path': 第一张图片的路径 ./data/train/0/0/align_face256/2.jpg
         'B_path': 其余图片的路径   ./data/train/0/0/align_face256/3~26.jpg
        }
    
    '''
    loader = {}     # 字典
    pair = np.arange(2, 27)     # ([2,3,...,26])
    im_pth = []     # 列表
    video_block = np.zeros((25,
                            config.image_size,
                            config.image_size,
                            config.image_channel_size))         # (25,256,256,3)  25张图片
    mfcc_block = np.zeros(( 25, 1,
                            config.mfcc_length,
                            config.mfcc_width,
                            ))                                  # (25,1,20,12)    25段音频特征

    if os.path.isdir(path):
        for block in (os.listdir(path)):           # align_face256, mfcc20
            block_dir = os.path.join(path, block)   # ./data/train/0/0/align_face256
            crop_x = 2
            crop_y = 2
            if mode == 'train':
                flip = np.random.randint(0, 2)      # 随机 0,1   决定是否翻转图片
                if random_crop:
                    crop_x = np.random.randint(0, 5)    # 随机 0,1,2,3,4
                    crop_y = np.random.randint(0, 5)
            else:
                flip = 0

            if os.path.isdir(block_dir):
                if block == config.image_block_name:        # image_block_name: align_face256
                    k1 = 0
                    for image_num in pair:                  # pair: ([2,3,...,26])    抽取 第 2-26 帧的图片，共25张
                        image_path = os.path.join(block_dir, str(image_num) + '.jpg')       # ./data/train/0/0/align_face256/2.jpg
                        im_pth.append(image_path)   
                        if os.path.exists(image_path):
                            image = cv2.imread(image_path)
                            if flip == 1:
                                image = np.fliplr(image)

                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            if ini == 'fan':
                                image = image / 255

                            video_block[k1] = image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size]
                        else:
                            print("video_block = 0")
                            shutil.rmtree(path)
                            break
                        k1 += 1

                if block == 'mfcc20':
                    if config.require_audio:
                        k4 = 0
                        for mfcc_num in pair:                # pair: ([2,3,...,26])    抽取 第 2-26 音频文件，共25个
                            # for s in range(-1,2):
                            mfcc_path = os.path.join(block_dir, str(mfcc_num) + '.bin')     # ./data/train/0/0/mfcc20/2.bin
                            if os.path.exists(mfcc_path):
                                mfcc = np.fromfile(mfcc_path)
                                mfcc = mfcc.reshape(20, 12)
                                mfcc_block[k4, 0, :, :] = mfcc

                                k4 += 1
                            else:
                                raise ("mfccs = 0")

        video_block = video_block.transpose((0, 3, 1, 2))       # (B,H,W,C) -> (B,C,H,W)
        loader['video'] = video_block       # 25 张图片
        loader['mfcc20'] = mfcc_block       # 25 个音频特征
        loader['A_path'] = im_pth[0]        # 第一张图片的路径 ./data/train/0/0/align_face256/2.jpg
        loader['B_path'] = im_pth[1:]       # 其余图片的路径   ./data/train/0/0/align_face256/3~26.jpg
        # loader['label_map'] = label_map_block[:, 1:, :, :]
        if not np.abs(np.mean(mfcc_block)) < 1e5:
            print(np.mean(mfcc_block))
            print(im_pth)
            shutil.rmtree(path)
    return loader


class VideoFolder(Dataset):
    '''
        root:./data/train
        mode: train/val/test
    '''
    def __init__(self, root, config=config, transform=None, target_transform=None,
                 loader=lip_reading_loader, mode='train'):
        classes, class_to_idx = find_classes(root)
        videos = make_dataset(root, class_to_idx, mode)
        if len(videos) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        self.root = root        # ./data/train
        self.vids = videos      # 元组列表
        self.classes = classes  # 字符串列表
        self.class_to_idx = class_to_idx    # 数字列表
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        
        self.config = config
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.vids[index]
        vid = self.loader(path, config=self.config, mode=self.mode)

        return vid, target

    def __len__(self):
        return len(self.vids)

