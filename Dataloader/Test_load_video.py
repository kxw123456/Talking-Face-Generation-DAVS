# coding=utf-8
from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


'''
    path: ./0572_0019_0003/video             将要转换的视频文件，里面是100张图片
    A_path: ./demo_images/test_sample1.jpg   目标人物图片

'''
def Test_Outside_Loader(path, A_path, config, require_video=True):
    loader = {}
    video_data_length = config.test_audio_video_length          # 99
    video_pair = range(2, video_data_length)                    # [2,3,4,...,98]
    im_pth = []

    video_block = np.zeros((config.test_audio_video_length,
                            config.image_size,
                            config.image_size,
                            config.image_channel_size))         # 99    256     256     3

    crop_x = 2
    crop_y = 2
    A_image = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)   # 颜色空间转换   目标人物图片
    A_image = A_image.astype(np.float)
    A_image = A_image / 255     # 像素归一化
    A_image = cv2.resize(A_image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size],
                         (config.image_size, config.image_size))      # [2:258,2:258]     图片大小不变，x、y各平移2个像素
    if os.path.isdir(path):

        k1 = 0
   
        for image_num in video_pair:            # 2-98 循环   将 video 中的图片进行 颜色空间转换、像素归一化、裁剪。其中图片像素 260*260

            image_path = os.path.join(path, str(image_num) + '.jpg')        # ./0572_0019_0003/video/i.jpg
            im_pth.append(image_path)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = image / 255

                video_block[k1] = image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size]
            else:
                print("video_block = 0")
                break

            k1 += 1

        video_block = video_block.transpose((0, 3, 1, 2))     # [第几张图片，C，W，H]
        A_image = A_image.transpose((2, 0, 1))                # [C,W,H]
        loader['A'] = A_image           # 目标图片
        if require_video:
            loader['B'] = video_block   # 待转换视频（图片序列）
        loader['A_path'] = A_path       # ./demo_images/test_sample1.jpg   目标人物图片
        loader['B_path'] = im_pth       # ./0572_0019_0003/video/i.jpg    
    return loader


class Test_VideoFolder(Dataset):

    def __init__(self, root, A_path, config, transform=None, target_transform=None,
                 loader=Test_Outside_Loader, mode='test'):

        self.root = root
        self.A_path = A_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.config = config
        self.mode = mode
        self.vid = self.loader(self.root, self.A_path,  config=self.config)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        loader = {}

        loader['A'] = self.vid['A']
        loader['B'] = self.vid['B'][index:index + self.config.sequence_length, :, :, :]     # 待转换图片序列  6张图片 gt
        loader['A_path'] = self.A_path
        loader['B_path'] = self.vid['B_path'][index:self.config.sequence_length + index]    # 图片路径
        return loader

    def __len__(self):
        return self.config.test_audio_video_length - self.config.sequence_length + 1
