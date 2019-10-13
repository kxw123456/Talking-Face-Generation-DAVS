# encoding:UTF-8
"""
    program to expend the LRW dataset
    seq 0 499 | parallel --eta -j 24 python process256_224.py --word_label {}
"""
from __future__ import division
import cv2
import os
import time
import csv
import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser()

# word_label 指明当前处理词列表中的哪一个词 范围（0,499）
parser.add_argument('--word_label', type=int, default=0, help='number of threads')   
opt = parser.parse_args()

class Config(object):
    def __init__(self):
        self.main_PATH = "/home/hzhou/data/LRW/lipread_mp4/"   #LRW数据集路径
        self.save_PATH = "/home/hzhou/SSD/new_data2/"          #处理后数据保存路径
        self.image_size = 224
        self.save_size = 228
        self.video_length = 25
        self.avg_three_points = np.array([[101.19, 97.79],
                                     [155.24, 97.53],
                                     [126.75, 127.71]])
        self.eye_avg = (self.avg_three_points[0, :] + self.avg_three_points[1, :]) / 2	# ([128.215,  97.66 ])
        self.orig_resize = int(256 * self.save_size / (self.eye_avg[1] * 2 + 10))	# 284
        self.top = 5
        self.bottom = int(self.eye_avg[1] * 2 + 15)			 # 210
        self.left = int(self.eye_avg[0] - self.eye_avg[1] - 5)		 # 25
        self.right = int(self.eye_avg[0] + self.eye_avg[1] + 5)          # 230
        self.scale = float(self.save_size / (self.eye_avg[1] * 2 + 10))  # 1.1104617182933958

config = Config()
print(config.scale)

def create_video_folders(main_PATH, name_id):
    """Creates a directory for each label name in the dataset."""
    name_id_path = os.path.join(main_PATH, name_id)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path


def find_three_points(face_points):         
    ''' face_points 指的是什么？   face_points'''
    three_points = np.zeros((5, 2))
    three_points[0, :] = face_points[74, :]
    three_points[1, :] = face_points[77, :]
    three_points[2, :] = face_points[74, :]
    three_points[3, :] = face_points[77, :]
    three_points[4, :] = face_points[46, :]
    return three_points


class ImageLoader256(object):
    def __init__(self, face_points, mode='train'):
	'''
		face_points: 
			类型：np.array
			维度：（n,2）    n ： 106
	''' 
        self.scale = 2
        self.face_points = face_points
        self.three_points = np.zeros((5, 2))
        self.crop_height = 134 * self.scale     # 134*2 = 268
        self.crop_width = 134 * self.scale
        self.crop_center_y_offset = 10 * self.scale    #10 * 2 = 20
        self.output_scale = (260, 260)
        self.ori_scale = (178 * self.scale, 218 * self.scale)   # 178*2 = 356    218*2 = 436
        if mode == 'train':
            self.flip = np.random.randint(0, 2)          # 随机翻转方向
            self.random_x = np.random.randint(-3, 4)     # 
            self.random_y = np.random.randint(-3, 4)     #

    def image_loader(self, img):
        self.find_three_points()
        self.M = self.transformation_from_points(self.three_points, self.scale)
	
        # 对img做仿射变换，  边界填充 [127 127 127]
        align_img = cv2.warpAffine(img, self.M, self.ori_scale, borderValue=[127, 127, 127])   

        self.l = int(round(self.ori_scale[0] / 2 - self.crop_width / 2))    # 44
        self.r = int(round(self.ori_scale[0] / 2 + self.crop_width / 2))    # 312
        self.t = int(round(self.ori_scale[1] / 2 - self.crop_height / 2 + self.crop_center_y_offset))   # 104
        self.d = int(round(self.ori_scale[1] / 2 + self.crop_height / 2 + self.crop_center_y_offset))   # 372
        align_img2 = align_img[self.t:self.d, self.l:self.r, :]     # [104:372, 44:312]
        # cv2.imwrite('/home/wyang/temp/align_img2.jpg', align_img2)
        align_img2 = cv2.resize(align_img2, self.output_scale)      # [260,260,3]
        self.compute_transpoints()

        return align_img2



    def transformation_from_points(self, points1, scale):
        points = [[70, 112],
                  [110, 112],
                  [70, 112],
                  [110, 112],
                  [90,  150]]                    # points 意义？    等腰三角形
        points2 = np.array(points) * scale       # 尺度放大   
        points2 = points2.astype(np.float64)        # (5,2)
        points1 = points1.astype(np.float64)        # (5,2)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2               # -均值  /标准差   

        U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))      
        R = (np.matmul(U, Vt)).T
        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
        M = np.concatenate((sR, T), axis=1)
        return M        # 维度：（2,3）

    def find_three_points(self):

        self.three_points[0, :] = self.face_points[74, :]
        self.three_points[1, :] = self.face_points[77, :]
        self.three_points[2, :] = self.face_points[74, :]
        self.three_points[3, :] = self.face_points[77, :]
        self.three_points[4, :] = (self.face_points[84, :] + self.face_points[90, :]) / 2
        self.three_points.astype(np.float32)

    def compute_transpoints(self):
        mouth_points_T = np.concatenate((self.face_points.T, np.ones((1, 106))), 0)     # 数组拼接  拼接行   维度：(3,106)
        transmouth = (np.matmul(self.M, mouth_points_T))      # 维度：（2,106）
        self.transmouth = transmouth.T      # （106,2）
        self.transmouth[:, 0] -= self.l
        self.transmouth[:, 1] -= self.t
        self.transmouth = self.transmouth * self.output_scale[0] / self.crop_height
        # return transmouth



def warp_and_save(M, frame, config=config):
    a1 = cv2.warpAffine(frame, M, (256, 256), borderValue=[127, 127, 127])    
    croped_a1 = a1[0: config.bottom, config.left: config.right]     # a1[0:210, 25:230]   大小：（210,205）
    croped_image = cv2.resize(croped_a1, dsize=(config.save_size, config.save_size))    # （228,228）
    return croped_image


def write_images(dir, num, image):
    cv2.imwrite(os.path.join(dir, str(num) + ".jpg"), image)


def save_transpoints(face_points, M, config=config):
    mouth_points_T = np.concatenate((face_points.T, np.ones((1, 106))), 0)
    transmouth = (np.matmul(M, mouth_points_T))
    transmouth = transmouth.T
    transmouth[:, 0] -= float(config.left) - 1
    transmouth[:, 1] -= 15
    transmouth = transmouth * config.scale
    return transmouth

end = time.time()

face_data_name = "_face.txt"
listnames = config.main_PATH + "LRW_list.txt"     # LRW 词列表文件
filenames = "_filenames.txt"                      # LRW数据下每个文件夹中 .mp4 文件列表
train = "train"
test = "test"
val = "val"
p_lists = {0 : "test" , 1 : "train", 2 : "val"}


if (not os.path.exists(config.save_PATH)):
    os.mkdir(config.save_PATH)
for p in range(3):                                    # create path train, val, test / image, flow

    p_name = p_lists[p]
    save_dir = create_video_folders(config.save_PATH, p_name)
    # if (not os.path.exists(hsv_dir)):
    #     os.mkdir(hsv_dir)


for p in range(3):                                   # test, train, val loop
    total_data_num = 0
    p_name = p_lists[p]                              # "test"
    # dir to save data
    video_savedir = config.save_PATH + p_lists[p]    # /home/.../data/test

    with open(listnames, 'r') as w:                  # read all class names
        word_names = w.read().splitlines()           # 词列表
    start = end
    word_label = opt.word_label                      # word_label < 500
    # for word_label in range(2):


    ABOUT_dir = create_video_folders(video_savedir, str(word_label))      # 创建0文件夹  /home/.../data/test/0
    word = word_names[word_label]                                         # 取第一个单词 ABOUT
    word_dir = config.main_PATH + word                                    # /home/data2/LRW/ABOUT
    word_dir_p = os.path.join(word_dir, p_name)                           # /home/data2/LRW/ABOUT/test
    video_names_dir = os.path.join(word_dir, p_name + filenames)          #/home/data2/LRW/ABOUT/test_filenames.txt

    with open(video_names_dir, "r") as v:            # 获取文件夹下的 .mp4 文件名列表
        video_names = v.read().splitlines()
    video_names.sort()
    vid = 0
    file_list = {}
    for video in video_names:                        # ABOUT_000xx.mp4 loop

        video_dir = os.path.join(word_dir_p, video)  # /home/data2/LRW/ABOUT/test/ABOUT_000xx.mp4

        # facep_dir 表示什么意义？ 推测：脸部特征点文件 face_points_dir
        facep_dir = video_dir[:-4] + face_data_name  # /home/data2/LRW/ABOUT/test/ABOUT_00023_face.txt
        if os.path.getsize(facep_dir) == 0:
            continue
        else:
            face_data = np.loadtxt(facep_dir)       # face_data:维度：（29*106，2）
            s2, _ = np.shape(face_data)
            if s2 != 29 * 106:                       # 每个视频29帧，106 ？     
                continue

        ABOUT_00001_dir = create_video_folders(ABOUT_dir, str(vid))     # /home/.../data/test/0/0

        # create folders for data  /home/.../data/test/0/0/align_face256
        align_face_dir256 = create_video_folders(ABOUT_00001_dir, "align_face256")

        # /home/.../data/test/0/0/flow256
        flow_dir256 = create_video_folders(ABOUT_00001_dir, "flow256")

        # save file idx
        file_list[vid] = video       # file_list[i] = AOUBT_00001.mp4

        # /home/.../data/test/0/video_look_up_table.csv  记录 pid
        w = csv.writer(open(os.path.join(ABOUT_dir, "video_look_up_table.csv"), "w"))
        for key, val in file_list.items():
            w.writerow([key, val])

        #
        prvs1 = np.zeros([config.save_size, config.save_size], dtype=int)         # flow buffer  （228，228）
        Sum_align_face_temp = np.zeros((config.save_size, config.save_size, 3))     # （228,228,3）
        Sum_flow_temp = np.zeros((config.save_size, config.save_size, 3))           # （228,228,3）
        # orig_image_state = os.path.exists(origin_image_dir + "/1.jpg")            # find if origin images have already been saved

        cropped_align_state256 = os.path.exists(align_face_dir256 + "/27.jpg")  # /home/.../data/test/0/0/align_face256/27.jpg 最后一张图片

        transpoints256 = []
        # capture the video  读取视频
        capture = cv2.VideoCapture(video_dir)
        if capture.isOpened():
            n = 0
            n2 = 0

            M_save256 = []
            for i in range(29):
                face_points = face_data[i * 106: i * 106 + 106, :]
                Align256 = ImageLoader256(face_points)

                ret, frame = capture.read()
                if ret:
		    
                    if not cropped_align_state256:     # 没有对齐
                        img256 = Align256.image_loader(frame)
                        cv2.imwrite(os.path.join(align_face_dir256, str(i) + ".jpg"), img256)       # /home/.../data/test/0/0/align_face256/i。jpg
                        face2 = cv2.cvtColor(img256, cv2.COLOR_BGR2GRAY)                  # grey scale image
                        flow = 0
                        if (n2 >= 1):
                            flow = cv2.calcOpticalFlowFarneback(prvs2, face2, 0.5, 3, 10, 5, 5, 1.1, 0)
                            flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
                            flow = flow.astype(np.uint8)
                            flow = np.concatenate((flow, np.zeros((260, 260, 1))), 2)
                            cv2.imwrite(os.path.join(flow_dir256, str(i) + ".jpg"), flow)               # save flow files
                        prvs2 = face2
                        n2 += 1
                        transface256 = Align256.transmouth
                        transpoints256.append(transface256)
                        M_save256.append(Align256.M)



            # /home/.../data/test/0/0/M256.npy
            if not os.path.exists(os.path.join(ABOUT_00001_dir, "M256.npy")):
                np.save(os.path.join(ABOUT_00001_dir, "M256.npy"), M_save256)
            # /home/.../data/test/0/0/transpoints256.npy
            if not os.path.exists(os.path.join(ABOUT_00001_dir, "transpoints256.npy")):
                np.save(os.path.join(ABOUT_00001_dir, "transpoints256.npy"), transpoints256)

            # /home/.../data/test/0/0/0.wav
            if not os.path.exists(os.path.join(ABOUT_00001_dir, str(vid) + ".wav")):
                command = ['ffmpeg -i', video_dir,
                            '-f wav -acodec pcm_s16le -ar 16000',
                            os.path.join(ABOUT_00001_dir, str(vid) + ".wav")]
                command = ' '.join(command)
                try:
                    output = subprocess.check_output(command, shell=True,
                                                     stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as err:
                    print(err)



        vid += 1

    end = time.time()
    print(end - start)
    print("converted " + word + " " + str(vid) + ".wav")


