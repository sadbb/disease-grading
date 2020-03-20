import os
from torch.utils import data
from torchvision import transforms as T 
from PIL import Image
import torch as t 
import csv
from random import shuffle, sample
import numpy as np 
import random
import math


class Eye_img(data.Dataset):
    def __init__(self, train=True, test=False,w=None):
        self.test = test
        self.path = 'I:/kaggle/'
        self.imgs = []
        self.imgs_val = []
        img_list = [[[] for _ in range(2)] for _ in range(5)]
        # w_val = [500,45,100,20,20]
        w_val = [30 for _ in range(5)]
        w = [300 for _ in range(5)]
        if test:
            csv_file = csv.reader(open(self.path+'retinopathy_solution.csv','r'))
            flag = 1
            for line in csv_file:
                if flag:
                    flag = 0
                    continue
                if line[2] == 'Public':
                    self.imgs.append([self.path+'test/{}.jpeg'.format(line[0]),int(line[1])])
            # for img in os.listdir(self.path+'test'):
            #     self.imgs.append([self.path+'test/'+img, 0])
        else:
            csv_file = csv.reader(open(self.path+'trainLabels.csv','r'))
            flag = 0
            for line in csv_file:
                if flag == 0:
                    flag = 1
                    continue
                if 'left' in line[0]:
                    img_list[int(line[1])][0].append(line[0])
                else:
                    img_list[int(line[1])][1].append(line[0])
            
            img_num = [[0,0] for _ in range(5)]
            for i in range(5):
                img_num[i][0] = len(img_list[i][0])
                img_num[i][1] = len(img_list[i][1])
            # print(img_num)
            if train:
                for i in range(5):
                    for j in range(2):
                        for img in sample(img_list[i][j][:-w_val[i]],w[i]):
                            self.imgs.append([self.path+'train/{}.jpeg'.format(img),i])
                shuffle(self.imgs)
            else:
                for i in range(5):
                    for j in range(2):
                        for img in img_list[i][j][-w_val[i]:]:
                            self.imgs.append([self.path+'train/{}.jpeg'.format(img),i])

            # self.imgs = sorted(self.imgs)
            # self.imglen = len(self.imgs)
            # if train:
            #     # self.imgs = self.imgs[:int(0.8*self.imglen)]
            #     pass
            # else:
            #     # self.imgs = self.imgs[int(0.8*self.imglen):]
            #     self.imgs = self.imgs_val
        nor = T.Normalize(
            mean=[.426, .298, .213],
            std=[.277, .203, .169]
        )
        if test or not train:
            self.transforms = T.Compose([
                cutimg(1246),
                # T.Resize(448),
                # T.CenterCrop(448),
                # T.ToTensor(),
                # nor
            ])
        else:
            self.transforms = T.Compose([
                cutimg(1246),
                # T.Resize(480),
                # T.CenterCrop(480),
                # T.RandomResizedCrop(448),
                T.RandomHorizontalFlip(),
                randomRotation(np.random.randint(1, 360)),
                # T.ColorJitter(0.25,0.25,0.25,0),
                # T.ToTensor(),
                # nor
            ])
        self.resize = T.Resize(512)
        self.transforms_2 = T.Compose([
            T.ToTensor(),
            nor
        ])
        # print(self.len)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = img[1]
        img_ = Image.open(img[0])
        img0 = self.transforms_2(self.transforms(img_))
        img1 = self.transforms_2(self.resize(self.transforms(img_)))
        return img0, img1, label

    def __len__(self):
        return len(self.imgs)


class cutimg(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        img_ = np.array(img)
        self.bbox = self.getbbox(img_)
        img = img.crop(self.bbox)
        return img.resize((self.img_size,self.img_size))

    def __repr__(self):
        return self.__class__.__name__ + '(bbox={})'.format(self.bbox)

    def getbbox(self,img):
        h, w, _ = img.shape
        line = 5000
        x = 10
        left, upper, right, lower = 0,0,0,0  
        for i in range(w):
            if img[:,i,0].sum()>line:
                left = max(i-x,0)
                break
        for i in range(w-1,-1,-1):
            if img[:,i,0].sum()>line:
                right = min(i+x,w-1)
                break
        for i in range(h):
            if img[i,:,0].sum()>line:
                upper = max(i-x,0)
                break
        for i in range(h-1,-1,-1):
            if img[i,:,0].sum()>line:
                lower = min(i+x,h-1)
                break
        return left, upper, right, lower


class randomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return img.rotate(self.angle)

    def __repr__(self):
        return self.__class__.__name__ + '(angle={})'.format(self.angle)


def oversample(data,k):
    if len(data) >= k:
        return sample(data,k)
    else:
        turn = math.ceil(k*1.0/len(data))
        sam = []
        for i in range(turn-1):
            sam.extend(data)
        sam.extend(sample(data,k % len(data)))
        return sam
        





