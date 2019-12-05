import collections
import os
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import cv2 as cv
import random

class VOCClassSegBase(data.Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    mean = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self, root, split='train', transform=True):
        self.root = root
        self.split = split
        self._transform = transform
        data_pth = os.path.join(self.root, 'VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split_file in ['train', 'val']:
            file_name = os.path.join(data_pth, 'ImageSets/Segmentation/%s.txt' % split_file)
            for img_name in file_name:
                img_name = img_name.strip()
                img_file = os.path.join(data_pth, 'JPEGImages/%s.jpg' % img_name)
                lbl_file = os.path.join(data_pth, 'SegmentationClass/%s.png' % img_name)
                self.files[split_file].append({
                    'img': img_file,
                    'lbl': lbl_file
                })
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl[lbl == 255] = 0
        img, lbl = self.randomFlip(img, lbl)
        img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
    def transform(self, img, lbl):
        img = img[:, :, ::-1]    # RGB - > BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.transpose(2, 0, 1)  # whc -> cwh
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(img).long()
        return img, lbl

    def resize(self, img, lbl, s=320):
        img = cv.resize(img, (s, s), interpolation=cv.INTER_LINEAR)
        label = cv.resize(img, (s, s), interpolation=cv.INTER_NEAREST)
        return img, label

    def randomFlip(self, img, lbl):
        if random.random() < 0.5:
            img = np.fliplr(img)
            lbl = np.fliplr(lbl)
        return img, lbl

    def randomCrop(self, img, lbl):
        h, w, _ = img.shape
        shorter = min(h, w)
        rand_s = random.randrange(int(0.7 * shorter), shorter)
        x = random.randrange(0, w - rand_s)
        y = random.randrange(0, h - rand_s)
        return img[y:y + rand_s, x:x + rand_s], lbl[y:y + rand_s, x:x + rand_s]

    def augmentation(self, img, lbl):
        img, lbl = self.randomFlip(img, lbl)
        img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)
        return img, lbl

class VOC2012ClassSeg(VOCClassSegBase):
    def __init__(self, root, split='train', transform=False):
        super(VOCClassSegBase, self).__init__(root, split=split, transform=transform)
