import pytorch_lightning as pl
import torch
from torchvision import datasets
import torchvision.transforms as T
import os
import glob
from PIL import Image
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

def get_pathlist(root, phase='train'):
    path_target_jpg = os.path.join(root, phase, '*.jpg') #JPEGファイルのパスを取得
    path_target_png = os.path.join(root, phase, '*.png') #PNGファイルのパスを取得
    pathlist = glob.glob(path_target_jpg) + glob.glob(path_target_png) #JPEGとPNGのパスを結合
    return pathlist

path_imgs_train = get_pathlist(root='/home/ryosuke/ImageGPT', phase='train')
path_imgs_val = get_pathlist(root='/home/ryosuke/ImageGPT', phase='val')
path_imgs_test = get_pathlist(root='/home/ryosuke/ImageGPT', phase='test')

print(path_imgs_train)
print(path_imgs_val)
print(path_imgs_test)

class ImageTransform():
    def __init__(self, resize, interpolation):
        self.data_transform = {
            'train': T.Compose([
                T.Resize(resize, interpolation),
                T.ToTensor()
            ]),
            
            'val': T.Compose([
                T.Resize(resize, interpolation),
                T.ToTensor()
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

size = (128, 64)
interpolation = T.InterpolationMode.BILINEAR 
img = Image.open('/home/ryosuke/ImageGPT/train/tooth1_improve.jpg').convert("L")
transform = ImageTransform(resize = size, interpolation = interpolation)
img_train = transform(img, phase='train') #train用の変換
img_val = transform(img, phase='val') #val用の変換

print(img_train.shape, img_train.min(), img_train.max(), img_val.shape, img_val.min(), img_val.max())
print(img_train)
print(img_train.dtype)

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list #画像のパスを格納したリスト
        self.transform = transform #前処理クラスのインスタンス
        self.phase = phase #train or valの指定

    def __len__(self):
        return len(self.file_list) #画像の枚数を返す

    def __getitem__(self, index):
        #画像の前処理
        img_path = self.file_list[index] #index番目の画像のパスを取得
        img = Image.open(img_path).convert("L") #PIL形式で画像を読み込み
        img_transformed = self.transform(img, self.phase) #画像の前処理を実施
        binary_image = torch.where(img_transformed > 0.05, torch.tensor(1.0), torch.tensor(0.0))
        img_transformed = binary_image.long()
        return img_transformed

train_dataset = MyDataset(file_list=path_imgs_train,
                               transform=ImageTransform(size, interpolation),
                               phase='train')

val_dataset = MyDataset(file_list=path_imgs_val,
                               transform=ImageTransform(size, interpolation),
                               phase='val')

test_dataset = MyDataset(file_list=path_imgs_test,
                               transform=ImageTransform(size, interpolation),
                               phase='val')
                               
print(train_dataset[0])
print(train_dataset[0].dtype)
print(train_dataset[0].shape)
print(f'trainデータ数: {len(train_dataset)}, valデータ数: {len(val_dataset)}, testデータ数: {len(test_dataset)}') #データ数の確認

train_dl = DataLoader(train_dataset, batch_size= 1, shuffle = True, num_workers = 6, pin_memory = True)
val_dl = DataLoader(val_dataset, batch_size= 1, shuffle = False, num_workers = 6, pin_memory = True)
test_dl = DataLoader(test_dataset, batch_size= 1, shuffle = False, num_workers = 6, pin_memory = True)



print('train:', len(train_dl), 'val:', len(val_dl), 'test:', len(test_dl)) #minibatch数の確認
print('train:', len(train_dl.dataset), 'val:', len(val_dl.dataset), 'test:', len(test_dl.dataset)) #データ数の確認
