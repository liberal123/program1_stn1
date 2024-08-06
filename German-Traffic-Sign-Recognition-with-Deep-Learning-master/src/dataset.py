import os

import torch
import pandas as pd
import cv2
import albumentations
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

orginal_path=r'D:\TSR\GTSRB'
class GTSRDataset(Dataset):
    def __init__(self, images, labels, tfms=None):
        self.images = images
        self.labels = labels

        # apply augmentations
        if tfms == 0:  # if validating
            self.aug = albumentations.Compose([
                # 48x48 resizing is required
                albumentations.Resize(48, 48, always_apply=True),

            ])
        else:  # if training
            self.aug = albumentations.Compose([
                # 48x48 resizing is required
                albumentations.Resize(48, 48, always_apply=True),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # # image = plt.imread(self.images[index]+'.ppm')
        # print(self.images[index]) #Train/2/00002_00030_00026.png
        # # self.images[index]=self.images[index].replace('\\','/')#注意不要忘了接收 '/','\\'
        self.images[index]= self.images[index].replace('/', '\\')
        # print(self.images[index]) #Train\2\00002_00030_00026.png
        # print(type(self.images[index])) #<class 'str'>
        image_path=orginal_path+'\\'+self.images[index]
        # # image_path='r'+'"'+image_path.replace('/','\\')+'"'
        # # image_path = image_path.replace('/', '\\')
        # # image_path=os.path.abspath(image_path) #绝对路径
        # print(image_path)
        #下面这个是正确的 路径里面有空格也会错
        # image_path=r'D:\WeChat\WeChat Files\wxid_gah2faxrlhq622\FileStorage\File\eleven\GTSRB\GTSRB\Train\2\00002_00030_00026.png'
        # print(image_path)
        # print(type(image_path))
        image = plt.imread(image_path)
        # print(image)
        image = image / 255.
        image = self.aug(image=np.array(image))['image']
        # print(image)
        image = np.transpose(image, (2, 0, 1))
        label = self.labels[index] #label相当于ClassId

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }# 返回了一个字典


df = pd.read_csv(r'D:\TSR\GTSRB\Train.csv',
                 nrows=5000)

# X = df.image_path.values
# y = df.label.values
# X = df.image_path.values 可能csv文件不一样 应改为：
X = df.Path.values
y = df.ClassId.values

(xtrain, xtest, ytrain, ytest) = train_test_split(X, y,
                                                  test_size=0.10, random_state=42)
print(f"Training instances: {len(xtrain)}")
print(f"Validation instances: {len(xtest)}")

train_data = GTSRDataset(xtrain, ytrain, tfms=1)
val_data = GTSRDataset(xtest, ytest, tfms=0)

batch_size = 256
train_data_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=1,
)
val_data_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=1,
)

# visualization
# visualize = False
visualize = True
if visualize:
    for i in range(1):
        # sign_df = pd.read_csv(
        #     'D:\微信\WeChat Files\wxid_gah2faxrlhq622\FileStorage\File\eleven\GTSRB\GTSRB\Meta.csv'
        # )
        sign_df = pd.read_csv(
            r'D:\TSR\GTSRB\Train.csv'
        )
        sample = train_data[i]
        # print(sample) #一个字典
        image = sample['image']
        label = sample['label']
        image = np.array(np.transpose(image, (1, 2, 0)))
        plt.imshow(image)
        plt.title(str(sign_df.loc[int(label), 'ClassId'])) #signname改为ClassId
        plt.show()
