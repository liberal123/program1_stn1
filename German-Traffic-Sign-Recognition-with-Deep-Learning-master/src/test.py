import glob as glob
import albumentations
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from model import Net
matplotlib.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# load the model checkpoint
checkpoint = torch.load('D:\TSR\program1\German-Traffic-Sign-Recognition-with-Deep-Learning-master\Outputs1/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# read all image paths
root_dir = 'D:\TSR\GTSRB'

# read the test dataframe
test_df = pd.read_csv(
    r'D:\TSR\GTSRB\Test.csv',
    sep=',', nrows=10) #sep=';'这时用df.set_index会报错，可能是堆在一起了 改为sep=','就更好看了 就不会堆在一起 而且不会报错
# change index to filename for easier access to labels
print(test_df)
#    Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
# 0                   53,54,6,5,48,49,16,Test/00000.png
# 1                    42,45,5,5,36,40,1,Test/00001.png
# 2                   48,52,6,6,43,47,38,Test/00002.png
gt_df = test_df.set_index('Path', drop=True)
print(gt_df)
# read sign label dataframes

#-*- coding : utf-8 -*-
# coding: utf-8
sign_df = pd.read_csv(
        'D:\TSR\GTSRB\sign_df.csv',encoding='unicode_escape'
        )

aug = albumentations.Compose([
                # 48x48 resizing is required for this network model
                albumentations.Resize(48, 48, always_apply=True),
            ])


for i in range(len(test_df)):
    # image_path = root_dir+test_df.loc[i, 'Filename'] Path
    image_path = root_dir +'\\'+ test_df.loc[i, 'Path']
    print(image_path) #D:\TSR\GTSRB\Train\Test/00000.png
    image_path=image_path.replace('/', '\\')
    print(image_path)
    image = plt.imread(image_path)
    orig = image.copy()

    model.eval()
    with torch.no_grad():
        image = image / 255.
        image = aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
         
    # get the prediction label
    # label = sign_df.loc[int(preds), 'SignName'] sign_df应该是从每个交通标识的文件夹对应的名字的csv文件读取出来的
    label = sign_df.loc[int(preds), 'SignName']
    # get the ground truth label
    filename = test_df.loc[i, 'Path']
    print(filename)
    gt_id = gt_df.loc[filename,'ClassId']
    print(gt_id)
    gt_label = sign_df.loc[int(gt_id), 'SignName']
    # filename = image_path.split('/')[-1]
    # gt_id = gt_df.loc[filename].ClassId
    # gt_label = sign_df.loc[int(gt_id), 'SignName']

    # image = image.detach().cpu().numpy()
    # image = image.squeeze(0)
    # image = np.transpose(image, (1, 2, 0))
    # plt.imshow(image)
    # plt.title('Image that the model sees')
    # plt.show()
    
    plt.imshow(orig)
    plt.title(f"Prediction - {str(label)}\nGround Truth - {str(gt_label)}")
    plt.axis('off')
    #FileNotFoundError: [Errno 2] No such file or directory: 'D:/TSR/program1/German-Traffic-Sign-Recognition-with-Deep-Learning-master/Outputs1/Test/00000.png'
    #是因为多了一个Test文件，因为上面的代码改掉了
    plt.savefig(f"D:/TSR/program1/German-Traffic-Sign-Recognition-with-Deep-Learning-master/Outputs1/{filename.split('.')[0]}.png")
    plt.show()
    plt.close()