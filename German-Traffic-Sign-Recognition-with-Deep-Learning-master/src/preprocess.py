import pandas as pd
import os
import numpy as np

from tqdm import tqdm

root_path = 'D:\TSR\GTSRB\Train'

# get all the image folder paths
all_paths = os.listdir(root_path)
all_paths.sort()
# print(all_paths)

# create a new dataframe 
data = pd.DataFrame()
labels = []
counter = 0
# store all images in the dataframe
for i, path in tqdm(enumerate(all_paths), total=len(all_paths)):
    # all_paths[i]=all_paths[i].replace('/', '\\')
    # print(all_paths)#['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23'...]
    all_images = os.listdir(root_path+'\\'+all_paths[i]) #路径这里加了+'\\'
    for image in all_images:
        if image.split('.')[1] == 'png':
            image_name = image.split('.')[0]
            # image_name=image_name.replace('/', '\\') image_name是00003_00016_00010这样的
            #f"{root_path}{all_paths[i]}/{image_name}" 问题在这里
            data.loc[counter, 'image_path'] = f"{root_path}{all_paths[i]}\\{image_name}"
            data.loc[counter, 'label'] = i
            counter += 1

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# save the new dataframe 只需要到SavedCsv/即可
data.to_csv('D:\TSR\program1\German-Traffic-Sign-Recognition-with-Deep-Learning-master\SavedCsv/data.csv', index=False)

print(data.head(5))