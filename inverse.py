import cv2
import os
import numpy as np

file_path="/media/8TB/mw/datasets/cocoinpainting/cocomask_modified"
save_path="/media/8TB/mw/datasets/cocoinpainting/newcocomask"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

pathname=os.listdir(file_path)
for i in range(0,len(pathname)):
    path=os.path.join(file_path,pathname[i])
    mask=cv2.imread(path,1)
    res=rgb2gray(mask)
    cv2.imwrite(os.path.join(save_path,pathname[i]),res)
    print(i,pathname[i])