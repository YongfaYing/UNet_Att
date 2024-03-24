import tifffile as tiff
import torch
import numpy as np
from skimage import io
from skimage import util
import os

path_dir = r'./datasets/poissin' # input poissin images path
path_list = os.listdir(path_dir)
file_list = []
for file_name in path_list:
    if os.path.splitext(file_name)[1] == '.tif':
        file_list.append(file_name)
file_list.sort()

output_dir = path_dir + '/gaussian'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file in file_list:
    path_file = path_dir + '/' + file
    img = tiff.imread(path_file)

    noise_img = util.random_noise(img,mode='gaussian',var=0.0625)
    noise_img = noise_img*65535
    noise_img=noise_img.astype('uint16')

    noise_name = output_dir + '/' + file
    io.imsave(noise_name, noise_img)