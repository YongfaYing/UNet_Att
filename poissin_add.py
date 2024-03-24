import tifffile as tiff
import torch
import numpy as np
from skimage import io
from skimage import util
import os

path_dir = r'D:\database\image\STORM\STORM'   #input images path
path_list = os.listdir(path_dir)
file_list = []
for file_name in path_list:
    if os.path.splitext(file_name)[1] == '.tif':
        file_list.append(file_name)
file_list.sort()

output_dir = path_dir + '/poissin'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file in file_list:
    path_file = path_dir + '/' + file
    img = tiff.imread(path_file)
    #print(img.shape, np.max(img))

    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    if img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    # Ensure image is exclusively positive
    if low_clip == -1.:
        old_max = img.max()
        img = (img + 1.) / (old_max + 1.)

    # Generating noise for each unique value in image.
    noise_img = np.random.poisson(img)
    print(noise_img.max())
    # Return image to original range if input was signed
    if low_clip == -1.:
        out = out * (old_max + 1.) - 1.
    
    print(noise_img.max())
    noise_img = noise_img.astype('uint16')

    noise_name = output_dir + '/' + file
    io.imsave(noise_name, noise_img)
