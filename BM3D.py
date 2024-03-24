# -*- coding: utf-8 -*-
import tifffile as tiff
import cv2
import numpy as np
import math
from bm3d import bm3d
import skimage.metrics
from skimage import io

# 生成示例三维图像数据（大小为 32x32x32）
# data = np.random.randn(32, 32, 32)

gt = tiff.imread(r'./datasets/test_1_gt/1907005-4-003c1.tif')
gt_img = gt[300:1324, 1500:2524]

vars = [0.01, 0.0225, 0.04, 0.0625, 0.09]
for v in vars:
	path = './datasets/add_noise_cut/cut_' + str(v) + '.tif'
	image = tiff.imread(path)



	img = image
	# img = np.array(img, dtype=np.float32).transpose(1,2,0)
	print(img.shape)
	# 添加高斯白噪声
	# noise = np.random.randn(32, 32, 32) * 0.1
	# noisy_data = data + noise

	# 设置BM3D参数
	sigma = image.std()  # 噪声标准差
	block_size = 8  # 区块大小
	group_size = 16  # 组大小

	# 对数据进行BM3D处理
	# denoised_img = bm3d(img, sigma, block_size, group_size)
	denoised_img = bm3d(img, sigma)
	print(denoised_img.shape)
	print(denoised_img.max())

	# 打印处理前后的信噪比
	# print("处理前的信噪比：", 20 * np.log10(np.max(data)) - 20 * np.log10(np.linalg.norm(data - noisy_data)))
	# print("处理后的信噪比：", 20 * np.log10(np.max(data)) - 20 * np.log10(np.linalg.norm(data - denoised_data)))




	psnr = skimage.metrics.peak_signal_noise_ratio(
			gt_img.astype(np.float), denoised_img.astype(np.float), data_range=255)
	print(psnr)


	out_name='./results/BM3D/cut_bm3d_' +str(v) + '.tif'
	io.imsave(out_name, denoised_img)



# import numpy as np
# from bm3d import bm3d

# # 生成示例三维图像数据（大小为 32x32x32）
# data = np.random.randn(32, 32, 32)

# # 添加高斯白噪声
# noise = np.random.randn(32, 32, 32) * 0.1
# noisy_data = data + noise

# # 设置BM3D参数
# sigma = 0.1  # 噪声标准差
# block_size = 8  # 区块大小
# group_size = 16  # 组大小

# # 对数据进行BM3D处理
# denoised_data = bm3d(noisy_data, sigma, block_size, group_size)

# # 打印处理前后的信噪比
# print("处理前的信噪比：", 20 * np.log10(np.max(data)) - 20 * np.log10(np.linalg.norm(data - noisy_data)))
# print("处理后的信噪比：", 20 * np.log10(np.max(data)) - 20 * np.log10(np.linalg.norm(data - denoised_data)))
