import skimage.metrics
import numpy as np
import tifffile as tiff


path1='./datasets/test_1_gt/1907005-4-003c1.tif'
img1 = tiff.imread(path1)
img1 = img1[300:1324, 1500:2524]    #cut


#path2='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'

# path2='./datasets/black.tif'
# img2 = tiff.imread(path2)

# psnr = skimage.metrics.peak_signal_noise_ratio(
#             img1.astype(np.float), img2.astype(np.float), data_range=255)
# print(psnr)   #deepCAD   #n2n  #attention_unet++

print('au_crg')

#vars = [0.0025, 0.01, 0.0225, 0.04]
#vars = [0.0625, 0.09, 0.16, 0.25]
vars = [0.0225, 0.04, 0.0625, 0.09]
for v in vars:
#     path2='./datasets/add_noise_cut/cut_' + str(v) + '.tif'
#     img2 = tiff.imread(path2)

#     psnr = skimage.metrics.peak_signal_noise_ratio(
#                 img1.astype(np.float), img2.astype(np.float), data_range=255)
#     print(psnr)

    print('var:', v)
    for i in range(1,101):
        if i<10:
            path2='./results/DataFolderIs_add_noise_cut_noise_1_3_au_crg/'+'E_0'+str(i)+'_Iter_2042/cut_' + str(v) + '_E_0'+str(i)+'_Iter_2042_output.tif'
        else:
            path2='./results/DataFolderIs_add_noise_cut_noise_1_3_au_crg/'+'E_'+str(i)+'_Iter_2042/cut_' + str(v) + '_E_'+str(i)+'_Iter_2042_output.tif'
    
        #img=img*255/img_max
        outimg=tiff.imread(path2)
        #outimg=outimg[14:256,:,:]

        psnr = skimage.metrics.peak_signal_noise_ratio(
                img1.astype(np.float), outimg.astype(np.float), data_range=255)
        #psnr_all.append(psnr)

        print(psnr)




# for i in range(1,301):
#     if i<10:
#         path2='./results/DataFolderIs_cut_au_lr/'+'E_0'+str(i)+'_Iter_2042/cut_E_0'+str(i)+'_Iter_2042_output.tif'
#     else:
#         path2='./results/DataFolderIs_cut_au_lr/'+'E_'+str(i)+'_Iter_2042/cut_E_'+str(i)+'_Iter_2042_output.tif'
