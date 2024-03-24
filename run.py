import os
import sys

flag = sys.argv[1]

###################################################################################################################################################################
# Only train, using 1 GPU and batch_size=1
if flag == 'train':
    os.system('python train.py --datasets_train noise_1_3 \
                               --datasets_target target_1 \
                               --n_epochs 100 --GPU 0 --batch_size 64 \
                               --img_h 64 --img_w 64')  

if flag == 'test':
    os.system('python test.py --denoise_model noise_1_3_deepCAD --datasets_train add_noise_cut \
                              --GPU 0 --batch_size 1')
