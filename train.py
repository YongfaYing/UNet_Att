import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from numpy import *
import csv
from network import Network_3D_Unet
from data_process import train_preprocess_lessMemoryMulStacks, shuffle_datasets_lessMemory, trainset
from utils import save_yaml

#############################################################################################################################################
#argparse是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
parser = argparse.ArgumentParser() 
#add_argument()添加参数  
parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation")

parser.add_argument('--batch_size', type=int, default=2, help="batch size")
parser.add_argument('--img_w', type=int, default=150, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=150, help="the height of image sequence")


parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')  #原0.00005
parser.add_argument("--b1", type=float, default=0.9, help="Adam: bata1")    #一阶矩估计的指数衰减率
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")  #二阶矩估计的指数衰减率
parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_train', type=str, default='train', help="A folder containing files for training")
parser.add_argument('--datasets_target', type=str, default='target', help="A folder containing files for training")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--select_img_num', type=int, default=1000000, help='select the number of images used for training')
parser.add_argument('--train_datasets_size', type=int, default=4000, help='datasets size for training')  #可以理解为小片段的总个数
opt = parser.parse_args()  #parse_args() 解析添加的参数

# default image gap is 0.75*image_dim
#gap_存在的意义是图像切割的间隔，*0.75的话是重叠25%，*0.5的话是重叠50%
opt.gap_w=int(opt.img_w*0.75)
opt.gap_h=int(opt.img_h*0.75)
opt.ngpu=str(opt.GPU).count(',')+1    #通过计算GPU索引中','的个数，得到GPU的个数
print('\033[1;31mTraining parameters -----> \033[0m')  #什么意思？
print(opt)

########################################################################################################################
if not os.path.exists(opt.output_dir):   #判断输出路径是否存在
    os.mkdir(opt.output_dir)   #创建目录
#下面这几行代码是保存训练过程文件用的，每周期保存一个模型
current_time = opt.datasets_train+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")   #文件夹名称，比如“DataForPytorch_202110282017”
output_path = opt.output_dir + '/' + current_time  #输出结果路径
pth_path = 'pth//'+ current_time   #每周期模型保存路径
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

yaml_name = pth_path+'//para.yaml'
save_yaml(opt, yaml_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)  #运行环境
batch_size = opt.batch_size
lr = opt.lr

#name_list是分割好的每个小片段的名称，noise_im是截取需要的帧后读取到的所有的像素数据，coordinate_list是一个字典，里面是每个小片段的名称，以及每个维度开始与结束的位置信息
name_list, noise_img, target_img, coordinate_list = train_preprocess_lessMemoryMulStacks(opt)
# print('name_list -----> ',name_list)

########################################################################################################################
L1_pixelwise = torch.nn.L1Loss()   #平均绝对误差，即L1范数
L2_pixelwise = torch.nn.MSELoss()  #均方损失函数，即L2范数

denoise_generator = Network_3D_Unet(in_channels = 1, out_channels = 1, f_maps=opt.fmap, final_sigmoid = True)   
#denoise_generator = Network_3D_Unet3(in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=False)

if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()
########################################################################################################################
optimizer_G = torch.optim.Adam(denoise_generator.parameters(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.99999)   #指数衰减调整学习率

########################################################################################################################
#判断使用cuda还是cpu
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()

########################################################################################################################
time_start=time.time()
black_i = 0
# start training
Total_Loss_mean=list(range(0,opt.n_epochs))   #新增，为了方便可视化loss
L1_Loss_mean=list(range(0,opt.n_epochs))   #新增，为了方便可视化loss
L2_Loss_mean=list(range(0,opt.n_epochs))   #新增，为了方便可视化loss
for epoch in range(0, opt.n_epochs):
    #print(optimizer_G.param_groups[0]['lr'])  #打印学习率
    name_list = shuffle_datasets_lessMemory(name_list)   #将name_list的顺序打乱
    train_data = trainset(name_list, coordinate_list, noise_img, target_img)  #获得每个小段的训练数据
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for iteration, (input, target) in enumerate(trainloader):   #这里才真正运行类trainset中的__getitem__()
        if torch.equal(target[0],torch.zeros(target[0].shape)) and (iteration + 1) % (len(trainloader)) != 0:  #判断全黑的区域
            black_i += 1
            print(black_i)
            continue
        input=input.cuda()
        target = target.cuda()
        real_A=input
        real_B=target
        
        real_A = Variable(real_A)
        #print('real_A shape -----> ', real_A.shape)
        #print('real_B shape -----> ',real_B.shape)
        fake_B = denoise_generator(real_A)   #UNet训练后的结果
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)
        ################################################################################################################
        optimizer_G.zero_grad()  #梯度置零
        # Total loss
        #Total_loss =  L2_loss
        Total_loss =  0.5*L1_loss + 0.5*L2_loss    #真的合理吗？L1_loss只有几百，而L2_loss有十几万，各乘0.5的话，不就是L2主导吗
        Total_loss.backward()   #反向传播
        optimizer_G.step()
        scheduler.step()  #学习率更新
        ################################################################################################################
        batches_done = epoch * len(trainloader) + iteration   #已经完成多少批了
        batches_left = opt.n_epochs * len(trainloader) - batches_done   #还剩多少批
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))  #还有多少时间可以全部完成
        prev_time = time.time()   #time.time() - prev_time  就是上一批量所用的时间

        Total_loss_=[]   #新增
        L1_loss_=[]     #新增
        L2_loss_=[]     #新增
        if iteration%1 == 0:
            time_end=time.time()
            print('\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] [ETA: %s] [Time cost: %.2d s]        ' 
            % (
                epoch+1,
                opt.n_epochs,
                iteration+1,
                len(trainloader),
                Total_loss.item(),
                L1_loss.item(),
                L2_loss.item(),
                time_left,
                time_end-time_start  #已经运行的时间
            ), end=' ')              #程序运行时，窗口输出的东西
            Total_loss_.append(Total_loss.item())   #新增
            L1_loss_.append(L1_loss.item())         #新增
            L2_loss_.append(L2_loss.item())         #新增
            #新增，保存每批次loss
            rows_=zip(Total_loss_,L1_loss_,L2_loss_)
            loss_path=pth_path+'//loss_detail.csv'
            with open(loss_path,'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                for row_ in rows_:
                    writer.writerow(row_)

        if (iteration+1)%len(trainloader) == 0:
            print('\n', end=' ')
            
        ################################################################################################################
        # save model  每周期保存一次模型
        if (iteration + 1) % (len(trainloader)) == 0:    #本epoch结束
            model_save_name = pth_path + '//E_' + str(epoch+1).zfill(2) + '_Iter_' + str(iteration+1).zfill(4) + '.pth'
            if isinstance(denoise_generator, nn.DataParallel): 
                torch.save(denoise_generator.module.state_dict(), model_save_name)  # parallel
            else:
                torch.save(denoise_generator.state_dict(), model_save_name)         # not parallel
            Total_Loss_mean[epoch]=mean(Total_loss_)    #新增
            L1_Loss_mean[epoch]=mean(L1_loss_)          #新增
            L2_Loss_mean[epoch]=mean(L2_loss_)          #新增
            
#新增，保存每周期loss
rows=zip(Total_Loss_mean,L1_Loss_mean,L2_Loss_mean)
loss_path_=pth_path+'//loss.csv'
with open(loss_path_,'w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    for row in rows:
        writer.writerow(row)
