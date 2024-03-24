# images denoised of UNet-Att

### Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.7.1
* NVIDIA GPU (GeForce GTX 1060) + CUDA (10.1)
* tifffile 2020.6.3

## Description of the programs

- BM3D_api.py: BM3D Algorithm
- buildingblocks.py: Some neural network layer algorithms
- data_process.py: Preprocessing of data
- gaussian_add.py: add gaussian noise
- grid_attention_layer.py: Attention Module Algorithm
- layers.py: convolutional layer algorithm
- model.py: Some related algorithmic models
- network.py: The neural network model used
- poissin_add.py: add poissin noise
- psnr.py: Evaluation of denoising results
- run.py: Quick Run Scripts
- test.py: Arithmetic on test sets
- train.py: Arithmetic on train sets
- utils.py: Some functions


## run
### 1. add poissin noise and gaussian noise

```
python poissin_add.py 
python gaussian_add.py
```

### 2. train

```
python run.py train
```

### 3. test

```
python run.py test
```
