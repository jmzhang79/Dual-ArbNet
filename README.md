# Dual-ArbNet
Official Pytorch implementation of "Dual Arbitrary Scale Super-Resolution for Multi-Contrast MRI"

## Requirements
- Python 3.9
- asposestorage==1.0.2
- imageio==2.22.4
- matplotlib==3.6.2
- numpy==1.23.5
- opencv_python==4.6.0.66
- scikit_image==0.19.3
- scipy==1.10.1
- skimage==0.0
- thop==0.1.1.post2209072238
- torch==1.13.0
- torchvision==0.14.0
- tqdm==4.64.1

## Train
### 1. Prepare training data
1.1 Downkload fastMRI dataset and IXI dataset.  
1.2 Filter the multi contrast MRI datasets.
### 2. Begin to train
Run `./main.sh` to train on the training dataset. Please update `name_train`, `dir_data`, `save`, `ref_mat`, `ref_list` in the bash file as your needs.

## Quick Test on An LR MR Image
Download [pre-trained weights](https://1drv.ms/u/s!Amr2hw2GQjYIhRF46VujiNq-TNrL?e=nvO780) and put it in the `experiment` folder.

Run `./test_save.sh` to enlarge an LR image to an arbitrary size. Please update `dir_data` and `pre_train` in the bash file as `your_path`.

## Visual Results
### SR with Arbitrary Scale Factors
You can change the --scale `./test_save.sh` to obtain the results of different scale factors.
You can also change the --ref_type_test `./test_save.sh` to use HR(1) or LR(2) reference image.
