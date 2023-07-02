import torch
import torch.nn as nn
import numpy as np
import math


class KLoss(nn.Module):
    def __init__(self, args):
        super(KLoss, self).__init__()


    def forward(self, sr, FSsr, hr, shape1, shape2):
        b,c,h,w = hr.shape
        hr_comp = hr[:,0:1,:,:]+1j*hr[:,1:2,:,:]

        if FSsr is None:
            sr_comp = sr[:,0:1,:,:]+1j*sr[:,1:2,:,:]
            FSsr = 1 / math.sqrt(h*w) * torch.fft.fftn(sr_comp, dim=[2,3])
            FSsr = torch.fft.fftshift(FSsr, dim=[2,3])
        FShr = 1 / math.sqrt(h*w) * torch.fft.fftn(hr_comp, dim=[2,3])
        FShr = torch.fft.fftshift(FShr, dim=[2,3])
        mask = torch.ones_like(FSsr)
        mask[:,:,h//2-math.floor(shape1/2):h//2+math.ceil(shape1/2),w//2-math.floor(shape2/2):w//2+math.ceil(shape2/2)] = 0        

        loss = torch.mean(torch.abs((FSsr-FShr)*mask))      
        return loss

if __name__ == "__main__":
    model_test = KLoss(0)
    a = torch.randn([5,2,20,20])
    b = torch.randn([5,2,20,20])
    loss = model_test(a,b,5,5)
    print(loss)