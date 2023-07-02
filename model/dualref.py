import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.resblock import ResBlock
def make_model(args, parent=False):
    return DUALRef(args)

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

def patch_norm_2d(x, kernel_size=3):
    mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
    mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
    var = mean_sq - mean**2
    return (x-mean)/(var + 1e-6)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[64, 64, 64, 64, 64]):
        super().__init__()

        last_dim_K = in_channels * 9 + in_channels * 9
        
        last_dim_Q = 4

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                        nn.ReLU(),
                                        ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                        ))    
            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim*2
            last_dim_Q = hidden_dim
        self.last_layer = nn.Conv2d(hidden_dims[-1], 2, 1)
        self.ref_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],2, 1),
                            nn.ReLU())
        self.in_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],2, 1),
                            nn.ReLU())
        
    def _make_pos_encoding(self, x, size): 
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact'))
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, ref, syn_inp):
        q = syn_inp
        q_ref =syn_inp
        k = x
        k_ref = ref
        kk = torch.cat([k,k_ref],dim=1)
        for i in range(len(self.K)):
            kk = self.K[i](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]*self.Q[i](q)
            q_ref = kk[:,dim:]*self.Q[i](q_ref)
        q = self.last_layer(q)
        q_ref = self.last_layer(q_ref)
        return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)

    def batched_step(self, x, syn_inp, bsize):
        with torch.no_grad():
            h, w = syn_inp.shape[-2:]
            ql = 0
            preds = []
            while ql < w:
                qr = min(ql + bsize//h, w)
                pred = self.step(x[:, :, :, ql: qr], syn_inp[:, :, :, ql: qr])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=-1)
        return pred


    def forward(self, x, ref, size, bsize=None):
        B, C, H_in, W_in = x.shape
        Bref, Cref, H_in_ref, W_in_ref = ref.shape
        rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size))
        ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size))
        ratio_ref = (ref.new_tensor([math.sqrt((H_in_ref*W_in_ref)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(Bref, -1, *size))
        syn_inp = torch.cat([rel_coord, ratio, ratio_ref], dim=1)
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='bilinear')
        ref = F.interpolate(F.unfold(ref, 3, padding=1).view(B, C*9, H_in_ref, W_in_ref), size=syn_inp.shape[-2:], mode='bilinear')
        if bsize is None: 
            pred = self.step(x, ref, syn_inp)
        else:
            pred = self.batched_step(x, syn_inp, bsize)
        return pred


class DUALRef(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = make_rdn()
        self.decoder = ImplicitDecoder()
        self.mixer = nn.Conv2d(64*2, 64, 1, padding=0, stride=1)
    
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    def forward(self, inp, bsize = None):
        if len(inp)==5:
            epoch = inp[4]
        else:
            epoch = None
        ref_type = inp[3]
        if ref_type == None:
            ref_type = random.randint(1,2) 
            if epoch is not None and epoch < 10:
                ref_type = 1
        ref = inp[ref_type] 
        inp = inp[0]

        B,C,H,W = inp.shape
        B,C,H_ref,W_ref = ref.shape
        H_hr = round(H*self.scale)
        W_hr = round(W*self.scale2)
        feat = self.encoder((inp-0.5)/0.5)
        with torch.no_grad():
            ref = self.encoder((ref-0.5)/0.5)
        ref.requires_grad = True
        size = [H_hr, W_hr]
        pred,pred_ref = self.decoder(feat, ref, size, bsize)

        return pred*0.5+0.5, pred_ref*0.5+0.5
