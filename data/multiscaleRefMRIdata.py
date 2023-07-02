import os
import glob
from data import common
import pickle
import numpy as np
import imageio
import torch
import torch.utils.data as data
import math
import scipy.io as sio

def fft2c(x):
    """
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    res = 1 / math.sqrt(S[0]*S[1]) * np.fft.fftshift(np.fft.fft2(x,axes=[0,1]),axes=[0,1])
    return res

def ifft2c(x):
    """
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    x = np.fft.ifftshift(x,axes=[0,1])
    res = math.sqrt(S[0]*S[1]) * np.fft.ifft2(x,axes=[0,1])
    return res

def cal_step(scale):
    if abs(scale - round(scale)) < 0.001:
        step = 1
    elif abs(scale * 2 - round(scale * 2)) < 0.001:
        step = 2
    elif abs(scale * 5 - round(scale * 5)) < 0.001:
        step = 5
    elif abs(scale * 10 - round(scale * 10)) < 0.001:
        step = 10
    elif abs(scale * 20 - round(scale * 20)) < 0.001:
        step = 20
    return step


def paired_crop(x, ref, scale):
    """
    input:
        x: input HR complex matrix [H,W,C]
        lq_size: target LR shape
        scale: downsample scale of x
    output:
        x_crop: HR image corresponding to LR image
        x_lq: downsampled LR image shape is [lq_size,lq_size,C]
    """
    h_hr,w_hr = x.shape
    H_hr = round(math.floor(h_hr / 24) * 24)
    W_hr = round(math.floor(w_hr / 24) * 24)
    x = x[h_hr//2-math.floor(H_hr/2):h_hr//2+math.ceil(H_hr/2),w_hr//2-math.floor(W_hr/2):w_hr//2+math.ceil(W_hr/2)]
    ref = ref[h_hr//2-math.floor(H_hr/2):h_hr//2+math.ceil(H_hr/2),w_hr//2-math.floor(W_hr/2):w_hr//2+math.ceil(W_hr/2)]

    fs = fft2c(x)
    H_lr,W_lr = round(H_hr/scale[0]),round(W_hr/scale[1])
    H_c,W_c = H_hr//2,W_hr//2
    fsref = fft2c(ref)

    fs_crop_lr = fs[H_c-math.floor(H_lr/2):H_c+math.ceil(H_lr/2),W_c-math.floor(W_lr/2):W_c+math.ceil(W_lr/2)]
    x_hr = x
    x_lr = ifft2c(fs_crop_lr)

    fsref_crop_lr = fsref[H_c-math.floor(H_lr/2):H_c+math.ceil(H_lr/2),W_c-math.floor(W_lr/2):W_c+math.ceil(W_lr/2)]
    ref_hr = ref
    ref_lr = ifft2c(fsref_crop_lr)

    x_hr_real = x_hr.real
    x_hr_real = x_hr_real[ :, :,np.newaxis]
    x_hr_imag = x_hr.imag
    x_hr_imag = x_hr_imag[ :, :,np.newaxis]
    x_hr = np.concatenate((x_hr_real,x_hr_imag),2)

    x_lr_real = x_lr.real
    x_lr_real = x_lr_real[ :, :,np.newaxis]
    x_lr_imag = x_lr.imag
    x_lr_imag = x_lr_imag[ :, :,np.newaxis]
    x_lr = np.concatenate((x_lr_real,x_lr_imag),2)

    ref_hr_real = ref_hr.real
    ref_hr_real = ref_hr_real[ :, :,np.newaxis]
    ref_hr_imag = ref_hr.imag
    ref_hr_imag = ref_hr_imag[ :, :,np.newaxis]
    ref_hr = np.concatenate((ref_hr_real,ref_hr_imag),2)

    ref_lr_real = ref_lr.real
    ref_lr_real = ref_lr_real[ :, :,np.newaxis]
    ref_lr_imag = ref_lr.imag
    ref_lr_imag = ref_lr_imag[ :, :,np.newaxis]
    ref_lr = np.concatenate((ref_lr_real,ref_lr_imag),2)
    if np.max(x_hr)!=0 and np.max(ref_hr)!=0:
        x_lr = x_lr/(math.sqrt(scale[0]*scale[1])*np.max(x_hr))
        x_hr = x_hr/np.max(x_hr)
        ref_lr = ref_lr/(math.sqrt(scale[0]*scale[1])*np.max(ref_hr))
        ref_hr = ref_hr/np.max(ref_hr)
    return x_hr, x_lr, ref_hr, ref_lr

class RefMRIData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.scale2 = args.scale2
        self.idx_scale = 0
        self.first_epoch =False


        self._set_filesystem(args.dir_data)
        self._read_reflist(args.dir_data, args.ref_list, args.ref_mat)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr = self._scan()
        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr = self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr = list_hr
            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )

                self.images_hr = [], [[] for _ in range(len(self.scale))]
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

        if train:
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )

        return names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = self.apath
        self.ext = ('.mat','.mat')


    def _read_reflist(self, dir_data, ref_list, ref_mat):
        dir_ref = os.path.join(dir_data, ref_mat)
        ref_file = os.path.join(dir_data, ref_list)
        self.dictref = {}
        with open(ref_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                lr, ref = line.split(' ')
                self.dictref[lr] = os.path.join(dir_ref,ref+'.mat')        

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f:
                    ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f:
                pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        lr, hr, ref_hr, ref_lr, filename = self._load_file(idx)
        lr_cat = np.concatenate([lr,ref_lr],axis=2)
        lr, hr, ref_hr = self.get_patch(lr_cat, hr, ref_hr)
        lr_tensor, hr_tensor, ref_hr_tensor = common.np2Tensor(
            lr, hr, ref_hr, rgb_range=1
        )
        return (lr_tensor[:2,:,:], ref_hr_tensor, lr_tensor[2:,:,:]), hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        hr = sio.loadmat(f_hr)['dcm']
        
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        filename_withoutdcm = filename.split('.')[0]
        f_ref = self.dictref[filename_withoutdcm]
        refimg = sio.loadmat(f_ref)['dcm']
        hr , lr, ref_hr, ref_lr = paired_crop(hr, refimg, (self.scale[self.idx_scale],self.scale2[self.idx_scale]))
        return lr, hr, ref_hr, ref_lr, filename

    def get_patch(self, lr, hr, ref_hr):
        scale = self.scale[self.idx_scale]
        scale2 = self.scale2[self.idx_scale]
        if self.train:
            if self.args.asymm:
                lr, hr, ref_hr = common.get_patch(
                    lr,
                    hr,
                    ref_hr,
                    patch_size=self.args.patch_size,
                    scale=scale,
                    scale2=scale2
                )
            else:
                lr, hr, ref_hr = common.get_patch(
                    lr,
                    hr,
                    ref_hr,
                    patch_size=self.args.patch_size,
                    scale=scale,
                    scale2=scale
                )

            if not self.args.no_augment:
                lr, hr, ref_hr = common.augment(lr, hr, ref_hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:int(ih * scale), 0:int(iw * scale2)]
            ref_hr = ref_hr[0:int(ih * scale), 0:int(iw * scale2)]

        return lr, hr, ref_hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
