import os
import math
import matplotlib
matplotlib.use('Agg')
import utility
import torch
import numpy as np
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class DataConsistency(nn.Module):
    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, FSsr, hr, shape1, shape2):
        b,c,h,w = hr.shape
        hr_comp = hr[:,0:1,:,:]+1j*hr[:,1:2,:,:]
        k_hr = 1 / math.sqrt(hr.shape[2]*hr.shape[3]) * torch.fft.fftshift(torch.fft.fftn(hr_comp, dim=[2,3]))

        mask1 = torch.ones_like(k_hr)
        mask1[:,:,h//2-math.floor(shape1/2):h//2+math.ceil(shape1/2),w//2-math.floor(shape2/2):w//2+math.ceil(shape2/2)] = 0
        k_out = FSsr*mask1 + k_hr*(1-mask1)

        k_out = torch.fft.ifftshift((k_out), dim=[2,3])
        x_res = math.sqrt(h*w) * torch.fft.ifftn(k_out, dim=[2,3])
        x_res = torch.cat([x_res.real,x_res.imag],dim=1)
        return x_res

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.psnr_max = None

        self.DataConsistency = DataConsistency()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if epoch == 1 and self.args.load == '.':
            self.loader_train.dataset.first_epoch = True
            # adjust learning rate
            lr = 5e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # train on all scale factors for remaining epochs
        else:
            self.loader_train.dataset.first_epoch = False
            # adjust learning rate
            lr = self.args.lr * (2 ** -(epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            if isinstance(lr,list):
                lr, ref_hr, ref_lr, hr = self.prepare(lr[0], lr[1], lr[2], hr)
            else:
                lr, hr = self.prepare(lr, hr)
                ref_hr = None
            scale = hr.size(2) / lr.size(2)
            scale2 = hr.size(3) / lr.size(3)
            timer_data.hold()
            self.optimizer.zero_grad()

            # inference
            self.model.get_model().set_scale(scale, scale2)
            if ref_hr is None:
                sr = self.model(lr)
            else:
                sr = self.model((lr, ref_hr, ref_lr, self.args.ref_type, epoch))
            if isinstance(sr,tuple):
                sr,Refsr = sr
            else:
                Refsr = None
            # loss function
            loss = self.loss(sr, Refsr, None, hr, ref_hr, lr.shape[2], lr.shape[3])

            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_latest.pt')
        )
        if epoch % self.args.save_every == 0:
            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
            self.ckp.write_log('save ckpt epoch{:.4f}'.format(epoch))

    def test(self):
        self.model.eval()

        with torch.no_grad():
            if self.args.test_only:
                scale_list = range(len(self.args.scale))
                logger = print
            else:
                scale_list = [9,19,29]
                logger = self.ckp.write_log

            eval_psnr_avg = []
            for idx_scale in scale_list:
                self.loader_test.dataset.set_scale(idx_scale)
                scale = self.args.scale[idx_scale]
                scale2 = self.args.scale2[idx_scale]

                eval_psnr = 0
                eval_ssim = 0
                for idx_img, (lr, hr, filename, _) in tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
                    filename = filename[0]
                    # prepare LR & HR images
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        if isinstance(lr,list):
                            lr, ref_hr, ref_lr, hr = self.prepare(lr[0], lr[1], lr[2], hr)
                        else:
                            lr, hr = self.prepare(lr, hr)
                            ref_hr = None
                            ref_lr = None
                    else:
                        if isinstance(lr,list):
                            lr, ref_hr, ref_lr = self.prepare(lr[0], lr[1], lr[2])
                        else:
                            lr, = self.prepare(lr)
                            ref_hr = None
                            ref_lr = None
                    lr, hr, ref_hr, ref_lr = self.crop_border(lr, hr, ref_hr, ref_lr, scale, scale2)
                    # inference
                    self.model.get_model().set_scale(scale, scale2)
                    if ref_hr is None:
                        sr = self.model(lr)
                    else:
                        sr = self.model((lr, ref_hr, ref_lr, self.args.ref_type_test))
                    if isinstance(sr,tuple):
                        sr,Refsr = sr                    

                    if not no_eval:
                        psnr, ssim, mse = utility.calc_psnr(
                            lr, sr,  hr, img_name=filename, scale=[scale, scale2], 
                            save = self.args.save_results, savefile = self.args.savefigfilename,ref = ref_hr
                        )
                        eval_psnr += psnr
                        eval_ssim += ssim 


                if scale == scale2:
                    logger('[{} x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
                else:
                    logger('[{} x{}/x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        scale2,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
                eval_psnr_avg.append(eval_psnr / len(self.loader_test))
            eval_psnr_avg = np.mean(eval_psnr_avg)
        if not self.args.test_only: #training mode and save the best model
            if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
                self.psnr_max = eval_psnr_avg
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckp.dir, 'model', 'model_best.pt')
                )
                logger('save ckpt PSNR:{:.4f}'.format(eval_psnr_avg))


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def crop_border(self, img_lr, img_hr, img_ref_hr, img_ref_lr, scale, scale2):
        N, C, H_lr, W_lr = img_lr.size()
        N, C, H_hr, W_hr = img_hr.size()
        H = H_lr if round(H_lr * scale) <= H_hr else math.floor(H_hr / scale)
        W = W_lr if round(W_lr * scale2) <= W_hr else math.floor(W_hr / scale2)

        step = []
        for s in [scale, scale2]:
            if s == int(s):
                step.append(1)
            elif s * 2 == int(s * 2):
                step.append(2)
            elif s * 5 == int(s * 5):
                step.append(5)
            elif s * 10 == int(s * 10):
                step.append(10)
            elif s * 20 == int(s * 20):
                step.append(20)
            elif s * 50 == int(s * 50):
                step.append(50)

        H_new = H // step[0] * step[0]
        if H_new % 2 == 1:
            H_new = H // (step[0] * 2) * step[0] * 2

        W_new = W // step[1] * step[1]
        if W_new % 2 == 1:
            W_new = W // (step[1] * 2) * step[1] * 2

        img_lr = img_lr[:, :, :H_new, :W_new]
        img_hr = img_hr[:, :, :round(scale * H_new), :round(scale2 * W_new)]
        if img_ref_hr is not None:
            img_ref_hr = img_ref_hr[:, :, :round(scale * H_new), :round(scale2 * W_new)]
            img_ref_lr = img_ref_lr[:, :, :H_new, :W_new]
        return img_lr, img_hr, img_ref_hr, img_ref_lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
