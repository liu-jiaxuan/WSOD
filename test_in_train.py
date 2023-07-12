import os
import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from net_agg import SCWSSOD
import time
import logging as logger
import tqdm
from tools import *
from torchvision.utils import make_grid


class Test(object):
    def __init__(self, datapath='DUTS_Test',mode='test', Dataset=dataset):
        ## dataset    
        self.test_dir_path = '../../../dataset/sod_scribble/test/'
        self.train_dir_path = '../../../dataset/sod_scribble/train/' 
            
        if mode=='test':
            self.datapath = datapath.split("/")[-1]
            print("Testing on %s"%self.datapath)
            self.cfg = Dataset.Config(datapath=datapath, mode='test')
            self.cfg.base_dir=self.test_dir_path
        elif mode=='vis':
            self.datapath=datapath
            print("Visualizing on %s" % self.datapath)
            self.cfg = Dataset.Config(datapath=datapath, mode='vis')
            self.cfg.base_dir = self.train_dir_path
        self.cfg.mode=mode
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)


    def accuracy(self,net,logger,args):
        net.train(False)
        net.eval()
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask,bbox, (H, W), maskpath in self.loader:
                image, mask            = image.cuda().float(), mask.cuda().float()
                start_time = time.time()
                out2,out2_bbox, out3, out4, out5 = net(image, 'Test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                out2_bbox = F.interpolate(out2_bbox, size=(H, W), mode='bilinear', align_corners=False)

                if args.use_boxsup:
                    pred= (torch.sigmoid((out2[0, 0])) *torch.sigmoid(out2_bbox[0, 0]))
                else:
                    pred= torch.sigmoid((out2[0, 0]))
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time

                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                if self.datapath in ['ECSSD','PASCAL']:
                    precision = torch.zeros(number)
                    recall    = torch.zeros(number)
                    for i in range(number):
                        temp         = (pred >= threshod[i]).float()
                        precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                        recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                    mean_pr += precision
                    mean_re += recall
                    fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                    fscore=fscore.mean()/cnt
                else:
                    fscore=0.0
            fps = len(self.loader.dataset) / cost_time
            msg = 'Eval %s MAE=%.6f, meanF-score=%.6f, len(imgs)=%s, fps=%.4f' % (
            self.datapath, mae / cnt, fscore, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)
        net.train(True)
        return mae/cnt,fscore


    def save(self,TAG,net,args):
        net.eval()
        with torch.no_grad():
            tbar=tqdm.tqdm(self.loader)
            for image, mask,bbox, (H, W), name in tbar:
                out2,out2_bbox, out3, out4, out5 = net(image.cuda().float(), 'Test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                out2_bbox = F.interpolate(out2_bbox, size=(H, W), mode='bilinear', align_corners=False)
                if args.use_boxsup:
                    pred= (torch.sigmoid(out2[0, 0])*torch.sigmoid(out2_bbox[0, 0])).cpu().numpy()
                else:
                    pred=torch.sigmoid(out2[0, 0]).cpu().numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head     = './{}/pred_maps/{}/'.format(TAG,TAG) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))
        net.train(True)


    def visualize(self,TAG,net,args,sw):
        net.eval()
        with torch.no_grad():
            tbar=tqdm.tqdm(self.loader)
            step=0
            for image, mask,bbox,global_gt, (H, W), name in tbar:
                step+=1
                if step>200:
                    break
                out2,out2_bbox, out3, out4, out5,feature = net(image.cuda().float(), 'Train')
                mask=mask.cuda().float()
                bbox=bbox.cuda().float()
                global_gt=global_gt.cuda().float()

                out2_init = out2.detach()
                if args.use_boxsup:
                    out2 = out2 * out2_bbox

                min_rate, max_rate, mean_rate = batch_fill_rate(mask.clone(), bbox.clone())
                if args.fillrate == 'min':
                    rate = min_rate
                elif args.fillrate == 'mean':
                    rate = mean_rate
                elif args.fillrate == 'max':
                    rate = max_rate

                if args.inbox_fg:
                    fr_mask, bg_fr_mask = get_mask_from_fill_inbox(out2.clone().detach(), bbox.clone(),
                                                                   rate * args.inbox_fg_rate, args.bg_score)
                else:
                    fr_mask, bg_fr_mask = get_mask_from_fill(out2.clone().detach(), bbox.clone(),
                                                             rate * args.inbox_fg_rate, args.bg_score)
                if args.inbox_bg_rate > 0:
                    bg_fr_mask_rate = get_bgmask_from_fill(out2.clone().detach(), bbox.clone(),
                                                           (1 - max_rate) * args.inbox_bg_rate)
                    bg_fr_mask_rate = bg_fr_mask_rate.squeeze(1).long()


                out2_bbox = torch.cat((1 - out2_bbox, out2_bbox), 1)
                bbox = bbox.squeeze(1).long()
                out2 = torch.cat((1 - out2, out2), 1)
                ######   label for partial cross-entropy loss  ######
                gt = mask.squeeze(1).long()
                fr_mask = fr_mask.squeeze(1).long()
                bg_fr_mask=bg_fr_mask.squeeze(1).long()

                grid_image = make_grid(image[0].clone().cpu().data, 1, normalize=True)

                grid_image1 = make_grid(
                    [grid_image, bbox[0].unsqueeze(0).clone().cpu().data.expand(3, bbox[0].shape[0], bbox[0].shape[1]),
                     mask[0].clone().cpu().data.expand(3, bbox[0].shape[0], bbox[0].shape[1]),
                     global_gt[0].clone().cpu().data.expand(3, bbox[0].shape[0], bbox[0].shape[1])],
                    4, normalize=False)

                cm = plt.get_cmap('jet')
                np_out_init = torch.from_numpy(
                    cm(np.array(bg_fr_mask[0].detach().cpu().numpy() * 255).astype(np.uint8))[:, :, :3].transpose(2,0,1))
                np_out_bbox = torch.from_numpy(
                    cm(np.array(out2_bbox[0, 1].detach().cpu().numpy() * 255).astype(np.uint8))[:, :, :3].transpose(2,0,1))
                np_out2 = torch.from_numpy(
                    cm(np.array(out2[0, 1].detach().cpu().numpy() * 255).astype(np.uint8))[:, :, :3].transpose(2, 0, 1))
                np_fill_mask = torch.from_numpy(
                    cm(np.array(fr_mask[0].detach().cpu().numpy() * 255).astype(np.uint8))[:, :, :3].transpose(2, 0, 1))

                grid_image2 = make_grid([np_out_init, np_out_bbox, np_out2, np_fill_mask], 4, normalize=False)

                np_feature_mean_sigmoid = torch.from_numpy(cm(
                    np.array(torch.sigmoid(torch.mean(feature[0].detach(), dim=0)).cpu().numpy() * 255).astype(
                        np.uint8))[:, :, :3].transpose(2, 0, 1))
                np_feature_mean_sigmoid = \
                F.interpolate(np_feature_mean_sigmoid.unsqueeze(0), size=out2.size()[2:], mode='bilinear',
                              align_corners=False)[0]
                np_feature_mean_normalize = torch.from_numpy(cm(
                    np.array(torch_normalize(torch.mean(feature[0].detach(), dim=0)).cpu().numpy() * 255).astype(
                        np.uint8))[:, :, :3].transpose(2, 0, 1))
                np_feature_mean_normalize = \
                F.interpolate(np_feature_mean_normalize.unsqueeze(0), size=out2.size()[2:], mode='bilinear',
                              align_corners=False)[0]
                np_feature_max_sigmoid = torch.from_numpy(cm(
                    np.array(torch.sigmoid(torch.max(feature[0].detach(), dim=0)[0]).cpu().numpy() * 255).astype(
                        np.uint8))[:, :, :3].transpose(2, 0, 1))
                np_feature_max_sigmoid = \
                F.interpolate(np_feature_max_sigmoid.unsqueeze(0), size=out2.size()[2:], mode='bilinear',
                              align_corners=False)[0]
                np_feature_max_normalize = torch.from_numpy(cm(
                    np.array(torch_normalize(torch.max(feature[0].detach(), dim=0)[0]).cpu().numpy() * 255).astype(
                        np.uint8))[:, :, :3].transpose(2, 0, 1))
                np_feature_max_normalize = \
                F.interpolate(np_feature_max_normalize.unsqueeze(0), size=out2.size()[2:], mode='bilinear',
                              align_corners=False)[0]
                grid_image = make_grid([np_feature_mean_sigmoid, np_feature_mean_normalize, np_feature_max_sigmoid,
                                        np_feature_max_normalize], 4, normalize=False)
               
                grid_image = make_grid([grid_image1, grid_image2, grid_image], 1, normalize=False)
                grid_image=F.interpolate(grid_image.unsqueeze(0), scale_factor=0.5, mode='bilinear',align_corners=False)[0]

                sw.add_image('Train Images Visualize',grid_image, step)

        net.train(True)