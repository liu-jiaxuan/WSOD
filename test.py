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
import ast
import argparse
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='grabcut_bfillrate_cbg_edge')
parser.add_argument('--use_boxsup', type=ast.literal_eval, default=False)
parser.add_argument('--save', type=ast.literal_eval, default=True)
parser.add_argument('--num_worker', type=int, default=8)
args = parser.parse_args()


TAG = args.tag
SAVE_PATH = TAG
GPU_ID=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="{}/test_{}.log".format(TAG,TAG), filemode="w")
DATASETS = ['ECSSD', 'DUT', 'PASCAL', 'HKU-IS', 'THUR', 'DUTS_Test']


class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.cfg.base_dir='../../../dataset/sod_scribble/test/'
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=args.num_worker)
        ## network
        self.net    = Network(self.cfg)
        path = './{}/model-best.pt'.format(TAG)
        state_dict = torch.load(path)
        print('complete loading: {}'.format(path))
        self.net.load_state_dict(state_dict)
        print('model has {} parameters in total'.format(sum(x.numel() for x in self.net.parameters())))
        self.net.train(False)
        self.net.cuda()
        self.net.eval()


    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask,bbox, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                start_time = time.time()
                out2, out2_bbox, out3, out4, out5 = self.net(image, 'Test')

                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                out2_bbox = F.interpolate(out2_bbox, size=(H, W), mode='bilinear', align_corners=False)

                if args.use_boxsup:
                    pred= (torch.sigmoid((out2[0, 0])) *torch.sigmoid(out2_bbox[0, 0]))
                else:
                    pred = torch.sigmoid((out2[0, 0]))
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time
                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                # ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.mean()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)


    def save(self):
        with torch.no_grad():
            tbar=tqdm.tqdm(self.loader)
            for image, mask,bbox, (H, W), name in tbar:
                out2, out2_bbox, _, _, _ = self.net(image.cuda().float(), 'Test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                out2_bbox = F.interpolate(out2_bbox, size=(H, W), mode='bilinear', align_corners=False)
                if args.use_boxsup:
                    pred= (torch.sigmoid((out2[0, 0])) *torch.sigmoid(out2_bbox[0, 0])).cpu().numpy()
                else:
                    pred=(torch.sigmoid((out2[0, 0]))).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head     = './{}/pred_maps/{}/'.format(TAG,TAG) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))


if __name__=='__main__':
    for e in DATASETS:
        print(e)
        t =Test(dataset, e, SCWSSOD)
        if e in [ 'DUT', 'DUTS_Test','ECSSD', 'HKU-IS', 'PASCAL', 'THUR']:
            t.accuracy()
        if args.save:
            t.save()