"""
CVPR 2022
Paper ID: 5498
"""

import sys
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
import shutil
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from data import dataset
from net_agg import SCWSSOD
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lscloss import *
import time
import ast
import numpy as np
from tools import *
import matplotlib.pyplot as plt
from test_in_train import Test
import argparse
from crf.config2d import config
from crf.convcrf2d import ConvCRF2d


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='grabcut_bfillrate_cbg_edge')
parser.add_argument('--train_txt', type=str, default='train')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--num_worker', type=int, default=8)
parser.add_argument('--transform_mode', type=int, default=0)
parser.add_argument('--lr_style',
                    type=str,
                    default='triangle',
                    help='can be in [step,triangle,cos]')
parser.add_argument('--cos_step',
                    type=int,
                    default=40,
                    help='can be in [step,triangle]')
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--lr_decay_epoch', type=int, default=20)
parser.add_argument('--use_clip', type=ast.literal_eval, default=False)
parser.add_argument('--use_boxsup', type=ast.literal_eval, default=False)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--base_lr', type=float, default=1e-5)
parser.add_argument('--max_lr', type=float, default=1e-2)
parser.add_argument('--fillrate', type=str, default='min')
parser.add_argument('--inbox_loss', type=ast.literal_eval,
                    default=False)  #Box in Sturcture loss
parser.add_argument('--inbox_fg', type=ast.literal_eval,
                    default=False)  #inbox background loss
parser.add_argument('--inbox_fg_rate', type=float, default=1.0)
parser.add_argument('--inbox_bg', type=ast.literal_eval,
                    default=False)  #inbox background loss
parser.add_argument('--inbox_bg_rate', type=float, default=0)
parser.add_argument('--bg_score', type=float, default=0.01)
parser.add_argument('--color_contrast', type=ast.literal_eval,
                    default=False)  #Contrastive loss
parser.add_argument('--color_inbox', type=ast.literal_eval, default=False)
parser.add_argument('--color_rate', type=float, default=100.0)
parser.add_argument('--color_k', type=int, default=3)
parser.add_argument('--color_d', type=int, default=1)
parser.add_argument('--color_s', type=int, default=1)
parser.add_argument('--color_tau', type=float, default=0.9)
parser.add_argument('--color_temp', type=float, default=0.3)
parser.add_argument('--contrast', type=ast.literal_eval,
                    default=False)  #Contrastive loss
parser.add_argument('--contrast_rate', type=float, default=100.0)
parser.add_argument('--feature_down', type=ast.literal_eval, default=False)
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--pos_num', type=int, default=512)
parser.add_argument('--neg_num', type=int, default=512)
parser.add_argument('--patch_loss', type=ast.literal_eval,
                    default=False)  #patch consistent loss
parser.add_argument('--patch_kernel', type=int, default=11)
parser.add_argument('--patch_stride', type=int, default=5)
parser.add_argument('--patch_num', type=int, default=1024)
parser.add_argument('--edge_loss', type=ast.literal_eval,
                    default=False)  # Edge guidance Loss
parser.add_argument('--edge_rate', type=float, default=20.0)
parser.add_argument('--edge_partial', type=ast.literal_eval, default=False)
parser.add_argument('--label_update', type=ast.literal_eval, default=False)
parser.add_argument('--crf_update', type=ast.literal_eval,
                    default=False)  #CRF update loss
parser.add_argument('--crf_kernel', type=int, default=3)
parser.add_argument('--combine_update', type=ast.literal_eval,
                    default=False)  #A combine update label mechanism
parser.add_argument('--rand_gate', type=ast.literal_eval,
                    default=False)  #Random gated input
parser.add_argument('--resume', type=ast.literal_eval, default=False)
parser.add_argument('--vis_only', type=ast.literal_eval, default=False)
parser.add_argument('--cuda', type=ast.literal_eval, default=True)
parser.add_argument('--gpus', type=ast.literal_eval, default=False)
args = parser.parse_args()
print(args)


# file paths
summary_save_path = "../summary2"
dataset_path = "../../../dataset/sod_scribble/train"


TAG = args.tag
SAVE_PATH = args.tag
summary_path = os.path.join(summary_save_path, TAG)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
else:
    if not args.resume:
        print('summary_dir: ', SAVE_PATH,
              'already exist, will be removed !!!!!!')
        time.sleep(5)
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
else:
    if not args.resume:
        print('save_dir: ', SAVE_PATH, 'already exist, will be removed !!!!!!')
        time.sleep(5)
        shutil.rmtree(SAVE_PATH)
        os.makedirs(SAVE_PATH)
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="{}/train_{}.log".format(SAVE_PATH,TAG), filemode="w")

shutil.copy('./train.py', os.path.join(SAVE_PATH, 'train.py'))
shutil.copy('./net_agg.py', os.path.join(SAVE_PATH, 'net_agg.py'))
shutil.copy('./tools.py', os.path.join(SAVE_PATH, 'tools.py'))
logger.info(str(args))
if args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    print('gpus 0,1')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('gpu 0')


def time_change(time_init):
    time_list = []
    if time_init / 3600 > 1:
        time_h = int(time_init / 3600)
        time_m = int((time_init - time_h * 3600) / 60)
        time_s = int(time_init - time_h * 3600 - time_m * 60)
        time_list.append(str(time_h))
        time_list.append('h ')
        time_list.append(str(time_m))
        time_list.append('m ')

    elif time_init / 60 > 1:
        time_m = int(time_init / 60)
        time_s = int(time_init - time_m * 60)
        time_list.append(str(time_m))
        time_list.append('m ')
    else:
        time_s = int(time_init)

    time_list.append(str(time_s))
    time_list.append('s')
    time_str = ''.join(time_list)
    return time_str


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first -
              base_lr * total_steps) / (first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] -
                                       momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps - 1) / num_steps)**power


def step_lr(base_lr, epoch, decay_rate=0.1, decay_epoch=20):
    decay = decay_rate**(epoch // decay_epoch)
    lr = decay * base_lr
    return lr


BASE_LR = args.base_lr
MAX_LR = args.max_lr
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
batch = args.batch_size
l = 0.3


def train(Dataset, Network):
    # dataset
    cfg = Dataset.Config(datapath=dataset_path,
                         savepath=SAVE_PATH,
                         train_txt=args.train_txt,
                         mode='train',
                         batch=batch,
                         lr=1e-3,
                         momen=0.9,
                         decay=5e-4,
                         epoch=args.epoch,
                         transform_mode=args.transform_mode)
    data = Dataset.Data(cfg)
    loader = DataLoader(data,
                        batch_size=cfg.batch,
                        shuffle=True,
                        num_workers=args.num_worker)
    db_size = len(loader)
    # network
    net = Network(cfg)
    criterion = torch.nn.CrossEntropyLoss(weight=None,
                                          ignore_index=255,
                                          reduction='mean')
    criterion_mse = torch.nn.MSELoss()
    edge_loss = GradLoss().cuda()
    net.train(True)
    if args.gpus:
        net = nn.DataParallel(net)
    net.cuda()
    criterion.cuda()
    model_crf = ConvCRF2d(config, kernel_size=args.crf_kernel).cuda()

    # parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    if args.lr_style == 'cos':

        optimizer = torch.optim.SGD([{
            'params': base,
            'lr': args.max_lr * 0.1
        }, {
            'params': head
        }],
                                    lr=args.max_lr,
                                    momentum=cfg.momen,
                                    weight_decay=cfg.decay,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cos_step * db_size, eta_min=args.base_lr)
    else:
        optimizer = torch.optim.SGD([{
            'params': base
        }, {
            'params': head
        }],
                                    lr=args.max_lr,
                                    momentum=cfg.momen,
                                    weight_decay=cfg.decay,
                                    nesterov=True)
    #Load form checkpoints
    if args.resume and os.path.exists(cfg.savepath + '/model_newest.pt'):
        checkpoint = torch.load(cfg.savepath + '/model_newest.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.lr_style == 'cos':
            scheduler.last_epoch = checkpoint['epoch']

    sw = SummaryWriter(summary_path)

    max_iterations = cfg.epoch * db_size
    global_step = args.start_epoch * db_size
    best_mae, best_mae_ecssd, best_mae_pascal, best_meanf_ecssd, best_meanf_pascal = 1.0, 1.0, 1.0, 1.0, 1.0
    start = time.time()
    # -------------------------- training ------------------------------------
    for epoch in range(args.start_epoch, cfg.epoch):
        if args.vis_only:
            break
        loss_epoch = 0.0
        loss_step = 0
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask, bbox, global_gt = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            if args.lr_style == 'triangle':
                lr, momentum = get_triangle_lr(BASE_LR,
                                               MAX_LR,
                                               cfg.epoch * db_size,
                                               niter,
                                               ratio=1.)
                optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
                optimizer.param_groups[1]['lr'] = lr
                optimizer.momentum = momentum
            elif args.lr_style == 'step':
                lr = step_lr(BASE_LR, epoch, args.lr_decay_rate,
                             args.lr_decay_epoch)
                optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
                optimizer.param_groups[1]['lr'] = lr
            elif args.lr_style == 'cos':
                scheduler.step()
            batch_idx += 1
            global_step += 1

            ######  saliency structure consistency loss  ######
            if args.rand_gate:
                if np.random.randint(2) == 1:
                    out2, out2_bbox, out3, out4, out5, feature = net(
                        image * bbox, 'Train')
                else:
                    out2, out2_bbox, out3, out4, out5, feature = net(
                        image, 'Train')
            else:
                out2, out2_bbox, out3, out4, out5, feature = net(
                    image, 'Train')
            if args.inbox_loss:
                out2_inbox, _, _, _, _, _ = net(image * bbox, 'Train')
                loss_inbox = SaliencyStructureConsistency(
                    out2, out2_inbox, 0.85)
            else:
                loss_inbox = torch.tensor(0.0).cuda()

            out2_init = out2.detach()
            if args.use_boxsup:
                out2 = out2 * out2_bbox
            if args.edge_loss:
                if args.edge_partial:
                    loss_edge = edge_loss_partial(out2, mask) * args.edge_rate
                else:
                    loss_edge = edge_loss(out2, mask) * args.edge_rate
            else:
                loss_edge = torch.tensor(0.0).cuda()

            min_rate, max_rate, mean_rate = batch_fill_rate(
                mask.clone(), bbox.clone())
            if args.fillrate == 'min':
                rate = min_rate
            elif args.fillrate == 'mean':
                rate = mean_rate
            elif args.fillrate == 'max':
                rate = max_rate

            if args.inbox_fg:
                fr_mask, bg_fr_mask = get_mask_from_fill_inbox(
                    out2.clone().detach(), bbox.clone(),
                    rate * args.inbox_fg_rate, args.bg_score)
            else:
                fr_mask, bg_fr_mask = get_mask_from_fill(
                    out2.clone().detach(), bbox.clone(),
                    rate * args.inbox_fg_rate, args.bg_score)
            if args.inbox_bg_rate > 0:
                bg_fr_mask_rate = get_bgmask_from_fill(out2.clone().detach(),
                                                       bbox.clone(),
                                                       args.inbox_bg_rate)
                bg_fr_mask_rate = bg_fr_mask_rate.squeeze(1).long()
            if args.contrast:
                if args.feature_down:
                    loss_const = args.contrast_rate * feature_contrastive_loss_downsample(
                        feature,
                        F.interpolate(bbox,
                                      scale_factor=0.25,
                                      mode='bilinear',
                                      align_corners=False),
                        F.interpolate(fr_mask,
                                      scale_factor=0.25,
                                      mode='bilinear',
                                      align_corners=False),
                        similarity_temperature=args.temperature)
                else:
                    loss_const = args.contrast_rate * feature_contrastive_sampled_loss(
                        feature,
                        F.interpolate(bbox,
                                      scale_factor=0.25,
                                      mode='bilinear',
                                      align_corners=False),
                        F.interpolate(fr_mask,
                                      scale_factor=0.25,
                                      mode='bilinear',
                                      align_corners=False),
                        similarity_temperature=args.temperature,
                        pos_num=args.pos_num,
                        neg_num=args.neg_num)
            else:
                loss_const = torch.tensor(0.0).cuda()
            if args.color_contrast:
                if not args.color_inbox:
                    loss_color = args.color_rate * color_feature_contrast_loss_downsampled(
                        image,
                        feature,
                        kernel_size=args.color_k,
                        stride=args.color_s,
                        dilation_rate=args.color_d,
                        color_tau=args.color_tau,
                        sim_temp=args.color_temp)
                else:
                    loss_color = args.color_rate * color_feature_contrast_loss_downsampled_inbox(
                        image,
                        feature,
                        bbox,
                        kernel_size=args.color_k,
                        stride=args.color_s,
                        dilation_rate=args.color_d,
                        color_tau=args.color_tau,
                        sim_temp=args.color_temp)
            else:
                loss_color = torch.tensor(0.0).cuda()

            if args.patch_loss:
                loss_patch = feature_loca_consist_loss_sample(
                    feature,
                    kernel=args.patch_kernel,
                    stride=args.patch_stride,
                    patches_num=args.patch_num)
            else:
                loss_patch = torch.tensor(0.0).cuda()

            out2_bbox = torch.cat((1 - out2_bbox, out2_bbox), 1)
            bbox = bbox.squeeze(1).long()

            if args.use_boxsup:
                loss_bbox = criterion(out2_bbox, bbox)
            else:
                loss_bbox = torch.tensor(0.0).cuda()

            out2 = torch.cat((1 - out2, out2), 1)
            ######   label for partial cross-entropy loss  ######
            gt = mask.squeeze(1).long()
            fr_mask = fr_mask.squeeze(1).long()
            bg_fr_mask = bg_fr_mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255

            if args.inbox_bg and epoch > 1:
                if args.inbox_bg_rate > 0:
                    inbox_bg = bg_fr_mask_rate
                else:
                    inbox_bg = bg_fr_mask * bbox
                bg_label[bbox == 1] = 255
                bg_label[inbox_bg == 1] = 0
            else:
                bg_label[bbox == 1] = 255  #certain background
            fg_label[fr_mask == 0] = 255

            if args.combine_update:
                crf_label = model_crf(image, out2)
                crf_label = torch.argmax(crf_label, dim=1)
                loss_new_gt = criterion(out2, crf_label)
            else:
                if args.label_update:
                    new_gt = (out2[:, 1:2] * out2_bbox[:, 1:2] +
                              gt.unsqueeze(1) * (1 - out2_bbox[:, 1:2]))
                    loss_new_gt = criterion_mse(out2[:, 1:2],
                                                new_gt) + criterion_mse(
                                                    out2_bbox[:, 1:2],
                                                    bbox.unsqueeze(1).float())
                elif args.crf_update:
                    crf_label = model_crf(image, out2)
                    crf_label = torch.argmax(crf_label, dim=1)
                    loss_new_gt = criterion(out2, crf_label)
                else:
                    loss_new_gt = torch.tensor(0.0).cuda()

            #Debug
            fg_loss = criterion(out2, fg_label)
            bg_loss = criterion(out2, bg_label)
            if args.combine_update:
                alpha = epoch * (1.0 / (args.epoch - 1))
                loss2 = (1 - alpha) * (
                    fg_loss + bg_loss
                ) + alpha * loss_new_gt + loss_bbox + loss_inbox + loss_edge + loss_const + loss_patch + loss_color
            else:
                loss2 = fg_loss + bg_loss + loss_bbox + loss_inbox + loss_edge + loss_const + loss_patch + loss_new_gt + loss_color

            loss = loss2
            loss_epoch += loss2.data
            loss_step += 1
            optimizer.zero_grad()

            loss.backward()
            if args.lr_style == 'step' and args.use_clip:
                clip_gradient(optimizer, 0.5)
            optimizer.step()

            if global_step % 200 == 0 or global_step == 1:
                with torch.no_grad():
                    sw.add_scalar('loss', loss.item(), global_step=global_step)
                    sw.add_scalar('fg_loss',
                                  fg_loss.item(),
                                  global_step=global_step)
                    sw.add_scalar('bg_loss',
                                  bg_loss.item(),
                                  global_step=global_step)
                    sw.add_scalar('bbox_loss',
                                  loss_bbox.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_inbox',
                                  loss_inbox.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_edge',
                                  loss_edge.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_const',
                                  loss_const.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_patch',
                                  loss_patch.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_new_gt',
                                  loss_new_gt.item(),
                                  global_step=global_step)
                    sw.add_scalar('loss_color',
                                  loss_color.item(),
                                  global_step=global_step)
                    sw.add_scalar('lr',
                                  optimizer.param_groups[0]['lr'],
                                  global_step=global_step)
                    grid_image = make_grid(image[0].clone().cpu().data,
                                           1,
                                           normalize=True)

                    grid_image1 = make_grid([
                        grid_image,
                        bbox[0].unsqueeze(0).clone().cpu().data.expand(
                            3, bbox[0].shape[0], bbox[0].shape[1]),
                        mask[0].clone().cpu().data.expand(
                            3, bbox[0].shape[0], bbox[0].shape[1]),
                        global_gt[0].clone().cpu().data.expand(
                            3, bbox[0].shape[0], bbox[0].shape[1])
                    ],
                                            4,
                                            normalize=False)
                    cm = plt.get_cmap('jet')
                    np_out_init = torch.from_numpy(
                        cm(
                            np.array(bg_fr_mask[0].detach().cpu().numpy() *
                                     255).astype(
                                         np.uint8))[:, :, :3].transpose(
                                             2, 0, 1))
                    np_out_bbox = torch.from_numpy(
                        cm(
                            np.array(out2_bbox[0, 1].detach().cpu().numpy() *
                                     255).astype(
                                         np.uint8))[:, :, :3].transpose(
                                             2, 0, 1))
                    np_out2 = torch.from_numpy(
                        cm(
                            np.array(out2[0, 1].detach().cpu().numpy() *
                                     255).astype(
                                         np.uint8))[:, :, :3].transpose(
                                             2, 0, 1))
                    np_fill_mask = torch.from_numpy(
                        cm(
                            np.array(fr_mask[0].detach().cpu().numpy() *
                                     255).astype(
                                         np.uint8))[:, :, :3].transpose(
                                             2, 0, 1))

                    grid_image2 = make_grid(
                        [np_out_init, np_out_bbox, np_out2, np_fill_mask],
                        4,
                        normalize=False)

                    np_feature_mean_sigmoid = torch.from_numpy(
                        cm(
                            np.array(
                                torch.sigmoid(
                                    torch.mean(feature[0].detach(),
                                               dim=0)).cpu().numpy() *
                                255).astype(np.uint8))[:, :, :3].transpose(
                                    2, 0, 1))
                    np_feature_mean_sigmoid = F.interpolate(
                        np_feature_mean_sigmoid.unsqueeze(0),
                        size=out2.size()[2:],
                        mode='bilinear',
                        align_corners=False)[0]
                    np_feature_mean_normalize = torch.from_numpy(
                        cm(
                            np.array(
                                torch_normalize(
                                    torch.mean(feature[0].detach(),
                                               dim=0)).cpu().numpy() *
                                255).astype(np.uint8))[:, :, :3].transpose(
                                    2, 0, 1))
                    np_feature_mean_normalize = F.interpolate(
                        np_feature_mean_normalize.unsqueeze(0),
                        size=out2.size()[2:],
                        mode='bilinear',
                        align_corners=False)[0]
                    np_feature_max_sigmoid = torch.from_numpy(
                        cm(
                            np.array(
                                torch.sigmoid(
                                    torch.max(feature[0].detach(),
                                              dim=0)[0]).cpu().numpy() *
                                255).astype(np.uint8))[:, :, :3].transpose(
                                    2, 0, 1))
                    np_feature_max_sigmoid = F.interpolate(
                        np_feature_max_sigmoid.unsqueeze(0),
                        size=out2.size()[2:],
                        mode='bilinear',
                        align_corners=False)[0]
                    np_feature_max_normalize = torch.from_numpy(
                        cm(
                            np.array(
                                torch_normalize(
                                    torch.max(feature[0].detach(),
                                              dim=0)[0]).cpu().numpy() *
                                255).astype(np.uint8))[:, :, :3].transpose(
                                    2, 0, 1))
                    np_feature_max_normalize = F.interpolate(
                        np_feature_max_normalize.unsqueeze(0),
                        size=out2.size()[2:],
                        mode='bilinear',
                        align_corners=False)[0]
                    grid_image = make_grid([
                        np_feature_mean_sigmoid, np_feature_mean_normalize,
                        np_feature_max_sigmoid, np_feature_max_normalize
                    ],
                                           4,
                                           normalize=False)
                    grid_image = make_grid(
                        [grid_image1, grid_image2, grid_image],
                        1,
                        normalize=False)
                    grid_image = F.interpolate(grid_image.unsqueeze(0),
                                               scale_factor=0.5,
                                               mode='bilinear',
                                               align_corners=False)[0]
                    sw.add_image(
                        'ROW1_Image---Bbox---Mask--GT ROW2_Out_init---Outbbox---OutFinal---Fill_rate_mask',
                        grid_image, global_step)

            if batch_idx % 10 == 0:
                process = global_step * 1.0 / max_iterations
                end = time.time()
                use_time = end - start

                all_time = use_time / process
                res_time = all_time - use_time
                str_ues_time = time_change(use_time)
                str_res_time = time_change(res_time)
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss_fg=%.6f | loss_bg=%.6f | loss_bbox=%.6f |' \
                      ' loss_inbox=%.6f | loss_edge=%.6f | loss_const=%.6f| loss_patch=%.6f | loss_new_gt=%.6f | loss_color=%.6f | Eval best MAE-meanF= D%.6f--%.6f | E%.6f--%.6f | P%.6f--%.6f | Used [%s] Res [%s]'\
                      % (SAVE_PATH, datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'],
                         loss.item(), fg_loss.item(), bg_loss.item(), loss_bbox.item(),loss_inbox.item(),loss_edge.item(),
                         loss_const.item(),loss_patch.item(),loss_new_gt.item(),loss_color.item(),best_mae,0.0,best_mae_ecssd,best_meanf_ecssd,best_mae_pascal,best_meanf_pascal,str_ues_time,str_res_time)
                print(msg)
                logger.info(msg)
            image, mask, bbox, global_gt = prefetcher.next()
        loss_epoch /= loss_step
        sw.add_scalar('Epoch/Loss', loss_epoch, global_step=epoch + 1)
        sw.add_scalar('loss_epoch', loss_epoch, global_step=epoch + 1)
        if epoch > 28:
            if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epoch:
                torch.save(net.state_dict(),
                           cfg.savepath + '/model-' + str(epoch + 1) + '.pt')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, cfg.savepath + '/model_newest.pt')
        if (epoch + 1) % 5 == 0:
            test = Test()
            mae, _ = test.accuracy(net, logger, args)
            sw.add_scalar('Epoch/MAE', mae, global_step=epoch + 1)

            test = Test(datapath='ECSSD')
            mae_ecssd, meanf_ecssd = test.accuracy(net, logger, args)
            sw.add_scalar('Epoch/MAE_ECSSD', mae_ecssd, global_step=epoch + 1)
            sw.add_scalar('Epoch/MeanF_ECSSD',
                          meanf_ecssd,
                          global_step=epoch + 1)

            test = Test('PASCAL')
            mae_pascal, meanf_pascal = test.accuracy(net, logger, args)
            sw.add_scalar('Epoch/MAE_PASCAL',
                          mae_pascal,
                          global_step=epoch + 1)
            sw.add_scalar('Epoch/MeanF_PASCAL',
                          meanf_pascal,
                          global_step=epoch + 1)
            if mae < best_mae:
                best_mae = mae
                best_mae_ecssd = mae_ecssd
                best_mae_pascal = mae_pascal
                best_meanf_ecssd = meanf_ecssd
                best_meanf_pascal = meanf_pascal
                torch.save(net.state_dict(), cfg.savepath + '/model-best.pt')
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epoch:
            test = Test(datapath='../../../dataset/sod_scribble/train/',
                        mode='vis')
            test.visualize(TAG, net, args, sw)
    if args.vis_only:
        test = Test(datapath='../../../dataset/sod_scribble/train/',
                    mode='vis')
        test.visualize(TAG, net, args, sw)


if __name__ == '__main__':
    train(dataset, SCWSSOD)