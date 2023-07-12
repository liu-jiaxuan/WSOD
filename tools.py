import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random
from skimage import color


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def ToLabel(E):
    fgs = np.argmax(E, axis=1).astype(np.float32)
    return fgs.astype(np.uint8)


def np_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def torch_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def SSIM(x, y):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y
    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)


# RGB
# RGB: All three of Red, Green and Blue in [0..1].
def xyz2rgb(xyz):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    transform_tensor = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                                     [-0.9692660, 1.8760108, 0.0415560],
                                     [0.0556434, -0.2040259, 1.0572252]])
    if xyz.is_cuda:
        transform_tensor = transform_tensor.cuda()
    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    if len(xyz.shape) == 4:
        convolved = F.conv2d(xyz, transform_tensor)
    else:
        convolved = F.conv2d(xyz.unsqueeze(0), transform_tensor).squeeze(0)
    # return convolved
    return torch.where(convolved > 0.0031308,
                       1.055 * (convolved.pow(1. / 2.4)) - 0.055,
                       12.92 * convolved)


def rgb2xyz(rgb):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    rgb = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4),
                      rgb / 12.92)

    transform_tensor = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                     [0.2126729, 0.7151522, 0.0721750],
                                     [0.0193339, 0.1191920, 0.9503041]])
    if rgb.is_cuda:
        transform_tensor = transform_tensor.cuda()
    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    if len(rgb.shape) == 4:
        return F.conv2d(rgb, transform_tensor)
    else:
        return F.conv2d(rgb.unsqueeze(0), transform_tensor).squeeze(0)


# LAB
# CIE-L*a*b*: A perceptually uniform color space,
# i.e. distances are meaningful. L* in [0..1] and a*, b* almost in [-1..1].
D65 = [0.95047, 1.00000, 1.08883]


def lab_f(t):
    return torch.where(t > 0.008856451679035631, t.pow(1 / 3),
                       t * 7.787037037037035 + 0.13793103448275862)


def lab_finv(t):
    return torch.where(t > 0.20689655172413793, t.pow(3),
                       0.12841854934601665 * (t - 0.13793103448275862))


def lab2xyz(lab, wref=None):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if wref is None:
        wref = D65
    dim = 1 if len(lab.shape) == 4 else 0
    l, a, b = lab.chunk(3, dim=dim)

    l2 = (l + 0.16) / 1.16
    x = wref[0] * lab_finv(l2 + a / 5)
    y = wref[1] * lab_finv(l2)
    z = wref[2] * lab_finv(l2 - b / 2)
    xyz = torch.cat([x, y, z], dim=dim)

    return xyz


def xyz2lab(xyz, wref=None):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if wref is None:
        wref = D65
    dim = 1 if len(xyz.shape) == 4 else 0
    x, y, z = xyz.chunk(3, dim=dim)

    fy = lab_f(y / wref[1])
    l = 1.16 * fy - 0.16
    a = 5.0 * (lab_f(x / wref[0]) - fy)
    b = 2.0 * (fy - lab_f(z / wref[2]))
    xyz = torch.cat([l, a, b], dim=dim)

    return xyz


# composed functions
def lab2rgb(lab):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    return xyz2rgb(lab2xyz(lab))


def rgb2lab(rgb):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    return xyz2lab(rgb2xyz(rgb))


def SaliencyStructureConsistency(x, y, alpha):
    ssim = torch.mean(SSIM(x, y))
    l1_loss = torch.mean(torch.abs(x - y))
    loss_ssc = alpha * ssim + (1 - alpha) * l1_loss
    return loss_ssc


def SaliencyStructureConsistencynossim(x, y):
    l1_loss = torch.mean(torch.abs(x - y))
    return l1_loss


def batch_fill_rate(mask, bbox):
    """
    :param mask: B*C*H*W
    :param bbox: B*C*H*W
    :return: rate
    """
    assert mask.shape == bbox.shape
    B, C, H, W = mask.shape
    mask = mask.reshape(B, C, H * W)
    bbox = bbox.reshape(B, C, H * W)
    sum_mask = torch.sum(mask, dim=2)
    sum_bbox = torch.sum(bbox, dim=2)
    rate = sum_mask * 1.0 / sum_bbox

    return rate.min(), rate.max(), rate.mean()


def img_fill_rate(mask, bbox):
    """
    :param mask:  B*C*H*W
    :param bbox: B*C*H*W
    :return: rate
    """
    assert mask.shape == bbox.shape
    B, C, H, W = mask.shape
    mask = mask.reshape(B, C, H * W)
    bbox = bbox.reshape(B, C, H * W)
    sum_mask = torch.sum(mask, dim=2)
    sum_bbox = torch.sum(bbox, dim=2)
    rate = sum_mask * 1.0 / sum_bbox
    return rate.squeeze(1)


def get_mask_from_fill(pred, bbox, rate, bg_score):
    """
    :param pred: B*C*H*W
    :param bbox: B*C*H*W
    :param rate: filling rate
    :return: mask that pred value prob>fillrate
    """
    assert bbox.shape == pred.shape
    with torch.no_grad():
        B, C, H, W = bbox.shape
        bbox = bbox.reshape(B, H * W)
        num_pixels = torch.floor(torch.sum(bbox > 0, dim=1) * rate).long()
        pred_sort = pred.reshape(B, H * W)
        pred_sort, indices = torch.sort(pred_sort, dim=1, descending=True)
        mask = torch.zeros_like(pred)
        mask_background = torch.zeros_like(pred)
        for i in range(B):
            threshold = pred_sort[i, num_pixels[i]]
            mask[i, pred[i] >= threshold] = 1.0
            mask_background[i, pred[i] <= bg_score] = 1.0
        return mask, mask_background


def get_mask_from_fill_inbox(pred, bbox, rate, bg_score):
    """
    :param pred: B*C*H*W
    :param bbox: B*C*H*W
    :param rate: filling rate
    :return: mask that pred value prob>fillrate
    """
    assert bbox.shape == pred.shape
    with torch.no_grad():
        B, C, H, W = bbox.shape
        pred[bbox == 0] = 0
        bbox = bbox.reshape(B, H * W)
        num_pixels = torch.floor(torch.sum(bbox > 0, dim=1) * rate).long()
        pred_sort = pred.reshape(B, H * W)
        pred_sort, indices = torch.sort(pred_sort, dim=1, descending=True)
        mask = torch.zeros_like(pred)
        mask_background = torch.zeros_like(pred)
        for i in range(B):
            if num_pixels[i] == 102400:
                threshold = pred_sort[i, 102399]
            else:
                threshold = pred_sort[i, num_pixels[i]]
            mask[i, pred[i] >= threshold] = 1.0
            mask_background[i, pred[i] <= bg_score] = 1.0
        return mask, mask_background


def get_bgmask_from_fill(pred, bbox, rate):
    """
    :param pred: B*C*H*W
    :param bbox: B*C*H*W
    :param rate: filling rate
    :return: mask that pred value prob<fillrate
    """
    assert bbox.shape == pred.shape
    with torch.no_grad():
        B, C, H, W = bbox.shape
        bbox_reshape = bbox.reshape(B, H * W)
        num_pixels = torch.floor(torch.sum(bbox_reshape > 0, dim=1) *
                                 rate).long()
        pred[bbox == 0] = 1.0
        pred_sort = pred.reshape(B, H * W)
        pred_sort, indices = torch.sort(pred_sort, dim=1, descending=False)
        mask = torch.zeros_like(pred)
        for i in range(B):
            if num_pixels[i] == 102400:
                threshold = pred_sort[i, 102399]
            else:
                threshold = pred_sort[i, num_pixels[i]]
            mask[i, pred[i] <= threshold] = 1.0
        return mask


def get_mask_from_fill_img(pred, bbox, rate, bg_score):
    """
    :param pred: B*C*H*W
    :param bbox: B*C*H*W
    :param rate: filling rate
    :return: mask that pred value prob>fillrate
    """
    assert bbox.shape == pred.shape
    B, C, H, W = bbox.shape
    pred[bbox == 0] = 0
    bbox = bbox.reshape(B, H * W)
    pred_sort = pred.reshape(B, H * W)
    mask = torch.zeros_like(pred)
    mask_background = torch.zeros_like(pred)
    for i in range(B):
        num_pixels = torch.floor(torch.sum(bbox[i] > 0) * rate[i]).long()
        pred_sort_i, indices = torch.sort(pred_sort[i], dim=0, descending=True)
        threshold = pred_sort_i[num_pixels]
        mask[i, pred[i] >= threshold] = 1.0
        mask_background[i, pred[i] <= bg_score] = 1.0
    return mask, mask_background


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_h = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
        kernel_v = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)


class PartialGradLoss(nn.Module):
    def __init__(self):
        super(PartialGradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        mask = torch.zeros_like(gt_grad)
        mask[gt_grad > 0.0] = 1.0

        return self.loss(output_grad * mask, gt_grad * mask)


def feature_contrastive_sampled_loss(feature,
                                     bbox,
                                     fill_mask,
                                     similarity_temperature=0.3,
                                     pos_num=100,
                                     neg_num=100):
    """

    :param feature: B*C*H*W: [B,256,80,80]
    :param bbox: B*C*H*W
    :param fill_mask: B*1*H*W
    :return:
    """
    B, C, H, W = feature.shape

    fill_mask = fill_mask * bbox
    negative_mask = 1 - bbox

    flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
    feature_flatten, fill_mask, negative_mask = list(
        map(flatten, [feature, fill_mask, negative_mask]))
    loss = 0
    for bi in range(B):
        bi_feature = feature_flatten[bi]
        bi_fill_mask = fill_mask[bi]
        bi_negative_mask = negative_mask[bi]
        positive_feature = bi_feature[:, bi_fill_mask.squeeze().bool()]
        if positive_feature.shape[1] < pos_num:
            sampled_positive_feature = positive_feature
        else:
            sampled_positive_index = random.sample(
                list(range(0, positive_feature.shape[1])), pos_num)
            sampled_positive_feature = positive_feature.index_select(
                1,
                torch.tensor(sampled_positive_index).cuda())

        negative_feature = bi_feature[:, bi_negative_mask.squeeze().bool()]
        if negative_feature.shape[1] < neg_num:
            sampled_negative_feature = negative_feature
        else:
            sampled_negative_index = random.sample(
                list(range(0, negative_feature.shape[1])), neg_num)
            sampled_negative_feature = negative_feature.index_select(
                1,
                torch.tensor(sampled_negative_index).cuda())

        similarity_positive = F.cosine_similarity(
            positive_feature[..., :, None],
            sampled_positive_feature[..., None, :],
            dim=0) / similarity_temperature
        similarity_negative = F.cosine_similarity(
            positive_feature[..., :, None],
            sampled_negative_feature[..., None, :],
            dim=0) / similarity_temperature
        if similarity_positive.shape[0] == 0 or similarity_positive.shape[
                1] == 0 or similarity_negative.shape[
                    0] == 0 or similarity_negative.shape[1] == 0:
            bi_loss = torch.tensor(0.0).cuda()
        else:
            bi_loss = -torch.log(similarity_positive.exp().sum() /
                                 (similarity_negative.exp().sum() +
                                  similarity_positive.exp().sum() + 1e-8))
        loss += bi_loss
    loss /= B
    return loss


def feature_contrastive_loss_downsample(feature,
                                        bbox,
                                        fill_mask,
                                        similarity_temperature=0.3):
    """

    :param feature: B*C*H*W: [B,256,80,80]
    :param bbox: B*C*H*W
    :param fill_mask: B*1*H*W
    :return:
    """
    feature = F.interpolate(feature,
                            scale_factor=0.25,
                            mode='bilinear',
                            align_corners=False)
    bbox = F.interpolate(bbox,
                         scale_factor=0.25,
                         mode='bilinear',
                         align_corners=False)
    fill_mask = F.interpolate(fill_mask,
                              scale_factor=0.25,
                              mode='bilinear',
                              align_corners=False)

    B, C, H, W = feature.shape

    fill_mask = fill_mask * bbox
    negative_mask = 1 - bbox

    flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
    feature_flatten, fill_mask, negative_mask = list(
        map(flatten, [feature, fill_mask, negative_mask]))
    loss = 0
    for bi in range(B):
        bi_feature = feature_flatten[bi]
        bi_fill_mask = fill_mask[bi]
        bi_negative_mask = negative_mask[bi]
        positive_feature = bi_feature[:, bi_fill_mask.squeeze().bool()]
        negative_feature = bi_feature[:, bi_negative_mask.squeeze().bool()]
        similarity_positive = F.cosine_similarity(
            positive_feature[..., :, None],
            positive_feature[..., None, :],
            dim=0) / similarity_temperature
        similarity_negative = F.cosine_similarity(
            positive_feature[..., :, None],
            negative_feature[..., None, :],
            dim=0) / similarity_temperature
        if similarity_positive.shape[0] == 0 or similarity_positive.shape[
                1] == 0 or similarity_negative.shape[
                    0] == 0 or similarity_negative.shape[1] == 0:
            bi_loss = torch.tensor(0.0).cuda()
        else:
            bi_loss = -torch.log(similarity_positive.exp().sum() /
                                 (similarity_negative.exp().sum() +
                                  similarity_positive.exp().sum() + 1e-8))
        loss += bi_loss
    loss /= B
    return loss


def extract_patches(x, kernel=25, stride=1):
    """
    :param x: (B,C,H,W)
    :param kernel: the size (k) of local patch
    :param stride: step size
    :return: (B,C,H,W,k,k)
    """
    if kernel != 1:
        x = nn.ReplicationPad2d(int((kernel - 1) / 2))(x)
    all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)


def feature_local_consist_loss(feature, kernel=3):
    """l
    :param feature: (B,C,H,W)
    :param kernel: the size (k) of the patch
    :return: loss
    """
    feature_patches = extract_patches(x=feature, kernel=kernel, stride=1)
    B, C, H, W, k, k = feature_patches.shape
    feature_center = extract_patches(x=feature, kernel=1,
                                     stride=1).expand(-1, -1, -1, -1, k, k)

    similarity_patches = F.cosine_similarity(feature_center,
                                             feature_patches,
                                             dim=1)
    loss = (1 - similarity_patches).sum() / (B * H * W * k * k)

    return loss


def feature_loca_consist_loss_sample(feature,
                                     kernel=3,
                                     stride=1,
                                     patches_num=1024):
    """l
    :param feature: (B,C,H,W)
    :param kernel: the size (k) of the patch
    :return: loss
    """

    feature_patches = extract_patches(x=feature, kernel=kernel, stride=stride)
    B, C, H, W, k, k = feature_patches.shape
    feature_center = extract_patches(x=feature, kernel=1,
                                     stride=stride).expand(
                                         -1, -1, -1, -1, k, k)

    similarity_patches = F.cosine_similarity(feature_center,
                                             feature_patches,
                                             dim=1)

    loss = (1 - similarity_patches).sum() / (B * H * W * k * k)
    return loss


def unfold_wo_center(x, kernel_size, dilation, stride):
    """
    :param x: B,C,H,W
    :param kernel_size:
    :param dilation:
    :return: B,C,kh*kw-1,H,W
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    B, C, _, _ = x.shape
    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x,
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation,
                          stride=stride)
    if stride == 1:
        unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), -1, x.size(2),
                                        x.size(3))
    else:
        H = int(np.sqrt((unfolded_x.shape[2])))
        W = H
        unfolded_x = unfolded_x.reshape(B, C, -1, H, W)

    # remove the center pixels
    size = kernel_size**2
    unfolded_x = torch.cat(
        (unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]),
        dim=2)

    return unfolded_x


def unfold_with_center(x, kernel_size, dilation, stride):
    """
    :param x: B,C,H,W
    :param kernel_size:
    :param dilation:
    :return: B,C,kh*kw,H,W
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    B, C, _, _ = x.shape
    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x,
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation,
                          stride=stride)

    if stride == 1:
        unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), -1, x.size(2),
                                        x.size(3))
    else:
        H = int(np.sqrt((unfolded_x.shape[2])))
        W = H
        unfolded_x = unfolded_x.reshape(B, C, -1, H, W)

    # remove the center pixels

    return unfolded_x


def get_images_color_similarity(images, kernel_size, dilation, stride):
    """
    :param images: B,C,H,W
    :param image_masks:
    :param kernel_size:
    :param dilation:
    :return: B,kh*kw-1,H,W
    """
    assert images.dim() == 4

    unfolded_images = unfold_wo_center(images,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       stride=stride)
    if stride != 1:
        images = unfold_with_center(images,
                                    kernel_size=1,
                                    dilation=dilation,
                                    stride=stride)
        diff = images - unfolded_images
    else:
        diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity


def color_feature_contrast_loss_downsampled(images, feature, kernel_size,
                                            stride, dilation_rate, color_tau,
                                            sim_temp):
    """
    :param images: B,C,H,W (input 320)
    :param feature: input (B,C,80*80)
    :param kernel_size:
    :param stride:
    :param dilation_rate:
    :return:
    """

    feature = F.interpolate(feature,
                            scale_factor=0.5,
                            mode='bilinear',
                            align_corners=False)
    images = F.interpolate(images,
                           scale_factor=0.125,
                           mode='bilinear',
                           align_corners=False)
    images = (images - torch.min(images)) / (torch.max(images) -
                                             torch.min(images))

    images_lab = rgb2lab(images)
    color_smilarity = get_images_color_similarity(images_lab,
                                                  kernel_size=kernel_size,
                                                  dilation=dilation_rate,
                                                  stride=stride)
    # similarity B,kh*kw-1,H,W
    unfolded_feature = unfold_wo_center(feature,
                                        kernel_size=kernel_size,
                                        dilation=dilation_rate,
                                        stride=stride)
    anchor_feature = unfold_with_center(feature,
                                        kernel_size=1,
                                        dilation=dilation_rate,
                                        stride=stride)
    #feature: B,C,kh*kw-1,H,W   anchor feature: B,C,1,H,W
    positive_mask = color_smilarity >= color_tau
    # negative_mask=color_smilarity<color_tau

    similarity_feature = F.cosine_similarity(
        anchor_feature[:, :, :, None, :, :],
        unfolded_feature[:, :, None, :, :, :],
        dim=1) / sim_temp
    #Similarity_fature: B,1,kh*kw-1,H,W

    try:
        loss = -torch.log(
            similarity_feature.masked_select(positive_mask[:, None,
                                                           ...]).exp().sum() /
            (similarity_feature.exp().sum() + 1e-8)) / images.shape[0]
    except:
        if torch.sum(positive_mask) == 0:
            return torch.tensor(0.0).cuda()
    return loss


def color_feature_contrast_loss_downsampled_inbox(images, feature, bbox,
                                                  kernel_size, stride,
                                                  dilation_rate, color_tau,
                                                  sim_temp):
    """
    :param images: B,C,H,W (input 320)
    :param feature: input (B,C,80*80)
    :bbox: B,1,H,W
    :param kernel_size:
    :param stride:
    :param dilation_rate:
    :return:
    """

    feature = F.interpolate(feature,
                            scale_factor=0.5,
                            mode='bilinear',
                            align_corners=False)
    images = F.interpolate(images,
                           scale_factor=0.125,
                           mode='bilinear',
                           align_corners=False)
    bbox = F.interpolate(bbox, scale_factor=0.125, mode='nearest')
    images = (images - torch.min(images)) / (torch.max(images) -
                                             torch.min(images))
    B, C, H, W = images.shape

    images_lab = rgb2lab(images)
    color_smilarity = get_images_color_similarity(images_lab,
                                                  kernel_size=kernel_size,
                                                  dilation=dilation_rate,
                                                  stride=stride)
    # similarity B,kh*kw-1,H,W
    unfolded_feature = unfold_wo_center(feature,
                                        kernel_size=kernel_size,
                                        dilation=dilation_rate,
                                        stride=stride)
    anchor_feature = unfold_with_center(feature,
                                        kernel_size=1,
                                        dilation=dilation_rate,
                                        stride=stride)
    #feature: B,C,kh*kw-1,H,W   anchor feature: B,C,1,H,W
    positive_mask = (color_smilarity >= color_tau) * bbox.reshape(B, 1, H,
                                                                  W).bool()
    negative_mask = (color_smilarity < color_tau) * bbox.reshape(B, 1, H,
                                                                 W).bool()

    similarity_feature = F.cosine_similarity(
        anchor_feature[:, :, :, None, :, :],
        unfolded_feature[:, :, None, :, :, :],
        dim=1) / sim_temp
    #Similarity_fature: B,1,kh*kw-1,H,W
    if torch.sum(positive_mask) == 0:
        return torch.tensor(0.0).cuda()
    pos_sim = similarity_feature.masked_select(positive_mask[:, None,
                                                             ...]).exp().sum()
    neg_sim = similarity_feature.masked_select(negative_mask[:, None,
                                                             ...]).exp().sum()
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)) / images.shape[0]

    return loss


def pixConv_soft(kernel, imgs, ksize=25):  #kernel:(B,kh*kw,H,W)
    kernel = kernel.permute(0, 2, 3, 1).contiguous()  #kernel:(B,H,W,kh*kw)
    patches = extract_patches(imgs, kernel=ksize)
    B, C, H, W, kh, kw = patches.shape
    patches = patches.reshape(B, C, H, W, kh * kw)
    sums = kernel.sum(dim=3).reshape(B, H, W, 1)
    kernel = kernel.div(sums + 1e-9)
    kernel = kernel.reshape(B, 1, H, W, kh * kw)
    out = patches.mul(kernel)
    out = out.sum(dim=4)  #(B,C,H,W)
    return out


def pixConv(kernel, imgs, ksize=25):  #kernel:(B,kh*kw,H,W)
    kernel = kernel.permute(0, 2, 3, 1).contiguous()  #kernel:(B,H,W,kh*kw)
    patches = extract_patches(imgs, kernel=ksize)
    B, C, H, W, kh, kw = patches.shape
    patches = patches.reshape(B, C, H, W, -1)
    kernel = kernel.reshape(B, 1, H, W, -1)
    out = patches.mul(kernel)
    out = out.sum(dim=4)  #(B,C,H,W)
    return out


def pixConv_single(kernel, imgs, ksize=25):  #kernel:(B,1,kh,kw)
    kernel = kernel.permute(0, 2, 3, 1).contiguous()  #kernel:(B,H,W,kh*kw)
    patches = extract_patches(imgs, kernel=ksize)
    B, C, H, W, kh, kw = patches.shape
    patches = patches.reshape(B, C, H, W, -1)
    kernel = kernel.reshape(B, 1, 1, 1, -1)
    out = patches.mul(kernel)
    out = out.sum(dim=4)
    return out


def get_identity_kernel(ksize, B, H, W):
    kernel = torch.zeros([B, ksize * ksize, H, W])
    kernel[:, (ksize * ksize) // 2, :, :] = 1.0
    return kernel


def iou_loss(preds, bbox, thr=0.5, eps=1e-6, reduction='mean'):
    '''
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    '''
    pred_maps: C*1*H*W bbox: 
    '''
    preds = preds > thr
    torch.max(preds, 2)

    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (
        preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    ious = (inters / uni).clamp(min=eps)
    loss = -ious.log()

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import torch
    from data.dataset import Config, Data
    from torch.utils.data import Dataset, DataLoader
    from lib.data_prefetcher import DataPrefetcher
    cfg = Config(mode='train',
                 datapath='../../../../dataset/sod_scribble/train',
                 transform_mode=0)
    data = Data(cfg)
    loader = DataLoader(data, batch_size=2, shuffle=False, num_workers=0)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask, bbox, _ = prefetcher.next()

    rate, _, _ = batch_fill_rate(mask, bbox)
    pred = torch.rand_like(mask)
    fill_mask, _ = get_mask_from_fill(pred, bbox, rate, 0.01)
    feature = torch.rand([2, 256, 80, 80]).cuda()
    color_feature_contrast_loss_downsampled_inbox(image,
                                                  feature,
                                                  bbox,
                                                  kernel_size=3,
                                                  stride=1,
                                                  dilation_rate=1,
                                                  color_tau=0.85,
                                                  sim_temp=0.3)