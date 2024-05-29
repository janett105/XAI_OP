# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
# import cv2
import numpy as np
import segmentation_models_pytorch as smp
from collections import OrderedDict

import torch.nn.functional as F


class MaskedAutoencoderCNN(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, checkpoint_type, img_size=224, patch_size=16, 
                 model_arch='Unet', encoder_name='densenet121', 
                 pretrained_path=None, mask_strategy='random'):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.model = smp.__dict__[model_arch](
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
        # Load pre-trained weights if a path is provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path, checkpoint_type)
            # binary classification을 위한 model classifier(FC layer) 변경
            # num_ftrs = self.model.segmentation_head[0].out_channels
            # self.model.segmentation_head = nn.Conv2d(num_ftrs, 1, kernel_size=(1, 1), bias=True)

        self.img_size = img_size
        self.patch_size = patch_size

    def patchify_heatmap(self, heatmaps, patch_size):
        """
        heatmaps batch를 받아서 (torch.Size([8, 224, 224]))
        [8, 224,224] 확률 분포 heatmap -> patch로 나눠서 [8, 14, 14, 16, 16] 
        patch 안의 값들은 sum -> [8, 14*14]
        batch별로 각 값들은 확률분포를 따름
        """
        batch_size = heatmaps.shape[0]

        heatmaps = heatmaps.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        heatmaps = heatmaps.contiguous().view(batch_size, -1, patch_size*patch_size).sum(-1)
        return heatmaps 

    def load_pretrained_weights(self, path, checkpoint_type):
        # Load pre-trained weights from a .pth file
        checkpoint = torch.load(path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % path)

        if 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        if checkpoint_type == 'smp_encoder':
            state_dict = checkpoint_model

            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                if 'model.encoder.' in key:
                    new_key = key.replace('model.encoder.', '')
                    new_state_dict[new_key] = value
            checkpoint_model = new_state_dict
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, imgs, mask_ratio, heatmaps=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        8(batch, N), 196(L), ?(Dim) 
        """
        x = self.patchify(imgs)
        heatmaps = self.patchify_heatmap(heatmaps, self.patch_size) #[8 batch_size, 224, 224] -> [8 batch_size, 196 patch_num]
        
        N, L, D = x.shape  # batch, length, dim

        len_keep = int(L * (1 - mask_ratio))

        if heatmaps is not None:
            # noise = torch.mul(torch.rand(N, L, device=x.device), heatmaps) # 중요한 patch(마스킹 되어야 하는 patch)일 수록 높은 값
            noise = torch.multinomial(heatmaps, num_samples=L) # batch별로 heatmap 확률분포 따라 sampling
            ids_shuffle = torch.flip(noise, dims=(1,))
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1, descending=False) # 모든 batch에 대해서, masking 확률 적은 patch 부터 index 정렬
        
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = self.unpatchify(x * (1 - mask.unsqueeze(-1)))
        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = imgs

        loss = (pred - target) ** 2
        loss = loss.mean()
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, heatmaps=None):
        if heatmaps is not None:
            imgs_masked, mask, ids_restore = self.random_masking(imgs, mask_ratio, heatmaps)
        else:
            imgs_masked, mask, ids_restore = self.random_masking(imgs, mask_ratio)
        
        pred = self.model(imgs_masked)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


