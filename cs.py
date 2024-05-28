import numpy as np
import torch

def patchify_heatmap(heatmaps, patch_size):
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

heatmap = np.load('data/DB_BBox/Bbox.npy')
heatmap = torch.from_numpy(heatmap)
heatmaps = heatmap.unsqueeze(0).repeat(8,1,1)
heatmaps = patchify_heatmap(heatmaps, 16)

noise = torch.multinomial(heatmaps, num_samples=196) # [8,196]
ids_shuffle = torch.flip(noise, dims=(1,)) # 모든 batch에 대해서, 중요도가 낮은 patch(masking 확률 적은 patch)부터 정렬 
ids_restore = torch.argsort(ids_shuffle, dim=1)

mask = torch.ones([8, 196])
mask[:, :17] = 0
# unshuffle to get the binary mask
mask = torch.gather(mask, dim=1, index=ids_restore)

def unpatchify_mask(mask):
    """
    [N(batch size), L(patch num, 14*14)]인 mask를 [N, 14, 14]로
    """
    batch_size, num_patches = mask.shape
    mask = mask.view(8, 14, 14)


# print(ids_shuffle.shape)
# print(ids_shuffle)
# print(heatmaps.shape)