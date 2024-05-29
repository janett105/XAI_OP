import numpy as np
import torch
import pandas as pd

def patchify_heatmap(heatmaps, patch_size):
    """
    heatmaps batch를 받아서 (torch.Size([8, 224, 224]))
    [8, 224,224] 확률 분포 heatmap -> patch로 나눠서 [8, 14, 14, 16, 16] 
    patch 안의 값들은 sum -> [8, 14*14(196)]
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

# 90% -> 196개 중 19개 남김/ 177개 지움
# 75% -> 196개 중 49개 남김 / 147개 지움 / 가장자리 한 줄이 52개

h = pd.DataFrame(heatmaps[0, :]) 
print(h[h==0.0])

# 0.000000    143
# 0.000917~ 0.098089 53