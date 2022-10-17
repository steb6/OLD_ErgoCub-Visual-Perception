from datetime import datetime
import numpy as np
import torch
import time
from cuml.cluster.dbscan import DBSCAN
# from sklearn.cluster import DBSCAN

from utils.timer import Timer

print('Initializing CUDA')
start = time.perf_counter()
torch.tensor([0]).cuda()
print(f'Cuda initialized in {time.perf_counter() - start} seconds')

data = [np.load(f'./denoising/assets/inputs/pc_noise{i}.npy') for i in range(10)]

for i, x in enumerate(data):
    with Timer('DBSCAN'):
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(x)  # 0.1 10 are perfect but slow
        close = clustering.labels_[x.argmax(axis=0)[2]]
        denoised_pc = x[clustering.labels_ == close]

    np.save(f'./denoising/assets/outputs/rapids/pc_out{i}', denoised_pc)

# sklearn-windows 6 fps
# sklearn-wsl 5.87 fps
# cuml-wsl 40 fps