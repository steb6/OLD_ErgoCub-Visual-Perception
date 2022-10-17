from datetime import datetime

import requests
# from denoising.src.denoise import denoise

print(f'Importing numpy - {datetime.now()}')
import numpy as np
print(f'Numpy imported - {datetime.now()}')
print(f'Importing torch - {datetime.now()}')
import torch
print(f'Torch imported - {datetime.now()}')
import time
# from cuml.cluster.dbscan import DBSCAN
# from sklearn.cluster import DBSCAN

# from utils.timer import Timer

# print('Initializing CUDA')
# start = time.perf_counter()
# torch.tensor([0]).cuda()
# print(f'Cuda initialized in {time.perf_counter() - start} seconds')

data = [np.load(f'./denoising/assets/inputs/pc_noise{i}.npy') for i in range(10)]
print('Profiling...')
i = 0

start = time.perf_counter()

s = requests.Session()
for _ in range(100):
    for x in data:
        # with Timer('DBSCAN'):
            # clustering = DBSCAN(eps=0.05, min_samples=10).fit(x)  # 0.1 10 are perfect but slow
            files = {'input': x}
            response = s.post(
                'http://denoising:8000/file',
                files=files,
                data={'index': i}
            )
            clustering = np.frombuffer(response.content).reshape(-1, 3).shape
            # print(response.headers['index'])
            # i += 1
        # denoise(x)

print(1 / ((time.perf_counter() - start) / 1000))
print(i)
# print(f'{(Timer.timers["DBSCAN"] / Timer.counters["DBSCAN"])} seconds')
# print(f'{1 / (Timer.timers["DBSCAN"] / Timer.counters["DBSCAN"])} fps')

# sklearn-windows 6 fps
# sklearn-wsl 5.87 fps
# cuml-wsl 40 fps