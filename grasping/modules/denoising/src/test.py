# client.py
import numpy as np
import requests
#
# file = np.zeros([1024, 3])
# files = {'my_file': file.tobytes()}
#
# response = requests.post(
#     'http://127.0.0.1:8000/input',
#     files=files,
# )

file = np.load('denoising/assets/inputs/pc_noise0.npy')
files = {'input': file}

response = requests.post(
    'http://127.0.0.1:8000/file',
    files=files,
    data={'index': 0},
)

print(file.shape)
print(np.frombuffer(response.content).reshape(-1, 3).shape)
