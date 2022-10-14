import numpy as np
import torch
from torch import nn


def main():
    ransac = BuildRansac()

    it = 5000
    points = torch.tensor(np.load('grasping/modules/ransac/assets/test_pcd.npy'))
    idx = torch.randint(0, points.shape[0], size=[it * 3])
    subsets = points[idx].reshape(it, 3, 3)

    torch.onnx.export(ransac, (points, subsets, 0.005), 'grasping/modules/ransac/assets/ransac.onnx', input_names=['points', 'subsets', 'eps'],
                      output_names=[f'plane'], opset_version=11)

# Actual code that gets compiled into TRT
def parallel_ransac(points, subsets, eps):
    device = points.device

    v1 = subsets[:, 1] - subsets[:, 0]
    v2 = subsets[:, 2] - subsets[:, 0]
    p = subsets[:, 0]

    normals = cross_product(v1, v2)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    d = - torch.sum(normals * p, dim=1, keepdim=True)

    distances = torch.abs(normals @ points.transpose(0, 1) + d)

    scores = torch.sum(distances < eps, dim=1)
    planes = torch.concat([normals, d], dim=1)
    planes = planes * -torch.sign(d)  # normalization

    return planes, scores


# Utility Module used to generate the engine
class BuildRansac(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, subsets, eps):
        return parallel_ransac(points, subsets, eps)


def cross_product(v1, v2):
    return torch.stack([v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1],
                        v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2],
                        v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]], dim=1)

if __name__ == '__main__':
    main()