import copy
import time

import numpy as np
import torch
import tqdm
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
import open3d as o3d
from scipy.spatial.transform import Rotation

from ransac.utils.dataset import TestSet
from ransac.utils.inference import Runner
from seg_pcr_ge.delete import GraspEstimator
# from ransac.utils.grasp_estimator import GraspEstimator

from shape_reconstruction.pytorch.datasets.BoxNetPOVDepth import BoxNet
from shape_reconstruction.tensorrt.utils.dataset import DataSet
from shape_reconstruction.tensorrt.utils.decoder import Decoder
from shape_reconstruction.tensorrt.utils.inference import Infer
from utils.timer import Timer


def main():
    it = 1000
    data_loader = DataSet(iterations=it)

    a = torch.zeros([1]).to('cuda')
    backbone = Infer('./shape_reconstruction/tensorrt/assets/pcr.engine')
    decoder = Decoder()
    grasp_estimator = GraspEstimator(Runner('./ransac/assets/ransac_5000.engine'))

    for i, x in enumerate(data_loader):
        x = x['input'][0]
        x = np.load(f'./seg_pcr_ge/test{i}.npy')
        with Timer('total'):
            with Timer('backbone'):
                fast_weights = backbone(x)
            with Timer('decoder'):
                res = decoder(fast_weights)
            if res.shape[0] < 10_000:
                with Timer('poses'):
                    poses = grasp_estimator.find_poses(res * [1, 1, -1], 0.001, 5000)
                    # if len(poses) == 1:
                    #     print('Error')
                    #
                    aux1 = PointCloud()
                    aux1.points = Vector3dVector(res * [1, 1, -1])
                    aux1.paint_uniform_color(np.random.rand(3))

                    aux2 = PointCloud()
                    aux2.points = Vector3dVector(x * [1, 1, -1])
                    aux2.paint_uniform_color(np.random.rand(3))

                    # for s, poses in [['new', poses1], ['old', poses2]]:
                    #     print(s)
                    c1, r1, c2, r2, planes, lines, vertices = poses

                    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(vertices)
                    line_set.lines = o3d.utility.Vector2iVector(lines)

                    right_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1). \
                        rotate(r1, center=[0, 0, 0]).translate(c1, relative=False)
                    left_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1). \
                        rotate(r2, center=[0, 0, 0]).translate(c2, relative=False)

                    # print(np.array([1, 0, 0]) @ r1)
                    # if i >= 20:

                    draw_geometries(
                        [aux1, aux2, line_set, right_hand, left_hand, TriangleMesh.create_coordinate_frame(size=0.5)] +
                        [TriangleMesh.create_coordinate_frame(origin=v, size=0.1) for v in vertices])

    for k in Timer.timers:
        print(k, 1 / (Timer.timers[k] / Timer.counters[k]))


def plot_line(line):
    t = (np.random.rand(1000000, 1) - 0.5) * 2
    l0, l = line

    aux = PointCloud()
    aux.points = Vector3dVector(l0 + t * l)
    return aux


if __name__ == '__main__':
    # Total: 14 fps
    #   Backbone: 32 fps
    #   Decoder: 31 fps
    #   Pose: 163
    np.seterr(all='raise')
    main()