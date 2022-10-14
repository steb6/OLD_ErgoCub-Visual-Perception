import copy

import numpy as np
import torch
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.pipelines.registration import registration_icp
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries


def icp(model, target, threshold=0.01):
    model_o3d = PointCloud()
    model_o3d.points = Vector3dVector(model)
    model_o3d.paint_uniform_color([1, 0, 0])

    target_o3d = PointCloud(points=Vector3dVector(target))
    target_o3d.points = Vector3dVector(target)
    target_o3d.paint_uniform_color([0, 1, 0])

    # draw_geometries([model_o3d, target_o3d])

    reg_p2p = registration_icp(
        model_o3d,
        target_o3d,
        threshold,
        init=np.eye(4),
    )

    # draw_geometries([copy.deepcopy(model_o3d).transform(reg_p2p.transformation), target_o3d])
    return reg_p2p.transformation


if __name__ == '__main__':
    source = np.load('grasping/modules/tracking/source.npy')
    target = np.load('grasping/modules/tracking/target.npy')

    idx = np.random.choice(source.shape[0], 2024, replace=False)
    source = source[idx]

    model_o3d = PointCloud()
    model_o3d.points = Vector3dVector(source)
    model_o3d.paint_uniform_color([1, 0, 0])

    target_o3d = PointCloud(points=Vector3dVector(target))
    target_o3d.points = Vector3dVector(target)
    target_o3d.paint_uniform_color([0, 1, 0])

    draw_geometries([model_o3d, target_o3d])
    T = icp(source, target)

    # res = copy.deepcopy(model_o3d)
    # res.transform(T)
    res = PointCloud()
    res.points = Vector3dVector((np.block([source, np.ones([source.shape[0], 1])]) @ T.T)[:, :3])
    res.paint_uniform_color([0, 0, 1])
    draw_geometries([model_o3d, target_o3d, res])