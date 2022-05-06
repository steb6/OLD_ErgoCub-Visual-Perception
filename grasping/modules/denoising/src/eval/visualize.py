import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

for i in range(10):
    inp = np.load(f'./denoising/assets/inputs/pc_noise{i}.npy')

    rapids_pc = np.load(f'./denoising/assets/outputs/rapids/pc_out{i}.npy')
    torch_pc = np.load(f'./denoising/assets/outputs/torch/pc_out{i}.npy')

    aux1 = PointCloud()
    aux1.points = Vector3dVector(rapids_pc)
    aux1.paint_uniform_color(np.random.rand(3, 1))

    aux2 = PointCloud()
    aux2.points = Vector3dVector(torch_pc)
    aux2.paint_uniform_color(np.random.rand(3, 1))

    aux3 = PointCloud()
    aux3.points = Vector3dVector(inp)
    aux3.paint_uniform_color(np.random.rand(3, 1))

    draw_geometries([aux3])
    draw_geometries([aux1])
    draw_geometries([aux2])
    # draw_geometries([aux1, aux2, aux3])