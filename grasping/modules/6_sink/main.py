import atexit
import gc
import io
import pickle
import socket
import cv2
import numpy as np

import open3d as o3d
#Creatreconstructionject.
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer

from utils.misc import draw_mask

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, size)
sock.bind(("127.0.0.1", 5054)) # 172.30.160.1
sock.listen()


def close_socket():
    print('Closing socket')
    sock.close()
atexit.register(close_socket)

#Accept client.
print('Wainting for connections...')
client, addr = sock.accept()
print('Connection received')

#Receive all the bytes and write them into the file.
vis = Visualizer()
vis.create_window('Pose Estimation')

scene_pcd = PointCloud()
part_pcd = PointCloud()
part_pcd.points = Vector3dVector(np.array([[0,0,0]]))
pred_pcd = PointCloud()
coords_mesh = [TriangleMesh.create_coordinate_frame(size=0.1) for _ in range(2)]
render_setup = False

while True:

    data_length = int.from_bytes(client.recv(24), 'big')
    stream = io.BytesIO()
    while (data_length - stream.tell()) > 0:
        stream.write(client.recv(data_length - stream.tell()))

    gc.disable()
    data = pickle.loads(stream.getbuffer())
    gc.disable()


    # cv2.imshow('test', data['image'][..., None].astype(float))
    res1, res2 = draw_mask(data['image'], data['mask'])
    cv2.imshow('res1', res1)
    cv2.imshow('res2', res2)

    cv2.waitKey(1)

    if data['reconstruction'] is not None:
        if data['reconstruction'].shape[0] < 10_000:
            part_pcd.clear()
            pred_pcd.clear()
            aux = PointCloud()
            aux.points = Vector3dVector(data['normalized_pc'])
            aux.paint_uniform_color([1, 0, 0])

            aux2 = PointCloud()
            aux2.points = Vector3dVector(data['reconstruction'])
            aux2.paint_uniform_color([0, 1, 0])

            part_pcd += aux
            pred_pcd += aux2

            if not render_setup:
                vis.add_geometry(part_pcd)
                vis.add_geometry(pred_pcd)
                render_setup = True

            vis.update_geometry(part_pcd)
            vis.update_geometry(pred_pcd)

            vis.poll_events()
            vis.update_renderer()



