# import pycuda.driver as cuda
# cuda.init()
# cuda.Device(0).make_context()
import pycuda.autoinit

import atexit
import gc

import numpy as np
import io
import pickle
import time

import open3d as o3d

import cv2
import socket
# import numpy as np
from segmentation.container.src.inference import Runner


def main():
    print('Connecting to process...')
    while True:
        try:  # moved this line here
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect(("127.0.0.1", 5051))  # no longer throws error - 172.30.160.1
            break
        except socket.error:
            pass
    print('Connected to process')

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.bind(("0.0.0.0", 5050))
    in_sock.listen()

    def close_socket():
        in_sock.close(), out_sock.close()
    atexit.register(close_socket)

    model = Runner('./segmentation/tensorrt/assets/seg_int8.engine')

    # Accept client.
    print('Waiting for connections...')
    client, addr = in_sock.accept()
    print('Connection received')

    i, avg = 1, 0
    while True:
        start = time.perf_counter()

        data_length = int.from_bytes(client.recv(24), 'big')
        stream = io.BytesIO()
        while (data_length - stream.tell()) > 0:
            stream.write(client.recv(data_length - stream.tell()))

        inp = stream.getbuffer()
        gc.disable()
        data = pickle.loads(inp)
        gc.enable()

        mask = model(data['image'])[0]
        mask = mask.reshape(192, 256, 1)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        data['mask'] = mask


        segmented_depth = data['depth']
        segmented_depth[mask != 1] = 0

        if len(segmented_depth.nonzero()[0]) >= 4096:

            segmented_pc = depth_pointcloud(segmented_depth)
            idx = np.random.choice(segmented_pc.shape[0], 4096, replace=False)
            downsampled_pc = segmented_pc[idx]

            data['pc'] = downsampled_pc

        else:
            data['pc'] = None

        gc.disable()
        out = pickle.dumps(data)
        gc.enable()

        out_sock.sendall(len(out).to_bytes(24, 'big') + out)

        fps = 1 / (time.perf_counter() - start)
        avg += (fps - avg) / i
        i += 1
        print(f'\rfps={avg:.2f} count={i}', end='')

        # print(np.unique(mask))
        # if np.any(mask == 1):
        #     print('Object detected')


def depth_pointcloud(depth_image, intrinsics=None):
    depth_image = o3d.geometry.Image(depth_image)

    if intrinsics is None:
        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047, 'width': 640, 'height': 480}

    camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                               intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    return np.array(pcd.points)


if __name__ == '__main__':
    main()